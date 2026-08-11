"""Fail-closed offline compiler for paired LongMemEval-S reranker evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from _longmemeval_common import LONGMEMEVAL_S_SHA256, LONGMEMEVAL_S_URL
from pydantic import ValidationError

from swarmbrain.domain.reranking import (
    LearnedRerankerIdentity,
    LearnedRerankPolicy,
    LearnedRerankReceipt,
    LearnedRerankResult,
    LearnedRerankTrace,
)
from swarmbrain.retrieval.learned_reranking import (
    request_usage_dimensions,
    validate_learned_rerank_result,
)

from .contracts import (
    BASELINE_ARM,
    CANDIDATE_WINDOW,
    K_VALUES,
    PROTOCOL_VERSION,
    REPORT_ARTIFACT_TYPE,
    RUN_ARTIFACT_TYPE,
    RUN_SCHEMA_VERSION,
    SLICE_CATEGORIES,
    TREATMENT_ARM,
    LongMemEvalRerankerEvidenceError,
    finite_number,
    integer,
    required_text,
    sha256_text,
)
from .evidence import (
    ARM_INPUT_FIELDS,
    TRACE_SCHEMA_VERSION,
    arm_input_evidence,
    build_core_request,
    candidate_document_evidence,
    canonical_json,
    canonical_policy,
    implementation_fingerprint,
    parse_candidate_position,
    protocol_evidence,
    query_text,
    sha256_bytes,
)
from .metrics import PairedRankingCase, paired_summary, percentile

EXPECTED_QUESTION_COUNT = 500
EXPECTED_ABSTENTION_QUESTIONS = 30

_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "protocol_version",
        "created_at_utc",
        "implementation",
        "dataset",
        "source_retrieval_artifact",
        "traces_artifact",
        "protocol",
        "reranker_identity",
        "rerank_policy",
        "call_accounting",
    }
)
_ARTIFACT_FIELDS = frozenset({"path", "bytes", "sha256"})
_TRACE_ARTIFACT_FIELDS = frozenset({"path", "bytes", "sha256", "rows"})
_TRACE_FIELDS = frozenset(
    {
        "trace_schema_version",
        "case_index",
        "case_id",
        "category",
        "abstention_question",
        "candidate_documents",
        "baseline",
        "learned",
    }
)
_BASELINE_FIELDS = frozenset({"arm", "input", "ranked_ids"})
_LEARNED_FIELDS = frozenset({"arm", "input", "trace"})
_USAGE_TOTAL_FIELDS = (
    "candidate_count",
    "query_characters",
    "document_characters",
    "temporal_characters",
    "query_bytes",
    "document_bytes",
    "temporal_bytes",
    "request_bytes",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)
_ACCOUNTING_FIELDS = frozenset(
    {
        "source",
        "requests",
        "responses",
        "successful_responses",
        "unique_request_ids",
        "unique_provider_request_ids",
        "unique_tokenized_input_sha256",
        *_USAGE_TOTAL_FIELDS,
    }
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise LongMemEvalRerankerEvidenceError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise LongMemEvalRerankerEvidenceError(f"non-finite JSON number {value!r} is forbidden")


def _strict_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, LongMemEvalRerankerEvidenceError) as exc:
        raise LongMemEvalRerankerEvidenceError(f"{label} is not strict JSON: {exc}") from exc


def _strict_object(raw: bytes, *, label: str) -> dict[str, Any]:
    value = _strict_json(raw, label=label)
    if not isinstance(value, dict):
        raise LongMemEvalRerankerEvidenceError(f"{label} must contain one JSON object")
    return value


def _strict_jsonl(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise LongMemEvalRerankerEvidenceError(f"{label} is not UTF-8") from exc
    if not text or not text.endswith("\n"):
        raise LongMemEvalRerankerEvidenceError(f"{label} must be non-empty and newline terminated")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise LongMemEvalRerankerEvidenceError(f"{label}:{line_number} cannot be blank")
        value = _strict_json(line.encode("utf-8"), label=f"{label}:{line_number}")
        if not isinstance(value, dict):
            raise LongMemEvalRerankerEvidenceError(f"{label}:{line_number} must be an object")
        rows.append(value)
    return rows


def _resolve_input(
    path: str | Path,
    *,
    root: Path,
    label: str,
    require_relative: bool,
) -> tuple[Path, str]:
    supplied = Path(path)
    if require_relative and supplied.is_absolute():
        raise LongMemEvalRerankerEvidenceError(f"{label} must be a repository-local relative path")
    if ".." in supplied.parts:
        raise LongMemEvalRerankerEvidenceError(f"{label} cannot contain '..'")
    root = root.resolve()
    candidate = supplied if supplied.is_absolute() else root / supplied
    current = Path(candidate.anchor) if candidate.is_absolute() else root
    parts = candidate.parts[1:] if candidate.is_absolute() else candidate.parts
    for part in parts:
        current /= part
        if current.is_symlink():
            raise LongMemEvalRerankerEvidenceError(f"{label} cannot traverse symbolic links")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LongMemEvalRerankerEvidenceError(f"{label} is missing") from exc
    if not resolved.is_file():
        raise LongMemEvalRerankerEvidenceError(f"{label} must resolve to a regular file")
    if require_relative and not resolved.is_relative_to(root):
        raise LongMemEvalRerankerEvidenceError(f"{label} must remain inside the repository")
    display = (
        resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(resolved)
    )
    return resolved, display


def _resolve_output(
    path: str | Path,
    *,
    root: Path,
    overwrite: bool,
    protected: set[Path],
) -> Path:
    supplied = Path(path)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise LongMemEvalRerankerEvidenceError(
            "--output must be a repository-local relative path without '..'"
        )
    root = root.resolve()
    candidate = root / supplied
    if not candidate.resolve().is_relative_to(root):
        raise LongMemEvalRerankerEvidenceError("--output must remain inside the repository")
    current = root
    for part in supplied.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise LongMemEvalRerankerEvidenceError("--output cannot traverse symbolic links")
    if candidate.is_symlink() or candidate.resolve() in protected:
        raise LongMemEvalRerankerEvidenceError(
            "--output cannot be a symlink or overwrite an input artifact"
        )
    if candidate.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite reranker report: {candidate}")
    return candidate


def _identity(raw: bytes, *, path: str) -> dict[str, Any]:
    return {"path": path, "bytes": len(raw), "sha256": sha256_bytes(raw)}


def _validate_bound_artifact(
    value: Any,
    *,
    root: Path,
    label: str,
    fields: frozenset[str],
) -> tuple[Path, str, bytes]:
    if not isinstance(value, dict) or set(value) != fields:
        raise LongMemEvalRerankerEvidenceError(f"{label} must bind exactly {sorted(fields)}")
    raw_path = required_text(value.get("path"), label=f"{label}.path")
    resolved, relative = _resolve_input(
        raw_path,
        root=root,
        label=f"{label}.path",
        require_relative=True,
    )
    if relative != raw_path:
        raise LongMemEvalRerankerEvidenceError(
            f"{label}.path must use canonical repository-relative POSIX form"
        )
    raw = resolved.read_bytes()
    if integer(value.get("bytes"), label=f"{label}.bytes") != len(raw):
        raise LongMemEvalRerankerEvidenceError(f"{label}.bytes does not match the file")
    if sha256_text(value.get("sha256"), label=f"{label}.sha256") != sha256_bytes(raw):
        raise LongMemEvalRerankerEvidenceError(f"{label}.sha256 does not match the file")
    return resolved, relative, raw


def _validate_created_at(value: Any) -> None:
    text = required_text(value, label="run.created_at_utc")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LongMemEvalRerankerEvidenceError("run.created_at_utc must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise LongMemEvalRerankerEvidenceError("run.created_at_utc must carry UTC offset")


def _validate_run(
    value: dict[str, Any],
    *,
    code_root: Path,
) -> tuple[LearnedRerankerIdentity, LearnedRerankPolicy]:
    if set(value) != _RUN_FIELDS:
        raise LongMemEvalRerankerEvidenceError(
            f"run fields differ from schema: expected {sorted(_RUN_FIELDS)}"
        )
    expected = {
        "schema_version": RUN_SCHEMA_VERSION,
        "artifact_type": RUN_ARTIFACT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
    }
    for field, wanted in expected.items():
        if type(value.get(field)) is not type(wanted) or value.get(field) != wanted:
            raise LongMemEvalRerankerEvidenceError(f"run.{field} must be {wanted!r}")
    _validate_created_at(value.get("created_at_utc"))
    if value.get("implementation") != implementation_fingerprint(code_root):
        raise LongMemEvalRerankerEvidenceError(
            "run implementation fingerprint does not match the current compiler/core tree"
        )
    if value.get("protocol") != protocol_evidence():
        raise LongMemEvalRerankerEvidenceError("run protocol differs from the fixed paired design")
    try:
        identity = LearnedRerankerIdentity.model_validate(value.get("reranker_identity"))
        policy = LearnedRerankPolicy.model_validate(value.get("rerank_policy"))
    except ValidationError as exc:
        raise LongMemEvalRerankerEvidenceError(
            f"run learned-reranker identity/policy is invalid: {exc}"
        ) from exc
    if policy.identity != identity or policy != canonical_policy(identity):
        raise LongMemEvalRerankerEvidenceError(
            "run rerank_policy must be the canonical alpha=1/window=50 core policy"
        )
    return identity, policy


def _load_dataset(
    raw: bytes,
    *,
    expected_sha256: str,
    expected_questions: int,
) -> list[dict[str, Any]]:
    if sha256_bytes(raw) != expected_sha256:
        raise LongMemEvalRerankerEvidenceError(
            "--dataset bytes do not match the expected pinned digest"
        )
    value = _strict_json(raw, label="--dataset")
    if not isinstance(value, list) or len(value) != expected_questions:
        raise LongMemEvalRerankerEvidenceError(
            f"--dataset must contain exactly {expected_questions} questions"
        )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(value):
        label = f"dataset[{index}]"
        if not isinstance(record, dict):
            raise LongMemEvalRerankerEvidenceError(f"{label} must be an object")
        case_id = required_text(record.get("question_id"), label=f"{label}.question_id")
        if case_id in seen:
            raise LongMemEvalRerankerEvidenceError(f"{label} repeats question_id {case_id!r}")
        seen.add(case_id)
        required_text(record.get("question_type"), label=f"{label}.question_type")
        query_text(record)
        required_text(record.get("question_date"), label=f"{label}.question_date")
        arrays = (
            record.get("haystack_session_ids"),
            record.get("haystack_dates"),
            record.get("haystack_sessions"),
        )
        if not all(isinstance(item, list) for item in arrays):
            raise LongMemEvalRerankerEvidenceError(f"{label} haystack arrays must be lists")
        if not len(arrays[0]) == len(arrays[1]) == len(arrays[2]):
            raise LongMemEvalRerankerEvidenceError(f"{label} haystack arrays are misaligned")
        if len(arrays[0]) < max(K_VALUES):
            raise LongMemEvalRerankerEvidenceError(
                f"{label} has fewer candidates than the largest evaluated k"
            )
        answer_ids = record.get("answer_session_ids")
        if (
            not isinstance(answer_ids, list)
            or not answer_ids
            or not all(isinstance(item, str) and item for item in answer_ids)
        ):
            raise LongMemEvalRerankerEvidenceError(
                f"{label}.answer_session_ids must be non-empty strings"
            )
        if not set(answer_ids).issubset(set(arrays[0])):
            raise LongMemEvalRerankerEvidenceError(
                f"{label}.answer_session_ids point outside the question haystack"
            )
        records.append(record)
    return records


def _gold_ids(record: dict[str, Any]) -> tuple[str, ...]:
    answers = set(record["answer_session_ids"])
    return tuple(
        f"{position:03d}:{session_id}"
        for position, session_id in enumerate(record["haystack_session_ids"])
        if session_id in answers
    )


def _string_list(value: Any, *, label: str, unique: bool = True) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise LongMemEvalRerankerEvidenceError(f"{label} must be a list of non-empty strings")
    if unique and len(value) != len(set(value)):
        raise LongMemEvalRerankerEvidenceError(f"{label} contains duplicate candidates")
    return value


def _validate_source_retrieval(
    source: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    dataset_sha256: str,
    require_publishable: bool,
    require_current_implementation: bool,
) -> list[tuple[dict[str, Any], list[str]]]:
    expected = {
        "artifact_type": "swarmbrain-retrieval-eval-run",
        "schema_version": 2,
        "protocol_version": "swarmbrain-longmemeval-retrieval-v2",
        "track": "longmemeval-s",
        "granularity": "one memory per haystack session",
    }
    for field, wanted in expected.items():
        if type(source.get(field)) is not type(wanted) or source.get(field) != wanted:
            raise LongMemEvalRerankerEvidenceError(f"source retrieval {field} must be {wanted!r}")
    dataset = source.get("dataset")
    if not isinstance(dataset, dict):
        raise LongMemEvalRerankerEvidenceError("source retrieval lacks dataset metadata")
    source_dataset_expected = {
        "name": "LongMemEval-S",
        "sha256": dataset_sha256,
        "total_questions": len(records),
        "evaluated_questions": len(records),
        "sample_seed": None,
    }
    for field, wanted in source_dataset_expected.items():
        if type(dataset.get(field)) is not type(wanted) or dataset.get(field) != wanted:
            raise LongMemEvalRerankerEvidenceError(
                f"source retrieval dataset.{field} must be {wanted!r}"
            )
    recall_limit = integer(source.get("recall_limit"), label="source recall_limit", minimum=10)
    saved_depth = integer(
        source.get("saved_ranking_depth"), label="source saved_ranking_depth", minimum=50
    )
    if saved_depth < CANDIDATE_WINDOW:
        raise LongMemEvalRerankerEvidenceError(
            "source retrieval does not preserve the complete rerank window"
        )
    raw_cases = source.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != len(records):
        raise LongMemEvalRerankerEvidenceError(
            "source retrieval case coverage differs from the dataset"
        )

    if require_publishable:
        from run_longmemeval_qa import (
            retrieval_publishability_errors,
            validate_retrieval_run_protocol,
        )

        try:
            validate_retrieval_run_protocol(source)
            errors = retrieval_publishability_errors(source)
        except ValueError as exc:
            raise LongMemEvalRerankerEvidenceError(
                f"source retrieval protocol validation failed: {exc}"
            ) from exc
        if errors:
            raise LongMemEvalRerankerEvidenceError(
                "source retrieval is not publishable: " + "; ".join(errors)
            )
    if require_current_implementation:
        from run_retrieval_eval import retrieval_implementation_fingerprint

        if source.get("implementation") != retrieval_implementation_fingerprint():
            raise LongMemEvalRerankerEvidenceError(
                "source retrieval implementation does not match the current tree"
            )

    output: list[tuple[dict[str, Any], list[str]]] = []
    abstentions = 0
    for index, (record, case) in enumerate(zip(records, raw_cases, strict=True)):
        label = f"source.cases[{index}]"
        if not isinstance(case, dict):
            raise LongMemEvalRerankerEvidenceError(f"{label} must be an object")
        case_id = str(record["question_id"])
        category = str(record["question_type"])
        abstention = case_id.endswith("_abs")
        abstentions += int(abstention)
        expected_fields = {
            "case_id": case_id,
            "category": category,
            "abstention_question": abstention,
            "haystack_sessions": len(record["haystack_session_ids"]),
            "relevant_ids": list(_gold_ids(record)),
            "degraded_lanes": [],
        }
        for field, wanted in expected_fields.items():
            if type(case.get(field)) is not type(wanted) or case.get(field) != wanted:
                raise LongMemEvalRerankerEvidenceError(
                    f"{label}.{field} disagrees with the pinned dataset/protocol"
                )
        rankings = case.get("rankings")
        if not isinstance(rankings, dict):
            raise LongMemEvalRerankerEvidenceError(f"{label}.rankings must be an object")
        fused = _string_list(rankings.get("fused"), label=f"{label}.rankings.fused")
        expected_pool_size = min(CANDIDATE_WINDOW, len(record["haystack_session_ids"]))
        if len(fused) != expected_pool_size:
            raise LongMemEvalRerankerEvidenceError(
                f"{label}.rankings.fused must equal the complete fixed rerank window"
            )
        candidate_ids = fused[:expected_pool_size]
        for candidate_id in candidate_ids:
            parse_candidate_position(record, candidate_id)
        final = _string_list(rankings.get("final"), label=f"{label}.rankings.final")
        expected_final = fused[: min(recall_limit, len(fused))]
        if final != expected_final:
            raise LongMemEvalRerankerEvidenceError(
                f"{label} is not an unrereanked fixed-fusion baseline"
            )
        output.append((case, candidate_ids))
    if len(records) == EXPECTED_QUESTION_COUNT and abstentions != EXPECTED_ABSTENTION_QUESTIONS:
        raise LongMemEvalRerankerEvidenceError(
            "canonical LongMemEval-S evidence must contain exactly 30 abstention questions"
        )
    return output


def _validate_arm_input(value: Any, *, expected: dict[str, Any], label: str) -> None:
    if not isinstance(value, dict) or set(value) != ARM_INPUT_FIELDS:
        raise LongMemEvalRerankerEvidenceError(
            f"{label} must contain exactly the shared core input contract"
        )
    if value != expected:
        raise LongMemEvalRerankerEvidenceError(
            f"{label} changes query, candidate text/temporal payload, tokenizer input, or k"
        )


def _validate_core_trace(
    raw_trace: Any,
    *,
    record: dict[str, Any],
    candidate_ids: list[str],
    identity: LearnedRerankerIdentity,
    policy: LearnedRerankPolicy,
    label: str,
) -> tuple[LearnedRerankTrace, Any]:
    try:
        trace = LearnedRerankTrace.model_validate(raw_trace)
    except ValidationError as exc:
        raise LongMemEvalRerankerEvidenceError(f"{label} is not a valid core trace: {exc}") from exc
    if trace.identity != identity or trace.policy != policy:
        raise LongMemEvalRerankerEvidenceError(
            f"{label} does not use the run's immutable identity and policy"
        )
    if (
        not trace.attempted
        or not trace.applied
        or trace.degraded
        or trace.degradation_reason is not None
        or trace.request_id is None
        or trace.provider_request_id is None
        or trace.usage is None
        or trace.response_sha256 is None
    ):
        raise LongMemEvalRerankerEvidenceError(
            f"{label} is not one complete successfully applied provider result"
        )
    if len(trace.input_ids) != len(candidate_ids) or len(trace.input_ids) != len(
        set(trace.input_ids)
    ):
        raise LongMemEvalRerankerEvidenceError(
            f"{label}.input_ids do not align one-to-one with the source candidate pool"
        )
    for provider_id in trace.input_ids:
        try:
            if str(UUID(provider_id)) != provider_id:
                raise ValueError
        except ValueError as exc:
            raise LongMemEvalRerankerEvidenceError(
                f"{label}.input_ids must be canonical runtime memory UUIDs"
            ) from exc
    request = build_core_request(
        record,
        candidate_ids,
        list(trace.input_ids),
        policy,
        request_id=trace.request_id,
    )
    expected_trace_fields = {
        "serializer_revision": request.serializer_revision,
        "request_sha256": request.request_sha256,
        "query_sha256": request.query_sha256,
        "candidate_pool_sha256": request.candidate_pool_sha256,
        "candidate_document_sha256": {
            item.candidate_id: item.document_sha256 for item in request.candidates
        },
        "candidate_temporal_sha256": {
            item.candidate_id: item.temporal_sha256 for item in request.candidates
        },
        "input_ids": tuple(item.candidate_id for item in request.candidates),
    }
    for field, wanted in expected_trace_fields.items():
        if getattr(trace, field) != wanted:
            raise LongMemEvalRerankerEvidenceError(
                f"{label}.{field} does not match the dataset-reconstructed core request"
            )
    expected_dimensions = request_usage_dimensions(request)
    for field, wanted in expected_dimensions.items():
        if getattr(trace.usage, field) != wanted:
            raise LongMemEvalRerankerEvidenceError(
                f"{label}.usage.{field} does not match the reconstructed request"
            )
    try:
        receipt = LearnedRerankReceipt(
            identity=identity,
            request_sha256=request.request_sha256,
            provider_request_id=trace.provider_request_id,
            usage=trace.usage,
            response_sha256=trace.response_sha256,
        )
        result = LearnedRerankResult(scores=trace.scores, receipt=receipt)
        validate_learned_rerank_result(request, result, expected_identity=identity)
    except (ValidationError, ValueError) as exc:
        raise LongMemEvalRerankerEvidenceError(
            f"{label} response receipt does not bind the reconstructed request: {exc}"
        ) from exc
    return trace, request


def _validate_trace_rows(
    rows: list[dict[str, Any]],
    *,
    records: list[dict[str, Any]],
    source_cases: list[tuple[dict[str, Any], list[str]]],
    identity: LearnedRerankerIdentity,
    policy: LearnedRerankPolicy,
) -> tuple[list[PairedRankingCase], dict[str, Any], list[float]]:
    if len(rows) != len(records):
        raise LongMemEvalRerankerEvidenceError("trace row coverage differs from the dataset")
    request_ids: set[str] = set()
    provider_request_ids: set[str] = set()
    tokenized_input_digests: set[str] = set()
    totals = {field: 0 for field in _USAGE_TOTAL_FIELDS}
    latencies: list[float] = []
    paired_cases: list[PairedRankingCase] = []

    for index, (row, record, source_info) in enumerate(
        zip(rows, records, source_cases, strict=True)
    ):
        label = f"trace[{index}]"
        if set(row) != _TRACE_FIELDS:
            raise LongMemEvalRerankerEvidenceError(f"{label} fields differ from the trace schema")
        source_case, candidate_ids = source_info
        case_id = str(record["question_id"])
        category = str(record["question_type"])
        abstention = case_id.endswith("_abs")
        expected_scalars = {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "case_index": index,
            "case_id": case_id,
            "category": category,
            "abstention_question": abstention,
        }
        for field, wanted in expected_scalars.items():
            if type(row.get(field)) is not type(wanted) or row.get(field) != wanted:
                raise LongMemEvalRerankerEvidenceError(f"{label}.{field} is invalid")

        baseline = row.get("baseline")
        if not isinstance(baseline, dict) or set(baseline) != _BASELINE_FIELDS:
            raise LongMemEvalRerankerEvidenceError(f"{label}.baseline fields are malformed")
        if baseline.get("arm") != BASELINE_ARM:
            raise LongMemEvalRerankerEvidenceError(f"{label}.baseline arm is invalid")
        baseline_ids = _string_list(
            baseline.get("ranked_ids"), label=f"{label}.baseline.ranked_ids"
        )
        if baseline_ids != candidate_ids:
            raise LongMemEvalRerankerEvidenceError(
                f"{label}.baseline must preserve the source fused order"
            )

        learned = row.get("learned")
        if not isinstance(learned, dict) or set(learned) != _LEARNED_FIELDS:
            raise LongMemEvalRerankerEvidenceError(f"{label}.learned fields are malformed")
        if learned.get("arm") != TREATMENT_ARM:
            raise LongMemEvalRerankerEvidenceError(f"{label}.learned arm is invalid")
        trace, request = _validate_core_trace(
            learned.get("trace"),
            record=record,
            candidate_ids=candidate_ids,
            identity=identity,
            policy=policy,
            label=f"{label}.learned.trace",
        )
        assert trace.request_id is not None
        assert trace.provider_request_id is not None
        assert trace.usage is not None
        if trace.request_id in request_ids:
            raise LongMemEvalRerankerEvidenceError("learned arm reuses a request ID")
        if trace.provider_request_id in provider_request_ids:
            raise LongMemEvalRerankerEvidenceError("learned arm reuses a provider request ID")
        if trace.request_id == trace.provider_request_id:
            raise LongMemEvalRerankerEvidenceError(
                "provider request ID must differ from the client request ID"
            )
        request_ids.add(trace.request_id)
        provider_request_ids.add(trace.provider_request_id)
        tokenized_input_digests.add(trace.usage.tokenized_input_sha256)

        expected_input = arm_input_evidence(
            request,
            evaluation_candidate_ids=candidate_ids,
            tokenized_input_sha256=trace.usage.tokenized_input_sha256,
        )
        _validate_arm_input(
            baseline.get("input"), expected=expected_input, label=f"{label}.baseline.input"
        )
        _validate_arm_input(
            learned.get("input"), expected=expected_input, label=f"{label}.learned.input"
        )
        if baseline.get("input") != learned.get("input"):
            raise LongMemEvalRerankerEvidenceError(
                f"{label} counterfactual arms do not share one exact input"
            )
        if row.get("candidate_documents") != candidate_document_evidence(request, candidate_ids):
            raise LongMemEvalRerankerEvidenceError(
                f"{label}.candidate_documents disagree with the reconstructed core request"
            )

        provider_output_ids = list(trace.output_ids)
        if len(provider_output_ids) != len(candidate_ids) or len(provider_output_ids) != len(
            set(provider_output_ids)
        ):
            raise LongMemEvalRerankerEvidenceError(
                f"{label}.learned output omits or duplicates candidates"
            )
        missing = set(trace.input_ids).difference(provider_output_ids)
        added = set(provider_output_ids).difference(trace.input_ids)
        if missing or added:
            raise LongMemEvalRerankerEvidenceError(
                f"{label}.learned output is not an exact candidate permutation; "
                f"missing={sorted(missing)}, added={sorted(added)}"
            )
        scores = {item.candidate_id: float(item.score) for item in trace.scores}
        source_rank = {candidate_id: rank for rank, candidate_id in enumerate(trace.input_ids)}
        expected_order = sorted(
            trace.input_ids,
            key=lambda item: (-scores[item], source_rank[item]),
        )
        if provider_output_ids != expected_order:
            raise LongMemEvalRerankerEvidenceError(
                f"{label}.learned output is not stable descending score order"
            )
        evaluation_by_provider = dict(zip(trace.input_ids, candidate_ids, strict=True))
        output_ids = [evaluation_by_provider[item] for item in provider_output_ids]

        for field in _USAGE_TOTAL_FIELDS:
            totals[field] += int(getattr(trace.usage, field))
        latency = finite_number(trace.latency_ms, label=f"{label}.latency_ms", minimum=0.0)
        latencies.append(latency)
        paired_cases.append(
            PairedRankingCase(
                case_id=case_id,
                category=category,
                abstention_question=abstention,
                relevant_ids=frozenset(_gold_ids(record)),
                baseline_ids=tuple(baseline_ids),
                treatment_ids=tuple(output_ids),
            )
        )
        if source_case.get("case_id") != case_id:
            raise LongMemEvalRerankerEvidenceError(f"{label} source pairing changed")

    accounting = {
        "source": "provider-observed-and-offline-reconciled",
        "requests": len(rows),
        "responses": len(rows),
        "successful_responses": len(rows),
        "unique_request_ids": len(request_ids),
        "unique_provider_request_ids": len(provider_request_ids),
        "unique_tokenized_input_sha256": len(tokenized_input_digests),
        **totals,
    }
    return paired_cases, accounting, latencies


def _validate_accounting(value: Any, *, expected: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != _ACCOUNTING_FIELDS:
        raise LongMemEvalRerankerEvidenceError("run.call_accounting fields are malformed")
    for field, wanted in expected.items():
        if type(value.get(field)) is not type(wanted) or value.get(field) != wanted:
            raise LongMemEvalRerankerEvidenceError(
                f"run.call_accounting.{field} does not reconcile with raw traces"
            )


def _slice_metrics(cases: list[PairedRankingCase]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, category in SLICE_CATEGORIES.items():
        selected = (
            cases if category is None else [case for case in cases if case.category == category]
        )
        if not selected:
            raise LongMemEvalRerankerEvidenceError(f"required paired slice {name!r} is empty")
        output[name] = {f"k={k}": paired_summary(selected, k=k) for k in K_VALUES}
    return output


def _validate_dataset_metadata(
    value: Any,
    *,
    dataset_sha256: str,
    questions: int,
) -> None:
    expected = {
        "name": "LongMemEval-S",
        "source": LONGMEMEVAL_S_URL,
        "sha256": dataset_sha256,
        "questions": questions,
    }
    if not isinstance(value, dict) or value != expected:
        raise LongMemEvalRerankerEvidenceError(
            "run.dataset does not bind the supplied LongMemEval-S bytes"
        )


def compile_longmemeval_reranker_report(
    run: str | Path,
    dataset: str | Path,
    output: str | Path,
    *,
    artifact_root: Path,
    code_root: Path,
    overwrite: bool = False,
    expected_dataset_sha256: str = LONGMEMEVAL_S_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    require_publishable_source: bool = True,
    require_current_source_implementation: bool = True,
) -> dict[str, Any]:
    """Reparse raw core traces and derive all paired metrics without model calls."""

    root = artifact_root.resolve()
    run_path, run_display = _resolve_input(run, root=root, label="--run", require_relative=True)
    dataset_path, dataset_display = _resolve_input(
        dataset, root=root, label="--dataset", require_relative=False
    )
    run_raw = run_path.read_bytes()
    dataset_raw = dataset_path.read_bytes()
    run_payload = _strict_object(run_raw, label="--run")
    identity, policy = _validate_run(run_payload, code_root=code_root)
    records = _load_dataset(
        dataset_raw,
        expected_sha256=expected_dataset_sha256,
        expected_questions=expected_question_count,
    )
    _validate_dataset_metadata(
        run_payload.get("dataset"),
        dataset_sha256=expected_dataset_sha256,
        questions=expected_question_count,
    )

    source_path, source_display, source_raw = _validate_bound_artifact(
        run_payload.get("source_retrieval_artifact"),
        root=root,
        label="run.source_retrieval_artifact",
        fields=_ARTIFACT_FIELDS,
    )
    traces_path, traces_display, traces_raw = _validate_bound_artifact(
        run_payload.get("traces_artifact"),
        root=root,
        label="run.traces_artifact",
        fields=_TRACE_ARTIFACT_FIELDS,
    )
    source_payload = _strict_object(source_raw, label="source retrieval artifact")
    source_cases = _validate_source_retrieval(
        source_payload,
        records,
        dataset_sha256=expected_dataset_sha256,
        require_publishable=require_publishable_source,
        require_current_implementation=require_current_source_implementation,
    )
    trace_rows = _strict_jsonl(traces_raw, label="reranker traces artifact")
    expected_trace_rows = integer(
        run_payload["traces_artifact"].get("rows"),
        label="run.traces_artifact.rows",
        minimum=1,
    )
    if expected_trace_rows != len(trace_rows):
        raise LongMemEvalRerankerEvidenceError(
            "run.traces_artifact.rows does not match the JSONL evidence"
        )
    paired_cases, observed_accounting, latencies = _validate_trace_rows(
        trace_rows,
        records=records,
        source_cases=source_cases,
        identity=identity,
        policy=policy,
    )
    _validate_accounting(run_payload.get("call_accounting"), expected=observed_accounting)

    canonical_official = (
        expected_dataset_sha256 == LONGMEMEVAL_S_SHA256
        and expected_question_count == EXPECTED_QUESTION_COUNT
        and require_publishable_source
        and require_current_source_implementation
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "benchmark": {
            "name": "LongMemEval-S learned-reranker paired A/B",
            "canonical_official_dataset": canonical_official,
            "retrieval_only": True,
            "reader_or_judge_called": False,
        },
        "design": {
            **protocol_evidence(),
            "candidate_payload_source": "byte-pinned official dataset",
            "gold_fields_excluded_from_core_request": True,
            "one_source_retrieval_for_both_arms": True,
        },
        "provenance": {
            "run_artifact": _identity(run_raw, path=run_display),
            "dataset_artifact": _identity(dataset_raw, path=dataset_display),
            "source_retrieval_artifact": _identity(source_raw, path=source_display),
            "traces_artifact": {
                **_identity(traces_raw, path=traces_display),
                "rows": len(trace_rows),
            },
            "implementation": run_payload["implementation"],
            "reranker_identity": identity.model_dump(mode="json"),
            "rerank_policy": policy.model_dump(mode="json"),
            "receipt_attestation": {
                "source": "unsigned-local-process-receipt",
                "digest_bound": True,
                "cryptographically_signed": False,
                "component_artifact_bytes_reopened_by_compiler": False,
            },
        },
        "coverage": {
            "questions": len(paired_cases),
            "abstention_questions": sum(case.abstention_question for case in paired_cases),
            "categories": {
                category: sum(case.category == category for case in paired_cases)
                for category in sorted({case.category for case in paired_cases})
            },
            "candidate_inputs": observed_accounting["candidate_count"],
        },
        "paired_metrics": _slice_metrics(paired_cases),
        "reranker_execution": {
            "call_accounting": observed_accounting,
            "latency_ms": {
                "mean": sum(latencies) / len(latencies),
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "total": sum(latencies),
            },
        },
        "validation": {
            "full_question_coverage": len(paired_cases) == expected_question_count,
            "source_retrieval_byte_bound": True,
            "dataset_byte_bound": True,
            "all_candidate_payloads_reconstructed_with_core_serializer": True,
            "baseline_is_source_fusion": True,
            "same_query_candidate_text_temporal_tokenizer_input_and_k": True,
            "learned_outputs_are_exact_permutations": True,
            "scores_finite_normalized_and_stably_sorted": True,
            "request_ids_unique": True,
            "provider_request_ids_unique": True,
            "core_request_digests_recomputed": True,
            "core_response_receipts_recomputed": True,
            "composite_identity_bundle_digests_reconciled": True,
            "provider_usage_bytes_characters_and_tokens_reconciled": True,
            "paired_metrics_recomputed_offline": True,
            "pass_threshold_encoded": False,
        },
        "limitations": [
            "Retrieval metrics do not establish end-to-end LongMemEval answer accuracy.",
            "Provider-reported token counts are response-bound but cannot be recomputed without "
            "the pinned tokenizer artifacts.",
            "Local process receipts are digest-bound but unsigned. Publishable model-weight "
            "identity still requires artifact replay or an external signed attestation.",
            "No pass threshold is encoded until a preregistered comparison target is adopted.",
        ],
    }
    canonical_json(report)

    output_path = _resolve_output(
        output,
        root=root,
        overwrite=overwrite,
        protected={run_path, dataset_path, source_path, traces_path},
    )
    if (
        run_path.read_bytes() != run_raw
        or dataset_path.read_bytes() != dataset_raw
        or source_path.read_bytes() != source_raw
        or traces_path.read_bytes() != traces_raw
    ):
        raise LongMemEvalRerankerEvidenceError("an input artifact changed during compilation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "EXPECTED_ABSTENTION_QUESTIONS",
    "EXPECTED_QUESTION_COUNT",
    "compile_longmemeval_reranker_report",
]
