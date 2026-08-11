"""Fail-closed offline compiler for paired LongMemEval selection QA evidence."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _longmemeval_common import LONGMEMEVAL_S_SHA256, LONGMEMEVAL_S_URL

from .contracts import (
    ARMS,
    BASELINE_ARM,
    CANDIDATE_ARM,
    CASE_ARTIFACT_TYPE,
    CASE_SCHEMA_VERSION,
    E7_A_INPUT_PROFILE,
    E7_ABSTRACTIVE_GROUNDING_CLAIM,
    E7_B_INPUT_PROFILE,
    E7_BYTE_GROUNDED_CLAIM,
    E7_C_INPUT_PROFILE,
    E7_NO_CONSTRUCTOR_RECEIPTS,
    E7_PROTOCOL_EVIDENCE_FIELDS,
    E7_RAW_GROUNDING_CLAIM,
    HELDOUT_CONFIRMATION_ARTIFACT_TYPE,
    HELDOUT_CONFIRMATION_SCHEMA_VERSION,
    HELDOUT_PREREGISTRATION_ARTIFACT_TYPE,
    HELDOUT_PREREGISTRATION_SCHEMA_VERSION,
    INTRINSICALLY_INELIGIBLE_CANDIDATE_PROFILES,
    MAX_TYPE_REGRESSION,
    PRIMARY_PROMPT_BUDGET,
    PROTOCOL_VERSION,
    RECEIPT_AUTHENTICATION,
    REPORT_ARTIFACT_TYPE,
    REPORT_SCHEMA_VERSION,
    RUN_ARTIFACT_TYPE,
    RUN_SCHEMA_VERSION,
    SELECTION_INPUT_PROFILES,
    LongMemEvalSelectionEvidenceError,
    finite_number,
    fixed_protocol,
    integer,
    receipt_envelope_sha256,
    required_text,
    selection_input_profile_digest,
    sha256_bytes,
    sha256_json,
    sha256_text,
    sha256_utf8,
)
from .metrics import (
    ArmOutcome,
    PairedQACase,
    baseline_dominates_candidate,
    context_summary,
    efficiency_summary,
    paired_qa_summary,
    qa_by_question_type,
)

EXPECTED_QUESTION_COUNT = 500

_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "protocol_version",
        "created_at_utc",
        "experiment_id",
        "protocol",
        "dataset",
        "evidence",
        "baseline_cell",
        "candidate_cell",
        "prompt_serializer",
        "reader_identity",
        "judge_identity",
        "receipt_authentication",
    }
)
_DATASET_BINDING_FIELDS = frozenset({"path", "bytes", "sha256", "questions"})
_EVIDENCE_BINDING_FIELDS = frozenset({"path", "bytes", "sha256", "rows"})
_FILE_BINDING_FIELDS = frozenset({"path", "bytes", "sha256"})
_CELL_IDENTITY_FIELDS = frozenset(
    {
        "name",
        "version",
        "artifact_sha256",
        "selection_input_profile",
        "selection_input_profile_sha256",
    }
)
_PROMPT_SERIALIZER_FIELDS = frozenset(
    {
        "name",
        "version",
        "artifact_sha256",
        "tokenizer",
        "tokenizer_revision",
        "tokenizer_artifact_sha256",
    }
)
_MODEL_IDENTITY_FIELDS = frozenset(
    {
        "provider",
        "model",
        "revision",
        "deployment_sha256",
        "tokenizer",
        "tokenizer_revision",
        "tokenizer_artifact_sha256",
        "prompt_template_sha256",
    }
)
_CASE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "protocol_version",
        "case_index",
        "question_id",
        "question_type",
        "dataset_record_sha256",
        "question_sha256",
        "question_date_sha256",
        "reference_answer_sha256",
        "source_corpus_sha256",
        BASELINE_ARM,
        CANDIDATE_ARM,
    }
)
_ARM_FIELDS = frozenset(
    {
        "arm",
        "cell_identity_sha256",
        "selection",
        "candidate_pool",
        "delivered_context",
        "reader",
        "judge",
        "accounting",
    }
)
_SELECTION_FIELDS = frozenset(
    {
        "query_sha256",
        "source_corpus_sha256",
        "input_artifact_sha256",
        "trace_sha256",
        "input_profile",
        "input_profile_sha256",
        "input_field_names",
        "gold_fields_used",
        "protocol_evidence",
        "protocol_evidence_sha256",
    }
)
_CANDIDATE_POOL_FIELDS = frozenset({"unit", "candidate_ids", "candidate_pool_sha256"})
_DELIVERED_CONTEXT_FIELDS = frozenset({"candidate_ids", "context_sha256"})
_READER_FIELDS = frozenset(
    {
        "identity_sha256",
        "serializer_identity_sha256",
        "budget_tokens",
        "prompt_material_sha256",
        "prompt_sha256",
        "tokenized_prompt_sha256",
        "prompt_tokens",
        "answer_sha256",
        "request_id",
        "provider_request_id",
        "request_sha256",
        "response_sha256",
        "receipt_envelope_sha256",
        "receipt_authentication",
    }
)
_JUDGE_FIELDS = frozenset(
    {
        "identity_sha256",
        "prompt_material_sha256",
        "prompt_sha256",
        "response_sha256",
        "request_id",
        "provider_request_id",
        "request_sha256",
        "reference_answer_sha256",
        "model_answer_sha256",
        "label",
        "receipt_envelope_sha256",
        "receipt_authentication",
    }
)
_ACCOUNTING_FIELDS = frozenset({"embedding", "reranker", "constructor", "reader", "judge"})
_EMBEDDING_ACCOUNTING_FIELDS = frozenset(
    {"calls", "ingestion_tokens", "query_tokens", "cost_usd", "latency_ms"}
)
_GENERIC_ACCOUNTING_FIELDS = frozenset(
    {"calls", "input_tokens", "output_tokens", "cost_usd", "latency_ms"}
)
_HELDOUT_PREREGISTRATION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "preregistration_id",
        "registered_at_utc",
        "primary_dataset_sha256",
        "heldout_dataset",
        "protocol_version",
        "baseline_cell_sha256",
        "candidate_cell_sha256",
        "prompt_serializer_sha256",
        "reader_identity_sha256",
        "judge_identity_sha256",
        "evidence_schema",
    }
)
_HELDOUT_CONFIRMATION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "completed_at_utc",
        "preregistration",
        "dataset",
        "evidence",
        "receipt_authentication",
    }
)
_HELDOUT_CASE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "protocol_version",
        "case_index",
        "question_id",
        "question_type",
        "dataset_record_sha256",
        "baseline_cell_sha256",
        "candidate_cell_sha256",
        "prompt_serializer_sha256",
        "reader_identity_sha256",
        "judge_identity_sha256",
        "baseline_label",
        "candidate_label",
        "baseline_receipt_sha256",
        "candidate_receipt_sha256",
    }
)
_HELDOUT_CASE_SCHEMA_VERSION = 1
_HELDOUT_CASE_ARTIFACT_TYPE = "swarmbrain-selection-heldout-paired-label"
_HELDOUT_EVIDENCE_SCHEMA = "paired-binary-labels-with-receipt-digests-v1"
_E7_PROFILES = frozenset({E7_A_INPUT_PROFILE, E7_B_INPUT_PROFILE, E7_C_INPUT_PROFILE})

_IMPLEMENTATION_PATHS = (
    "benchmarks/integrations/longmemeval_selection_report/__init__.py",
    "benchmarks/integrations/longmemeval_selection_report/contracts.py",
    "benchmarks/integrations/longmemeval_selection_report/metrics.py",
    "benchmarks/integrations/longmemeval_selection_report/report.py",
    "scripts/build_longmemeval_selection_report.py",
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise LongMemEvalSelectionEvidenceError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise LongMemEvalSelectionEvidenceError(f"non-finite JSON number {value!r} is forbidden")


def _strict_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, LongMemEvalSelectionEvidenceError) as exc:
        raise LongMemEvalSelectionEvidenceError(f"{label} is not strict JSON: {exc}") from exc


def _strict_object(raw: bytes, *, label: str) -> dict[str, Any]:
    value = _strict_json(raw, label=label)
    if not isinstance(value, dict):
        raise LongMemEvalSelectionEvidenceError(f"{label} must contain one JSON object")
    return value


def _strict_jsonl(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    if b"\r" in raw:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must use byte-exact LF line endings; CR and CRLF are forbidden"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise LongMemEvalSelectionEvidenceError(f"{label} is not UTF-8") from exc
    if not text or not text.endswith("\n"):
        raise LongMemEvalSelectionEvidenceError(f"{label} must be non-empty and newline terminated")
    lines = text.split("\n")
    if lines[-1] != "":
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must have exactly one terminal LF delimiter"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines[:-1], start=1):
        if not line:
            raise LongMemEvalSelectionEvidenceError(f"{label}:{line_number} cannot be blank")
        value = _strict_json(line.encode("utf-8"), label=f"{label}:{line_number}")
        if not isinstance(value, dict):
            raise LongMemEvalSelectionEvidenceError(f"{label}:{line_number} must be an object")
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
        raise LongMemEvalSelectionEvidenceError(f"{label} must be a repository-local relative path")
    if ".." in supplied.parts:
        raise LongMemEvalSelectionEvidenceError(f"{label} cannot contain '..'")
    root = root.resolve()
    candidate = supplied if supplied.is_absolute() else root / supplied
    current = Path(candidate.anchor) if candidate.is_absolute() else root
    parts = candidate.parts[1:] if candidate.is_absolute() else candidate.parts
    for part in parts:
        current /= part
        if current.is_symlink():
            raise LongMemEvalSelectionEvidenceError(f"{label} cannot traverse symbolic links")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LongMemEvalSelectionEvidenceError(f"{label} is missing") from exc
    if not resolved.is_file():
        raise LongMemEvalSelectionEvidenceError(f"{label} must resolve to a regular file")
    if require_relative and not resolved.is_relative_to(root):
        raise LongMemEvalSelectionEvidenceError(f"{label} must remain inside the artifact root")
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
        raise LongMemEvalSelectionEvidenceError(
            "output must be an artifact-root-relative path without '..'"
        )
    root = root.resolve()
    candidate = root / supplied
    if not candidate.resolve().is_relative_to(root):
        raise LongMemEvalSelectionEvidenceError("output must remain inside the artifact root")
    current = root
    for part in supplied.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise LongMemEvalSelectionEvidenceError("output cannot traverse symbolic links")
    if candidate.is_symlink():
        raise LongMemEvalSelectionEvidenceError("output cannot be a symbolic link")
    if candidate.exists():
        if not candidate.is_file():
            raise LongMemEvalSelectionEvidenceError("output must be a regular file")
        if any(os.path.samefile(candidate, input_path) for input_path in protected):
            raise LongMemEvalSelectionEvidenceError(
                "output cannot share an inode with a protected input artifact"
            )
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite selection QA report: {candidate}")
        existing = _strict_object(candidate.read_bytes(), label="existing output report")
        if existing.get("artifact_type") != REPORT_ARTIFACT_TYPE:
            raise LongMemEvalSelectionEvidenceError(
                "--force can replace only an existing LongMemEval selection QA report"
            )
    return candidate


def _atomic_write_report(path: Path, report: dict[str, Any]) -> None:
    encoded = (
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_utc(value: Any, *, label: str) -> datetime:
    text = required_text(value, label=label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LongMemEvalSelectionEvidenceError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise LongMemEvalSelectionEvidenceError(f"{label} must carry the UTC offset")
    return parsed


def _identity(value: Any, *, fields: frozenset[str], label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != fields:
        raise LongMemEvalSelectionEvidenceError(f"{label} must contain exactly {sorted(fields)}")
    output: dict[str, str] = {}
    for field in sorted(fields):
        if field.endswith("sha256"):
            output[field] = sha256_text(value.get(field), label=f"{label}.{field}")
        else:
            output[field] = required_text(value.get(field), label=f"{label}.{field}")
    return output


def _cell_identity(value: Any, *, label: str) -> dict[str, str]:
    output = _identity(value, fields=_CELL_IDENTITY_FIELDS, label=label)
    profile = output["selection_input_profile"]
    if profile not in SELECTION_INPUT_PROFILES:
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.selection_input_profile is not a frozen E1/E2/E6/E7 profile"
        )
    expected_digest = selection_input_profile_digest(profile)
    if output["selection_input_profile_sha256"] != expected_digest:
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.selection_input_profile_sha256 does not bind its exact ordered allowlist"
        )
    return output


def _validate_serializer_reader_tokenizer(
    prompt_serializer: dict[str, Any],
    reader_identity: dict[str, Any],
) -> None:
    fields = ("tokenizer", "tokenizer_revision", "tokenizer_artifact_sha256")
    serializer_tuple = tuple(prompt_serializer.get(field) for field in fields)
    reader_tuple = tuple(reader_identity.get(field) for field in fields)
    if serializer_tuple != reader_tuple:
        raise LongMemEvalSelectionEvidenceError(
            "prompt serializer tokenizer/model/revision/artifact tuple must exactly equal "
            "the reader tokenizer tuple"
        )


def _bound_artifact(
    value: Any,
    *,
    root: Path,
    fields: frozenset[str],
    label: str,
    count_field: str,
) -> tuple[Path, str, bytes, int]:
    if not isinstance(value, dict) or set(value) != fields:
        raise LongMemEvalSelectionEvidenceError(f"{label} must bind exactly {sorted(fields)}")
    raw_path = required_text(value.get("path"), label=f"{label}.path")
    path, relative = _resolve_input(
        raw_path,
        root=root,
        label=f"{label}.path",
        require_relative=True,
    )
    if raw_path != relative:
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.path must use canonical artifact-root-relative POSIX form"
        )
    raw = path.read_bytes()
    if integer(value.get("bytes"), label=f"{label}.bytes") != len(raw):
        raise LongMemEvalSelectionEvidenceError(f"{label}.bytes does not match the file")
    if sha256_text(value.get("sha256"), label=f"{label}.sha256") != sha256_bytes(raw):
        raise LongMemEvalSelectionEvidenceError(f"{label}.sha256 does not match the file")
    count = integer(value.get(count_field), label=f"{label}.{count_field}", minimum=1)
    return path, relative, raw, count


def _bound_file(
    value: Any,
    *,
    root: Path,
    label: str,
) -> tuple[Path, str, bytes]:
    if not isinstance(value, dict) or set(value) != _FILE_BINDING_FIELDS:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must bind exactly {sorted(_FILE_BINDING_FIELDS)}"
        )
    raw_path = required_text(value.get("path"), label=f"{label}.path")
    path, relative = _resolve_input(
        raw_path,
        root=root,
        label=f"{label}.path",
        require_relative=True,
    )
    if raw_path != relative:
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.path must use canonical artifact-root-relative POSIX form"
        )
    raw = path.read_bytes()
    if integer(value.get("bytes"), label=f"{label}.bytes") != len(raw):
        raise LongMemEvalSelectionEvidenceError(f"{label}.bytes does not match the file")
    if sha256_text(value.get("sha256"), label=f"{label}.sha256") != sha256_bytes(raw):
        raise LongMemEvalSelectionEvidenceError(f"{label}.sha256 does not match the file")
    return path, relative, raw


def implementation_fingerprint(code_root: str | Path) -> dict[str, Any]:
    root = Path(code_root).resolve()
    files: dict[str, str] = {}
    for relative in _IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise LongMemEvalSelectionEvidenceError(
                f"compiler implementation file is missing or unsafe: {relative}"
            )
        files[relative] = sha256_bytes(path.read_bytes())
    return {"tree_sha256": sha256_json(files), "files": files}


def build_run_manifest(
    *,
    created_at_utc: str,
    experiment_id: str,
    dataset_path: str | Path,
    evidence_path: str | Path,
    baseline_cell: dict[str, Any],
    candidate_cell: dict[str, Any],
    prompt_serializer: dict[str, Any],
    reader_identity: dict[str, Any],
    judge_identity: dict[str, Any],
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Build the unsigned integrity manifest an external runner must persist."""

    root = Path(artifact_root).resolve()
    _validate_utc(created_at_utc, label="created_at_utc")
    required_text(experiment_id, label="experiment_id")
    checked_baseline = _cell_identity(baseline_cell, label="baseline_cell")
    checked_candidate = _cell_identity(candidate_cell, label="candidate_cell")
    if checked_baseline["artifact_sha256"] == checked_candidate["artifact_sha256"]:
        raise LongMemEvalSelectionEvidenceError(
            "baseline and candidate cell artifact_sha256 values must differ"
        )
    checked_serializer = _identity(
        prompt_serializer, fields=_PROMPT_SERIALIZER_FIELDS, label="prompt_serializer"
    )
    checked_reader = _identity(
        reader_identity, fields=_MODEL_IDENTITY_FIELDS, label="reader_identity"
    )
    _validate_serializer_reader_tokenizer(checked_serializer, checked_reader)
    _identity(judge_identity, fields=_MODEL_IDENTITY_FIELDS, label="judge_identity")
    dataset_resolved, dataset_display = _resolve_input(
        dataset_path,
        root=root,
        label="dataset_path",
        require_relative=True,
    )
    evidence_resolved, evidence_display = _resolve_input(
        evidence_path,
        root=root,
        label="evidence_path",
        require_relative=True,
    )
    dataset_raw = dataset_resolved.read_bytes()
    evidence_raw = evidence_resolved.read_bytes()
    dataset_value = _strict_json(dataset_raw, label="dataset")
    evidence_rows = _strict_jsonl(evidence_raw, label="evidence")
    if not isinstance(dataset_value, list) or not dataset_value:
        raise LongMemEvalSelectionEvidenceError("dataset must be a non-empty JSON array")
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "artifact_type": RUN_ARTIFACT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": created_at_utc,
        "experiment_id": experiment_id,
        "protocol": fixed_protocol(),
        "dataset": {
            "path": dataset_display,
            "bytes": len(dataset_raw),
            "sha256": sha256_bytes(dataset_raw),
            "questions": len(dataset_value),
        },
        "evidence": {
            "path": evidence_display,
            "bytes": len(evidence_raw),
            "sha256": sha256_bytes(evidence_raw),
            "rows": len(evidence_rows),
        },
        "baseline_cell": baseline_cell,
        "candidate_cell": candidate_cell,
        "prompt_serializer": prompt_serializer,
        "reader_identity": reader_identity,
        "judge_identity": judge_identity,
        "receipt_authentication": RECEIPT_AUTHENTICATION,
    }


def _load_run(
    run_raw: bytes,
    *,
    root: Path,
) -> tuple[dict[str, Any], Path, bytes, Path, bytes]:
    run = _strict_object(run_raw, label="run manifest")
    if set(run) != _RUN_FIELDS:
        raise LongMemEvalSelectionEvidenceError("run manifest fields differ from the schema")
    expected = {
        "schema_version": RUN_SCHEMA_VERSION,
        "artifact_type": RUN_ARTIFACT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "receipt_authentication": RECEIPT_AUTHENTICATION,
    }
    for field, wanted in expected.items():
        if type(run.get(field)) is not type(wanted) or run.get(field) != wanted:
            raise LongMemEvalSelectionEvidenceError(f"run.{field} must be {wanted!r}")
    _validate_utc(run.get("created_at_utc"), label="run.created_at_utc")
    required_text(run.get("experiment_id"), label="run.experiment_id")
    if run.get("protocol") != fixed_protocol():
        raise LongMemEvalSelectionEvidenceError("run.protocol differs from the frozen design")
    checked_baseline = _cell_identity(run.get("baseline_cell"), label="baseline_cell")
    checked_candidate = _cell_identity(run.get("candidate_cell"), label="candidate_cell")
    if run.get("baseline_cell") == run.get("candidate_cell"):
        raise LongMemEvalSelectionEvidenceError("baseline and candidate cells must be distinct")
    if checked_baseline["artifact_sha256"] == checked_candidate["artifact_sha256"]:
        raise LongMemEvalSelectionEvidenceError(
            "baseline and candidate cell artifact_sha256 values must differ"
        )
    checked_serializer = _identity(
        run.get("prompt_serializer"),
        fields=_PROMPT_SERIALIZER_FIELDS,
        label="prompt_serializer",
    )
    checked_reader = _identity(
        run.get("reader_identity"), fields=_MODEL_IDENTITY_FIELDS, label="reader_identity"
    )
    _validate_serializer_reader_tokenizer(checked_serializer, checked_reader)
    _identity(run.get("judge_identity"), fields=_MODEL_IDENTITY_FIELDS, label="judge_identity")
    dataset_path, _, dataset_raw, dataset_count = _bound_artifact(
        run.get("dataset"),
        root=root,
        fields=_DATASET_BINDING_FIELDS,
        label="run.dataset",
        count_field="questions",
    )
    evidence_path, _, evidence_raw, evidence_count = _bound_artifact(
        run.get("evidence"),
        root=root,
        fields=_EVIDENCE_BINDING_FIELDS,
        label="run.evidence",
        count_field="rows",
    )
    if dataset_count != evidence_count:
        raise LongMemEvalSelectionEvidenceError(
            "run dataset and evidence counts do not form complete paired coverage"
        )
    return run, dataset_path, dataset_raw, evidence_path, evidence_raw


def _load_dataset(
    raw: bytes,
    *,
    expected_sha256: str,
    expected_questions: int,
) -> list[dict[str, Any]]:
    if sha256_bytes(raw) != expected_sha256:
        raise LongMemEvalSelectionEvidenceError(
            "dataset bytes do not match the expected pinned digest"
        )
    value = _strict_json(raw, label="dataset")
    if not isinstance(value, list) or len(value) != expected_questions:
        raise LongMemEvalSelectionEvidenceError(
            f"dataset must contain exactly {expected_questions} questions"
        )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(value):
        label = f"dataset[{index}]"
        if not isinstance(record, dict):
            raise LongMemEvalSelectionEvidenceError(f"{label} must be an object")
        question_id = required_text(record.get("question_id"), label=f"{label}.question_id")
        if question_id in seen:
            raise LongMemEvalSelectionEvidenceError(f"{label} repeats question_id {question_id!r}")
        seen.add(question_id)
        required_text(record.get("question_type"), label=f"{label}.question_type")
        question = record.get("question")
        if not isinstance(question, str) or not question:
            raise LongMemEvalSelectionEvidenceError(f"{label}.question must be non-empty text")
        try:
            question.encode("utf-8")
        except UnicodeError as exc:
            raise LongMemEvalSelectionEvidenceError(
                f"{label}.question must be valid UTF-8"
            ) from exc
        question_date = record.get("question_date")
        if not isinstance(question_date, str) or not question_date:
            raise LongMemEvalSelectionEvidenceError(f"{label}.question_date must be non-empty text")
        if "answer" not in record:
            raise LongMemEvalSelectionEvidenceError(f"{label} lacks the reference answer")
        # This is both a finiteness/UTF-8 check and the canonical reference hash domain.
        sha256_json(record["answer"])

        session_ids = record.get("haystack_session_ids")
        dates = record.get("haystack_dates")
        sessions = record.get("haystack_sessions")
        if not all(isinstance(item, list) for item in (session_ids, dates, sessions)):
            raise LongMemEvalSelectionEvidenceError(f"{label} haystack arrays must be lists")
        if not len(session_ids) == len(dates) == len(sessions) or not session_ids:
            raise LongMemEvalSelectionEvidenceError(
                f"{label} haystack arrays must be non-empty and aligned"
            )
        checked_ids = [
            required_text(item, label=f"{label}.haystack_session_ids[{position}]")
            for position, item in enumerate(session_ids)
        ]
        if len(checked_ids) != len(set(checked_ids)):
            raise LongMemEvalSelectionEvidenceError(f"{label} repeats a haystack session ID")
        for position, (date, turns) in enumerate(zip(dates, sessions, strict=True)):
            required_text(date, label=f"{label}.haystack_dates[{position}]")
            if not isinstance(turns, list) or not turns:
                raise LongMemEvalSelectionEvidenceError(
                    f"{label}.haystack_sessions[{position}] must be a non-empty turn list"
                )
            for turn_position, turn in enumerate(turns):
                turn_label = f"{label}.haystack_sessions[{position}][{turn_position}]"
                if not isinstance(turn, dict):
                    raise LongMemEvalSelectionEvidenceError(f"{turn_label} must be an object")
                required_text(turn.get("role"), label=f"{turn_label}.role")
                content = turn.get("content")
                if not isinstance(content, str):
                    raise LongMemEvalSelectionEvidenceError(f"{turn_label}.content must be text")
                try:
                    content.encode("utf-8")
                except UnicodeError as exc:
                    raise LongMemEvalSelectionEvidenceError(
                        f"{turn_label}.content must be valid UTF-8"
                    ) from exc

        answer_session_ids = record.get("answer_session_ids")
        if answer_session_ids is not None:
            if not isinstance(answer_session_ids, list):
                raise LongMemEvalSelectionEvidenceError(
                    f"{label}.answer_session_ids must be a list when supplied"
                )
            checked_answers = [
                required_text(item, label=f"{label}.answer_session_ids[{position}]")
                for position, item in enumerate(answer_session_ids)
            ]
            if len(checked_answers) != len(set(checked_answers)):
                raise LongMemEvalSelectionEvidenceError(
                    f"{label}.answer_session_ids cannot contain duplicates"
                )
            unknown = set(checked_answers) - set(checked_ids)
            if unknown:
                raise LongMemEvalSelectionEvidenceError(
                    f"{label}.answer_session_ids point outside the haystack"
                )
        records.append(record)
    return records


def _source_corpus_sha256(record: dict[str, Any]) -> str:
    return sha256_json(
        {
            "haystack_session_ids": record["haystack_session_ids"],
            "haystack_dates": record["haystack_dates"],
            "haystack_sessions": record["haystack_sessions"],
        }
    )


def _string_list(value: Any, *, label: str, allow_empty: bool) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        raise LongMemEvalSelectionEvidenceError(f"{label} must be a {qualifier} list")
    output = [required_text(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    if len(output) != len(set(output)):
        raise LongMemEvalSelectionEvidenceError(f"{label} cannot contain duplicates")
    return output


def _candidate_parent_session(
    candidate_id: str,
    *,
    unit: str,
    record: dict[str, Any],
    label: str,
) -> str:
    session_ids = record["haystack_session_ids"]
    if unit == "session":
        if candidate_id not in session_ids:
            raise LongMemEvalSelectionEvidenceError(
                f"{label} names a session outside the question haystack"
            )
        return candidate_id
    if unit != "turn":
        raise LongMemEvalSelectionEvidenceError(
            f"{label} candidate unit must be 'session' or 'turn'"
        )
    try:
        parsed = json.loads(candidate_id)
    except json.JSONDecodeError as exc:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} is not a canonical immutable turn ID"
        ) from exc
    if (
        not isinstance(parsed, list)
        or len(parsed) != 3
        or parsed[0] != record["question_id"]
        or isinstance(parsed[1], bool)
        or not isinstance(parsed[1], int)
        or parsed[1] < 0
        or isinstance(parsed[2], bool)
        or not isinstance(parsed[2], int)
        or parsed[2] < 0
    ):
        raise LongMemEvalSelectionEvidenceError(
            f"{label} is not a question-local immutable turn ID"
        )
    if candidate_id != json.dumps(parsed, ensure_ascii=False, separators=(",", ":")):
        raise LongMemEvalSelectionEvidenceError(f"{label} is not canonically serialized")
    session_position = parsed[1]
    turn_position = parsed[2]
    sessions = record["haystack_sessions"]
    if session_position >= len(session_ids) or turn_position >= len(sessions[session_position]):
        raise LongMemEvalSelectionEvidenceError(f"{label} points outside the source corpus")
    return str(session_ids[session_position])


def _candidate_pool(
    value: Any,
    *,
    record: dict[str, Any],
    label: str,
) -> tuple[str, list[str], list[str], str]:
    if not isinstance(value, dict) or set(value) != _CANDIDATE_POOL_FIELDS:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must contain exactly {sorted(_CANDIDATE_POOL_FIELDS)}"
        )
    unit = required_text(value.get("unit"), label=f"{label}.unit")
    candidate_ids = _string_list(
        value.get("candidate_ids"), label=f"{label}.candidate_ids", allow_empty=False
    )
    parents = [
        _candidate_parent_session(
            candidate_id,
            unit=unit,
            record=record,
            label=f"{label}.candidate_ids[{index}]",
        )
        for index, candidate_id in enumerate(candidate_ids)
    ]
    expected_digest = sha256_json({"unit": unit, "candidate_ids": candidate_ids})
    if (
        sha256_text(value.get("candidate_pool_sha256"), label=f"{label}.candidate_pool_sha256")
        != expected_digest
    ):
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.candidate_pool_sha256 does not match the exact ordered pool"
        )
    return unit, candidate_ids, parents, expected_digest


def _delivered_context(
    value: Any,
    *,
    pool_ids: list[str],
    pool_parents: list[str],
    label: str,
) -> tuple[list[str], list[str], str]:
    if not isinstance(value, dict) or set(value) != _DELIVERED_CONTEXT_FIELDS:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must contain exactly {sorted(_DELIVERED_CONTEXT_FIELDS)}"
        )
    candidate_ids = _string_list(
        value.get("candidate_ids"), label=f"{label}.candidate_ids", allow_empty=True
    )
    positions = {candidate_id: index for index, candidate_id in enumerate(pool_ids)}
    try:
        selected_positions = [positions[candidate_id] for candidate_id in candidate_ids]
    except KeyError as exc:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} contains a candidate outside the selected pool"
        ) from exc
    if selected_positions != sorted(selected_positions):
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must preserve pool order under skip-and-continue packing"
        )
    context_sha256 = sha256_text(value.get("context_sha256"), label=f"{label}.context_sha256")
    return (
        candidate_ids,
        [pool_parents[position] for position in selected_positions],
        context_sha256,
    )


def _selection_protocol_evidence(
    value: Any,
    *,
    profile: str,
    input_artifact_sha256: str,
    declared_sha256: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LongMemEvalSelectionEvidenceError(f"{label} must be an object")
    expected_digest = sha256_json(value)
    if sha256_text(declared_sha256, label=f"{label}_sha256") != expected_digest:
        raise LongMemEvalSelectionEvidenceError(
            f"{label}_sha256 does not bind the exact protocol evidence"
        )
    if profile not in _E7_PROFILES:
        if value:
            raise LongMemEvalSelectionEvidenceError(
                f"{label} is permitted only for frozen E7 profiles"
            )
        return {}
    if set(value) != set(E7_PROTOCOL_EVIDENCE_FIELDS):
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must contain exactly {sorted(E7_PROTOCOL_EVIDENCE_FIELDS)}"
        )
    digests = {
        field: sha256_text(value.get(field), label=f"{label}.{field}")
        for field in (
            "source_e1_b_selection_trace_sha256",
            "preflight_manifest_sha256",
            "window_batch_trace_sha256",
            "normalized_constructor_receipts_sha256",
            "reader_context_sha256",
        )
    }
    if digests["source_e1_b_selection_trace_sha256"] != input_artifact_sha256:
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.source_e1_b_selection_trace_sha256 differs from "
            "selection.input_artifact_sha256"
        )
    receipt_count = integer(
        value.get("normalized_constructor_receipt_count"),
        label=f"{label}.normalized_constructor_receipt_count",
    )
    receipt_authentication = required_text(
        value.get("constructor_receipt_authentication"),
        label=f"{label}.constructor_receipt_authentication",
    )
    grounding_claim = required_text(value.get("grounding_claim"), label=f"{label}.grounding_claim")
    faithfulness = value.get("authenticated_faithfulness_evidence_sha256")
    if faithfulness is not None:
        sha256_text(
            faithfulness,
            label=f"{label}.authenticated_faithfulness_evidence_sha256",
        )
        raise LongMemEvalSelectionEvidenceError(
            "selection QA protocol v2 cannot authenticate an E7 faithfulness artifact; "
            "a new protocol with a verifying trust boundary is required"
        )

    if profile == E7_A_INPUT_PROFILE:
        if receipt_count != 0:
            raise LongMemEvalSelectionEvidenceError(
                f"{label} E7-A must bind zero normalized constructor receipts"
            )
        if digests["normalized_constructor_receipts_sha256"] != sha256_json([]):
            raise LongMemEvalSelectionEvidenceError(
                f"{label} E7-A constructor-receipt digest must bind the empty list"
            )
        if receipt_authentication != E7_NO_CONSTRUCTOR_RECEIPTS:
            raise LongMemEvalSelectionEvidenceError(
                f"{label} E7-A constructor receipt authentication is invalid"
            )
        if grounding_claim != E7_RAW_GROUNDING_CLAIM:
            raise LongMemEvalSelectionEvidenceError(
                f"{label} E7-A must claim raw source-byte grounding"
            )
    else:
        if receipt_count < 1:
            raise LongMemEvalSelectionEvidenceError(
                f"{label} E7-B/C must bind at least one normalized constructor receipt"
            )
        if receipt_authentication != RECEIPT_AUTHENTICATION:
            raise LongMemEvalSelectionEvidenceError(
                f"{label} E7-B/C receipts must remain {RECEIPT_AUTHENTICATION!r}"
            )
        allowed_claims = {E7_BYTE_GROUNDED_CLAIM}
        if profile == E7_B_INPUT_PROFILE:
            allowed_claims.add(E7_ABSTRACTIVE_GROUNDING_CLAIM)
        if grounding_claim not in allowed_claims:
            qualifier = "byte-grounded" if profile == E7_C_INPUT_PROFILE else "registered"
            raise LongMemEvalSelectionEvidenceError(
                f"{label} must carry a {qualifier} E7 grounding claim"
            )
    return {
        **value,
        "normalized_constructor_receipt_count": receipt_count,
        "constructor_receipt_authentication": receipt_authentication,
        "grounding_claim": grounding_claim,
    }


def _selection(
    value: Any,
    *,
    query_sha256: str,
    source_corpus_sha256: str,
    cell_identity: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _SELECTION_FIELDS:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must contain exactly {sorted(_SELECTION_FIELDS)}"
        )
    if sha256_text(value.get("query_sha256"), label=f"{label}.query_sha256") != query_sha256:
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.query_sha256 disagrees with the exact dataset question"
        )
    if (
        sha256_text(value.get("source_corpus_sha256"), label=f"{label}.source_corpus_sha256")
        != source_corpus_sha256
    ):
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.source_corpus_sha256 changes the shared question corpus"
        )
    input_artifact_sha256 = sha256_text(
        value.get("input_artifact_sha256"), label=f"{label}.input_artifact_sha256"
    )
    sha256_text(value.get("trace_sha256"), label=f"{label}.trace_sha256")
    expected_profile = str(cell_identity["selection_input_profile"])
    expected_profile_sha256 = str(cell_identity["selection_input_profile_sha256"])
    if value.get("input_profile") != expected_profile:
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.input_profile differs from the arm's cell identity"
        )
    if value.get("input_profile_sha256") != expected_profile_sha256:
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.input_profile_sha256 differs from the arm's cell identity"
        )
    expected_fields = list(SELECTION_INPUT_PROFILES[expected_profile])
    if value.get("input_field_names") != expected_fields:
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.input_field_names must exactly equal the registered ordered allowlist"
        )
    if value.get("gold_fields_used") is not False:
        raise LongMemEvalSelectionEvidenceError(f"{label}.gold_fields_used must be exactly false")
    return _selection_protocol_evidence(
        value.get("protocol_evidence"),
        profile=expected_profile,
        input_artifact_sha256=input_artifact_sha256,
        declared_sha256=value.get("protocol_evidence_sha256"),
        label=f"{label}.protocol_evidence",
    )


def _embedding_accounting(value: Any, *, label: str) -> dict[str, int | float]:
    if not isinstance(value, dict) or set(value) != _EMBEDDING_ACCOUNTING_FIELDS:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must contain exactly {sorted(_EMBEDDING_ACCOUNTING_FIELDS)}"
        )
    output: dict[str, int | float] = {
        "calls": integer(value.get("calls"), label=f"{label}.calls"),
        "ingestion_tokens": integer(
            value.get("ingestion_tokens"), label=f"{label}.ingestion_tokens"
        ),
        "query_tokens": integer(value.get("query_tokens"), label=f"{label}.query_tokens"),
        "cost_usd": finite_number(value.get("cost_usd"), label=f"{label}.cost_usd", minimum=0),
        "latency_ms": finite_number(
            value.get("latency_ms"), label=f"{label}.latency_ms", minimum=0
        ),
    }
    if output["calls"] == 0 and any(
        output[field] != 0
        for field in ("ingestion_tokens", "query_tokens", "cost_usd", "latency_ms")
    ):
        raise LongMemEvalSelectionEvidenceError(f"{label} reports usage for zero calls")
    return output


def _generic_accounting(
    value: Any,
    *,
    label: str,
    required_calls: int | None = None,
) -> dict[str, int | float]:
    if not isinstance(value, dict) or set(value) != _GENERIC_ACCOUNTING_FIELDS:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must contain exactly {sorted(_GENERIC_ACCOUNTING_FIELDS)}"
        )
    output: dict[str, int | float] = {
        "calls": integer(value.get("calls"), label=f"{label}.calls"),
        "input_tokens": integer(value.get("input_tokens"), label=f"{label}.input_tokens"),
        "output_tokens": integer(value.get("output_tokens"), label=f"{label}.output_tokens"),
        "cost_usd": finite_number(value.get("cost_usd"), label=f"{label}.cost_usd", minimum=0),
        "latency_ms": finite_number(
            value.get("latency_ms"), label=f"{label}.latency_ms", minimum=0
        ),
    }
    if required_calls is not None and output["calls"] != required_calls:
        raise LongMemEvalSelectionEvidenceError(f"{label}.calls must be exactly {required_calls}")
    if output["calls"] == 0 and any(
        output[field] != 0 for field in ("input_tokens", "output_tokens", "cost_usd", "latency_ms")
    ):
        raise LongMemEvalSelectionEvidenceError(f"{label} reports usage for zero calls")
    return output


def _accounting(value: Any, *, label: str) -> dict[str, dict[str, int | float]]:
    if not isinstance(value, dict) or set(value) != _ACCOUNTING_FIELDS:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must contain exactly {sorted(_ACCOUNTING_FIELDS)}"
        )
    return {
        "embedding": _embedding_accounting(value.get("embedding"), label=f"{label}.embedding"),
        "reranker": _generic_accounting(value.get("reranker"), label=f"{label}.reranker"),
        "constructor": _generic_accounting(value.get("constructor"), label=f"{label}.constructor"),
        "reader": _generic_accounting(
            value.get("reader"), label=f"{label}.reader", required_calls=1
        ),
        "judge": _generic_accounting(value.get("judge"), label=f"{label}.judge", required_calls=1),
    }


def _reader(
    value: Any,
    *,
    case_index: int,
    question_id: str,
    arm: str,
    question_sha256: str,
    question_date_sha256: str,
    candidate_pool_sha256: str,
    delivered_candidate_ids: list[str],
    context_sha256: str,
    cell_identity_sha256: str,
    serializer_identity_sha256: str,
    reader_identity_sha256: str,
    accounting: dict[str, int | float],
    label: str,
    client_request_ids: set[str],
    provider_request_ids: set[str],
    receipt_envelope_digests: set[str],
) -> tuple[int, str]:
    if not isinstance(value, dict) or set(value) != _READER_FIELDS:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must contain exactly {sorted(_READER_FIELDS)}"
        )
    expected_scalars = {
        "identity_sha256": reader_identity_sha256,
        "serializer_identity_sha256": serializer_identity_sha256,
        "budget_tokens": PRIMARY_PROMPT_BUDGET,
        "receipt_authentication": RECEIPT_AUTHENTICATION,
    }
    for field, wanted in expected_scalars.items():
        if type(value.get(field)) is not type(wanted) or value.get(field) != wanted:
            raise LongMemEvalSelectionEvidenceError(f"{label}.{field} must be {wanted!r}")
    expected_material = sha256_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "serializer_identity_sha256": serializer_identity_sha256,
            "reader_identity_sha256": reader_identity_sha256,
            "budget_tokens": PRIMARY_PROMPT_BUDGET,
            "question_sha256": question_sha256,
            "question_date_sha256": question_date_sha256,
            "candidate_pool_sha256": candidate_pool_sha256,
            "delivered_candidate_ids": delivered_candidate_ids,
            "context_sha256": context_sha256,
        }
    )
    if (
        sha256_text(value.get("prompt_material_sha256"), label=f"{label}.prompt_material_sha256")
        != expected_material
    ):
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.prompt_material_sha256 does not bind the exact delivered context"
        )
    digests = {}
    for field in (
        "prompt_sha256",
        "tokenized_prompt_sha256",
        "answer_sha256",
        "request_sha256",
        "response_sha256",
    ):
        digests[field] = sha256_text(value.get(field), label=f"{label}.{field}")
    prompt_tokens = integer(value.get("prompt_tokens"), label=f"{label}.prompt_tokens", minimum=1)
    if prompt_tokens > PRIMARY_PROMPT_BUDGET:
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.prompt_tokens exceeds the full 8192-token prompt budget"
        )
    request_id = required_text(value.get("request_id"), label=f"{label}.request_id")
    provider_id = required_text(
        value.get("provider_request_id"), label=f"{label}.provider_request_id"
    )
    if request_id == provider_id:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} provider request ID must differ from the client request ID"
        )
    if request_id in client_request_ids or request_id in provider_request_ids:
        raise LongMemEvalSelectionEvidenceError(f"{label} reuses a receipt request ID")
    if provider_id in provider_request_ids or provider_id in client_request_ids:
        raise LongMemEvalSelectionEvidenceError(f"{label} reuses a receipt request ID")
    expected_envelope = receipt_envelope_sha256(
        case_index=case_index,
        question_id=question_id,
        arm=arm,
        stage="reader",
        identities={
            "cell": cell_identity_sha256,
            "prompt_serializer": serializer_identity_sha256,
            "reader": reader_identity_sha256,
        },
        request_id=request_id,
        provider_request_id=provider_id,
        request_sha256=digests["request_sha256"],
        response_sha256=digests["response_sha256"],
        outcome={
            "prompt_material_sha256": expected_material,
            "prompt_sha256": digests["prompt_sha256"],
            "tokenized_prompt_sha256": digests["tokenized_prompt_sha256"],
            "prompt_tokens": prompt_tokens,
            "answer_sha256": digests["answer_sha256"],
        },
        accounting=accounting,
    )
    if (
        sha256_text(
            value.get("receipt_envelope_sha256"),
            label=f"{label}.receipt_envelope_sha256",
        )
        != expected_envelope
    ):
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.receipt_envelope_sha256 does not bind the complete reader receipt"
        )
    if expected_envelope in receipt_envelope_digests:
        raise LongMemEvalSelectionEvidenceError(f"{label} reuses a receipt envelope digest")
    client_request_ids.add(request_id)
    provider_request_ids.add(provider_id)
    receipt_envelope_digests.add(expected_envelope)
    return prompt_tokens, digests["answer_sha256"]


def _judge(
    value: Any,
    *,
    case_index: int,
    question_id: str,
    arm: str,
    question_sha256: str,
    reference_answer_sha256: str,
    model_answer_sha256: str,
    cell_identity_sha256: str,
    serializer_identity_sha256: str,
    reader_identity_sha256: str,
    judge_identity_sha256: str,
    accounting: dict[str, int | float],
    label: str,
    client_request_ids: set[str],
    provider_request_ids: set[str],
    receipt_envelope_digests: set[str],
) -> bool:
    if not isinstance(value, dict) or set(value) != _JUDGE_FIELDS:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must contain exactly {sorted(_JUDGE_FIELDS)}"
        )
    expected_scalars = {
        "identity_sha256": judge_identity_sha256,
        "reference_answer_sha256": reference_answer_sha256,
        "model_answer_sha256": model_answer_sha256,
        "receipt_authentication": RECEIPT_AUTHENTICATION,
    }
    for field, wanted in expected_scalars.items():
        if type(value.get(field)) is not type(wanted) or value.get(field) != wanted:
            raise LongMemEvalSelectionEvidenceError(f"{label}.{field} must be {wanted!r}")
    expected_material = sha256_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "judge_identity_sha256": judge_identity_sha256,
            "question_sha256": question_sha256,
            "reference_answer_sha256": reference_answer_sha256,
            "model_answer_sha256": model_answer_sha256,
        }
    )
    if (
        sha256_text(value.get("prompt_material_sha256"), label=f"{label}.prompt_material_sha256")
        != expected_material
    ):
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.prompt_material_sha256 does not bind the exact judged answer"
        )
    digests = {
        field: sha256_text(value.get(field), label=f"{label}.{field}")
        for field in ("prompt_sha256", "response_sha256", "request_sha256")
    }
    judged = value.get("label")
    if not isinstance(judged, bool):
        raise LongMemEvalSelectionEvidenceError(f"{label}.label must be a JSON boolean")
    request_id = required_text(value.get("request_id"), label=f"{label}.request_id")
    provider_id = required_text(
        value.get("provider_request_id"), label=f"{label}.provider_request_id"
    )
    if request_id == provider_id:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} provider request ID must differ from the client request ID"
        )
    if request_id in client_request_ids or request_id in provider_request_ids:
        raise LongMemEvalSelectionEvidenceError(f"{label} reuses a receipt request ID")
    if provider_id in provider_request_ids or provider_id in client_request_ids:
        raise LongMemEvalSelectionEvidenceError(f"{label} reuses a receipt request ID")
    expected_envelope = receipt_envelope_sha256(
        case_index=case_index,
        question_id=question_id,
        arm=arm,
        stage="judge",
        identities={
            "cell": cell_identity_sha256,
            "judge": judge_identity_sha256,
            "prompt_serializer": serializer_identity_sha256,
            "reader": reader_identity_sha256,
        },
        request_id=request_id,
        provider_request_id=provider_id,
        request_sha256=digests["request_sha256"],
        response_sha256=digests["response_sha256"],
        outcome={
            "prompt_material_sha256": expected_material,
            "prompt_sha256": digests["prompt_sha256"],
            "reference_answer_sha256": reference_answer_sha256,
            "model_answer_sha256": model_answer_sha256,
            "label": judged,
        },
        accounting=accounting,
    )
    if (
        sha256_text(
            value.get("receipt_envelope_sha256"),
            label=f"{label}.receipt_envelope_sha256",
        )
        != expected_envelope
    ):
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.receipt_envelope_sha256 does not bind the complete judge receipt"
        )
    if expected_envelope in receipt_envelope_digests:
        raise LongMemEvalSelectionEvidenceError(f"{label} reuses a receipt envelope digest")
    client_request_ids.add(request_id)
    provider_request_ids.add(provider_id)
    receipt_envelope_digests.add(expected_envelope)
    return judged


def _arm_outcome(
    value: Any,
    *,
    case_index: int,
    arm: str,
    record: dict[str, Any],
    question_sha256: str,
    question_date_sha256: str,
    reference_answer_sha256: str,
    source_corpus_sha256: str,
    cell_identity: dict[str, Any],
    cell_identity_sha256: str,
    serializer_identity_sha256: str,
    reader_identity_sha256: str,
    judge_identity_sha256: str,
    label: str,
    client_request_ids: set[str],
    provider_request_ids: set[str],
    receipt_envelope_digests: set[str],
) -> ArmOutcome:
    if arm not in ARMS:
        raise LongMemEvalSelectionEvidenceError(f"unknown QA arm {arm!r}")
    if not isinstance(value, dict) or set(value) != _ARM_FIELDS:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must contain exactly {sorted(_ARM_FIELDS)}"
        )
    if value.get("arm") != arm:
        raise LongMemEvalSelectionEvidenceError(f"{label}.arm must be {arm!r}")
    if value.get("cell_identity_sha256") != cell_identity_sha256:
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.cell_identity_sha256 does not match the run manifest"
        )
    protocol_evidence = _selection(
        value.get("selection"),
        query_sha256=question_sha256,
        source_corpus_sha256=source_corpus_sha256,
        cell_identity=cell_identity,
        label=f"{label}.selection",
    )
    _, pool_ids, pool_parents, pool_sha256 = _candidate_pool(
        value.get("candidate_pool"), record=record, label=f"{label}.candidate_pool"
    )
    delivered_ids, delivered_parents, context_sha256 = _delivered_context(
        value.get("delivered_context"),
        pool_ids=pool_ids,
        pool_parents=pool_parents,
        label=f"{label}.delivered_context",
    )
    accounting = _accounting(value.get("accounting"), label=f"{label}.accounting")
    profile = str(cell_identity["selection_input_profile"])
    constructor_calls = int(accounting["constructor"]["calls"])
    if profile in _E7_PROFILES:
        if protocol_evidence["reader_context_sha256"] != context_sha256:
            raise LongMemEvalSelectionEvidenceError(
                f"{label}.selection E7 reader-context digest differs from delivered context"
            )
        if protocol_evidence["normalized_constructor_receipt_count"] != constructor_calls:
            raise LongMemEvalSelectionEvidenceError(
                f"{label}.accounting.constructor.calls differs from normalized E7 receipts"
            )
    elif constructor_calls != 0:
        raise LongMemEvalSelectionEvidenceError(
            f"{label}.accounting.constructor must be zero outside frozen E7 profiles"
        )
    prompt_tokens, model_answer_sha256 = _reader(
        value.get("reader"),
        case_index=case_index,
        question_id=str(record["question_id"]),
        arm=arm,
        question_sha256=question_sha256,
        question_date_sha256=question_date_sha256,
        candidate_pool_sha256=pool_sha256,
        delivered_candidate_ids=delivered_ids,
        context_sha256=context_sha256,
        cell_identity_sha256=cell_identity_sha256,
        serializer_identity_sha256=serializer_identity_sha256,
        reader_identity_sha256=reader_identity_sha256,
        accounting=accounting["reader"],
        label=f"{label}.reader",
        client_request_ids=client_request_ids,
        provider_request_ids=provider_request_ids,
        receipt_envelope_digests=receipt_envelope_digests,
    )
    judged = _judge(
        value.get("judge"),
        case_index=case_index,
        question_id=str(record["question_id"]),
        arm=arm,
        question_sha256=question_sha256,
        reference_answer_sha256=reference_answer_sha256,
        model_answer_sha256=model_answer_sha256,
        cell_identity_sha256=cell_identity_sha256,
        serializer_identity_sha256=serializer_identity_sha256,
        reader_identity_sha256=reader_identity_sha256,
        judge_identity_sha256=judge_identity_sha256,
        accounting=accounting["judge"],
        label=f"{label}.judge",
        client_request_ids=client_request_ids,
        provider_request_ids=provider_request_ids,
        receipt_envelope_digests=receipt_envelope_digests,
    )

    gold_value = record.get("answer_session_ids")
    gold = set(gold_value) if isinstance(gold_value, list) and gold_value else None
    if gold is None:
        any_gold: bool | None = None
        all_gold: bool | None = None
        answer_mrr: float | None = None
    else:
        delivered = set(delivered_parents)
        any_gold = bool(delivered & gold)
        all_gold = gold.issubset(delivered)
        unique_ranked_sessions = list(dict.fromkeys(pool_parents))
        first_rank = next(
            (
                rank
                for rank, parent_session_id in enumerate(unique_ranked_sessions, start=1)
                if parent_session_id in gold
            ),
            None,
        )
        answer_mrr = 0.0 if first_rank is None else 1.0 / first_rank

    operational_latency = sum(
        float(accounting[phase]["latency_ms"])
        for phase in ("embedding", "reranker", "constructor", "reader")
    )
    judge_latency = float(accounting["judge"]["latency_ms"])
    construction_plus_query_cost = sum(
        float(accounting[phase]["cost_usd"]) for phase in ("embedding", "reranker", "constructor")
    )
    reader_cost = float(accounting["reader"]["cost_usd"])
    judge_cost = float(accounting["judge"]["cost_usd"])
    return ArmOutcome(
        correct=judged,
        any_gold_in_context=any_gold,
        all_gold_in_context=all_gold,
        answer_session_mrr=answer_mrr,
        prompt_tokens=prompt_tokens,
        operational_latency_ms=operational_latency,
        end_to_end_latency_ms=operational_latency + judge_latency,
        construction_plus_query_cost_usd=construction_plus_query_cost,
        reader_cost_usd=reader_cost,
        judge_cost_usd=judge_cost,
        total_cost_usd=construction_plus_query_cost + reader_cost + judge_cost,
        accounting=accounting,
    )


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    records: list[dict[str, Any]],
    run: dict[str, Any],
) -> list[PairedQACase]:
    if len(rows) != len(records):
        raise LongMemEvalSelectionEvidenceError(
            "evidence row coverage differs from the pinned dataset"
        )
    baseline_cell_sha256 = sha256_json(run["baseline_cell"])
    candidate_cell_sha256 = sha256_json(run["candidate_cell"])
    serializer_identity_sha256 = sha256_json(run["prompt_serializer"])
    reader_identity_sha256 = sha256_json(run["reader_identity"])
    judge_identity_sha256 = sha256_json(run["judge_identity"])
    client_request_ids: set[str] = set()
    provider_request_ids: set[str] = set()
    receipt_envelope_digests: set[str] = set()
    seen_question_ids: set[str] = set()
    e7_preflight_manifest_sha256: str | None = None
    e7_question_bindings: dict[str, dict[str, str]] = {}
    e7_digest_owners: dict[str, dict[str, str]] = {
        "source_e1_b_selection_trace_sha256": {},
        "window_batch_trace_sha256": {},
        "normalized_constructor_receipts_sha256": {},
    }
    cases: list[PairedQACase] = []
    for index, (row, record) in enumerate(zip(rows, records, strict=True)):
        label = f"evidence[{index}]"
        if set(row) != _CASE_FIELDS:
            raise LongMemEvalSelectionEvidenceError(f"{label} fields differ from the case schema")
        expected_scalars = {
            "schema_version": CASE_SCHEMA_VERSION,
            "artifact_type": CASE_ARTIFACT_TYPE,
            "protocol_version": PROTOCOL_VERSION,
            "case_index": index,
            "question_id": record["question_id"],
            "question_type": record["question_type"],
        }
        for field, wanted in expected_scalars.items():
            if type(row.get(field)) is not type(wanted) or row.get(field) != wanted:
                raise LongMemEvalSelectionEvidenceError(
                    f"{label}.{field} disagrees with the exact dataset/order"
                )
        question_id = str(record["question_id"])
        if question_id in seen_question_ids:
            raise LongMemEvalSelectionEvidenceError(f"{label} duplicates a question ID")
        seen_question_ids.add(question_id)
        expected_hashes = {
            "dataset_record_sha256": sha256_json(record),
            "question_sha256": sha256_utf8(record["question"]),
            "question_date_sha256": sha256_utf8(record["question_date"]),
            "reference_answer_sha256": sha256_json(record["answer"]),
            "source_corpus_sha256": _source_corpus_sha256(record),
        }
        for field, wanted in expected_hashes.items():
            if sha256_text(row.get(field), label=f"{label}.{field}") != wanted:
                raise LongMemEvalSelectionEvidenceError(
                    f"{label}.{field} disagrees with the exact dataset"
                )
        baseline = _arm_outcome(
            row.get(BASELINE_ARM),
            case_index=index,
            arm=BASELINE_ARM,
            record=record,
            question_sha256=expected_hashes["question_sha256"],
            question_date_sha256=expected_hashes["question_date_sha256"],
            reference_answer_sha256=expected_hashes["reference_answer_sha256"],
            source_corpus_sha256=expected_hashes["source_corpus_sha256"],
            cell_identity=run["baseline_cell"],
            cell_identity_sha256=baseline_cell_sha256,
            serializer_identity_sha256=serializer_identity_sha256,
            reader_identity_sha256=reader_identity_sha256,
            judge_identity_sha256=judge_identity_sha256,
            label=f"{label}.{BASELINE_ARM}",
            client_request_ids=client_request_ids,
            provider_request_ids=provider_request_ids,
            receipt_envelope_digests=receipt_envelope_digests,
        )
        candidate = _arm_outcome(
            row.get(CANDIDATE_ARM),
            case_index=index,
            arm=CANDIDATE_ARM,
            record=record,
            question_sha256=expected_hashes["question_sha256"],
            question_date_sha256=expected_hashes["question_date_sha256"],
            reference_answer_sha256=expected_hashes["reference_answer_sha256"],
            source_corpus_sha256=expected_hashes["source_corpus_sha256"],
            cell_identity=run["candidate_cell"],
            cell_identity_sha256=candidate_cell_sha256,
            serializer_identity_sha256=serializer_identity_sha256,
            reader_identity_sha256=reader_identity_sha256,
            judge_identity_sha256=judge_identity_sha256,
            label=f"{label}.{CANDIDATE_ARM}",
            client_request_ids=client_request_ids,
            provider_request_ids=provider_request_ids,
            receipt_envelope_digests=receipt_envelope_digests,
        )
        for arm, profile in (
            (BASELINE_ARM, str(run["baseline_cell"]["selection_input_profile"])),
            (CANDIDATE_ARM, str(run["candidate_cell"]["selection_input_profile"])),
        ):
            if profile not in _E7_PROFILES:
                continue
            protocol_evidence = row[arm]["selection"]["protocol_evidence"]
            preflight_sha256 = protocol_evidence["preflight_manifest_sha256"]
            if e7_preflight_manifest_sha256 is None:
                e7_preflight_manifest_sha256 = preflight_sha256
            elif preflight_sha256 != e7_preflight_manifest_sha256:
                raise LongMemEvalSelectionEvidenceError(
                    "all E7 rows and arms must bind one shared preflight manifest"
                )
            case_bindings = e7_question_bindings.setdefault(question_id, {})
            shared_digest_fields = [
                "source_e1_b_selection_trace_sha256",
                "window_batch_trace_sha256",
            ]
            for field in shared_digest_fields:
                digest = protocol_evidence[field]
                previous_case_digest = case_bindings.setdefault(field, digest)
                if digest != previous_case_digest:
                    raise LongMemEvalSelectionEvidenceError(
                        f"E7 arms for question {question_id!r} bind different {field} values"
                    )
                owner = e7_digest_owners[field].setdefault(digest, question_id)
                if owner != question_id:
                    raise LongMemEvalSelectionEvidenceError(
                        f"E7 {field} reuses a question-bound digest across questions"
                    )
            if profile != E7_A_INPUT_PROFILE:
                field = "normalized_constructor_receipts_sha256"
                digest = protocol_evidence[field]
                owner = e7_digest_owners[field].setdefault(digest, question_id)
                if owner != question_id:
                    raise LongMemEvalSelectionEvidenceError(
                        f"E7 {field} reuses a question-bound digest across questions"
                    )
        cases.append(
            PairedQACase(
                question_id=question_id,
                question_type=str(record["question_type"]),
                baseline=baseline,
                candidate=candidate,
            )
        )
    if seen_question_ids != {str(record["question_id"]) for record in records}:
        raise LongMemEvalSelectionEvidenceError(
            "evidence does not provide exact complete question-ID coverage"
        )
    return cases


def _heldout_dataset_records(raw: bytes, *, questions: int) -> list[dict[str, Any]]:
    value = _strict_json(raw, label="held-out dataset")
    if not isinstance(value, list) or len(value) != questions:
        raise LongMemEvalSelectionEvidenceError(
            "held-out dataset count differs from its preregistered binding"
        )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(value):
        label = f"heldout.dataset[{index}]"
        if not isinstance(record, dict):
            raise LongMemEvalSelectionEvidenceError(f"{label} must be an object")
        question_id = required_text(record.get("question_id"), label=f"{label}.question_id")
        question_type = required_text(record.get("question_type"), label=f"{label}.question_type")
        if question_id in seen:
            raise LongMemEvalSelectionEvidenceError(f"{label} repeats question_id {question_id!r}")
        seen.add(question_id)
        # Hashing the complete record rejects non-finite/non-UTF-8 held-out inputs.
        sha256_json(record)
        records.append({"question_id": question_id, "question_type": question_type, **record})
    return records


def _validate_heldout_confirmation(
    path: str | Path | None,
    *,
    root: Path,
    primary_dataset_sha256: str,
    run: dict[str, Any],
) -> tuple[dict[str, Any], set[Path]]:
    if path is None:
        return (
            {
                "required": True,
                "present": False,
                "structurally_complete_and_distinct": False,
                "directional_qa_benefit": False,
                "reason": "no structurally preregistered byte-distinct held-out confirmation supplied",
                "authentication": None,
            },
            set(),
        )
    confirmation_path, confirmation_display = _resolve_input(
        path,
        root=root,
        label="held-out confirmation",
        require_relative=True,
    )
    confirmation_raw = confirmation_path.read_bytes()
    confirmation = _strict_object(confirmation_raw, label="held-out confirmation")
    if set(confirmation) != _HELDOUT_CONFIRMATION_FIELDS:
        raise LongMemEvalSelectionEvidenceError(
            "held-out confirmation fields differ from the schema"
        )
    expected = {
        "schema_version": HELDOUT_CONFIRMATION_SCHEMA_VERSION,
        "artifact_type": HELDOUT_CONFIRMATION_ARTIFACT_TYPE,
        "receipt_authentication": RECEIPT_AUTHENTICATION,
    }
    for field, wanted in expected.items():
        if type(confirmation.get(field)) is not type(wanted) or confirmation.get(field) != wanted:
            raise LongMemEvalSelectionEvidenceError(
                f"held-out confirmation.{field} must be {wanted!r}"
            )
    completed_at = _validate_utc(
        confirmation.get("completed_at_utc"), label="held-out confirmation.completed_at_utc"
    )
    prereg_path, prereg_display, prereg_raw = _bound_file(
        confirmation.get("preregistration"),
        root=root,
        label="held-out confirmation.preregistration",
    )
    preregistration = _strict_object(prereg_raw, label="held-out preregistration")
    if set(preregistration) != _HELDOUT_PREREGISTRATION_FIELDS:
        raise LongMemEvalSelectionEvidenceError(
            "held-out preregistration fields differ from the schema"
        )
    prereg_expected = {
        "schema_version": HELDOUT_PREREGISTRATION_SCHEMA_VERSION,
        "artifact_type": HELDOUT_PREREGISTRATION_ARTIFACT_TYPE,
        "primary_dataset_sha256": primary_dataset_sha256,
        "protocol_version": PROTOCOL_VERSION,
        "baseline_cell_sha256": sha256_json(run["baseline_cell"]),
        "candidate_cell_sha256": sha256_json(run["candidate_cell"]),
        "prompt_serializer_sha256": sha256_json(run["prompt_serializer"]),
        "reader_identity_sha256": sha256_json(run["reader_identity"]),
        "judge_identity_sha256": sha256_json(run["judge_identity"]),
        "evidence_schema": _HELDOUT_EVIDENCE_SCHEMA,
    }
    for field, wanted in prereg_expected.items():
        if (
            type(preregistration.get(field)) is not type(wanted)
            or preregistration.get(field) != wanted
        ):
            raise LongMemEvalSelectionEvidenceError(
                f"held-out preregistration.{field} must match the frozen primary protocol"
            )
    required_text(
        preregistration.get("preregistration_id"),
        label="held-out preregistration.preregistration_id",
    )
    registered_at = _validate_utc(
        preregistration.get("registered_at_utc"),
        label="held-out preregistration.registered_at_utc",
    )
    if registered_at >= completed_at:
        raise LongMemEvalSelectionEvidenceError(
            "held-out preregistration must predate the completed confirmation"
        )

    prereg_dataset = preregistration.get("heldout_dataset")
    dataset_path, dataset_display, dataset_raw, dataset_questions = _bound_artifact(
        prereg_dataset,
        root=root,
        fields=_DATASET_BINDING_FIELDS,
        label="held-out preregistration.heldout_dataset",
        count_field="questions",
    )
    if confirmation.get("dataset") != prereg_dataset:
        raise LongMemEvalSelectionEvidenceError(
            "held-out confirmation dataset differs from the preregistered byte binding"
        )
    heldout_dataset_sha256 = sha256_bytes(dataset_raw)
    if heldout_dataset_sha256 == primary_dataset_sha256:
        raise LongMemEvalSelectionEvidenceError(
            "held-out confirmation must use a dataset distinct from the primary track"
        )
    heldout_records = _heldout_dataset_records(dataset_raw, questions=dataset_questions)

    evidence_path, evidence_display, evidence_raw, evidence_rows = _bound_artifact(
        confirmation.get("evidence"),
        root=root,
        fields=_EVIDENCE_BINDING_FIELDS,
        label="held-out confirmation.evidence",
        count_field="rows",
    )
    if evidence_rows != dataset_questions:
        raise LongMemEvalSelectionEvidenceError(
            "held-out evidence must provide complete paired dataset coverage"
        )
    rows = _strict_jsonl(evidence_raw, label="held-out evidence")
    if len(rows) != dataset_questions:
        raise LongMemEvalSelectionEvidenceError(
            "held-out evidence row count differs from its byte binding"
        )
    receipt_digests: set[str] = set()
    baseline_correct = 0
    candidate_correct = 0
    question_types: dict[str, int] = {}
    for index, (row, record) in enumerate(zip(rows, heldout_records, strict=True)):
        label = f"heldout.evidence[{index}]"
        if set(row) != _HELDOUT_CASE_FIELDS:
            raise LongMemEvalSelectionEvidenceError(
                f"{label} fields differ from the paired-label schema"
            )
        scalars = {
            "schema_version": _HELDOUT_CASE_SCHEMA_VERSION,
            "artifact_type": _HELDOUT_CASE_ARTIFACT_TYPE,
            "protocol_version": PROTOCOL_VERSION,
            "case_index": index,
            "question_id": record["question_id"],
            "question_type": record["question_type"],
            "dataset_record_sha256": sha256_json(record),
            "baseline_cell_sha256": preregistration["baseline_cell_sha256"],
            "candidate_cell_sha256": preregistration["candidate_cell_sha256"],
            "prompt_serializer_sha256": preregistration["prompt_serializer_sha256"],
            "reader_identity_sha256": preregistration["reader_identity_sha256"],
            "judge_identity_sha256": preregistration["judge_identity_sha256"],
        }
        for field, wanted in scalars.items():
            if type(row.get(field)) is not type(wanted) or row.get(field) != wanted:
                raise LongMemEvalSelectionEvidenceError(
                    f"{label}.{field} disagrees with the preregistered dataset/order"
                )
        baseline_label = row.get("baseline_label")
        candidate_label = row.get("candidate_label")
        if not isinstance(baseline_label, bool) or not isinstance(candidate_label, bool):
            raise LongMemEvalSelectionEvidenceError(f"{label} must contain two binary judge labels")
        for field in ("baseline_receipt_sha256", "candidate_receipt_sha256"):
            digest = sha256_text(row.get(field), label=f"{label}.{field}")
            if digest in receipt_digests:
                raise LongMemEvalSelectionEvidenceError(
                    f"{label}.{field} reuses a held-out receipt digest"
                )
            receipt_digests.add(digest)
        baseline_correct += int(baseline_label)
        candidate_correct += int(candidate_label)
        question_type = str(record["question_type"])
        question_types[question_type] = question_types.get(question_type, 0) + 1
    delta = (candidate_correct - baseline_correct) / dataset_questions
    return (
        {
            "required": True,
            "present": True,
            "structurally_complete_and_distinct": True,
            "directional_qa_benefit": delta > 0.0,
            "qa_delta": delta,
            "questions": dataset_questions,
            "question_type_counts": dict(sorted(question_types.items())),
            "preregistration_id": preregistration["preregistration_id"],
            "registered_at_utc": preregistration["registered_at_utc"],
            "completed_at_utc": confirmation["completed_at_utc"],
            "dataset": {
                "path": dataset_display,
                "bytes": len(dataset_raw),
                "sha256": heldout_dataset_sha256,
            },
            "evidence": {
                "path": evidence_display,
                "bytes": len(evidence_raw),
                "sha256": sha256_bytes(evidence_raw),
            },
            "preregistration": {
                "path": prereg_display,
                "bytes": len(prereg_raw),
                "sha256": sha256_bytes(prereg_raw),
            },
            "confirmation": {
                "path": confirmation_display,
                "bytes": len(confirmation_raw),
                "sha256": sha256_bytes(confirmation_raw),
            },
            "authentication": RECEIPT_AUTHENTICATION,
            "receipt_digests_reopened": False,
            "labels_recomputed_from_complete_paired_rows": True,
            "evidence_authenticity_verified": False,
            "independence_claim_authenticated": False,
            "preregistration_timestamp_authenticated": False,
            "scope": (
                "structural policy evidence over a distinct byte-bound dataset; "
                "not an authenticated provider attestation or an empirical SOTA claim"
            ),
        },
        {confirmation_path, prereg_path, dataset_path, evidence_path},
    )


def compile_longmemeval_selection_report(
    run_path: str | Path,
    output_path: str | Path,
    *,
    artifact_root: str | Path,
    code_root: str | Path,
    heldout_confirmation_path: str | Path | None = None,
    expected_dataset_sha256: str = LONGMEMEVAL_S_SHA256,
    expected_question_count: int = EXPECTED_QUESTION_COUNT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compile complete paired QA rows without executing any external system.

    Reader, judge, tokenizer, selection, and provider receipts are externally
    attested and unsigned.  This compiler checks their structure, cross-row
    identity, digests, pairing, accounting, and byte-bound inputs; it does not
    claim to authenticate which model weights an external process served.
    """

    root = Path(artifact_root).resolve()
    sha256_text(expected_dataset_sha256, label="expected_dataset_sha256")
    integer(expected_question_count, label="expected_question_count", minimum=1)
    resolved_run, run_display = _resolve_input(
        run_path,
        root=root,
        label="run_path",
        require_relative=True,
    )
    run_raw = resolved_run.read_bytes()
    run, dataset_path, dataset_raw, evidence_path, evidence_raw = _load_run(run_raw, root=root)
    if run["dataset"]["sha256"] != expected_dataset_sha256:
        raise LongMemEvalSelectionEvidenceError(
            "run dataset digest differs from the compiler's pinned expectation"
        )
    if run["dataset"]["questions"] != expected_question_count:
        raise LongMemEvalSelectionEvidenceError(
            "run dataset count differs from the compiler's pinned expectation"
        )
    records = _load_dataset(
        dataset_raw,
        expected_sha256=expected_dataset_sha256,
        expected_questions=expected_question_count,
    )
    rows = _strict_jsonl(evidence_raw, label="evidence")
    if len(rows) != run["evidence"]["rows"]:
        raise LongMemEvalSelectionEvidenceError(
            "evidence row count differs from the run byte binding"
        )
    cases = _validate_rows(rows, records=records, run=run)

    qa = paired_qa_summary(cases)
    by_type = qa_by_question_type(cases)
    context = context_summary(cases)
    efficiency = efficiency_summary(cases)

    qa_ci_gate = float(qa["paired_delta"]["ci_low"]) > 0.0
    regressed_types: list[dict[str, Any]] = []
    for question_type, metric in by_type.items():
        baseline_correct = int(metric["baseline_correct"])
        candidate_correct = int(metric["candidate_correct"])
        questions = int(metric["questions"])
        net_regression_count = baseline_correct - candidate_correct
        # Compare raw integers: net_regression / questions > 2 / 100.
        if net_regression_count > 0 and net_regression_count * 100 > 2 * questions:
            regressed_types.append(
                {
                    "question_type": question_type,
                    "questions": questions,
                    "baseline_correct": baseline_correct,
                    "candidate_correct": candidate_correct,
                    "net_regression_count": net_regression_count,
                    "regression_percentage_points": 100.0 * net_regression_count / questions,
                }
            )
    type_gate = not regressed_types
    context_available = context.get("available") is True
    any_gold_gate = bool(
        context_available
        and int(context["candidate"]["any_gold_questions"])
        >= int(context["baseline"]["any_gold_questions"])
    )
    expected_gold_context_questions = sum(not case.question_id.endswith("_abs") for case in cases)
    all_metrics_gate = bool(
        context_available
        and expected_gold_context_questions > 0
        and int(context["questions"]) == expected_gold_context_questions
        and all(
            case.baseline.prompt_tokens > 0 and case.candidate.prompt_tokens > 0 for case in cases
        )
    )
    baseline_dominates = baseline_dominates_candidate(qa=qa, efficiency=efficiency)
    pareto_gate = not baseline_dominates

    heldout, heldout_paths = _validate_heldout_confirmation(
        heldout_confirmation_path,
        root=root,
        primary_dataset_sha256=expected_dataset_sha256,
        run=run,
    )
    track_gates = {
        "qa_ci_lower_strictly_above_zero": {
            "passed": qa_ci_gate,
            "ci_low": qa["paired_delta"]["ci_low"],
            "required_strict_lower_bound": 0.0,
        },
        "no_question_type_regression_over_two_percentage_points": {
            "passed": type_gate,
            "maximum_regression_percentage_points": MAX_TYPE_REGRESSION * 100.0,
            "violations": regressed_types,
            "comparison_uses_raw_correct_counts": True,
        },
        "any_gold_in_context_noninferior": {
            "passed": any_gold_gate,
            "noninferiority_margin": 0.0,
            "baseline_any_gold_questions": (
                context["baseline"]["any_gold_questions"] if context_available else None
            ),
            "candidate_any_gold_questions": (
                context["candidate"]["any_gold_questions"] if context_available else None
            ),
        },
        "all_required_metrics_reported": {
            "passed": all_metrics_gate,
            "required": [
                "paired QA and confidence interval",
                "per-question-type raw counts and deltas",
                "any/all gold in delivered context",
                "answer-session MRR",
                "exact prompt tokens",
                "operational and end-to-end latency",
                "stage call/token/cost accounting",
            ],
        },
        "pareto_non_dominated": {
            "passed": pareto_gate,
            "baseline_dominates_candidate": baseline_dominates,
            "semantics": (
                "baseline is at least as good on QA and no worse on p95 prompt tokens, "
                "p95 operational latency, and total construction-plus-query cost, with "
                "at least one strict advantage"
            ),
        },
    }
    primary_track_passed = all(bool(value["passed"]) for value in track_gates.values())
    structural_heldout_gate_passed = bool(
        heldout["present"]
        and heldout["structurally_complete_and_distinct"]
        and heldout["directional_qa_benefit"]
    )
    canonical_official = (
        expected_dataset_sha256 == LONGMEMEVAL_S_SHA256
        and expected_question_count == EXPECTED_QUESTION_COUNT
    )
    candidate_profile = run["candidate_cell"]["selection_input_profile"]
    e7_b_requires_authenticated_faithfulness = candidate_profile == E7_B_INPUT_PROFILE
    authenticated_faithfulness_verified = False
    candidate_intrinsically_eligible = (
        candidate_profile not in INTRINSICALLY_INELIGIBLE_CANDIDATE_PROFILES
    )
    structural_offline_policy_eligible = bool(
        canonical_official
        and candidate_intrinsically_eligible
        and primary_track_passed
        and structural_heldout_gate_passed
    )
    serving_promotion_eligible = False
    if structural_offline_policy_eligible:
        verdict = "structurally-eligible-authenticated-serving-approval-required"
    elif not canonical_official:
        verdict = "refused-noncanonical-primary-dataset"
    elif not candidate_intrinsically_eligible:
        verdict = "refused-intrinsically-ineligible-candidate-profile"
    elif not primary_track_passed:
        verdict = "refused-primary-track-gates-not-satisfied"
    elif not heldout["present"]:
        verdict = "refused-structural-preregistered-heldout-artifact-required"
    else:
        verdict = "refused-heldout-directional-benefit-not-proven"

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "experiment_id": run["experiment_id"],
        "protocol": fixed_protocol(),
        "compiler_implementation": implementation_fingerprint(code_root),
        "benchmark": {
            "name": "LongMemEval-S" if canonical_official else "fixture-or-noncanonical",
            "source": LONGMEMEVAL_S_URL if canonical_official else run["dataset"]["path"],
            "dataset_sha256": expected_dataset_sha256,
            "canonical_official_cleaned_dataset": canonical_official,
            "questions": len(records),
        },
        "source_artifacts": {
            "run": {
                "path": run_display,
                "bytes": len(run_raw),
                "sha256": sha256_bytes(run_raw),
            },
            "dataset": {
                "path": run["dataset"]["path"],
                "bytes": len(dataset_raw),
                "sha256": sha256_bytes(dataset_raw),
            },
            "evidence": {
                "path": run["evidence"]["path"],
                "bytes": len(evidence_raw),
                "sha256": sha256_bytes(evidence_raw),
                "rows": len(rows),
            },
        },
        "identities": {
            "baseline_cell": run["baseline_cell"],
            "candidate_cell": run["candidate_cell"],
            "prompt_serializer": run["prompt_serializer"],
            "reader": run["reader_identity"],
            "judge": run["judge_identity"],
            "receipt_authentication": RECEIPT_AUTHENTICATION,
        },
        "coverage": {
            "dataset_questions": len(records),
            "paired_questions": len(cases),
            "missing_questions": 0,
            "unpaired_questions": 0,
            "duplicate_question_ids": 0,
            "question_type_counts": {
                question_type: sum(case.question_type == question_type for case in cases)
                for question_type in sorted({case.question_type for case in cases})
            },
        },
        "metrics": {
            "qa": qa,
            "qa_by_question_type": by_type,
            "gold_context": context,
            "efficiency": efficiency,
        },
        "promotion": {
            "primary_track_gates": track_gates,
            "primary_track_passed": primary_track_passed,
            "canonical_official_primary_required": True,
            "canonical_official_primary_passed": canonical_official,
            "candidate_intrinsic_eligibility": {
                "passed": candidate_intrinsically_eligible,
                "candidate_input_profile": candidate_profile,
                "intrinsically_ineligible_profiles": list(
                    INTRINSICALLY_INELIGIBLE_CANDIDATE_PROFILES
                ),
                "authenticated_abstractive_faithfulness_required": (
                    e7_b_requires_authenticated_faithfulness
                ),
                "authenticated_abstractive_faithfulness_verified": (
                    authenticated_faithfulness_verified
                ),
                "reason": (
                    "selection QA protocol v2 has no authenticated E7-B faithfulness verifier"
                    if e7_b_requires_authenticated_faithfulness
                    else None
                ),
            },
            "heldout_structural_evidence": heldout,
            "structural_heldout_gate_passed": structural_heldout_gate_passed,
            "structural_offline_policy_eligible": structural_offline_policy_eligible,
            "serving_promotion_eligible": serving_promotion_eligible,
            "serving_promotion_reason": (
                "serving promotion is never authorized by this offline compiler; "
                "authenticated execution evidence and explicit serving approval are required"
            ),
            "structural_eligibility_scope": (
                "structural integrity policy result over unsigned, unreopened external "
                "receipts and a byte-distinct held-out artifact; not semantic independence, "
                "authenticated model execution, serving authorization, or an empirical SOTA claim"
            ),
            "verdict": verdict,
        },
        "validation": {
            "complete_paired_coverage_replayed": True,
            "dataset_record_hashes_recomputed": True,
            "question_ids_and_types_reconciled": True,
            "shared_query_and_corpus_hashes_recomputed": True,
            "candidate_ids_reconciled_to_dataset": True,
            "baseline_candidate_cell_artifact_sha256_distinct": True,
            "prompt_serializer_reader_tokenizer_tuple_equal": True,
            "gold_context_metrics_computed_only_after_selection": True,
            "declared_selection_input_fields_are_allowlisted": True,
            "selection_protocol_evidence_digests_recomputed": True,
            "e7_reader_context_and_constructor_receipts_reconciled": True,
            "e7_preflight_and_question_scoped_bindings_reconciled": True,
            "selection_input_artifacts_reopened": False,
            "selection_gold_leakage_absence_cryptographically_proven": False,
            "exact_prompt_token_counts_bounded": True,
            "exact_prompt_token_counts_recomputed": False,
            "exact_prompt_token_counts_status": RECEIPT_AUTHENTICATION,
            "provider_receipts_replayed": False,
            "provider_receipts_cryptographically_authenticated": False,
            "provider_receipts_status": RECEIPT_AUTHENTICATION,
            "receipt_envelope_digests_recomputed": True,
            "receipt_envelope_bytes_reopened": False,
            "receipt_envelope_digests_globally_unique": True,
            "receipt_envelope_digests": len(cases) * len(ARMS) * 2,
            "construction_plus_query_cost_formula": (
                "embedding_cost_plus_reranker_cost_plus_constructor_cost"
            ),
            "reader_cost_excluded_from_construction_plus_query": True,
            "content_text_fields_omitted": True,
            "arbitrary_identifier_metadata_content_free_proven": False,
            "output_overwrite_type_checked_and_atomic": True,
        },
    }
    protected = {resolved_run, dataset_path, evidence_path, *heldout_paths}
    output = _resolve_output(
        output_path,
        root=root,
        overwrite=overwrite,
        protected={path.resolve() for path in protected},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_report(output, report)
    return report


__all__ = [
    "EXPECTED_QUESTION_COUNT",
    "build_run_manifest",
    "compile_longmemeval_selection_report",
    "implementation_fingerprint",
]
