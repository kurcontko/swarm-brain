"""Canonical core inputs and raw trace bridge for the reranker paired A/B."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from _longmemeval_common import (
    LONGMEMEVAL_CLOCK_START,
    LONGMEMEVAL_S_URL,
    parse_longmemeval_datetime,
    session_text,
)

from swarmbrain.domain.memory import Memory, MemoryKind, MemoryState, Visibility
from swarmbrain.domain.reranking import (
    LEARNED_RERANK_REQUEST_SCHEMA,
    LEARNED_RERANK_RESPONSE_SCHEMA,
    LearnedRerankerIdentity,
    LearnedRerankPolicy,
    LearnedRerankRequest,
    LearnedRerankTrace,
    canonical_rerank_json,
    learned_rerank_candidate_pool_payload,
    rerank_sha256_json,
)
from swarmbrain.retrieval.learned_reranking import (
    SWARMBRAIN_MEMORY_RERANK_SERIALIZER_REVISION,
    build_learned_rerank_request,
    canonical_memory_rerank_input,
    utf8_prefix,
)

from .contracts import (
    ARMS,
    BASELINE_ARM,
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_METHOD,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CANDIDATE_WINDOW,
    K_VALUES,
    PROTOCOL_VERSION,
    RUN_ARTIFACT_TYPE,
    RUN_SCHEMA_VERSION,
    SCORE_MAXIMUM,
    SCORE_MINIMUM,
    SLICE_CATEGORIES,
    TREATMENT_ARM,
    LongMemEvalRerankerEvidenceError,
)

CANDIDATE_SERIALIZER_VERSION = SWARMBRAIN_MEMORY_RERANK_SERIALIZER_REVISION
REQUEST_SERIALIZER_VERSION = LEARNED_RERANK_REQUEST_SCHEMA
TRACE_SCHEMA_VERSION = 1

ARM_INPUT_FIELDS = frozenset(
    {
        "serializer_revision",
        "query_sha256",
        "candidate_pool_sha256",
        "request_sha256",
        "candidate_document_sha256",
        "candidate_temporal_sha256",
        "candidate_id_mapping_sha256",
        "tokenized_input_sha256",
        "evaluation_candidate_ids",
        "provider_candidate_ids",
        "k_values",
    }
)

_BENCHMARK_IMPLEMENTATION_PATHS = (
    "benchmarks/integrations/longmemeval_reranker/__init__.py",
    "benchmarks/integrations/longmemeval_reranker/contracts.py",
    "benchmarks/integrations/longmemeval_reranker/evidence.py",
    "benchmarks/integrations/longmemeval_reranker/metrics.py",
    "benchmarks/integrations/longmemeval_reranker/report.py",
    "scripts/build_longmemeval_reranker_report.py",
    "scripts/run_longmemeval_reranker_ab.py",
    "scripts/_longmemeval_common.py",
    "scripts/_longmemeval_tokenizer.py",
    "scripts/run_longmemeval_qa.py",
    "scripts/run_retrieval_eval.py",
    "pyproject.toml",
    "uv.lock",
    "src/swarmbrain/domain/common.py",
    "src/swarmbrain/domain/evidence.py",
    "src/swarmbrain/domain/memory.py",
    "src/swarmbrain/domain/reranking.py",
    "src/swarmbrain/domain/retrieval.py",
    "src/swarmbrain/ports/reranking.py",
    "src/swarmbrain/application/memory_policy.py",
    "src/swarmbrain/application/retrieval_service.py",
    "src/swarmbrain/adapters/embeddings/openai_compatible.py",
    "src/swarmbrain/adapters/memory/__init__.py",
    "src/swarmbrain/adapters/memory/in_memory.py",
    "src/swarmbrain/adapters/memory/retrieval.py",
    "src/swarmbrain/ports/embeddings.py",
    "src/swarmbrain/retrieval/fusion.py",
    "src/swarmbrain/retrieval/learned_reranking.py",
    "src/swarmbrain/retrieval/planner.py",
    "src/swarmbrain/retrieval/projection.py",
    "src/swarmbrain/retrieval/relevance.py",
)


def canonical_json(value: Any) -> str:
    """Use the core learned-reranker canonical JSON, including ASCII escaping."""

    try:
        return canonical_rerank_json(value)
    except (TypeError, ValueError) as exc:
        raise LongMemEvalRerankerEvidenceError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return rerank_sha256_json(value)


def canonical_policy(identity: LearnedRerankerIdentity) -> LearnedRerankPolicy:
    """The preregistered benchmark policy; scorer identity remains generic."""

    return LearnedRerankPolicy(identity=identity, window=CANDIDATE_WINDOW, alpha=1.0)


def protocol_evidence() -> dict[str, Any]:
    """Return the one accepted paired design; no outcome threshold is encoded."""

    return {
        "paired": True,
        "arms": list(ARMS),
        "baseline_order_source": "source_retrieval.case.rankings.fused",
        "candidate_window": CANDIDATE_WINDOW,
        "candidate_serializer": CANDIDATE_SERIALIZER_VERSION,
        "request_schema": REQUEST_SERIALIZER_VERSION,
        "response_schema": LEARNED_RERANK_RESPONSE_SCHEMA,
        "learned_alpha": 1.0,
        "k_values": list(K_VALUES),
        "score_range": [SCORE_MINIMUM, SCORE_MAXIMUM],
        "score_direction": "descending",
        "stable_tie_break": "source_fusion_rank",
        "same_query_candidate_text_temporal_tokenizer_input_and_k": True,
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "unit": "question",
            "paired": True,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": BOOTSTRAP_CONFIDENCE,
        },
        "slices": SLICE_CATEGORIES,
        "pass_threshold": None,
    }


def query_text(record: dict[str, Any]) -> str:
    value = record.get("question")
    if not isinstance(value, str) or not value:
        raise LongMemEvalRerankerEvidenceError("dataset question must be non-empty text")
    return value


def candidate_key(record: dict[str, Any], position: int) -> str:
    session_ids = record.get("haystack_session_ids")
    if not isinstance(session_ids, list) or not 0 <= position < len(session_ids):
        raise LongMemEvalRerankerEvidenceError("candidate position is outside the haystack")
    session_id = session_ids[position]
    if not isinstance(session_id, str) or not session_id:
        raise LongMemEvalRerankerEvidenceError("dataset session ID must be non-empty text")
    return f"{position:03d}:{session_id}"


def parse_candidate_position(record: dict[str, Any], candidate_id: str) -> int:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise LongMemEvalRerankerEvidenceError("candidate ID must be non-empty text")
    try:
        prefix, _ = candidate_id.split(":", 1)
        position = int(prefix)
    except (ValueError, TypeError) as exc:
        raise LongMemEvalRerankerEvidenceError(
            f"candidate ID is not a LongMemEval session key: {candidate_id!r}"
        ) from exc
    if candidate_key(record, position) != candidate_id:
        raise LongMemEvalRerankerEvidenceError(
            f"candidate ID disagrees with the pinned dataset: {candidate_id!r}"
        )
    return position


def canonical_candidate_memory(record: dict[str, Any], position: int) -> Memory:
    """Recreate the memory written by ``retrieve_question`` before retrieval."""

    session_ids = record.get("haystack_session_ids")
    dates = record.get("haystack_dates")
    sessions = record.get("haystack_sessions")
    if not all(isinstance(value, list) for value in (session_ids, dates, sessions)):
        raise LongMemEvalRerankerEvidenceError("dataset haystack arrays must be lists")
    if not len(session_ids) == len(dates) == len(sessions):
        raise LongMemEvalRerankerEvidenceError("dataset haystack arrays are misaligned")
    if not 0 <= position < len(session_ids):
        raise LongMemEvalRerankerEvidenceError("candidate position is outside the haystack")
    date = dates[position]
    turns = sessions[position]
    if not isinstance(date, str) or not isinstance(turns, list):
        raise LongMemEvalRerankerEvidenceError("dataset session date/turns are malformed")
    case_id = str(record.get("question_id") or "")
    namespace = f"swarmbrain-lme-rerank/{case_id}/{position}"
    content = session_text(turns).strip() or "(empty session)"
    return Memory(
        memory_id=str(uuid5(NAMESPACE_URL, namespace + "/memory")),
        tenant_id=str(uuid5(NAMESPACE_URL, namespace + "/tenant")),
        project_id=str(uuid5(NAMESPACE_URL, namespace + "/project")),
        repository_id=str(uuid5(NAMESPACE_URL, namespace + "/repository")),
        swarm_id=str(uuid5(NAMESPACE_URL, namespace + "/swarm")),
        run_id=str(uuid5(NAMESPACE_URL, namespace + "/run")),
        author_agent_id=str(uuid5(NAMESPACE_URL, namespace + "/agent")),
        kind=MemoryKind.OBSERVATION,
        state=MemoryState.CONFIRMED,
        visibility=Visibility.REPOSITORY,
        title=f"Conversation session recorded {date}" if date else "Conversation session",
        content=content,
        tags=("longmemeval", "session"),
        valid_from=parse_longmemeval_datetime(date),
        recorded_from=LONGMEMEVAL_CLOCK_START + timedelta(seconds=position + 1),
        metadata={
            "policy": {
                "operation": "add",
                "reason": "append independent memory; no explicit supersession target",
                "confidence": 0.9,
            }
        },
    )


def candidate_core_input(
    record: dict[str, Any],
    position: int,
    policy: LearnedRerankPolicy,
) -> tuple[str, str]:
    return canonical_memory_rerank_input(canonical_candidate_memory(record, position), policy)


def candidate_document_text(
    record: dict[str, Any],
    position: int,
    policy: LearnedRerankPolicy,
) -> str:
    return candidate_core_input(record, position, policy)[0]


def build_core_request(
    record: dict[str, Any],
    evaluation_candidate_ids: list[str] | tuple[str, ...],
    provider_candidate_ids: list[str] | tuple[str, ...],
    policy: LearnedRerankPolicy,
    *,
    request_id: str,
) -> LearnedRerankRequest:
    query = query_text(record)[: policy.max_query_characters]
    query = utf8_prefix(query, policy.max_query_bytes)
    if len(evaluation_candidate_ids) != len(provider_candidate_ids):
        raise LongMemEvalRerankerEvidenceError(
            "provider IDs must align positionally with the evaluation candidate pool"
        )
    candidates = tuple(
        (
            provider_id,
            *candidate_core_input(
                record,
                parse_candidate_position(record, evaluation_id),
                policy,
            ),
        )
        for evaluation_id, provider_id in zip(
            evaluation_candidate_ids[: policy.window],
            provider_candidate_ids[: policy.window],
            strict=True,
        )
    )
    return build_learned_rerank_request(
        policy,
        serializer_revision=SWARMBRAIN_MEMORY_RERANK_SERIALIZER_REVISION,
        query=query,
        candidates=candidates,
        request_id=request_id,
    )


def candidate_document_evidence(
    request: LearnedRerankRequest,
    evaluation_candidate_ids: list[str] | tuple[str, ...],
) -> list[dict[str, str | int]]:
    if len(request.candidates) != len(evaluation_candidate_ids):
        raise LongMemEvalRerankerEvidenceError(
            "candidate evidence requires one evaluation ID per provider candidate"
        )
    return [
        {
            "evaluation_candidate_id": evaluation_id,
            "provider_candidate_id": candidate.candidate_id,
            "document_sha256": candidate.document_sha256,
            "temporal_sha256": candidate.temporal_sha256,
            "document_utf8_bytes": len(candidate.document.encode("utf-8")),
            "temporal_utf8_bytes": len(candidate.temporal_context.encode("utf-8")),
        }
        for candidate, evaluation_id in zip(
            request.candidates,
            evaluation_candidate_ids,
            strict=True,
        )
    ]


def candidate_pool_sha256(request: LearnedRerankRequest) -> str:
    return rerank_sha256_json(learned_rerank_candidate_pool_payload(request.candidates))


def request_sha256(request: LearnedRerankRequest) -> str:
    return request.request_sha256


def arm_input_evidence(
    request: LearnedRerankRequest,
    *,
    evaluation_candidate_ids: list[str] | tuple[str, ...],
    tokenized_input_sha256: str,
) -> dict[str, Any]:
    provider_candidate_ids = [item.candidate_id for item in request.candidates]
    mapping = [
        {"evaluation_candidate_id": evaluation_id, "provider_candidate_id": provider_id}
        for evaluation_id, provider_id in zip(
            evaluation_candidate_ids,
            provider_candidate_ids,
            strict=True,
        )
    ]
    return {
        "serializer_revision": request.serializer_revision,
        "query_sha256": request.query_sha256,
        "candidate_pool_sha256": request.candidate_pool_sha256,
        "request_sha256": request.request_sha256,
        "candidate_document_sha256": {
            item.candidate_id: item.document_sha256 for item in request.candidates
        },
        "candidate_temporal_sha256": {
            item.candidate_id: item.temporal_sha256 for item in request.candidates
        },
        "candidate_id_mapping_sha256": sha256_json(mapping),
        "tokenized_input_sha256": tokenized_input_sha256,
        "evaluation_candidate_ids": list(evaluation_candidate_ids),
        "provider_candidate_ids": provider_candidate_ids,
        "k_values": list(K_VALUES),
    }


def build_trace_row(
    *,
    case_index: int,
    record: dict[str, Any],
    source_case: dict[str, Any],
    policy: LearnedRerankPolicy,
    learned_trace: LearnedRerankTrace | dict[str, Any],
) -> dict[str, Any]:
    """Project one authoritative core trace into the raw JSONL schema."""

    trace = (
        learned_trace
        if isinstance(learned_trace, LearnedRerankTrace)
        else LearnedRerankTrace.model_validate(learned_trace)
    )
    if not trace.applied or trace.request_id is None or trace.usage is None:
        raise LongMemEvalRerankerEvidenceError(
            "paired evidence requires one successfully applied core learned-rerank trace"
        )
    rankings = source_case.get("rankings")
    fused = rankings.get("fused") if isinstance(rankings, dict) else None
    if not isinstance(fused, list) or not all(isinstance(item, str) for item in fused):
        raise LongMemEvalRerankerEvidenceError("source case lacks a fused ranking")
    candidate_ids = fused[: policy.window]
    request = build_core_request(
        record,
        candidate_ids,
        list(trace.input_ids),
        policy,
        request_id=trace.request_id,
    )
    shared_input = arm_input_evidence(
        request,
        evaluation_candidate_ids=candidate_ids,
        tokenized_input_sha256=trace.usage.tokenized_input_sha256,
    )
    case_id = record.get("question_id")
    category = record.get("question_type")
    if not isinstance(case_id, str) or not isinstance(category, str):
        raise LongMemEvalRerankerEvidenceError("dataset case identity is malformed")
    return {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "case_index": case_index,
        "case_id": case_id,
        "category": category,
        "abstention_question": case_id.endswith("_abs"),
        "candidate_documents": candidate_document_evidence(request, candidate_ids),
        "baseline": {
            "arm": BASELINE_ARM,
            "input": shared_input,
            "ranked_ids": list(candidate_ids),
        },
        "learned": {
            "arm": TREATMENT_ARM,
            "input": shared_input,
            "trace": trace.model_dump(mode="json"),
        },
    }


def build_run_manifest(
    *,
    created_at_utc: str,
    dataset_sha256: str,
    question_count: int,
    source_retrieval_path: Path,
    traces_path: Path,
    identity: LearnedRerankerIdentity,
    policy: LearnedRerankPolicy,
    artifact_root: Path,
    code_root: Path,
    trace_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build the byte-bound envelope after raw JSONL has been durably written."""

    if policy.identity != identity or policy != canonical_policy(identity):
        raise LongMemEvalRerankerEvidenceError(
            "run manifest requires the canonical core alpha=1/window=50 policy"
        )
    root = artifact_root.resolve()

    def artifact(path: Path) -> dict[str, Any]:
        if path.is_symlink():
            raise LongMemEvalRerankerEvidenceError("run artifacts cannot be symbolic links")
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise LongMemEvalRerankerEvidenceError(
                "run artifacts must be regular repository-local files"
            )
        raw = resolved.read_bytes()
        return {
            "path": resolved.relative_to(root).as_posix(),
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        }

    trace_identity = artifact(traces_path)
    raw_trace_rows = traces_path.read_text(encoding="utf-8").splitlines()
    if len(raw_trace_rows) != len(trace_rows):
        raise LongMemEvalRerankerEvidenceError(
            "trace_rows do not match the durably written JSONL row count"
        )
    for index, (raw, supplied) in enumerate(zip(raw_trace_rows, trace_rows, strict=True)):
        if json.loads(raw) != supplied:
            raise LongMemEvalRerankerEvidenceError(
                f"trace_rows[{index}] differs from the durably written JSONL"
            )

    usage_fields = (
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
    totals = {field: 0 for field in usage_fields}
    request_ids: set[str] = set()
    provider_request_ids: set[str] = set()
    tokenized_digests: set[str] = set()
    for index, row in enumerate(trace_rows):
        try:
            trace = LearnedRerankTrace.model_validate(row["learned"]["trace"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LongMemEvalRerankerEvidenceError(
                f"trace_rows[{index}] does not contain a valid core trace"
            ) from exc
        if (
            not trace.applied
            or trace.request_id is None
            or trace.provider_request_id is None
            or trace.usage is None
        ):
            raise LongMemEvalRerankerEvidenceError(
                f"trace_rows[{index}] does not contain a successful provider receipt"
            )
        request_ids.add(trace.request_id)
        provider_request_ids.add(trace.provider_request_id)
        tokenized_digests.add(trace.usage.tokenized_input_sha256)
        for field in usage_fields:
            totals[field] += int(getattr(trace.usage, field))
    call_accounting = {
        "source": "provider-observed-and-offline-reconciled",
        "requests": len(trace_rows),
        "responses": len(trace_rows),
        "successful_responses": len(trace_rows),
        "unique_request_ids": len(request_ids),
        "unique_provider_request_ids": len(provider_request_ids),
        "unique_tokenized_input_sha256": len(tokenized_digests),
        **totals,
    }
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "artifact_type": RUN_ARTIFACT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": created_at_utc,
        "implementation": implementation_fingerprint(code_root),
        "dataset": {
            "name": "LongMemEval-S",
            "source": LONGMEMEVAL_S_URL,
            "sha256": dataset_sha256,
            "questions": question_count,
        },
        "source_retrieval_artifact": artifact(source_retrieval_path),
        "traces_artifact": {**trace_identity, "rows": len(trace_rows)},
        "protocol": protocol_evidence(),
        "reranker_identity": identity.model_dump(mode="json"),
        "rerank_policy": policy.model_dump(mode="json"),
        "call_accounting": call_accounting,
    }


def implementation_fingerprint(code_root: Path) -> dict[str, Any]:
    """Bind the evidence bridge/compiler and authoritative core rerank seam."""

    root = code_root.resolve()
    paths = [root / relative for relative in _BENCHMARK_IMPLEMENTATION_PATHS]
    package_root = root / "src" / "swarmbrain"
    if package_root.is_dir():
        # The runner executes the ordinary memory publication and retrieval
        # stack, not only the learned-scoring seam.  Bind every package module
        # so a mid-run edit to a transitively imported dependency cannot be
        # omitted from the start/end provenance comparison.
        paths.extend(package_root.rglob("*.py"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        relative = [str(path.relative_to(root)) for path in missing]
        raise LongMemEvalRerankerEvidenceError(
            f"reranker evidence implementation files are missing: {relative}"
        )
    files = {
        path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes())
        for path in sorted(set(paths))
    }
    return {"tree_sha256": sha256_json(files), "files": files}


__all__ = [
    "ARM_INPUT_FIELDS",
    "CANDIDATE_SERIALIZER_VERSION",
    "REQUEST_SERIALIZER_VERSION",
    "TRACE_SCHEMA_VERSION",
    "arm_input_evidence",
    "build_core_request",
    "build_run_manifest",
    "build_trace_row",
    "candidate_core_input",
    "candidate_document_evidence",
    "candidate_document_text",
    "candidate_key",
    "candidate_pool_sha256",
    "canonical_candidate_memory",
    "canonical_json",
    "canonical_policy",
    "implementation_fingerprint",
    "parse_candidate_position",
    "protocol_evidence",
    "query_text",
    "request_sha256",
    "sha256_bytes",
    "sha256_json",
]
