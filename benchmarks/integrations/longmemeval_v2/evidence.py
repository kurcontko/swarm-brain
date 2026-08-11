"""Bind opaque Swarm operations to official LongMemEval-V2 evidence rows."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from .contracts import (
    ADAPTER_REVISION,
    EXPECTED_JUDGE_MODEL,
    EXPECTED_QUESTIONS,
    EXPECTED_READER_MODEL,
    MEMORY_TYPE,
    METHOD,
    PINNED_REPOSITORY_COMMIT,
    RAW_TRACE_DIGEST_METADATA_KEY,
    SIDECAR_SCHEMA_VERSION,
    TRACE_DIGEST_METADATA_KEY,
    AdapterConfig,
    BoundQueryEvidence,
    EmbeddingRuntimeEvidence,
    JsonObject,
    LongMemEvalV2AdapterError,
    RawOperation,
    RawQueryTrace,
)

LEDGER_SCHEMA_VERSION = 3
_RUN_LOCATION_FIELDS = {
    "domain",
    "haystack_path",
    "memory_config_path",
    "output_dir",
    "questions_path",
    "started_at_utc",
    "trajectories_path",
}
_OPERATION_FIELDS = {
    "sequence",
    "invocation_id",
    "operation",
    "success",
    "depth",
    "seed_memory_ids",
    "result_memory_ids",
    "delivered_tokens",
    "latency_ms",
}
_QUERY_FIELDS = {
    "question_id",
    "domain",
    "query_tokens",
    "query_latency_ms",
    "operations",
    "embedding",
    "trace_sha256",
}


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise LongMemEvalV2AdapterError("evidence must be canonical-JSON serializable") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def expected_invocation_id(
    *, tier: str, operating_point: str, question_id: str, sequence: int, operation: str
) -> str:
    return "inv_" + canonical_sha256(
        {
            "operation": operation,
            "operating_point": operating_point,
            "question_id": question_id,
            "sequence": sequence,
            "tier": tier,
        }
    )


def _memory_pseudonym(
    raw_memory_id: str, *, tier: str, operating_point: str, question_id: str
) -> str:
    return "mem_" + canonical_sha256(
        {
            "memory_id": raw_memory_id,
            "operating_point": operating_point,
            "question_id": question_id,
            "tier": tier,
        }
    )


def raw_trace_sha256(trace: RawQueryTrace) -> str:
    """Digest a run-local trace without serializing its opaque handle."""

    return canonical_sha256(
        {
            "operations": [
                {
                    "sequence": operation.sequence,
                    "operation": operation.operation,
                    "depth": operation.depth,
                    "seed_memory_ids": list(operation.seed_memory_ids),
                    "result_memory_ids": list(operation.result_memory_ids),
                    "latency_ms": operation.latency_ms,
                }
                for operation in trace.operations
            ],
            "embedding": trace.embedding.as_json(),
        }
    )


def _nonnegative_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise LongMemEvalV2AdapterError(f"{label} must be a finite non-negative number")
    return float(value)


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LongMemEvalV2AdapterError(f"{label} must be a non-negative integer")
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongMemEvalV2AdapterError(f"{label} must be a non-empty string")
    return value.strip()


def _validate_raw_trace(trace: RawQueryTrace) -> tuple[RawOperation, RawOperation]:
    if len(trace.operations) != 2:
        raise LongMemEvalV2AdapterError(
            "the pinned protocol requires exactly recall_memory then read_expand_memory"
        )
    recall, expand = trace.operations
    if (
        recall.sequence != 0
        or recall.operation != "recall_memory"
        or recall.depth != 0
        or recall.seed_memory_ids
        or not recall.result_memory_ids
    ):
        raise LongMemEvalV2AdapterError("raw recall_memory trace violates the pinned protocol")
    if (
        expand.sequence != 1
        or expand.operation != "read_expand_memory"
        or not 1 <= expand.depth <= 2
        or not 1 <= len(expand.seed_memory_ids) <= 8
        or not expand.result_memory_ids
    ):
        raise LongMemEvalV2AdapterError("raw read_expand_memory trace violates the pinned protocol")
    if not set(expand.seed_memory_ids).issubset(recall.result_memory_ids):
        raise LongMemEvalV2AdapterError(
            "read_expand_memory seeds must come from the preceding recall_memory result"
        )
    for operation in trace.operations:
        _nonnegative_number(operation.latency_ms, f"{operation.operation}.latency_ms")
    return recall, expand


def bind_query_trace(
    trace: RawQueryTrace,
    official_prompt_row: Mapping[str, Any],
    *,
    tier: str,
    operating_point: str,
    domain: str,
) -> BoundQueryEvidence:
    """Restore benchmark IDs only after the private backend query has returned.

    The official harness computes token truncation after ``post_query_hook``.
    Consequently this binder is the first truthful point at which both the
    stable question ID and the exact context delivered to the reader exist.
    """

    if tier not in {"small", "medium"}:
        raise LongMemEvalV2AdapterError("tier must be small or medium")
    if domain not in {"web", "enterprise"}:
        raise LongMemEvalV2AdapterError("domain must be web or enterprise")
    operating_point = _required_text(operating_point, "operating_point")
    question_id = _required_text(official_prompt_row.get("question_id"), "question_id")
    opaque_id = _required_text(
        official_prompt_row.get("query_invocation_id"), "query_invocation_id"
    )
    if opaque_id != trace.opaque_invocation_id:
        raise LongMemEvalV2AdapterError("official prompt row used a different opaque invocation")
    recall, expand = _validate_raw_trace(trace)

    query_tokens = _nonnegative_integer(
        official_prompt_row.get("memory_context_token_count"),
        "memory_context_token_count",
    )
    if query_tokens > 16_384:
        raise LongMemEvalV2AdapterError(
            "official delivered context exceeds the operation-sidecar token bound"
        )
    query_latency_ms = 1000.0 * _nonnegative_number(
        official_prompt_row.get("memory_query_duration_seconds"),
        "memory_query_duration_seconds",
    )
    operation_latency_ms = recall.latency_ms + expand.latency_ms
    if operation_latency_ms > query_latency_ms and not math.isclose(
        operation_latency_ms, query_latency_ms, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise LongMemEvalV2AdapterError(
            "measured Swarm operation latency exceeds the official memory query latency"
        )

    metadata = official_prompt_row.get("memory_post_query_metadata")
    if not isinstance(metadata, dict):
        raise LongMemEvalV2AdapterError("memory_post_query_metadata must be an object")
    expected_raw_digest = raw_trace_sha256(trace)
    if metadata.get(RAW_TRACE_DIGEST_METADATA_KEY) != expected_raw_digest:
        raise LongMemEvalV2AdapterError(
            "post_query_hook raw trace digest does not match the backend journal"
        )

    raw_ids = tuple(
        dict.fromkeys(
            (
                *recall.result_memory_ids,
                *expand.seed_memory_ids,
                *expand.result_memory_ids,
            )
        )
    )
    pseudonyms = {
        memory_id: _memory_pseudonym(
            memory_id,
            tier=tier,
            operating_point=operating_point,
            question_id=question_id,
        )
        for memory_id in raw_ids
    }
    operations: tuple[JsonObject, ...] = (
        {
            "sequence": 0,
            "invocation_id": expected_invocation_id(
                tier=tier,
                operating_point=operating_point,
                question_id=question_id,
                sequence=0,
                operation="recall_memory",
            ),
            "operation": "recall_memory",
            "success": True,
            "depth": 0,
            "seed_memory_ids": [],
            "result_memory_ids": [pseudonyms[item] for item in recall.result_memory_ids],
            "delivered_tokens": 0,
            "latency_ms": recall.latency_ms,
        },
        {
            "sequence": 1,
            "invocation_id": expected_invocation_id(
                tier=tier,
                operating_point=operating_point,
                question_id=question_id,
                sequence=1,
                operation="read_expand_memory",
            ),
            "operation": "read_expand_memory",
            "success": True,
            "depth": expand.depth,
            "seed_memory_ids": [pseudonyms[item] for item in expand.seed_memory_ids],
            "result_memory_ids": [pseudonyms[item] for item in expand.result_memory_ids],
            "delivered_tokens": query_tokens,
            "latency_ms": expand.latency_ms,
        },
    )
    trace_digest = canonical_sha256(
        {"operations": operations, "embedding": trace.embedding.as_json()}
    )
    return BoundQueryEvidence(
        question_id=question_id,
        domain=domain,
        query_tokens=query_tokens,
        query_latency_ms=query_latency_ms,
        operations=operations,
        embedding=trace.embedding,
        trace_sha256=trace_digest,
    )


class EvidenceLedger:
    """Thread-safe collection of post-harness, content-free query evidence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queries: dict[str, BoundQueryEvidence] = {}

    def record(self, evidence: BoundQueryEvidence) -> None:
        with self._lock:
            if evidence.question_id in self._queries:
                raise LongMemEvalV2AdapterError(
                    f"duplicate bound query evidence for {evidence.question_id!r}"
                )
            self._queries[evidence.question_id] = evidence

    def snapshot(self) -> tuple[BoundQueryEvidence, ...]:
        with self._lock:
            return tuple(self._queries[key] for key in sorted(self._queries))


def ledger_payload(
    ledger: EvidenceLedger,
    *,
    tier: str,
    operating_point: str,
    domain: str,
    dataset_revision: str,
    dataset_manifest_sha256: str,
    run_args: Mapping[str, Any],
    memory_config: Mapping[str, Any],
) -> JsonObject:
    config = _validated_adapter_config(memory_config)
    if (
        config.tier != tier
        or config.operating_point != operating_point
        or config.dataset_revision != dataset_revision
        or config.dataset_manifest_sha256 != dataset_manifest_sha256
    ):
        raise LongMemEvalV2AdapterError(
            "ledger identity differs from the exact runtime memory config"
        )
    if run_args.get("domain") != domain:
        raise LongMemEvalV2AdapterError("ledger domain differs from the exact runtime arguments")
    if run_args.get("model") != EXPECTED_READER_MODEL:
        raise LongMemEvalV2AdapterError("runtime arguments use a non-pinned reader model")
    if run_args.get("evaluator_model") != EXPECTED_JUDGE_MODEL:
        raise LongMemEvalV2AdapterError("runtime arguments use a non-pinned judge model")
    memory_config_payload = dict(memory_config)
    protocol_digest = protocol_sha256(run_args, memory_config_payload, METHOD)
    memory_config_digest = canonical_sha256(memory_config_payload)
    rows = [
        {
            "question_id": evidence.question_id,
            "domain": evidence.domain,
            "query_tokens": evidence.query_tokens,
            "query_latency_ms": evidence.query_latency_ms,
            "operations": [dict(operation) for operation in evidence.operations],
            "embedding": evidence.embedding.as_json(),
            "trace_sha256": evidence.trace_sha256,
        }
        for evidence in ledger.snapshot()
    ]
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "benchmark_repository_commit": PINNED_REPOSITORY_COMMIT,
        "tier": tier,
        "operating_point": operating_point,
        "domain": domain,
        "method": METHOD,
        "reader_model": EXPECTED_READER_MODEL,
        "judge_model": EXPECTED_JUDGE_MODEL,
        "dataset_revision": dataset_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "protocol_sha256": protocol_digest,
        "memory_config_sha256": memory_config_digest,
        "queries": rows,
    }


def write_ledger(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise LongMemEvalV2AdapterError(f"refusing to overwrite operation ledger: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n")


def protocol_sha256(run_args: Mapping[str, Any], memory_config: Any, method: str) -> str:
    filtered_args = {
        key: value for key, value in run_args.items() if key not in _RUN_LOCATION_FIELDS
    }
    return canonical_sha256(
        {"memory_config": memory_config, "method": method, "run_args": filtered_args}
    )


def _validated_adapter_config(memory_config: Mapping[str, Any]) -> AdapterConfig:
    if not isinstance(memory_config, dict) or set(memory_config) != {
        "memory_type",
        "memory_params",
    }:
        raise LongMemEvalV2AdapterError(
            "official memory_config fields differ from the pinned adapter protocol"
        )
    if memory_config.get("memory_type") != MEMORY_TYPE:
        raise LongMemEvalV2AdapterError(
            f"official memory_config.memory_type must equal {MEMORY_TYPE!r}"
        )
    return AdapterConfig.from_memory_params(memory_config.get("memory_params"))


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> JsonObject:
    value: JsonObject = {}
    for key, item in pairs:
        if key in value:
            raise LongMemEvalV2AdapterError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _read_json(path: Path) -> JsonObject:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_duplicate_rejecting_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LongMemEvalV2AdapterError(f"cannot read JSON object {path}") from exc
    if not isinstance(value, dict):
        raise LongMemEvalV2AdapterError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[JsonObject]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LongMemEvalV2AdapterError(f"cannot read JSONL {path}") from exc
    rows: list[JsonObject] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_duplicate_rejecting_object)
        except json.JSONDecodeError as exc:
            raise LongMemEvalV2AdapterError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise LongMemEvalV2AdapterError(f"JSONL row at {path}:{line_number} must be an object")
        rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    except OSError as exc:
        raise LongMemEvalV2AdapterError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def package_hashes(package_dir: Path) -> dict[str, str]:
    package_dir = package_dir.resolve()
    if package_dir.is_symlink() or not package_dir.is_dir():
        raise LongMemEvalV2AdapterError("official package must be a real directory")
    files: dict[str, str] = {}
    for root, directories, filenames in os.walk(package_dir, followlinks=False):
        root_path = Path(root)
        for name in directories:
            if (root_path / name).is_symlink():
                raise LongMemEvalV2AdapterError("official package cannot contain symlinks")
        for name in filenames:
            path = root_path / name
            if path.is_symlink() or not path.is_file():
                raise LongMemEvalV2AdapterError("official package contains a non-regular file")
            files[path.relative_to(package_dir).as_posix()] = _sha256_file(path)
    if not files:
        raise LongMemEvalV2AdapterError("official package is empty")
    return dict(sorted(files.items()))


def _query_summary(rows: Sequence[Mapping[str, Any]]) -> JsonObject:
    if not rows:
        raise LongMemEvalV2AdapterError("an operating point cannot have zero queries")
    tokens = [int(row["query_tokens"]) for row in rows]
    latencies = [float(row["query_latency_ms"]) for row in rows]
    ordered_latencies = sorted(latencies)
    return {
        "questions": len(rows),
        "query_token_observations": len(rows),
        "query_tokens_total": sum(tokens),
        "query_tokens_mean": sum(tokens) / len(tokens),
        "query_latency_observations": len(rows),
        "query_latency_total_ms": sum(latencies),
        "query_latency_mean_ms": mean(latencies),
        "query_latency_p95_ms": ordered_latencies[
            min(len(ordered_latencies) - 1, int(0.95 * len(ordered_latencies)))
        ],
        "query_failures": sum(bool(row["query_failed"]) for row in rows),
        "unanswered_questions": sum(bool(row["unanswered"]) for row in rows),
    }


def _validate_ledger(path: Path) -> tuple[JsonObject, dict[str, JsonObject]]:
    ledger = _read_json(path)
    expected_header = {
        "schema_version",
        "adapter_revision",
        "benchmark_repository_commit",
        "tier",
        "operating_point",
        "domain",
        "method",
        "reader_model",
        "judge_model",
        "dataset_revision",
        "dataset_manifest_sha256",
        "protocol_sha256",
        "memory_config_sha256",
        "queries",
    }
    if set(ledger) != expected_header:
        raise LongMemEvalV2AdapterError(f"{path} fields differ from ledger schema")
    exact = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "benchmark_repository_commit": PINNED_REPOSITORY_COMMIT,
        "method": METHOD,
        "reader_model": EXPECTED_READER_MODEL,
        "judge_model": EXPECTED_JUDGE_MODEL,
    }
    for key, expected in exact.items():
        if ledger.get(key) != expected:
            raise LongMemEvalV2AdapterError(f"{path} has mismatched {key}")
    for key in ("protocol_sha256", "memory_config_sha256"):
        value = ledger.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise LongMemEvalV2AdapterError(f"{path}.{key} must be a lowercase SHA-256 digest")
    rows = ledger.get("queries")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise LongMemEvalV2AdapterError(f"{path}.queries must be a list of objects")
    indexed: dict[str, JsonObject] = {}
    for row in rows:
        if set(row) != _QUERY_FIELDS:
            raise LongMemEvalV2AdapterError(f"{path} query fields differ from ledger schema")
        question_id = _required_text(row.get("question_id"), "ledger question_id")
        if question_id in indexed:
            raise LongMemEvalV2AdapterError(f"{path} repeats question {question_id!r}")
        operations = row.get("operations")
        if (
            not isinstance(operations, list)
            or len(operations) != 2
            or any(not isinstance(operation, dict) for operation in operations)
            or any(set(operation) != _OPERATION_FIELDS for operation in operations)
        ):
            raise LongMemEvalV2AdapterError(f"{path} has malformed operations")
        embedding = EmbeddingRuntimeEvidence.from_json(row.get("embedding"))
        if row.get("trace_sha256") != canonical_sha256(
            {"operations": operations, "embedding": embedding.as_json()}
        ):
            raise LongMemEvalV2AdapterError(f"{path} has a mismatched trace digest")
        indexed[question_id] = row
    return ledger, indexed


def build_operation_sidecar(package_dir: Path, ledger_paths: Sequence[Path]) -> JsonObject:
    """Build schema-v3 evidence only from a complete official package and ledgers."""

    package_dir = package_dir.resolve()
    overview = _read_json(package_dir / "submission_overview.json")
    tier = overview.get("tier")
    if tier not in {"small", "medium"}:
        raise LongMemEvalV2AdapterError("official package tier must be small or medium")
    if overview.get("method") != METHOD:
        raise LongMemEvalV2AdapterError(f"official package method must be {METHOD!r}")
    raw_points = overview.get("operating_points")
    if not isinstance(raw_points, list) or not raw_points:
        raise LongMemEvalV2AdapterError("official package must contain operating points")
    point_names: list[str] = []
    for point in raw_points:
        if not isinstance(point, dict):
            raise LongMemEvalV2AdapterError("operating point overview must be an object")
        point_names.append(_required_text(point.get("name"), "operating point name"))
    if len(set(point_names)) != len(point_names):
        raise LongMemEvalV2AdapterError("official package repeats an operating point")

    ledgers: dict[tuple[str, str], tuple[JsonObject, dict[str, JsonObject]]] = {}
    for path in ledger_paths:
        header, rows = _validate_ledger(path.resolve())
        key = (str(header.get("operating_point")), str(header.get("domain")))
        if key in ledgers:
            raise LongMemEvalV2AdapterError(f"duplicate ledger for {key}")
        if header.get("tier") != tier:
            raise LongMemEvalV2AdapterError("ledger tier differs from official package")
        if key[1] not in {"web", "enterprise"}:
            raise LongMemEvalV2AdapterError("ledger domain must be web or enterprise")
        ledgers[key] = (header, rows)
    expected_ledger_keys = {
        (point_name, domain) for point_name in point_names for domain in ("web", "enterprise")
    }
    if set(ledgers) != expected_ledger_keys:
        raise LongMemEvalV2AdapterError(
            "ledger operating-point/domain coverage differs from the official package"
        )

    sidecar_points: list[JsonObject] = []
    reference_dataset: tuple[str, str] | None = None
    for point_name in sorted(point_names):
        point_rows: list[JsonObject] = []
        point_protocol: str | None = None
        seen_question_ids: set[str] = set()
        for domain in ("web", "enterprise"):
            run_dir = package_dir / "operating_points" / point_name / domain
            run_args = _read_json(run_dir / "run_args.json")
            memory_config = _read_json(run_dir / "runtime_inputs/memory_config.json")
            if run_args.get("model") != EXPECTED_READER_MODEL:
                raise LongMemEvalV2AdapterError("official run uses a non-pinned reader model")
            if run_args.get("evaluator_model") != EXPECTED_JUDGE_MODEL:
                raise LongMemEvalV2AdapterError("official run uses a non-pinned judge model")
            header, ledger_rows = ledgers[(point_name, domain)]
            config = _validated_adapter_config(memory_config)
            dataset_identity = (
                _required_text(header.get("dataset_revision"), "dataset_revision"),
                _required_text(header.get("dataset_manifest_sha256"), "dataset_manifest_sha256"),
            )
            if reference_dataset is None:
                reference_dataset = dataset_identity
            elif dataset_identity != reference_dataset:
                raise LongMemEvalV2AdapterError("ledgers mix dataset revisions or manifests")
            if (
                config.dataset_revision != dataset_identity[0]
                or config.dataset_manifest_sha256 != dataset_identity[1]
                or config.operating_point != point_name
                or config.tier != tier
            ):
                raise LongMemEvalV2AdapterError(
                    "official memory config differs from the bound ledger protocol"
                )
            current_memory_config_digest = canonical_sha256(memory_config)
            if header.get("memory_config_sha256") != current_memory_config_digest:
                raise LongMemEvalV2AdapterError(
                    "ledger memory config digest differs from the official package"
                )
            current_protocol = protocol_sha256(run_args, memory_config, METHOD)
            if header.get("protocol_sha256") != current_protocol:
                raise LongMemEvalV2AdapterError(
                    "ledger protocol digest differs from the official package"
                )
            if point_protocol is None:
                point_protocol = current_protocol
            elif point_protocol != current_protocol:
                raise LongMemEvalV2AdapterError(
                    "web and enterprise runs use different operating-point protocols"
                )

            records = _read_jsonl(run_dir / "per_question.jsonl")
            records_by_id: dict[str, JsonObject] = {}
            for record in records:
                question_id = _required_text(record.get("question_id"), "record question_id")
                if question_id in records_by_id or question_id in seen_question_ids:
                    raise LongMemEvalV2AdapterError(
                        f"official package repeats question {question_id!r}"
                    )
                records_by_id[question_id] = record
                seen_question_ids.add(question_id)
            if set(records_by_id) != set(ledger_rows):
                raise LongMemEvalV2AdapterError(
                    f"{point_name}/{domain} ledger coverage differs from official records"
                )
            for question_id in sorted(records_by_id):
                record = records_by_id[question_id]
                bound = ledger_rows[question_id]
                if bound.get("domain") != domain:
                    raise LongMemEvalV2AdapterError("ledger query domain differs from run domain")
                tokens = _nonnegative_integer(
                    record.get("memory_context_token_count"), "official context tokens"
                )
                latency_ms = 1000.0 * _nonnegative_number(
                    record.get("memory_query_duration_seconds"), "official query latency"
                )
                if bound.get("query_tokens") != tokens or not math.isclose(
                    _nonnegative_number(bound.get("query_latency_ms"), "ledger query latency"),
                    latency_ms,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise LongMemEvalV2AdapterError(
                        "ledger token or latency accounting differs from official output"
                    )
                metadata = record.get("memory_post_query_metadata")
                if not isinstance(metadata, dict) or metadata.get(
                    TRACE_DIGEST_METADATA_KEY
                ) != bound.get("trace_sha256"):
                    raise LongMemEvalV2AdapterError(
                        "official record is not bound to the ledger operation trace"
                    )
                response = record.get("response_raw")
                if not isinstance(response, str):
                    raise LongMemEvalV2AdapterError("official response_raw must be a string")
                embedding = EmbeddingRuntimeEvidence.from_json(bound.get("embedding"))
                if not embedding.sota_capable or embedding.retrieval_mode != "openai_hybrid":
                    raise LongMemEvalV2AdapterError(
                        "publishable LongMemEval-V2 sidecars require openai_hybrid SOTA evidence"
                    )
                point_rows.append(
                    {
                        "question_id": question_id,
                        "domain": domain,
                        "query_tokens": tokens,
                        "query_latency_ms": latency_ms,
                        "query_failed": False,
                        "unanswered": not response.strip(),
                        "operations": bound["operations"],
                        "embedding": embedding.as_json(),
                    }
                )
        if len(point_rows) != EXPECTED_QUESTIONS:
            raise LongMemEvalV2AdapterError(
                f"operating point {point_name!r} must cover exactly {EXPECTED_QUESTIONS} questions"
            )
        assert point_protocol is not None
        point_rows.sort(key=lambda row: str(row["question_id"]))
        sidecar_points.append(
            {
                "name": point_name,
                "protocol_sha256": point_protocol,
                "summary": _query_summary(point_rows),
                "queries": point_rows,
            }
        )

    files = package_hashes(package_dir)
    assert reference_dataset is not None
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "benchmark_repository_commit": PINNED_REPOSITORY_COMMIT,
        "tier": tier,
        "method": METHOD,
        "reader_model": EXPECTED_READER_MODEL,
        "judge_model": EXPECTED_JUDGE_MODEL,
        "dataset_revision": reference_dataset[0],
        "dataset_manifest_sha256": reference_dataset[1],
        "dataset_identity_source": "caller-pinned-run-ledger",
        "iterative_search_read_expand": True,
        "semantic_embedding_proof": True,
        "package": {
            "tree_sha256": canonical_sha256(files),
            "files_sha256": files,
        },
        "operating_points": sidecar_points,
    }


def write_operation_sidecar(
    package_dir: Path, ledger_paths: Sequence[Path], output_path: Path
) -> JsonObject:
    package_dir = package_dir.resolve()
    output_path = output_path.resolve()
    if output_path == package_dir or package_dir in output_path.parents:
        raise LongMemEvalV2AdapterError("operation sidecar must remain external to the package")
    if output_path.exists() or output_path.is_symlink():
        raise LongMemEvalV2AdapterError(f"refusing to overwrite operation sidecar: {output_path}")
    payload = build_operation_sidecar(package_dir, ledger_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _validate_with_strict_compiler(
            package_dir, temporary_path, expected_tier=str(payload["tier"])
        )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return payload


def _validate_with_strict_compiler(
    package_dir: Path, sidecar_path: Path, *, expected_tier: str
) -> None:
    """Make the canonical compiler the final write gate, not a later suggestion."""

    compiler_path = (
        Path(__file__).resolve().parents[3] / "scripts" / "build_longmemeval_v2_report.py"
    )
    if not compiler_path.is_file():
        raise LongMemEvalV2AdapterError(
            "strict LongMemEval-V2 evidence compiler is missing from this checkout"
        )
    module_name = "_swarmbrain_longmemeval_v2_strict_compiler"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, compiler_path)
        if spec is None or spec.loader is None:
            raise LongMemEvalV2AdapterError("cannot load strict LongMemEval-V2 compiler")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    try:
        module.load_tier_evidence(
            package_dir,
            sidecar_path,
            expected_tier=expected_tier,
        )
    except Exception as exc:
        raise LongMemEvalV2AdapterError(
            f"strict compiler rejected generated operation sidecar: {exc}"
        ) from exc


__all__ = [
    "EvidenceLedger",
    "bind_query_trace",
    "build_operation_sidecar",
    "canonical_json",
    "canonical_sha256",
    "expected_invocation_id",
    "ledger_payload",
    "package_hashes",
    "protocol_sha256",
    "raw_trace_sha256",
    "write_ledger",
    "write_operation_sidecar",
]
