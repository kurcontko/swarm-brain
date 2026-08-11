#!/usr/bin/env python3
"""Compile strict LongMemEval-V2 SOTA evidence from official packages.

The official leaderboard package records answer quality and average memory-query
latency, but it does not preserve Swarm Brain's query-token accounting or a
machine-checkable declaration that iterative search/read/expand was enabled.
This compiler therefore requires one external Swarm evidence sidecar per tier.
It verifies every package file against the sidecar, independently recomputes
the official metrics and LAFS score, and binds every content-free operation
trace to the canonical digest preserved in the official per-question record.
Only then does it emit the schema consumed by ``benchmarks/sota/manifest.json``.

The output deliberately contains no wall-clock timestamp or absolute input
path. Identical evidence therefore produces byte-identical JSON regardless of
where the package is copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

PINNED_REPOSITORY_COMMIT = "ef67f10aacd9080c75aeb2dd527a0af25dc26f1b"
OFFICIAL_REPOSITORY = "https://github.com/xiaowu0162/LongMemEval-V2"
EXPECTED_QUESTIONS = 451
EXPECTED_READER_MODEL = "Qwen/Qwen3.5-9B"
EXPECTED_JUDGE_MODEL = "gpt-5.2"
SIDECAR_SCHEMA_VERSION = 3
TRACE_DIGEST_METADATA_KEY = "swarmbrain_operation_trace_sha256"
SOTA_EMBEDDING_PROVIDER = "OpenAICompatibleEmbeddingProvider"
SOTA_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
SOTA_EMBEDDING_DIMENSIONS = 4_096
SOTA_EMBEDDING_QUERY_INSTRUCTION = (
    "Given a question about past agent trajectories, retrieve relevant memory entries "
    "that help answer it."
)
SOTA_EMBEDDING_QUERY_INSTRUCTION_SHA256 = hashlib.sha256(
    SOTA_EMBEDDING_QUERY_INSTRUCTION.encode("utf-8")
).hexdigest()

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OPAQUE_MEMORY_ID_RE = re.compile(r"mem_[0-9a-f]{64}")
_INVOCATION_ID_RE = re.compile(r"inv_[0-9a-f]{64}")
_SAFE_ROOT_FILE_RE = re.compile(r"[A-Za-z0-9._-]+")
_TRACE_OPERATION_FIELDS = frozenset(
    {
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
)
_QUERY_EVIDENCE_FIELDS = frozenset(
    {
        "question_id",
        "domain",
        "query_tokens",
        "query_latency_ms",
        "query_failed",
        "unanswered",
        "operations",
        "embedding",
    }
)
_EMBEDDING_EVIDENCE_FIELDS = frozenset(
    {
        "retrieval_mode",
        "sota_capable",
        "provider",
        "model",
        "model_revision",
        "dimensions",
        "response_model_requirement",
        "query_instruction_sha256",
        "inserted_memories",
        "embedding_work_completed",
        "call_accounting",
        "exact_response_model_verified",
        "deterministic_fallback_used",
    }
)
_EMBEDDING_ACCOUNTING_FIELDS = frozenset(
    {
        "source",
        "document_inputs",
        "document_batch_calls",
        "document_successful_http_calls",
        "document_http_attempts",
        "query_calls",
        "query_successful_http_calls",
        "query_http_attempts",
    }
)
_MAX_TRACE_OPERATIONS = 16
_MAX_RECALL_MEMORY_IDS = 100
_MAX_READ_EXPAND_SEED_IDS = 8
_MAX_READ_EXPAND_RESULT_IDS = 100
_MAX_READ_EXPAND_DEPTH = 2
_MAX_OPERATION_DELIVERED_TOKENS = 16_384
_REQUIRED_RUN_FILES = (
    "aggregated_metrics.json",
    "per_question.jsonl",
    "run_args.json",
    "runtime_inputs/haystack.json",
    "runtime_inputs/memory_config.json",
    "runtime_inputs/questions.json",
)
_REQUIRED_RUN_ARG_KEYS = frozenset(
    {
        "api_key_env",
        "api_key_file",
        "base_url",
        "domain",
        "evaluator_api_key_env",
        "evaluator_api_key_file",
        "evaluator_base_url",
        "evaluator_max_completion_tokens",
        "evaluator_model",
        "evaluator_reasoning_effort",
        "evaluator_timeout_seconds",
        "haystack_path",
        "load_memory_dir",
        "max_completion_tokens",
        "memory_config_path",
        "memory_context_max_tokens",
        "model",
        "output_dir",
        "presence_penalty",
        "prompt_build_max_workers",
        "questions_path",
        "reader_enable_thinking",
        "reader_max_concurrent_requests",
        "reasoning_effort",
        "repetition_penalty",
        "save_memory",
        "shuffle_questions_seed",
        "skip_evaluation",
        "started_at_utc",
        "temperature",
        "timeout_seconds",
        "top_k",
        "top_p",
        "trajectories_path",
    }
)
_RUN_LOCATION_FIELDS = frozenset(
    {
        "domain",
        "haystack_path",
        "memory_config_path",
        "output_dir",
        "questions_path",
        "started_at_utc",
        "trajectories_path",
    }
)
_CATEGORY_MAP = {
    "static-environment": "static",
    "static-environment-abs": "static-abs",
    "dynamic-environment": "dynamic",
    "dynamic-environment-abs": "dynamic-abs",
    "procedure": "procedure",
    "procedure-abs": "procedure-abs",
    "errors-gotchas": "gotchas",
}
_NON_ABSTENTION_CATEGORIES = ("static", "dynamic", "procedure", "gotchas")
_ABSTENTION_CATEGORIES = ("static-abs", "dynamic-abs", "procedure-abs")
_COMBINED_CATEGORY_PAIRS = {
    "static": ("static", "static-abs"),
    "dynamic": ("dynamic", "dynamic-abs"),
    "procedure": ("procedure", "procedure-abs"),
}


class LongMemEvalV2EvidenceError(ValueError):
    """Inputs cannot support a comparable LongMemEval-V2 claim."""


@dataclass(frozen=True, slots=True)
class _LafsPoint:
    name: str
    accuracy_percentage_points: float
    latency_seconds: float


_REFERENCE_POINTS = {
    "small": (
        _LafsPoint("RAG: query -> slice + notes", 51.0, 0.2),
        _LafsPoint("Codex", 69.9, 177.2),
        _LafsPoint("AgentRunbook-R", 58.6, 26.9),
        _LafsPoint("AgentRunbook-C", 74.9, 108.3),
    ),
    "medium": (
        _LafsPoint("RAG: query -> slice + notes", 45.9, 0.3),
        _LafsPoint("Codex", 68.7, 185.8),
        _LafsPoint("AgentRunbook-R", 57.0, 25.8),
        _LafsPoint("AgentRunbook-C", 70.1, 139.9),
    ),
}


@dataclass(frozen=True, slots=True)
class TraceOperation:
    sequence: int
    invocation_id: str
    operation: str
    success: bool
    depth: int
    seed_memory_ids: tuple[str, ...]
    result_memory_ids: tuple[str, ...]
    delivered_tokens: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class EmbeddingEvidence:
    retrieval_mode: str
    provider: str
    model: str
    model_revision: str
    dimensions: int
    response_model_requirement: str
    query_instruction_sha256: str
    inserted_memories: int
    embedding_work_completed: int
    document_inputs: int
    document_batch_calls: int
    document_successful_http_calls: int
    document_http_attempts: int
    query_calls: int
    query_successful_http_calls: int
    query_http_attempts: int

    def as_json(self) -> dict[str, Any]:
        return {
            "retrieval_mode": self.retrieval_mode,
            "sota_capable": True,
            "provider": self.provider,
            "model": self.model,
            "model_revision": self.model_revision,
            "dimensions": self.dimensions,
            "response_model_requirement": self.response_model_requirement,
            "query_instruction_sha256": self.query_instruction_sha256,
            "inserted_memories": self.inserted_memories,
            "embedding_work_completed": self.embedding_work_completed,
            "call_accounting": {
                "source": "provider-observed",
                "document_inputs": self.document_inputs,
                "document_batch_calls": self.document_batch_calls,
                "document_successful_http_calls": self.document_successful_http_calls,
                "document_http_attempts": self.document_http_attempts,
                "query_calls": self.query_calls,
                "query_successful_http_calls": self.query_successful_http_calls,
                "query_http_attempts": self.query_http_attempts,
            },
            "exact_response_model_verified": True,
            "deterministic_fallback_used": False,
        }


@dataclass(frozen=True, slots=True)
class QueryEvidence:
    question_id: str
    domain: str
    query_tokens: int
    query_latency_ms: float
    query_failed: bool
    unanswered: bool
    operations: tuple[TraceOperation, ...]
    embedding: EmbeddingEvidence
    trace_sha256: str


@dataclass(frozen=True, slots=True)
class RunEvidence:
    domain: str
    question_ids: tuple[str, ...]
    question_signatures: tuple[tuple[str, str], ...]
    questions_json: str
    haystack_json: str
    records: tuple[dict[str, Any], ...]
    records_by_id: Mapping[str, dict[str, Any]]
    protocol_sha256: str
    dataset_revision: str
    dataset_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class OperatingPointEvidence:
    name: str
    questions: int
    accuracy: float
    memory_query_avg_seconds: float
    protocol_sha256: str
    queries: tuple[QueryEvidence, ...]


@dataclass(frozen=True, slots=True)
class TierEvidence:
    tier: str
    submission_name: str
    method: str
    dataset_revision: str
    dataset_manifest_sha256: str
    package_tree_sha256: str
    package_files_sha256: Mapping[str, str]
    sidecar_sha256: str
    question_ids: tuple[str, ...]
    question_signatures: tuple[tuple[str, str], ...]
    question_ids_sha256: str
    question_content_sha256: str
    lafs_gain: float
    operating_points: tuple[OperatingPointEvidence, ...]


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise LongMemEvalV2EvidenceError(f"duplicate JSON object key {key!r}")
        payload[key] = value
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw, object_pairs_hook=_duplicate_rejecting_object)
    except LongMemEvalV2EvidenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LongMemEvalV2EvidenceError(
            f"cannot read JSON object {path}: {type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise LongMemEvalV2EvidenceError(f"{path} must contain one JSON object")
    return payload


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw, object_pairs_hook=_duplicate_rejecting_object)
    except LongMemEvalV2EvidenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LongMemEvalV2EvidenceError(
            f"cannot read JSON list {path}: {type(exc).__name__}"
        ) from exc
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise LongMemEvalV2EvidenceError(f"{path} must contain a list of JSON objects")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LongMemEvalV2EvidenceError(f"cannot read JSONL {path}: {type(exc).__name__}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, object_pairs_hook=_duplicate_rejecting_object)
        except LongMemEvalV2EvidenceError:
            raise
        except json.JSONDecodeError as exc:
            raise LongMemEvalV2EvidenceError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise LongMemEvalV2EvidenceError(
                f"JSONL record at {path}:{line_number} must be an object"
            )
        records.append(value)
    return records


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LongMemEvalV2EvidenceError(f"{label} must be an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise LongMemEvalV2EvidenceError(f"{label} must be a list")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongMemEvalV2EvidenceError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LongMemEvalV2EvidenceError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise LongMemEvalV2EvidenceError(f"{label} must be at least {minimum}")
    return value


def _number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LongMemEvalV2EvidenceError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise LongMemEvalV2EvidenceError(f"{label} must be a finite number")
    if positive and result <= 0:
        raise LongMemEvalV2EvidenceError(f"{label} must be positive")
    if minimum is not None and result < minimum:
        raise LongMemEvalV2EvidenceError(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise LongMemEvalV2EvidenceError(f"{label} must be at most {maximum}")
    return result


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise LongMemEvalV2EvidenceError(f"{label} must be boolean")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    except OSError as exc:
        raise LongMemEvalV2EvidenceError(f"cannot hash {path}: {type(exc).__name__}") from exc
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _package_hashes(package_dir: Path) -> dict[str, str]:
    if package_dir.is_symlink() or not package_dir.is_dir():
        raise LongMemEvalV2EvidenceError(
            f"official package must be a real directory, not a symlink: {package_dir}"
        )
    files: dict[str, str] = {}
    for root, directories, filenames in os.walk(package_dir, followlinks=False):
        root_path = Path(root)
        for name in sorted(directories):
            path = root_path / name
            if path.is_symlink():
                raise LongMemEvalV2EvidenceError(f"official package contains symlink: {path}")
        for name in sorted(filenames):
            path = root_path / name
            if path.is_symlink() or not path.is_file():
                raise LongMemEvalV2EvidenceError(
                    f"official package contains a non-regular file: {path}"
                )
            relative = path.relative_to(package_dir).as_posix()
            files[relative] = _sha256_file(path)
    if not files:
        raise LongMemEvalV2EvidenceError(f"official package is empty: {package_dir}")
    return dict(sorted(files.items()))


def _tree_sha256(files: Mapping[str, str]) -> str:
    return _canonical_sha256(dict(sorted(files.items())))


def _checked_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise LongMemEvalV2EvidenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _assert_close(actual: Any, expected: float, *, label: str) -> None:
    value = _number(actual, label=label)
    if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise LongMemEvalV2EvidenceError(
            f"{label} is inconsistent: expected {expected!r}, got {value!r}"
        )


def _percentile_index(values: Sequence[float], probability: float) -> float:
    if not values:
        raise LongMemEvalV2EvidenceError("cannot compute a percentile of no values")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(probability * len(ordered)))]


def _pareto_frontier(points: Sequence[_LafsPoint]) -> list[_LafsPoint]:
    ordered = sorted(
        points,
        key=lambda point: (point.latency_seconds, -point.accuracy_percentage_points),
    )
    frontier: list[_LafsPoint] = []
    best_accuracy = -float("inf")
    for point in ordered:
        if point.latency_seconds <= 0:
            raise LongMemEvalV2EvidenceError(f"LAFS point {point.name!r} has nonpositive latency")
        if point.accuracy_percentage_points > best_accuracy:
            frontier.append(point)
            best_accuracy = point.accuracy_percentage_points
    return frontier


def _lafs(points: Sequence[_LafsPoint]) -> float:
    t_min = 1.0
    t_max = 200.0
    frontier = _pareto_frontier(points)
    breakpoints = {t_min, t_max}
    breakpoints.update(
        point.latency_seconds for point in frontier if t_min < point.latency_seconds < t_max
    )
    ordered = sorted(breakpoints)
    area = 0.0
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        available = [
            point.accuracy_percentage_points for point in frontier if point.latency_seconds <= left
        ]
        area += (max(available) if available else 0.0) * math.log(right / left)
    return area / math.log(t_max / t_min)


def _frontier_payload(points: Sequence[_LafsPoint]) -> list[dict[str, Any]]:
    return [
        {
            "name": point.name,
            "accuracy": point.accuracy_percentage_points,
            "latency_seconds": point.latency_seconds,
        }
        for point in _pareto_frontier(points)
    ]


def _lafs_summary(tier: str, submission_points: Sequence[_LafsPoint]) -> dict[str, Any]:
    reference = _REFERENCE_POINTS[tier]
    reference_lafs = _lafs(reference)
    combined = (*reference, *submission_points)
    submission_lafs = _lafs(combined)
    return {
        "tier": tier,
        "t_min_seconds": 1.0,
        "t_max_seconds": 200.0,
        "floor_accuracy": 0.0,
        "accuracy_unit": "percentage_points",
        "reference_lafs": reference_lafs,
        "submission_lafs": submission_lafs,
        "lafs_gain": submission_lafs - reference_lafs,
        "reference_frontier": _frontier_payload(reference),
        "submission_frontier": _frontier_payload(combined),
    }


def _compare_json(actual: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, float):
        _assert_close(actual, expected, label=label)
        return
    if isinstance(expected, bool):
        if not isinstance(actual, bool) or actual is not expected:
            raise LongMemEvalV2EvidenceError(
                f"{label} is inconsistent: expected {expected!r}, got {actual!r}"
            )
        return
    if isinstance(expected, int):
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise LongMemEvalV2EvidenceError(
                f"{label} is inconsistent: expected integer {expected!r}, got {actual!r}"
            )
        return
    if isinstance(expected, dict):
        actual_mapping = _mapping(actual, label=label)
        if set(actual_mapping) != set(expected):
            raise LongMemEvalV2EvidenceError(
                f"{label} fields differ: expected {sorted(expected)}, got {sorted(actual_mapping)}"
            )
        for key, value in expected.items():
            _compare_json(actual_mapping[key], value, label=f"{label}.{key}")
        return
    if isinstance(expected, list):
        actual_list = _list(actual, label=label)
        if len(actual_list) != len(expected):
            raise LongMemEvalV2EvidenceError(
                f"{label} length differs: expected {len(expected)}, got {len(actual_list)}"
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual_list, expected, strict=True)
        ):
            _compare_json(actual_item, expected_item, label=f"{label}[{index}]")
        return
    if actual != expected:
        raise LongMemEvalV2EvidenceError(
            f"{label} is inconsistent: expected {expected!r}, got {actual!r}"
        )


def _index_unique(
    records: Iterable[dict[str, Any]], *, key: str, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = _nonempty_string(record.get(key), label=f"{label} {key}")
        if identifier in indexed:
            raise LongMemEvalV2EvidenceError(f"{label} contains duplicate {key} {identifier!r}")
        indexed[identifier] = record
    return indexed


def _question_text(question: dict[str, Any], *, label: str) -> str:
    value = question.get("question")
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        return _nonempty_string(value.get("text"), label=f"{label} question text")
    raise LongMemEvalV2EvidenceError(f"{label} question must be text or a text/image object")


def _question_signature(question: dict[str, Any], *, domain: str) -> str:
    question_id = _nonempty_string(question.get("id"), label="runtime question id")
    if question.get("domain") != domain:
        raise LongMemEvalV2EvidenceError(
            f"runtime question {question_id!r} domain must be {domain!r}"
        )
    question_type = _nonempty_string(
        question.get("question_type"), label=f"runtime question {question_id} question_type"
    )
    if question_type not in _CATEGORY_MAP:
        raise LongMemEvalV2EvidenceError(
            f"runtime question {question_id!r} has unsupported question_type {question_type!r}"
        )
    answer = question.get("answer")
    if not isinstance(answer, str):
        raise LongMemEvalV2EvidenceError(f"runtime question {question_id!r} answer must be text")
    eval_function = _nonempty_string(
        question.get("eval_function"), label=f"runtime question {question_id} eval_function"
    )
    return _canonical_json(
        {
            "answer": answer,
            "domain": domain,
            "eval_function": eval_function,
            "id": question_id,
            "question": _question_text(question, label=f"runtime question {question_id}"),
            "question_type": question_type,
        }
    )


def _breakdown(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if not count:
        return {
            "count": 0,
            "pct_correct": None,
            "pct_answered_wrong": None,
            "pct_unknown": None,
        }
    unknown = sum(record["is_unknown"] for record in records)
    correct = sum(record["score_bool"] and not record["is_unknown"] for record in records)
    return {
        "count": count,
        "pct_correct": correct / count,
        "pct_answered_wrong": (count - correct - unknown) / count,
        "pct_unknown": unknown / count,
    }


def _timing_summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "avg_seconds": sum(values) / len(values),
        "p50_seconds": ordered[len(ordered) // 2],
        "p95_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_seconds": ordered[-1],
        "total_seconds": sum(values),
    }


def _aggregate_expected(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    non_abstention = [record for record in records if not record["is_abstention_problem"]]
    abstention = [record for record in records if record["is_abstention_problem"]]

    def score(rows: Sequence[dict[str, Any]]) -> float | None:
        return mean(float(record["score"]) for record in rows) if rows else None

    prompt_tokens = sum(record["usage"]["prompt_tokens"] for record in records)
    completion_tokens = sum(record["usage"]["completion_tokens"] for record in records)
    query_durations = [record["memory_query_duration_seconds"] for record in records]
    post_query_durations = [record["memory_post_query_duration_seconds"] for record in records]
    return {
        "overall": {
            "overall_full_set": score(records),
            "overall_non_abstention_only": score(non_abstention),
            "overall_abstention_only": score(abstention),
            "count_all_questions": len(records),
            "count_non_abstention": len(non_abstention),
            "count_abstention": len(abstention),
        },
        "non_abstention_by_category": {
            category: _breakdown(
                [record for record in non_abstention if record["category"] == category]
            )
            for category in _NON_ABSTENTION_CATEGORIES
        },
        "abstention_by_category": {
            category: _breakdown(
                [record for record in abstention if record["category"] == category]
            )
            for category in _ABSTENTION_CATEGORIES
        },
        "combined_abstention_by_category": {
            category: _breakdown([record for record in records if record["category"] in pair])
            for category, pair in _COMBINED_CATEGORY_PAIRS.items()
        },
        "abstention_overall": _breakdown(abstention),
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "avg_prompt_tokens": prompt_tokens / len(records),
            "avg_completion_tokens": completion_tokens / len(records),
            "avg_total_tokens": (prompt_tokens + completion_tokens) / len(records),
        },
        "memory_context": {
            "avg_original_tokens": mean(
                record["memory_context_original_token_count"] for record in records
            ),
            "avg_final_tokens": mean(record["memory_context_token_count"] for record in records),
            "num_truncated_sequences": sum(
                record["memory_context_was_truncated"] for record in records
            ),
        },
        "memory_query": _timing_summary(query_durations),
        "memory_post_query": _timing_summary(post_query_durations),
    }


def _validate_aggregated_metrics(
    actual: dict[str, Any], expected: dict[str, Any], *, label: str
) -> None:
    for section, expected_value in expected.items():
        if section not in actual:
            raise LongMemEvalV2EvidenceError(f"{label} is missing {section}")
        _compare_json(actual[section], expected_value, label=f"{label}.{section}")


def _validate_run_record(
    record: dict[str, Any], question: dict[str, Any], *, domain: str, label: str
) -> None:
    question_id = _nonempty_string(record.get("question_id"), label=f"{label} question_id")
    question_type = _nonempty_string(record.get("question_type"), label=f"{label} question_type")
    if question_type != question["question_type"]:
        raise LongMemEvalV2EvidenceError(f"{label} question_type differs from runtime input")
    expected_category = _CATEGORY_MAP[question_type]
    if record.get("category") != expected_category:
        raise LongMemEvalV2EvidenceError(f"{label} category differs from official mapping")
    expected_abstention = expected_category.endswith("-abs")
    if record.get("is_abstention_problem") is not expected_abstention:
        raise LongMemEvalV2EvidenceError(f"{label} abstention flag differs from question type")
    if record.get("eval_function") != question["eval_function"]:
        raise LongMemEvalV2EvidenceError(f"{label} eval_function differs from runtime input")
    if record.get("answer_gold") != question["answer"]:
        raise LongMemEvalV2EvidenceError(f"{label} gold answer differs from runtime input")
    if not isinstance(record.get("response_raw"), str):
        raise LongMemEvalV2EvidenceError(f"{label} response_raw must be text")
    is_unknown = _boolean(record.get("is_unknown"), label=f"{label} is_unknown")
    score_bool = _boolean(record.get("score_bool"), label=f"{label} score_bool")
    score = _number(record.get("score"), label=f"{label} score", minimum=0, maximum=1)
    if score not in {0.0, 1.0} or score != float(score_bool):
        raise LongMemEvalV2EvidenceError(f"{label} score and score_bool are inconsistent")
    if is_unknown and score_bool:
        raise LongMemEvalV2EvidenceError(f"{label} cannot be both unknown and correct")
    usage = _mapping(record.get("usage"), label=f"{label} usage")
    prompt_tokens = _integer(
        usage.get("prompt_tokens"), label=f"{label} usage.prompt_tokens", minimum=0
    )
    completion_tokens = _integer(
        usage.get("completion_tokens"), label=f"{label} usage.completion_tokens", minimum=0
    )
    total_tokens = _integer(
        usage.get("total_tokens"), label=f"{label} usage.total_tokens", minimum=0
    )
    if total_tokens != prompt_tokens + completion_tokens:
        raise LongMemEvalV2EvidenceError(f"{label} usage token totals are inconsistent")
    _number(
        record.get("memory_query_duration_seconds"),
        label=f"{label} memory_query_duration_seconds",
        minimum=0,
    )
    _number(
        record.get("memory_post_query_duration_seconds"),
        label=f"{label} memory_post_query_duration_seconds",
        minimum=0,
    )
    _integer(
        record.get("memory_context_original_token_count"),
        label=f"{label} memory_context_original_token_count",
        minimum=0,
    )
    _integer(
        record.get("memory_context_token_count"),
        label=f"{label} memory_context_token_count",
        minimum=0,
    )
    _boolean(
        record.get("memory_context_was_truncated"),
        label=f"{label} memory_context_was_truncated",
    )
    if question.get("domain") != domain:
        raise LongMemEvalV2EvidenceError(
            f"{label} runtime question {question_id!r} has a cross-domain value"
        )


def _protocol_sha256(run_args: Mapping[str, Any], memory_config: Any, method: str) -> str:
    filtered_args = {
        key: value for key, value in run_args.items() if key not in _RUN_LOCATION_FIELDS
    }
    return _canonical_sha256(
        {"memory_config": memory_config, "method": method, "run_args": filtered_args}
    )


def _expected_invocation_id(
    *, tier: str, point_name: str, question_id: str, sequence: int, operation: str
) -> str:
    """Return the path-independent invocation pseudonym required by the sidecar."""

    digest = _canonical_sha256(
        {
            "operation": operation,
            "operating_point": point_name,
            "question_id": question_id,
            "sequence": sequence,
            "tier": tier,
        }
    )
    return f"inv_{digest}"


def _opaque_memory_ids(
    value: Any,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> tuple[str, ...]:
    raw = _list(value, label=label)
    if not minimum <= len(raw) <= maximum:
        raise LongMemEvalV2EvidenceError(
            f"{label} must contain between {minimum} and {maximum} opaque memory IDs"
        )
    identifiers: list[str] = []
    for index, identifier in enumerate(raw):
        if not isinstance(identifier, str) or not _OPAQUE_MEMORY_ID_RE.fullmatch(identifier):
            raise LongMemEvalV2EvidenceError(
                f"{label}[{index}] must be a content-free mem_<sha256> identifier"
            )
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise LongMemEvalV2EvidenceError(f"{label} must contain unique opaque memory IDs")
    return tuple(identifiers)


def _validate_embedding_evidence(raw: Any, *, label: str) -> EmbeddingEvidence:
    payload = _mapping(raw, label=label)
    if set(payload) != _EMBEDDING_EVIDENCE_FIELDS:
        raise LongMemEvalV2EvidenceError(
            f"{label} must contain only content-free embedding evidence fields "
            f"{sorted(_EMBEDDING_EVIDENCE_FIELDS)}"
        )
    exact = {
        "retrieval_mode": "openai_hybrid",
        "sota_capable": True,
        "provider": SOTA_EMBEDDING_PROVIDER,
        "model": SOTA_EMBEDDING_MODEL,
        "response_model_requirement": SOTA_EMBEDDING_MODEL,
        "query_instruction_sha256": SOTA_EMBEDDING_QUERY_INSTRUCTION_SHA256,
        "exact_response_model_verified": True,
        "deterministic_fallback_used": False,
    }
    for field, expected in exact.items():
        if payload.get(field) != expected or type(payload.get(field)) is not type(expected):
            raise LongMemEvalV2EvidenceError(
                f"{label}.{field} must equal pinned value {expected!r}"
            )
    revision = _nonempty_string(payload.get("model_revision"), label=f"{label}.model_revision")
    if (
        revision != revision.strip()
        or len(revision) > 255
        or any(ord(character) < 32 for character in revision)
    ):
        raise LongMemEvalV2EvidenceError(
            f"{label}.model_revision must be a public identifier of at most 255 characters"
        )
    dimensions = _integer(payload.get("dimensions"), label=f"{label}.dimensions", minimum=32)
    if dimensions > SOTA_EMBEDDING_DIMENSIONS:
        raise LongMemEvalV2EvidenceError(
            f"{label}.dimensions must not exceed {SOTA_EMBEDDING_DIMENSIONS}"
        )
    inserted = _integer(
        payload.get("inserted_memories"), label=f"{label}.inserted_memories", minimum=1
    )
    completed = _integer(
        payload.get("embedding_work_completed"),
        label=f"{label}.embedding_work_completed",
        minimum=1,
    )
    if completed != inserted:
        raise LongMemEvalV2EvidenceError(
            f"{label} does not prove complete inserted-memory embedding coverage"
        )
    accounting = _mapping(payload.get("call_accounting"), label=f"{label}.call_accounting")
    if set(accounting) != _EMBEDDING_ACCOUNTING_FIELDS:
        raise LongMemEvalV2EvidenceError(
            f"{label}.call_accounting fields differ from the provider-observed schema"
        )
    if accounting.get("source") != "provider-observed":
        raise LongMemEvalV2EvidenceError(
            f"{label}.call_accounting.source must be 'provider-observed'"
        )
    document_inputs = _integer(
        accounting.get("document_inputs"),
        label=f"{label}.call_accounting.document_inputs",
        minimum=0,
    )
    document_batches = _integer(
        accounting.get("document_batch_calls"),
        label=f"{label}.call_accounting.document_batch_calls",
        minimum=0,
    )
    document_successes = _integer(
        accounting.get("document_successful_http_calls"),
        label=f"{label}.call_accounting.document_successful_http_calls",
        minimum=0,
    )
    document_attempts = _integer(
        accounting.get("document_http_attempts"),
        label=f"{label}.call_accounting.document_http_attempts",
        minimum=0,
    )
    if not document_inputs == document_batches == document_successes == inserted:
        raise LongMemEvalV2EvidenceError(
            f"{label} document provider calls do not reconcile with inserted memories"
        )
    if document_attempts < document_successes:
        raise LongMemEvalV2EvidenceError(
            f"{label} document HTTP attempts are below successful calls"
        )
    query_calls = _integer(
        accounting.get("query_calls"),
        label=f"{label}.call_accounting.query_calls",
        minimum=0,
    )
    query_successes = _integer(
        accounting.get("query_successful_http_calls"),
        label=f"{label}.call_accounting.query_successful_http_calls",
        minimum=0,
    )
    query_attempts = _integer(
        accounting.get("query_http_attempts"),
        label=f"{label}.call_accounting.query_http_attempts",
        minimum=0,
    )
    if query_calls != 1 or query_successes != 1 or query_attempts < 1:
        raise LongMemEvalV2EvidenceError(
            f"{label} does not prove one successful query embedding HTTP call"
        )
    return EmbeddingEvidence(
        retrieval_mode="openai_hybrid",
        provider=SOTA_EMBEDDING_PROVIDER,
        model=SOTA_EMBEDDING_MODEL,
        model_revision=revision,
        dimensions=dimensions,
        response_model_requirement=SOTA_EMBEDDING_MODEL,
        query_instruction_sha256=SOTA_EMBEDDING_QUERY_INSTRUCTION_SHA256,
        inserted_memories=inserted,
        embedding_work_completed=completed,
        document_inputs=document_inputs,
        document_batch_calls=document_batches,
        document_successful_http_calls=document_successes,
        document_http_attempts=document_attempts,
        query_calls=query_calls,
        query_successful_http_calls=query_successes,
        query_http_attempts=query_attempts,
    )


def _operation_trace_sha256(operations: Sequence[TraceOperation]) -> str:
    payload = [
        {
            "sequence": operation.sequence,
            "invocation_id": operation.invocation_id,
            "operation": operation.operation,
            "success": operation.success,
            "depth": operation.depth,
            "seed_memory_ids": list(operation.seed_memory_ids),
            "result_memory_ids": list(operation.result_memory_ids),
            "delivered_tokens": operation.delivered_tokens,
            "latency_ms": operation.latency_ms,
        }
        for operation in operations
    ]
    return _canonical_sha256(payload)


def _query_proof_sha256(operations: Sequence[TraceOperation], embedding: EmbeddingEvidence) -> str:
    operations_payload = [
        {
            "sequence": operation.sequence,
            "invocation_id": operation.invocation_id,
            "operation": operation.operation,
            "success": operation.success,
            "depth": operation.depth,
            "seed_memory_ids": list(operation.seed_memory_ids),
            "result_memory_ids": list(operation.result_memory_ids),
            "delivered_tokens": operation.delivered_tokens,
            "latency_ms": operation.latency_ms,
        }
        for operation in operations
    ]
    return _canonical_sha256({"operations": operations_payload, "embedding": embedding.as_json()})


def _validate_operation_trace(
    raw_operations: Any,
    *,
    tier: str,
    point_name: str,
    question_id: str,
    query_tokens: int,
    query_latency_ms: float,
    official_record: Mapping[str, Any],
    seen_invocation_ids: set[str],
    label: str,
    embedding: EmbeddingEvidence | None = None,
) -> tuple[tuple[TraceOperation, ...], str]:
    raw = _list(raw_operations, label=f"{label}.operations")
    if not 2 <= len(raw) <= _MAX_TRACE_OPERATIONS:
        raise LongMemEvalV2EvidenceError(
            f"{label}.operations must contain between 2 and {_MAX_TRACE_OPERATIONS} operations"
        )
    operations: list[TraceOperation] = []
    preceding_search_results: list[frozenset[str]] = []
    linked_successful_expands = 0
    local_invocations: set[str] = set()
    for expected_sequence, raw_operation in enumerate(raw):
        operation_payload = _mapping(
            raw_operation, label=f"{label}.operations[{expected_sequence}]"
        )
        if set(operation_payload) != _TRACE_OPERATION_FIELDS:
            raise LongMemEvalV2EvidenceError(
                f"{label}.operations[{expected_sequence}] must contain only content-free "
                f"trace fields {sorted(_TRACE_OPERATION_FIELDS)}"
            )
        sequence = _integer(
            operation_payload.get("sequence"),
            label=f"{label}.operations[{expected_sequence}].sequence",
            minimum=0,
        )
        if sequence != expected_sequence:
            raise LongMemEvalV2EvidenceError(
                f"{label}.operations sequence numbers must be contiguous from zero"
            )
        operation = _nonempty_string(
            operation_payload.get("operation"),
            label=f"{label}.operations[{sequence}].operation",
        )
        if operation not in {"recall_memory", "read_expand_memory"}:
            raise LongMemEvalV2EvidenceError(
                f"{label}.operations[{sequence}].operation is not an evidence operation"
            )
        invocation_id = _nonempty_string(
            operation_payload.get("invocation_id"),
            label=f"{label}.operations[{sequence}].invocation_id",
        )
        if not _INVOCATION_ID_RE.fullmatch(invocation_id):
            raise LongMemEvalV2EvidenceError(
                f"{label}.operations[{sequence}].invocation_id must be inv_<sha256>"
            )
        expected_invocation_id = _expected_invocation_id(
            tier=tier,
            point_name=point_name,
            question_id=question_id,
            sequence=sequence,
            operation=operation,
        )
        if invocation_id != expected_invocation_id:
            raise LongMemEvalV2EvidenceError(
                f"{label}.operations[{sequence}] has a nondeterministic invocation_id"
            )
        if invocation_id in local_invocations or invocation_id in seen_invocation_ids:
            raise LongMemEvalV2EvidenceError(
                f"{label}.operations[{sequence}] reuses invocation_id {invocation_id!r}"
            )
        local_invocations.add(invocation_id)
        success = _boolean(
            operation_payload.get("success"),
            label=f"{label}.operations[{sequence}].success",
        )
        delivered_tokens = _integer(
            operation_payload.get("delivered_tokens"),
            label=f"{label}.operations[{sequence}].delivered_tokens",
            minimum=0,
        )
        if delivered_tokens > _MAX_OPERATION_DELIVERED_TOKENS:
            raise LongMemEvalV2EvidenceError(
                f"{label}.operations[{sequence}].delivered_tokens exceeds "
                f"{_MAX_OPERATION_DELIVERED_TOKENS}"
            )
        latency_ms = _number(
            operation_payload.get("latency_ms"),
            label=f"{label}.operations[{sequence}].latency_ms",
            minimum=0,
        )
        if latency_ms > query_latency_ms and not math.isclose(
            latency_ms, query_latency_ms, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise LongMemEvalV2EvidenceError(
                f"{label}.operations[{sequence}].latency_ms exceeds official query latency"
            )
        if operation == "recall_memory":
            depth = _integer(
                operation_payload.get("depth"),
                label=f"{label}.operations[{sequence}].depth",
                minimum=0,
            )
            if depth != 0:
                raise LongMemEvalV2EvidenceError(
                    f"{label}.operations[{sequence}] recall_memory depth must be zero"
                )
            seeds = _opaque_memory_ids(
                operation_payload.get("seed_memory_ids"),
                label=f"{label}.operations[{sequence}].seed_memory_ids",
                minimum=0,
                maximum=0,
            )
            results = _opaque_memory_ids(
                operation_payload.get("result_memory_ids"),
                label=f"{label}.operations[{sequence}].result_memory_ids",
                minimum=0,
                maximum=_MAX_RECALL_MEMORY_IDS,
            )
            if success and results:
                preceding_search_results.append(frozenset(results))
        else:
            depth = _integer(
                operation_payload.get("depth"),
                label=f"{label}.operations[{sequence}].depth",
                minimum=1,
            )
            if depth > _MAX_READ_EXPAND_DEPTH:
                raise LongMemEvalV2EvidenceError(
                    f"{label}.operations[{sequence}].depth exceeds {_MAX_READ_EXPAND_DEPTH}"
                )
            seeds = _opaque_memory_ids(
                operation_payload.get("seed_memory_ids"),
                label=f"{label}.operations[{sequence}].seed_memory_ids",
                minimum=1,
                maximum=_MAX_READ_EXPAND_SEED_IDS,
            )
            results = _opaque_memory_ids(
                operation_payload.get("result_memory_ids"),
                label=f"{label}.operations[{sequence}].result_memory_ids",
                minimum=1 if success else 0,
                maximum=_MAX_READ_EXPAND_RESULT_IDS,
            )
            seed_set = frozenset(seeds)
            linked_to_search = any(
                seed_set.issubset(search_results) for search_results in preceding_search_results
            )
            if not linked_to_search:
                raise LongMemEvalV2EvidenceError(
                    f"{label}.operations[{sequence}] read_expand_memory seeds did not come "
                    "from one preceding successful recall_memory search"
                )
            if success:
                linked_successful_expands += 1
        if not success and (results or delivered_tokens):
            raise LongMemEvalV2EvidenceError(
                f"{label}.operations[{sequence}] failed operations cannot deliver IDs or tokens"
            )
        operations.append(
            TraceOperation(
                sequence=sequence,
                invocation_id=invocation_id,
                operation=operation,
                success=success,
                depth=depth,
                seed_memory_ids=seeds,
                result_memory_ids=results,
                delivered_tokens=delivered_tokens,
                latency_ms=latency_ms,
            )
        )
    if not preceding_search_results or linked_successful_expands == 0:
        raise LongMemEvalV2EvidenceError(
            f"{label}.operations must prove successful recall_memory followed by linked "
            "successful read_expand_memory"
        )
    operation_tokens = sum(operation.delivered_tokens for operation in operations)
    if operation_tokens != query_tokens:
        raise LongMemEvalV2EvidenceError(
            f"{label}.operations delivered-token total does not equal query_tokens"
        )
    official_delivered_tokens = _integer(
        official_record.get("memory_context_token_count"),
        label=f"{label} official memory_context_token_count",
        minimum=0,
    )
    if operation_tokens != official_delivered_tokens:
        raise LongMemEvalV2EvidenceError(
            f"{label}.operations delivered-token total does not equal official delivered context"
        )
    operation_latency = sum(operation.latency_ms for operation in operations)
    if operation_latency > query_latency_ms and not math.isclose(
        operation_latency, query_latency_ms, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise LongMemEvalV2EvidenceError(
            f"{label}.operations total latency exceeds official query latency"
        )
    normalized_operations = tuple(operations)
    trace_sha256 = (
        _operation_trace_sha256(normalized_operations)
        if embedding is None
        else _query_proof_sha256(normalized_operations, embedding)
    )
    post_query_metadata = _mapping(
        official_record.get("memory_post_query_metadata"),
        label=f"{label} official memory_post_query_metadata",
    )
    bound_digest = _checked_sha256(
        post_query_metadata.get(TRACE_DIGEST_METADATA_KEY),
        label=f"{label} official memory_post_query_metadata.{TRACE_DIGEST_METADATA_KEY}",
    )
    if bound_digest != trace_sha256:
        raise LongMemEvalV2EvidenceError(
            f"{label} canonical query-proof SHA does not match official memory_post_query_metadata"
        )
    seen_invocation_ids.update(local_invocations)
    return normalized_operations, trace_sha256


def _load_run(run_dir: Path, *, domain: str, method: str, tier: str) -> RunEvidence:
    for relative in _REQUIRED_RUN_FILES:
        path = run_dir / relative
        if not path.is_file() or path.is_symlink():
            raise LongMemEvalV2EvidenceError(f"missing official run file {path}")
    run_args = _read_json(run_dir / "run_args.json")
    missing_args = sorted(_REQUIRED_RUN_ARG_KEYS - set(run_args))
    if missing_args:
        raise LongMemEvalV2EvidenceError(
            f"{run_dir / 'run_args.json'} is missing official harness fields: {missing_args}"
        )
    if run_args.get("domain") != domain:
        raise LongMemEvalV2EvidenceError(f"{run_dir} run_args domain must be {domain!r}")
    if run_args.get("model") != EXPECTED_READER_MODEL:
        raise LongMemEvalV2EvidenceError(
            f"{run_dir} must use exact fixed reader {EXPECTED_READER_MODEL!r}"
        )
    if run_args.get("evaluator_model") != EXPECTED_JUDGE_MODEL:
        raise LongMemEvalV2EvidenceError(f"{run_dir} must use exact judge {EXPECTED_JUDGE_MODEL!r}")
    if "method" in run_args and run_args["method"] != method:
        raise LongMemEvalV2EvidenceError(f"{run_dir} run_args method differs from package method")
    if "tier" in run_args and run_args["tier"] != tier:
        raise LongMemEvalV2EvidenceError(f"{run_dir} run_args tier differs from package tier")
    if run_args.get("skip_evaluation") is not False:
        raise LongMemEvalV2EvidenceError(f"{run_dir} must be a completed evaluation run")
    _boolean(run_args.get("reader_enable_thinking"), label=f"{run_dir} reader_enable_thinking")
    _nonempty_string(run_args.get("started_at_utc"), label=f"{run_dir} started_at_utc")

    questions_path = run_dir / "runtime_inputs/questions.json"
    questions = _read_json_list(questions_path)
    questions_by_id = _index_unique(questions, key="id", label=str(questions_path))
    signatures = {
        question_id: _question_signature(question, domain=domain)
        for question_id, question in questions_by_id.items()
    }
    haystack_path = run_dir / "runtime_inputs/haystack.json"
    haystack = _read_json(haystack_path)
    if set(haystack) != set(questions_by_id):
        missing = sorted(set(questions_by_id) - set(haystack))[:3]
        extra = sorted(set(haystack) - set(questions_by_id))[:3]
        raise LongMemEvalV2EvidenceError(
            f"{haystack_path} coverage differs from questions; missing={missing}, extra={extra}"
        )
    for question_id, trajectory_ids in haystack.items():
        values = _list(trajectory_ids, label=f"{haystack_path} haystack {question_id}")
        if not all(isinstance(value, str) and value for value in values):
            raise LongMemEvalV2EvidenceError(
                f"{haystack_path} haystack {question_id!r} contains invalid trajectory IDs"
            )
        if len(values) != len(set(values)):
            raise LongMemEvalV2EvidenceError(
                f"{haystack_path} haystack {question_id!r} contains duplicate trajectory IDs"
            )

    records_path = run_dir / "per_question.jsonl"
    records = _read_jsonl(records_path)
    records_by_id = _index_unique(records, key="question_id", label=str(records_path))
    if set(records_by_id) != set(questions_by_id):
        missing = sorted(set(questions_by_id) - set(records_by_id))[:3]
        extra = sorted(set(records_by_id) - set(questions_by_id))[:3]
        raise LongMemEvalV2EvidenceError(
            f"{records_path} coverage differs from questions; missing={missing}, extra={extra}"
        )
    normalized_records: list[dict[str, Any]] = []
    for question_id in sorted(records_by_id):
        record = records_by_id[question_id]
        _validate_run_record(
            record,
            questions_by_id[question_id],
            domain=domain,
            label=f"{records_path} record {question_id}",
        )
        normalized_records.append(record)
    expected_aggregated = _aggregate_expected(normalized_records)
    _validate_aggregated_metrics(
        _read_json(run_dir / "aggregated_metrics.json"),
        expected_aggregated,
        label=str(run_dir / "aggregated_metrics.json"),
    )
    memory_config = _read_json(run_dir / "runtime_inputs/memory_config.json")
    if memory_config.get("memory_type") != "swarmbrain":
        raise LongMemEvalV2EvidenceError(
            f"{run_dir} memory_config must use the Swarm Brain adapter"
        )
    memory_params = _mapping(
        memory_config.get("memory_params"), label=f"{run_dir} memory_config.memory_params"
    )
    dataset_revision = _nonempty_string(
        memory_params.get("dataset_revision"),
        label=f"{run_dir} memory_config.memory_params.dataset_revision",
    )
    dataset_manifest_sha256 = _checked_sha256(
        memory_params.get("dataset_manifest_sha256"),
        label=f"{run_dir} memory_config.memory_params.dataset_manifest_sha256",
    )
    return RunEvidence(
        domain=domain,
        question_ids=tuple(sorted(questions_by_id)),
        question_signatures=tuple(sorted(signatures.items())),
        questions_json=_canonical_json(questions),
        haystack_json=_canonical_json(haystack),
        records=tuple(normalized_records),
        records_by_id=records_by_id,
        protocol_sha256=_protocol_sha256(run_args, memory_config, method),
        dataset_revision=dataset_revision,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )


def _metric_overview(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    def category_accuracy(categories: set[str]) -> float:
        selected = [record for record in records if record["category"] in categories]
        if not selected:
            raise LongMemEvalV2EvidenceError(
                f"combined official records contain no questions for {sorted(categories)}"
            )
        correct = sum(record["score_bool"] and not record["is_unknown"] for record in selected)
        return correct / len(selected)

    return {
        "overall_full_set": mean(float(record["score"]) for record in records),
        "gotchas_accuracy": category_accuracy({"gotchas"}),
        "static_accuracy": category_accuracy({"static", "static-abs"}),
        "dynamic_accuracy": category_accuracy({"dynamic", "dynamic-abs"}),
        "procedure_accuracy": category_accuracy({"procedure", "procedure-abs"}),
        "memory_query_avg_seconds": mean(
            record["memory_query_duration_seconds"] for record in records
        ),
    }


def _validate_metric_overview(
    actual: dict[str, Any], expected: Mapping[str, float], *, label: str
) -> None:
    if set(actual) != set(expected):
        raise LongMemEvalV2EvidenceError(
            f"{label} fields differ: expected {sorted(expected)}, got {sorted(actual)}"
        )
    for key, value in expected.items():
        _assert_close(actual[key], value, label=f"{label}.{key}")
    for key in (
        "overall_full_set",
        "gotchas_accuracy",
        "static_accuracy",
        "dynamic_accuracy",
        "procedure_accuracy",
    ):
        _number(actual[key], label=f"{label}.{key}", minimum=0, maximum=1)
    _number(
        actual["memory_query_avg_seconds"],
        label=f"{label}.memory_query_avg_seconds",
        positive=True,
    )


def _sidecar_points(sidecar: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    values = _list(sidecar.get("operating_points"), label=f"{label}.operating_points")
    if not values or not all(isinstance(value, dict) for value in values):
        raise LongMemEvalV2EvidenceError(
            f"{label}.operating_points must contain at least one object"
        )
    return _index_unique(values, key="name", label=f"{label}.operating_points")


def _validate_sidecar_queries(
    sidecar_point: dict[str, Any],
    *,
    tier: str,
    point_name: str,
    records_by_id: Mapping[str, tuple[str, dict[str, Any]]],
    protocol_sha256: str,
    seen_invocation_ids: set[str],
    label: str,
) -> tuple[QueryEvidence, ...]:
    if sidecar_point.get("protocol_sha256") != protocol_sha256:
        raise LongMemEvalV2EvidenceError(f"{label} protocol_sha256 differs from run artifacts")
    raw_queries = _list(sidecar_point.get("queries"), label=f"{label}.queries")
    if not all(isinstance(query, dict) for query in raw_queries):
        raise LongMemEvalV2EvidenceError(f"{label}.queries must contain objects")
    indexed = _index_unique(raw_queries, key="question_id", label=f"{label}.queries")
    if set(indexed) != set(records_by_id):
        missing = sorted(set(records_by_id) - set(indexed))[:3]
        extra = sorted(set(indexed) - set(records_by_id))[:3]
        raise LongMemEvalV2EvidenceError(
            f"{label}.queries coverage differs from official records; missing={missing}, extra={extra}"
        )
    queries: list[QueryEvidence] = []
    for question_id in sorted(indexed):
        query = indexed[question_id]
        if set(query) != _QUERY_EVIDENCE_FIELDS:
            raise LongMemEvalV2EvidenceError(
                f"{label} {question_id} must contain only content-free query evidence "
                f"fields {sorted(_QUERY_EVIDENCE_FIELDS)}"
            )
        expected_domain, record = records_by_id[question_id]
        domain = _nonempty_string(query.get("domain"), label=f"{label} {question_id} domain")
        if domain != expected_domain:
            raise LongMemEvalV2EvidenceError(
                f"{label} {question_id} domain differs from official run"
            )
        query_tokens = _integer(
            query.get("query_tokens"), label=f"{label} {question_id} query_tokens", minimum=0
        )
        latency_ms = _number(
            query.get("query_latency_ms"),
            label=f"{label} {question_id} query_latency_ms",
            minimum=0,
        )
        _assert_close(
            latency_ms,
            float(record["memory_query_duration_seconds"]) * 1000.0,
            label=f"{label} {question_id} query_latency_ms",
        )
        query_failed = _boolean(
            query.get("query_failed"), label=f"{label} {question_id} query_failed"
        )
        unanswered = _boolean(query.get("unanswered"), label=f"{label} {question_id} unanswered")
        expected_unanswered = not record["response_raw"].strip()
        if unanswered is not expected_unanswered:
            raise LongMemEvalV2EvidenceError(
                f"{label} {question_id} unanswered flag differs from official output"
            )
        embedding = _validate_embedding_evidence(
            query.get("embedding"), label=f"{label} {question_id}.embedding"
        )
        operations, trace_sha256 = _validate_operation_trace(
            query.get("operations"),
            tier=tier,
            point_name=point_name,
            question_id=question_id,
            query_tokens=query_tokens,
            query_latency_ms=latency_ms,
            official_record=record,
            seen_invocation_ids=seen_invocation_ids,
            label=f"{label} {question_id}",
            embedding=embedding,
        )
        queries.append(
            QueryEvidence(
                question_id=question_id,
                domain=domain,
                query_tokens=query_tokens,
                query_latency_ms=latency_ms,
                query_failed=query_failed,
                unanswered=unanswered,
                operations=operations,
                embedding=embedding,
                trace_sha256=trace_sha256,
            )
        )
    if not queries:
        raise LongMemEvalV2EvidenceError(f"{label}.queries cannot be empty")
    token_total = sum(query.query_tokens for query in queries)
    if token_total <= 0:
        raise LongMemEvalV2EvidenceError(f"{label} must record a positive query-token total")
    latencies = [query.query_latency_ms for query in queries]
    expected_summary: dict[str, int | float] = {
        "questions": len(queries),
        "query_token_observations": len(queries),
        "query_tokens_total": token_total,
        "query_tokens_mean": token_total / len(queries),
        "query_latency_observations": len(queries),
        "query_latency_total_ms": sum(latencies),
        "query_latency_mean_ms": mean(latencies),
        "query_latency_p95_ms": _percentile_index(latencies, 0.95),
        "query_failures": sum(query.query_failed for query in queries),
        "unanswered_questions": sum(query.unanswered for query in queries),
    }
    summary = _mapping(sidecar_point.get("summary"), label=f"{label}.summary")
    if set(summary) != set(expected_summary):
        raise LongMemEvalV2EvidenceError(
            f"{label}.summary fields differ: expected {sorted(expected_summary)}, "
            f"got {sorted(summary)}"
        )
    _compare_json(summary, expected_summary, label=f"{label}.summary")
    return tuple(queries)


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    package_name: str,
    point_name: str,
    method: str,
    tier: str,
    runs: Mapping[str, RunEvidence],
    label: str,
) -> None:
    expected_scalars = {
        "submission_name": package_name,
        "operating_point_name": point_name,
        "method": method,
        "tier": tier,
    }
    for key, expected in expected_scalars.items():
        if metadata.get(key) != expected:
            raise LongMemEvalV2EvidenceError(
                f"{label}.{key} must be {expected!r}, got {metadata.get(key)!r}"
            )
    _nonempty_string(metadata.get("generated_at_utc"), label=f"{label}.generated_at_utc")
    included = _list(metadata.get("included_run_files"), label=f"{label}.included_run_files")
    expected_included = [
        "aggregated_metrics.json",
        "per_question.jsonl",
        "run_args.json",
        "runtime_inputs/",
    ]
    if included != expected_included:
        raise LongMemEvalV2EvidenceError(f"{label}.included_run_files is not official")
    run_metadata = _mapping(metadata.get("runs"), label=f"{label}.runs")
    if set(run_metadata) != {"web", "enterprise"}:
        raise LongMemEvalV2EvidenceError(f"{label}.runs must contain web and enterprise")
    for domain, run in runs.items():
        item = _mapping(run_metadata[domain], label=f"{label}.runs.{domain}")
        if item.get("domain") != domain or item.get("question_count") != len(run.question_ids):
            raise LongMemEvalV2EvidenceError(
                f"{label}.runs.{domain} question metadata is inconsistent"
            )
        if item.get("model") != EXPECTED_READER_MODEL:
            raise LongMemEvalV2EvidenceError(f"{label}.runs.{domain} must name exact fixed reader")
        if item.get("evaluator_model") != EXPECTED_JUDGE_MODEL:
            raise LongMemEvalV2EvidenceError(f"{label}.runs.{domain} must name exact judge")


def _validate_lafs(
    actual: dict[str, Any], *, tier: str, points: Sequence[OperatingPointEvidence], label: str
) -> float:
    for field in ("reference_lafs", "submission_lafs", "lafs_gain"):
        if field not in actual:
            raise LongMemEvalV2EvidenceError(f"{label} is missing official field {field}")
        _number(actual[field], label=f"{label}.{field}", positive=True)
    expected = _lafs_summary(
        tier,
        [
            _LafsPoint(
                point.name,
                point.accuracy * 100.0,
                point.memory_query_avg_seconds,
            )
            for point in points
        ],
    )
    _compare_json(actual, expected, label=label)
    return float(expected["lafs_gain"])


def _validate_sidecar_package_hashes(
    sidecar: dict[str, Any], *, actual_files: Mapping[str, str], label: str
) -> str:
    package = _mapping(sidecar.get("package"), label=f"{label}.package")
    declared_raw = _mapping(package.get("files_sha256"), label=f"{label}.package.files_sha256")
    declared: dict[str, str] = {}
    for relative, digest in declared_raw.items():
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise LongMemEvalV2EvidenceError(
                f"{label}.package.files_sha256 contains an unsafe path"
            )
        normalized = Path(relative).as_posix()
        if normalized != relative or ".." in Path(relative).parts:
            raise LongMemEvalV2EvidenceError(
                f"{label}.package.files_sha256 contains a noncanonical path {relative!r}"
            )
        declared[relative] = _checked_sha256(
            digest, label=f"{label}.package.files_sha256[{relative!r}]"
        )
    if declared != dict(actual_files):
        missing = sorted(set(actual_files) - set(declared))[:3]
        extra = sorted(set(declared) - set(actual_files))[:3]
        changed = sorted(
            path
            for path in set(actual_files) & set(declared)
            if actual_files[path] != declared[path]
        )[:3]
        raise LongMemEvalV2EvidenceError(
            f"{label} package evidence hashes differ; missing={missing}, "
            f"extra={extra}, changed={changed}"
        )
    tree_digest = _tree_sha256(actual_files)
    if package.get("tree_sha256") != tree_digest:
        raise LongMemEvalV2EvidenceError(f"{label}.package.tree_sha256 is inconsistent")
    return tree_digest


def load_tier_evidence(
    package_dir: Path, sidecar_path: Path, *, expected_tier: str
) -> TierEvidence:
    """Validate and load one official package plus its external Swarm sidecar."""

    if expected_tier not in _REFERENCE_POINTS:
        raise LongMemEvalV2EvidenceError(f"unsupported tier {expected_tier!r}")
    package_dir = package_dir.resolve()
    sidecar_path = sidecar_path.resolve()
    if sidecar_path.is_relative_to(package_dir):
        raise LongMemEvalV2EvidenceError(
            "Swarm evidence sidecar must be external to the official package"
        )
    actual_files = _package_hashes(package_dir)
    sidecar = _read_json(sidecar_path)
    sidecar_label = f"{expected_tier} sidecar"
    if sidecar.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise LongMemEvalV2EvidenceError(
            f"{sidecar_label}.schema_version must be {SIDECAR_SCHEMA_VERSION}"
        )
    if sidecar.get("benchmark_repository_commit") != PINNED_REPOSITORY_COMMIT:
        raise LongMemEvalV2EvidenceError(
            f"{sidecar_label} must pin benchmark commit {PINNED_REPOSITORY_COMMIT}"
        )
    if sidecar.get("tier") != expected_tier:
        raise LongMemEvalV2EvidenceError(f"{sidecar_label}.tier must be {expected_tier!r}")
    if sidecar.get("reader_model") != EXPECTED_READER_MODEL:
        raise LongMemEvalV2EvidenceError(f"{sidecar_label} must name exact fixed reader")
    if sidecar.get("judge_model") != EXPECTED_JUDGE_MODEL:
        raise LongMemEvalV2EvidenceError(f"{sidecar_label} must name exact judge")
    dataset_revision = _nonempty_string(
        sidecar.get("dataset_revision"), label=f"{sidecar_label}.dataset_revision"
    )
    if dataset_revision != dataset_revision.strip():
        raise LongMemEvalV2EvidenceError(
            f"{sidecar_label}.dataset_revision must be a canonical caller-pinned identifier"
        )
    dataset_manifest_sha256 = _checked_sha256(
        sidecar.get("dataset_manifest_sha256"),
        label=f"{sidecar_label}.dataset_manifest_sha256",
    )
    if sidecar.get("dataset_identity_source") != "caller-pinned-run-ledger":
        raise LongMemEvalV2EvidenceError(
            f"{sidecar_label}.dataset_identity_source must be 'caller-pinned-run-ledger'"
        )
    if sidecar.get("iterative_search_read_expand") is not True:
        raise LongMemEvalV2EvidenceError(
            f"{sidecar_label}.iterative_search_read_expand must be true"
        )
    if sidecar.get("semantic_embedding_proof") is not True:
        raise LongMemEvalV2EvidenceError(f"{sidecar_label}.semantic_embedding_proof must be true")
    method = _nonempty_string(sidecar.get("method"), label=f"{sidecar_label}.method")
    package_tree_sha256 = _validate_sidecar_package_hashes(
        sidecar, actual_files=actual_files, label=sidecar_label
    )
    sidecar_points = _sidecar_points(sidecar, label=sidecar_label)

    overview_path = package_dir / "submission_overview.json"
    overview = _read_json(overview_path)
    submission_name = _nonempty_string(
        overview.get("submission_name"), label="submission_overview.submission_name"
    )
    if package_dir.name != submission_name:
        raise LongMemEvalV2EvidenceError(
            "official package directory name must equal submission_name"
        )
    if overview.get("method") != method:
        raise LongMemEvalV2EvidenceError("official package and sidecar methods differ")
    if overview.get("tier") != expected_tier:
        raise LongMemEvalV2EvidenceError(f"official package tier must be {expected_tier!r}")
    _nonempty_string(overview.get("generated_at_utc"), label="submission_overview.generated_at_utc")
    if overview.get("archive_name") != f"{submission_name}.tar.gz":
        raise LongMemEvalV2EvidenceError("submission_overview.archive_name is inconsistent")
    if overview.get("system_description_file") != "SYSTEM_DESCRIPTION.md":
        raise LongMemEvalV2EvidenceError(
            "submission_overview.system_description_file must be SYSTEM_DESCRIPTION.md"
        )
    code_file = _nonempty_string(overview.get("code_file"), label="submission_overview.code_file")
    if not _SAFE_ROOT_FILE_RE.fullmatch(code_file) or code_file in {
        "SYSTEM_DESCRIPTION.md",
        "submission_overview.json",
        "operating_points",
    }:
        raise LongMemEvalV2EvidenceError("submission_overview.code_file is unsafe or reserved")
    for root_file in ("SYSTEM_DESCRIPTION.md", code_file):
        if root_file not in actual_files:
            raise LongMemEvalV2EvidenceError(f"official package is missing root file {root_file}")

    top_points_raw = _list(
        overview.get("operating_points"), label="submission_overview.operating_points"
    )
    if not all(isinstance(point, dict) for point in top_points_raw):
        raise LongMemEvalV2EvidenceError(
            "submission_overview.operating_points must contain objects"
        )
    top_points = _index_unique(
        top_points_raw, key="name", label="submission_overview.operating_points"
    )
    points_dir = package_dir / "operating_points"
    if not points_dir.is_dir() or points_dir.is_symlink():
        raise LongMemEvalV2EvidenceError("official package is missing operating_points")
    directory_points = {
        path.name for path in points_dir.iterdir() if path.is_dir() and not path.is_symlink()
    }
    if set(top_points) != directory_points or set(top_points) != set(sidecar_points):
        raise LongMemEvalV2EvidenceError(
            "operating-point sets differ across overview, package, and sidecar"
        )
    if not top_points:
        raise LongMemEvalV2EvidenceError("official package has no operating points")

    point_evidence: list[OperatingPointEvidence] = []
    seen_invocation_ids: set[str] = set()
    reference_questions: dict[str, str] | None = None
    reference_haystacks: dict[str, str] | None = None
    reference_signatures: tuple[tuple[str, str], ...] | None = None
    for point_name in sorted(top_points):
        point_dir = points_dir / point_name
        metric_path = point_dir / "metric_overview.json"
        metadata_path = point_dir / "operating_point_metadata.json"
        metric = _read_json(metric_path)
        metadata = _read_json(metadata_path)
        runs = {
            domain: _load_run(
                point_dir / domain,
                domain=domain,
                method=method,
                tier=expected_tier,
            )
            for domain in ("web", "enterprise")
        }
        if runs["web"].protocol_sha256 != runs["enterprise"].protocol_sha256:
            raise LongMemEvalV2EvidenceError(
                f"operating point {point_name!r} mixes web and enterprise protocols"
            )
        run_dataset_identities = {
            (run.dataset_revision, run.dataset_manifest_sha256) for run in runs.values()
        }
        if run_dataset_identities != {(dataset_revision, dataset_manifest_sha256)}:
            raise LongMemEvalV2EvidenceError(
                f"operating point {point_name!r} memory configs differ from the "
                "caller-pinned sidecar dataset identity"
            )
        web_ids = set(runs["web"].question_ids)
        enterprise_ids = set(runs["enterprise"].question_ids)
        if web_ids & enterprise_ids:
            raise LongMemEvalV2EvidenceError(
                f"operating point {point_name!r} repeats question IDs across domains"
            )
        question_ids = tuple(sorted(web_ids | enterprise_ids))
        if len(question_ids) != EXPECTED_QUESTIONS:
            raise LongMemEvalV2EvidenceError(
                f"operating point {point_name!r} must cover exactly {EXPECTED_QUESTIONS} "
                f"combined web+enterprise questions, got {len(question_ids)}"
            )
        current_questions = {
            domain: runs[domain].questions_json for domain in ("web", "enterprise")
        }
        current_haystacks = {domain: runs[domain].haystack_json for domain in ("web", "enterprise")}
        signatures = tuple(
            sorted((*runs["web"].question_signatures, *runs["enterprise"].question_signatures))
        )
        if reference_questions is None:
            reference_questions = current_questions
            reference_haystacks = current_haystacks
            reference_signatures = signatures
        elif current_questions != reference_questions or current_haystacks != reference_haystacks:
            raise LongMemEvalV2EvidenceError(
                f"operating point {point_name!r} uses different questions or haystacks"
            )
        elif signatures != reference_signatures:
            raise LongMemEvalV2EvidenceError(
                f"operating point {point_name!r} uses different question content"
            )
        combined_records = (*runs["web"].records, *runs["enterprise"].records)
        expected_metric = _metric_overview(combined_records)
        _validate_metric_overview(metric, expected_metric, label=str(metric_path))
        _validate_metadata(
            metadata,
            package_name=submission_name,
            point_name=point_name,
            method=method,
            tier=expected_tier,
            runs=runs,
            label=str(metadata_path),
        )
        top_point = top_points[point_name]
        expected_top = {
            "name": point_name,
            "metric_overview_file": f"operating_points/{point_name}/metric_overview.json",
            "overall_full_set": expected_metric["overall_full_set"],
            "memory_query_avg_seconds": expected_metric["memory_query_avg_seconds"],
            "lafs_accuracy_percentage_points": expected_metric["overall_full_set"] * 100.0,
            "lafs_latency_seconds": expected_metric["memory_query_avg_seconds"],
        }
        _compare_json(top_point, expected_top, label=f"submission_overview point {point_name}")
        records_by_id = {
            question_id: (domain, record)
            for domain, run in runs.items()
            for question_id, record in run.records_by_id.items()
        }
        queries = _validate_sidecar_queries(
            sidecar_points[point_name],
            tier=expected_tier,
            point_name=point_name,
            records_by_id=records_by_id,
            protocol_sha256=runs["web"].protocol_sha256,
            seen_invocation_ids=seen_invocation_ids,
            label=f"{sidecar_label} operating point {point_name}",
        )
        point_evidence.append(
            OperatingPointEvidence(
                name=point_name,
                questions=len(question_ids),
                accuracy=expected_metric["overall_full_set"],
                memory_query_avg_seconds=expected_metric["memory_query_avg_seconds"],
                protocol_sha256=runs["web"].protocol_sha256,
                queries=queries,
            )
        )
    assert reference_signatures is not None
    question_ids = tuple(question_id for question_id, _ in reference_signatures)
    lafs = _mapping(overview.get("lafs"), label="submission_overview.lafs")
    lafs_gain = _validate_lafs(
        lafs,
        tier=expected_tier,
        points=point_evidence,
        label="submission_overview.lafs",
    )
    return TierEvidence(
        tier=expected_tier,
        submission_name=submission_name,
        method=method,
        dataset_revision=dataset_revision,
        dataset_manifest_sha256=dataset_manifest_sha256,
        package_tree_sha256=package_tree_sha256,
        package_files_sha256=actual_files,
        sidecar_sha256=_sha256_file(sidecar_path),
        question_ids=question_ids,
        question_signatures=reference_signatures,
        question_ids_sha256=_canonical_sha256(question_ids),
        question_content_sha256=_canonical_sha256(reference_signatures),
        lafs_gain=lafs_gain,
        operating_points=tuple(point_evidence),
    )


def _query_summary(queries: Sequence[QueryEvidence]) -> dict[str, Any]:
    if not queries:
        raise LongMemEvalV2EvidenceError("cannot summarize zero query records")
    latencies = [query.query_latency_ms for query in queries]
    token_total = sum(query.query_tokens for query in queries)
    return {
        "latency": {
            "query_mean_ms": mean(latencies),
            "query_p95_ms": _percentile_index(latencies, 0.95),
            "query_total_ms": sum(latencies),
            "observations": len(queries),
        },
        "tokens": {
            "query_mean": token_total / len(queries),
            "query_total": token_total,
            "observations": len(queries),
        },
        "failures": {
            "query_failures": sum(query.query_failed for query in queries),
            "unanswered_questions": sum(query.unanswered for query in queries),
        },
    }


def _agentic_proof_summary(queries: Sequence[QueryEvidence]) -> dict[str, Any]:
    if not queries:
        raise LongMemEvalV2EvidenceError("cannot summarize zero agentic proof traces")
    operations = tuple(operation for query in queries for operation in query.operations)
    trace_sha256s = [query.trace_sha256 for query in queries]
    if len(trace_sha256s) != len(set(trace_sha256s)):
        raise LongMemEvalV2EvidenceError("agentic proof reuses canonical operation-trace digests")
    invocation_ids = [operation.invocation_id for operation in operations]
    if len(invocation_ids) != len(set(invocation_ids)):
        raise LongMemEvalV2EvidenceError("agentic proof reuses invocation IDs")
    successful_recalls = [
        operation
        for operation in operations
        if operation.operation == "recall_memory"
        and operation.success
        and operation.result_memory_ids
    ]
    successful_expands = [
        operation
        for operation in operations
        if operation.operation == "read_expand_memory" and operation.success
    ]
    memory_ids = {
        memory_id
        for operation in operations
        for memory_id in (*operation.seed_memory_ids, *operation.result_memory_ids)
    }
    delivered_tokens = sum(operation.delivered_tokens for operation in operations)
    operation_latency_ms = sum(operation.latency_ms for operation in operations)
    query_count = len(queries)
    return {
        "queries": query_count,
        "search_then_read_expand_queries": query_count,
        "proof_rate": 1.0,
        "all_queries_proved": True,
        "package_bound_trace_queries": query_count,
        "all_traces_package_bound": True,
        "trace_sha256s": len(trace_sha256s),
        "trace_sha256s_unique": True,
        "content_free_trace_queries": query_count,
        "contiguous_sequence_queries": query_count,
        "token_reconciled_queries": query_count,
        "token_reconciliation_exact": True,
        "latency_bounded_queries": query_count,
        "operation_latency_bounded": True,
        "operations": len(operations),
        "successful_recall_operations": len(successful_recalls),
        "successful_read_expand_operations": len(successful_expands),
        "invocation_ids": len(invocation_ids),
        "invocation_ids_unique": True,
        "unique_opaque_memory_ids": len(memory_ids),
        "delivered_tokens": delivered_tokens,
        "operation_latency_ms": operation_latency_ms,
        "max_depth": max(operation.depth for operation in operations),
        "max_seed_memory_ids": max(len(operation.seed_memory_ids) for operation in operations),
        "max_result_memory_ids": max(len(operation.result_memory_ids) for operation in operations),
    }


def _semantic_embedding_proof_summary(
    queries: Sequence[QueryEvidence],
) -> dict[str, Any]:
    if not queries:
        raise LongMemEvalV2EvidenceError("cannot summarize zero semantic embedding proofs")
    embeddings = [query.embedding for query in queries]
    identities = {
        (
            item.retrieval_mode,
            item.provider,
            item.model,
            item.model_revision,
            item.dimensions,
            item.response_model_requirement,
            item.query_instruction_sha256,
        )
        for item in embeddings
    }
    if len(identities) != 1:
        raise LongMemEvalV2EvidenceError(
            "semantic embedding proofs mix provider, model, revision, or protocol identity"
        )
    first = embeddings[0]
    inserted = sum(item.inserted_memories for item in embeddings)
    completed = sum(item.embedding_work_completed for item in embeddings)
    document_inputs = sum(item.document_inputs for item in embeddings)
    document_batches = sum(item.document_batch_calls for item in embeddings)
    document_successes = sum(item.document_successful_http_calls for item in embeddings)
    document_attempts = sum(item.document_http_attempts for item in embeddings)
    query_calls = sum(item.query_calls for item in embeddings)
    query_successes = sum(item.query_successful_http_calls for item in embeddings)
    query_attempts = sum(item.query_http_attempts for item in embeddings)
    query_count = len(queries)
    if not inserted == completed == document_inputs == document_batches == document_successes:
        raise LongMemEvalV2EvidenceError(
            "aggregate document embedding accounting does not reconcile"
        )
    if query_calls != query_count or query_successes != query_count:
        raise LongMemEvalV2EvidenceError(
            "aggregate query embedding accounting does not cover every query"
        )
    return {
        "retrieval_mode": first.retrieval_mode,
        "sota_capable": True,
        "provider": first.provider,
        "model": first.model,
        "model_revision": first.model_revision,
        "dimensions": first.dimensions,
        "response_model_requirement": first.response_model_requirement,
        "query_instruction_sha256": first.query_instruction_sha256,
        "inserted_memories": inserted,
        "embedding_work_completed": completed,
        "inserted_memory_embedding_coverage": 1.0,
        "queries": query_count,
        "query_embedding_call_coverage": 1.0,
        "exact_response_model_verified": True,
        "deterministic_fallback_used": False,
        "call_accounting": {
            "source": "provider-observed",
            "document_inputs": document_inputs,
            "document_batch_calls": document_batches,
            "document_successful_http_calls": document_successes,
            "document_http_attempts": document_attempts,
            "query_calls": query_calls,
            "query_successful_http_calls": query_successes,
            "query_http_attempts": query_attempts,
            "successful_http_calls": document_successes + query_successes,
            "http_attempts": document_attempts + query_attempts,
        },
    }


def _tier_report(tier: TierEvidence) -> dict[str, Any]:
    best = sorted(tier.operating_points, key=lambda point: (-point.accuracy, point.name))[0]
    all_queries = tuple(query for point in tier.operating_points for query in point.queries)
    summary = _query_summary(all_queries)
    return {
        "questions": len(tier.question_ids),
        "accuracy": best.accuracy,
        "accuracy_operating_point": best.name,
        "lafs_gain": tier.lafs_gain,
        "operating_points": {
            point.name: {
                "questions": point.questions,
                "accuracy": point.accuracy,
                "memory_query_avg_seconds": point.memory_query_avg_seconds,
                "protocol_sha256": point.protocol_sha256,
                "query_latency_p95_ms": _percentile_index(
                    [query.query_latency_ms for query in point.queries], 0.95
                ),
                "query_tokens_mean": mean(query.query_tokens for query in point.queries),
                "query_failures": sum(query.query_failed for query in point.queries),
                "unanswered_questions": sum(query.unanswered for query in point.queries),
            }
            for point in tier.operating_points
        },
        "agentic_proof": _agentic_proof_summary(all_queries),
        "semantic_embedding_proof": _semantic_embedding_proof_summary(all_queries),
        **summary,
    }


def build_report(small: TierEvidence, medium: TierEvidence) -> dict[str, Any]:
    """Join fully validated small and medium evidence into the SOTA schema."""

    if small.tier != "small" or medium.tier != "medium":
        raise LongMemEvalV2EvidenceError("build_report requires small then medium evidence")
    if small.method != medium.method:
        raise LongMemEvalV2EvidenceError("small and medium packages use different methods")
    if (
        small.dataset_revision != medium.dataset_revision
        or small.dataset_manifest_sha256 != medium.dataset_manifest_sha256
    ):
        raise LongMemEvalV2EvidenceError(
            "small and medium packages use different caller-pinned dataset identities"
        )
    if small.question_ids != medium.question_ids:
        raise LongMemEvalV2EvidenceError("small and medium packages cover different question IDs")
    if small.question_signatures != medium.question_signatures:
        raise LongMemEvalV2EvidenceError("small and medium packages use different question content")
    small_protocols = {point.name: point.protocol_sha256 for point in small.operating_points}
    medium_protocols = {point.name: point.protocol_sha256 for point in medium.operating_points}
    if small_protocols != medium_protocols:
        raise LongMemEvalV2EvidenceError(
            "small and medium packages use different operating-point protocols"
        )
    all_queries = tuple(
        query
        for tier in (small, medium)
        for point in tier.operating_points
        for query in point.queries
    )
    invocation_ids = [
        operation.invocation_id for query in all_queries for operation in query.operations
    ]
    if len(invocation_ids) != len(set(invocation_ids)):
        raise LongMemEvalV2EvidenceError("small and medium proof traces reuse invocation IDs")
    summary = _query_summary(all_queries)
    return {
        "schema_version": 1,
        "benchmark": {
            "name": "LongMemEval-V2",
            "repository": OFFICIAL_REPOSITORY,
            "repository_commit": PINNED_REPOSITORY_COMMIT,
            "swarm_sidecar_schema_version": SIDECAR_SCHEMA_VERSION,
        },
        "dataset": {
            "questions": len(small.question_ids),
            "domains": ["enterprise", "web"],
            "revision": small.dataset_revision,
            "manifest_sha256": small.dataset_manifest_sha256,
            "identity_source": "caller-pinned-run-ledger",
            "question_ids_sha256": small.question_ids_sha256,
            "question_content_sha256": small.question_content_sha256,
        },
        "evaluation": {
            "method": small.method,
            "fixed_reader": True,
            "reader_model": EXPECTED_READER_MODEL,
            "judge_model": EXPECTED_JUDGE_MODEL,
            "iterative_search_read_expand": True,
            "iterative_search_read_expand_declaration_role": "secondary-metadata",
            "retrieval_control_flow": "fixed_two_stage_recall_then_expand",
            "adaptive_multi_round_retrieval": False,
            "primary_agentic_evidence": "per-query-content-free-operation-traces",
            "semantic_embedding_required": True,
            "primary_embedding_evidence": ("per-query-content-free-provider-and-http-accounting"),
            "operation_trace_binding": (
                f"official-per_question.memory_post_query_metadata.{TRACE_DIGEST_METADATA_KEY}"
            ),
            "protocol_sha256": _canonical_sha256(small_protocols),
        },
        "tiers": {
            "small": _tier_report(small),
            "medium": _tier_report(medium),
        },
        "latency": summary["latency"],
        "tokens": summary["tokens"],
        "semantic_embedding_proof": _semantic_embedding_proof_summary(all_queries),
        "failures": summary["failures"],
        "agentic_proof": _agentic_proof_summary(all_queries),
        "evidence": {
            tier.tier: {
                "submission_name": tier.submission_name,
                "package_tree_sha256": tier.package_tree_sha256,
                "package_files_sha256": dict(tier.package_files_sha256),
                "sidecar_sha256": tier.sidecar_sha256,
            }
            for tier in (small, medium)
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-package", type=Path, required=True)
    parser.add_argument("--small-sidecar", type=Path, required=True)
    parser.add_argument("--medium-package", type=Path, required=True)
    parser.add_argument("--medium-sidecar", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Explicit output path; use benchmarks/sota/longmemeval-v2-report.json only for real evidence",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output artifact",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    package_paths = (args.small_package.resolve(), args.medium_package.resolve())
    if any(output.is_relative_to(package) for package in package_paths):
        print("cannot build LongMemEval-V2 report: output must be outside evidence packages")
        return 2
    if output.exists() and not args.force:
        print(f"cannot build LongMemEval-V2 report: output already exists: {output}")
        return 2
    try:
        small = load_tier_evidence(args.small_package, args.small_sidecar, expected_tier="small")
        medium = load_tier_evidence(
            args.medium_package, args.medium_sidecar, expected_tier="medium"
        )
        report = build_report(small, medium)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except (LongMemEvalV2EvidenceError, OSError) as exc:
        print(f"cannot build LongMemEval-V2 report: {exc}")
        return 2
    print(f"Wrote {output}")
    print(
        "Validated LongMemEval-V2 accuracy: "
        f"small={report['tiers']['small']['accuracy']:.4f}, "
        f"medium={report['tiers']['medium']['accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
