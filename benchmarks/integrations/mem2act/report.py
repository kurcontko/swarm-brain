"""Strict offline compiler for hash-bound Mem2Act prediction evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from .contracts import (
    FailureRecord,
    Mem2ActContractError,
    Mem2ActDataset,
    PredictionRecord,
    ReaderRequest,
    TaskMetrics,
    ToolPrediction,
)
from .dataset import (
    KNOWN_TOOL_NAME_REPAIRS,
    MEM2ACT_REPO_COMMIT,
    OFFICIAL_MEM2ACT_SPEC,
    canonical_json,
    load_mem2act_dataset,
)
from .metrics import aggregate_arm, paired_bootstrap, parse_tool_prediction, score_prediction
from .openai_reader import request_protocol_evidence
from .provenance import (
    PROTOCOL_VERSION,
    RUN_ARTIFACT_TYPE,
    RUN_SCHEMA_VERSION,
    implementation_fingerprint,
    repository_root,
)

NO_MEMORY_ARM = "no_memory"
SWARM_ARM = "swarm"
ORACLE_ARM = "oracle"
REQUIRED_ARMS = (NO_MEMORY_ARM, SWARM_ARM, ORACLE_ARM)
TARGET_TOOL_GIVEN = "target_tool_given"
FULL_CATALOG = "full_catalog"
REQUIRED_CONDITIONS = (TARGET_TOOL_GIVEN, FULL_CATALOG)

_SHA256_LENGTH = 64
_PREDICTION_FIELDS = frozenset(field.name for field in fields(PredictionRecord))
_METRIC_FIELDS = frozenset(field.name for field in fields(TaskMetrics))
_FAILURE_FIELDS = frozenset(field.name for field in fields(FailureRecord))
_CONFIGURATION_FIELDS = frozenset(
    {
        "retrieval_limit",
        "retrieval_token_budget",
        "bootstrap_resamples",
        "bootstrap_seed",
        "bootstrap_confidence",
        "task_limit",
        "expected_reader_model",
        "reader_revision",
    }
)
_RUN_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "protocol_version",
        "created_at_utc",
        "implementation",
        "dataset",
        "tool_catalog",
        "configuration",
        "implementations",
        "memory_bridge_evidence",
        "ingestion",
        "predictions_artifact",
    }
)
_PREDICTIONS_ARTIFACT_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "bytes",
        "rows",
        "preserves_raw_predictions",
        "preserves_reader_contexts",
        "preserves_failures_latency_and_tokens",
    }
)


class Mem2ActReportError(Mem2ActContractError):
    """Raw Mem2Act evidence cannot support a reproducible report."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise Mem2ActReportError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise Mem2ActReportError(f"non-finite JSON number {value!r} is forbidden")


def _strict_json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, Mem2ActReportError) as exc:
        raise Mem2ActReportError(f"{label} is not strict JSON: {exc}") from exc


def _strict_object(raw: bytes, *, label: str) -> dict[str, Any]:
    value = _strict_json(raw, label=label)
    if not isinstance(value, dict):
        raise Mem2ActReportError(f"{label} must contain one JSON object")
    return value


def _required_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Mem2ActReportError(f"{label} must be an object")
    canonical_json(value)
    return value


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Mem2ActReportError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, label=label)


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Mem2ActReportError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Mem2ActReportError(f"{label} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise Mem2ActReportError(f"{label} must be a finite number >= {minimum}")
    return result


def _boolean(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise Mem2ActReportError(f"{label} must be boolean")
    return value


def _sha256(value: Any, *, label: str) -> str:
    text = _required_text(value, label=label)
    if len(text) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise Mem2ActReportError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _string_tuple(
    value: Any, *, label: str, unique: bool = False, nonempty_items: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise Mem2ActReportError(f"{label} must be a list of strings")
    result = tuple(value)
    if nonempty_items and any(not item for item in result):
        raise Mem2ActReportError(f"{label} must contain only non-empty strings")
    if unique and len(result) != len(set(result)):
        raise Mem2ActReportError(f"{label} must contain unique values")
    return result


def _resolve_input(
    path: str | Path,
    *,
    root: Path,
    label: str,
    require_relative: bool,
) -> tuple[Path, str]:
    supplied = Path(path)
    if require_relative and supplied.is_absolute():
        raise Mem2ActReportError(f"{label} must be a repository-local relative path")
    if ".." in supplied.parts:
        raise Mem2ActReportError(f"{label} cannot contain '..'")
    root = root.resolve()
    candidate = supplied if supplied.is_absolute() else root / supplied
    current = Path(candidate.anchor) if candidate.is_absolute() else root
    parts = candidate.parts[1:] if candidate.is_absolute() else candidate.parts
    for part in parts:
        current /= part
        if current.is_symlink():
            raise Mem2ActReportError(f"{label} cannot traverse symbolic links")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Mem2ActReportError(f"{label} is missing") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise Mem2ActReportError(f"{label} must resolve to a regular file inside the artifact root")
    return resolved, resolved.relative_to(root).as_posix()


def _resolve_output(
    path: str | Path,
    *,
    root: Path,
    require_relative: bool,
    overwrite: bool,
) -> Path:
    supplied = Path(path)
    if require_relative and supplied.is_absolute():
        raise Mem2ActReportError("--output must be a repository-local relative path")
    if ".." in supplied.parts:
        raise Mem2ActReportError("--output cannot contain '..'")
    root = root.resolve()
    candidate = supplied if supplied.is_absolute() else root / supplied
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise Mem2ActReportError("--output must remain inside the artifact root") from exc
    if not candidate.resolve().is_relative_to(root):
        raise Mem2ActReportError("--output must remain inside the artifact root")
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise Mem2ActReportError("--output cannot traverse symbolic links")
    if candidate.is_symlink():
        raise Mem2ActReportError("--output cannot be a symbolic link")
    if candidate.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite Mem2Act report: {candidate}")
    return candidate


def _resolve_directory(
    path: str | Path,
    *,
    root: Path,
    label: str,
    require_relative: bool,
) -> Path:
    supplied = Path(path)
    if require_relative and supplied.is_absolute():
        raise Mem2ActReportError(f"{label} must be a repository-local relative path")
    if ".." in supplied.parts:
        raise Mem2ActReportError(f"{label} cannot contain '..'")
    root = root.resolve()
    candidate = supplied if supplied.is_absolute() else root / supplied
    if not supplied.is_absolute():
        current = root
        for part in supplied.parts:
            current /= part
            if current.is_symlink():
                raise Mem2ActReportError(f"{label} cannot traverse symbolic links")
    elif candidate.is_symlink():
        raise Mem2ActReportError(f"{label} cannot be a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Mem2ActReportError(f"{label} is missing") from exc
    if not resolved.is_dir():
        raise Mem2ActReportError(f"{label} must resolve to a directory")
    if require_relative and not resolved.is_relative_to(root):
        raise Mem2ActReportError(f"{label} must remain inside the artifact root")
    return resolved


def _strict_jsonl(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise Mem2ActReportError(f"{label} is not UTF-8") from exc
    lines = text.splitlines()
    if not lines:
        raise Mem2ActReportError(f"{label} cannot be empty")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise Mem2ActReportError(f"{label}:{line_number} cannot be blank")
        value = _strict_json(line.encode("utf-8"), label=f"{label}:{line_number}")
        if not isinstance(value, dict):
            raise Mem2ActReportError(f"{label}:{line_number} must be an object")
        rows.append(value)
    return rows


def _prediction_record(payload: dict[str, Any], *, index: int) -> PredictionRecord:
    label = f"prediction[{index}]"
    if set(payload) != _PREDICTION_FIELDS:
        raise Mem2ActReportError(f"{label} fields differ from PredictionRecord schema")
    qa_id = _required_text(payload["qa_id"], label=f"{label}.qa_id")
    condition = _required_text(payload["condition"], label=f"{label}.condition")
    arm = _required_text(payload["arm"], label=f"{label}.arm")
    query = _required_text(payload["query"], label=f"{label}.query")
    complexity = _required_text(payload["complexity_level"], label=f"{label}.complexity_level")
    contexts = _string_tuple(payload["memory_contexts"], label=f"{label}.memory_contexts")
    memory_ids = _string_tuple(
        payload["retrieved_memory_ids"],
        label=f"{label}.retrieved_memory_ids",
        unique=True,
        nonempty_items=True,
    )
    raw_scores = payload["retrieved_scores"]
    if not isinstance(raw_scores, list):
        raise Mem2ActReportError(f"{label}.retrieved_scores must be a list")
    scores = tuple(_number(value, label=f"{label}.retrieved_scores[]") for value in raw_scores)
    if any(value > 1.0 for value in scores):
        raise Mem2ActReportError(f"{label}.retrieved_scores values must be <= 1")
    raw_reasons = payload["retrieval_reasons"]
    if not isinstance(raw_reasons, list):
        raise Mem2ActReportError(f"{label}.retrieval_reasons must be a list")
    reasons = tuple(
        _string_tuple(value, label=f"{label}.retrieval_reasons[]") for value in raw_reasons
    )
    if not len(memory_ids) == len(scores) == len(reasons):
        raise Mem2ActReportError(f"{label} retrieval arrays must have equal lengths")
    total_candidates = _integer(
        payload["retrieval_total_candidates"], label=f"{label}.retrieval_total_candidates"
    )
    if total_candidates < len(memory_ids):
        raise Mem2ActReportError(
            f"{label}.retrieval_total_candidates is below the returned-memory count"
        )
    truncated = _boolean(payload["retrieval_truncated"], label=f"{label}.retrieval_truncated")
    retrieval_latency = _number(
        payload["retrieval_latency_ms"], label=f"{label}.retrieval_latency_ms"
    )
    reader_wall = _number(
        payload["reader_wall_latency_ms"], label=f"{label}.reader_wall_latency_ms"
    )
    reader_reported_raw = payload["reader_reported_latency_ms"]
    reader_reported = (
        None
        if reader_reported_raw is None
        else _number(reader_reported_raw, label=f"{label}.reader_reported_latency_ms")
    )
    total_latency = _number(payload["total_latency_ms"], label=f"{label}.total_latency_ms")
    if total_latency + 1e-9 < retrieval_latency + reader_wall:
        raise Mem2ActReportError(f"{label}.total_latency_ms is below component latency")
    prompt_tokens = _integer(payload["prompt_tokens"], label=f"{label}.prompt_tokens")
    completion_tokens = _integer(payload["completion_tokens"], label=f"{label}.completion_tokens")
    reader_model = _optional_text(payload["reader_model"], label=f"{label}.reader_model")
    reader_metadata = _required_object(payload["reader_metadata"], label=f"{label}.reader_metadata")
    raw_prediction = payload["raw_prediction"]
    if raw_prediction is not None and not isinstance(raw_prediction, str):
        raise Mem2ActReportError(f"{label}.raw_prediction must be a string or null")
    parsed_payload = payload["parsed_prediction"]
    parsed: ToolPrediction | None = None
    if parsed_payload is not None:
        parsed_object = _required_object(parsed_payload, label=f"{label}.parsed_prediction")
        if set(parsed_object) != {"name", "arguments"}:
            raise Mem2ActReportError(f"{label}.parsed_prediction fields differ from schema")
        parsed = ToolPrediction(
            name=_required_text(parsed_object["name"], label=f"{label}.parsed_prediction.name"),
            arguments=_required_object(
                parsed_object["arguments"], label=f"{label}.parsed_prediction.arguments"
            ),
        )
    gold_tool = _required_text(payload["gold_tool_name"], label=f"{label}.gold_tool_name")
    gold_arguments = _required_object(payload["gold_arguments"], label=f"{label}.gold_arguments")
    metric_payload = _required_object(payload["metrics"], label=f"{label}.metrics")
    if set(metric_payload) != _METRIC_FIELDS:
        raise Mem2ActReportError(f"{label}.metrics fields differ from TaskMetrics schema")
    failure_payload = payload["failure"]
    failure: FailureRecord | None = None
    if failure_payload is not None:
        failure_object = _required_object(failure_payload, label=f"{label}.failure")
        if set(failure_object) != _FAILURE_FIELDS:
            raise Mem2ActReportError(f"{label}.failure fields differ from FailureRecord schema")
        message = failure_object["message"]
        if not isinstance(message, str):
            raise Mem2ActReportError(f"{label}.failure.message must be a string")
        failure = FailureRecord(
            stage=_required_text(failure_object["stage"], label=f"{label}.failure.stage"),
            error_type=_required_text(
                failure_object["error_type"], label=f"{label}.failure.error_type"
            ),
            message=message,
        )

    reparsed: ToolPrediction | None = None
    parse_error: Exception | None = None
    if raw_prediction is not None:
        try:
            reparsed = parse_tool_prediction(raw_prediction)
        except Exception as exc:  # the typed failure is validated immediately below
            parse_error = exc
    if raw_prediction is None:
        if (
            parsed is not None
            or failure is None
            or failure.stage
            not in {
                "memory_retrieval",
                "reader_call",
            }
        ):
            raise Mem2ActReportError(f"{label} has inconsistent missing-reader evidence")
        if failure.stage == "memory_retrieval" and arm != SWARM_ARM:
            raise Mem2ActReportError(f"{label} assigns retrieval failure to a non-swarm arm")
        if (
            reader_model is not None
            or reader_reported is not None
            or reader_metadata
            or prompt_tokens
            or completion_tokens
        ):
            raise Mem2ActReportError(f"{label} missing-reader evidence reports reader output")
    elif parse_error is not None:
        if parsed is not None or failure is None or failure.stage != "prediction_parse":
            raise Mem2ActReportError(f"{label} has inconsistent prediction-parse evidence")
        if reader_model is None:
            raise Mem2ActReportError(f"{label} parsed reader failure is missing reader_model")
    else:
        if (
            failure is not None
            or parsed is None
            or canonical_json(asdict(parsed)) != canonical_json(asdict(reparsed))
        ):
            raise Mem2ActReportError(f"{label}.parsed_prediction differs from raw_prediction")
        if reader_model is None:
            raise Mem2ActReportError(f"{label} successful prediction is missing reader_model")

    recomputed = score_prediction(
        reparsed,
        gold_tool_name=gold_tool,
        gold_arguments=gold_arguments,
    )
    if canonical_json(metric_payload) != canonical_json(asdict(recomputed)):
        raise Mem2ActReportError(f"{label}.metrics do not recompute from raw prediction and gold")
    catalog_digest = _sha256(payload["tool_catalog_sha256"], label=f"{label}.tool_catalog_sha256")
    return PredictionRecord(
        qa_id=qa_id,
        condition=condition,
        arm=arm,
        query=query,
        complexity_level=complexity,
        memory_contexts=contexts,
        retrieved_memory_ids=memory_ids,
        retrieved_scores=scores,
        retrieval_reasons=reasons,
        retrieval_total_candidates=total_candidates,
        retrieval_truncated=truncated,
        retrieval_latency_ms=retrieval_latency,
        reader_wall_latency_ms=reader_wall,
        reader_reported_latency_ms=reader_reported,
        total_latency_ms=total_latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reader_model=reader_model,
        reader_metadata=reader_metadata,
        raw_prediction=raw_prediction,
        parsed_prediction=reparsed,
        gold_tool_name=gold_tool,
        gold_arguments=gold_arguments,
        metrics=recomputed,
        failure=failure,
        tool_catalog_sha256=catalog_digest,
    )


def _validate_dataset(value: Any) -> tuple[dict[str, Any], bool]:
    dataset = _required_object(value, label="run.dataset")
    expected = {
        "repo_commit",
        "files_sha256",
        "task_count",
        "session_count",
        "unresolved_source_ids",
        "known_data_repairs",
    }
    if set(dataset) != expected:
        raise Mem2ActReportError("run.dataset fields differ from DatasetFingerprint schema")
    _required_text(dataset["repo_commit"], label="run.dataset.repo_commit")
    files_payload = _required_object(dataset["files_sha256"], label="run.dataset.files_sha256")
    files: dict[str, str] = {}
    for relative, digest in files_payload.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
        ):
            raise Mem2ActReportError("run.dataset.files_sha256 contains an unsafe path")
        files[relative] = _sha256(digest, label=f"run.dataset.files_sha256[{relative!r}]")
    task_count = _integer(dataset["task_count"], label="run.dataset.task_count", minimum=1)
    session_count = _integer(dataset["session_count"], label="run.dataset.session_count", minimum=1)
    unresolved = _string_tuple(
        dataset["unresolved_source_ids"],
        label="run.dataset.unresolved_source_ids",
        unique=True,
        nonempty_items=True,
    )
    repairs = _string_tuple(
        dataset["known_data_repairs"],
        label="run.dataset.known_data_repairs",
        unique=True,
        nonempty_items=True,
    )
    normalized = {
        "repo_commit": dataset["repo_commit"],
        "files_sha256": dict(sorted(files.items())),
        "task_count": task_count,
        "session_count": session_count,
        "unresolved_source_ids": list(unresolved),
        "known_data_repairs": list(repairs),
    }
    official_repairs = tuple(
        f"{qa_id}:tool_call.name={raw!r}->{replacement}"
        for qa_id, (raw, replacement) in sorted(KNOWN_TOOL_NAME_REPAIRS.items())
    )
    official = (
        normalized["repo_commit"] == MEM2ACT_REPO_COMMIT
        and normalized["files_sha256"] == OFFICIAL_MEM2ACT_SPEC.files_sha256
        and task_count == OFFICIAL_MEM2ACT_SPEC.task_count
        and session_count == OFFICIAL_MEM2ACT_SPEC.session_count
        and unresolved == tuple(sorted(OFFICIAL_MEM2ACT_SPEC.allowed_unresolved_source_ids))
        and repairs == official_repairs
    )
    return normalized, official


def _validate_configuration(value: Any) -> dict[str, Any]:
    config = _required_object(value, label="run.configuration")
    if set(config) != _CONFIGURATION_FIELDS:
        raise Mem2ActReportError("run.configuration fields differ from BenchmarkConfig schema")
    retrieval_limit = _integer(
        config["retrieval_limit"], label="run.configuration.retrieval_limit", minimum=1
    )
    if retrieval_limit > 100:
        raise Mem2ActReportError("run.configuration.retrieval_limit must be <= 100")
    token_budget = config["retrieval_token_budget"]
    if token_budget is not None:
        token_budget = _integer(
            token_budget, label="run.configuration.retrieval_token_budget", minimum=1
        )
    resamples = _integer(
        config["bootstrap_resamples"],
        label="run.configuration.bootstrap_resamples",
        minimum=1,
    )
    seed = _integer(config["bootstrap_seed"], label="run.configuration.bootstrap_seed")
    confidence = _number(
        config["bootstrap_confidence"], label="run.configuration.bootstrap_confidence"
    )
    if not 0.0 < confidence < 1.0:
        raise Mem2ActReportError("run.configuration.bootstrap_confidence must be in (0, 1)")
    task_limit = config["task_limit"]
    if task_limit is not None:
        task_limit = _integer(task_limit, label="run.configuration.task_limit", minimum=1)
    expected_model = _optional_text(
        config["expected_reader_model"], label="run.configuration.expected_reader_model"
    )
    revision = _optional_text(config["reader_revision"], label="run.configuration.reader_revision")
    return {
        "retrieval_limit": retrieval_limit,
        "retrieval_token_budget": token_budget,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "bootstrap_confidence": confidence,
        "task_limit": task_limit,
        "expected_reader_model": expected_model,
        "reader_revision": revision,
    }


def _validate_run(payload: dict[str, Any], *, code_root: Path) -> dict[str, Any]:
    if set(payload) != _RUN_FIELDS:
        raise Mem2ActReportError("run artifact fields differ from schema")
    if payload["schema_version"] != RUN_SCHEMA_VERSION:
        raise Mem2ActReportError(f"run.schema_version must be {RUN_SCHEMA_VERSION}")
    if payload["artifact_type"] != RUN_ARTIFACT_TYPE:
        raise Mem2ActReportError(f"run.artifact_type must be {RUN_ARTIFACT_TYPE!r}")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise Mem2ActReportError(f"run.protocol_version must be {PROTOCOL_VERSION!r}")
    created_at = _required_text(payload["created_at_utc"], label="run.created_at_utc")
    try:
        parsed_time = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise Mem2ActReportError("run.created_at_utc must be ISO-8601") from exc
    if parsed_time.tzinfo is None:
        raise Mem2ActReportError("run.created_at_utc must include a timezone")
    try:
        current_implementation = implementation_fingerprint(repo_root=code_root)
    except Mem2ActContractError as exc:
        raise Mem2ActReportError(
            "current Mem2Act implementation tree is incomplete or unsafe"
        ) from exc
    if payload["implementation"] != current_implementation:
        raise Mem2ActReportError("run implementation fingerprint does not match current tree")
    dataset, official_dataset = _validate_dataset(payload["dataset"])
    tool_catalog = _required_object(payload["tool_catalog"], label="run.tool_catalog")
    if set(tool_catalog) != {"entries", "sha256"}:
        raise Mem2ActReportError("run.tool_catalog fields differ from schema")
    tool_catalog = {
        "entries": _integer(tool_catalog["entries"], label="run.tool_catalog.entries", minimum=1),
        "sha256": _sha256(tool_catalog["sha256"], label="run.tool_catalog.sha256"),
    }
    config = _validate_configuration(payload["configuration"])
    implementations = _required_object(payload["implementations"], label="run.implementations")
    if set(implementations) != {"memory_bridge", "reader"}:
        raise Mem2ActReportError("run.implementations fields differ from schema")
    implementations = {
        name: _required_text(implementations[name], label=f"run.implementations.{name}")
        for name in ("memory_bridge", "reader")
    }
    bridge = _required_object(payload["memory_bridge_evidence"], label="run.memory_bridge_evidence")
    ingestion = _required_object(payload["ingestion"], label="run.ingestion")
    if set(ingestion) != {"memory_count", "latency_ms", "metadata"}:
        raise Mem2ActReportError("run.ingestion fields differ from IngestionResult schema")
    ingestion = {
        "memory_count": _integer(
            ingestion["memory_count"], label="run.ingestion.memory_count", minimum=1
        ),
        "latency_ms": _number(ingestion["latency_ms"], label="run.ingestion.latency_ms"),
        "metadata": _required_object(ingestion["metadata"], label="run.ingestion.metadata"),
    }
    if ingestion["memory_count"] != dataset["session_count"]:
        raise Mem2ActReportError("run ingestion memory count differs from dataset sessions")
    artifact = _required_object(payload["predictions_artifact"], label="run.predictions_artifact")
    if set(artifact) != _PREDICTIONS_ARTIFACT_FIELDS:
        raise Mem2ActReportError("run.predictions_artifact fields differ from schema")
    for flag in (
        "preserves_raw_predictions",
        "preserves_reader_contexts",
        "preserves_failures_latency_and_tokens",
    ):
        if _boolean(artifact[flag], label=f"run.predictions_artifact.{flag}") is not True:
            raise Mem2ActReportError(f"run.predictions_artifact.{flag} must be true")
    artifact = {
        **artifact,
        "path": _required_text(artifact["path"], label="run.predictions_artifact.path"),
        "sha256": _sha256(artifact["sha256"], label="run.predictions_artifact.sha256"),
        "bytes": _integer(artifact["bytes"], label="run.predictions_artifact.bytes", minimum=1),
        "rows": _integer(artifact["rows"], label="run.predictions_artifact.rows", minimum=1),
    }
    return {
        "created_at_utc": created_at,
        "implementation": current_implementation,
        "dataset": dataset,
        "official_dataset": official_dataset,
        "tool_catalog": tool_catalog,
        "configuration": config,
        "implementations": implementations,
        "memory_bridge_evidence": bridge,
        "ingestion": ingestion,
        "predictions_artifact": artifact,
    }


def _validate_coverage(
    records: tuple[PredictionRecord, ...], *, run: dict[str, Any]
) -> tuple[str, ...]:
    by_task: dict[str, list[PredictionRecord]] = {}
    seen: set[tuple[str, str, str]] = set()
    expected_cells = {
        (condition, arm) for condition in REQUIRED_CONDITIONS for arm in REQUIRED_ARMS
    }
    for record in records:
        if record.condition not in REQUIRED_CONDITIONS or record.arm not in REQUIRED_ARMS:
            raise Mem2ActReportError("prediction uses an unsupported condition or arm")
        if record.arm == SWARM_ARM and len(record.memory_contexts) != len(
            record.retrieved_memory_ids
        ):
            raise Mem2ActReportError(
                "swarm prediction contexts must correspond one-to-one with retrieved memories"
            )
        if len(record.retrieved_memory_ids) > run["configuration"]["retrieval_limit"]:
            raise Mem2ActReportError("prediction exceeds the configured retrieval limit")
        key = (record.qa_id, record.condition, record.arm)
        if key in seen:
            raise Mem2ActReportError(f"duplicate prediction cell {key!r}")
        seen.add(key)
        by_task.setdefault(record.qa_id, []).append(record)
    for qa_id, rows in by_task.items():
        cells = {(row.condition, row.arm) for row in rows}
        if cells != expected_cells:
            raise Mem2ActReportError(f"task {qa_id!r} does not contain all six paired cells")
        signatures = {
            canonical_json(
                {
                    "query": row.query,
                    "complexity_level": row.complexity_level,
                    "gold_tool_name": row.gold_tool_name,
                    "gold_arguments": row.gold_arguments,
                }
            )
            for row in rows
        }
        if len(signatures) != 1:
            raise Mem2ActReportError(f"task {qa_id!r} changes query or gold labels across cells")
        for condition in REQUIRED_CONDITIONS:
            condition_rows = [row for row in rows if row.condition == condition]
            if len({row.tool_catalog_sha256 for row in condition_rows}) != 1:
                raise Mem2ActReportError(f"task {qa_id!r} changes its catalog across arms")
            if condition == FULL_CATALOG and (
                condition_rows[0].tool_catalog_sha256 != run["tool_catalog"]["sha256"]
            ):
                raise Mem2ActReportError(f"task {qa_id!r} full catalog digest is not global")
        for arm in REQUIRED_ARMS:
            arm_rows = [row for row in rows if row.arm == arm]
            retrieval_signatures = {
                canonical_json(
                    {
                        "contexts": row.memory_contexts,
                        "ids": row.retrieved_memory_ids,
                        "scores": row.retrieved_scores,
                        "reasons": row.retrieval_reasons,
                        "total_candidates": row.retrieval_total_candidates,
                        "truncated": row.retrieval_truncated,
                        "latency_ms": row.retrieval_latency_ms,
                    }
                )
                for row in arm_rows
            }
            if len(retrieval_signatures) != 1:
                raise Mem2ActReportError(
                    f"task {qa_id!r} does not reuse one frozen {arm!r} context"
                )
            if arm in {NO_MEMORY_ARM, ORACLE_ARM} and any(
                (
                    arm_rows[0].retrieved_memory_ids,
                    arm_rows[0].retrieved_scores,
                    arm_rows[0].retrieval_reasons,
                    arm_rows[0].retrieval_total_candidates,
                    arm_rows[0].retrieval_truncated,
                    arm_rows[0].retrieval_latency_ms,
                )
            ):
                raise Mem2ActReportError(f"task {qa_id!r} {arm!r} arm claims retrieval")
            if arm == NO_MEMORY_ARM and arm_rows[0].memory_contexts:
                raise Mem2ActReportError(f"task {qa_id!r} no-memory arm contains context")
    task_ids = tuple(sorted(by_task))
    expected_tasks = run["dataset"]["task_count"]
    task_limit = run["configuration"]["task_limit"]
    if task_limit is not None:
        expected_tasks = min(expected_tasks, task_limit)
    if len(task_ids) != expected_tasks or len(records) != expected_tasks * len(expected_cells):
        raise Mem2ActReportError("prediction task/row coverage differs from run configuration")
    expected_model = run["configuration"]["expected_reader_model"]
    observed_models = {record.reader_model for record in records if record.reader_model is not None}
    if expected_model is not None and observed_models - {expected_model}:
        raise Mem2ActReportError("prediction reader model differs from pinned configuration")
    return task_ids


def _catalog_for_task(
    dataset: Mem2ActDataset,
    *,
    qa_id: str,
    condition: str,
) -> tuple[dict[str, Any], ...]:
    if condition == FULL_CATALOG:
        return dataset.tool_catalog
    if condition != TARGET_TOOL_GIVEN:
        raise Mem2ActReportError(f"prediction condition {condition!r} is unsupported")
    task = next((item for item in dataset.tasks if item.qa_id == qa_id), None)
    if task is None:
        raise Mem2ActReportError(f"prediction task {qa_id!r} is absent from the pinned dataset")
    target = canonical_json(task.target_tool_schema)
    matches = tuple(
        entry for entry in dataset.tool_catalog if canonical_json(entry.get("schema")) == target
    )
    if len(matches) != 1:
        raise Mem2ActReportError(
            f"pinned task {qa_id!r} does not resolve to one target catalog entry"
        )
    return matches


def _catalog_sha256(catalog: tuple[dict[str, Any], ...]) -> str:
    return hashlib.sha256(canonical_json(catalog).encode("utf-8")).hexdigest()


def _validate_official_dataset_evidence(
    records: tuple[PredictionRecord, ...],
    *,
    run: dict[str, Any],
    dataset: Mem2ActDataset,
) -> bool:
    if canonical_json(asdict(dataset.fingerprint)) != canonical_json(run["dataset"]):
        raise Mem2ActReportError("run dataset identity differs from the pinned raw dataset")
    expected_catalog = {
        "entries": len(dataset.tool_catalog),
        "sha256": dataset.tool_catalog_sha256,
    }
    if run["tool_catalog"] != expected_catalog:
        raise Mem2ActReportError("run tool catalog differs from the pinned raw dataset")

    selected_tasks = dataset.tasks
    task_limit = run["configuration"]["task_limit"]
    if task_limit is not None:
        selected_tasks = selected_tasks[:task_limit]
    tasks = {task.qa_id: task for task in selected_tasks}
    if set(tasks) != {record.qa_id for record in records}:
        raise Mem2ActReportError("prediction task IDs differ from the pinned raw dataset")
    canonical_reader = (
        run["implementations"]["reader"]
        == "benchmarks.integrations.mem2act.openai_reader.OpenAICompatibleToolSelectionReader"
    )
    reader_protocol_verified = canonical_reader
    for record in records:
        task = tasks[record.qa_id]
        expected_task = {
            "qa_id": task.qa_id,
            "query": task.query,
            "complexity_level": task.complexity_level,
            "gold_tool_name": task.gold_tool_name,
            "gold_arguments": task.gold_arguments,
        }
        observed_task = {
            "qa_id": record.qa_id,
            "query": record.query,
            "complexity_level": record.complexity_level,
            "gold_tool_name": record.gold_tool_name,
            "gold_arguments": record.gold_arguments,
        }
        if canonical_json(observed_task) != canonical_json(expected_task):
            raise Mem2ActReportError(
                f"prediction task {record.qa_id!r} differs from the pinned raw dataset"
            )
        catalog = _catalog_for_task(
            dataset,
            qa_id=record.qa_id,
            condition=record.condition,
        )
        if record.tool_catalog_sha256 != _catalog_sha256(catalog):
            raise Mem2ActReportError(
                f"prediction task {record.qa_id!r} catalog differs from the pinned raw dataset"
            )
        if record.arm == ORACLE_ARM:
            expected_contexts = tuple(memory.render() for memory in task.oracle_memories)
            if record.memory_contexts != expected_contexts:
                raise Mem2ActReportError(
                    f"prediction task {record.qa_id!r} oracle context differs from the dataset"
                )
        if record.raw_prediction is None:
            reader_protocol_verified = False
            continue
        request = ReaderRequest(
            condition=record.condition,
            query=record.query,
            memory_contexts=record.memory_contexts,
            tool_catalog=catalog,
        )
        expected_evidence = request_protocol_evidence(request)
        if any(
            record.reader_metadata.get(key) != value for key, value in expected_evidence.items()
        ):
            reader_protocol_verified = False
        if record.reader_metadata.get("revision") != run["configuration"]["reader_revision"]:
            reader_protocol_verified = False
    return reader_protocol_verified


def _load_official_dataset_evidence(
    dataset_dir: str | Path,
    *,
    artifact_root: Path,
    enforce_repository_local: bool,
) -> Mem2ActDataset:
    root = _resolve_directory(
        dataset_dir,
        root=artifact_root,
        label="--dataset-dir",
        require_relative=enforce_repository_local,
    )
    for relative in OFFICIAL_MEM2ACT_SPEC.files_sha256:
        _resolve_input(
            relative,
            root=root,
            label=f"--dataset-dir/{relative}",
            require_relative=True,
        )
    try:
        return load_mem2act_dataset(root, spec=OFFICIAL_MEM2ACT_SPEC, verify_git=False)
    except Mem2ActContractError as exc:
        raise Mem2ActReportError(f"pinned raw Mem2Act dataset is invalid: {exc}") from exc


def _gate_metrics(
    parameter_summary: dict[str, Any], strict_summary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "parameter_condition": TARGET_TOOL_GIVEN,
        "parameter_f1": parameter_summary["micro_parameter_f1"],
        "parameter_precision": parameter_summary["micro_parameter_precision"],
        "parameter_recall": parameter_summary["micro_parameter_recall"],
        "slot_accuracy": parameter_summary["slot_accuracy"],
        "tool_selection_condition": FULL_CATALOG,
        "exact_tool_and_arguments": strict_summary["exact_tool_and_arguments"],
        "tool_accuracy": strict_summary["tool_accuracy"],
        "failures": {
            TARGET_TOOL_GIVEN: parameter_summary["failure_count"],
            FULL_CATALOG: strict_summary["failure_count"],
        },
        "tokens": {
            TARGET_TOOL_GIVEN: parameter_summary["tokens"],
            FULL_CATALOG: strict_summary["tokens"],
        },
        "latency_ms": {
            TARGET_TOOL_GIVEN: parameter_summary["latency_ms"],
            FULL_CATALOG: strict_summary["latency_ms"],
        },
    }


def _build_report(
    records: tuple[PredictionRecord, ...],
    *,
    run: dict[str, Any],
    run_identity: dict[str, Any],
    predictions_identity: dict[str, Any],
    official_dataset_verified: bool,
    canonical_reader_protocol_verified: bool,
) -> dict[str, Any]:
    task_ids = _validate_coverage(records, run=run)
    config = run["configuration"]
    condition_reports: dict[str, Any] = {}
    for condition in REQUIRED_CONDITIONS:
        condition_records = tuple(record for record in records if record.condition == condition)
        by_arm = {
            arm: tuple(record for record in condition_records if record.arm == arm)
            for arm in REQUIRED_ARMS
        }
        condition_reports[condition] = {
            "paper_parameter_grounding_comparable": condition == TARGET_TOOL_GIVEN,
            "candidate_tool_count": (
                1 if condition == TARGET_TOOL_GIVEN else run["tool_catalog"]["entries"]
            ),
            "arms": {arm: aggregate_arm(by_arm[arm]) for arm in REQUIRED_ARMS},
            "paired_bootstrap": paired_bootstrap(
                condition_records,
                arm_pairs=((SWARM_ARM, NO_MEMORY_ARM), (ORACLE_ARM, SWARM_ARM)),
                resamples=config["bootstrap_resamples"],
                seed=config["bootstrap_seed"],
                confidence=config["bootstrap_confidence"],
            ),
        }
    primary = condition_reports[TARGET_TOOL_GIVEN]
    strict = condition_reports[FULL_CATALOG]
    arms = primary["arms"]
    bootstrap = primary["paired_bootstrap"]
    memory_delta = bootstrap["pairs"]["swarm-minus-no_memory"]["parameter_f1"]
    failures = sum(record.failure is not None for record in records)
    observed_models = sorted(
        {record.reader_model for record in records if record.reader_model is not None}
    )
    fixed_reader = len(observed_models) == 1
    reader_pinned = (
        config["expected_reader_model"] is not None
        and fixed_reader
        and observed_models[0] == config["expected_reader_model"]
    )
    complete = (
        len(task_ids) == 400
        and run["dataset"]["task_count"] == 400
        and official_dataset_verified
        and canonical_reader_protocol_verified
        and failures == 0
        and reader_pinned
        and config["reader_revision"] is not None
        and config["task_limit"] is None
    )
    return {
        "schema_version": 1,
        "benchmark": "Mem2ActBench",
        "generated_at": run["created_at_utc"],
        "claim_status": "measurement artifact only; no comparative or SOTA claim",
        "protocol": {
            "arms": list(REQUIRED_ARMS),
            "conditions": list(REQUIRED_CONDITIONS),
            "normal_memory_input": "all pinned public conversation sessions",
            "normal_retrieval_input": "query text only",
            "target_tool_given": (
                "one published generic target schema/name; excludes target arguments, "
                "grounding labels, and evidence"
            ),
            "full_catalog": "same complete deduplicated catalog for every task and arm",
            "oracle_input": "published evolution-chain evidence; excludes target call labels",
            "parameter_matching": "top-level slot, type-sensitive exact JSON value",
            "wrong_tool_parameter_credit": "zero",
            "bootstrap_unit": "question",
            "official_dataset_verified": official_dataset_verified,
            "canonical_reader_protocol_verified": canonical_reader_protocol_verified,
            "complete_400_task_protocol": complete,
        },
        "evaluation": {
            "oracle_arm": True,
            "no_memory_arm": True,
            "paired": True,
            "primary_parameter_condition": TARGET_TOOL_GIVEN,
            "strict_tool_selection_condition": FULL_CATALOG,
            "reader_model": observed_models[0] if fixed_reader else None,
            "reader_revision": config["reader_revision"],
            "reader_model_pinned": reader_pinned,
            "fixed_reader_model": fixed_reader,
            "official_dataset_verified": official_dataset_verified,
            "canonical_reader_protocol_verified": canonical_reader_protocol_verified,
            "total_failures": failures,
            "complete_400_task_protocol": complete,
        },
        "scoring": {
            "official_evaluator_released": False,
            "implementation": "strict reimplementation from paper section 4.1 text",
            "parameter_unit": "top-level argument slot",
            "value_match": "type-sensitive exact JSON",
            "aggregation": "micro precision/recall/F1 with macro diagnostics",
            "warning": (
                "These metrics are not outputs of an upstream official scorer; "
                "comparability depends on the disclosed target-tool-given condition."
            ),
        },
        "paper_references": {
            "table_4_hybrid_at_5_parameter_f1": {
                "value": 0.307,
                "role": "passive-retrieval ablation baseline; not the SOTA frontier",
            },
            "table_3_a_mem_qwen2_5_72b_parameter_f1": {
                "value": 0.3593,
                "role": "highest reported main-table parameter F1",
                "reader_model": "Qwen2.5-72B-Instruct",
            },
        },
        "dataset": run["dataset"],
        "tool_catalog": run["tool_catalog"],
        "configuration": config,
        "implementations": run["implementations"],
        "memory_bridge_evidence": run["memory_bridge_evidence"],
        "ingestion": run["ingestion"],
        "evaluated_task_count": len(task_ids),
        "prediction_count": len(records),
        "conditions": condition_reports,
        "arms": arms,
        "memory": _gate_metrics(arms[SWARM_ARM], strict["arms"][SWARM_ARM]),
        "no_memory": _gate_metrics(arms[NO_MEMORY_ARM], strict["arms"][NO_MEMORY_ARM]),
        "oracle": _gate_metrics(arms[ORACLE_ARM], strict["arms"][ORACLE_ARM]),
        "comparison": {
            "memory_vs_no_memory_ci95": {
                "metric": "micro_parameter_f1",
                "delta": memory_delta["delta"],
                "lower": memory_delta["ci_low"],
                "upper": memory_delta["ci_high"],
            }
        },
        "paired_bootstrap": bootstrap,
        "reader_models": observed_models,
        "predictions_artifact": predictions_identity,
        "run_artifact": run_identity,
        "provenance": {
            "compiled_offline": True,
            "raw_predictions_reparsed": True,
            "stored_metrics_recomputed": True,
            "current_tree_verified": True,
            "official_dataset_reconstructed": official_dataset_verified,
            "canonical_reader_requests_recomputed": canonical_reader_protocol_verified,
            "implementation": run["implementation"],
        },
    }


def compile_mem2act_report(
    run_path: str | Path,
    output_path: str | Path,
    *,
    dataset_dir: str | Path | None = None,
    artifact_root: Path | None = None,
    code_root: Path | None = None,
    enforce_repository_local: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Recompute a Mem2Act report from a bound run manifest and raw JSONL."""

    artifact_root = (artifact_root or repository_root()).resolve()
    code_root = (code_root or repository_root()).resolve()
    run_file, _ = _resolve_input(
        run_path,
        root=artifact_root,
        label="--run",
        require_relative=enforce_repository_local,
    )
    supplied_output = Path(output_path)
    if ".." in supplied_output.parts:
        raise Mem2ActReportError("--output cannot contain '..'")
    if supplied_output.is_absolute():
        # Readiness replay owns an isolated absolute temporary output. Normalize
        # its parent first so the platform's system temp symlink (for example
        # /var -> /private/var on macOS) cannot defeat the component checks.
        output_root = supplied_output.parent.resolve()
        checked_output: Path = output_root / supplied_output.name
        require_relative_output = False
    else:
        output_root = artifact_root
        checked_output = supplied_output
        require_relative_output = enforce_repository_local
    output_file = _resolve_output(
        checked_output,
        root=output_root,
        require_relative=require_relative_output,
        overwrite=overwrite,
    )
    if output_file.resolve() == run_file:
        raise Mem2ActReportError("--output cannot overwrite --run")
    run_raw = run_file.read_bytes()
    run_payload = _strict_object(run_raw, label="--run")
    run = _validate_run(run_payload, code_root=code_root)
    official_dataset: Mem2ActDataset | None = None
    if run["official_dataset"]:
        if dataset_dir is None:
            raise Mem2ActReportError(
                "official Mem2Act evidence requires --dataset-dir for raw dataset replay"
            )
        official_dataset = _load_official_dataset_evidence(
            dataset_dir,
            artifact_root=artifact_root,
            enforce_repository_local=enforce_repository_local,
        )

    prediction_reference = Path(run["predictions_artifact"]["path"])
    if (
        prediction_reference.is_absolute()
        or len(prediction_reference.parts) != 1
        or prediction_reference.name != run["predictions_artifact"]["path"]
    ):
        raise Mem2ActReportError("run.predictions_artifact.path must be one safe sibling filename")
    predictions_file, _ = _resolve_input(
        run_file.parent / prediction_reference,
        root=artifact_root,
        label="run.predictions_artifact.path",
        require_relative=False,
    )
    if output_file.resolve() == predictions_file:
        raise Mem2ActReportError("--output cannot overwrite raw predictions")
    predictions_raw = predictions_file.read_bytes()
    descriptor = run["predictions_artifact"]
    if len(predictions_raw) != descriptor["bytes"]:
        raise Mem2ActReportError("raw prediction byte count differs from run artifact")
    if hashlib.sha256(predictions_raw).hexdigest() != descriptor["sha256"]:
        raise Mem2ActReportError("raw prediction SHA-256 differs from run artifact")
    rows = _strict_jsonl(predictions_raw, label="raw predictions")
    if len(rows) != descriptor["rows"]:
        raise Mem2ActReportError("raw prediction row count differs from run artifact")
    records = tuple(_prediction_record(row, index=index) for index, row in enumerate(rows))
    official_dataset_verified = False
    canonical_reader_protocol_verified = False
    if official_dataset is not None:
        canonical_reader_protocol_verified = _validate_official_dataset_evidence(
            records,
            run=run,
            dataset=official_dataset,
        )
        official_dataset_verified = True
    predictions_identity = {
        **descriptor,
        "path": predictions_file.name,
    }
    run_identity = {
        "path": run_file.name,
        "sha256": hashlib.sha256(run_raw).hexdigest(),
        "bytes": len(run_raw),
    }
    report = _build_report(
        records,
        run=run,
        run_identity=run_identity,
        predictions_identity=predictions_identity,
        official_dataset_verified=official_dataset_verified,
        canonical_reader_protocol_verified=canonical_reader_protocol_verified,
    )
    if run_file.read_bytes() != run_raw or predictions_file.read_bytes() != predictions_raw:
        raise Mem2ActReportError("raw evidence changed during offline compilation")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["Mem2ActReportError", "compile_mem2act_report"]
