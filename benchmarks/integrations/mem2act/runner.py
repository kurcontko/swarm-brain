"""Leakage-safe three-arm Mem2ActBench evaluation runner."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from .contracts import (
    FailureRecord,
    Mem2ActContractError,
    Mem2ActDataset,
    Mem2ActTask,
    MemoryBridge,
    PredictionRecord,
    ReaderRequest,
    ReaderResult,
    RetrievalResult,
    TaskMetrics,
    ToolPrediction,
    ToolSelectionReader,
)
from .dataset import canonical_json, sha256_file
from .metrics import aggregate_arm, paired_bootstrap, parse_tool_prediction, score_prediction
from .provenance import (
    PROTOCOL_VERSION,
    RUN_ARTIFACT_TYPE,
    RUN_SCHEMA_VERSION,
    implementation_fingerprint,
)

NO_MEMORY_ARM = "no_memory"
SWARM_ARM = "swarm"
ORACLE_ARM = "oracle"
REQUIRED_ARMS = (NO_MEMORY_ARM, SWARM_ARM, ORACLE_ARM)
TARGET_TOOL_GIVEN = "target_tool_given"
FULL_CATALOG = "full_catalog"
REQUIRED_CONDITIONS = (TARGET_TOOL_GIVEN, FULL_CATALOG)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    retrieval_limit: int = 5
    retrieval_token_budget: int | None = 8_192
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 2_026_080_9
    bootstrap_confidence: float = 0.95
    task_limit: int | None = None
    expected_reader_model: str | None = None
    reader_revision: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.retrieval_limit, int)
            or isinstance(self.retrieval_limit, bool)
            or not 1 <= self.retrieval_limit <= 100
        ):
            raise Mem2ActContractError("retrieval_limit must be an integer in [1, 100]")
        if self.retrieval_token_budget is not None and (
            not isinstance(self.retrieval_token_budget, int)
            or isinstance(self.retrieval_token_budget, bool)
            or self.retrieval_token_budget < 1
        ):
            raise Mem2ActContractError("retrieval_token_budget must be a positive integer")
        if (
            not isinstance(self.bootstrap_resamples, int)
            or isinstance(self.bootstrap_resamples, bool)
            or self.bootstrap_resamples < 1
        ):
            raise Mem2ActContractError("bootstrap_resamples must be a positive integer")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise Mem2ActContractError("bootstrap_confidence must be between zero and one")
        if self.task_limit is not None and (
            not isinstance(self.task_limit, int)
            or isinstance(self.task_limit, bool)
            or self.task_limit < 1
        ):
            raise Mem2ActContractError("task_limit must be a positive integer")
        for name, value in (
            ("expected_reader_model", self.expected_reader_model),
            ("reader_revision", self.reader_revision),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise Mem2ActContractError(f"{name} must be a non-empty string when set")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    records: tuple[PredictionRecord, ...]
    report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OutputPaths:
    predictions: Path
    run: Path
    report: Path


class Mem2ActEvaluator:
    def __init__(
        self,
        dataset: Mem2ActDataset,
        memory: MemoryBridge,
        reader: ToolSelectionReader,
        *,
        config: BenchmarkConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.memory = memory
        self.reader = reader
        self.config = config or BenchmarkConfig()
        self._observed_reader_model: str | None = None
        self._reader_model_consistent = True

    async def run(self) -> BenchmarkResult:
        tasks = self.dataset.tasks
        if self.config.task_limit is not None:
            tasks = tasks[: self.config.task_limit]
        if not tasks:
            raise Mem2ActContractError("no Mem2ActBench tasks selected")

        # One fixed corpus is ingested once for every query.  No task label is
        # consulted when deciding what enters Swarm Brain.
        public_sessions = tuple(session.public_view() for session in self.dataset.sessions)
        ingestion = await self.memory.ingest(public_sessions)
        if ingestion.memory_count != len(public_sessions):
            raise Mem2ActContractError(
                "memory bridge did not acknowledge every published Mem2ActBench session"
            )
        _validate_nonnegative_finite(ingestion.latency_ms, "ingestion latency")
        canonical_json(ingestion.metadata)

        records: list[PredictionRecord] = []
        for task in tasks:
            shared_retrieval = RetrievalResult(
                memories=(), latency_ms=0.0, total_candidates=0, truncated=False
            )
            retrieval_error: Exception | None = None
            try:
                shared_retrieval = await self.memory.retrieve(
                    task.query,
                    limit=self.config.retrieval_limit,
                    token_budget=self.config.retrieval_token_budget,
                )
                _validate_retrieval(shared_retrieval)
            except Exception as exc:
                retrieval_error = exc
            for condition in REQUIRED_CONDITIONS:
                for arm in REQUIRED_ARMS:
                    records.append(
                        await self._run_case(
                            task,
                            condition,
                            arm,
                            shared_retrieval=shared_retrieval,
                            retrieval_error=retrieval_error,
                        )
                    )

        condition_reports: dict[str, Any] = {}
        for condition in REQUIRED_CONDITIONS:
            condition_records = [record for record in records if record.condition == condition]
            by_arm = {
                arm: [record for record in condition_records if record.arm == arm]
                for arm in REQUIRED_ARMS
            }
            condition_reports[condition] = {
                "paper_parameter_grounding_comparable": condition == TARGET_TOOL_GIVEN,
                "candidate_tool_count": (
                    1 if condition == TARGET_TOOL_GIVEN else len(self.dataset.tool_catalog)
                ),
                "arms": {arm: aggregate_arm(by_arm[arm]) for arm in REQUIRED_ARMS},
                "paired_bootstrap": paired_bootstrap(
                    condition_records,
                    arm_pairs=((SWARM_ARM, NO_MEMORY_ARM), (ORACLE_ARM, SWARM_ARM)),
                    resamples=self.config.bootstrap_resamples,
                    seed=self.config.bootstrap_seed,
                    confidence=self.config.bootstrap_confidence,
                ),
            }
        primary = condition_reports[TARGET_TOOL_GIVEN]
        strict = condition_reports[FULL_CATALOG]
        arm_summaries = primary["arms"]
        bootstrap = primary["paired_bootstrap"]
        memory_delta = bootstrap["pairs"]["swarm-minus-no_memory"]["parameter_f1"]
        total_failures = sum(record.failure is not None for record in records)
        fixed_reader_model = (
            self._observed_reader_model is not None and self._reader_model_consistent
        )
        memory_bridge_evidence = _memory_bridge_evidence(self.memory)
        completed_full_protocol = (
            len(tasks) == 400
            and len(self.dataset.tasks) == 400
            and total_failures == 0
            and fixed_reader_model
            and self.config.expected_reader_model is not None
            and self.config.reader_revision is not None
        )
        report = {
            "schema_version": 1,
            "benchmark": "Mem2ActBench",
            "generated_at": datetime.now(UTC).isoformat(),
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
                "complete_400_task_protocol": completed_full_protocol,
            },
            # Stable gate-facing projection.  The richer per-arm accounting
            # remains under ``arms``; these fields intentionally match the
            # SOTA readiness manifest without manufacturing a result artifact.
            "evaluation": {
                "oracle_arm": True,
                "no_memory_arm": True,
                "paired": True,
                "primary_parameter_condition": TARGET_TOOL_GIVEN,
                "strict_tool_selection_condition": FULL_CATALOG,
                "reader_model": self._observed_reader_model,
                "reader_revision": self.config.reader_revision,
                "reader_model_pinned": (
                    self.config.expected_reader_model is not None and self._reader_model_consistent
                ),
                "fixed_reader_model": fixed_reader_model,
                "total_failures": total_failures,
                "complete_400_task_protocol": completed_full_protocol,
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
            "dataset": asdict(self.dataset.fingerprint),
            "tool_catalog": {
                "entries": len(self.dataset.tool_catalog),
                "sha256": self.dataset.tool_catalog_sha256,
            },
            "configuration": asdict(self.config),
            "implementations": {
                "memory_bridge": _qualified_name(self.memory),
                "reader": _qualified_name(self.reader),
            },
            "memory_bridge_evidence": memory_bridge_evidence,
            "ingestion": asdict(ingestion),
            "evaluated_task_count": len(tasks),
            "prediction_count": len(records),
            "conditions": condition_reports,
            # ``arms`` remains a concise alias for the paper-comparable
            # target-tool-given condition.
            "arms": arm_summaries,
            "memory": _gate_metrics(arm_summaries[SWARM_ARM], strict["arms"][SWARM_ARM]),
            "no_memory": _gate_metrics(arm_summaries[NO_MEMORY_ARM], strict["arms"][NO_MEMORY_ARM]),
            "oracle": _gate_metrics(arm_summaries[ORACLE_ARM], strict["arms"][ORACLE_ARM]),
            "comparison": {
                "memory_vs_no_memory_ci95": {
                    "metric": "micro_parameter_f1",
                    "delta": memory_delta["delta"],
                    "lower": memory_delta["ci_low"],
                    "upper": memory_delta["ci_high"],
                }
            },
            "paired_bootstrap": bootstrap,
            "reader_models": [self._observed_reader_model]
            if self._observed_reader_model is not None
            else [],
        }
        canonical_json(report)
        return BenchmarkResult(records=tuple(records), report=report)

    async def _run_case(
        self,
        task: Mem2ActTask,
        condition: str,
        arm: str,
        *,
        shared_retrieval: RetrievalResult,
        retrieval_error: Exception | None,
    ) -> PredictionRecord:
        total_started = perf_counter()
        retrieval = RetrievalResult(
            memories=(), latency_ms=0.0, total_candidates=0, truncated=False
        )
        if arm == SWARM_ARM:
            retrieval = shared_retrieval
            if retrieval_error is not None:
                return self._failure_record(
                    task,
                    condition,
                    arm,
                    retrieval=retrieval,
                    total_started=total_started,
                    stage="memory_retrieval",
                    exc=retrieval_error,
                )
            contexts = tuple(memory.content for memory in retrieval.memories)
        elif arm == ORACLE_ARM:
            contexts = tuple(memory.render() for memory in task.oracle_memories)
        elif arm == NO_MEMORY_ARM:
            contexts = ()
        else:
            raise Mem2ActContractError(f"unknown Mem2ActBench arm: {arm!r}")

        # Construct an allowlisted object rather than passing the label-bearing
        # task.  This is mechanically inspectable in fake-reader tests.
        request = ReaderRequest(
            condition=condition,
            query=task.query,
            memory_contexts=contexts,
            tool_catalog=self._reader_catalog(task, condition),
        )
        reader_started = perf_counter()
        reader_result: ReaderResult | None = None
        try:
            reader_result = await self.reader.select_tool(request)
            if not isinstance(reader_result, ReaderResult):
                raise Mem2ActContractError("reader must return ReaderResult")
            canonical_json(reader_result.metadata)
            self._record_reader_model(reader_result.model)
        except Exception as exc:
            return self._failure_record(
                task,
                condition,
                arm,
                retrieval=retrieval,
                contexts=contexts,
                total_started=total_started,
                reader_started=reader_started,
                stage="reader_call",
                exc=exc,
            )
        reader_wall_ms = (perf_counter() - reader_started) * 1000.0

        prediction: ToolPrediction | None = None
        failure: FailureRecord | None = None
        try:
            prediction = parse_tool_prediction(reader_result.raw_prediction)
        except Exception as exc:
            failure = _failure("prediction_parse", exc)
        metrics = score_prediction(
            prediction,
            gold_tool_name=task.gold_tool_name,
            gold_arguments=task.gold_arguments,
        )
        return _prediction_record(
            task=task,
            condition=condition,
            arm=arm,
            contexts=contexts,
            retrieval=retrieval,
            reader_wall_ms=reader_wall_ms,
            total_ms=(perf_counter() - total_started) * 1000.0 + retrieval.latency_ms,
            reader_result=reader_result,
            prediction=prediction,
            metrics=metrics,
            failure=failure,
            tool_catalog_sha256=_catalog_sha256(request.tool_catalog),
        )

    def _failure_record(
        self,
        task: Mem2ActTask,
        condition: str,
        arm: str,
        *,
        retrieval: RetrievalResult,
        total_started: float,
        stage: str,
        exc: Exception,
        contexts: tuple[str, ...] = (),
        reader_started: float | None = None,
    ) -> PredictionRecord:
        reader_wall_ms = (
            0.0 if reader_started is None else (perf_counter() - reader_started) * 1000.0
        )
        return _prediction_record(
            task=task,
            condition=condition,
            arm=arm,
            contexts=contexts,
            retrieval=retrieval,
            reader_wall_ms=reader_wall_ms,
            total_ms=(perf_counter() - total_started) * 1000.0 + retrieval.latency_ms,
            reader_result=None,
            prediction=None,
            metrics=score_prediction(
                None,
                gold_tool_name=task.gold_tool_name,
                gold_arguments=task.gold_arguments,
            ),
            failure=_failure(stage, exc),
            tool_catalog_sha256=_catalog_sha256(self._reader_catalog(task, condition)),
        )

    def _reader_catalog(self, task: Mem2ActTask, condition: str) -> tuple[dict[str, Any], ...]:
        if condition == FULL_CATALOG:
            return self.dataset.tool_catalog
        if condition != TARGET_TOOL_GIVEN:
            raise Mem2ActContractError(f"unknown Mem2ActBench condition: {condition!r}")
        target = canonical_json(task.target_tool_schema)
        matches = tuple(
            entry
            for entry in self.dataset.tool_catalog
            if canonical_json(entry.get("schema")) == target
        )
        if len(matches) != 1:
            raise Mem2ActContractError(
                f"{task.qa_id} target schema does not resolve uniquely in the tool catalog"
            )
        return matches

    def _record_reader_model(self, model: str) -> None:
        expected = self.config.expected_reader_model
        if expected is not None and model != expected:
            self._reader_model_consistent = False
            raise Mem2ActContractError(
                f"reader model mismatch: expected {expected!r}, got {model!r}"
            )
        if self._observed_reader_model is None:
            self._observed_reader_model = model
        elif model != self._observed_reader_model:
            self._reader_model_consistent = False
            raise Mem2ActContractError(
                f"reader model changed within one run: {self._observed_reader_model!r} -> {model!r}"
            )


def write_benchmark_outputs(
    result: BenchmarkResult,
    output_prefix: str | Path,
    *,
    dataset_dir: str | Path | None = None,
    overwrite: bool = False,
) -> OutputPaths:
    prefix = Path(output_prefix).expanduser().resolve()
    predictions_path = prefix.with_name(f"{prefix.name}-predictions.jsonl")
    run_path = prefix.with_name(f"{prefix.name}-run.json")
    report_path = prefix.with_name(f"{prefix.name}-report.json")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = [path for path in (predictions_path, run_path, report_path) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite benchmark output: {existing[0]}")

    mode = "w" if overwrite else "x"
    with predictions_path.open(mode, encoding="utf-8") as handle:
        for record in result.records:
            handle.write(canonical_json(asdict(record)))
            handle.write("\n")
    predictions_hash = sha256_file(predictions_path)

    predictions_artifact = {
        "path": predictions_path.name,
        "sha256": predictions_hash,
        "bytes": predictions_path.stat().st_size,
        "rows": len(result.records),
        "preserves_raw_predictions": True,
        "preserves_reader_contexts": True,
        "preserves_failures_latency_and_tokens": True,
    }
    run_artifact = {
        "schema_version": RUN_SCHEMA_VERSION,
        "artifact_type": RUN_ARTIFACT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": result.report["generated_at"],
        "implementation": implementation_fingerprint(),
        "dataset": result.report["dataset"],
        "tool_catalog": result.report["tool_catalog"],
        "configuration": result.report["configuration"],
        "implementations": result.report["implementations"],
        "memory_bridge_evidence": result.report["memory_bridge_evidence"],
        "ingestion": result.report["ingestion"],
        "predictions_artifact": predictions_artifact,
    }
    with run_path.open(mode, encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                run_artifact,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
        handle.write("\n")
    # Import lazily so the offline compiler can import runner contracts without
    # creating a module cycle. The written report is always reconstructed from
    # the just-persisted raw files rather than trusted from ``result.report``.
    from .report import compile_mem2act_report

    compile_mem2act_report(
        run_path,
        report_path,
        dataset_dir=dataset_dir,
        artifact_root=prefix.parent,
        enforce_repository_local=False,
        overwrite=overwrite,
    )
    return OutputPaths(predictions=predictions_path, run=run_path, report=report_path)


async def resolve_factory(value: Any) -> Any:
    """Await async factories while allowing simple synchronous injection."""

    return await value if inspect.isawaitable(value) else value


def _prediction_record(
    *,
    task: Mem2ActTask,
    condition: str,
    arm: str,
    contexts: tuple[str, ...],
    retrieval: RetrievalResult,
    reader_wall_ms: float,
    total_ms: float,
    reader_result: ReaderResult | None,
    prediction: ToolPrediction | None,
    metrics: TaskMetrics,
    failure: FailureRecord | None,
    tool_catalog_sha256: str,
) -> PredictionRecord:
    return PredictionRecord(
        qa_id=task.qa_id,
        condition=condition,
        arm=arm,
        query=task.query,
        complexity_level=task.complexity_level,
        memory_contexts=contexts,
        retrieved_memory_ids=tuple(memory.memory_id for memory in retrieval.memories),
        retrieved_scores=tuple(memory.score for memory in retrieval.memories),
        retrieval_reasons=tuple(memory.reasons for memory in retrieval.memories),
        retrieval_total_candidates=retrieval.total_candidates,
        retrieval_truncated=retrieval.truncated,
        retrieval_latency_ms=retrieval.latency_ms,
        reader_wall_latency_ms=reader_wall_ms,
        reader_reported_latency_ms=(None if reader_result is None else reader_result.latency_ms),
        total_latency_ms=total_ms,
        prompt_tokens=0 if reader_result is None else reader_result.prompt_tokens,
        completion_tokens=0 if reader_result is None else reader_result.completion_tokens,
        reader_model=None if reader_result is None else reader_result.model,
        reader_metadata={} if reader_result is None else reader_result.metadata,
        raw_prediction=None if reader_result is None else reader_result.raw_prediction,
        parsed_prediction=prediction,
        gold_tool_name=task.gold_tool_name,
        gold_arguments=task.gold_arguments,
        metrics=metrics,
        failure=failure,
        tool_catalog_sha256=tool_catalog_sha256,
    )


def _validate_retrieval(result: RetrievalResult) -> None:
    if not isinstance(result, RetrievalResult):
        raise Mem2ActContractError("memory bridge must return RetrievalResult")
    _validate_nonnegative_finite(result.latency_ms, "retrieval latency")
    if (
        not isinstance(result.total_candidates, int)
        or isinstance(result.total_candidates, bool)
        or result.total_candidates < 0
    ):
        raise Mem2ActContractError("retrieval total_candidates must be non-negative")
    if not isinstance(result.truncated, bool):
        raise Mem2ActContractError("retrieval truncated must be boolean")
    canonical_json(result.metadata)
    seen: set[str] = set()
    for memory in result.memories:
        if not memory.memory_id or memory.memory_id in seen:
            raise Mem2ActContractError("retrieval memory IDs must be non-empty and unique")
        seen.add(memory.memory_id)
        if not isinstance(memory.content, str):
            raise Mem2ActContractError("retrieval memory content must be a string")
        _validate_nonnegative_finite(memory.score, "retrieval score")
        if memory.score > 1.0:
            raise Mem2ActContractError("retrieval score must be <= 1.0")
        if any(not isinstance(reason, str) for reason in memory.reasons):
            raise Mem2ActContractError("retrieval reasons must be strings")


def _validate_nonnegative_finite(value: float, field: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise Mem2ActContractError(f"{field} must be a non-negative finite number")


def _failure(stage: str, exc: Exception) -> FailureRecord:
    return FailureRecord(
        stage=stage,
        error_type=type(exc).__name__,
        message=str(exc),
    )


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


def _catalog_sha256(catalog: tuple[dict[str, Any], ...]) -> str:
    return hashlib.sha256(canonical_json(catalog).encode("utf-8")).hexdigest()


def _qualified_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _memory_bridge_evidence(memory: MemoryBridge) -> dict[str, Any]:
    evidence = getattr(memory, "evidence_metadata", None)
    if not callable(evidence):
        return {"available": False}
    payload = evidence()
    if not isinstance(payload, dict):
        raise Mem2ActContractError("memory bridge evidence_metadata must return an object")
    _reject_secret_fields(payload, path="memory_bridge_evidence")
    canonical_json(payload)
    return {"available": True, **payload}


def _reject_secret_fields(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = "".join(character for character in key.lower() if character.isalnum())
            env_reference = normalized.endswith(("env", "envvar", "envname"))
            if not env_reference and any(
                marker in normalized
                for marker in ("apikey", "authorization", "cookie", "password", "secret")
            ):
                raise Mem2ActContractError(
                    f"{path}.{key} cannot persist credential material; store an env name"
                )
            _reject_secret_fields(item, path=f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_secret_fields(item, path=f"{path}[{index}]")


def request_sha256(request: ReaderRequest) -> str:
    """Stable audit fingerprint for custom reader integrations."""

    return hashlib.sha256(canonical_json(asdict(request)).encode("utf-8")).hexdigest()


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "Mem2ActEvaluator",
    "NO_MEMORY_ARM",
    "ORACLE_ARM",
    "OutputPaths",
    "REQUIRED_ARMS",
    "REQUIRED_CONDITIONS",
    "SWARM_ARM",
    "TARGET_TOOL_GIVEN",
    "FULL_CATALOG",
    "request_sha256",
    "resolve_factory",
    "write_benchmark_outputs",
]
