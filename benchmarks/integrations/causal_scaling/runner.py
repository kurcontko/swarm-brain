"""Deterministic paired runner for 1/2/4-agent causal-scaling trials."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    AGENT_COUNTS,
    AgentBudget,
    AgentUsage,
    CausalScalingError,
    CausalTask,
    EvaluationProvenance,
    ExecutionResult,
    FailureRecord,
    FailureStage,
    OutcomeScorer,
    RolloutExecutor,
    RolloutRequest,
    RuntimeEventEnvelope,
    RuntimeEvidenceReader,
    ScoreResult,
    TotalBudget,
    canonical_json,
    sha256_json,
)
from .evidence import project_memory_use
from .workload import WorkloadManifest

MEMORY_CONDITION = "memory"
NO_MEMORY_CONDITION = "no_memory"
CONDITIONS = (NO_MEMORY_CONDITION, MEMORY_CONDITION)
SCHEDULE_VERSION = "paired-seeded-mt19937-v1"
RUNNER_VERSION = "swarmbrain-causal-scaling-v1"


@dataclass(frozen=True, slots=True)
class CausalScalingConfig:
    total_budget: TotalBudget
    seeds: tuple[int, ...] = (104_729, 130_363, 155_921, 181_081, 206_369)
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 2_026_080_9
    bootstrap_confidence: float = 0.95

    def __post_init__(self) -> None:
        if len(self.seeds) < 5 or len(set(self.seeds)) != len(self.seeds):
            raise CausalScalingError("at least five unique rollout seeds are required")
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in self.seeds):
            raise CausalScalingError("rollout seeds must be integers")
        if self.bootstrap_resamples < 10_000:
            raise CausalScalingError("canonical cluster bootstrap requires >= 10,000 resamples")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise CausalScalingError("bootstrap_confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class _ScheduledCase:
    schedule_index: int
    seed: int
    task: CausalTask
    agent_count: int
    memory_enabled: bool
    rollout_key: str


@dataclass(frozen=True, slots=True)
class RawRunResult:
    payload: dict[str, Any]


class CausalScalingRunner:
    """Run every measured cell while preserving failures as typed raw rows."""

    def __init__(
        self,
        *,
        tasks: tuple[CausalTask, ...],
        executor: RolloutExecutor,
        scorer: OutcomeScorer,
        evidence_reader: RuntimeEvidenceReader,
        provenance: EvaluationProvenance,
        workload_manifest: WorkloadManifest,
        config: CausalScalingConfig,
    ) -> None:
        self.tasks = tasks
        self.executor = executor
        self.scorer = scorer
        self.evidence_reader = evidence_reader
        self.provenance = provenance
        self.workload_manifest = workload_manifest
        self.config = config
        _validate_tasks(tasks)
        workload_manifest.validate(tasks)

    async def run(self) -> RawRunResult:
        schedule = build_schedule(self.tasks, self.config.seeds)
        records = [await self._run_case(case) for case in schedule]
        task_projection = [
            {
                "task_id": task.task_id,
                "cluster_id": task.cluster_id,
                "fingerprint": task.fingerprint,
            }
            for task in sorted(self.tasks, key=lambda item: item.task_id)
        ]
        protocol = {
            "runner_version": RUNNER_VERSION,
            "agent_counts": list(AGENT_COUNTS),
            "conditions": list(CONDITIONS),
            "paired": True,
            "isolated_run_per_cell": True,
            "randomized": True,
            "schedule_version": SCHEDULE_VERSION,
            "seeds": list(self.config.seeds),
            "total_budget_per_rollout": asdict(self.config.total_budget),
            "budget_semantics": (
                "provider-reported input plus output model tokens and all tool calls; "
                "one fixed total cap divided across agent slots"
            ),
            "bootstrap": {
                "unit": "task_cluster",
                "paired": True,
                "resamples": self.config.bootstrap_resamples,
                "seed": self.config.bootstrap_seed,
                "confidence": self.config.bootstrap_confidence,
            },
        }
        protocol["config_sha256"] = sha256_json(protocol)
        return RawRunResult(
            payload={
                "schema_version": 1,
                "benchmark": {
                    "name": "Swarm Brain 1/2/4-agent causal scaling",
                    "runner_version": RUNNER_VERSION,
                },
                "execution": {
                    "provenance": asdict(self.provenance),
                    "provenance_sha256": self.provenance.digest,
                },
                "protocol": protocol,
                "workload": {
                    "schema": self.workload_manifest.schema,
                    "workload_id": self.workload_manifest.workload_id,
                    "workload_revision": self.workload_manifest.workload_revision,
                    "source": self.workload_manifest.source,
                    "source_revision": self.workload_manifest.source_revision,
                    "verifier_schema": self.workload_manifest.verifier_schema,
                    "review_status": self.workload_manifest.review_status,
                    "review_revision": self.workload_manifest.review_revision,
                    "task_count": self.workload_manifest.task_count,
                    "cluster_count": self.workload_manifest.cluster_count,
                    "manifest_sha256": self.workload_manifest.digest,
                },
                "task_set": {
                    "task_count": len(task_projection),
                    "cluster_count": len({task.cluster_id for task in self.tasks}),
                    "sha256": sha256_json(task_projection),
                    "tasks": task_projection,
                },
                "schedule": {
                    "sha256": sha256_json([case.rollout_key for case in schedule]),
                    "record_count": len(schedule),
                    "rollout_keys": [case.rollout_key for case in schedule],
                },
                "records": records,
            }
        )

    async def _run_case(self, case: _ScheduledCase) -> dict[str, Any]:
        budgets = split_budget(self.config.total_budget, case.agent_count)
        profiles = self.provenance.model_profiles[: case.agent_count]
        request = RolloutRequest(
            rollout_key=case.rollout_key,
            schedule_index=case.schedule_index,
            seed=case.seed,
            agent_count=case.agent_count,
            memory_enabled=case.memory_enabled,
            task=case.task.public_view(),
            total_budget=self.config.total_budget,
            agent_budgets=budgets,
            model_profiles=profiles,
        )
        failures: list[FailureRecord] = []
        execution: ExecutionResult | None = None
        score: ScoreResult | None = None
        envelope: Any = None
        proof: Any = None
        try:
            execution = await self.executor.execute(request)
            if not isinstance(execution, ExecutionResult):
                raise CausalScalingError("executor must return ExecutionResult")
        except Exception as exc:
            failures.append(_failure(FailureStage.EXECUTION, exc))

        if execution is not None:
            try:
                _validate_usage(request, execution.agent_usage)
            except Exception as exc:
                failures.append(_failure(FailureStage.USAGE_VALIDATION, exc))
            try:
                score = await self.scorer.score(case.task, execution)
                if not isinstance(score, ScoreResult):
                    raise CausalScalingError("scorer must return ScoreResult")
            except Exception as exc:
                failures.append(_failure(FailureStage.OUTCOME_SCORING, exc))
            try:
                candidate = await self.evidence_reader.read_events(request, execution)
                if not isinstance(candidate, RuntimeEventEnvelope):
                    raise CausalScalingError("evidence reader must return RuntimeEventEnvelope")
                envelope = candidate
            except Exception as exc:
                failures.append(_failure(FailureStage.EVIDENCE_READ, exc))
            if envelope is not None:
                try:
                    proof = project_memory_use(envelope, request=request, result=execution)
                except Exception as exc:
                    failures.append(_failure(FailureStage.EVIDENCE_VALIDATION, exc))

        usage = () if execution is None else execution.agent_usage
        return {
            "rollout_key": case.rollout_key,
            "schedule_index": case.schedule_index,
            "task_id": case.task.task_id,
            "task_cluster_id": case.task.cluster_id,
            "task_fingerprint": case.task.fingerprint,
            "seed": case.seed,
            "agent_count": case.agent_count,
            "condition": MEMORY_CONDITION if case.memory_enabled else NO_MEMORY_CONDITION,
            "memory_enabled": case.memory_enabled,
            "provenance_sha256": self.provenance.digest,
            "budget": {
                "total": asdict(self.config.total_budget),
                "allocations": [asdict(item) for item in budgets],
            },
            "execution": None
            if execution is None
            else {
                "run_id": execution.run_id,
                "raw_output": execution.raw_output,
                "raw_output_sha256": sha256_json(execution.raw_output),
                "agent_usage": [asdict(item) for item in usage],
                "model_tokens_used": sum(item.model_tokens for item in usage),
                "tool_calls_used": sum(item.tool_calls for item in usage),
                "metadata": execution.metadata,
            },
            "outcome": None if score is None else asdict(score),
            "runtime_event_envelope": None if envelope is None else asdict(envelope),
            "memory_use_proof": None if proof is None else asdict(proof),
            "failures": [asdict(item) for item in failures],
        }


def split_budget(total: TotalBudget, agent_count: int) -> tuple[AgentBudget, ...]:
    """Divide both caps deterministically; remainders go to lower slots."""

    if agent_count not in AGENT_COUNTS:
        raise CausalScalingError("agent_count must be one of 1, 2, or 4")
    model_base, model_remainder = divmod(total.model_tokens, agent_count)
    tool_base, tool_remainder = divmod(total.tool_calls, agent_count)
    allocations = tuple(
        AgentBudget(
            agent_slot=slot,
            model_tokens=model_base + (slot <= model_remainder),
            tool_calls=tool_base + (slot <= tool_remainder),
        )
        for slot in range(1, agent_count + 1)
    )
    if sum(item.model_tokens for item in allocations) != total.model_tokens:
        raise AssertionError("model budget split lost tokens")
    if sum(item.tool_calls for item in allocations) != total.tool_calls:
        raise AssertionError("tool budget split lost calls")
    return allocations


def build_schedule(
    tasks: tuple[CausalTask, ...],
    seeds: tuple[int, ...],
) -> tuple[_ScheduledCase, ...]:
    """Return a reproducible random order over every paired factorial cell."""

    ordered_tasks = tuple(sorted(tasks, key=lambda item: item.task_id))
    scheduled: list[_ScheduledCase] = []
    for seed in seeds:
        cells = [
            (task, agent_count, condition == MEMORY_CONDITION)
            for task in ordered_tasks
            for agent_count in AGENT_COUNTS
            for condition in CONDITIONS
        ]
        random.Random(seed).shuffle(cells)
        for task, agent_count, memory_enabled in cells:
            rollout_key = sha256_json(
                {
                    "schedule_version": SCHEDULE_VERSION,
                    "seed": seed,
                    "task_id": task.task_id,
                    "task_fingerprint": task.fingerprint,
                    "agent_count": agent_count,
                    "memory_enabled": memory_enabled,
                }
            )
            scheduled.append(
                _ScheduledCase(
                    schedule_index=len(scheduled),
                    seed=seed,
                    task=task,
                    agent_count=agent_count,
                    memory_enabled=memory_enabled,
                    rollout_key=rollout_key,
                )
            )
    return tuple(scheduled)


def write_raw_evidence(
    result: RawRunResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8") as handle:
        handle.write(canonical_json(result.payload))
        handle.write("\n")
    return destination


def _validate_tasks(tasks: tuple[CausalTask, ...]) -> None:
    if len(tasks) < 2:
        raise CausalScalingError("causal scaling requires at least two tasks")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise CausalScalingError("causal-scaling task IDs must be unique")
    if len({task.cluster_id for task in tasks}) < 2:
        raise CausalScalingError("task-cluster bootstrap requires at least two clusters")


def _validate_usage(request: RolloutRequest, usages: tuple[AgentUsage, ...]) -> None:
    if len(usages) != request.agent_count:
        raise CausalScalingError("executor must report usage for every configured agent slot")
    by_slot = {usage.agent_slot: usage for usage in usages}
    if set(by_slot) != set(range(1, request.agent_count + 1)):
        raise CausalScalingError("executor usage slots are missing or duplicated")
    budget_by_slot = {budget.agent_slot: budget for budget in request.agent_budgets}
    profile_by_slot = {profile.agent_slot: profile for profile in request.model_profiles}
    for slot, usage in by_slot.items():
        profile = profile_by_slot[slot]
        if (usage.provider, usage.model, usage.revision) != (
            profile.provider,
            profile.model,
            profile.revision,
        ):
            raise CausalScalingError("observed model identity differs from pinned provenance")
        budget = budget_by_slot[slot]
        if usage.model_tokens > budget.model_tokens or usage.tool_calls > budget.tool_calls:
            raise CausalScalingError("agent usage exceeds its allocated share of the total budget")
    if sum(usage.model_tokens for usage in usages) > request.total_budget.model_tokens:
        raise CausalScalingError("rollout exceeds total model-token budget")
    if sum(usage.tool_calls for usage in usages) > request.total_budget.tool_calls:
        raise CausalScalingError("rollout exceeds total tool-call budget")


def _failure(stage: FailureStage, exc: Exception) -> FailureRecord:
    return FailureRecord(stage=stage, error_type=type(exc).__name__, message=str(exc))


__all__ = [
    "CONDITIONS",
    "MEMORY_CONDITION",
    "NO_MEMORY_CONDITION",
    "RUNNER_VERSION",
    "SCHEDULE_VERSION",
    "CausalScalingConfig",
    "CausalScalingRunner",
    "RawRunResult",
    "build_schedule",
    "split_budget",
    "write_raw_evidence",
]
