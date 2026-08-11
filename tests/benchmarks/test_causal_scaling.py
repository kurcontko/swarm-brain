from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from benchmarks.integrations.causal_scaling import (
    AgentUsage,
    CausalScalingConfig,
    CausalScalingError,
    CausalScalingRunner,
    CausalTask,
    EvaluationProvenance,
    ExecutionKind,
    ExecutionResult,
    ModelProfile,
    RuntimeEventEnvelope,
    ScoreResult,
    TotalBudget,
    WorkloadManifest,
    WorkloadTaskEntry,
    build_causal_scaling_report,
)
from benchmarks.integrations.causal_scaling.contracts import sha256_json
from benchmarks.integrations.causal_scaling.evidence import PUBLIC_EVENT_SOURCE
from benchmarks.integrations.causal_scaling.runner import split_budget
from benchmarks.integrations.causal_scaling.workload import WORKLOAD_SCHEMA


def _tasks() -> tuple[CausalTask, ...]:
    return (
        CausalTask(
            task_id="task-a",
            cluster_id="question-a",
            prompt="Recover the hidden deployment invariant.",
            evaluation_payload={"answer": "alpha"},
        ),
        CausalTask(
            task_id="task-b",
            cluster_id="question-b",
            prompt="Recover the hidden rollback invariant.",
            evaluation_payload={"answer": "beta"},
        ),
    )


def _workload(
    tasks: tuple[CausalTask, ...], *, review_status: str = "reviewed"
) -> WorkloadManifest:
    return WorkloadManifest(
        schema=WORKLOAD_SCHEMA,
        workload_id="fixture-reviewed-memory-dependent-v1",
        workload_revision="fixture-revision-1",
        source="tests/benchmarks/test_causal_scaling.py",
        source_revision="fixture-source-1",
        verifier_schema="exact-hidden-answer-v1",
        review_status=review_status,
        review_revision="fixture-review-1",
        task_count=len(tasks),
        cluster_count=len({task.cluster_id for task in tasks}),
        tasks=tuple(
            WorkloadTaskEntry(
                task_id=task.task_id,
                cluster_id=task.cluster_id,
                public_task_sha256=task.public_fingerprint,
                hidden_verifier_sha256=task.hidden_verifier_fingerprint,
                task_fingerprint=task.fingerprint,
            )
            for task in sorted(tasks, key=lambda item: item.task_id)
        ),
    )


def _provenance(
    kind: ExecutionKind = ExecutionKind.MEASURED_EXTERNAL,
) -> EvaluationProvenance:
    return EvaluationProvenance(
        execution_kind=kind,
        comparable=kind is ExecutionKind.MEASURED_EXTERNAL,
        code_revision="fixture-code-revision",
        adapter_name="fixture-provider-adapter",
        adapter_revision="fixture-adapter-revision",
        environment_digest="fixture-environment-digest",
        tool_catalog_sha256=sha256_json({"tools": ["complete_task"]}),
        tool_runtime_revision="fixture-tool-runtime-revision",
        model_profiles=tuple(
            ModelProfile(
                agent_slot=slot,
                provider="fixture-provider",
                model="fixture-model",
                revision="fixture-model-revision",
                decoding_config_sha256=sha256_json({"temperature": 0, "slot": slot}),
            )
            for slot in range(1, 5)
        ),
    )


class _Executor:
    def __init__(self, *, over_budget: bool = False) -> None:
        self.requests: list[Any] = []
        self.over_budget = over_budget

    async def execute(self, request: Any) -> ExecutionResult:
        self.requests.append(request)
        allocations = split_budget(request.total_budget, request.agent_count)
        usages = tuple(
            AgentUsage(
                agent_slot=slot,
                provider="fixture-provider",
                model="fixture-model",
                revision="fixture-model-revision",
                input_tokens=(allocation.model_tokens + 1)
                if self.over_budget and slot == 1
                else min(7, allocation.model_tokens),
                output_tokens=0,
                tool_calls=min(1, allocation.tool_calls),
            )
            for slot, allocation in enumerate(allocations, start=1)
        )
        return ExecutionResult(
            run_id=f"run-{request.rollout_key}",
            raw_output=json.dumps(
                {
                    "memory_enabled": request.memory_enabled,
                    "agent_count": request.agent_count,
                }
            ),
            agent_usage=usages,
            metadata={"provider_request_id": request.rollout_key[:12]},
        )


class _Scorer:
    async def score(self, task: CausalTask, result: ExecutionResult) -> ScoreResult:
        del task
        payload = json.loads(result.raw_output)
        success = bool(payload["memory_enabled"] and payload["agent_count"] == 4)
        return ScoreResult(success=success, score=float(success), details={"verifier": "exact"})


class _Events:
    async def read_events(self, request: Any, result: ExecutionResult) -> RuntimeEventEnvelope:
        lease_id = f"lease-{request.rollout_key}"
        memory_id = f"memory-{request.task.task_id}"
        events: list[dict[str, Any]] = []
        if request.memory_enabled:
            events.append(
                {
                    "event_id": f"activation-{request.rollout_key}",
                    "event_type": "memory.activated",
                    "run_id": result.run_id,
                    "task_id": request.task.task_id,
                    "agent_id": "agent-1",
                    "payload": {"lease_id": lease_id, "memory_ids": [memory_id]},
                }
            )
        events.append(
            {
                "event_id": f"completion-{request.rollout_key}",
                "event_type": "task.completed",
                "run_id": result.run_id,
                "task_id": request.task.task_id,
                "agent_id": "agent-1",
                "payload": {
                    "lease_id": lease_id,
                    "memory_ids": [memory_id] if request.memory_enabled else [],
                },
            }
        )
        return RuntimeEventEnvelope(
            source=PUBLIC_EVENT_SOURCE,
            run_id=result.run_id,
            complete=True,
            page_count=1,
            events=tuple(events),
        )


async def _raw(
    *,
    provenance: EvaluationProvenance | None = None,
    executor: _Executor | None = None,
) -> tuple[dict[str, Any], WorkloadManifest, _Executor]:
    tasks = _tasks()
    workload = _workload(tasks)
    selected_executor = executor or _Executor()
    runner = CausalScalingRunner(
        tasks=tasks,
        executor=selected_executor,
        scorer=_Scorer(),
        evidence_reader=_Events(),
        provenance=provenance or _provenance(),
        workload_manifest=workload,
        config=CausalScalingConfig(total_budget=TotalBudget(model_tokens=100, tool_calls=8)),
    )
    result = await runner.run()
    return result.payload, workload, selected_executor


@pytest.mark.asyncio
async def test_runner_is_paired_budget_matched_and_compiler_derives_did() -> None:
    raw, workload, executor = await _raw()

    assert len(raw["records"]) == 2 * 5 * 3 * 2
    assert len(executor.requests) == len(raw["records"])
    assert all(not hasattr(request.task, "evaluation_payload") for request in executor.requests)
    for request in executor.requests:
        assert sum(item.model_tokens for item in request.agent_budgets) == 100
        assert sum(item.tool_calls for item in request.agent_budgets) == 8

    report = build_causal_scaling_report(
        raw,
        workload_manifest=_manifest_dict(workload),
    )

    assert report["design"]["agent_counts"] == [1, 2, 4]
    assert report["arms"]["4"]["memory"]["proven_activation_citation_rate"] == 1.0
    assert report["arms"]["4"]["no_memory"]["proven_memory_absence_rate"] == 1.0
    effect = report["comparisons"]["4_vs_1"]["memory_dependent_gain"]
    assert effect["estimand"] == "paired_difference_in_differences"
    assert effect["estimate"] == effect["ci95"]["lower"] == 1.0
    assert report["raw"]["outcomes_included"]
    assert len(report["raw"]["outcomes"]) == 10


@pytest.mark.asyncio
async def test_compiler_rejects_incomplete_or_unequal_budget_cells() -> None:
    raw, workload, _ = await _raw()
    manifest = _manifest_dict(workload)
    incomplete = copy.deepcopy(raw)
    incomplete["records"].pop()
    with pytest.raises(CausalScalingError, match="incomplete"):
        build_causal_scaling_report(incomplete, workload_manifest=manifest)

    unequal = copy.deepcopy(raw)
    unequal["records"][0]["budget"]["total"]["model_tokens"] = 101
    with pytest.raises(CausalScalingError, match="budget drift"):
        build_causal_scaling_report(unequal, workload_manifest=manifest)


@pytest.mark.asyncio
async def test_compiler_recomputes_activation_citation_evidence() -> None:
    raw, workload, _ = await _raw()
    memory_record = next(record for record in raw["records"] if record["memory_enabled"])
    completion = next(
        event
        for event in memory_record["runtime_event_envelope"]["events"]
        if event["event_type"] == "task.completed"
    )
    completion["payload"]["memory_ids"] = []

    with pytest.raises(CausalScalingError, match="matching activation"):
        build_causal_scaling_report(raw, workload_manifest=_manifest_dict(workload))


@pytest.mark.asyncio
async def test_typed_usage_failure_cannot_compile() -> None:
    raw, workload, _ = await _raw(executor=_Executor(over_budget=True))
    failures = [failure for record in raw["records"] for failure in record["failures"]]
    assert {failure["stage"] for failure in failures} == {"usage_validation"}
    with pytest.raises(CausalScalingError, match="usage_validation"):
        build_causal_scaling_report(raw, workload_manifest=_manifest_dict(workload))


@pytest.mark.asyncio
async def test_smoke_fake_and_unreviewed_workload_are_non_comparable() -> None:
    smoke, workload, _ = await _raw(provenance=_provenance(ExecutionKind.SMOKE_FAKE))
    with pytest.raises(CausalScalingError, match="measured_external"):
        build_causal_scaling_report(smoke, workload_manifest=_manifest_dict(workload))

    raw, _, _ = await _raw()
    draft = replace(workload, review_status="draft")
    with pytest.raises(CausalScalingError, match="separately reviewed"):
        build_causal_scaling_report(raw, workload_manifest=_manifest_dict(draft))


@pytest.mark.asyncio
async def test_workload_hidden_verifier_digest_is_pinned_separately() -> None:
    raw, workload, _ = await _raw()
    manifest = _manifest_dict(workload)
    manifest["tasks"][0]["hidden_verifier_sha256"] = "0" * 64
    with pytest.raises(CausalScalingError, match="workload reference"):
        build_causal_scaling_report(raw, workload_manifest=manifest)


def test_sota_gate_stays_blocked_until_reviewed_workload_digest_is_pinned() -> None:
    manifest = json.loads(Path("benchmarks/sota/manifest.json").read_text(encoding="utf-8"))
    gate = next(item for item in manifest["gates"] if item["id"] == "multi_agent_causal_scaling")
    checks = {item["pointer"]: item for item in gate["checks"]}

    assert checks["/workload/manifest_sha256"]["expected"] == (
        "PIN_REVIEWED_WORKLOAD_SHA256_BEFORE_RUNNING"
    )
    assert all(item["operator"] != "exists" for item in gate["checks"])


def _manifest_dict(workload: WorkloadManifest) -> dict[str, Any]:
    return {
        "schema": workload.schema,
        "workload_id": workload.workload_id,
        "workload_revision": workload.workload_revision,
        "source": workload.source,
        "source_revision": workload.source_revision,
        "verifier_schema": workload.verifier_schema,
        "review_status": workload.review_status,
        "review_revision": workload.review_revision,
        "task_count": workload.task_count,
        "cluster_count": workload.cluster_count,
        "tasks": [
            {
                "task_id": entry.task_id,
                "cluster_id": entry.cluster_id,
                "public_task_sha256": entry.public_task_sha256,
                "hidden_verifier_sha256": entry.hidden_verifier_sha256,
                "task_fingerprint": entry.task_fingerprint,
            }
            for entry in workload.tasks
        ],
    }
