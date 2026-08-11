"""Fail-closed compiler for the canonical causal-scaling SOTA artifact."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import (
    AGENT_COUNTS,
    AgentBudget,
    AgentUsage,
    CausalScalingError,
    EvaluationProvenance,
    ExecutionKind,
    ExecutionResult,
    ModelProfile,
    PublicTask,
    RolloutRequest,
    RuntimeEventEnvelope,
    TotalBudget,
    canonical_json,
    sha256_json,
)
from .evidence import project_memory_use
from .runner import (
    CONDITIONS,
    MEMORY_CONDITION,
    NO_MEMORY_CONDITION,
    RUNNER_VERSION,
    SCHEDULE_VERSION,
    split_budget,
)
from .workload import REVIEWED_STATUS, WorkloadManifest, load_workload_manifest


def build_causal_scaling_report(
    raw_evidence: dict[str, Any] | str | Path,
    *,
    workload_manifest: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate raw measured rollouts and derive a deterministic report.

    Smoke/fake evidence is intentionally rejected here.  It remains useful for
    exercising the runner, but it cannot create the canonical SOTA artifact.
    """

    raw = _load(raw_evidence)
    pinned_workload = load_workload_manifest(workload_manifest)
    _require(
        pinned_workload.review_status == REVIEWED_STATUS,
        "canonical report requires a separately reviewed workload manifest",
    )
    _require(raw.get("schema_version") == 1, "raw evidence schema_version must be 1")
    execution = _object(raw.get("execution"), "execution")
    provenance_raw = _object(execution.get("provenance"), "execution.provenance")
    provenance = _parse_provenance(provenance_raw)
    _require(
        provenance.execution_kind is ExecutionKind.MEASURED_EXTERNAL and provenance.comparable,
        "canonical report requires measured_external comparable executions",
    )
    provenance_sha = _sha(execution.get("provenance_sha256"), "provenance_sha256")
    _require(provenance_sha == provenance.digest, "execution provenance digest mismatch")

    protocol = _object(raw.get("protocol"), "protocol")
    protocol_without_digest = dict(protocol)
    config_sha = _sha(protocol_without_digest.pop("config_sha256", None), "config_sha256")
    _require(config_sha == sha256_json(protocol_without_digest), "protocol config digest mismatch")
    _validate_protocol(protocol)
    seeds = tuple(_integer(seed, "rollout seed") for seed in _list(protocol["seeds"], "seeds"))
    total_budget = TotalBudget(**_object(protocol["total_budget_per_rollout"], "total budget"))
    bootstrap = _object(protocol["bootstrap"], "bootstrap")

    workload = _object(raw.get("workload"), "workload")
    _validate_workload_reference(workload, pinned_workload)

    task_set = _object(raw.get("task_set"), "task_set")
    tasks = _validate_task_projection(task_set)
    expected_workload_tasks = [
        {
            "task_id": entry.task_id,
            "cluster_id": entry.cluster_id,
            "fingerprint": entry.task_fingerprint,
        }
        for entry in pinned_workload.tasks
    ]
    _require(
        tasks == expected_workload_tasks,
        "raw task projection differs from the separately pinned workload",
    )
    expected_schedule = _expected_schedule(tasks, seeds)
    schedule = _object(raw.get("schedule"), "schedule")
    rollout_keys = _list(schedule.get("rollout_keys"), "schedule rollout_keys")
    _require(rollout_keys == expected_schedule, "stored schedule is not the deterministic schedule")
    _require(
        schedule.get("record_count") == len(expected_schedule),
        "schedule record_count is incomplete",
    )
    _require(
        schedule.get("sha256") == sha256_json(expected_schedule),
        "schedule digest mismatch",
    )

    records = _list(raw.get("records"), "records")
    _require(len(records) == len(expected_schedule), "raw record set is incomplete")
    task_by_id = {str(task["task_id"]): task for task in tasks}
    profiles = provenance.model_profiles
    validated: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    seen_keys: set[str] = set()
    for index, value in enumerate(records):
        record = _object(value, f"record {index}")
        validated_record = _validate_record(
            record,
            index=index,
            expected_key=expected_schedule[index],
            task_by_id=task_by_id,
            seeds=seeds,
            total_budget=total_budget,
            profiles=profiles,
            provenance_sha=provenance_sha,
        )
        run_id = validated_record["run_id"]
        rollout_key = validated_record["rollout_key"]
        _require(run_id not in seen_runs, "each factorial cell requires an isolated run_id")
        _require(rollout_key not in seen_keys, "raw evidence contains duplicate rollout keys")
        seen_runs.add(run_id)
        seen_keys.add(rollout_key)
        validated.append(validated_record)

    expected_cells = {
        (str(task["task_id"]), seed, agent_count, condition)
        for task in tasks
        for seed in seeds
        for agent_count in AGENT_COUNTS
        for condition in CONDITIONS
    }
    observed_cells = {
        (item["task_id"], item["seed"], item["agent_count"], item["condition"])
        for item in validated
    }
    _require(observed_cells == expected_cells, "paired factorial arms are incomplete")

    outcomes = _raw_outcomes(validated, tasks=tasks, seeds=seeds)
    arms = {
        str(agent_count): _arm_summary(validated, agent_count=agent_count)
        for agent_count in AGENT_COUNTS
    }
    bootstrap_seed = _integer(bootstrap["seed"], "bootstrap seed")
    resamples = _integer(bootstrap["resamples"], "bootstrap resamples")
    confidence = _number(bootstrap["confidence"], "bootstrap confidence")
    comparisons = {
        "4_vs_1": _comparison(
            validated,
            larger=4,
            smaller=1,
            resamples=resamples,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
        ),
        "4_vs_2": _comparison(
            validated,
            larger=4,
            smaller=2,
            resamples=resamples,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
        ),
    }
    raw_sha = sha256_json(raw)
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": {
            "name": "Swarm Brain 1/2/4-agent causal scaling",
            "runner_version": RUNNER_VERSION,
        },
        "design": {
            "paired": True,
            "counterfactual_conditions": list(CONDITIONS),
            "agent_counts": list(AGENT_COUNTS),
            "equal_total_model_budget": True,
            "equal_total_tool_budget": True,
            "budget_split_not_multiplied": True,
            "model_token_budget_per_rollout": total_budget.model_tokens,
            "tool_call_budget_per_rollout": total_budget.tool_calls,
            "randomized_rollouts": len(seeds),
            "deterministic_randomization": True,
            "schedule_version": SCHEDULE_VERSION,
            "isolated_run_per_cell": True,
        },
        "provenance": {
            "fixed": True,
            "sha256": provenance_sha,
            "protocol_config_sha256": config_sha,
            "task_set_sha256": task_set["sha256"],
            "tool_catalog_sha256": provenance.tool_catalog_sha256,
            "execution_kind": provenance.execution_kind.value,
            "comparable": provenance.comparable,
            "details": provenance_raw,
        },
        "workload": {
            **workload,
            "task_count": len(tasks),
            "task_cluster_count": task_set["cluster_count"],
            "separately_pinned": True,
            "hidden_verifier_digests_pinned": True,
        },
        "raw": {
            "evidence_sha256": raw_sha,
            "task_count": len(tasks),
            "task_cluster_count": task_set["cluster_count"],
            "seed_count": len(seeds),
            "record_count": len(validated),
            "outcomes_included": True,
            "outcomes_sha256": sha256_json(outcomes),
            "outcomes": outcomes,
        },
        "arms": arms,
        "comparisons": comparisons,
        "bootstrap": {
            "unit": "task_cluster",
            "paired": True,
            "resamples": resamples,
            "seed": bootstrap_seed,
            "confidence": confidence,
        },
        "validation": {
            "all_factorial_cells_complete": True,
            "counterfactuals_complete": True,
            "isolated_run_ids": True,
            "equal_budget_allocations": True,
            "model_identity_matches_provenance": True,
            "event_proofs_recomputed": True,
            "memory_activation_and_citation_proven": True,
            "no_memory_absence_proven": True,
        },
        "failures": {
            "total": 0,
            "incomplete_cells": 0,
            "duplicate_cells": 0,
            "budget_violations": 0,
            "provenance_violations": 0,
            "memory_evidence_violations": 0,
            "unscored_rollouts": 0,
        },
        "estimands": {
            "memory_dependent_gain": (
                "[(memory success - no-memory success) at the larger team size] - "
                "[(memory success - no-memory success) at the smaller team size], "
                "paired on task_id and seed"
            ),
            "task_success_delta": (
                "memory-enabled larger-team success minus memory-enabled smaller-team "
                "success; diagnostic only, not the causal memory claim"
            ),
        },
        "limitations": [
            "This protocol identifies gain attributable to memory under the tested tasks, "
            "models, tools, budgets, and rollout seeds; it does not prove universal scaling.",
            "Activation plus citation proves delivered-memory use, not that every cited token "
            "was necessary for the model's internal reasoning.",
            "Cluster-bootstrap intervals cover the supplied task clusters, not deployment drift.",
        ],
    }
    canonical_json(report)
    return report


def _validate_protocol(protocol: dict[str, Any]) -> None:
    _require(protocol.get("runner_version") == RUNNER_VERSION, "runner version mismatch")
    _require(protocol.get("agent_counts") == list(AGENT_COUNTS), "agent arms must be 1/2/4")
    _require(protocol.get("conditions") == list(CONDITIONS), "counterfactual arms are missing")
    for key in ("paired", "isolated_run_per_cell", "randomized"):
        _require(protocol.get(key) is True, f"protocol {key} must be true")
    _require(protocol.get("schedule_version") == SCHEDULE_VERSION, "schedule version mismatch")
    seeds = _list(protocol.get("seeds"), "seeds")
    _require(len(seeds) >= 5 and len(set(seeds)) == len(seeds), "five unique seeds are required")
    bootstrap = _object(protocol.get("bootstrap"), "bootstrap")
    _require(bootstrap.get("unit") == "task_cluster", "bootstrap unit must be task_cluster")
    _require(bootstrap.get("paired") is True, "bootstrap must preserve paired cells")
    _require(
        _integer(bootstrap.get("resamples"), "bootstrap resamples") >= 10_000,
        "bootstrap requires >= 10,000 resamples",
    )
    _require(
        _number(bootstrap.get("confidence"), "bootstrap confidence") == 0.95,
        "canonical report requires a 95% interval",
    )


def _validate_task_projection(task_set: dict[str, Any]) -> list[dict[str, Any]]:
    tasks_raw = _list(task_set.get("tasks"), "task_set.tasks")
    tasks = [_object(task, "task projection") for task in tasks_raw]
    _require(len(tasks) >= 2, "at least two tasks are required")
    ids: list[str] = []
    clusters: set[str] = set()
    for task in tasks:
        task_id = _string(task.get("task_id"), "task_id")
        cluster = _string(task.get("cluster_id"), "cluster_id")
        _sha(task.get("fingerprint"), "task fingerprint")
        ids.append(task_id)
        clusters.add(cluster)
    _require(len(ids) == len(set(ids)), "task IDs must be unique")
    _require(len(clusters) >= 2, "at least two task clusters are required")
    _require(task_set.get("task_count") == len(tasks), "task_count mismatch")
    _require(task_set.get("cluster_count") == len(clusters), "cluster_count mismatch")
    _require(task_set.get("sha256") == sha256_json(tasks), "task-set digest mismatch")
    _require(ids == sorted(ids), "task projection must be ordered by task_id")
    return tasks


def _validate_workload_reference(raw: dict[str, Any], manifest: WorkloadManifest) -> None:
    expected = {
        "schema": manifest.schema,
        "workload_id": manifest.workload_id,
        "workload_revision": manifest.workload_revision,
        "source": manifest.source,
        "source_revision": manifest.source_revision,
        "verifier_schema": manifest.verifier_schema,
        "review_status": manifest.review_status,
        "review_revision": manifest.review_revision,
        "task_count": manifest.task_count,
        "cluster_count": manifest.cluster_count,
        "manifest_sha256": manifest.digest,
    }
    _require(raw == expected, "raw evidence workload reference does not match its manifest")


def _expected_schedule(tasks: list[dict[str, Any]], seeds: tuple[int, ...]) -> list[str]:
    expected: list[str] = []
    for seed in seeds:
        cells = [
            (task, agent_count, condition == MEMORY_CONDITION)
            for task in tasks
            for agent_count in AGENT_COUNTS
            for condition in CONDITIONS
        ]
        random.Random(seed).shuffle(cells)
        expected.extend(
            sha256_json(
                {
                    "schedule_version": SCHEDULE_VERSION,
                    "seed": seed,
                    "task_id": task["task_id"],
                    "task_fingerprint": task["fingerprint"],
                    "agent_count": agent_count,
                    "memory_enabled": memory_enabled,
                }
            )
            for task, agent_count, memory_enabled in cells
        )
    return expected


def _validate_record(
    record: dict[str, Any],
    *,
    index: int,
    expected_key: str,
    task_by_id: dict[str, dict[str, Any]],
    seeds: tuple[int, ...],
    total_budget: TotalBudget,
    profiles: tuple[ModelProfile, ...],
    provenance_sha: str,
) -> dict[str, Any]:
    rollout_key = _sha(record.get("rollout_key"), "rollout_key")
    _require(rollout_key == expected_key, "record does not match deterministic schedule")
    _require(record.get("schedule_index") == index, "record schedule_index mismatch")
    task_id = _string(record.get("task_id"), "record task_id")
    _require(task_id in task_by_id, "record references an unknown task")
    task = task_by_id[task_id]
    _require(record.get("task_cluster_id") == task["cluster_id"], "task cluster drift")
    _require(record.get("task_fingerprint") == task["fingerprint"], "task instance drift")
    seed = _integer(record.get("seed"), "record seed")
    _require(seed in seeds, "record seed is not configured")
    agent_count = _integer(record.get("agent_count"), "record agent_count")
    _require(agent_count in AGENT_COUNTS, "record agent_count is not 1/2/4")
    condition = _string(record.get("condition"), "record condition")
    _require(condition in CONDITIONS, "record condition is invalid")
    memory_enabled = record.get("memory_enabled")
    _require(isinstance(memory_enabled, bool), "memory_enabled must be boolean")
    _require(memory_enabled == (condition == MEMORY_CONDITION), "condition flag mismatch")
    _require(record.get("provenance_sha256") == provenance_sha, "record provenance drift")

    budget = _object(record.get("budget"), "record budget")
    _require(budget.get("total") == asdict(total_budget), "rollout total budget drift")
    expected_allocations = split_budget(total_budget, agent_count)
    _require(
        budget.get("allocations") == [asdict(item) for item in expected_allocations],
        "rollout budget was multiplied or split incorrectly",
    )
    failures = _list(record.get("failures"), "record failures")
    if failures:
        first = _object(failures[0], "failure")
        stage = first.get("stage", "unknown")
        error_type = first.get("error_type", "unknown")
        raise CausalScalingError(f"rollout {rollout_key} failed at {stage}: {error_type}")

    execution_raw = _object(record.get("execution"), "record execution")
    run_id = _string(execution_raw.get("run_id"), "execution run_id")
    raw_output = execution_raw.get("raw_output")
    _require(isinstance(raw_output, str), "execution raw_output must be preserved")
    _require(
        execution_raw.get("raw_output_sha256") == sha256_json(raw_output),
        "raw output digest mismatch",
    )
    usage_raw = _list(execution_raw.get("agent_usage"), "agent_usage")
    usages = tuple(AgentUsage(**_object(item, "agent usage")) for item in usage_raw)
    _validate_usage(
        usages,
        expected_allocations=expected_allocations,
        profiles=profiles[:agent_count],
        total_budget=total_budget,
    )
    model_tokens = sum(usage.model_tokens for usage in usages)
    tool_calls = sum(usage.tool_calls for usage in usages)
    _require(execution_raw.get("model_tokens_used") == model_tokens, "token total mismatch")
    _require(execution_raw.get("tool_calls_used") == tool_calls, "tool-call total mismatch")
    canonical_json(_object(execution_raw.get("metadata"), "execution metadata"))
    execution = ExecutionResult(
        run_id=run_id,
        raw_output=raw_output,
        agent_usage=usages,
        metadata=execution_raw["metadata"],
    )

    outcome = _object(record.get("outcome"), "record outcome")
    success = outcome.get("success")
    _require(isinstance(success, bool), "outcome success must be boolean")
    _number(outcome.get("score"), "outcome score", minimum=0.0, maximum=1.0)
    canonical_json(_object(outcome.get("details"), "outcome details"))

    envelope_raw = _object(record.get("runtime_event_envelope"), "runtime event envelope")
    envelope = RuntimeEventEnvelope(
        source=_string(envelope_raw.get("source"), "event source"),
        run_id=_string(envelope_raw.get("run_id"), "event run_id"),
        complete=envelope_raw.get("complete"),
        page_count=_integer(envelope_raw.get("page_count"), "event page_count"),
        events=tuple(
            _object(item, "runtime event") for item in _list(envelope_raw.get("events"), "events")
        ),
    )
    request = RolloutRequest(
        rollout_key=rollout_key,
        schedule_index=index,
        seed=seed,
        agent_count=agent_count,
        memory_enabled=memory_enabled,
        task=PublicTask(task_id=task_id, prompt="redacted from report compiler"),
        total_budget=total_budget,
        agent_budgets=expected_allocations,
        model_profiles=profiles[:agent_count],
    )
    recomputed_proof = project_memory_use(envelope, request=request, result=execution)
    proof_raw = _object(record.get("memory_use_proof"), "memory use proof")
    _require(
        canonical_json(proof_raw) == canonical_json(asdict(recomputed_proof)),
        "stored memory-use proof is not reproducible",
    )

    return {
        "rollout_key": rollout_key,
        "task_id": task_id,
        "cluster_id": task["cluster_id"],
        "seed": seed,
        "agent_count": agent_count,
        "condition": condition,
        "success": int(success),
        "score": float(outcome["score"]),
        "run_id": run_id,
        "model_tokens_used": model_tokens,
        "tool_calls_used": tool_calls,
        "matched_activation_citations": recomputed_proof.matched_activation_citations,
        "memory_absence_proven": recomputed_proof.memory_absence_proven,
        "event_stream_sha256": recomputed_proof.event_stream_sha256,
        "raw_output_sha256": execution_raw["raw_output_sha256"],
    }


def _validate_usage(
    usages: tuple[AgentUsage, ...],
    *,
    expected_allocations: tuple[AgentBudget, ...],
    profiles: tuple[ModelProfile, ...],
    total_budget: TotalBudget,
) -> None:
    _require(len(usages) == len(expected_allocations), "usage missing an agent slot")
    _require(
        tuple(usage.agent_slot for usage in usages) == tuple(range(1, len(usages) + 1)),
        "usage must be ordered and complete by agent slot",
    )
    for usage, allocation, profile in zip(usages, expected_allocations, profiles, strict=True):
        _require(usage.agent_slot == allocation.agent_slot == profile.agent_slot, "slot drift")
        _require(
            (usage.provider, usage.model, usage.revision)
            == (profile.provider, profile.model, profile.revision),
            "observed model identity differs from pinned provenance",
        )
        _require(
            usage.model_tokens <= allocation.model_tokens
            and usage.tool_calls <= allocation.tool_calls,
            "agent exceeded its allocated budget share",
        )
    _require(
        sum(usage.model_tokens for usage in usages) <= total_budget.model_tokens,
        "rollout exceeded total model-token budget",
    )
    _require(
        sum(usage.tool_calls for usage in usages) <= total_budget.tool_calls,
        "rollout exceeded total tool-call budget",
    )


def _raw_outcomes(
    records: list[dict[str, Any]],
    *,
    tasks: list[dict[str, Any]],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    indexed = {
        (item["task_id"], item["seed"], item["agent_count"], item["condition"]): item
        for item in records
    }
    outcomes: list[dict[str, Any]] = []
    for task in tasks:
        for seed in seeds:
            cells: dict[str, Any] = {}
            for agent_count in AGENT_COUNTS:
                for condition in CONDITIONS:
                    item = indexed[(task["task_id"], seed, agent_count, condition)]
                    cells[f"{agent_count}_{condition}"] = {
                        "success": item["success"],
                        "score": item["score"],
                        "model_tokens_used": item["model_tokens_used"],
                        "tool_calls_used": item["tool_calls_used"],
                        "matched_activation_citations": item["matched_activation_citations"],
                        "memory_absence_proven": item["memory_absence_proven"],
                        "run_id": item["run_id"],
                        "raw_output_sha256": item["raw_output_sha256"],
                        "event_stream_sha256": item["event_stream_sha256"],
                    }
            outcomes.append(
                {
                    "task_id": task["task_id"],
                    "task_cluster_id": task["cluster_id"],
                    "seed": seed,
                    "cells": cells,
                }
            )
    return outcomes


def _arm_summary(records: list[dict[str, Any]], *, agent_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {"agent_count": agent_count}
    for condition in CONDITIONS:
        selected = [
            item
            for item in records
            if item["agent_count"] == agent_count and item["condition"] == condition
        ]
        result[condition] = {
            "records": len(selected),
            "task_success": _mean([item["success"] for item in selected]),
            "mean_model_tokens_used": _mean([item["model_tokens_used"] for item in selected]),
            "mean_tool_calls_used": _mean([item["tool_calls_used"] for item in selected]),
            "proven_activation_citation_rate": _mean(
                [int(item["matched_activation_citations"] > 0) for item in selected]
            ),
            "proven_memory_absence_rate": _mean(
                [int(item["memory_absence_proven"]) for item in selected]
            ),
        }
    paired = _paired_values(records, agent_count=agent_count)
    result["paired_memory_effect"] = _mean([value[1] for value in paired])
    return result


def _comparison(
    records: list[dict[str, Any]],
    *,
    larger: int,
    smaller: int,
    resamples: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    indexed = {
        (item["task_id"], item["seed"], item["agent_count"], item["condition"]): item
        for item in records
    }
    pairs: list[tuple[str, float, float]] = []
    task_seeds = sorted({(item["task_id"], item["seed"]) for item in records})
    for task_id, seed in task_seeds:
        memory_larger = indexed[(task_id, seed, larger, MEMORY_CONDITION)]
        no_memory_larger = indexed[(task_id, seed, larger, NO_MEMORY_CONDITION)]
        memory_smaller = indexed[(task_id, seed, smaller, MEMORY_CONDITION)]
        no_memory_smaller = indexed[(task_id, seed, smaller, NO_MEMORY_CONDITION)]
        cluster = memory_larger["cluster_id"]
        _require(
            {
                no_memory_larger["cluster_id"],
                memory_smaller["cluster_id"],
                no_memory_smaller["cluster_id"],
            }
            == {cluster},
            "paired record cluster mismatch",
        )
        did = (memory_larger["success"] - no_memory_larger["success"]) - (
            memory_smaller["success"] - no_memory_smaller["success"]
        )
        raw_delta = memory_larger["success"] - memory_smaller["success"]
        pairs.append((cluster, float(did), float(raw_delta)))
    did_values = [(cluster, did) for cluster, did, _raw in pairs]
    raw_values = [(cluster, raw_delta) for cluster, _did, raw_delta in pairs]
    did_interval = _cluster_bootstrap(
        did_values,
        resamples=resamples,
        seed=_derived_seed(bootstrap_seed, f"{larger}_vs_{smaller}_did"),
        confidence=confidence,
    )
    raw_interval = _cluster_bootstrap(
        raw_values,
        resamples=resamples,
        seed=_derived_seed(bootstrap_seed, f"{larger}_vs_{smaller}_raw"),
        confidence=confidence,
    )
    return {
        "larger_agent_count": larger,
        "smaller_agent_count": smaller,
        "paired_observations": len(pairs),
        "memory_dependent_gain": {
            "estimand": "paired_difference_in_differences",
            "formula": (
                f"(memory_{larger} - no_memory_{larger}) - (memory_{smaller} - no_memory_{smaller})"
            ),
            "estimate": _mean([value for _cluster, value in did_values]),
            "ci95": did_interval,
        },
        "task_success_delta": {
            "estimand": "memory_arm_success_difference_diagnostic",
            "formula": f"memory_{larger} - memory_{smaller}",
            "estimate": _mean([value for _cluster, value in raw_values]),
            "ci95": raw_interval,
        },
    }


def _paired_values(records: list[dict[str, Any]], *, agent_count: int) -> list[tuple[str, float]]:
    indexed = {
        (item["task_id"], item["seed"], item["condition"]): item
        for item in records
        if item["agent_count"] == agent_count
    }
    return [
        (
            memory["cluster_id"],
            float(memory["success"] - indexed[(task_id, seed, NO_MEMORY_CONDITION)]["success"]),
        )
        for task_id, seed, condition in sorted(indexed)
        if condition == MEMORY_CONDITION
        for memory in (indexed[(task_id, seed, MEMORY_CONDITION)],)
    ]


def _cluster_bootstrap(
    values: list[tuple[str, float]],
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> dict[str, float]:
    by_cluster: dict[str, list[float]] = {}
    for cluster, value in values:
        by_cluster.setdefault(cluster, []).append(value)
    clusters = sorted(by_cluster)
    _require(len(clusters) >= 2, "cluster bootstrap requires at least two clusters")
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        drawn = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        observations = [value for cluster in drawn for value in by_cluster[cluster]]
        samples.append(_mean(observations))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "lower": _quantile(samples, alpha),
        "upper": _quantile(samples, 1.0 - alpha),
    }


def _quantile(values: list[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _derived_seed(seed: int, label: str) -> int:
    return int(sha256_json({"seed": seed, "label": label})[:16], 16)


def _parse_provenance(raw: dict[str, Any]) -> EvaluationProvenance:
    try:
        profiles = tuple(
            ModelProfile(**_object(item, "model profile"))
            for item in _list(raw.get("model_profiles"), "model_profiles")
        )
        provenance = EvaluationProvenance(
            execution_kind=ExecutionKind(raw.get("execution_kind")),
            comparable=raw.get("comparable"),
            code_revision=raw.get("code_revision"),
            adapter_name=raw.get("adapter_name"),
            adapter_revision=raw.get("adapter_revision"),
            environment_digest=raw.get("environment_digest"),
            tool_catalog_sha256=raw.get("tool_catalog_sha256"),
            tool_runtime_revision=raw.get("tool_runtime_revision"),
            model_profiles=profiles,
        )
    except (TypeError, ValueError) as exc:
        raise CausalScalingError("execution provenance is invalid") from exc
    _require(
        canonical_json(raw) == canonical_json(asdict(provenance)),
        "execution provenance has unknown or coerced fields",
    )
    return provenance


def _load(value: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, dict):
        # Exercise the same JSON boundary as the standalone compiler even when
        # a caller passes the runner payload directly in a test or notebook.
        return json.loads(canonical_json(value))
    path = Path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CausalScalingError(f"cannot read raw causal evidence: {type(exc).__name__}") from exc
    return _object(payload, "raw evidence")


def _mean(values: list[int | float]) -> float:
    _require(bool(values), "cannot aggregate an empty cell")
    return sum(values) / len(values)


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CausalScalingError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise CausalScalingError(f"{field} must be a list")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CausalScalingError(f"{field} must be a non-empty string")
    return value


def _sha(value: Any, field: str) -> str:
    result = _string(value, field)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise CausalScalingError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CausalScalingError(f"{field} must be an integer")
    return value


def _number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise CausalScalingError(f"{field} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise CausalScalingError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise CausalScalingError(f"{field} must be <= {maximum}")
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CausalScalingError(message)


__all__ = ["build_causal_scaling_report"]
