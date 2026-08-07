"""The A/B comparison measures the real run and dominates the simulated baseline."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.demo import build_scenario
from swarmbrain.demo.ab import (
    BASELINE_KIND,
    SWARM_KIND,
    WORKLOAD,
    ABReport,
    simulate_uncoordinated_baseline,
)
from swarmbrain.demo.ab import run_comparison as run_ab
from swarmbrain.transports.http import create_app


class _Clock:
    """Ticks 1 ms per read so strictly-ordered bitemporal writes become visible."""

    def __init__(self) -> None:
        self._now = datetime.now(UTC)

    def __call__(self) -> datetime:
        self._now += timedelta(milliseconds=1)
        return self._now

    def advance_past(self, moment: datetime) -> None:
        if self._now <= moment:
            self._now = moment + timedelta(seconds=1)


async def _comparison() -> ABReport:
    clock = _Clock()
    runtime = build_in_memory_runtime("demo-ab-test-secret-0123456789", clock=clock)
    assert runtime.coordination_store is not None
    app = create_app(runtime)

    async def expire(expires_at: datetime) -> None:
        clock.advance_past(expires_at)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://demo-ab") as client:
        return await run_ab(
            client=client,
            tokens=runtime.tokens,
            store=runtime.coordination_store,
            scenario=build_scenario(),
            expire_leases=expire,
            now=clock,
        )


@pytest.mark.asyncio
async def test_swarm_arm_strictly_dominates_the_uncoordinated_baseline() -> None:
    report = await _comparison()
    baseline = report.baseline
    swarm = report.swarm

    assert report.ok
    # The measured arm does each task exactly once and nothing twice.
    assert swarm.task_executions == WORKLOAD.minimum_executions
    assert swarm.duplicate_task_executions == 0
    assert swarm.investigations_performed == WORKLOAD.minimum_investigations
    assert swarm.duplicate_investigations == 0
    assert swarm.discovery_derivations == WORKLOAD.minimum_discovery_derivations
    assert swarm.discoveries_rederived == 0
    assert swarm.discovery_reuses >= 2

    # The simulated fleet duplicates every axis the swarm brain serializes.
    assert baseline.duplicate_task_executions > 0
    assert baseline.duplicate_investigations > 0
    assert baseline.discoveries_rederived > 0
    assert baseline.discovery_reuses == 0

    assert report.deltas["duplicate_task_executions_avoided"] == (
        baseline.duplicate_task_executions
    )
    assert report.deltas["duplicate_investigations_avoided"] == baseline.duplicate_investigations
    assert report.deltas["discovery_rederivations_avoided"] == baseline.discoveries_rederived
    assert report.deltas["steps_saved"] > 0
    assert report.deltas["work_multiplier"] > 1.0


@pytest.mark.asyncio
async def test_crashed_worker_keeps_its_progress_only_with_the_swarm_brain() -> None:
    report = await _comparison()
    assert report.swarm.crash_steps_preserved == WORKLOAD.crash_after_steps
    assert report.swarm.crash_steps_lost == 0
    assert report.baseline.crash_steps_preserved == 0
    assert report.baseline.crash_steps_lost == WORKLOAD.crash_after_steps
    assert report.deltas["crash_steps_preserved"] == WORKLOAD.crash_after_steps


@pytest.mark.asyncio
async def test_arm_b_numbers_are_traceable_to_the_real_run() -> None:
    report = await _comparison()
    evidence = report.evidence
    assert evidence["ok"] is True
    assert evidence["metrics"]["tasks_completed"] == 5
    # Claim arbitration refusing the losing racers is a store-recorded number.
    assert report.swarm.duplicate_executions_prevented >= 8
    assert set(report.swarm.sources) >= {
        "task_executions",
        "discovery_reuses",
        "crash_steps_preserved",
        "duplicate_executions_prevented",
    }
    beat_keys = [beat["key"] for beat in evidence["beats"]]
    assert "shared_discovery" in beat_keys and "crash_handoff" in beat_keys


@pytest.mark.asyncio
async def test_artifact_serializes_with_both_arms_labelled() -> None:
    report = await _comparison()
    payload = json.loads(report.to_json())

    assert payload["arms"]["baseline"]["kind"] == BASELINE_KIND
    assert payload["arms"]["baseline"]["simulated"] is True
    assert payload["arms"]["swarm_brain"]["kind"] == SWARM_KIND
    assert payload["arms"]["swarm_brain"]["simulated"] is False
    assert "simulated" in payload["honesty"]["arm_a"]
    assert "measured" in payload["honesty"]["arm_b"]
    assert any(note.startswith("SIMULATED.") for note in payload["arms"]["baseline"]["notes"])
    assert payload["workload"]["minimum_steps"] == WORKLOAD.minimum_steps
    assert payload["swarm_run_evidence"]["ok"] is True

    table = report.table()
    assert BASELINE_KIND in table
    assert SWARM_KIND in table
    assert "total steps" in table


def test_baseline_simulation_is_deterministic_and_per_vendor() -> None:
    scenario = build_scenario()
    first = simulate_uncoordinated_baseline(scenario)
    second = simulate_uncoordinated_baseline(build_scenario())
    assert first.to_dict() == second.to_dict()

    fleets = len({agent.provider for agent in scenario.workers})
    assert first.fleets == fleets
    # Every fleet redoes every task and re-derives the one shared discovery.
    assert first.task_executions == fleets * WORKLOAD.minimum_executions
    assert first.discovery_derivations == fleets
    assert set(first.executions_by_task.values()) == {fleets}
    assert first.total_steps > first.minimum_steps
