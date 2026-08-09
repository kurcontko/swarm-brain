"""The scripted swarm demo passes end to end over the in-memory HTTP app."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.demo import DemoRunner, build_scenario
from swarmbrain.demo.scenario import WORKER_CAPABILITIES
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


@pytest.mark.asyncio
async def test_demo_runs_every_beat_and_produces_evidence() -> None:
    clock = _Clock()
    runtime = build_in_memory_runtime("demo-test-secret", clock=clock)
    assert runtime.coordination_store is not None
    app = create_app(runtime)

    async def expire(expires_at: datetime) -> None:
        clock.advance_past(expires_at)

    transport = httpx.ASGITransport(app=app)
    scenario = build_scenario()
    async with httpx.AsyncClient(transport=transport, base_url="http://demo-test") as client:
        runner = DemoRunner(
            client=client,
            tokens=runtime.tokens,
            store=runtime.coordination_store,
            scenario=scenario,
            expire_leases=expire,
            now=clock,
        )
        report = runner_report = await runner.run()

    failed = [
        f"{beat.key}:{check.name}: {check.detail}"
        for beat in runner_report.beats
        for check in beat.checks
        if not check.ok
    ]
    assert report.ok, f"failed checks: {failed}"
    assert [beat.key for beat in report.beats] == [
        "join",
        "claim_race",
        "shared_discovery",
        "supersession_poisoning",
        "complete_replay",
        "crash_handoff",
        "dag_unblock",
        "telemetry",
    ]
    assert report.metrics["tasks_total"] == 4
    assert report.metrics["tasks_completed"] == 4
    assert report.metrics["crash_handoffs"] >= 1
    assert report.metrics["duplicate_mutations_replayed"] >= 1
    assert report.metrics["memories_cited"] >= 6
    assert report.metrics["cross_agent_memory_uses"] >= 4

    # The artifact serializes cleanly and keeps the cross-vendor story visible.
    payload = report.to_json()
    assert "crash_handoff" in payload
    assert report.scenario["mode"] == "four_agent_causal"
    assert len(report.scenario["agents"]) == 4
    handoff = next(beat for beat in report.beats if beat.key == "crash_handoff")
    assert handoff.data["crashed"]["provider"] != handoff.data["successor"]["provider"]
    causal = handoff.data["causal_verification"]
    assert causal["kind"] == "measured_deterministic_context_ablation"
    assert causal["without_memory"]["passed"] is False
    assert causal["without_memory"]["delivered_memory_ids"] == []
    assert causal["with_memory"]["passed"] is True
    assert set(causal["with_memory"]["accepted_memory_ids"]) == set(
        causal["with_memory"]["cited_memory_ids"]
    )
    assert set(causal["resumed_memory_ids"]) == set(causal["with_memory"]["accepted_memory_ids"])

    stale = next(beat for beat in report.beats if beat.key == "supersession_poisoning")
    telemetry = next(beat for beat in report.beats if beat.key == "telemetry")
    assert stale.data["wrong_memory_id"] not in telemetry.data["cited_memory_ids"]
    assert stale.data["poison_memory_id"] is None

    # Opaque values are present only in source/memory state, never leaked by the report.
    assert scenario.challenge is not None
    assert scenario.challenge.guard_token not in payload
    assert scenario.challenge.procedure_token not in payload


def test_scenario_roster_is_heterogeneous_and_least_privilege() -> None:
    scenario = build_scenario()
    assert len(scenario.all_agents) == 4
    assert len(scenario.racers) == 4
    assert len({agent.agent_id for agent in scenario.all_agents}) == 4
    assert len({agent.provider for agent in scenario.all_agents}) == 4
    assert scenario.scout.capabilities == WORKER_CAPABILITIES
    assert "memory:confirm" in WORKER_CAPABILITIES
    for key in ("implement-replay-guard", "verify-replay-guard"):
        assert scenario.task(key).depends_on == ("replay-invariant", "replay-procedure")
    assert scenario.challenge is not None
    assert len(scenario.challenge.guard_token) == 32
    assert len(scenario.challenge.procedure_token) == 32
    assert len(scenario.challenge.expected_sha256) == 64
