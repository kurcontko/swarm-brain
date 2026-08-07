"""The scripted swarm demo passes end to end over the in-memory HTTP app."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.demo import DemoRunner, build_scenario
from swarmbrain.demo.scenario import SCOUT_CAPABILITIES, WORKER_CAPABILITIES
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
    async with httpx.AsyncClient(transport=transport, base_url="http://demo-test") as client:
        runner = DemoRunner(
            client=client,
            tokens=runtime.tokens,
            store=runtime.coordination_store,
            scenario=build_scenario(),
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
    assert report.metrics["tasks_completed"] == 5
    assert report.metrics["crash_handoffs"] >= 1
    assert report.metrics["duplicate_mutations_replayed"] >= 1

    # The artifact serializes cleanly and keeps the cross-vendor story visible.
    payload = report.to_json()
    assert "crash_handoff" in payload
    handoff = next(beat for beat in report.beats if beat.key == "crash_handoff")
    assert handoff.data["crashed"]["provider"] != handoff.data["successor"]["provider"]


def test_scenario_roster_is_heterogeneous_and_least_privilege() -> None:
    scenario = build_scenario()
    assert len(scenario.racers) == 12
    assert len({agent.provider for agent in scenario.racers}) >= 4
    assert scenario.scout.capabilities == SCOUT_CAPABILITIES
    assert "task:claim" not in SCOUT_CAPABILITIES
    assert "memory:confirm" in WORKER_CAPABILITIES
    blocked = scenario.task("apply-fix")
    assert blocked.depends_on == ("duplicate-charge",)
