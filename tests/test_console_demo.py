"""The one-click demo trigger is off unless an operator turns it on.

It takes no body, so it can only ever create a fresh scenario run; it hands
back a read-only credential for that run and nothing else; and it refuses to
start a second scenario on top of the first.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import httpx
import pytest

from conftest import make_actor
from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.config import ApiSettings, BackendKind
from swarmbrain.transports.http import create_app
from swarmbrain.transports.http.app import DEMO_VIEWER_CAPABILITIES

pytestmark = pytest.mark.asyncio

SECRET = "console-demo-secret-0123456789ab"


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://console-demo-test",
    )


async def _stop(app: Any) -> None:
    task = app.state.console_demo_gate.task
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def test_trigger_is_absent_unless_the_operator_enables_it() -> None:
    app = create_app(build_in_memory_runtime(SECRET))

    async with _client(app) as client:
        assert (await client.post("/console/demo")).status_code == 404
        assert (await client.get("/console/demo/status")).status_code == 404
        # The console page itself stays reachable either way.
        assert (await client.get("/console")).status_code == 200


async def test_the_setting_defaults_to_disabled_and_only_the_exact_flag_enables_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SWARMBRAIN_BACKEND", "memory")
    monkeypatch.setenv("SWARMBRAIN_TOKEN_SECRET", SECRET)
    monkeypatch.delenv("SWARMBRAIN_CONSOLE_DEMO", raising=False)

    assert ApiSettings.from_env().console_demo_enabled is False
    assert ApiSettings.from_env().backend is BackendKind.MEMORY

    for value in ("", "true", "1", "yes", "Enable", "disabled"):
        monkeypatch.setenv("SWARMBRAIN_CONSOLE_DEMO", value)
        assert ApiSettings.from_env().console_demo_enabled is False, value

    monkeypatch.setenv("SWARMBRAIN_CONSOLE_DEMO", "ENABLED")
    assert ApiSettings.from_env().console_demo_enabled is True


async def test_trigger_creates_a_fresh_run_and_returns_a_read_only_viewer_token() -> None:
    runtime = build_in_memory_runtime(SECRET)
    app = create_app(runtime, console_demo=True)

    try:
        async with _client(app) as client:
            status = await client.get("/console/demo/status")
            assert status.status_code == 200
            assert status.json()["enabled"] is True

            # A caller-supplied run_id cannot steer the trigger: it takes no body.
            response = await client.post(
                "/console/demo",
                json={"run_id": "00000000-0000-0000-0000-000000000000"},
            )
            assert response.status_code == 202
            body = response.json()
            assert body["run_id"] != "00000000-0000-0000-0000-000000000000"
            assert body["expires_in"] == 1800

            viewer = runtime.tokens.verify(body["viewer_token"])
            assert viewer.run_id == body["run_id"]
            assert viewer.capabilities == DEMO_VIEWER_CAPABILITIES

            second = await client.post("/console/demo")
            assert second.status_code == 429
            assert second.json()["error"]["code"] in {"demo_already_running", "demo_cooldown"}
    finally:
        await _stop(app)


async def test_the_viewer_token_can_read_its_run_and_mutate_nothing() -> None:
    runtime = build_in_memory_runtime(SECRET)
    app = create_app(runtime, console_demo=True)

    try:
        async with _client(app) as client:
            body = (await client.post("/console/demo")).json()
            run_id = body["run_id"]
            auth = {"Authorization": f"Bearer {body['viewer_token']}"}

            assert (await client.get(f"/v1/runs/{run_id}/events", headers=auth)).status_code == 200
            assert (await client.get(f"/v1/runs/{run_id}/metrics", headers=auth)).status_code == 200

            mutations = (
                ("/v1/tasks:claim", {"lease_seconds": 60}),
                ("/v1/memories", {"kind": "observation", "content": "should never land"}),
                (
                    f"/v1/tasks/{run_id}:complete",
                    {
                        "lease_id": run_id,
                        "expected_task_version": 1,
                        "expected_lease_version": 1,
                        "summary": "should never land",
                    },
                ),
            )
            for path, payload in mutations:
                denied = await client.post(
                    path,
                    json=payload,
                    headers={**auth, "Idempotency-Key": "viewer-should-not-write"},
                )
                assert denied.status_code == 403, path
                assert denied.json()["error"]["code"] == "forbidden", path

            # Scoped to its own run: another run is simply not there.
            other = await client.get(
                "/v1/runs/00000000-0000-0000-0000-000000000000/events", headers=auth
            )
            assert other.status_code == 404
    finally:
        await _stop(app)


async def test_enabling_the_trigger_does_not_weaken_any_existing_route(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    app = create_app(runtime, console_demo=True)
    token = runtime.tokens.issue(make_actor(scope_ids))
    run_id = scope_ids["run_id"]

    async with _client(app) as client:
        for path in (
            f"/v1/runs/{run_id}/events",
            f"/v1/runs/{run_id}/metrics",
            "/v1/memories/00000000-0000-0000-0000-000000000000/lineage",
        ):
            anonymous = await client.get(path)
            assert anonymous.status_code == 401
            assert anonymous.headers["WWW-Authenticate"] == "Bearer"

        anonymous_write = await client.post(
            "/v1/tasks:claim", json={}, headers={"Idempotency-Key": "anon"}
        )
        assert anonymous_write.status_code == 401

        headers = {"Authorization": f"Bearer {token}"}
        assert (await client.get(f"/v1/runs/{run_id}/events", headers=headers)).status_code == 200


async def test_cooldown_blocks_a_restart_even_after_the_first_scenario_stops() -> None:
    app = create_app(build_in_memory_runtime(SECRET), console_demo=True)

    try:
        async with _client(app) as client:
            assert (await client.post("/console/demo")).status_code == 202
            await _stop(app)

            blocked = await client.post("/console/demo")
            assert blocked.status_code == 429
            assert blocked.json()["error"]["code"] == "demo_cooldown"
            assert blocked.json()["error"]["details"]["cooldown_seconds"] > 0
            assert blocked.json()["error"]["retryable"] is True
    finally:
        await _stop(app)
