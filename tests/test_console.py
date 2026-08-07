"""The console page is public, inert, and never a way around authentication."""

from __future__ import annotations

import httpx
import pytest

from conftest import make_actor
from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.transports.http import create_app
from swarmbrain.transports.http.console import console_html

SECRET = "console-test-secret-0123456789ab"


@pytest.mark.asyncio
async def test_console_page_is_served_unauthenticated_and_self_contained() -> None:
    app = create_app(build_in_memory_runtime(SECRET))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://console-test") as client:
        response = await client.get("/console")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert body.startswith("<!doctype html>")
    assert "swarmbrain" in body
    # Offline-capable: no external origin is referenced by the document.
    assert "http://" not in body
    assert "https://" not in body
    assert "//cdn" not in body
    # The page holds no secret, no token, and no run data.
    assert SECRET not in body
    assert "Bearer ey" not in body
    assert "sessionStorage" in body


@pytest.mark.asyncio
async def test_console_does_not_relax_data_route_authentication(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(scope_ids)
    token = runtime.tokens.issue(actor)
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)
    run_id = scope_ids["run_id"]

    async with httpx.AsyncClient(transport=transport, base_url="http://console-test") as client:
        for path in (
            f"/v1/runs/{run_id}/events",
            f"/v1/runs/{run_id}/metrics",
            "/v1/memories/00000000-0000-0000-0000-000000000000/lineage",
        ):
            anonymous = await client.get(path)
            assert anonymous.status_code == 401
            assert anonymous.json()["error"]["code"] == "authentication_required"
            assert anonymous.headers["WWW-Authenticate"] == "Bearer"

        headers = {"Authorization": f"Bearer {token}"}
        events = await client.get(f"/v1/runs/{run_id}/events", headers=headers)
        assert events.status_code == 200
        assert events.json()["events"] == []
        metrics = await client.get(f"/v1/runs/{run_id}/metrics", headers=headers)
        assert metrics.status_code == 200
        assert metrics.json()["run_id"] == run_id

        # The console route stays a page, never a data endpoint.
        other_run = await client.get(
            "/v1/runs/00000000-0000-0000-0000-000000000000/events", headers=headers
        )
        assert other_run.status_code == 404


def test_console_document_reads_only_the_canonical_routes() -> None:
    body = console_html()

    assert "/v1/runs/${encodeURIComponent(state.runId)}/events" in body
    assert "/v1/runs/${encodeURIComponent(state.runId)}/metrics" in body
    assert "/v1/memories/${encodeURIComponent(memoryId)}/lineage" in body
    assert "events:read" in body and "metrics:read" in body and "memory:recall" in body

    # Read-only against the data plane: the only POST the page can issue is the
    # operator-gated demo trigger, and it never carries the viewer's bearer token.
    api_call = body.index("async function api(path)")
    assert 'method: "POST"' not in body[api_call : body.index("async function pullEvents")]
    posts = body.count('method: "POST"')
    assert posts == 1
    assert "`${state.base}/console/demo`" in body
    assert "/v1/" not in body[body.index("async function triggerDemo") :]


def test_console_prefers_enriched_payload_labels_with_uuid_fallback() -> None:
    body = console_html()

    # The board, the roster, and the ledger all read the additive payload keys.
    assert "task_title" in body
    assert "agent_display" in body
    assert "agent_harness" in body
    # ...and every label degrades to the short id when an event predates them.
    assert "state.agentLabels.get(agentId)) || short(agentId)" in body
    assert "esc(short(task.taskId))" in body
