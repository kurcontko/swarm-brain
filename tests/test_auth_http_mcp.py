from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from conftest import make_actor, make_task
from swarmbrain.adapters.auth import ExpiredTokenError, InvalidTokenError, RunTokenCodec
from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.config import BridgeSettings
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    EvidenceKind,
    RegisterEvidenceSourceCommand,
)
from swarmbrain.transports.http import create_app
from swarmbrain.transports.mcp.client import SwarmBrainHttpClient
from swarmbrain.transports.mcp.server import create_server

SECRET = "0123456789abcdef-local-test-secret"


def test_run_token_is_signed_scoped_and_short_lived(scope_ids: dict[str, str]) -> None:
    actor = make_actor(scope_ids)
    codec = RunTokenCodec(SECRET)
    now = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)
    token = codec.issue(actor, ttl=timedelta(minutes=5), now=now)

    verified = codec.verify(token, now=now + timedelta(minutes=1))
    assert verified.agent_id == actor.agent_id
    assert verified.run_id == actor.run_id
    assert verified.token_id is not None
    assert verified.expires_at == now + timedelta(minutes=5)

    with pytest.raises(ExpiredTokenError):
        codec.verify(token, now=now + timedelta(minutes=6))
    prefix, payload, signature = token.split(".")
    replacement = "A" if payload[-1] != "A" else "B"
    tampered = f"{prefix}.{payload[:-1]}{replacement}.{signature}"
    with pytest.raises(InvalidTokenError):
        codec.verify(tampered, now=now)


@pytest.mark.asyncio
async def test_http_auth_scope_strict_contract_and_idempotency_replay(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(scope_ids)
    task = make_task(scope_ids)
    await runtime.kernel.add_task(task)
    token = runtime.tokens.issue(actor)
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.post(
            "/v1/tasks:claim",
            json={},
            headers={"Idempotency-Key": "missing-auth"},
        )
        assert missing.status_code == 401
        assert missing.json()["error"]["code"] == "authentication_required"

        forged = await client.post(
            "/v1/tasks:claim",
            json={"agent_id": "00000000-0000-0000-0000-000000000000"},
            headers={
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": "forged-actor",
            },
        )
        assert forged.status_code == 422
        assert forged.json()["error"]["code"] == "validation_error"

        headers = {
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": "stable-claim",
        }
        first = await client.post("/v1/tasks:claim", json={}, headers=headers)
        second = await client.post("/v1/tasks:claim", json={}, headers=headers)

    assert first.status_code == 200
    assert first.json()["lease"]["owner_agent_id"] == actor.agent_id
    assert first.json()["task"]["task_id"] == task.task_id
    assert second.status_code == 200
    assert second.json()["lease"]["lease_id"] == first.json()["lease"]["lease_id"]
    assert second.json()["replayed"] is True
    assert second.headers["Idempotency-Replayed"] == "true"


@pytest.mark.asyncio
async def test_full_in_memory_vertical_slice_through_canonical_http(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(scope_ids)
    first_task = make_task(scope_ids, title="Fix serializer")
    second_task = make_task(scope_ids, title="Review documentation")
    await runtime.kernel.add_task(first_task)
    await runtime.kernel.add_task(second_task)
    source = await runtime.kernel.register_source(
        actor,
        RegisterEvidenceSourceCommand(
            idempotency_key="http-source",
            kind=EvidenceKind.SOURCE_CODE,
            content_sha256="c" * 64,
            occurrence_key="http:file.py",
            observed_at=datetime.now(UTC),
        ),
    )
    evidence = await runtime.kernel.add_evidence(
        actor,
        AddEvidenceCommand(
            idempotency_key="http-evidence",
            source_id=source.source_id,
            kind=EvidenceKind.SOURCE_CODE,
            locator="file.py:12",
            excerpt="PORT = 26257",
        ),
    )
    token = runtime.tokens.issue(actor)
    auth = {"Authorization": f"Bearer {token}"}
    transport = httpx.ASGITransport(app=create_app(runtime))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        joined = await client.post(f"/v1/runs/{actor.run_id}/agents:join", json={}, headers=auth)
        assert joined.status_code == 200

        claimed = await client.post(
            "/v1/tasks:claim",
            json={"task_id": first_task.task_id, "lease_seconds": 120},
            headers={**auth, "Idempotency-Key": "http-claim"},
        )
        assert claimed.status_code == 200
        claim = claimed.json()

        renewed = await client.post(
            f"/v1/leases/{claim['lease']['lease_id']}:renew",
            json={"expected_version": claim["lease"]["version"], "extension_seconds": 120},
            headers={**auth, "Idempotency-Key": "http-renew"},
        )
        assert renewed.status_code == 200, renewed.text
        lease_version = renewed.json()["lease"]["version"]

        wrong = await client.post(
            "/v1/memories",
            json={
                "kind": "hypothesis",
                "content": "The SQL port is 5432",
                "visibility": "repository",
            },
            headers={**auth, "Idempotency-Key": "http-wrong-memory"},
        )
        assert wrong.status_code == 200
        wrong_id = wrong.json()["memory"]["memory_id"]
        corrected = await client.post(
            "/v1/memories",
            json={
                "kind": "invariant",
                "content": "The CockroachDB SQL port is 26257",
                "desired_state": "confirmed",
                "visibility": "repository",
                "evidence": [evidence.as_ref().model_dump(mode="json")],
                "supersedes_memory_id": wrong_id,
            },
            headers={**auth, "Idempotency-Key": "http-correct-memory"},
        )
        assert corrected.status_code == 200
        correct_id = corrected.json()["memory"]["memory_id"]

        recalled = await client.post(
            "/v1/memories:recall",
            json={"text": "SQL port", "include_evidence": True},
            headers=auth,
        )
        assert [hit["memory"]["memory_id"] for hit in recalled.json()["hits"]] == [correct_id]
        lineage = await client.get(f"/v1/memories/{correct_id}/lineage", headers=auth)
        assert len(lineage.json()["memories"]) == 2

        checkpointed = await client.post(
            f"/v1/tasks/{first_task.task_id}/checkpoints",
            json={
                "lease_id": claim["lease"]["lease_id"],
                "expected_task_version": claim["task"]["version"],
                "expected_lease_version": lease_version,
                "summary": "serializer patched",
                "discoveries": ["wrong port documented"],
                "completed_work": ["patched serializer"],
                "remaining_work": ["run full suite"],
            },
            headers={**auth, "Idempotency-Key": "http-checkpoint"},
        )
        assert checkpointed.status_code == 200
        checkpoint = checkpointed.json()

        conflict = await client.post(
            "/v1/conflicts",
            json={
                "memory_ids": [wrong_id, correct_id],
                "description": "incompatible port values",
                "evidence": [evidence.as_ref().model_dump(mode="json")],
            },
            headers={**auth, "Idempotency-Key": "http-conflict"},
        )
        assert conflict.status_code == 200
        conflict_body = conflict.json()["conflict"]
        resolution = await client.post(
            f"/v1/conflicts/{conflict_body['conflict_id']}:resolve",
            json={
                "expected_version": conflict_body["version"],
                "resolution": {
                    "kind": "prefer_newer",
                    "rationale": "source code is authoritative",
                    "evidence": [evidence.as_ref().model_dump(mode="json")],
                    "supported_memory_ids": [correct_id],
                    "refuted_memory_ids": [wrong_id],
                },
            },
            headers={**auth, "Idempotency-Key": "http-resolve"},
        )
        assert resolution.status_code == 200
        assert resolution.json()["conflict"]["status"] == "resolved"

        completed = await client.post(
            f"/v1/tasks/{first_task.task_id}:complete",
            json={
                "lease_id": claim["lease"]["lease_id"],
                "expected_task_version": checkpoint["task"]["version"],
                "expected_lease_version": checkpoint["lease"]["version"],
                "summary": "all focused tests pass",
                "memory_ids": [correct_id],
            },
            headers={**auth, "Idempotency-Key": "http-complete"},
        )
        assert completed.status_code == 200

        second_claim = await client.post(
            "/v1/tasks:claim",
            json={"task_id": second_task.task_id},
            headers={**auth, "Idempotency-Key": "http-second-claim"},
        )
        second = second_claim.json()
        released = await client.post(
            f"/v1/tasks/{second_task.task_id}:release",
            json={
                "lease_id": second["lease"]["lease_id"],
                "expected_task_version": second["task"]["version"],
                "expected_lease_version": second["lease"]["version"],
                "reason": "return to queue",
            },
            headers={**auth, "Idempotency-Key": "http-release"},
        )
        assert released.status_code == 200
        assert released.json()["task"]["status"] == "ready"

        events = await client.get(f"/v1/runs/{actor.run_id}/events", headers=auth)
        metrics = await client.get(f"/v1/runs/{actor.run_id}/metrics", headers=auth)
        assert events.status_code == metrics.status_code == 200
        assert metrics.json()["tasks_completed"] == 1
        assert metrics.json()["conflicts_resolved"] == 1


def test_http_openapi_uses_typed_domain_contracts(scope_ids: dict[str, str]) -> None:
    schema = create_app(build_in_memory_runtime(SECRET)).openapi()
    claim = schema["paths"]["/v1/tasks:claim"]["post"]
    assert claim["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ClaimTaskBody"
    )
    assert claim["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ClaimTaskResult"
    )
    body = schema["components"]["schemas"]["ClaimTaskBody"]
    assert "agent_id" not in body["properties"]
    assert "tenant_id" not in body["properties"]


@pytest.mark.asyncio
async def test_mcp_registers_exactly_six_thin_scope_safe_tools() -> None:
    server = create_server(BridgeSettings("http://example.test", "token"))
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == {
        "claim_task",
        "recall_memory",
        "publish_memory",
        "checkpoint_task",
        "complete_task",
        "report_conflict",
    }
    forbidden = {
        "tenant_id",
        "project_id",
        "repository_id",
        "repo_id",
        "run_id",
        "agent_id",
        "provider",
        "model",
        "capabilities",
        "token",
    }
    for tool in tools:
        assert forbidden.isdisjoint(tool.inputSchema["properties"])


@pytest.mark.asyncio
async def test_mcp_http_client_uses_canonical_fields_header_and_version_fencing() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/tasks:claim":
            return httpx.Response(
                200,
                json={
                    "task": {"task_id": "task", "version": 2},
                    "lease": {"lease_id": "lease", "version": 1},
                },
            )
        if request.url.path == "/v1/tasks/task/checkpoints":
            return httpx.Response(
                200,
                json={
                    "task": {"task_id": "task", "version": 3},
                    "lease": {"lease_id": "lease", "version": 2},
                    "checkpoint": {},
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    raw = httpx.AsyncClient(
        base_url="http://example.test",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer signed-token"},
    )
    bridge = SwarmBrainHttpClient(BridgeSettings("http://example.test", "signed-token"), client=raw)
    try:
        await bridge.claim_task(
            task_id=None,
            required_tags=["python"],
            lease_seconds=120,
            idempotency_key="claim-key",
        )
        await bridge.checkpoint_task(
            task_id="task",
            summary="handoff",
            discoveries=["found race"],
            completed_work=["reproducer"],
            remaining_work=["patch"],
            idempotency_key="checkpoint-key",
        )
    finally:
        await bridge.close()
        await raw.aclose()

    claim_payload = json.loads(requests[0].content)
    checkpoint_payload = json.loads(requests[1].content)
    assert requests[0].headers["Idempotency-Key"] == "claim-key"
    assert requests[0].headers["Authorization"] == "Bearer signed-token"
    assert claim_payload["lease_seconds"] == 120
    assert "lease_ttl_seconds" not in claim_payload
    assert checkpoint_payload["expected_task_version"] == 2
    assert checkpoint_payload["expected_lease_version"] == 1
    assert checkpoint_payload["completed_work"] == ["reproducer"]
    assert "agent_id" not in checkpoint_payload
    assert bridge._renewals == {}


@pytest.mark.asyncio
async def test_bridge_renews_lease_without_a_model_visible_tool() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/tasks:claim":
            return httpx.Response(
                200,
                json={
                    "task": {"task_id": "task", "version": 2},
                    "lease": {"lease_id": "lease", "version": 1},
                },
            )
        if request.url.path == "/v1/leases/lease:renew":
            return httpx.Response(200, json={"lease": {"lease_id": "lease", "version": 2}})
        raise AssertionError(f"unexpected request: {request.url}")

    sleeps = 0

    async def one_tick(_delay: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps > 1:
            raise asyncio.CancelledError

    raw = httpx.AsyncClient(
        base_url="http://example.test",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer signed-token"},
    )
    bridge = SwarmBrainHttpClient(
        BridgeSettings("http://example.test", "signed-token"),
        client=raw,
        sleep=one_tick,
    )
    try:
        await bridge.claim_task(idempotency_key="claim-key")
        renewal = bridge._renewals["task"]
        with pytest.raises(asyncio.CancelledError):
            await renewal
        assert bridge._leases["task"].lease_version == 2
    finally:
        await bridge.close()
        await raw.aclose()

    renewals = [request for request in requests if request.url.path.endswith(":renew")]
    assert len(renewals) == 1
    assert json.loads(renewals[0].content) == {
        "expected_version": 1,
        "extension_seconds": 120,
    }
