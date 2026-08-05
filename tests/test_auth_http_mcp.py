from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest

from conftest import make_actor, make_task
from swarmbrain.adapters.auth import ExpiredTokenError, InvalidTokenError, RunTokenCodec
from swarmbrain.adapters.extraction import (
    STRUCTURED_MEMORY_SOURCE_KIND,
    DefaultRuleExtractor,
    InMemoryWorkStore,
)
from swarmbrain.application.extraction import ExtractionService
from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.config import BridgeSettings
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    EvidenceKind,
    RegisterEvidenceSourceCommand,
)
from swarmbrain.domain.memory import RememberCommand, Visibility
from swarmbrain.transports.http import create_app
from swarmbrain.transports.mcp.client import BridgeHttpError, SwarmBrainHttpClient
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
async def test_task_claim_capability_does_not_leak_initial_memory(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    publisher = make_actor(scope_ids)
    actor = make_actor(scope_ids, capabilities=frozenset({"task:claim"}))
    task = make_task(scope_ids)
    await runtime.kernel.add_task(task)
    await runtime.memory.publish(
        publisher,
        RememberCommand(
            idempotency_key="claim-private-memory",
            kind="decision",
            content="repository secret must not leak through task claim",
            visibility=Visibility.REPOSITORY,
        ),
    )
    transport = httpx.ASGITransport(app=create_app(runtime))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        claimed = await client.post(
            "/v1/tasks:claim",
            json={"task_id": task.task_id},
            headers={
                "Authorization": f"Bearer {runtime.tokens.issue(actor)}",
                "Idempotency-Key": "claim-only-capability",
            },
        )

    assert claimed.status_code == 200, claimed.text
    assert claimed.json()["task"]["status"] == "claimed"
    assert claimed.json()["memory"] is None


@pytest.mark.asyncio
async def test_initial_memory_failure_does_not_hide_committed_claim(
    scope_ids: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(
        scope_ids,
        capabilities=frozenset({"task:claim", "memory:recall"}),
    )
    task = make_task(scope_ids)
    await runtime.kernel.add_task(task)

    async def fail_recall(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("recall unavailable")

    assert runtime.memory.retrieval is not None
    monkeypatch.setattr(runtime.memory.retrieval, "execute", fail_recall)
    transport = httpx.ASGITransport(app=create_app(runtime))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        claimed = await client.post(
            "/v1/tasks:claim",
            json={"task_id": task.task_id},
            headers={
                "Authorization": f"Bearer {runtime.tokens.issue(actor)}",
                "Idempotency-Key": "claim-recall-outage",
            },
        )

    assert claimed.status_code == 200, claimed.text
    assert claimed.json()["memory"] is None
    persisted = runtime.kernel.tasks[task.task_id]
    assert persisted.status.value == "claimed"
    assert persisted.active_lease_id == claimed.json()["lease"]["lease_id"]


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

        structured = await client.post(
            "/v1/memories",
            json={
                "kind": "org.acme/preference",
                "content": {
                    "subject": "editor",
                    "preference": {"name": "vim", "strength": 0.8},
                },
            },
            headers={**auth, "Idempotency-Key": "http-structured-memory"},
        )
        assert structured.status_code == 200, structured.text
        assert structured.json()["memory"]["kind"] == "org.acme/preference"
        assert structured.json()["memory"]["content"]["preference"]["name"] == "vim"

        structured_recall = await client.post(
            "/v1/memories:recall",
            json={"text": "vim", "kinds": ["org.acme/preference"]},
            headers=auth,
        )
        assert structured_recall.status_code == 200
        assert structured_recall.json()["hits"][0]["memory"]["content"] == {
            "subject": "editor",
            "preference": {"name": "vim", "strength": 0.8},
        }

        recalled = await client.post(
            "/v1/memories:recall",
            json={"text": "SQL port", "kinds": ["invariant"], "include_evidence": True},
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
    ingest_ref = schema["paths"]["/v1/sources:ingest"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    ingest_body = schema["components"]["schemas"][ingest_ref.rsplit("/", 1)[-1]]
    assert {"trust", "use_provider", "chunk_chars", "priority"}.isdisjoint(
        ingest_body["properties"]
    )


@pytest.mark.asyncio
async def test_http_ingestion_receipt_status_scope_and_replay(
    scope_ids: dict[str, str],
) -> None:
    base = build_in_memory_runtime(SECRET)
    store = InMemoryWorkStore()
    extraction = ExtractionService(store, DefaultRuleExtractor())
    runtime = replace(
        base,
        extraction=extraction,
        source_store=store,
        work_queue=store,
    )
    actor = make_actor(scope_ids)
    auth = {"Authorization": f"Bearer {runtime.tokens.issue(actor)}"}
    payload = {
        "kind": STRUCTURED_MEMORY_SOURCE_KIND,
        "content": json.dumps(
            {
                "memories": [
                    {
                        "kind": "org.acme/preference",
                        "content": {"theme": "dark", "contexts": ["editor"]},
                    }
                ]
            }
        ),
        "metadata": {"channel": "test"},
    }
    transport = httpx.ASGITransport(app=create_app(runtime))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {**auth, "Idempotency-Key": "http-ingest-1"}
        first = await client.post("/v1/sources:ingest", json=payload, headers=headers)
        replay = await client.post("/v1/sources:ingest", json=payload, headers=headers)
        changed = await client.post(
            "/v1/sources:ingest",
            json={**payload, "content": payload["content"] + " "},
            headers=headers,
        )
        forged_policy = await client.post(
            "/v1/sources:ingest",
            json={**payload, "trust": "trusted"},
            headers={**auth, "Idempotency-Key": "http-ingest-forged"},
        )
        reader = make_actor(scope_ids, capabilities=frozenset({"memory:recall"}))
        denied = await client.post(
            "/v1/sources:ingest",
            json=payload,
            headers={
                "Authorization": f"Bearer {runtime.tokens.issue(reader)}",
                "Idempotency-Key": "http-ingest-denied",
            },
        )

        assert first.status_code == replay.status_code == 202
        receipt = first.json()
        assert receipt["chunk_count"] == 1
        assert "chunks" not in receipt
        assert receipt["source"]["trust"] == "unknown"
        assert replay.headers["Idempotency-Replayed"] == "true"
        assert replay.json()["work_id"] == receipt["work_id"]
        assert changed.status_code == 409
        assert forged_policy.status_code == 422
        assert denied.status_code == 403

        source_id = receipt["source"]["source_id"]
        queued = await client.get(f"/v1/sources/{source_id}/extraction", headers=auth)
        assert queued.status_code == 200
        assert queued.json()["status"] == "queued"
        assert {
            "lease_token",
            "locked_by",
            "payload",
            "provider",
            "model",
        }.isdisjoint(queued.json())

        applied = await runtime.extraction_worker().run_once("http-worker")
        assert len(applied) == 1
        completed = await client.get(f"/v1/sources/{source_id}/extraction", headers=auth)
        assert completed.json()["status"] == "completed"
        assert completed.json()["candidate_count"] == 1
        assert len(completed.json()["memory_ids"]) == 1

        foreign_scope = dict(scope_ids)
        foreign_scope["run_id"] = str(uuid4())
        foreign = make_actor(foreign_scope)
        foreign_auth = {"Authorization": f"Bearer {runtime.tokens.issue(foreign)}"}
        hidden = await client.get(
            f"/v1/sources/{source_id}/extraction",
            headers=foreign_auth,
        )
        malformed = await client.get("/v1/sources/not-a-uuid/extraction", headers=auth)
        assert hidden.status_code == 404
        assert malformed.status_code == 422


@pytest.mark.asyncio
async def test_memory_backend_fails_closed_for_durable_ingestion(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(scope_ids)
    transport = httpx.ASGITransport(app=create_app(runtime))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/sources:ingest",
            json={"kind": "document", "content": "not durable"},
            headers={
                "Authorization": f"Bearer {runtime.tokens.issue(actor)}",
                "Idempotency-Key": "memory-ingest-denied",
            },
        )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_inline_evidence_is_restart_idempotent_and_cannot_self_assign_trust(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(
        scope_ids,
        capabilities=frozenset({"memory:publish", "memory:recall"}),
    )
    token = runtime.tokens.issue(actor)
    app = create_app(runtime)
    schema = app.openapi()
    register_ref = schema["paths"]["/v1/evidence/sources"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    register_body = schema["components"]["schemas"][register_ref.rsplit("/", 1)[-1]]
    assert "trust" not in register_body["properties"]

    async def publish_once() -> dict[str, object]:
        raw = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {token}"},
        )
        bridge = SwarmBrainHttpClient(
            BridgeSettings("http://test", token),
            client=raw,
        )
        try:
            return await bridge.publish_memory(
                content={"invariant": "serialization retries only use 40001"},
                kind="invariant",
                evidence=[
                    {
                        "kind": "source_code",
                        "locator": "src/retry.py:42",
                        "excerpt": 'if sqlstate != "40001": raise',
                    }
                ],
                idempotency_key="bridge-inline-restart",
            )
        finally:
            await bridge.close()
            await raw.aclose()

    first = await publish_once()
    replay = await publish_once()
    assert replay["memory"]["memory_id"] == first["memory"]["memory_id"]  # type: ignore[index]
    assert len(runtime.kernel.sources) == 1
    assert len(runtime.kernel.evidence) == 1

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        forged = await client.post(
            "/v1/evidence/sources",
            json={
                "kind": "document",
                "content_sha256": "a" * 64,
                "observed_at": datetime.now(UTC).isoformat(),
                "trust": "trusted",
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": "forged-evidence-trust",
            },
        )
    assert forged.status_code == 422


@pytest.mark.asyncio
async def test_inline_evidence_cache_rejects_changed_content_for_parent_key() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/evidence/sources":
            return httpx.Response(200, json={"source_id": str(uuid4())})
        if request.url.path == "/v1/evidence":
            payload = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "evidence_id": str(uuid4()),
                    "source_id": payload["source_id"],
                    "locator": payload["locator"],
                    "excerpt": payload["excerpt"],
                    "content_sha256": payload["content_sha256"],
                    "metadata": payload["metadata"],
                },
            )
        if request.url.path == "/v1/memories":
            return httpx.Response(200, json={"memory": {"memory_id": str(uuid4())}})
        raise AssertionError(f"unexpected request: {request.url}")

    raw = httpx.AsyncClient(
        base_url="http://example.test",
        transport=httpx.MockTransport(handler),
    )
    bridge = SwarmBrainHttpClient(
        BridgeSettings("http://example.test", "signed-token"),
        client=raw,
    )
    evidence = {
        "kind": "source_code",
        "locator": "src/retry.py:42",
        "excerpt": 'if sqlstate != "40001": raise',
    }
    try:
        await bridge.publish_memory(
            content="retry invariant",
            evidence=[evidence],
            idempotency_key="same-parent",
        )
        await bridge.publish_memory(
            content="retry invariant",
            evidence=[dict(reversed(tuple(evidence.items())))],
            idempotency_key="same-parent",
        )
        before_conflict = len(requests)
        with pytest.raises(ValueError, match="different inline evidence"):
            await bridge.publish_memory(
                content="retry invariant",
                evidence=[{**evidence, "excerpt": "changed evidence"}],
                idempotency_key="same-parent",
            )
        assert len(requests) == before_conflict
    finally:
        await bridge.close()
        await raw.aclose()

    assert [request.url.path for request in requests].count("/v1/evidence/sources") == 1
    assert [request.url.path for request in requests].count("/v1/evidence") == 1
    assert [request.url.path for request in requests].count("/v1/memories") == 2


@pytest.mark.asyncio
async def test_inline_evidence_namespace_separates_parent_operations() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/evidence/sources":
            return httpx.Response(200, json={"source_id": str(uuid4())})
        if request.url.path == "/v1/evidence":
            payload = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "evidence_id": str(uuid4()),
                    "source_id": payload["source_id"],
                    "locator": payload["locator"],
                    "excerpt": payload["excerpt"],
                    "content_sha256": payload["content_sha256"],
                    "metadata": payload["metadata"],
                },
            )
        if request.url.path == "/v1/memories":
            return httpx.Response(200, json={"memory": {"memory_id": str(uuid4())}})
        if request.url.path == "/v1/conflicts":
            return httpx.Response(200, json={"conflict": {"conflict_id": str(uuid4())}})
        raise AssertionError(f"unexpected request: {request.url}")

    raw = httpx.AsyncClient(
        base_url="http://example.test",
        transport=httpx.MockTransport(handler),
    )
    bridge = SwarmBrainHttpClient(
        BridgeSettings(
            "http://example.test",
            "signed-token",
            expected_run_id="run-a",
            expected_agent_id="agent-a",
        ),
        client=raw,
    )
    try:
        await bridge.publish_memory(
            content="memory",
            evidence=[{"kind": "message", "excerpt": "memory evidence"}],
            idempotency_key="shared-parent-key",
        )
        await bridge.report_conflict(
            memory_ids=[str(uuid4()), str(uuid4())],
            description="conflict",
            evidence=[{"kind": "message", "excerpt": "conflict evidence"}],
            idempotency_key="shared-parent-key",
        )
    finally:
        await bridge.close()
        await raw.aclose()

    assert [request.url.path for request in requests].count("/v1/evidence/sources") == 2
    assert [request.url.path for request in requests].count("/v1/evidence") == 2
    child_keys = [
        request.headers["Idempotency-Key"]
        for request in requests
        if request.url.path == "/v1/evidence"
    ]
    assert len(set(child_keys)) == 2


@pytest.mark.asyncio
async def test_inline_evidence_namespace_separates_agents() -> None:
    occurrences: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if request.url.path == "/v1/evidence/sources":
            occurrences.append(payload["occurrence_key"])
            return httpx.Response(200, json={"source_id": str(uuid4())})
        if request.url.path == "/v1/evidence":
            return httpx.Response(
                200,
                json={
                    "evidence_id": str(uuid4()),
                    "source_id": payload["source_id"],
                    "locator": payload["locator"],
                    "excerpt": payload["excerpt"],
                    "content_sha256": payload["content_sha256"],
                    "metadata": payload["metadata"],
                },
            )
        if request.url.path == "/v1/memories":
            return httpx.Response(200, json={"memory": {"memory_id": str(uuid4())}})
        raise AssertionError(f"unexpected request: {request.url}")

    transport = httpx.MockTransport(handler)
    clients: list[tuple[SwarmBrainHttpClient, httpx.AsyncClient]] = []
    for agent_id in ("agent-a", "agent-b"):
        raw = httpx.AsyncClient(base_url="http://example.test", transport=transport)
        clients.append(
            (
                SwarmBrainHttpClient(
                    BridgeSettings(
                        "http://example.test",
                        "signed-token",
                        expected_run_id="shared-run",
                        expected_agent_id=agent_id,
                    ),
                    client=raw,
                ),
                raw,
            )
        )
    try:
        for position, (bridge, _raw) in enumerate(clients):
            await bridge.publish_memory(
                content=f"memory-{position}",
                evidence=[
                    {
                        "kind": "message",
                        "excerpt": f"evidence-{position}",
                    }
                ],
                idempotency_key="same-agent-local-key",
            )
    finally:
        for bridge, raw in clients:
            await bridge.close()
            await raw.aclose()

    assert len(occurrences) == 2
    assert len(set(occurrences)) == 2


@pytest.mark.asyncio
async def test_mcp_registers_exactly_seven_thin_scope_safe_tools() -> None:
    server = create_server(BridgeSettings("http://example.test", "token"))
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == {
        "claim_task",
        "recall_memory",
        "publish_memory",
        "ingest_memory_source",
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
    publish = next(tool for tool in tools if tool.name == "publish_memory")
    content_schema = publish.inputSchema["properties"]["content"]
    assert content_schema == {"$ref": "#/$defs/MemoryContent"}
    memory_content_schema = json.dumps(publish.inputSchema["$defs"]["MemoryContent"])
    assert '"string"' in memory_content_schema
    assert '"object"' in memory_content_schema
    assert '"array"' in memory_content_schema
    assert '"null"' not in memory_content_schema
    ingest = next(tool for tool in tools if tool.name == "ingest_memory_source")
    assert {"trust", "use_provider", "chunk_chars", "priority"}.isdisjoint(
        ingest.inputSchema["properties"]
    )


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


def test_bridge_expected_actor_identity_must_be_configured_as_a_pair() -> None:
    with pytest.raises(ValueError, match="configured together"):
        BridgeSettings(
            "http://example.test",
            "signed-token",
            expected_run_id="run",
        )
    with pytest.raises(ValueError, match="configured together"):
        BridgeSettings(
            "http://example.test",
            "signed-token",
            expected_agent_id="agent",
        )


@pytest.mark.asyncio
async def test_bridge_join_fails_closed_on_actor_identity_mismatch() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"run_id": "wrong-run", "agent_id": "expected-agent"},
        )

    raw = httpx.AsyncClient(
        base_url="http://example.test",
        transport=httpx.MockTransport(handler),
    )
    bridge = SwarmBrainHttpClient(
        BridgeSettings(
            "http://example.test",
            "signed-token",
            expected_run_id="expected-run",
            expected_agent_id="expected-agent",
        ),
        client=raw,
    )
    try:
        with pytest.raises(BridgeHttpError, match="expected run and agent") as caught:
            await bridge.join()
        assert caught.value.status_code == 409
        assert caught.value.code == "actor_identity_mismatch"
    finally:
        await bridge.close()
        await raw.aclose()


@pytest.mark.asyncio
async def test_bridge_renewal_retries_with_the_same_idempotency_key() -> None:
    requests: list[httpx.Request] = []
    renewal_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal renewal_attempts
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
            renewal_attempts += 1
            if renewal_attempts == 1:
                return httpx.Response(
                    503,
                    json={
                        "error": {
                            "code": "backend_unavailable",
                            "message": "API is restarting",
                            "retryable": True,
                        }
                    },
                )
            return httpx.Response(200, json={"lease": {"lease_id": "lease", "version": 2}})
        raise AssertionError(f"unexpected request: {request.url}")

    sleeps = 0

    async def retry_then_stop(_delay: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps > 2:
            raise asyncio.CancelledError

    raw = httpx.AsyncClient(
        base_url="http://example.test",
        transport=httpx.MockTransport(handler),
    )
    bridge = SwarmBrainHttpClient(
        BridgeSettings("http://example.test", "signed-token"),
        client=raw,
        sleep=retry_then_stop,
    )
    try:
        await bridge.claim_task(idempotency_key="claim-key")
        with pytest.raises(asyncio.CancelledError):
            await bridge._renewals["task"]
        assert bridge._leases["task"].lease_version == 2
    finally:
        await bridge.close()
        await raw.aclose()

    renewals = [request for request in requests if request.url.path.endswith(":renew")]
    assert len(renewals) == 2
    assert renewals[0].headers["Idempotency-Key"] == renewals[1].headers["Idempotency-Key"]
    assert json.loads(renewals[0].content) == json.loads(renewals[1].content)
