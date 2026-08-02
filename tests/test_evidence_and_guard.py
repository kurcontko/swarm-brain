"""Evidence registration surface and the tiered supersession guard."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from conftest import make_actor
from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.config import BridgeSettings
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    EvidenceKind,
    RegisterEvidenceSourceCommand,
)
from swarmbrain.domain.memory import MemoryOperation, MemoryState, RememberCommand
from swarmbrain.transports.http import create_app
from swarmbrain.transports.mcp.client import SwarmBrainHttpClient

SECRET = "0123456789abcdef-local-test-secret"
PUBLISH_ONLY = frozenset({"run:join", "memory:publish", "memory:recall"})


@pytest.mark.asyncio
async def test_http_evidence_registration_and_citation(scope_ids: dict[str, str]) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(scope_ids, capabilities=PUBLISH_ONLY)
    token = runtime.tokens.issue(actor)
    transport = httpx.ASGITransport(app=create_app(runtime))

    observed_at = datetime.now(UTC).isoformat()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"Authorization": f"Bearer {token}"}
        source = await client.post(
            "/v1/evidence/sources",
            json={
                "kind": "test_result",
                "content_sha256": "a" * 64,
                "observed_at": observed_at,
                "uri": "pytest://tests/test_serializer.py",
            },
            headers={**headers, "Idempotency-Key": "register-src-00001"},
        )
        assert source.status_code == 200, source.text
        source_id = source.json()["source_id"]

        evidence = await client.post(
            "/v1/evidence",
            json={
                "source_id": source_id,
                "kind": "test_result",
                "locator": "tests/test_serializer.py::test_roundtrip",
                "excerpt": "82 passed in 0.7s",
                "content_sha256": "a" * 64,
            },
            headers={**headers, "Idempotency-Key": "add-evidence-00001"},
        )
        assert evidence.status_code == 200, evidence.text
        row = evidence.json()

        published = await client.post(
            "/v1/memories",
            json={
                "kind": "outcome",
                "content": "serializer roundtrip is green after the fix",
                "evidence": [
                    {
                        "evidence_id": row["evidence_id"],
                        "source_id": row["source_id"],
                        "excerpt": row["excerpt"],
                        "content_sha256": row["content_sha256"],
                    }
                ],
            },
            headers={**headers, "Idempotency-Key": "publish-with-ev-1"},
        )
        assert published.status_code == 200, published.text
        memory = published.json()["memory"]
        assert memory["evidence"][0]["evidence_id"] == row["evidence_id"]

        replay = await client.post(
            "/v1/evidence/sources",
            json={
                "kind": "test_result",
                "content_sha256": "a" * 64,
                "observed_at": observed_at,
                "uri": "pytest://tests/test_serializer.py",
            },
            headers={**headers, "Idempotency-Key": "register-src-00001"},
        )
        assert replay.status_code == 200
        assert replay.json()["source_id"] == source_id


@pytest.mark.asyncio
async def test_evidence_registration_requires_publish_capability(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    reader = make_actor(scope_ids, capabilities=frozenset({"memory:recall"}))
    token = runtime.tokens.issue(reader)
    transport = httpx.ASGITransport(app=create_app(runtime))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        denied = await client.post(
            "/v1/evidence/sources",
            json={
                "kind": "log",
                "content_sha256": "b" * 64,
                "observed_at": datetime.now(UTC).isoformat(),
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": "denied-src-00001",
            },
        )
    assert denied.status_code == 403


@pytest.mark.asyncio
async def test_bridge_registers_inline_evidence_before_publishing(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    actor = make_actor(scope_ids, capabilities=PUBLISH_ONLY)
    token = runtime.tokens.issue(actor)
    transport = httpx.ASGITransport(app=create_app(runtime))
    settings = BridgeSettings(api_url="http://test", agent_token=token)
    http = httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )
    bridge = SwarmBrainHttpClient(settings, client=http)
    try:
        result = await bridge.publish_memory(
            content="retry only serialization failures, never ambiguity",
            kind="invariant",
            evidence=[
                {
                    "kind": "source_code",
                    "locator": "src/swarmbrain/adapters/cockroach/retry.py:92",
                    "excerpt": 'if sqlstate != "40001": raise',
                }
            ],
            idempotency_key="bridge-inline-ev-1",
        )
        refs = result["memory"]["evidence"]
        assert len(refs) == 1
        assert refs[0]["evidence_id"] in runtime.kernel.evidence
        assert refs[0]["source_id"] in runtime.kernel.sources
        assert refs[0]["locator"] == "src/swarmbrain/adapters/cockroach/retry.py:92"

        replay = await bridge.publish_memory(
            content="retry only serialization failures, never ambiguity",
            kind="invariant",
            evidence=[
                {
                    "kind": "source_code",
                    "locator": "src/swarmbrain/adapters/cockroach/retry.py:92",
                    "excerpt": 'if sqlstate != "40001": raise',
                }
            ],
            idempotency_key="bridge-inline-ev-1",
        )
        assert replay["memory"]["memory_id"] == result["memory"]["memory_id"]
        assert len(runtime.kernel.evidence) == 1
        assert len(runtime.kernel.sources) == 1

        with pytest.raises(ValueError, match="requires 'kind'"):
            await bridge.publish_memory(
                content="inline material without a kind is rejected client-side",
                evidence=[{"excerpt": "no kind supplied"}],
                idempotency_key="bridge-inline-ev-2",
            )
    finally:
        await bridge.close()
        await http.aclose()


@pytest.mark.asyncio
async def test_confirmed_memory_is_protected_even_without_evidence(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(SECRET)
    verifier = make_actor(scope_ids)
    worker = make_actor(scope_ids, capabilities=PUBLISH_ONLY)

    confirmed = await runtime.memory.publish(
        verifier,
        RememberCommand(
            idempotency_key="confirmed-no-evidence",
            kind="invariant",
            content="amounts are stored as integer cents",
            desired_state=MemoryState.CONFIRMED,
        ),
    )
    assert confirmed.memory is not None

    attack = await runtime.memory.publish(
        worker,
        RememberCommand(
            idempotency_key="tentative-overwrite",
            kind="invariant",
            content="amounts are stored as floating point dollars",
            supersedes_memory_id=confirmed.memory.memory_id,
        ),
    )
    assert attack.operation is MemoryOperation.NOOP
    assert attack.memory is None
    surviving = runtime.kernel.memories[confirmed.memory.memory_id]
    assert surviving.state is MemoryState.CONFIRMED


@pytest.mark.asyncio
async def test_supersession_tiers(scope_ids: dict[str, str]) -> None:
    runtime = build_in_memory_runtime(SECRET)
    verifier = make_actor(scope_ids)
    worker = make_actor(scope_ids, capabilities=PUBLISH_ONLY)

    # Tier 1: a worker freely supersedes its own tentative memory.
    draft = await runtime.memory.publish(
        worker,
        RememberCommand(
            idempotency_key="worker-draft-00001",
            kind="hypothesis",
            content="the flake is caused by clock skew",
        ),
    )
    assert draft.memory is not None
    corrected = await runtime.memory.publish(
        worker,
        RememberCommand(
            idempotency_key="worker-fix-0000001",
            kind="hypothesis",
            content="the flake is caused by an unawaited coroutine",
            supersedes_memory_id=draft.memory.memory_id,
        ),
    )
    assert corrected.operation is MemoryOperation.UPDATE

    # Tier 2: displacing confirmed memory needs confirm capability AND evidence.
    confirmed = await runtime.memory.publish(
        verifier,
        RememberCommand(
            idempotency_key="verifier-confirmed",
            kind="invariant",
            content="the API never applies DDL at startup",
            desired_state=MemoryState.CONFIRMED,
        ),
    )
    assert confirmed.memory is not None

    source = await runtime.kernel.register_source(
        verifier,
        RegisterEvidenceSourceCommand(
            idempotency_key="verifier-source-01",
            kind=EvidenceKind.SOURCE_CODE,
            content_sha256="d" * 64,
            observed_at=datetime.now(UTC),
        ),
    )
    evidence = await runtime.kernel.add_evidence(
        verifier,
        AddEvidenceCommand(
            idempotency_key="verifier-evidence1",
            source_id=source.source_id,
            kind=EvidenceKind.SOURCE_CODE,
            content_sha256="d" * 64,
        ),
    )

    unevidenced = await runtime.memory.publish(
        verifier,
        RememberCommand(
            idempotency_key="confirmed-no-proof",
            kind="invariant",
            content="the API applies DDL when a flag is set",
            desired_state=MemoryState.CONFIRMED,
            supersedes_memory_id=confirmed.memory.memory_id,
        ),
    )
    assert unevidenced.operation is MemoryOperation.NOOP

    replacement = await runtime.memory.publish(
        verifier,
        RememberCommand(
            idempotency_key="confirmed-w-proof1",
            kind="invariant",
            content="schema install is an explicit operator command",
            desired_state=MemoryState.CONFIRMED,
            supersedes_memory_id=confirmed.memory.memory_id,
            evidence=(evidence.as_ref(),),
        ),
    )
    assert replacement.operation is MemoryOperation.UPDATE
    displaced = runtime.kernel.memories[confirmed.memory.memory_id]
    assert displaced.state is MemoryState.SUPERSEDED
