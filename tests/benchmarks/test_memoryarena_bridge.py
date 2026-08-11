from __future__ import annotations

import json

import httpx
import pytest
from benchmarks.integrations.memoryarena.contracts import (
    SEMANTIC_EMBEDDING_DIMENSIONS,
    SEMANTIC_EMBEDDING_MODE,
    SEMANTIC_EMBEDDING_MODEL_ID,
    SEMANTIC_EMBEDDING_PROVIDER,
    SEMANTIC_EMBEDDING_QUERY_INSTRUCTION,
    SEMANTIC_EMBEDDING_QUERY_INSTRUCTION_SHA256,
    BridgeConfig,
    MemoryArenaContractError,
)
from benchmarks.integrations.memoryarena.evidence import assert_content_free
from benchmarks.integrations.memoryarena.runtime_bridge import MemoryArenaRuntimeBridge
from benchmarks.integrations.memoryarena.server import create_memoryarena_app


class _FakeEmbeddingResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeEmbeddingClient:
    def __init__(self, *, wrong_model_on_call: int | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.wrong_model_on_call = wrong_model_on_call

    async def post(self, url: str, **kwargs: object) -> _FakeEmbeddingResponse:
        self.calls.append({"url": url, **kwargs})
        request = kwargs["json"]
        assert isinstance(request, dict)
        inputs = request["input"]
        assert isinstance(inputs, list)
        model = (
            "wrong/embedding-model"
            if len(self.calls) == self.wrong_model_on_call
            else SEMANTIC_EMBEDDING_MODEL_ID
        )
        vector = [1.0] + [0.0] * (SEMANTIC_EMBEDDING_DIMENSIONS - 1)
        return _FakeEmbeddingResponse(
            {
                "model": model,
                "data": [
                    {"index": index, "embedding": vector} for index, _value in enumerate(inputs)
                ],
            }
        )


def _semantic_config(**overrides: object) -> BridgeConfig:
    values: dict[str, object] = {
        "embedding_mode": SEMANTIC_EMBEDDING_MODE,
        "embedding_dimensions": SEMANTIC_EMBEDDING_DIMENSIONS,
        "embedding_base_url": "https://memoryarena-embeddings.example.test",
        "embedding_api_key_env": "MEMORYARENA_TEST_EMBEDDING_KEY",
        "embedding_model_id": SEMANTIC_EMBEDDING_MODEL_ID,
        "embedding_model_revision": "a" * 40,
        "embedding_response_model": SEMANTIC_EMBEDDING_MODEL_ID,
    }
    values.update(overrides)
    return BridgeConfig(**values)


@pytest.mark.asyncio
async def test_official_api_isolated_scopes_embedding_drain_and_cleanup() -> None:
    bridge = MemoryArenaRuntimeBridge()
    await bridge.initialize("task-a-private", "swarmbrain")
    await bridge.initialize("task-b-private", "swarmbrain")
    add_a = await bridge.add(
        "task-a-private",
        "swarmbrain",
        "alpha-private-marker says reserve the Warsaw hotel",
    )
    await bridge.add(
        "task-b-private",
        "swarmbrain",
        "beta-private-marker says reserve the Lisbon hotel",
    )

    assert add_a["status"] == "ok"
    assert add_a["response"]["embedding_work_completed"] == 1
    wrapped_a = await bridge.wrap_user_prompt(
        "task-a-private", "swarmbrain", "What does alpha-private-marker say?"
    )
    assert wrapped_a["prompt"].startswith("<memory_context>\n<memory>")
    assert "Warsaw" in wrapped_a["prompt"]
    assert "Lisbon" not in wrapped_a["prompt"]
    assert wrapped_a["prompt"].endswith("User: What does alpha-private-marker say?")

    evidence = bridge.evidence()
    encoded = json.dumps(evidence, sort_keys=True)
    for secret in (
        "task-a-private",
        "task-b-private",
        "alpha-private-marker",
        "beta-private-marker",
        "Warsaw",
        "Lisbon",
    ):
        assert secret not in encoded
    assert evidence["event_count"] == 5
    assert evidence["embedding_execution"]["publishable"] is False
    assert evidence["embedding_execution"]["mode"] == "deterministic"
    assert any(
        row["operation"] == "add" and row["embedding_work_completed"] == 1
        for row in evidence["events"]
    )

    assert await bridge.cleanup("task-a-private") is True
    assert await bridge.cleanup("task-a-private") is False
    assert await bridge.active_scope_count() == 1
    await bridge.close()
    await bridge.close()


@pytest.mark.asyncio
async def test_reinitialize_resets_scope_and_empty_prompt_shape() -> None:
    bridge = MemoryArenaRuntimeBridge()
    await bridge.initialize("task-reset", "swarmbrain")
    await bridge.add("task-reset", "swarmbrain", "old-memory-marker")
    await bridge.initialize("task-reset", "swarmbrain")

    wrapped = await bridge.wrap_user_prompt("task-reset", "swarmbrain", "old-memory-marker")
    assert wrapped == {
        "status": "ok",
        "user_id": "task-reset",
        "prompt": ("<memory_context>\nNone\n</memory_context>\nUser: old-memory-marker"),
    }
    await bridge.close()


@pytest.mark.asyncio
async def test_canonical_packing_can_abstain_when_context_budget_is_exhausted() -> None:
    bridge = MemoryArenaRuntimeBridge(config=BridgeConfig(memory_context_token_budget=1))
    await bridge.initialize("packed-task", "swarmbrain")
    await bridge.add("packed-task", "swarmbrain", "needle " + "context " * 50)
    wrapped = await bridge.wrap_user_prompt("packed-task", "swarmbrain", "needle")

    assert "<memory_context>\nNone\n</memory_context>" in wrapped["prompt"]
    wrap_event = bridge.evidence()["events"][-1]
    assert wrap_event["selected_memory_count"] == 0
    assert wrap_event["dropped_memory_count"] >= 1
    await bridge.close()


@pytest.mark.asyncio
async def test_http_surface_matches_pinned_error_and_response_shapes() -> None:
    bridge = MemoryArenaRuntimeBridge()
    app = create_memoryarena_app(bridge, close_on_shutdown=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://memoryarena.test",
    ) as client:
        missing = await client.post(
            "/memory/wrap_user_prompt",
            json={
                "user_id": "missing",
                "memory_system_name": "swarmbrain",
                "question": "hello",
            },
        )
        assert missing.status_code == 404
        assert missing.json() == {"detail": "User not initialized"}

        unsupported = await client.post(
            "/memory/initialize",
            json={"user_id": "u", "memory_system_name": "mirix"},
        )
        assert unsupported.status_code == 400
        assert unsupported.json() == {"detail": "Unsupported memory_system: mirix"}

        initialized = await client.post(
            "/memory/initialize",
            json={"user_id": "u", "memory_system_name": "swarmbrain"},
        )
        assert initialized.json() == {
            "status": "ok",
            "user_id": "u",
            "memory_system_name": "swarmbrain",
        }
        mismatch = await client.post(
            "/memory/add",
            json={"user_id": "u", "memory_system_name": "bm25", "chunk": "x"},
        )
        assert mismatch.status_code == 400
        assert mismatch.json() == {"detail": "Mismatched memory_system for user"}
    await bridge.close()


def test_content_free_guard_rejects_payload_bearing_keys() -> None:
    with pytest.raises(ValueError, match="content-bearing"):
        assert_content_free({"question": "must not escape"})


@pytest.mark.asyncio
async def test_semantic_mode_attests_alias_and_coverage_but_not_served_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "fixture-secret-never-exported"
    monkeypatch.setenv("MEMORYARENA_TEST_EMBEDDING_KEY", secret)
    client = _FakeEmbeddingClient()
    bridge = MemoryArenaRuntimeBridge(
        config=_semantic_config(),
        openai_client=client,
    )
    await bridge.initialize("semantic-task-private", "swarmbrain")
    await bridge.add(
        "semantic-task-private",
        "swarmbrain",
        "The prior interaction selected the durable Warsaw itinerary.",
    )
    wrapped = await bridge.wrap_user_prompt(
        "semantic-task-private",
        "swarmbrain",
        "Which itinerary should the agent use?",
    )
    assert "Warsaw" in wrapped["prompt"]
    await bridge.close()

    evidence = bridge.evidence()
    embedding = evidence["embedding_execution"]
    assert embedding["mode"] == SEMANTIC_EMBEDDING_MODE
    assert embedding["publishable"] is False
    assert embedding["provider"] == SEMANTIC_EMBEDDING_PROVIDER
    assert embedding["model_id"] == SEMANTIC_EMBEDDING_MODEL_ID
    assert embedding["model_revision"] == "a" * 40
    assert embedding["model_revision_source"] == "operator-declared-unverified"
    assert embedding["provider_attested_model_revision"] is None
    assert embedding["deployment_manifest_bound_revision"] is False
    assert embedding["model_revision_binding_verified"] is False
    assert embedding["immutable_model_revision"] is False
    assert embedding["dimensions"] == SEMANTIC_EMBEDDING_DIMENSIONS
    assert embedding["required_response_model"] == SEMANTIC_EMBEDDING_MODEL_ID
    assert embedding["exact_response_model_verified"] is True
    assert embedding["query_instruction_sha256"] == SEMANTIC_EMBEDDING_QUERY_INSTRUCTION_SHA256
    assert embedding["call_accounting"] == {
        "source": "provider-observed",
        "document_inputs": 1,
        "document_batch_calls": 1,
        "query_calls": 1,
        "successful_http_calls": 2,
        "http_attempts": 2,
    }
    assert embedding["coverage"]["full_coverage"] is True
    assert embedding["coverage"]["zero_fallback"] is True
    assert embedding["provider_lifecycle"] == {"owner": "bridge", "closed": True}
    assert embedding["publishability_blockers"] == [
        "embedding_model_revision_not_attested_or_deployment_manifest_bound"
    ]
    wrap_event = evidence["events"][-1]
    assert wrap_event["dense_required"] is True
    assert wrap_event["dense_completed"] is True
    assert wrap_event["dense_fallback"] is False

    query_request = client.calls[-1]["json"]
    assert isinstance(query_request, dict)
    query_inputs = query_request["input"]
    assert isinstance(query_inputs, list)
    assert query_inputs[0].startswith(f"Instruct: {SEMANTIC_EMBEDDING_QUERY_INSTRUCTION}\nQuery: ")
    encoded = json.dumps(evidence, sort_keys=True)
    assert secret not in encoded
    assert "memoryarena-embeddings.example.test" not in encoded
    assert "MEMORYARENA_TEST_EMBEDDING_KEY" not in encoded
    assert SEMANTIC_EMBEDDING_QUERY_INSTRUCTION not in encoded


@pytest.mark.asyncio
async def test_semantic_mode_rejects_dense_fallback_on_response_model_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORYARENA_TEST_EMBEDDING_KEY", "fixture-key")
    bridge = MemoryArenaRuntimeBridge(
        config=_semantic_config(),
        openai_client=_FakeEmbeddingClient(wrong_model_on_call=2),
    )
    await bridge.initialize("semantic-fallback-task", "swarmbrain")
    await bridge.add("semantic-fallback-task", "swarmbrain", "A stored interaction.")

    with pytest.raises(MemoryArenaContractError, match="lexical fallback is forbidden"):
        await bridge.wrap_user_prompt(
            "semantic-fallback-task",
            "swarmbrain",
            "Retrieve the stored interaction.",
        )

    evidence = bridge.evidence()
    embedding = evidence["embedding_execution"]
    assert embedding["publishable"] is False
    assert embedding["exact_response_model_verified"] is False
    assert embedding["coverage"]["dense_fallbacks"] == 1
    assert embedding["coverage"]["zero_fallback"] is False
    wrap_event = evidence["events"][-1]
    assert wrap_event["success"] is False
    assert wrap_event["dense_required"] is True
    assert wrap_event["dense_completed"] is False
    assert wrap_event["dense_fallback"] is True
    await bridge.close()


@pytest.mark.asyncio
async def test_shared_semantic_provider_is_closed_only_by_bridge_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORYARENA_TEST_EMBEDDING_KEY", "fixture-key")
    bridge = MemoryArenaRuntimeBridge(
        config=_semantic_config(),
        openai_client=_FakeEmbeddingClient(),
    )
    provider = bridge._embedding_provider
    original_close = provider.close
    close_calls = 0

    async def tracked_close() -> None:
        nonlocal close_calls
        close_calls += 1
        await original_close()

    provider.close = tracked_close  # type: ignore[method-assign]
    await bridge.initialize("owner-scope-a", "swarmbrain")
    await bridge.initialize("owner-scope-b", "swarmbrain")
    await bridge.add("owner-scope-b", "swarmbrain", "Shared provider remains active.")
    assert await bridge.cleanup("owner-scope-a") is True
    assert close_calls == 0
    await bridge.wrap_user_prompt("owner-scope-b", "swarmbrain", "Is it still active?")

    await bridge.close()
    await bridge.close()
    assert close_calls == 1


def test_semantic_configuration_is_fixed_and_credentials_are_indirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(MemoryArenaContractError, match="40- or 64-character"):
        _semantic_config(embedding_model_revision="mutable-main")
    with pytest.raises(MemoryArenaContractError, match="model_id must be exactly"):
        _semantic_config(embedding_model_id="different/model")
    with pytest.raises(MemoryArenaContractError, match="dimensions must equal"):
        _semantic_config(embedding_dimensions=1_024)
    with pytest.raises(MemoryArenaContractError, match="response_model must be exactly"):
        _semantic_config(embedding_response_model="mutable-alias")
    with pytest.raises(MemoryArenaContractError, match="without credentials"):
        _semantic_config(embedding_base_url="https://user:secret@example.test/v1/embeddings")
    with pytest.raises(MemoryArenaContractError, match="must name an environment variable"):
        _semantic_config(embedding_api_key_env="not-valid")
    with pytest.raises(MemoryArenaContractError, match="semantic embedding settings require"):
        BridgeConfig(embedding_model_revision="a" * 40)

    monkeypatch.delenv("MEMORYARENA_TEST_EMBEDDING_KEY", raising=False)
    with pytest.raises(MemoryArenaContractError, match="unset or empty"):
        MemoryArenaRuntimeBridge(
            config=_semantic_config(),
            openai_client=_FakeEmbeddingClient(),
        )
