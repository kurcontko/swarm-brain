"""Durable embedding pipeline: enqueue, leased worker, ANN-backed hybrid recall."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence

import pytest

from conftest import make_actor
from swarmbrain.adapters.embeddings.bedrock import BedrockEmbeddingProvider, BedrockUnavailable
from swarmbrain.adapters.embeddings.deterministic import DeterministicEmbeddingProvider
from swarmbrain.application.memory_service import SEMANTIC_MATCH_REASON
from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.domain.memory import RecallQuery, RememberCommand
from swarmbrain.domain.work import WorkStatus

SECRET = "0123456789abcdef-local-test-secret"


class MappedProvider:
    """Fixed text-to-vector mapping so semantic hits are fully controlled."""

    model_name = "mapped-v1"
    dimensions = 4

    def __init__(self, mapping: dict[str, tuple[float, ...]]) -> None:
        self.mapping = mapping

    async def embed_query(self, text: str) -> tuple[float, ...]:
        return self.mapping[text]

    async def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self.mapping[text] for text in texts)


@pytest.mark.asyncio
async def test_publish_enqueues_worker_embeds_and_recall_blends(
    scope_ids: dict[str, str],
) -> None:
    provider = MappedProvider(
        {
            "cats purr when content": (1.0, 0.0, 0.0, 0.0),
            "dogs bark at couriers": (0.0, 1.0, 0.0, 0.0),
            "felines": (1.0, 0.0, 0.0, 0.0),
        }
    )
    runtime = build_in_memory_runtime(SECRET, embeddings=provider)
    actor = make_actor(scope_ids)

    for index, content in enumerate(("cats purr when content", "dogs bark at couriers")):
        result = await runtime.memory.publish(
            actor,
            RememberCommand(
                idempotency_key=f"embed-publish-{index:04d}",
                kind="observation",
                content=content,
            ),
        )
        assert result.memory is not None

    worker = runtime.embedding_worker()
    completed = await worker.run_once("embed-worker-1")
    assert len(completed) == 2
    assert all(item.item.status is WorkStatus.COMPLETED for item in completed)
    assert runtime.kernel is not None
    assert len(runtime.kernel.memory_embeddings) == 2

    bundle = await runtime.memory.recall(
        actor,
        RecallQuery(text="felines", min_score=0.3, limit=5),
    )
    assert len(bundle.hits) == 1
    hit = bundle.hits[0]
    assert hit.memory.content == "cats purr when content"
    assert SEMANTIC_MATCH_REASON in hit.reasons
    assert hit.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_replayed_publish_reuses_the_same_work_item(scope_ids: dict[str, str]) -> None:
    provider = MappedProvider({"idempotent content": (0.0, 0.0, 1.0, 0.0)})
    runtime = build_in_memory_runtime(SECRET, embeddings=provider)
    actor = make_actor(scope_ids)
    command = RememberCommand(
        idempotency_key="embed-replay-0001",
        kind="observation",
        content="idempotent content",
    )

    first = await runtime.memory.publish(actor, command)
    second = await runtime.memory.publish(actor, command)
    assert second.replayed is True
    assert first.memory is not None and second.memory is not None
    assert second.memory.memory_id == first.memory.memory_id

    worker = runtime.embedding_worker()
    completed = await worker.run_once("embed-worker-1", limit=10)
    assert len(completed) == 1


@pytest.mark.asyncio
async def test_deterministic_provider_is_stable_normalized_and_sized() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=64)
    first = await provider.embed_query("lease fencing prevents duplicate completion")
    second = await provider.embed_query("lease fencing prevents duplicate completion")
    other = await provider.embed_query("an entirely different sentence")

    assert first == second
    assert len(first) == 64
    assert sum(value * value for value in first) == pytest.approx(1.0)
    assert first != other

    (document,) = await provider.embed_documents(["lease fencing prevents duplicate completion"])
    assert document == first


class _FakeBedrockClient:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.calls: list[dict[str, str]] = []

    def invoke_model(self, **kwargs: str) -> dict[str, object]:
        self.calls.append(kwargs)
        payload = json.dumps({"embedding": [0.5] * self.dimensions})
        return {"body": io.BytesIO(payload.encode("utf-8"))}


@pytest.mark.asyncio
async def test_bedrock_provider_shapes_titan_requests() -> None:
    fake = _FakeBedrockClient(dimensions=8)
    provider = BedrockEmbeddingProvider(dimensions=8, client=fake)

    vector = await provider.embed_query("hello")
    assert vector == tuple([0.5] * 8)
    call = fake.calls[0]
    assert call["modelId"] == "amazon.titan-embed-text-v2:0"
    body = json.loads(call["body"])
    assert body == {"inputText": "hello", "dimensions": 8, "normalize": True}


@pytest.mark.asyncio
async def test_bedrock_provider_rejects_wrong_dimensionality() -> None:
    fake = _FakeBedrockClient(dimensions=4)
    provider = BedrockEmbeddingProvider(dimensions=8, client=fake)
    with pytest.raises(BedrockUnavailable, match="dimensions"):
        await provider.embed_query("hello")
