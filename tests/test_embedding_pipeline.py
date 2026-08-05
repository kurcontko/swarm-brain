"""Durable projection, embedding worker, and hybrid-recall core checks."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from conftest import make_actor, new_id
from swarmbrain.adapters.embeddings.bedrock import (
    BedrockEmbeddingProvider,
    BedrockUnavailable,
)
from swarmbrain.adapters.embeddings.deterministic import DeterministicEmbeddingProvider
from swarmbrain.adapters.extraction.in_memory import InMemoryWorkStore
from swarmbrain.adapters.memory import InMemoryKernel
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy, memory_content_text
from swarmbrain.application.memory_service import SEMANTIC_MATCH_REASON, MemoryService
from swarmbrain.application.work import DurableWorkService
from swarmbrain.domain.memory import (
    EmbeddingMatch,
    RecallBundle,
    RecallHit,
    RecallQuery,
    RememberCommand,
)
from swarmbrain.domain.work import WorkStatus
from swarmbrain.workers.durable import LeasedWorkWorker
from swarmbrain.workers.embedding import EmbedMemoryHandler


class MappedProvider:
    model_name = "mapped-v1"
    dimensions = 4

    def __init__(self, mapping: dict[str, tuple[float, ...]]) -> None:
        self.mapping = mapping
        self.document_calls: list[tuple[str, ...]] = []

    async def embed_query(self, text: str) -> tuple[float, ...]:
        return self.mapping[text]

    async def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        values = tuple(texts)
        self.document_calls.append(values)
        return tuple(self.mapping[text] for text in values)


class FailingQueryProvider:
    model_name = "failing-query-v1"
    dimensions = 4

    async def embed_query(self, text: str) -> tuple[float, ...]:
        raise RuntimeError("dense provider unavailable")

    async def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0, 0.0, 0.0) for _ in texts)


class FixedEmbeddingIndex:
    def __init__(self, matches: tuple[EmbeddingMatch, ...]) -> None:
        self.matches = matches

    async def search_embeddings(
        self, *_args: object, **_kwargs: object
    ) -> tuple[EmbeddingMatch, ...]:
        return self.matches


def test_memory_content_text_is_stable_json_projection_and_preserves_strings() -> None:
    first = {"z": 2, "a": [True, 1, {"nested": "yes"}]}
    second = {"a": [True, 1, {"nested": "yes"}], "z": 2}

    assert memory_content_text(first) == '{"a":[true,1,{"nested":"yes"}],"z":2}'
    assert memory_content_text(second) == memory_content_text(first)
    assert memory_content_text("  original text  ") == "  original text  "


@pytest.mark.parametrize(
    ("content", "case"),
    [("", "empty"), ("x" * 1_000_001, "large")],
    ids=lambda value: value if value in {"empty", "large"} else None,
)
@pytest.mark.asyncio
async def test_every_valid_string_memory_can_enqueue_a_durable_projection(
    scope_ids: dict[str, str],
    content: str,
    case: str,
) -> None:
    kernel = InMemoryKernel()
    queue = InMemoryWorkStore()
    provider = FailingQueryProvider()
    service = MemoryService(
        kernel,
        ConservativeMemoryPolicy(),
        review_store=kernel,
        embeddings=provider,
        embedding_index=queue,
        work=DurableWorkService(queue),
    )

    result = await service.publish(
        make_actor(scope_ids),
        RememberCommand(
            idempotency_key=f"flexible-embedding-projection-{case}",
            kind="org.acme/flexible",
            content=content,
        ),
    )

    assert result.memory is not None
    assert len(queue.items) == 1
    assert next(iter(queue.items.values())).payload["content"] == content


@pytest.mark.asyncio
async def test_structured_memory_enqueues_projection_worker_embeds_and_recall_restores_json(
    scope_ids: dict[str, str],
) -> None:
    content = {"z": 2, "a": ["cats", {"content": True}]}
    projection = '{"a":["cats",{"content":true}],"z":2}'
    provider = MappedProvider(
        {
            projection: (1.0, 0.0, 0.0, 0.0),
            "felines": (1.0, 0.0, 0.0, 0.0),
        }
    )
    kernel = InMemoryKernel()
    queue = InMemoryWorkStore()
    service = MemoryService(
        kernel,
        ConservativeMemoryPolicy(),
        review_store=kernel,
        embeddings=provider,
        embedding_index=queue,
        work=DurableWorkService(queue),
    )
    actor = make_actor(scope_ids)
    command = RememberCommand(
        idempotency_key="structured-embed-publish-1",
        kind="org.acme/preference",
        content=content,
    )

    first = await service.publish(actor, command)
    replay = await service.publish(actor, command)

    assert first.memory is not None and replay.memory is not None
    assert replay.replayed is True
    assert replay.memory.memory_id == first.memory.memory_id
    assert len(queue.items) == 1
    item = next(iter(queue.items.values()))
    assert item.payload == {
        "content": projection,
        "model": provider.model_name,
        "metadata": {},
    }
    assert isinstance(item.payload["content"], str)

    incompatible_provider = MappedProvider({projection: (0.0, 1.0, 0.0, 0.0)})
    incompatible_provider.model_name = "mapped-v2"
    incompatible_worker = LeasedWorkWorker(
        queue,
        [EmbedMemoryHandler(incompatible_provider)],
    )
    assert await incompatible_worker.run_once("incompatible-embedding-worker") == ()
    assert queue.items[item.work_id].status is WorkStatus.PENDING
    assert queue.items[item.work_id].attempts == 0
    assert incompatible_provider.document_calls == []

    worker = LeasedWorkWorker(queue, [EmbedMemoryHandler(provider)])
    completed = await worker.run_once("embedding-worker-1")

    assert len(completed) == 1
    assert completed[0].item.status is WorkStatus.COMPLETED
    assert provider.document_calls == [(projection,)]
    assert len(queue.memory_embeddings) == 1

    recalled = await service.recall(
        actor,
        RecallQuery(text="felines", min_score=0.3, limit=5),
    )

    assert len(recalled.hits) == 1
    hit = recalled.hits[0]
    assert hit.memory.content == content
    assert SEMANTIC_MATCH_REASON in hit.reasons
    assert hit.score == pytest.approx(1.0)

    other_project = actor.model_copy(update={"project_id": new_id()})
    assert (
        await queue.search_embeddings(
            other_project,
            (1.0, 0.0, 0.0, 0.0),
            model=provider.model_name,
        )
        == ()
    )


@pytest.mark.asyncio
async def test_hybrid_recall_preserves_explicit_memory_id_filter(
    scope_ids: dict[str, str],
) -> None:
    provider = MappedProvider(
        {
            "allowed memory": (0.8, 0.6, 0.0, 0.0),
            "forbidden memory": (1.0, 0.0, 0.0, 0.0),
            "semantic probe": (1.0, 0.0, 0.0, 0.0),
        }
    )
    kernel = InMemoryKernel()
    queue = InMemoryWorkStore()
    service = MemoryService(
        kernel,
        ConservativeMemoryPolicy(),
        review_store=kernel,
        embeddings=provider,
        embedding_index=queue,
        work=DurableWorkService(queue),
    )
    actor = make_actor(scope_ids)
    allowed = await service.publish(
        actor,
        RememberCommand(
            idempotency_key="memory-id-filter-allowed",
            kind="org.acme/fact",
            content="allowed memory",
        ),
    )
    forbidden = await service.publish(
        actor,
        RememberCommand(
            idempotency_key="memory-id-filter-forbidden",
            kind="org.acme/fact",
            content="forbidden memory",
        ),
    )
    assert allowed.memory is not None and forbidden.memory is not None
    worker = LeasedWorkWorker(queue, [EmbedMemoryHandler(provider)])
    assert len(await worker.run_once("memory-id-filter-worker", limit=10)) == 2

    recalled = await service.recall(
        actor,
        RecallQuery(
            text="semantic probe",
            memory_ids=frozenset({allowed.memory.memory_id}),
            min_score=0.0,
            limit=5,
        ),
    )

    assert [hit.memory.memory_id for hit in recalled.hits] == [allowed.memory.memory_id]


@pytest.mark.asyncio
async def test_semantic_merge_preserves_rrf_order_for_equal_public_scores(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    kernel = InMemoryKernel()
    first = await kernel.remember(
        actor,
        RememberCommand(idempotency_key="rrf-first", kind="fact", content="first"),
        ConservativeMemoryPolicy(),
    )
    second = await kernel.remember(
        actor,
        RememberCommand(idempotency_key="rrf-second", kind="fact", content="second"),
        ConservativeMemoryPolicy(),
    )
    assert first.memory is not None and second.memory is not None
    base = datetime(2026, 8, 5, 12, tzinfo=UTC)
    older_first = first.memory.model_copy(update={"recorded_from": base})
    newer_second = second.memory.model_copy(update={"recorded_from": base + timedelta(seconds=1)})
    query = RecallQuery(text="stable tie query")
    bundle = RecallBundle(
        query=query,
        hits=(
            RecallHit(memory=older_first, score=1.0, reasons=("rrf:first",)),
            RecallHit(memory=newer_second, score=1.0, reasons=("rrf:second",)),
        ),
        generated_at=base,
        total_candidates=2,
    )
    provider = MappedProvider({query.text: (1.0, 0.0, 0.0, 0.0)})
    index = FixedEmbeddingIndex(
        (
            EmbeddingMatch(memory_id=newer_second.memory_id, score=1.0),
            EmbeddingMatch(memory_id=older_first.memory_id, score=1.0),
        )
    )
    service = MemoryService(
        kernel,
        ConservativeMemoryPolicy(),
        embeddings=provider,
        embedding_index=index,
        canonical_reader=kernel,
    )

    merged = await service._merge_semantic(actor, query, bundle)

    assert [hit.memory.memory_id for hit in merged.hits] == [
        older_first.memory_id,
        newer_second.memory_id,
    ]


@pytest.mark.asyncio
async def test_dense_failure_degrades_to_lexical_recall(
    scope_ids: dict[str, str],
) -> None:
    kernel = InMemoryKernel()
    actor = make_actor(scope_ids)
    service = MemoryService(
        kernel,
        ConservativeMemoryPolicy(),
        review_store=kernel,
        embeddings=FailingQueryProvider(),
        embedding_index=kernel,
    )
    published = await service.publish(
        actor,
        RememberCommand(
            idempotency_key="dense-fallback-memory",
            kind="org.acme/fact",
            content="lexical fallback anchor",
        ),
    )
    assert published.memory is not None
    query = RecallQuery(text="lexical fallback", min_score=0.1, limit=5)

    expected = await kernel.recall(actor, query)
    recalled = await service.recall(actor, query)

    assert recalled.hits == expected.hits
    assert recalled.total_candidates == expected.total_candidates
    assert recalled.truncated == expected.truncated


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
    assert (await provider.embed_documents(["lease fencing prevents duplicate completion"])) == (
        first,
    )


class _FakeBedrockClient:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.calls: list[dict[str, str]] = []

    def invoke_model(self, **kwargs: str) -> dict[str, object]:
        self.calls.append(kwargs)
        payload = json.dumps({"embedding": [0.5] * self.dimensions})
        return {"body": io.BytesIO(payload.encode("utf-8"))}


@pytest.mark.asyncio
async def test_bedrock_provider_shapes_titan_requests_and_checks_dimensions() -> None:
    fake = _FakeBedrockClient(dimensions=8)
    provider = BedrockEmbeddingProvider(dimensions=8, client=fake)

    vector = await provider.embed_query("hello")

    assert vector == tuple([0.5] * 8)
    call = fake.calls[0]
    assert call["modelId"] == "amazon.titan-embed-text-v2:0"
    assert json.loads(call["body"]) == {
        "inputText": "hello",
        "dimensions": 8,
        "normalize": True,
    }

    wrong = BedrockEmbeddingProvider(dimensions=16, client=fake)
    with pytest.raises(BedrockUnavailable, match="dimensions"):
        await wrong.embed_query("hello")
