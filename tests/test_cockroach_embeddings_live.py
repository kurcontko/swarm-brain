"""Live CockroachDB checks for the prefix-scoped vector index path.

Requires an installed schema (``swarmbrain-schema install``) on a cluster
that supports ``VECTOR`` and ``CREATE VECTOR INDEX`` (CockroachDB v24.2+).
"""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import pytest

from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
from swarmbrain.adapters.embeddings.deterministic import DeterministicEmbeddingProvider
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.memory import EmbeddingVector, MemoryKind, RememberCommand

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set"),
]

DIMENSIONS = 1024  # must match memory_embeddings.embedding VECTOR(1024)


def _id() -> str:
    return str(uuid4())


def _actor() -> ActorContext:
    return ActorContext(
        tenant_id=_id(),
        project_id=_id(),
        repository_id=_id(),
        swarm_id=_id(),
        run_id=_id(),
        agent_id=_id(),
        harness="pytest",
        provider="local",
        model="none",
        capabilities=frozenset(),
    )


@pytest.fixture
async def database() -> CockroachDatabase:
    assert DATABASE_URL is not None
    value = CockroachDatabase(DATABASE_URL, min_size=1, max_size=4)
    await value.start()
    try:
        yield value
    finally:
        await value.close()


async def _insert_run(database: CockroachDatabase, actor: ActorContext) -> None:
    async def body(connection: Any) -> None:
        await connection.execute(
            """
            INSERT INTO runs (tenant_id, id, project_id, repository_id, swarm_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                actor.tenant_id,
                actor.run_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
            ),
        )

    await database.run(body)  # type: ignore[arg-type]


async def test_upsert_and_prefix_scoped_ann_search(database: CockroachDatabase) -> None:
    store = CockroachMemoryStore(database)
    policy = ConservativeMemoryPolicy()
    provider = DeterministicEmbeddingProvider(dimensions=DIMENSIONS)
    actor = _actor()
    await _insert_run(database, actor)

    contents = ("cats purr when content", "dogs bark at couriers")
    memory_ids: list[str] = []
    for index, content in enumerate(contents):
        result = await store.remember(
            actor,
            RememberCommand(
                idempotency_key=f"live-embed-{index:05d}",
                kind=MemoryKind.OBSERVATION,
                content=content,
            ),
            policy,
        )
        assert result.memory is not None
        memory_ids.append(result.memory.memory_id)

    vectors = [
        EmbeddingVector(
            memory_id=memory_id,
            model=provider.model_name,
            dimensions=DIMENSIONS,
            values=(await provider.embed_documents([content]))[0],
        )
        for memory_id, content in zip(memory_ids, contents, strict=True)
    ]
    await store.upsert_embeddings(actor, vectors, idempotency_key="live-embed-upsert")
    # Same-key replay must not error or duplicate.
    await store.upsert_embeddings(actor, vectors, idempotency_key="live-embed-upsert")

    query = await provider.embed_query("cats purr when content")
    matches = await store.search_embeddings(
        actor, query, model=provider.model_name, limit=2, min_score=0.5
    )
    assert matches
    assert matches[0].memory_id == memory_ids[0]
    assert matches[0].score > 0.99

    # A different tenant sees nothing through the prefix-scoped index.
    stranger = _actor()
    foreign = await store.search_embeddings(stranger, query, model=provider.model_name, limit=2)
    assert foreign == ()

    # An unknown model never mixes with another model's vectors.
    other_model = await store.search_embeddings(actor, query, model="unrelated-model", limit=2)
    assert other_model == ()
