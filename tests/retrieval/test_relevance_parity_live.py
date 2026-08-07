"""Cross-adapter parity for the calibrated relevance signal, on live CockroachDB.

``tests/retrieval/test_relevance.py`` asserts the design property in isolation:
relevance is invariant to a lane's raw-score scale.  This asserts the
consequence that matters operationally — the *same* corpus and the *same*
queries produce the *same* relevance, and therefore the same abstention at the
same ``min_score``, whether the lanes are the in-memory kernel or CockroachDB.

Skipped without ``SWARMBRAIN_TEST_DATABASE_URL``.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import pytest

from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
from swarmbrain.adapters.cockroach.retrieval import cockroach_retrieval_gateways
from swarmbrain.adapters.memory import InMemoryKernel, in_memory_retrieval_gateways
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.application.memory_service import MemoryService
from swarmbrain.application.retrieval_service import RetrievalExecution, RetrievalService
from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.memory import MemoryState, RecallQuery, RememberCommand, Visibility

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set"),
]

FLOOR = 0.25

CORPUS = (
    (
        "Webhook replay guard",
        "services/payments/webhook_router.py rejects duplicate delivery ids",
        ("payments", "webhooks"),
    ),
    (
        "Serializable retry budget",
        "SQLSTATE 40001 retries are capped at four attempts with jitter",
        ("database",),
    ),
    (
        "Catalogue index rebuild cost",
        "rebuilding the catalogue search index saturates the shard for an hour",
        ("search",),
    ),
    (
        "Ruff configuration decision",
        "lint configuration lives in pyproject.toml and is enforced in CI",
        ("build",),
    ),
)

QUERIES = (
    "services/payments/webhook_router.py",
    "duplicate webhook delivery ids",
    "catalogue search index rebuild",
    "kubernetes horizontal pod autoscaler configuration",
    "terraform remote state locking for the staging workspace",
)


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
        capabilities=frozenset(item.value for item in Capability),
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


async def _publish(service: MemoryService, actor: ActorContext) -> dict[str, str]:
    """Publish the shared corpus and key the canonical ids by title."""

    by_title: dict[str, str] = {}
    for title, content, tags in CORPUS:
        result = await service.publish(
            actor,
            RememberCommand(
                idempotency_key=f"relevance-parity-{title}-{_id()}",
                kind="invariant",
                desired_state=MemoryState.CONFIRMED,
                visibility=Visibility.REPOSITORY,
                title=title,
                content=content,
                tags=tags,
            ),
        )
        assert result.memory is not None
        by_title[title] = result.memory.memory_id
    return by_title


def _relevance_by_title(
    execution: RetrievalExecution,
    by_title: dict[str, str],
) -> dict[str, float]:
    title_by_id = {memory_id: title for title, memory_id in by_title.items()}
    return {
        title_by_id[item.canonical_id]: round(item.relevance, 9)
        for item in execution.trace.candidate_relevance
        if item.canonical_id in title_by_id
    }


def _titles_returned(
    execution: RetrievalExecution,
    by_title: dict[str, str],
) -> frozenset[str]:
    returned = {hit.memory.memory_id for hit in execution.bundle.hits}
    return frozenset(title for title, memory_id in by_title.items() if memory_id in returned)


async def test_relevance_and_abstention_match_across_adapters(
    database: CockroachDatabase,
) -> None:
    cockroach_actor = _actor()
    await _insert_run(database, cockroach_actor)
    store = CockroachMemoryStore(database)
    cockroach_service = MemoryService(
        store,
        ConservativeMemoryPolicy(),
        review_store=store,
        canonical_reader=store,
    )
    cockroach_titles = await _publish(cockroach_service, cockroach_actor)
    cockroach_retrieval = RetrievalService(cockroach_retrieval_gateways(database), store)

    kernel = InMemoryKernel()
    memory_actor = _actor()
    memory_service = MemoryService(kernel, ConservativeMemoryPolicy(), review_store=kernel)
    memory_titles = await _publish(memory_service, memory_actor)
    memory_retrieval = RetrievalService(in_memory_retrieval_gateways(kernel), kernel)

    for text in QUERIES:
        query = RecallQuery(text=text, limit=10)
        cockroach_execution = await cockroach_retrieval.execute(cockroach_actor, query)
        memory_execution = await memory_retrieval.execute(memory_actor, query)

        assert _relevance_by_title(cockroach_execution, cockroach_titles) == _relevance_by_title(
            memory_execution, memory_titles
        ), text

        floored = RecallQuery(text=text, limit=10, min_score=FLOOR)
        cockroach_floored = await cockroach_retrieval.execute(cockroach_actor, floored)
        memory_floored = await memory_retrieval.execute(memory_actor, floored)

        assert cockroach_floored.trace.abstained == memory_floored.trace.abstained, text
        assert cockroach_floored.trace.abstention_reason == memory_floored.trace.abstention_reason
        assert _titles_returned(cockroach_floored, cockroach_titles) == _titles_returned(
            memory_floored, memory_titles
        ), text

    # The two topically absent queries must abstain identically on both.
    for text in QUERIES[-2:]:
        floored = RecallQuery(text=text, limit=10, min_score=FLOOR)
        assert (await cockroach_retrieval.execute(cockroach_actor, floored)).bundle.hits == ()
        assert (await memory_retrieval.execute(memory_actor, floored)).bundle.hits == ()
