from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid5

import pytest

from conftest import make_actor
from swarmbrain.adapters.memory import InMemoryKernel, in_memory_retrieval_gateways
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.application.memory_service import MemoryService
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.domain.memory import Memory, MemoryState, RecallQuery, Visibility

FIXTURES = Path(__file__).parents[1] / "fixtures" / "retrieval_v1"


def _load(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _memory(
    actor: object,
    item: dict[str, object],
    now: datetime,
) -> Memory:
    return Memory(
        memory_id=str(item["memory_id"]),
        tenant_id=actor.tenant_id,  # type: ignore[attr-defined]
        project_id=actor.project_id,  # type: ignore[attr-defined]
        repository_id=actor.repository_id,  # type: ignore[attr-defined]
        swarm_id=actor.swarm_id,  # type: ignore[attr-defined]
        run_id=actor.run_id,  # type: ignore[attr-defined]
        author_agent_id=actor.agent_id,  # type: ignore[attr-defined]
        kind=str(item["kind"]),
        state=MemoryState.CONFIRMED,
        visibility=Visibility.REPOSITORY,
        title=str(item["title"]),
        content=str(item["content"]),
        tags=tuple(str(tag) for tag in item["tags"]),  # type: ignore[union-attr]
        valid_from=now - timedelta(days=30),
        recorded_from=now + timedelta(seconds=int(item["recorded_offset_seconds"])),
    )


@pytest.mark.asyncio
async def test_gold_corpus_abstains_and_finds_old_exact_lexical_and_fuzzy(
    scope_ids: dict[str, str],
) -> None:
    corpus = _load("corpus.json")
    cases = _load("cases.json")
    now = datetime.fromisoformat(str(corpus["clock"]).replace("Z", "+00:00")).astimezone(UTC)
    actor = make_actor(scope_ids)
    kernel = InMemoryKernel(clock=lambda: now)
    keyed: dict[str, str] = {}
    for raw in corpus["memories"]:  # type: ignore[union-attr]
        item = dict(raw)
        memory = _memory(actor, item, now)
        keyed[str(item["key"])] = memory.memory_id
        kernel.memories[memory.memory_id] = memory

    generated = corpus["generated_decoys"]
    namespace = UUID(str(generated["namespace"]))  # type: ignore[index]
    for index in range(int(generated["count"])):  # type: ignore[index]
        memory_id = str(uuid5(namespace, str(index)))
        kernel.memories[memory_id] = Memory(
            memory_id=memory_id,
            tenant_id=actor.tenant_id,
            project_id=actor.project_id,
            repository_id=actor.repository_id,
            swarm_id=actor.swarm_id,
            run_id=actor.run_id,
            author_agent_id=actor.agent_id,
            kind="observation",
            state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            content=f"{generated['content_prefix']} {index}",  # type: ignore[index]
            valid_from=now - timedelta(days=30),
            recorded_from=now + timedelta(seconds=index),
        )

    retrieval = RetrievalService(in_memory_retrieval_gateways(kernel), kernel)
    service = MemoryService(
        kernel,
        ConservativeMemoryPolicy(),
        retrieval=retrieval,
        canonical_reader=kernel,
    )
    for raw_case in cases["cases"]:  # type: ignore[union-attr]
        case = dict(raw_case)
        result = await service.recall(
            actor,
            RecallQuery(text=str(case["query"]), limit=5),
        )
        expected = [keyed[str(key)] for key in case["expected"]]  # type: ignore[union-attr]
        actual = [hit.memory.memory_id for hit in result.hits]
        if expected:
            assert actual[: len(expected)] == expected, case
        else:
            assert result.hits == (), case
        assert all(hit.score > 0.0 for hit in result.hits)
        assert all("scope_match" not in hit.reasons for hit in result.hits)
