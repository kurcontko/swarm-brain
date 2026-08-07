from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from conftest import make_actor, new_id
from swarmbrain.adapters.memory import InMemoryKernel, in_memory_retrieval_gateways
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.domain.memory import (
    Memory,
    MemoryLink,
    MemoryState,
    RecallQuery,
    Visibility,
)
from swarmbrain.domain.retrieval import Candidate, CandidateBatch, RetrievalPurpose, RetrievalSignal
from swarmbrain.retrieval import RetrievalPlanner


def _memory(
    actor: object,
    memory_id: str,
    content: str,
    now: datetime,
    *,
    visibility: Visibility = Visibility.REPOSITORY,
    run_id: str | None = None,
) -> Memory:
    return Memory(
        memory_id=memory_id,
        tenant_id=actor.tenant_id,  # type: ignore[attr-defined]
        project_id=actor.project_id,  # type: ignore[attr-defined]
        repository_id=actor.repository_id,  # type: ignore[attr-defined]
        swarm_id=actor.swarm_id,  # type: ignore[attr-defined]
        run_id=run_id or actor.run_id,  # type: ignore[attr-defined]
        author_agent_id=actor.agent_id,  # type: ignore[attr-defined]
        kind="observation",
        state=MemoryState.CONFIRMED,
        visibility=visibility,
        content=content,
        valid_from=now - timedelta(days=30),
        recorded_from=now - timedelta(days=2),
    )


def _link(
    source: str,
    target: str,
    kind: str,
    created_at: datetime,
) -> MemoryLink:
    return MemoryLink(
        link_id=new_id(),
        source_memory_id=source,
        target_memory_id=target,
        kind=kind,
        created_at=created_at,
    )


@pytest.mark.parametrize(
    ("purpose", "expected_hops"),
    (
        (RetrievalPurpose.INTERACTIVE_RECALL, 1),
        (RetrievalPurpose.HISTORICAL_AUDIT, 1),
        (RetrievalPurpose.TASK_BOOTSTRAP, 2),
        (RetrievalPurpose.CONFLICT_REVIEW, 2),
    ),
)
def test_planner_emits_a_complete_bounded_graph_policy(
    scope_ids: dict[str, str],
    purpose: RetrievalPurpose,
    expected_hops: int,
) -> None:
    plan = RetrievalPlanner().plan(
        make_actor(scope_ids),
        RecallQuery(text="graph policy", limit=10),
        purpose=purpose,
        available_signals=(RetrievalSignal.EXACT, RetrievalSignal.GRAPH),
    )

    assert plan.max_graph_hops == expected_hops
    assert plan.graph_seed_limit == 10
    assert plan.graph_max_fanout == 8
    assert plan.graph_edge_budget == 160
    assert plan.lane_budgets[RetrievalSignal.GRAPH.value] == 40
    assert plan.graph_link_types >= {"supports", "contradicts", "related_to"}


@pytest.mark.asyncio
async def test_graph_lane_expands_two_hops_with_cycle_safe_path_provenance(
    scope_ids: dict[str, str],
) -> None:
    now = datetime(2026, 8, 6, 12, tzinfo=UTC)
    actor = make_actor(scope_ids)
    kernel = InMemoryKernel(clock=lambda: now)
    seed_id, one_hop_id, two_hop_id, three_hop_id = (new_id() for _ in range(4))
    for memory in (
        _memory(actor, seed_id, "alpha sentinel failure", now),
        _memory(actor, one_hop_id, "rollback checklist context", now),
        _memory(actor, two_hop_id, "owner escalation record", now),
        _memory(actor, three_hop_id, "remote archive detail", now),
    ):
        kernel.memories[memory.memory_id] = memory
    for link in (
        _link(seed_id, one_hop_id, "supports", now - timedelta(hours=4)),
        _link(one_hop_id, two_hop_id, "derived_from", now - timedelta(hours=3)),
        _link(two_hop_id, three_hop_id, "related_to", now - timedelta(hours=2)),
    ):
        kernel.memory_links[link.link_id] = link

    execution = await RetrievalService(in_memory_retrieval_gateways(kernel), kernel).execute(
        actor,
        RecallQuery(text="alpha sentinel", limit=10),
        purpose=RetrievalPurpose.TASK_BOOTSTRAP,
    )

    graph_batch = next(
        batch for batch in execution.trace.batches if batch.lane is RetrievalSignal.GRAPH
    )
    graph_ids = tuple(candidate.canonical_id for candidate in graph_batch.candidates)
    assert graph_ids == (one_hop_id, two_hop_id)
    assert seed_id not in graph_ids
    assert three_hop_id not in graph_ids
    assert graph_batch.candidates[0].path[0] == f"memory:{seed_id}"
    assert graph_batch.candidates[0].path[-1] == f"memory:{one_hop_id}"
    assert len(graph_batch.candidates[0].path) == 3
    assert len(graph_batch.candidates[1].path) == 5
    assert "graph_path:supports>derived_from" in graph_batch.candidates[1].reasons
    assert {one_hop_id, two_hop_id}.issubset(
        {hit.memory.memory_id for hit in execution.bundle.hits}
    )
    two_hop_fused = next(
        candidate
        for candidate in execution.trace.fused_candidates
        if candidate.canonical_id == two_hop_id
    )
    assert {item.lane for item in two_hop_fused.contributions} == {RetrievalSignal.GRAPH}


@pytest.mark.asyncio
async def test_graph_lane_enforces_scope_relation_and_fanout_at_every_hop(
    scope_ids: dict[str, str],
) -> None:
    now = datetime(2026, 8, 6, 12, tzinfo=UTC)
    actor = make_actor(scope_ids)
    kernel = InMemoryKernel(clock=lambda: now)
    seed_id = new_id()
    hidden_id = new_id()
    hidden_descendant_id = new_id()
    custom_relation_id = new_id()
    supported_ids = tuple(new_id() for _ in range(10))
    memories = (
        _memory(actor, seed_id, "fanout sentinel primary", now),
        _memory(
            actor,
            hidden_id,
            "different run bridge",
            now,
            visibility=Visibility.RUN,
            run_id=new_id(),
        ),
        _memory(actor, hidden_descendant_id, "must not cross hidden bridge", now),
        _memory(actor, custom_relation_id, "custom relation target", now),
        *(
            _memory(actor, memory_id, f"neighbor payload {index}", now)
            for index, memory_id in enumerate(supported_ids)
        ),
    )
    for memory in memories:
        kernel.memories[memory.memory_id] = memory
    links = [
        _link(seed_id, hidden_id, "supports", now - timedelta(hours=3)),
        _link(hidden_id, hidden_descendant_id, "supports", now - timedelta(hours=2)),
        _link(seed_id, custom_relation_id, "causes", now - timedelta(hours=1)),
    ]
    links.extend(
        _link(seed_id, memory_id, "supports", now - timedelta(minutes=index))
        for index, memory_id in enumerate(supported_ids)
    )
    for link in links:
        kernel.memory_links[link.link_id] = link

    execution = await RetrievalService(in_memory_retrieval_gateways(kernel), kernel).execute(
        actor,
        RecallQuery(text="fanout sentinel", limit=100),
        purpose=RetrievalPurpose.PLANNING,
    )

    graph_batch = next(
        batch for batch in execution.trace.batches if batch.lane is RetrievalSignal.GRAPH
    )
    graph_ids = {candidate.canonical_id for candidate in graph_batch.candidates}
    assert len(graph_ids) == execution.trace.plan.graph_max_fanout == 8
    assert graph_ids.issubset(set(supported_ids))
    assert hidden_id not in graph_ids
    assert hidden_descendant_id not in graph_ids
    assert custom_relation_id not in graph_ids
    assert graph_batch.examined_count <= execution.trace.plan.graph_edge_budget
    assert graph_batch.truncated is True


@pytest.mark.asyncio
async def test_graph_lane_honors_recorded_time_for_edges(scope_ids: dict[str, str]) -> None:
    now = datetime(2026, 8, 6, 12, tzinfo=UTC)
    actor = make_actor(scope_ids)
    kernel = InMemoryKernel(clock=lambda: now)
    seed_id, target_id = new_id(), new_id()
    kernel.memories[seed_id] = _memory(actor, seed_id, "historical graph sentinel", now)
    kernel.memories[target_id] = _memory(actor, target_id, "future link target", now)
    link = _link(seed_id, target_id, "supports", now)
    kernel.memory_links[link.link_id] = link

    execution = await RetrievalService(in_memory_retrieval_gateways(kernel), kernel).execute(
        actor,
        RecallQuery(
            text="historical graph sentinel",
            recorded_at=now - timedelta(days=1),
        ),
        purpose=RetrievalPurpose.HISTORICAL_AUDIT,
    )

    graph_batch = next(
        batch for batch in execution.trace.batches if batch.lane is RetrievalSignal.GRAPH
    )
    assert graph_batch.candidates == ()
    assert target_id not in execution.trace.final_ids


@pytest.mark.asyncio
async def test_graph_failure_degrades_without_losing_direct_hits(
    scope_ids: dict[str, str],
) -> None:
    now = datetime(2026, 8, 6, 12, tzinfo=UTC)
    actor = make_actor(scope_ids)
    kernel = InMemoryKernel(clock=lambda: now)
    memory_id = new_id()
    kernel.memories[memory_id] = _memory(actor, memory_id, "direct survivor", now)

    class ExactGateway:
        signal = RetrievalSignal.EXACT

        async def retrieve(self, *_args: object) -> CandidateBatch:
            return CandidateBatch(
                lane=self.signal,
                candidates=(
                    Candidate(
                        resource_type="memory",
                        resource_id=memory_id,
                        resource_version=1,
                        canonical_id=memory_id,
                        domain_lane="knowledge",
                        signal=self.signal,
                        rank=1,
                        raw_score=1.0,
                    ),
                ),
                examined_count=1,
                latency_ms=0.1,
            )

    class BrokenGraphGateway:
        signal = RetrievalSignal.GRAPH

        async def expand(self, *_args: object) -> CandidateBatch:
            raise RuntimeError("graph unavailable")

    execution = await RetrievalService(
        (ExactGateway(), BrokenGraphGateway()),  # type: ignore[arg-type]
        kernel,
    ).execute(actor, RecallQuery(text="direct survivor"))

    assert execution.trace.final_ids == (memory_id,)
    assert execution.trace.degraded_lanes == frozenset({RetrievalSignal.GRAPH})
    graph_batch = next(
        batch for batch in execution.trace.batches if batch.lane is RetrievalSignal.GRAPH
    )
    assert graph_batch.degradation_reason == "RuntimeError"
