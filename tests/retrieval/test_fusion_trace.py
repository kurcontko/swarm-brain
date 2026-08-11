from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest

from conftest import make_actor, new_id
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.domain.memory import Memory, MemoryState, RecallHit, RecallQuery, Visibility
from swarmbrain.domain.retrieval import (
    Candidate,
    CandidateBatch,
    PackingPolicy,
    RetrievalSignal,
    RetrievalTrace,
)
from swarmbrain.retrieval import estimate_tokens, render_recall_hit


class _Gateway:
    def __init__(self, signal: RetrievalSignal, batch: CandidateBatch) -> None:
        self._signal = signal
        self.batch = batch

    @property
    def signal(self) -> RetrievalSignal:
        return self._signal

    async def retrieve(self, *_args: object) -> CandidateBatch:
        return self.batch


class _RaisingGateway:
    def __init__(self, signal: RetrievalSignal, error: BaseException) -> None:
        self._signal = signal
        self.error = error

    @property
    def signal(self) -> RetrievalSignal:
        return self._signal

    async def retrieve(self, *_args: object) -> CandidateBatch:
        raise self.error


class _Reader:
    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    async def hydrate_recallable(
        self, _actor: object, _query: object, candidate_ids: tuple[str, ...]
    ) -> tuple[Memory, ...]:
        return (self.memory,) if self.memory.memory_id in candidate_ids else ()


class _MultiReader:
    def __init__(self, memories: tuple[Memory, ...]) -> None:
        self.memories = {memory.memory_id: memory for memory in memories}

    async def hydrate_recallable(
        self, _actor: object, _query: object, candidate_ids: tuple[str, ...]
    ) -> tuple[Memory, ...]:
        return tuple(self.memories[item] for item in candidate_ids if item in self.memories)


class _TraceSink:
    def __init__(self) -> None:
        self.traces: list[RetrievalTrace] = []

    async def record(self, trace: RetrievalTrace) -> None:
        self.traces.append(trace)


@pytest.mark.asyncio
async def test_weighted_rrf_collapses_lanes_and_preserves_trace(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    memory_id = new_id()
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    memory = Memory(
        memory_id=memory_id,
        **scope_ids,
        author_agent_id=actor.agent_id,
        kind="procedure",
        state=MemoryState.CONFIRMED,
        visibility=Visibility.REPOSITORY,
        content="retry serialization failures",
        valid_from=now,
        recorded_from=now,
    )

    def candidate(signal: RetrievalSignal, rank: int, score: float) -> Candidate:
        return Candidate(
            resource_type="memory",
            resource_id=memory_id,
            resource_version=1,
            canonical_id=memory_id,
            domain_lane="playbook",
            signal=signal,
            rank=rank,
            raw_score=score,
        )

    exact = CandidateBatch(
        lane=RetrievalSignal.EXACT,
        candidates=(candidate(RetrievalSignal.EXACT, 1, 1.0),),
        examined_count=1,
        latency_ms=1.0,
    )
    lexical = CandidateBatch(
        lane=RetrievalSignal.LEXICAL,
        candidates=(candidate(RetrievalSignal.LEXICAL, 2, 0.17),),
        examined_count=5,
        latency_ms=2.0,
    )
    fuzzy = CandidateBatch(
        lane=RetrievalSignal.FUZZY,
        examined_count=0,
        latency_ms=0.0,
        degraded=True,
        degradation_reason="synthetic outage",
    )
    sink = _TraceSink()
    service = RetrievalService(
        (
            _Gateway(RetrievalSignal.EXACT, exact),
            _Gateway(RetrievalSignal.LEXICAL, lexical),
            _Gateway(RetrievalSignal.FUZZY, fuzzy),
        ),
        _Reader(memory),
        trace_sink=sink,
    )
    execution = await service.execute(actor, RecallQuery(text=memory_id))

    assert [hit.memory.memory_id for hit in execution.bundle.hits] == [memory_id]
    assert len(execution.trace.fused_candidates) == 1
    fused = execution.trace.fused_candidates[0]
    assert {item.lane for item in fused.contributions} == {
        RetrievalSignal.EXACT,
        RetrievalSignal.LEXICAL,
    }
    assert fused.raw_rrf == pytest.approx(5 / 61 + 3 / 62)
    assert execution.bundle.hits[0].score == pytest.approx(fused.normalized_score)
    assert execution.trace.degraded_lanes == frozenset({RetrievalSignal.FUZZY})
    assert sink.traces == [execution.trace]


@pytest.mark.asyncio
async def test_exact_rank_one_survives_min_score_with_empty_and_failed_lanes(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    memory_id = new_id()
    seed_id = new_id()
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    memory = Memory(
        memory_id=memory_id,
        **scope_ids,
        author_agent_id=actor.agent_id,
        kind="invariant",
        state=MemoryState.CONFIRMED,
        visibility=Visibility.REPOSITORY,
        content="canonical identifier target",
        valid_from=now,
        recorded_from=now,
    )
    exact_candidate = Candidate(
        resource_type="memory",
        resource_id=memory_id,
        resource_version=1,
        canonical_id=memory_id,
        domain_lane="knowledge",
        signal=RetrievalSignal.EXACT,
        rank=1,
        raw_score=1.0,
        reasons=("exact_memory_id",),
    )
    service = RetrievalService(
        (
            _Gateway(
                RetrievalSignal.EXACT,
                CandidateBatch(
                    lane=RetrievalSignal.EXACT,
                    candidates=(exact_candidate,),
                    examined_count=1,
                    latency_ms=0.1,
                ),
            ),
            _Gateway(
                RetrievalSignal.LEXICAL,
                CandidateBatch(
                    lane=RetrievalSignal.LEXICAL,
                    examined_count=0,
                    latency_ms=0.1,
                ),
            ),
            _RaisingGateway(RetrievalSignal.FUZZY, RuntimeError("fuzzy unavailable")),
        ),
        _Reader(memory),
    )

    execution = await service.execute(
        actor,
        RecallQuery(text=memory_id, min_score=0.6),
        seed_memory_ids=(seed_id,),
    )

    assert [hit.memory.memory_id for hit in execution.bundle.hits] == [memory_id]
    assert execution.bundle.hits[0].score == pytest.approx(1.0)
    assert execution.trace.fused_candidates[0].raw_rrf == pytest.approx(5 / 61)
    assert execution.trace.degraded_lanes == frozenset({RetrievalSignal.FUZZY})
    failed_batch = next(
        batch for batch in execution.trace.batches if batch.lane is RetrievalSignal.FUZZY
    )
    assert failed_batch.degradation_reason == "RuntimeError"
    assert execution.trace.parsed_identifiers == (memory_id, seed_id)


@pytest.mark.asyncio
async def test_runtime_budget_skips_oversized_top_hit_and_keeps_smaller_evidence(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    oversized = Memory(
        memory_id=new_id(),
        **scope_ids,
        author_agent_id=actor.agent_id,
        kind="observation",
        state=MemoryState.CONFIRMED,
        visibility=Visibility.REPOSITORY,
        content="needle " + ("large-context " * 300),
        valid_from=now,
        recorded_from=now,
    )
    compact = Memory(
        memory_id=new_id(),
        **scope_ids,
        author_agent_id=actor.agent_id,
        kind="procedure",
        state=MemoryState.CONFIRMED,
        visibility=Visibility.REPOSITORY,
        content="needle compact verified fix",
        valid_from=now,
        recorded_from=now,
    )
    candidates = tuple(
        Candidate(
            resource_type="memory",
            resource_id=memory.memory_id,
            resource_version=1,
            canonical_id=memory.memory_id,
            domain_lane="knowledge",
            signal=RetrievalSignal.LEXICAL,
            rank=rank,
            raw_score=1.0 / rank,
        )
        for rank, memory in enumerate((oversized, compact), start=1)
    )
    service = RetrievalService(
        (
            _Gateway(
                RetrievalSignal.LEXICAL,
                CandidateBatch(
                    lane=RetrievalSignal.LEXICAL,
                    candidates=candidates,
                    examined_count=2,
                    latency_ms=0.1,
                ),
            ),
        ),
        _MultiReader((oversized, compact)),
    )

    packed = await service.execute(
        actor,
        RecallQuery(text="needle", limit=2),
        token_budget=150,
    )
    unbounded = await service.execute(actor, RecallQuery(text="needle", limit=2))

    assert [hit.memory.memory_id for hit in packed.bundle.hits] == [compact.memory_id]
    assert packed.trace.final_ids == (compact.memory_id,)
    assert packed.trace.packing is not None
    assert packed.trace.packing.used_tokens <= 150
    assert packed.trace.packing.dropped_ids == (oversized.memory_id,)
    assert f"[memory:{compact.memory_id}]" in packed.rendered_context
    assert oversized.memory_id not in packed.rendered_context
    assert packed.bundle.truncated is True
    assert [hit.memory.memory_id for hit in unbounded.bundle.hits] == [
        oversized.memory_id,
        compact.memory_id,
    ]
    assert unbounded.trace.packing is None


@pytest.mark.asyncio
async def test_facility_location_is_an_explicit_runtime_policy(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    memories = tuple(
        Memory(
            memory_id=new_id(),
            **scope_ids,
            author_agent_id=actor.agent_id,
            kind="observation",
            state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            content=content,
            valid_from=now,
            recorded_from=now,
        )
        for content in ("alpha shared", "alpha shared", "beta distinct")
    )
    candidates = tuple(
        Candidate(
            resource_type="memory",
            resource_id=memory.memory_id,
            resource_version=1,
            canonical_id=memory.memory_id,
            domain_lane="knowledge",
            signal=RetrievalSignal.LEXICAL,
            rank=rank,
            raw_score=1.0 / rank,
        )
        for rank, memory in enumerate(memories, start=1)
    )
    gateway = _Gateway(
        RetrievalSignal.LEXICAL,
        CandidateBatch(
            lane=RetrievalSignal.LEXICAL,
            candidates=candidates,
            examined_count=3,
            latency_ms=0.1,
        ),
    )
    reader = _MultiReader(memories)
    greedy = RetrievalService((gateway,), reader)
    facility = RetrievalService(
        (gateway,),
        reader,
        packing_policy=PackingPolicy.FACILITY_LOCATION,
    )
    # Scores do not affect rendering; these lightweight copies mirror the
    # service's rank-order context blocks.
    rendered_hits = tuple(RecallHit(memory=memory, score=1.0) for memory in memories)
    rendered_sizes = tuple(
        estimate_tokens(("" if index == 0 else "\n\n") + render_recall_hit(hit))
        for index, hit in enumerate(rendered_hits)
    )
    budget = max(
        rendered_sizes[0] + rendered_sizes[1],
        rendered_sizes[0] + rendered_sizes[2],
    )

    greedy_execution = await greedy.execute(
        actor,
        RecallQuery(text="alpha beta", limit=2),
        token_budget=budget,
    )
    facility_execution = await facility.execute(
        actor,
        RecallQuery(text="alpha beta", limit=2),
        token_budget=budget,
    )

    assert [hit.memory.memory_id for hit in greedy_execution.bundle.hits] == [
        memories[0].memory_id,
        memories[1].memory_id,
    ]
    assert [hit.memory.memory_id for hit in facility_execution.bundle.hits] == [
        memories[0].memory_id,
        memories[2].memory_id,
    ]
    assert greedy_execution.trace.packing is not None
    assert facility_execution.trace.packing is not None
    assert greedy_execution.trace.packing.policy is PackingPolicy.GREEDY
    assert facility_execution.trace.packing.policy is PackingPolicy.FACILITY_LOCATION


@pytest.mark.asyncio
async def test_lane_cancellation_propagates(scope_ids: dict[str, str]) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    memory = Memory(
        memory_id=new_id(),
        **scope_ids,
        author_agent_id=actor.agent_id,
        kind="invariant",
        state=MemoryState.CONFIRMED,
        visibility=Visibility.REPOSITORY,
        content="unused",
        valid_from=now,
        recorded_from=now,
    )
    service = RetrievalService(
        (_RaisingGateway(RetrievalSignal.EXACT, asyncio.CancelledError()),),
        _Reader(memory),
    )

    with pytest.raises(asyncio.CancelledError):
        await service.execute(actor, RecallQuery(text=memory.memory_id))


@pytest.mark.asyncio
async def test_optional_reader_snapshot_and_clock_cover_lanes_and_hydration(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    memory = Memory(
        memory_id=new_id(),
        **scope_ids,
        author_agent_id=actor.agent_id,
        kind="invariant",
        state=MemoryState.CONFIRMED,
        visibility=Visibility.REPOSITORY,
        content="snapshot probe",
        valid_from=now,
        recorded_from=now,
    )
    events: list[str] = []

    class SnapshotReader(_Reader):
        active = False

        @asynccontextmanager
        async def retrieval_snapshot(self):  # type: ignore[no-untyped-def]
            events.append("enter")
            self.active = True
            try:
                yield
            finally:
                self.active = False
                events.append("exit")

        def retrieval_now(self) -> datetime:
            assert self.active
            events.append("now")
            return now

        async def hydrate_recallable(
            self, _actor: object, _query: object, candidate_ids: tuple[str, ...]
        ) -> tuple[Memory, ...]:
            assert self.active
            events.append("hydrate")
            return await super().hydrate_recallable(_actor, _query, candidate_ids)

    reader = SnapshotReader(memory)

    class SnapshotGateway(_Gateway):
        async def retrieve(self, *_args: object) -> CandidateBatch:
            assert reader.active
            events.append("lane")
            return await super().retrieve(*_args)

    service = RetrievalService(
        (
            SnapshotGateway(
                RetrievalSignal.EXACT,
                CandidateBatch(
                    lane=RetrievalSignal.EXACT,
                    examined_count=0,
                    latency_ms=0.0,
                ),
            ),
        ),
        reader,
    )

    execution = await service.execute(actor, RecallQuery(text="snapshot probe"))

    assert events == ["enter", "now", "lane", "hydrate", "now", "exit"]
    assert execution.bundle.generated_at == now
    assert execution.trace.started_at == now
    assert execution.trace.completed_at == now
