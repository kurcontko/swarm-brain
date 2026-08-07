"""Unit and adapter-parity coverage for the calibrated relevance signal.

The signal exists so ``RecallQuery.min_score`` can express "nothing here is
actually relevant", which the rank-anchored public score cannot.  The two
properties that make it usable are asserted here directly: it is independent of
rank, and it is invariant to the adapter that produced the candidate.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from conftest import make_actor, new_id
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.domain.memory import Memory, MemoryState, RecallQuery, Visibility
from swarmbrain.domain.retrieval import (
    Candidate,
    CandidateBatch,
    FusionContribution,
    RetrievalSignal,
)
from swarmbrain.retrieval.relevance import (
    RELEVANCE_VERSION,
    candidate_relevance,
    relevance_query,
)

NOW = datetime(2026, 8, 7, 12, tzinfo=UTC)


class _Gateway:
    def __init__(self, signal: RetrievalSignal, batch: CandidateBatch) -> None:
        self._signal = signal
        self.batch = batch

    @property
    def signal(self) -> RetrievalSignal:
        return self._signal

    async def retrieve(self, *_args: object) -> CandidateBatch:
        return self.batch


class _Reader:
    def __init__(self, *memories: Memory) -> None:
        self.memories = memories

    async def hydrate_recallable(
        self, _actor: object, _query: object, candidate_ids: tuple[str, ...]
    ) -> tuple[Memory, ...]:
        return tuple(memory for memory in self.memories if memory.memory_id in candidate_ids)


def _memory(scope_ids: dict[str, str], actor_agent_id: str, **overrides: object) -> Memory:
    fields: dict[str, object] = {
        "memory_id": new_id(),
        "kind": "invariant",
        "state": MemoryState.CONFIRMED,
        "visibility": Visibility.REPOSITORY,
        "title": "Webhook replay guard",
        "content": "services/payments/webhook_router.py rejects duplicate delivery ids",
        "tags": ("payments",),
        "valid_from": NOW,
        "recorded_from": NOW,
    }
    fields.update(overrides)
    return Memory(**scope_ids, author_agent_id=actor_agent_id, **fields)  # type: ignore[arg-type]


def _contribution(
    canonical_id: str,
    lane: RetrievalSignal,
    *,
    rank: int,
    raw_score: float | None,
    weight: float = 3.0,
) -> FusionContribution:
    return FusionContribution(
        canonical_id=canonical_id,
        lane=lane,
        rank=rank,
        lane_weight=weight,
        raw_score=raw_score,
        rrf_contribution=weight / (60 + rank),
    )


def _candidate(memory: Memory, lane: RetrievalSignal, *, rank: int, raw: float) -> Candidate:
    return Candidate(
        resource_type="memory",
        resource_id=memory.memory_id,
        resource_version=memory.version,
        canonical_id=memory.memory_id,
        domain_lane="knowledge",
        signal=lane,
        rank=rank,
        raw_score=raw,
    )


# --------------------------------------------------------------------------- #
# the two load-bearing properties
# --------------------------------------------------------------------------- #


def test_relevance_ignores_rank_and_the_adapters_raw_score_scale(
    scope_ids: dict[str, str],
) -> None:
    """The in-memory and CockroachDB lexical lanes disagree on scale, not evidence.

    The in-memory kernel reports token overlap as its lexical raw score, while
    CockroachDB reports ``ts_rank``, which is roughly an order of magnitude
    smaller for the same match.  A calibration of raw scores could not mean the
    same thing on both; recomputing coverage from the canonical text does.
    """

    actor = make_actor(scope_ids)
    memory = _memory(scope_ids, actor.agent_id)
    query = relevance_query("webhook duplicate delivery ids")

    in_memory_view = candidate_relevance(
        query,
        memory,
        (_contribution(memory.memory_id, RetrievalSignal.LEXICAL, rank=1, raw_score=1.0),),
    )
    cockroach_view = candidate_relevance(
        query,
        memory,
        (
            _contribution(
                memory.memory_id,
                RetrievalSignal.LEXICAL,
                rank=37,
                raw_score=0.06079,
            ),
        ),
    )

    assert in_memory_view.relevance == cockroach_view.relevance
    assert in_memory_view.lexical_coverage == cockroach_view.lexical_coverage
    assert in_memory_view.trigram_similarity == cockroach_view.trigram_similarity
    assert in_memory_view.relevance == pytest.approx(1.0)


def test_relevance_of_a_top_ranked_but_weak_candidate_stays_low(
    scope_ids: dict[str, str],
) -> None:
    """Rank one in the strongest lane is exactly the case the public score cannot judge."""

    actor = make_actor(scope_ids)
    memory = _memory(scope_ids, actor.agent_id)
    query = relevance_query("kubernetes horizontal pod autoscaler configuration")

    scored = candidate_relevance(
        query,
        memory,
        (_contribution(memory.memory_id, RetrievalSignal.LEXICAL, rank=1, raw_score=1.0),),
    )

    assert scored.relevance < 0.25


# --------------------------------------------------------------------------- #
# per-lane components
# --------------------------------------------------------------------------- #


def test_an_exact_lane_contribution_pins_relevance_to_one(scope_ids: dict[str, str]) -> None:
    """A whole-query term match or a caller-supplied id needs no further defence."""

    actor = make_actor(scope_ids)
    memory = _memory(scope_ids, actor.agent_id)
    query = relevance_query(memory.memory_id)

    scored = candidate_relevance(
        query,
        memory,
        (
            _contribution(
                memory.memory_id, RetrievalSignal.EXACT, rank=1, raw_score=1.0, weight=5.0
            ),
        ),
    )

    assert scored.exact_term == 1.0
    assert scored.relevance == 1.0
    # The components are still reported honestly rather than short-circuited.
    assert scored.lexical_coverage == 0.0


def test_a_graph_only_candidate_carries_no_independent_relevance(
    scope_ids: dict[str, str],
) -> None:
    """Graph activation derives from a seed's fused rank, so it is rank information."""

    actor = make_actor(scope_ids)
    memory = _memory(
        scope_ids,
        actor.agent_id,
        title="Rollout percentages",
        content="staged exposure ladder for the payment provider swap",
        tags=(),
    )
    query = relevance_query("kubernetes horizontal pod autoscaler")

    expanded = candidate_relevance(
        query,
        memory,
        (
            _contribution(
                memory.memory_id, RetrievalSignal.GRAPH, rank=1, raw_score=0.94, weight=1.75
            ),
        ),
    )
    unreached = candidate_relevance(query, memory, ())

    # A near-maximal graph activation adds nothing: the candidate is worth
    # exactly what its own text can defend, which here is far below any floor.
    assert expanded.relevance == unreached.relevance
    assert expanded.relevance < 0.25


def test_dense_cosine_is_read_from_the_lane_and_bounded(scope_ids: dict[str, str]) -> None:
    """Cosine is the one component that cannot be recomputed without the query vector."""

    actor = make_actor(scope_ids)
    memory = _memory(
        scope_ids,
        actor.agent_id,
        title="Rollout percentages",
        content="staged exposure ladder",
        tags=(),
    )
    query = relevance_query("gradual customer exposure plan")

    scored = candidate_relevance(
        query,
        memory,
        (_contribution(memory.memory_id, RetrievalSignal.DENSE, rank=4, raw_score=0.63),),
    )
    negative = candidate_relevance(
        query,
        memory,
        (_contribution(memory.memory_id, RetrievalSignal.DENSE, rank=4, raw_score=-0.4),),
    )

    assert scored.dense_cosine == pytest.approx(0.63)
    assert scored.relevance == pytest.approx(0.63)
    assert negative.dense_cosine == 0.0


def test_relevance_is_the_max_and_not_an_accumulation(scope_ids: dict[str, str]) -> None:
    """Several individually weak lanes must not manufacture a passing score."""

    actor = make_actor(scope_ids)
    memory = _memory(
        scope_ids,
        actor.agent_id,
        title="Rollout percentages",
        content="staged exposure ladder for the provider swap",
        tags=(),
    )
    query = relevance_query("terraform remote state locking for the staging workspace")

    scored = candidate_relevance(
        query,
        memory,
        (
            _contribution(memory.memory_id, RetrievalSignal.LEXICAL, rank=1, raw_score=0.2),
            _contribution(memory.memory_id, RetrievalSignal.FUZZY, rank=2, raw_score=0.26),
            _contribution(memory.memory_id, RetrievalSignal.DENSE, rank=3, raw_score=0.21),
            _contribution(memory.memory_id, RetrievalSignal.GRAPH, rank=1, raw_score=0.9),
        ),
    )

    assert scored.relevance == max(
        scored.exact_term,
        scored.lexical_coverage,
        scored.trigram_similarity,
        scored.dense_cosine,
    )
    assert scored.relevance < 0.4


# --------------------------------------------------------------------------- #
# query projection
# --------------------------------------------------------------------------- #


def test_function_words_do_not_manufacture_coverage(scope_ids: dict[str, str]) -> None:
    actor = make_actor(scope_ids)
    memory = _memory(
        scope_ids,
        actor.agent_id,
        title="Index rebuild cost",
        content="the catalogue index rebuild is expensive for the whole shard",
        tags=(),
    )
    unfiltered = relevance_query("how are translations sourced for the czech locales")

    assert "the" not in unfiltered.tokens
    assert "for" not in unfiltered.tokens
    assert "translations" in unfiltered.tokens
    scored = candidate_relevance(
        unfiltered,
        memory,
        (_contribution(memory.memory_id, RetrievalSignal.LEXICAL, rank=1, raw_score=0.5),),
    )
    assert scored.lexical_coverage == 0.0


def test_a_query_of_only_function_words_still_scores_coverage() -> None:
    """Filtering must never empty the token set and collapse every score to zero."""

    query = relevance_query("what is it")

    assert query.tokens == frozenset({"what", "is", "it"})


def test_query_tokens_are_bounded_like_the_lexical_lane() -> None:
    query = relevance_query(" ".join(f"opaque{index}" for index in range(300)))

    assert len(query.tokens) == 64


# --------------------------------------------------------------------------- #
# service wiring
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_min_score_gates_relevance_while_the_public_score_is_unchanged(
    scope_ids: dict[str, str],
) -> None:
    """The dropped candidate is the rank-one hit whose public score is exactly 1.00."""

    actor = make_actor(scope_ids)
    weak = _memory(
        scope_ids,
        actor.agent_id,
        title="Ruff configuration",
        content="lint configuration lives in pyproject.toml",
        tags=(),
    )
    service = RetrievalService(
        (
            _Gateway(
                RetrievalSignal.LEXICAL,
                CandidateBatch(
                    lane=RetrievalSignal.LEXICAL,
                    candidates=(_candidate(weak, RetrievalSignal.LEXICAL, rank=1, raw=0.2),),
                    examined_count=1,
                    latency_ms=0.1,
                ),
            ),
        ),
        _Reader(weak),
    )
    query_text = "kubernetes horizontal pod autoscaler configuration"

    open_floor = await service.execute(actor, RecallQuery(text=query_text, min_score=0.0))
    raised_floor = await service.execute(actor, RecallQuery(text=query_text, min_score=0.4))

    assert [hit.memory.memory_id for hit in open_floor.bundle.hits] == [weak.memory_id]
    assert open_floor.bundle.hits[0].score == pytest.approx(1.0)
    assert open_floor.trace.relevance_version == RELEVANCE_VERSION
    assert open_floor.trace.candidate_relevance[0].relevance < 0.4

    assert raised_floor.bundle.hits == ()
    assert raised_floor.trace.abstained is True
    assert raised_floor.trace.abstention_reason == "below_relevance_floor"


@pytest.mark.asyncio
async def test_a_relevant_candidate_survives_a_floor_that_drops_the_noise(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    strong = _memory(scope_ids, actor.agent_id)
    noise = _memory(
        scope_ids,
        actor.agent_id,
        title="Ruff configuration",
        content="lint configuration lives in pyproject.toml",
        tags=(),
    )
    service = RetrievalService(
        (
            _Gateway(
                RetrievalSignal.LEXICAL,
                CandidateBatch(
                    lane=RetrievalSignal.LEXICAL,
                    candidates=(
                        _candidate(noise, RetrievalSignal.LEXICAL, rank=1, raw=0.4),
                        _candidate(strong, RetrievalSignal.LEXICAL, rank=2, raw=0.3),
                    ),
                    examined_count=2,
                    latency_ms=0.1,
                ),
            ),
        ),
        _Reader(strong, noise),
    )

    execution = await service.execute(
        actor,
        RecallQuery(text="webhook duplicate delivery ids", min_score=0.4),
    )

    assert [hit.memory.memory_id for hit in execution.bundle.hits] == [strong.memory_id]
    # The public score still reports the rank the fused ranking produced.
    assert execution.bundle.hits[0].score < 1.0


@pytest.mark.asyncio
async def test_default_min_score_leaves_the_ranking_and_truncation_untouched(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    first = _memory(scope_ids, actor.agent_id)
    second = _memory(
        scope_ids,
        actor.agent_id,
        title="Ruff configuration",
        content="lint configuration lives in pyproject.toml",
        tags=(),
    )
    service = RetrievalService(
        (
            _Gateway(
                RetrievalSignal.LEXICAL,
                CandidateBatch(
                    lane=RetrievalSignal.LEXICAL,
                    candidates=(
                        _candidate(first, RetrievalSignal.LEXICAL, rank=1, raw=0.9),
                        _candidate(second, RetrievalSignal.LEXICAL, rank=2, raw=0.2),
                    ),
                    examined_count=2,
                    latency_ms=0.1,
                ),
            ),
        ),
        _Reader(first, second),
    )

    execution = await service.execute(actor, RecallQuery(text="webhook duplicate", limit=1))

    assert [hit.memory.memory_id for hit in execution.bundle.hits] == [first.memory_id]
    assert execution.bundle.truncated is True
    assert execution.bundle.total_candidates == 2
