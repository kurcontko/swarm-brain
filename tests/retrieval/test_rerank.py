"""Unit coverage for relevance-aware reranking of the fused head.

The stage exists because weighted RRF rewards agreement between lanes, which
buries a candidate that exactly one strong lane ranks well.  These tests pin the
properties that make it safe to enable by default: it is a no-op when disabled,
it never reaches past its window, it is deterministic, and it cannot promote a
candidate that has no relevance evidence of its own.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from conftest import make_actor
from swarmbrain.domain.memory import RecallQuery
from swarmbrain.domain.retrieval import (
    FusedCandidate,
    FusionContribution,
    RetrievalPlan,
    RetrievalPurpose,
    RetrievalSignal,
)
from swarmbrain.retrieval import RetrievalPlanner, relevance_reranked

# Fusion contracts require real UUIDs; the labels keep the tests readable.
IDS = {
    "a": "0f8b1d3e-0000-4000-8000-00000000000a",
    "b": "0f8b1d3e-0000-4000-8000-00000000000b",
    "c": "0f8b1d3e-0000-4000-8000-00000000000c",
    "d": "0f8b1d3e-0000-4000-8000-00000000000d",
    "graph-only": "0f8b1d3e-0000-4000-8000-00000000000e",
}
LABELS = {value: key for key, value in IDS.items()}


def _candidate(label: str, raw_rrf: float) -> FusedCandidate:
    canonical_id = IDS[label]
    return FusedCandidate(
        canonical_id=canonical_id,
        raw_rrf=raw_rrf,
        normalized_score=min(1.0, raw_rrf),
        contributions=(
            FusionContribution(
                canonical_id=canonical_id,
                lane=RetrievalSignal.LEXICAL,
                rank=1,
                lane_weight=3.0,
                raw_score=raw_rrf,
                rrf_contribution=raw_rrf,
            ),
        ),
        reasons=("signal:lexical",),
    )


def _ids(candidates: tuple[FusedCandidate, ...]) -> list[str]:
    return [LABELS[item.canonical_id] for item in candidates]


FUSED = (
    _candidate("a", 1.0),
    _candidate("b", 0.9),
    _candidate("c", 0.8),
    _candidate("d", 0.7),
)
# "d" is the case the stage exists for: fused last, but the only candidate any
# lane can defend on its own evidence.
RELEVANCE = {IDS["a"]: 0.10, IDS["b"]: 0.20, IDS["c"]: 0.15, IDS["d"]: 0.95}


def test_alpha_zero_is_an_exact_no_op() -> None:
    assert relevance_reranked(FUSED, RELEVANCE, alpha=0.0, window=4) == FUSED


def test_zero_window_is_an_exact_no_op() -> None:
    assert relevance_reranked(FUSED, RELEVANCE, alpha=1.0, window=0) == FUSED


def test_empty_input_is_returned_unchanged() -> None:
    assert relevance_reranked((), RELEVANCE, alpha=0.5, window=8) == ()


def test_relevant_candidate_is_promoted_over_fused_consensus() -> None:
    reranked = relevance_reranked(FUSED, RELEVANCE, alpha=0.5, window=4)
    assert _ids(reranked)[0] == "d"


def test_candidates_past_the_window_keep_their_fused_position() -> None:
    """The window is a hard boundary, not a hint."""

    reranked = relevance_reranked(FUSED, RELEVANCE, alpha=1.0, window=2)
    assert _ids(reranked) == ["b", "a", "c", "d"]


def test_published_score_stays_monotone_with_published_order() -> None:
    """Any caller that re-sorts a bundle by score must preserve this ranking."""

    reranked = relevance_reranked(FUSED, RELEVANCE, alpha=0.5, window=4)
    head = [item.normalized_score for item in reranked[:4]]
    assert head == sorted(head, reverse=True)
    assert all(0.0 < score <= 1.0 for score in head)


def test_a_candidate_without_relevance_evidence_is_not_promoted() -> None:
    """Graph-only candidates score zero relevance and must stay defended by rank."""

    graph_only = (*FUSED, _candidate("graph-only", 0.6))
    reranked = relevance_reranked(graph_only, {}, alpha=1.0, window=5)
    # With no relevance for anyone the blend is flat, so the tie-break on fused
    # position must reproduce the input order exactly.
    assert _ids(reranked) == _ids(graph_only)


def test_ties_fall_back_to_fused_position() -> None:
    flat = dict.fromkeys((IDS["a"], IDS["b"], IDS["c"], IDS["d"]), 0.5)
    assert _ids(relevance_reranked(FUSED, flat, alpha=1.0, window=4)) == _ids(FUSED)


@pytest.mark.parametrize("alpha", [-0.01, 1.01])
def test_alpha_outside_the_unit_interval_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        relevance_reranked(FUSED, RELEVANCE, alpha=alpha, window=4)


def test_negative_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="window"):
        relevance_reranked(FUSED, RELEVANCE, alpha=0.5, window=-1)


@pytest.mark.parametrize("purpose", list(RetrievalPurpose))
def test_no_purpose_ships_with_reranking_enabled(
    scope_ids: dict[str, str],
    purpose: RetrievalPurpose,
) -> None:
    """The stage is measured and deliberately off.

    It wins on the 40-query swarm corpus and loses on the 500-question
    LongMemEval-S track, so the independent measurement vetoes it; the reasoning
    is in ``RetrievalPlanner._rerank_alpha`` and the numbers are in
    ``docs/retrieval-benchmark.md``. If a future measurement turns it on, that
    is a deliberate edit here and not a default drifting back.
    """

    plan = RetrievalPlanner().plan(
        make_actor(scope_ids),
        RecallQuery(text="how did we handle duplicate charges", limit=10),
        purpose=purpose,
        available_signals=(RetrievalSignal.LEXICAL, RetrievalSignal.DENSE),
    )

    assert plan.rerank is False
    assert plan.rerank_alpha == 0.0
    assert plan.rerank_window == 0


def test_a_plan_cannot_ask_for_reranking_without_bounds(scope_ids: dict[str, str]) -> None:
    """The flag and its two bounds are one switch, not three independent knobs."""

    plan = RetrievalPlanner().plan(
        make_actor(scope_ids),
        RecallQuery(text="anything", limit=10),
        purpose=RetrievalPurpose.INTERACTIVE_RECALL,
        available_signals=(RetrievalSignal.LEXICAL,),
    )
    # Bounds without the flag: dead configuration a later reader could mistake
    # for an active setting.
    bounds_only = plan.model_dump() | {"rerank_alpha": 0.5, "rerank_window": 40}
    with pytest.raises(ValidationError, match="rerank"):
        RetrievalPlan.model_validate(bounds_only)

    # The flag without bounds: reordering that would silently do nothing.
    flag_only = plan.model_dump() | {"rerank": True}
    with pytest.raises(ValidationError, match="rerank"):
        RetrievalPlan.model_validate(flag_only)
