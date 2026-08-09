"""Deterministic weighted Reciprocal Rank Fusion, and relevance reranking.

``FusedCandidate.normalized_score`` as produced by :func:`weighted_rrf` is a
*rank* statement and nothing else: it is raw weighted RRF divided by the score a
rank-one hit in the strongest configured lane would earn.  It orders results
well and it is stable across lane availability, but it carries no information
about how well a candidate actually matches the query, so a threshold on it
cannot abstain.  The rank-independent counterpart lives in
:mod:`swarmbrain.retrieval.relevance` and is what ``RecallQuery.min_score`` is
gated on.

:func:`relevance_reranked` is the second stage.  RRF rewards *consensus*: a
candidate that several lanes each rank in the middle outscores one that a single
strong lane ranks well.  That is the right prior when lane scores are not
comparable, and it is the wrong one once a lane can state calibrated relevance,
which is exactly what a semantic dense lane plus the relevance module give us.
Reranking blends the two so agreement and evidence both count.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from swarmbrain.domain.retrieval import (
    Candidate,
    CandidateBatch,
    FusedCandidate,
    FusionContribution,
    RetrievalPlan,
)

RRF_K = 60


def weighted_rrf(
    batches: tuple[CandidateBatch, ...],
    plan: RetrievalPlan,
    *,
    k: int = RRF_K,
) -> tuple[FusedCandidate, ...]:
    if k < 1:
        raise ValueError("RRF k must be positive")

    strongest_lane_weight = max(
        (float(plan.lane_weights.get(signal.value, 0.0)) for signal in plan.signal_lanes),
        default=0.0,
    )
    if strongest_lane_weight <= 0.0:
        return ()
    # Keep raw weighted RRF as the ordering score.  The public [0, 1] score is
    # anchored to the best possible rank in the strongest configured lane,
    # rather than to simultaneous rank-one hits in every independent lane.
    # Otherwise merely enabling an empty or degraded lane makes a perfect
    # exact identifier match fail an otherwise stable public min_score.
    public_score_anchor = strongest_lane_weight / (k + 1)

    per_lane: dict[tuple[str, str], Candidate] = {}
    for batch in batches:
        if batch.degraded:
            continue
        for candidate in batch.candidates:
            if candidate.raw_score is not None and candidate.raw_score <= 0.0:
                continue
            key = (candidate.signal.value, candidate.canonical_id)
            current = per_lane.get(key)
            if current is None or (candidate.rank, candidate.resource_id) < (
                current.rank,
                current.resource_id,
            ):
                per_lane[key] = candidate

    contributions: dict[str, list[FusionContribution]] = defaultdict(list)
    reasons: dict[str, list[str]] = defaultdict(list)
    for candidate in per_lane.values():
        weight = float(plan.lane_weights.get(candidate.signal.value, 0.0))
        if weight <= 0.0:
            continue
        contribution = weight / (k + candidate.rank)
        contributions[candidate.canonical_id].append(
            FusionContribution(
                canonical_id=candidate.canonical_id,
                lane=candidate.signal,
                rank=candidate.rank,
                lane_weight=weight,
                raw_score=candidate.raw_score,
                rrf_contribution=contribution,
            )
        )
        reasons[candidate.canonical_id].extend(
            (f"signal:{candidate.signal.value}", *candidate.reasons)
        )

    fused: list[FusedCandidate] = []
    for canonical_id, items in contributions.items():
        ordered = tuple(sorted(items, key=lambda item: (item.lane.value, item.rank)))
        raw_rrf = sum(item.rrf_contribution for item in ordered)
        normalized = min(1.0, raw_rrf / public_score_anchor)
        if normalized <= 0.0:
            continue
        fused.append(
            FusedCandidate(
                canonical_id=canonical_id,
                raw_rrf=raw_rrf,
                normalized_score=normalized,
                contributions=ordered,
                reasons=tuple(dict.fromkeys(reasons[canonical_id])),
            )
        )
    fused.sort(key=lambda item: (-item.raw_rrf, item.canonical_id))
    return tuple(fused)


def relevance_reranked(
    fused: tuple[FusedCandidate, ...],
    relevance: Mapping[str, float],
    *,
    alpha: float,
    window: int,
) -> tuple[FusedCandidate, ...]:
    """Reorder the head of a fused ranking by rank consensus *and* relevance.

    The blend is ``(1 - alpha) * rrf + alpha * relevance``, where the RRF term
    is ``raw_rrf`` divided by the largest ``raw_rrf`` in the window.  Two
    properties of that normalisation matter:

    - it is strictly monotone in fused order, so ``alpha = 0`` reproduces
      weighted RRF exactly and the stage is a true no-op when disabled;
    - it is *relative to the window*, so the blend compares candidates against
      the best candidate this query actually produced rather than against the
      absolute anchor, which saturates at 1.0 for every strong hit and would
      collapse the fused ordering it is meant to preserve.

    Only the first ``window`` candidates are reordered.  Everything past the
    window keeps its fused position: relevance is not computed that deep, and a
    candidate the fusion buried at rank 200 should not be promoted on a single
    lane's say-so.

    Candidates missing from ``relevance`` score ``0.0`` on the relevance term.
    That is the correct reading rather than a gap: the only candidates that
    reach fusion without any relevance evidence are graph-only expansions,
    which :mod:`swarmbrain.retrieval.relevance` defines as carrying no
    independent relevance, so they must be defended by another lane.

    ``normalized_score`` on the reordered head is replaced by the blended value
    so that the published score stays monotone with the published order.  Any
    caller that sorts a bundle by score therefore preserves this ranking
    instead of silently undoing it.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("rerank alpha must be between 0 and 1")
    if window < 0:
        raise ValueError("rerank window must not be negative")
    if alpha == 0.0 or window == 0 or not fused:
        return fused

    head = fused[:window]
    tail = fused[window:]
    anchor = max((item.raw_rrf for item in head), default=0.0)
    if anchor <= 0.0:
        return fused

    blended: list[tuple[float, int, FusedCandidate]] = []
    for position, item in enumerate(head):
        score = (1.0 - alpha) * (item.raw_rrf / anchor) + alpha * relevance.get(
            item.canonical_id, 0.0
        )
        blended.append((score, position, item))
    # Ties fall back to the fused position, so the stage is deterministic and
    # never reorders candidates it cannot distinguish.
    blended.sort(key=lambda entry: (-entry[0], entry[1]))
    return (
        *(
            item.model_copy(update={"normalized_score": min(1.0, max(score, 1e-9))})
            for score, _position, item in blended
        ),
        *tail,
    )


__all__ = ["RRF_K", "relevance_reranked", "weighted_rrf"]
