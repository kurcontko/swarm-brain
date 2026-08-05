"""Deterministic weighted Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import defaultdict

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


__all__ = ["RRF_K", "weighted_rrf"]
