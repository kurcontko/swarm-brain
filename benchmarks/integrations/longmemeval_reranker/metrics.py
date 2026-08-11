"""Deterministic paired ranking metrics for the LongMemEval reranker A/B."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_METHOD,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    LongMemEvalRerankerEvidenceError,
)

METRIC_NAMES = ("recall_at_k", "mrr_at_k", "ndcg_at_k")


@dataclass(frozen=True, slots=True)
class PairedRankingCase:
    case_id: str
    category: str
    abstention_question: bool
    relevant_ids: frozenset[str]
    baseline_ids: tuple[str, ...]
    treatment_ids: tuple[str, ...]


def ranking_case_metrics(case: PairedRankingCase, *, k: int, arm: str) -> dict[str, float]:
    if k < 1:
        raise LongMemEvalRerankerEvidenceError("ranking depth must be positive")
    if not case.relevant_ids:
        raise LongMemEvalRerankerEvidenceError(
            f"case {case.case_id!r} has no relevance labels for paired ranking metrics"
        )
    ranking = case.baseline_ids if arm == "baseline" else case.treatment_ids
    returned = tuple(dict.fromkeys(ranking))[:k]
    ranks = [
        rank
        for rank, candidate_id in enumerate(returned, start=1)
        if candidate_id in case.relevant_ids
    ]
    dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks)
    ideal_count = min(k, len(case.relevant_ids))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "recall_at_k": len(ranks) / len(case.relevant_ids),
        "mrr_at_k": 0.0 if not ranks else 1.0 / ranks[0],
        "ndcg_at_k": 0.0 if ideal_dcg == 0.0 else dcg / ideal_dcg,
    }


def paired_summary(cases: Sequence[PairedRankingCase], *, k: int) -> dict[str, Any]:
    answerable = [case for case in cases if case.relevant_ids]
    if not answerable:
        raise LongMemEvalRerankerEvidenceError("paired metric slice has no answerable questions")
    baseline_rows = [ranking_case_metrics(case, k=k, arm="baseline") for case in answerable]
    treatment_rows = [ranking_case_metrics(case, k=k, arm="treatment") for case in answerable]
    deltas = {
        metric: [
            treatment[metric] - baseline[metric]
            for baseline, treatment in zip(baseline_rows, treatment_rows, strict=True)
        ]
        for metric in METRIC_NAMES
    }
    draws: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(answerable)
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = [rng.randrange(count) for _ in range(count)]
        for metric in METRIC_NAMES:
            values = deltas[metric]
            draws[metric].append(sum(values[index] for index in indices) / count)
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return {
        "questions": len(cases),
        "answerable_questions": count,
        "abstention_questions": sum(case.abstention_question for case in cases),
        "baseline": {
            metric: _mean([row[metric] for row in baseline_rows]) for metric in METRIC_NAMES
        },
        "learned_reranker": {
            metric: _mean([row[metric] for row in treatment_rows]) for metric in METRIC_NAMES
        },
        "paired_delta": {
            metric: {
                "delta": _mean(deltas[metric]),
                "ci_low": percentile(draws[metric], alpha),
                "ci_high": percentile(draws[metric], 1.0 - alpha),
                "improved_questions": sum(value > 0.0 for value in deltas[metric]),
                "regressed_questions": sum(value < 0.0 for value in deltas[metric]),
                "tied_questions": sum(value == 0.0 for value in deltas[metric]),
                "regression_rate": sum(value < 0.0 for value in deltas[metric]) / count,
            }
            for metric in METRIC_NAMES
        },
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "unit": "question",
            "paired": True,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": BOOTSTRAP_CONFIDENCE,
        },
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise LongMemEvalRerankerEvidenceError("cannot take a percentile of no values")
    if not 0.0 <= quantile <= 1.0:
        raise LongMemEvalRerankerEvidenceError("percentile quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise LongMemEvalRerankerEvidenceError("percentile input must be finite")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise LongMemEvalRerankerEvidenceError("cannot average no values")
    return sum(values) / len(values)


__all__ = ["METRIC_NAMES", "PairedRankingCase", "paired_summary", "percentile"]
