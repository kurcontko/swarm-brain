"""Dependency-free retrieval and ANN evaluation primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RankingCase:
    case_id: str
    relevant_ids: frozenset[str]
    returned_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    cases: int
    answerable_cases: int
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    no_answer_precision: float
    no_answer_recall: float


def evaluate_rankings(cases: Sequence[RankingCase], *, k: int) -> RankingMetrics:
    """Evaluate binary relevance plus explicit no-answer behavior."""

    if k < 1:
        raise ValueError("k must be positive")
    answerable = [case for case in cases if case.relevant_ids]
    recall_values: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    for case in answerable:
        returned = tuple(dict.fromkeys(case.returned_ids))[:k]
        relevant_ranks = [
            rank
            for rank, candidate_id in enumerate(returned, start=1)
            if candidate_id in case.relevant_ids
        ]
        recall_values.append(len(relevant_ranks) / len(case.relevant_ids))
        reciprocal_ranks.append(0.0 if not relevant_ranks else 1.0 / relevant_ranks[0])
        dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
        ideal_count = min(k, len(case.relevant_ids))
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcg_values.append(0.0 if ideal_dcg == 0.0 else dcg / ideal_dcg)

    predicted_no_answer = [case for case in cases if not case.returned_ids[:k]]
    true_no_answer = [case for case in cases if not case.relevant_ids]
    correct_no_answer = sum(1 for case in predicted_no_answer if not case.relevant_ids)
    return RankingMetrics(
        cases=len(cases),
        answerable_cases=len(answerable),
        recall_at_k=_mean(recall_values),
        mrr_at_k=_mean(reciprocal_ranks),
        ndcg_at_k=_mean(ndcg_values),
        no_answer_precision=(
            1.0 if not predicted_no_answer else correct_no_answer / len(predicted_no_answer)
        ),
        no_answer_recall=(1.0 if not true_no_answer else correct_no_answer / len(true_no_answer)),
    )


def evaluate_lanes(
    relevant_by_case: Mapping[str, Sequence[str]],
    rankings_by_lane: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    k: int,
) -> dict[str, RankingMetrics]:
    """Evaluate direct, graph, fused, or arbitrary saved lane runs."""

    return {
        lane: evaluate_rankings(
            tuple(
                RankingCase(
                    case_id=case_id,
                    relevant_ids=frozenset(relevant),
                    returned_ids=tuple(per_case.get(case_id, ())),
                )
                for case_id, relevant in relevant_by_case.items()
            ),
            k=k,
        )
        for lane, per_case in rankings_by_lane.items()
    }


def ann_recall_at_k(
    approximate_ids: Sequence[str],
    exact_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Measure ANN top-k overlap against the exact vector oracle."""

    if k < 1:
        raise ValueError("k must be positive")
    exact = tuple(dict.fromkeys(exact_ids))[:k]
    if not exact:
        return 1.0
    approximate = frozenset(tuple(dict.fromkeys(approximate_ids))[:k])
    return len(approximate.intersection(exact)) / len(exact)


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


__all__ = [
    "RankingCase",
    "RankingMetrics",
    "ann_recall_at_k",
    "evaluate_lanes",
    "evaluate_rankings",
]
