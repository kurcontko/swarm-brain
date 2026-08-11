"""Deterministic paired QA, context, and efficiency metrics."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_METHOD,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    LongMemEvalSelectionEvidenceError,
)


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    correct: bool
    any_gold_in_context: bool | None
    all_gold_in_context: bool | None
    answer_session_mrr: float | None
    prompt_tokens: int
    operational_latency_ms: float
    end_to_end_latency_ms: float
    construction_plus_query_cost_usd: float
    reader_cost_usd: float
    judge_cost_usd: float
    total_cost_usd: float
    accounting: dict[str, dict[str, int | float]]


@dataclass(frozen=True, slots=True)
class PairedQACase:
    question_id: str
    question_type: str
    baseline: ArmOutcome
    candidate: ArmOutcome


def percentile(values: Sequence[int | float], quantile: float) -> float:
    if not values:
        raise LongMemEvalSelectionEvidenceError("cannot take a percentile of no values")
    if not 0.0 <= quantile <= 1.0:
        raise LongMemEvalSelectionEvidenceError("percentile quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise LongMemEvalSelectionEvidenceError("percentile input must be finite")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        raise LongMemEvalSelectionEvidenceError("cannot summarize no values")
    checked = [float(value) for value in values]
    if any(not math.isfinite(value) for value in checked):
        raise LongMemEvalSelectionEvidenceError("summary values must be finite")
    total: int | float
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        total = sum(int(value) for value in values)
    else:
        total = sum(checked)
    return {
        "count": len(checked),
        "total": total,
        "mean": sum(checked) / len(checked),
        "min": min(checked),
        "p50": percentile(checked, 0.5),
        "p95": percentile(checked, 0.95),
        "max": max(checked),
    }


def _accuracy(cases: Sequence[PairedQACase], arm: str) -> dict[str, int | float]:
    values = [int(getattr(case, arm).correct) for case in cases]
    correct = sum(values)
    return {"questions": len(values), "correct": correct, "accuracy": correct / len(values)}


def _stratified_qa_draws(cases: Sequence[PairedQACase]) -> list[float]:
    by_type: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        by_type[case.question_type].append(
            float(case.candidate.correct) - float(case.baseline.correct)
        )
    ordered_strata = [(name, by_type[name]) for name in sorted(by_type)]
    total_count = len(cases)
    rng = random.Random(BOOTSTRAP_SEED)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        total = 0.0
        for _, values in ordered_strata:
            count = len(values)
            total += sum(values[rng.randrange(count)] for _ in range(count))
        draws.append(total / total_count)
    return draws


def paired_qa_summary(cases: Sequence[PairedQACase]) -> dict[str, Any]:
    if not cases:
        raise LongMemEvalSelectionEvidenceError("paired QA evidence has no questions")
    deltas = [float(case.candidate.correct) - float(case.baseline.correct) for case in cases]
    draws = _stratified_qa_draws(cases)
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return {
        "baseline": _accuracy(cases, "baseline"),
        "candidate": _accuracy(cases, "candidate"),
        "paired_delta": {
            "delta": sum(deltas) / len(deltas),
            "ci_low": percentile(draws, alpha),
            "ci_high": percentile(draws, 1.0 - alpha),
            "improved_questions": sum(value > 0.0 for value in deltas),
            "regressed_questions": sum(value < 0.0 for value in deltas),
            "tied_questions": sum(value == 0.0 for value in deltas),
        },
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "strata": "question_type",
            "unit": "question",
            "paired": True,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": BOOTSTRAP_CONFIDENCE,
        },
    }


def qa_by_question_type(cases: Sequence[PairedQACase]) -> dict[str, Any]:
    by_type: dict[str, list[PairedQACase]] = defaultdict(list)
    for case in cases:
        by_type[case.question_type].append(case)
    output: dict[str, Any] = {}
    for question_type in sorted(by_type):
        rows = by_type[question_type]
        baseline = _accuracy(rows, "baseline")
        candidate = _accuracy(rows, "candidate")
        output[question_type] = {
            "questions": len(rows),
            "baseline_correct": baseline["correct"],
            "candidate_correct": candidate["correct"],
            "baseline_accuracy": baseline["accuracy"],
            "candidate_accuracy": candidate["accuracy"],
            "paired_delta": float(candidate["accuracy"]) - float(baseline["accuracy"]),
            "improved_questions": sum(
                case.candidate.correct and not case.baseline.correct for case in rows
            ),
            "regressed_questions": sum(
                case.baseline.correct and not case.candidate.correct for case in rows
            ),
            "tied_questions": sum(case.baseline.correct == case.candidate.correct for case in rows),
        }
    return output


def context_summary(cases: Sequence[PairedQACase]) -> dict[str, Any]:
    eligible = [
        case
        for case in cases
        if not case.question_id.endswith("_abs")
        and case.baseline.any_gold_in_context is not None
        and case.baseline.all_gold_in_context is not None
        and case.baseline.answer_session_mrr is not None
        and case.candidate.any_gold_in_context is not None
        and case.candidate.all_gold_in_context is not None
        and case.candidate.answer_session_mrr is not None
    ]
    if not eligible:
        return {
            "available": False,
            "questions": 0,
            "reason": "dataset supplies no answer-session labels",
        }

    def arm_summary(arm: str) -> dict[str, Any]:
        rows = [getattr(case, arm) for case in eligible]
        any_values = [int(bool(row.any_gold_in_context)) for row in rows]
        all_values = [int(bool(row.all_gold_in_context)) for row in rows]
        mrr_values = [float(row.answer_session_mrr) for row in rows]
        return {
            "any_gold_in_context": sum(any_values) / len(any_values),
            "all_gold_in_context": sum(all_values) / len(all_values),
            "answer_session_mrr": sum(mrr_values) / len(mrr_values),
            "any_gold_questions": sum(any_values),
            "all_gold_questions": sum(all_values),
        }

    baseline = arm_summary("baseline")
    candidate = arm_summary("candidate")
    return {
        "available": True,
        "questions": len(eligible),
        "baseline": baseline,
        "candidate": candidate,
        "paired_delta": {
            metric: float(candidate[metric]) - float(baseline[metric])
            for metric in (
                "any_gold_in_context",
                "all_gold_in_context",
                "answer_session_mrr",
            )
        },
    }


def _phase_totals(cases: Sequence[PairedQACase], arm: str) -> dict[str, Any]:
    phases = ("embedding", "reranker", "constructor", "reader", "judge")
    output: dict[str, Any] = {}
    for phase in phases:
        rows = [getattr(case, arm).accounting[phase] for case in cases]
        fields = sorted(rows[0])
        output[phase] = {
            field: sum(row[field] for row in rows) for field in fields if field != "latency_ms"
        }
        output[phase]["latency_ms"] = numeric_summary([float(row["latency_ms"]) for row in rows])
    return output


def efficiency_summary(cases: Sequence[PairedQACase]) -> dict[str, Any]:
    if not cases:
        raise LongMemEvalSelectionEvidenceError("efficiency evidence has no questions")

    def arm_summary(arm: str) -> dict[str, Any]:
        rows = [getattr(case, arm) for case in cases]
        return {
            "prompt_tokens": numeric_summary([row.prompt_tokens for row in rows]),
            "operational_latency_ms": numeric_summary([row.operational_latency_ms for row in rows]),
            "end_to_end_latency_ms": numeric_summary([row.end_to_end_latency_ms for row in rows]),
            "construction_plus_query_cost_usd": numeric_summary(
                [row.construction_plus_query_cost_usd for row in rows]
            ),
            "reader_cost_usd": numeric_summary([row.reader_cost_usd for row in rows]),
            "judge_cost_usd": numeric_summary([row.judge_cost_usd for row in rows]),
            "total_cost_usd": numeric_summary([row.total_cost_usd for row in rows]),
            "accounting_totals": _phase_totals(cases, arm),
        }

    baseline = arm_summary("baseline")
    candidate = arm_summary("candidate")
    return {
        "baseline": baseline,
        "candidate": candidate,
        "paired_delta": {
            "prompt_tokens": numeric_summary(
                [case.candidate.prompt_tokens - case.baseline.prompt_tokens for case in cases]
            ),
            "operational_latency_ms": numeric_summary(
                [
                    case.candidate.operational_latency_ms - case.baseline.operational_latency_ms
                    for case in cases
                ]
            ),
            "construction_plus_query_cost_usd": numeric_summary(
                [
                    case.candidate.construction_plus_query_cost_usd
                    - case.baseline.construction_plus_query_cost_usd
                    for case in cases
                ]
            ),
        },
    }


def baseline_dominates_candidate(
    *,
    qa: dict[str, Any],
    efficiency: dict[str, Any],
) -> bool:
    """Return whether the baseline weakly dominates the candidate on all axes."""

    baseline_values = (
        float(qa["baseline"]["accuracy"]),
        -float(efficiency["baseline"]["prompt_tokens"]["p95"]),
        -float(efficiency["baseline"]["operational_latency_ms"]["p95"]),
        -float(efficiency["baseline"]["construction_plus_query_cost_usd"]["total"]),
    )
    candidate_values = (
        float(qa["candidate"]["accuracy"]),
        -float(efficiency["candidate"]["prompt_tokens"]["p95"]),
        -float(efficiency["candidate"]["operational_latency_ms"]["p95"]),
        -float(efficiency["candidate"]["construction_plus_query_cost_usd"]["total"]),
    )
    return all(
        left >= right for left, right in zip(baseline_values, candidate_values, strict=True)
    ) and any(left > right for left, right in zip(baseline_values, candidate_values, strict=True))


__all__ = [
    "ArmOutcome",
    "PairedQACase",
    "baseline_dominates_candidate",
    "context_summary",
    "efficiency_summary",
    "numeric_summary",
    "paired_qa_summary",
    "percentile",
    "qa_by_question_type",
]
