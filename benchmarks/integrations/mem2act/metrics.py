"""Exact Mem2ActBench scoring and deterministic paired bootstrap intervals."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .contracts import (
    Mem2ActContractError,
    PredictionRecord,
    TaskMetrics,
    ToolPrediction,
)


def parse_tool_prediction(raw_prediction: str) -> ToolPrediction:
    """Parse the benchmark's strict ``{"name", "arguments"}`` response."""

    if not isinstance(raw_prediction, str) or not raw_prediction.strip():
        raise Mem2ActContractError("tool prediction is empty")
    try:
        raw = json.loads(
            raw_prediction,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, Mem2ActContractError) as exc:
        raise Mem2ActContractError(f"tool prediction is not strict JSON: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {"name", "arguments"}:
        raise Mem2ActContractError(
            "tool prediction must contain exactly the keys 'name' and 'arguments'"
        )
    name = raw["name"]
    arguments = raw["arguments"]
    if not isinstance(name, str) or not name.strip():
        raise Mem2ActContractError("tool prediction name must be a non-empty string")
    if not isinstance(arguments, dict):
        raise Mem2ActContractError("tool prediction arguments must be an object")
    return ToolPrediction(name=name, arguments=arguments)


def strict_json_equal(left: Any, right: Any) -> bool:
    """Type-sensitive deep equality for JSON values.

    Python considers ``True == 1`` and ``1 == 1.0``.  Tool arguments do not:
    the official schemas distinguish booleans, integers, and floats, so exact
    matching must retain the JSON representation's type.
    """

    if type(left) is not type(right):  # noqa: E721 - exact type is the contract
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            strict_json_equal(lvalue, rvalue) for lvalue, rvalue in zip(left, right, strict=True)
        )
    return bool(left == right)


def score_prediction(
    prediction: ToolPrediction | None,
    *,
    gold_tool_name: str,
    gold_arguments: dict[str, Any],
) -> TaskMetrics:
    predicted_arguments = prediction.arguments if prediction is not None else {}
    tool_correct = int(prediction is not None and prediction.name == gold_tool_name)
    correct_slots = 0
    if tool_correct:
        correct_slots = sum(
            key in predicted_arguments and strict_json_equal(predicted_arguments[key], value)
            for key, value in gold_arguments.items()
        )
    predicted_slots = len(predicted_arguments)
    gold_slots = len(gold_arguments)
    true_positives = correct_slots
    false_positives = predicted_slots - true_positives
    false_negatives = gold_slots - true_positives
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _harmonic_mean(precision, recall)
    exact = int(
        bool(tool_correct)
        and prediction is not None
        and strict_json_equal(prediction.arguments, gold_arguments)
    )
    return TaskMetrics(
        tool_correct=tool_correct,
        exact_tool_and_arguments=exact,
        correct_slots=correct_slots,
        gold_slots=gold_slots,
        predicted_slots=predicted_slots,
        parameter_true_positives=true_positives,
        parameter_false_positives=false_positives,
        parameter_false_negatives=false_negatives,
        slot_accuracy=_safe_ratio(correct_slots, gold_slots),
        parameter_precision=precision,
        parameter_recall=recall,
        parameter_f1=f1,
    )


def aggregate_arm(records: Sequence[PredictionRecord]) -> dict[str, Any]:
    if not records:
        raise Mem2ActContractError("cannot aggregate an empty benchmark arm")
    task_count = len(records)
    totals = _metric_totals(records)
    failure_counts = Counter(
        record.failure.stage if record.failure is not None else "" for record in records
    )
    failure_counts.pop("", None)
    latencies = [record.total_latency_ms for record in records]
    reader_latencies = [record.reader_wall_latency_ms for record in records]
    return {
        "task_count": task_count,
        "successful_predictions": sum(record.failure is None for record in records),
        "failure_count": sum(failure_counts.values()),
        "failures_by_stage": dict(sorted(failure_counts.items())),
        "tool_accuracy": totals["tool_correct"] / task_count,
        "exact_tool_and_arguments": totals["exact"] / task_count,
        "slot_accuracy": _safe_ratio(totals["tp"], totals["gold_slots"]),
        "micro_parameter_precision": _safe_ratio(totals["tp"], totals["tp"] + totals["fp"]),
        "micro_parameter_recall": _safe_ratio(totals["tp"], totals["tp"] + totals["fn"]),
        "micro_parameter_f1": _harmonic_mean(
            _safe_ratio(totals["tp"], totals["tp"] + totals["fp"]),
            _safe_ratio(totals["tp"], totals["tp"] + totals["fn"]),
        ),
        "macro_parameter_precision": sum(record.metrics.parameter_precision for record in records)
        / task_count,
        "macro_parameter_recall": sum(record.metrics.parameter_recall for record in records)
        / task_count,
        "macro_parameter_f1": sum(record.metrics.parameter_f1 for record in records) / task_count,
        "slots": {
            "correct": totals["tp"],
            "gold": totals["gold_slots"],
            "predicted": totals["predicted_slots"],
            "false_positives": totals["fp"],
            "false_negatives": totals["fn"],
        },
        "tokens": {
            "prompt": sum(record.prompt_tokens for record in records),
            "completion": sum(record.completion_tokens for record in records),
            "total": sum(record.prompt_tokens + record.completion_tokens for record in records),
        },
        "latency_ms": {
            "mean_total": sum(latencies) / task_count,
            "p50_total": percentile(latencies, 0.50),
            "p95_total": percentile(latencies, 0.95),
            "mean_reader": sum(reader_latencies) / task_count,
            "p95_reader": percentile(reader_latencies, 0.95),
            "mean_retrieval": sum(record.retrieval_latency_ms for record in records) / task_count,
        },
    }


def paired_bootstrap(
    records: Sequence[PredictionRecord],
    *,
    arm_pairs: Sequence[tuple[str, str]],
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Question-cluster bootstrap for paired arm deltas.

    A sampled question contributes all of its parameter slots and both arm
    outcomes together.  This preserves within-question correlation and avoids
    treating individual slots as independent observations.
    """

    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise Mem2ActContractError("bootstrap resamples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise Mem2ActContractError("bootstrap confidence must be between zero and one")
    by_arm: dict[str, dict[str, PredictionRecord]] = {}
    for record in records:
        arm_records = by_arm.setdefault(record.arm, {})
        if record.qa_id in arm_records:
            raise Mem2ActContractError(
                f"duplicate prediction for arm={record.arm!r}, qa_id={record.qa_id!r}"
            )
        arm_records[record.qa_id] = record

    output: dict[str, Any] = {
        "method": "paired question-cluster bootstrap",
        "resamples": resamples,
        "seed": seed,
        "confidence": confidence,
        "pairs": {},
    }
    for pair_index, (left_arm, right_arm) in enumerate(arm_pairs):
        try:
            left = by_arm[left_arm]
            right = by_arm[right_arm]
        except KeyError as exc:
            raise Mem2ActContractError(f"bootstrap arm is missing: {exc.args[0]}") from exc
        if set(left) != set(right):
            raise Mem2ActContractError(
                f"paired arms {left_arm!r} and {right_arm!r} have different QA coverage"
            )
        qa_ids = sorted(left)
        if not qa_ids:
            raise Mem2ActContractError("bootstrap arms contain no tasks")
        left_records = [left[qa_id] for qa_id in qa_ids]
        right_records = [right[qa_id] for qa_id in qa_ids]
        point_left = _bootstrap_metrics(left_records, range(len(qa_ids)))
        point_right = _bootstrap_metrics(right_records, range(len(qa_ids)))
        metric_names = tuple(point_left)
        draws: dict[str, list[float]] = {name: [] for name in metric_names}
        rng = random.Random(seed + pair_index * 1_000_003)
        for _ in range(resamples):
            indices = [rng.randrange(len(qa_ids)) for _ in qa_ids]
            sampled_left = _bootstrap_metrics(left_records, indices)
            sampled_right = _bootstrap_metrics(right_records, indices)
            for name in metric_names:
                draws[name].append(sampled_left[name] - sampled_right[name])
        alpha = (1.0 - confidence) / 2.0
        output["pairs"][f"{left_arm}-minus-{right_arm}"] = {
            name: {
                "delta": point_left[name] - point_right[name],
                "ci_low": percentile(values, alpha),
                "ci_high": percentile(values, 1.0 - alpha),
            }
            for name, values in draws.items()
        }
    return output


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise Mem2ActContractError("cannot take a percentile of no values")
    if not 0.0 <= quantile <= 1.0:
        raise Mem2ActContractError("percentile quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise Mem2ActContractError("percentile input must be finite")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_totals(records: Sequence[PredictionRecord]) -> dict[str, int]:
    return {
        "tool_correct": sum(record.metrics.tool_correct for record in records),
        "exact": sum(record.metrics.exact_tool_and_arguments for record in records),
        "tp": sum(record.metrics.parameter_true_positives for record in records),
        "fp": sum(record.metrics.parameter_false_positives for record in records),
        "fn": sum(record.metrics.parameter_false_negatives for record in records),
        "gold_slots": sum(record.metrics.gold_slots for record in records),
        "predicted_slots": sum(record.metrics.predicted_slots for record in records),
    }


def _bootstrap_metrics(
    records: Sequence[PredictionRecord], indices: Sequence[int]
) -> dict[str, float]:
    selected = [records[index] for index in indices]
    totals = _metric_totals(selected)
    count = len(selected)
    precision = _safe_ratio(totals["tp"], totals["tp"] + totals["fp"])
    recall = _safe_ratio(totals["tp"], totals["tp"] + totals["fn"])
    return {
        "tool_accuracy": totals["tool_correct"] / count,
        "exact_tool_and_arguments": totals["exact"] / count,
        "slot_accuracy": _safe_ratio(totals["tp"], totals["gold_slots"]),
        "parameter_precision": precision,
        "parameter_recall": recall,
        "parameter_f1": _harmonic_mean(precision, recall),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _harmonic_mean(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Mem2ActContractError(f"duplicate prediction key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise Mem2ActContractError(f"non-finite prediction number is forbidden: {value}")


__all__ = [
    "aggregate_arm",
    "paired_bootstrap",
    "parse_tool_prediction",
    "percentile",
    "score_prediction",
    "strict_json_equal",
]
