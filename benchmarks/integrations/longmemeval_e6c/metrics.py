"""Frozen one-sided paired inference for E6c context and QA gates."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Final

BOOTSTRAP_METHOD: Final = "stratified-percentile-paired-lower-bound-v1"
BOOTSTRAP_RESAMPLES: Final = 100_000
BOOTSTRAP_SEED: Final = 20_260_810
BOOTSTRAP_CONFIDENCE: Final = 0.95


class E6CMetricsError(ValueError):
    """Paired E6c evidence is empty, non-finite, or misaligned."""


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise E6CMetricsError("percentile requires values and a quantile in [0,1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise E6CMetricsError("percentile values must be finite")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_lower_bound(
    deltas: Sequence[float],
    strata: Sequence[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Return the frozen stratified paired percentile lower confidence bound."""

    if len(deltas) != len(strata) or not deltas:
        raise E6CMetricsError("paired deltas and strata must be non-empty and aligned")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise E6CMetricsError("bootstrap resamples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise E6CMetricsError("bootstrap confidence must be in (0,1)")
    normalized = [float(value) for value in deltas]
    if any(not math.isfinite(value) for value in normalized):
        raise E6CMetricsError("paired deltas must be finite")
    if any(not isinstance(value, str) or not value for value in strata):
        raise E6CMetricsError("bootstrap strata must be non-empty strings")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, stratum in zip(normalized, strata, strict=True):
        grouped[stratum].append(value)
    ordered = [(name, grouped[name]) for name in sorted(grouped)]
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _name, values in ordered:
            total += sum(values[rng.randrange(len(values))] for _ in range(len(values)))
        draws.append(total / len(normalized))
    point = sum(normalized) / len(normalized)
    return {
        "count": len(normalized),
        "delta": point,
        "lower_bound": percentile(draws, 1.0 - confidence),
        "improved": sum(value > 0.0 for value in normalized),
        "regressed": sum(value < 0.0 for value in normalized),
        "tied": sum(value == 0.0 for value in normalized),
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "unit": "question-local-history",
            "paired": True,
            "strata": "question_type",
            "resamples": resamples,
            "seed": seed,
            "confidence": confidence,
            "tail": "one-sided-lower",
        },
    }


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_METHOD",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "E6CMetricsError",
    "paired_lower_bound",
    "percentile",
]
