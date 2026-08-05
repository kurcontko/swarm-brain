"""Shared fixed-width CockroachDB vector encoding."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

COCKROACH_VECTOR_DIMENSIONS = 1024


def vector_literal(values: Sequence[float]) -> str:
    normalized = tuple(float(value) for value in values)
    if len(normalized) != COCKROACH_VECTOR_DIMENSIONS:
        raise ValueError(
            f"CockroachDB vector index requires {COCKROACH_VECTOR_DIMENSIONS} dimensions"
        )
    if any(not isfinite(value) for value in normalized):
        raise ValueError("embedding values must be finite")
    return "[" + ",".join(repr(value) for value in normalized) + "]"


__all__ = ["COCKROACH_VECTOR_DIMENSIONS", "vector_literal"]
