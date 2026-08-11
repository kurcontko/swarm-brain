"""Benchmark-local envelope constants for core learned-reranker evidence.

The scorer, component, tokenizer, request, response, usage, and trace models are
the runtime contracts in :mod:`swarmbrain.domain.reranking`.  This module does
not define a parallel interpretation of them; it only fixes the paired
LongMemEval experimental design around those authoritative models.
"""

from __future__ import annotations

import math
import re
from typing import Any

RUN_SCHEMA_VERSION = 1
RUN_ARTIFACT_TYPE = "swarmbrain-longmemeval-reranker-ab-run"
REPORT_ARTIFACT_TYPE = "swarmbrain-longmemeval-reranker-ab-report"
PROTOCOL_VERSION = "swarmbrain-longmemeval-reranker-ab-v1"

BASELINE_ARM = "fixed_fusion_baseline"
TREATMENT_ARM = "learned_reranker"
ARMS = (BASELINE_ARM, TREATMENT_ARM)
K_VALUES = (5, 10)
CANDIDATE_WINDOW = 50

SCORE_MINIMUM = 0.0
SCORE_MAXIMUM = 1.0

BOOTSTRAP_METHOD = "percentile-paired-question-bootstrap-v1"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_809
BOOTSTRAP_CONFIDENCE = 0.95

SLICE_CATEGORIES = {
    "overall": None,
    "temporal": "temporal-reasoning",
    "conflict": "knowledge-update",
    "multi_session": "multi-session",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class LongMemEvalRerankerEvidenceError(ValueError):
    """Raw A/B evidence cannot support the paired reranker report."""


def required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongMemEvalRerankerEvidenceError(f"{label} must be a non-empty string")
    return value


def sha256_text(value: Any, *, label: str) -> str:
    text = required_text(value, label=label)
    if _SHA256_RE.fullmatch(text) is None:
        raise LongMemEvalRerankerEvidenceError(
            f"{label} must be a lowercase hexadecimal SHA-256 digest"
        )
    return text


def finite_number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LongMemEvalRerankerEvidenceError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise LongMemEvalRerankerEvidenceError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise LongMemEvalRerankerEvidenceError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise LongMemEvalRerankerEvidenceError(f"{label} must be <= {maximum}")
    return result


def integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LongMemEvalRerankerEvidenceError(f"{label} must be an integer >= {minimum}")
    return value


__all__ = [
    "ARMS",
    "BASELINE_ARM",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_METHOD",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CANDIDATE_WINDOW",
    "K_VALUES",
    "LongMemEvalRerankerEvidenceError",
    "PROTOCOL_VERSION",
    "REPORT_ARTIFACT_TYPE",
    "RUN_ARTIFACT_TYPE",
    "RUN_SCHEMA_VERSION",
    "SCORE_MAXIMUM",
    "SCORE_MINIMUM",
    "SLICE_CATEGORIES",
    "TREATMENT_ARM",
    "finite_number",
    "integer",
    "required_text",
    "sha256_text",
]
