"""Pure fixed-head hydration control for E6 representation results.

The representation evaluator records the complete ranking evidence before it
hydrates canonical raw values.  This module derives a packing-only control by
retaining the first 20 values in that already-validated order.  It does not
rerank values, rewrite source bytes, alter construction accounting, or execute
models, files, clocks, or networks.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Final

from .contracts import MAX_HYDRATED_VALUES, RepresentationError, sha256_json
from .experiment import RepresentationResult

HEAD_MATCHED_VALUE_COUNT: Final = 20


class HeadMatchedRepresentationError(RepresentationError):
    """A representation result cannot support the fixed 20-value control."""


def _validate_for_head_matching(result: RepresentationResult) -> None:
    trace = result.trace
    scores = trace.get("value_scores")
    if not isinstance(scores, list) or len(scores) != len(result.hydrated_values):
        raise HeadMatchedRepresentationError(
            "source representation value scores do not cover its hydration"
        )
    if trace.get("value_scores_sha256") != sha256_json(scores):
        raise HeadMatchedRepresentationError("source representation value-score digest drifted")
    for rank, (score, value) in enumerate(
        zip(scores, result.hydrated_values, strict=True),
        start=1,
    ):
        if (
            not isinstance(score, Mapping)
            or score.get("rank") != rank
            or score.get("value") != value.content_free_binding()
        ):
            raise HeadMatchedRepresentationError(
                "source representation value-score order differs from its hydration"
            )
    if trace.get("hydrated_value_count") != len(result.hydrated_values):
        raise HeadMatchedRepresentationError("source hydrated-value count drifted")
    if trace.get("hydrated_value_cap") != MAX_HYDRATED_VALUES:
        raise HeadMatchedRepresentationError("source hydrated-value cap drifted")
    frozen = trace.get("frozen_protocol")
    if not isinstance(frozen, Mapping) or frozen.get("hydrated_value_cap") != (MAX_HYDRATED_VALUES):
        raise HeadMatchedRepresentationError("source frozen-protocol hydration cap drifted")

    pre_cap = trace.get("hydrated_value_pre_cap_count")
    if (
        isinstance(pre_cap, bool)
        or not isinstance(pre_cap, int)
        or pre_cap < len(result.hydrated_values)
    ):
        raise HeadMatchedRepresentationError(
            "source representation has an inconsistent pre-cap hydration count"
        )
    if len(result.hydrated_values) < HEAD_MATCHED_VALUE_COUNT:
        raise HeadMatchedRepresentationError(
            f"source representation must hydrate at least {HEAD_MATCHED_VALUE_COUNT} values"
        )


def head_match_representation_result(result: RepresentationResult) -> RepresentationResult:
    """Return the first 20 hydrated values with exact bridge-valid trace bindings.

    The original result is never mutated.  ``hydrated_value_pre_cap_count`` and
    the frozen 128-value protocol cap remain unchanged because they describe
    the upstream ranking execution; only the derived hydration prefix and its
    dependent score evidence are narrowed.
    """

    if not isinstance(result, RepresentationResult):
        raise HeadMatchedRepresentationError("result must be RepresentationResult")
    _validate_for_head_matching(result)
    if len(result.hydrated_values) == HEAD_MATCHED_VALUE_COUNT:
        return result

    hydrated = result.hydrated_values[:HEAD_MATCHED_VALUE_COUNT]
    trace = deepcopy(result.trace)
    scores = trace["value_scores"][:HEAD_MATCHED_VALUE_COUNT]
    trace["value_scores"] = scores
    trace["value_scores_sha256"] = sha256_json(scores)
    trace["hydrated_value_ids"] = [value.value_id for value in hydrated]
    trace["hydrated_raw_value_hashes"] = [value.raw_value_sha256 for value in hydrated]
    trace["hydrated_value_count"] = len(hydrated)

    matched = RepresentationResult(
        cell=result.cell,
        hydrated_values=hydrated,
        trace=trace,
    )
    _validate_for_head_matching(matched)
    return matched


__all__ = [
    "HEAD_MATCHED_VALUE_COUNT",
    "HeadMatchedRepresentationError",
    "head_match_representation_result",
]
