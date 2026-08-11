"""Pure same-head raw-anchor control for LongMemEval representation results.

Equal-family RRF can demote an excellent raw match when several candidates are
supported by both a raw key and a correlated derived key.  This control keeps
the already-selected treatment head unchanged and promotes its raw rank-one
value to the first position when that value is already present.  It never adds
or removes a value and it does not inspect labels, answers, or outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Final

from .contracts import RepresentationCell, RepresentationError, sha256_json
from .experiment import RepresentationResult
from .head_matched import HEAD_MATCHED_VALUE_COUNT

RAW_ANCHOR_PROTOCOL: Final = "raw-top1-within-existing-head-v1"


class RawAnchorError(RepresentationError):
    """A pair of representation results cannot support the raw-anchor control."""


def _validated_scores(result: RepresentationResult, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(result, RepresentationResult):
        raise RawAnchorError(f"{label} must be a RepresentationResult")
    if len(result.hydrated_values) != HEAD_MATCHED_VALUE_COUNT:
        raise RawAnchorError(
            f"{label} must contain exactly {HEAD_MATCHED_VALUE_COUNT} hydrated values"
        )
    scores = result.trace.get("value_scores")
    if not isinstance(scores, list) or len(scores) != len(result.hydrated_values):
        raise RawAnchorError(f"{label} value scores do not cover its hydration")
    if result.trace.get("value_scores_sha256") != sha256_json(scores):
        raise RawAnchorError(f"{label} value-score digest drifted")
    normalized: list[dict[str, Any]] = []
    for rank, (score, value) in enumerate(
        zip(scores, result.hydrated_values, strict=True),
        start=1,
    ):
        if (
            not isinstance(score, Mapping)
            or score.get("rank") != rank
            or score.get("value") != value.content_free_binding()
        ):
            raise RawAnchorError(f"{label} score order differs from its hydration")
        normalized.append(dict(score))
    return normalized


def _corpus_binding(result: RepresentationResult, *, label: str) -> tuple[Any, ...]:
    corpus = result.trace.get("corpus")
    if not isinstance(corpus, Mapping):
        raise RawAnchorError(f"{label} lacks a corpus binding")
    fields = ("question_id", "source_artifact_sha256", "projection_sha256")
    if any(not isinstance(corpus.get(field), str) or not corpus[field] for field in fields):
        raise RawAnchorError(f"{label} corpus binding is incomplete")
    return tuple(corpus[field] for field in fields)


def raw_top1_anchor_within_head(
    raw: RepresentationResult,
    fused: RepresentationResult,
) -> RepresentationResult:
    """Return ``fused`` with its in-set raw rank-one value promoted to rank one.

    If the raw anchor is absent from the fused head, the fused order is retained.
    In both branches, membership is byte-for-byte identical to ``fused`` and a
    new trace records whether the outcome-free control changed the order.
    Neither input object is mutated.
    """

    raw_scores = _validated_scores(raw, label="raw result")
    fused_scores = _validated_scores(fused, label="fused result")
    if raw.cell is not RepresentationCell.RAW:
        raise RawAnchorError("raw result must use the raw representation cell")
    if fused.cell is not RepresentationCell.RAW_MERGED_SFK:
        raise RawAnchorError("fused result must use the raw-plus-merged-SFK cell")
    if _corpus_binding(raw, label="raw result") != _corpus_binding(
        fused,
        label="fused result",
    ):
        raise RawAnchorError("raw and fused results cross a corpus boundary")

    raw_anchor = raw.hydrated_values[0]
    fused_by_id = {value.value_id: value for value in fused.hydrated_values}
    if len(fused_by_id) != len(fused.hydrated_values):
        raise RawAnchorError("fused result repeats a canonical value")
    anchor_present = raw_anchor.value_id in fused_by_id
    source_anchor_rank = next(
        (
            rank
            for rank, value in enumerate(fused.hydrated_values, start=1)
            if value.value_id == raw_anchor.value_id
        ),
        None,
    )
    if anchor_present:
        ordered = (fused_by_id[raw_anchor.value_id],) + tuple(
            value for value in fused.hydrated_values if value.value_id != raw_anchor.value_id
        )
    else:
        ordered = fused.hydrated_values

    score_by_id = {str(score["value"]["value_id"]): score for score in fused_scores}
    reordered_scores: list[dict[str, Any]] = []
    for rank, value in enumerate(ordered, start=1):
        score = deepcopy(score_by_id[value.value_id])
        prior_rank = int(score["rank"])
        score["rank"] = rank
        score["post_fusion_control"] = {
            "protocol": RAW_ANCHOR_PROTOCOL,
            "source_fused_rank": prior_rank,
            "raw_anchor": bool(anchor_present and value.value_id == raw_anchor.value_id),
        }
        reordered_scores.append(score)

    trace = deepcopy(fused.trace)
    source_ranking = deepcopy(trace.get("ranking"))
    source_ids = [value.value_id for value in fused.hydrated_values]
    result_ids = [value.value_id for value in ordered]
    trace["ranking"] = {
        "method": RAW_ANCHOR_PROTOCOL,
        "source_ranking": source_ranking,
        "source_head_count": len(source_ids),
        "raw_anchor_source_rank": 1,
        "raw_anchor_value_id": raw_anchor.value_id,
        "raw_anchor_present_in_source_head": anchor_present,
        "source_fused_anchor_rank": source_anchor_rank,
        "order_changed": source_ids != result_ids,
        "set_changed": set(source_ids) != set(result_ids),
        "gold_answer_judge_or_outcome_fields_used": False,
        "absent_anchor_policy": "retain-source-fused-order",
        "tie_break": "source-fused-rank-then-canonical-value-id",
    }
    trace["value_scores"] = reordered_scores
    trace["value_scores_sha256"] = sha256_json(reordered_scores)
    trace["hydrated_value_ids"] = result_ids
    trace["hydrated_raw_value_hashes"] = [value.raw_value_sha256 for value in ordered]
    trace["hydrated_value_count"] = len(ordered)

    anchored = RepresentationResult(
        cell=fused.cell,
        hydrated_values=ordered,
        trace=trace,
    )
    _validated_scores(anchored, label="anchored result")
    if set(result_ids) != set(source_ids) or len(result_ids) != len(source_ids):
        raise RawAnchorError("raw-anchor control changed fused-head membership")
    if raw_scores[0]["value"] != raw_anchor.content_free_binding():
        raise RawAnchorError("raw rank-one score binding drifted")
    return anchored


__all__ = [
    "RAW_ANCHOR_PROTOCOL",
    "RawAnchorError",
    "raw_top1_anchor_within_head",
]
