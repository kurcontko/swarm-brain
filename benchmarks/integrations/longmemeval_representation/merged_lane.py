"""Pure merged-SFK lane selector for source-preserving LongMemEval evidence.

The derived text remains navigation-only: this selector follows the merged-SFK
family's frozen top-20 key order and hydrates the bound immutable raw values.
It does not deliver a derived key to the reader, inspect labels, or execute a
model, file, clock, or network operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Final

from .contracts import (
    RRF_K,
    CanonicalValue,
    KeyFamily,
    RankedFamilyObservation,
    RepresentationCell,
    RepresentationCorpus,
    RepresentationError,
    sha256_json,
)
from .experiment import RepresentationResult
from .head_matched import HEAD_MATCHED_VALUE_COUNT

MERGED_LANE_PROTOCOL: Final = "merged-sfk-lane-top20-hydrate-raw-v1"


class MergedLaneSelectionError(RepresentationError):
    """Merged-lane evidence cannot support the fixed source-preserving head."""


def _validate_source_result(result: RepresentationResult) -> None:
    if not isinstance(result, RepresentationResult):
        raise MergedLaneSelectionError("source result must be a RepresentationResult")
    if result.cell is not RepresentationCell.RAW_MERGED_SFK:
        raise MergedLaneSelectionError("source result must use the raw-plus-merged-SFK cell")
    if result.trace.get("cell") != result.cell.value:
        raise MergedLaneSelectionError("source result cell and trace cell differ")
    if len(result.hydrated_values) != HEAD_MATCHED_VALUE_COUNT:
        raise MergedLaneSelectionError("source result must be the exact head-matched treatment")
    scores = result.trace.get("value_scores")
    if not isinstance(scores, list) or len(scores) != len(result.hydrated_values):
        raise MergedLaneSelectionError("source value scores do not cover its hydration")
    if result.trace.get("value_scores_sha256") != sha256_json(scores):
        raise MergedLaneSelectionError("source value-score digest drifted")
    for rank, (score, value) in enumerate(
        zip(scores, result.hydrated_values, strict=True),
        start=1,
    ):
        if (
            not isinstance(score, Mapping)
            or score.get("rank") != rank
            or score.get("value") != value.content_free_binding()
        ):
            raise MergedLaneSelectionError("source score order differs from its hydration")


def _validate_bindings(
    result: RepresentationResult,
    corpus: RepresentationCorpus,
    observation: RankedFamilyObservation,
) -> None:
    if not isinstance(corpus, RepresentationCorpus):
        raise MergedLaneSelectionError("corpus must be a RepresentationCorpus")
    if not isinstance(observation, RankedFamilyObservation):
        raise MergedLaneSelectionError("observation must be a RankedFamilyObservation")
    if observation.family is not KeyFamily.MERGED_SFK:
        raise MergedLaneSelectionError("selector requires the merged-SFK family observation")
    if len(observation.ranked_keys) != HEAD_MATCHED_VALUE_COUNT:
        raise MergedLaneSelectionError("merged-SFK observation must expose exactly 20 keys")
    trace_corpus = result.trace.get("corpus")
    if not isinstance(trace_corpus, Mapping):
        raise MergedLaneSelectionError("source result lacks a corpus binding")
    expected = (
        corpus.question_id,
        corpus.source_artifact_sha256,
        corpus.projection_sha256,
        corpus.index_sha256,
        corpus.navigation_index_sha256,
    )
    observed = (
        trace_corpus.get("question_id"),
        trace_corpus.get("source_artifact_sha256"),
        trace_corpus.get("projection_sha256"),
        trace_corpus.get("index_sha256"),
        trace_corpus.get("navigation_index_sha256"),
    )
    if observed != expected:
        raise MergedLaneSelectionError("source result, corpus, and observation bindings differ")
    if (
        observation.question_id != corpus.question_id
        or observation.source_artifact_sha256 != corpus.source_artifact_sha256
        or observation.projection_sha256 != corpus.projection_sha256
        or observation.index_sha256 != corpus.index_sha256
    ):
        raise MergedLaneSelectionError("merged observation crosses a corpus boundary")


def _selected_values(
    corpus: RepresentationCorpus,
    observation: RankedFamilyObservation,
) -> tuple[tuple[CanonicalValue, str, int], ...]:
    keys = corpus.keys_by_id()
    values = corpus.values_by_id()
    selected: list[tuple[CanonicalValue, str, int]] = []
    seen: set[str] = set()
    for rank, ranked in enumerate(observation.ranked_keys, start=1):
        key = keys.get(ranked.key_id)
        if key is None or key.family is not KeyFamily.MERGED_SFK:
            raise MergedLaneSelectionError("merged observation contains an unknown-family key")
        if key.source_value_id in seen:
            raise MergedLaneSelectionError(
                "merged top-20 does not reach 20 unique canonical source values"
            )
        value = values.get(key.source_value_id)
        if value is None:
            raise MergedLaneSelectionError("merged key points to an unknown canonical value")
        selected.append((value, ranked.key_id, rank))
        seen.add(key.source_value_id)
    if len(selected) != HEAD_MATCHED_VALUE_COUNT:
        raise MergedLaneSelectionError("merged lane did not select the exact 20-value head")
    return tuple(selected)


def _merged_accounting(source: Mapping[str, Any]) -> dict[str, Any]:
    accounting = deepcopy(dict(source))
    derived_count = accounting.get("derived_key_count")
    derived_bytes = accounting.get("derived_key_utf8_bytes")
    if (
        isinstance(derived_count, bool)
        or not isinstance(derived_count, int)
        or derived_count < HEAD_MATCHED_VALUE_COUNT
        or isinstance(derived_bytes, bool)
        or not isinstance(derived_bytes, int)
        or derived_bytes < 1
    ):
        raise MergedLaneSelectionError("source construction accounting is malformed")
    accounting["active_indexed_key_count"] = derived_count
    accounting["active_indexed_key_utf8_bytes"] = derived_bytes
    objects = accounting.get("derived_objects_per_source")
    if not isinstance(objects, list):
        raise MergedLaneSelectionError("source derived-object accounting is malformed")
    merged_objects: list[dict[str, Any]] = []
    for row in objects:
        if not isinstance(row, Mapping) or not isinstance(row.get("value_id"), str):
            raise MergedLaneSelectionError("source derived-object row is malformed")
        counts = row.get("key_counts")
        if not isinstance(counts, Mapping) or counts.get(KeyFamily.MERGED_SFK.value) != 1:
            raise MergedLaneSelectionError("source value lacks exactly one merged-SFK key")
        merged_objects.append(
            {
                "value_id": row["value_id"],
                "key_counts": {KeyFamily.MERGED_SFK.value: 1},
            }
        )
    accounting["derived_objects_per_source"] = merged_objects
    accounting["derived_objects_per_source_sha256"] = sha256_json(merged_objects)
    orphan = accounting.get("orphan_keys")
    if isinstance(orphan, Mapping):
        accounting["orphan_keys"] = {**orphan, "denominator": derived_count}
    return accounting


def select_merged_lane_top20(
    source: RepresentationResult,
    *,
    corpus: RepresentationCorpus,
    observation: RankedFamilyObservation,
) -> RepresentationResult:
    """Hydrate raw values in the merged-SFK lane's exact top-20 order."""

    _validate_source_result(source)
    _validate_bindings(source, corpus, observation)
    selected = _selected_values(corpus, observation)
    hydrated = tuple(value for value, _key_id, _rank in selected)
    value_scores = [
        {
            "rank": rank,
            "value": value.content_free_binding(),
            "score": 1.0 / (RRF_K + rank),
            "best_prior_rank": rank,
            "family_contributions": [
                {
                    "family": KeyFamily.MERGED_SFK.value,
                    "best_rank": rank,
                    "witness_key_ids": [key_id],
                    "suppressed_same_family_hits": 0,
                    "rrf_contribution": 1.0 / (RRF_K + rank),
                }
            ],
            "entity_score": None,
            "graph_support_count": 0,
            "expanded": False,
            "selection_control": {
                "protocol": MERGED_LANE_PROTOCOL,
                "source_family_rank": rank,
            },
        }
        for value, key_id, rank in selected
    ]
    observation_binding = observation.content_free_binding()
    trace = deepcopy(source.trace)
    trace["observations"] = [observation_binding]
    trace["observations_sha256"] = sha256_json([observation_binding])
    trace["ranking"] = {
        "method": MERGED_LANE_PROTOCOL,
        "family_accounting": {
            KeyFamily.MERGED_SFK.value: {
                "indexed_keys": observation.indexed_key_count,
                "returned_key_hits": len(observation.ranked_keys),
                "unique_values_reached": len(hydrated),
                "same_family_fanout_hits_suppressed": 0,
            }
        },
        "family_weight": 1.0,
        "rrf_k": RRF_K,
        "tie_break": "merged-key-score-descending-then-key-id",
        "source_fused_trace_sha256": source.trace_sha256,
        "gold_answer_judge_or_outcome_fields_used": False,
    }
    frozen = deepcopy(trace.get("frozen_protocol"))
    if not isinstance(frozen, dict):
        raise MergedLaneSelectionError("source frozen protocol is malformed")
    frozen["key_families"] = [KeyFamily.MERGED_SFK.value]
    trace["frozen_protocol"] = frozen
    trace["value_scores"] = value_scores
    trace["value_scores_sha256"] = sha256_json(value_scores)
    trace["key_level_returned_count"] = len(observation.ranked_keys)
    trace["hydrated_value_pre_cap_count"] = len(hydrated)
    trace["hydrated_value_ids"] = [value.value_id for value in hydrated]
    trace["hydrated_raw_value_hashes"] = [value.raw_value_sha256 for value in hydrated]
    trace["hydrated_value_count"] = len(hydrated)
    construction = trace.get("construction_and_index_accounting")
    if not isinstance(construction, Mapping):
        raise MergedLaneSelectionError("source construction accounting is missing")
    trace["construction_and_index_accounting"] = _merged_accounting(construction)

    result = RepresentationResult(
        cell=source.cell,
        hydrated_values=hydrated,
        trace=trace,
    )
    _validate_source_result(result)
    if any(
        value.raw_value != corpus.values_by_id()[value.value_id].raw_value
        for value in result.hydrated_values
    ):
        raise MergedLaneSelectionError("selector rewrote a canonical raw value")
    return result


__all__ = [
    "MERGED_LANE_PROTOCOL",
    "MergedLaneSelectionError",
    "select_merged_lane_top20",
]
