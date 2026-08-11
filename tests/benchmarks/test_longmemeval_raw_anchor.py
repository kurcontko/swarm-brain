from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
from benchmarks.integrations.longmemeval_representation import (
    KeyFamily,
    RankedFamilyObservation,
    RankedKeyScore,
    RepresentationCell,
    RepresentationCorpus,
    ScorerIdentity,
    compile_question_canonical_values,
    evaluate_representation_cell,
    raw_key_id,
)
from benchmarks.integrations.longmemeval_representation.experiment import (
    RepresentationResult,
)
from benchmarks.integrations.longmemeval_representation.raw_anchor import (
    RAW_ANCHOR_PROTOCOL,
    RawAnchorError,
    raw_top1_anchor_within_head,
)
from benchmarks.integrations.longmemeval_turns import compile_dataset_bytes


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _source(question_id: str = "q-anchor") -> bytes:
    record: dict[str, Any] = {
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": "Which memory matters?",
        "answer": "fixture",
        "question_date": "2025/01/02 (Thu) 00:00",
        "haystack_session_ids": ["session"],
        "haystack_dates": ["2025/01/01 (Wed) 00:00"],
        "haystack_sessions": [
            [{"role": "user", "content": f"memory-{index:02d}"} for index in range(20)]
        ],
        "answer_session_ids": ["session"],
    }
    return (json.dumps([record], separators=(",", ":")) + "\n").encode()


def _scorer() -> ScorerIdentity:
    return ScorerIdentity(
        producer="fixture",
        protocol="fixture-v1",
        model_id="fixture/model",
        model_revision="revision",
        model_artifact_sha256=_digest("model"),
        identity_artifact_sha256=_digest("identity"),
    )


def _raw_result(raw: bytes, *, order: tuple[int, ...]) -> RepresentationResult:
    corpus = compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="fixture.json",
    )
    values = compile_question_canonical_values(corpus, question_id="q-anchor")
    representation = RepresentationCorpus(
        projection_corpus=corpus,
        values=values,
        derived_keys=(),
        construction_receipts=(),
    )
    observation = RankedFamilyObservation.create(
        family=KeyFamily.RAW,
        corpus=representation,
        query_sha256=_digest("query"),
        scorer=_scorer(),
        observation_artifact_sha256=_digest(str(order)),
        ranked_keys=tuple(
            RankedKeyScore(
                key_id=raw_key_id(values[position]),
                raw_score=100.0 - rank,
            )
            for rank, position in enumerate(order, start=1)
        ),
    )
    return evaluate_representation_cell(
        representation,
        cell=RepresentationCell.RAW,
        observations=(observation,),
    )


def _as_fused(raw: RepresentationResult, *, order: tuple[int, ...]) -> RepresentationResult:
    values = raw.hydrated_values
    ordered = tuple(values[position] for position in order)
    source_scores = {item["value"]["value_id"]: item for item in raw.trace["value_scores"]}
    scores = []
    for rank, value in enumerate(ordered, start=1):
        score = deepcopy(source_scores[value.value_id])
        score["rank"] = rank
        scores.append(score)
    trace = deepcopy(raw.trace)
    trace["cell"] = RepresentationCell.RAW_MERGED_SFK.value
    trace["value_scores"] = scores
    trace["value_scores_sha256"] = hashlib.sha256(
        json.dumps(scores, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    trace["hydrated_value_ids"] = [value.value_id for value in ordered]
    trace["hydrated_raw_value_hashes"] = [value.raw_value_sha256 for value in ordered]
    return RepresentationResult(
        cell=RepresentationCell.RAW_MERGED_SFK,
        hydrated_values=ordered,
        trace=trace,
    )


def test_raw_anchor_promotes_only_the_in_set_raw_top1_without_mutating_inputs() -> None:
    source = _source()
    raw = _raw_result(source, order=tuple(range(20)))
    fused_order = (1, 2, 0, *range(3, 20))
    fused = _as_fused(raw, order=fused_order)
    raw_before = deepcopy(raw.trace)
    fused_before = deepcopy(fused.trace)

    anchored = raw_top1_anchor_within_head(raw, fused)

    assert anchored.hydrated_values[0] == raw.hydrated_values[0]
    assert [value.value_id for value in anchored.hydrated_values[1:]] == [
        value.value_id for value in fused.hydrated_values if value != raw.hydrated_values[0]
    ]
    assert {value.value_id for value in anchored.hydrated_values} == {
        value.value_id for value in fused.hydrated_values
    }
    assert anchored.trace["ranking"]["method"] == RAW_ANCHOR_PROTOCOL
    assert anchored.trace["ranking"]["source_fused_anchor_rank"] == 3
    assert anchored.trace["ranking"]["order_changed"] is True
    assert anchored.trace["ranking"]["set_changed"] is False
    assert [score["rank"] for score in anchored.trace["value_scores"]] == list(range(1, 21))
    assert raw.trace == raw_before
    assert fused.trace == fused_before


def test_raw_anchor_is_deterministic_and_records_an_already_first_noop() -> None:
    source = _source()
    raw = _raw_result(source, order=tuple(range(20)))
    fused = _as_fused(raw, order=tuple(range(20)))

    first = raw_top1_anchor_within_head(raw, fused)
    second = raw_top1_anchor_within_head(raw, fused)

    assert first == second
    assert first.trace_sha256 == second.trace_sha256
    assert first.hydrated_values == fused.hydrated_values
    assert first.trace["ranking"]["order_changed"] is False
    assert first.trace["ranking"]["raw_anchor_present_in_source_head"] is True


def test_raw_anchor_rejects_cell_corpus_and_digest_drift() -> None:
    source = _source()
    raw = _raw_result(source, order=tuple(range(20)))
    fused = _as_fused(raw, order=tuple(range(20)))

    with pytest.raises(RawAnchorError, match="raw representation cell"):
        raw_top1_anchor_within_head(fused, fused)

    other_source = _source("q-other")
    other_corpus = compile_dataset_bytes(
        other_source,
        expected_sha256=hashlib.sha256(other_source).hexdigest(),
        source_label="other.json",
    )
    drifted_trace = deepcopy(fused.trace)
    drifted_trace["corpus"] = {
        **drifted_trace["corpus"],
        "source_artifact_sha256": other_corpus.source_artifact.sha256,
    }
    drifted = replace(fused, trace=drifted_trace)
    with pytest.raises(RawAnchorError, match="corpus boundary"):
        raw_top1_anchor_within_head(raw, drifted)

    digest_trace = deepcopy(fused.trace)
    digest_trace["value_scores_sha256"] = "0" * 64
    with pytest.raises(RawAnchorError, match="value-score digest"):
        raw_top1_anchor_within_head(raw, replace(fused, trace=digest_trace))
