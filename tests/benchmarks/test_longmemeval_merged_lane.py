from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
from benchmarks.integrations.longmemeval_representation import (
    EXTRACTOR_INPUT_FIELDS,
    ConstructionReceipt,
    DerivedKey,
    ExtractorIdentity,
    KeyFamily,
    RankedFamilyObservation,
    RankedKeyScore,
    RepresentationCell,
    RepresentationCorpus,
    ScorerIdentity,
    compile_question_canonical_values,
    derived_key_output_binding,
    evaluate_representation_cell,
    extraction_request_sha256,
    opaque_navigation_id,
    raw_key_id,
)
from benchmarks.integrations.longmemeval_representation.merged_lane import (
    MERGED_LANE_PROTOCOL,
    MergedLaneSelectionError,
    select_merged_lane_top20,
)
from benchmarks.integrations.longmemeval_turns import compile_dataset_bytes


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _source() -> bytes:
    record: dict[str, Any] = {
        "question_id": "q-merged-lane",
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


def _fixture():
    raw = _source()
    projection = compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="fixture.json",
    )
    values = compile_question_canonical_values(projection, question_id="q-merged-lane")
    extractor = ExtractorIdentity(
        producer="fixture",
        protocol="fixture-merged-v1",
        model_id="fixture/model",
        model_revision="revision",
        deployment_id="fixture-deployment",
        model_artifact_sha256=_digest("extractor-model"),
        prompt_sha256=_digest("prompt"),
        identity_artifact_sha256=_digest("extractor-identity"),
    )
    keys: list[DerivedKey] = []
    receipts: list[ConstructionReceipt] = []
    for index, value in enumerate(values):
        receipt_id = opaque_navigation_id(prefix="receipt", material=f"receipt:{index}")
        key = DerivedKey.create(
            key_id=opaque_navigation_id(prefix="key", material=f"merged:{index}"),
            family=KeyFamily.MERGED_SFK,
            source=value,
            key_text=f"merged-{index:02d}",
            construction_receipt_id=receipt_id,
        )
        output = derived_key_output_binding((key,))
        output_sha256 = hashlib.sha256(
            json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        receipts.append(
            ConstructionReceipt(
                receipt_id=receipt_id,
                family=KeyFamily.MERGED_SFK,
                source_value_id=value.value_id,
                question_id=value.question_id,
                source_artifact_sha256=value.source_artifact_sha256,
                projection_sha256=value.projection_sha256,
                source_version_sha256=value.source_version_sha256,
                raw_value_sha256=value.raw_value_sha256,
                raw_value_utf8_bytes=value.raw_value_utf8_bytes,
                extractor=extractor,
                construction_artifact_sha256=_digest(f"construction:{index}"),
                input_fields=EXTRACTOR_INPUT_FIELDS,
                source_input_sha256=value.raw_value_sha256,
                request_sha256=extraction_request_sha256(
                    family=KeyFamily.MERGED_SFK,
                    raw_value_sha256=value.raw_value_sha256,
                    raw_value_utf8_bytes=value.raw_value_utf8_bytes,
                    extractor=extractor,
                ),
                response_sha256=_digest(f"response:{index}"),
                output_key_ids=(key.key_id,),
                output_keys_sha256=output_sha256,
                input_tokens=1,
                output_tokens=1,
                latency_microseconds=1,
                cost_microusd=1,
            )
        )
        keys.append(key)
    corpus = RepresentationCorpus(
        projection_corpus=projection,
        values=values,
        derived_keys=tuple(keys),
        construction_receipts=tuple(receipts),
    )

    def observation(family: KeyFamily, ids: tuple[str, ...], label: str):
        return RankedFamilyObservation.create(
            family=family,
            corpus=corpus,
            query_sha256=_digest("query"),
            scorer=_scorer(),
            observation_artifact_sha256=_digest(label),
            ranked_keys=tuple(
                RankedKeyScore(key_id=key_id, raw_score=100.0 - rank)
                for rank, key_id in enumerate(ids, start=1)
            ),
        )

    raw_observation = observation(
        KeyFamily.RAW,
        tuple(raw_key_id(value) for value in values),
        "raw-observation",
    )
    merged_order = tuple(reversed(range(20)))
    merged_observation = observation(
        KeyFamily.MERGED_SFK,
        tuple(keys[index].key_id for index in merged_order),
        "merged-observation",
    )
    source = evaluate_representation_cell(
        corpus,
        cell=RepresentationCell.RAW_MERGED_SFK,
        observations=(raw_observation, merged_observation),
    )
    return corpus, merged_observation, source, values, merged_order


def test_merged_lane_selects_key_order_and_hydrates_only_raw_values() -> None:
    corpus, observation, source, values, merged_order = _fixture()
    source_before = deepcopy(source.trace)

    result = select_merged_lane_top20(source, corpus=corpus, observation=observation)

    assert result.hydrated_values == tuple(values[index] for index in merged_order)
    assert all(
        selected.raw_value == values[index].raw_value
        for selected, index in zip(result.hydrated_values, merged_order, strict=True)
    )
    assert result.trace["ranking"]["method"] == MERGED_LANE_PROTOCOL
    assert result.trace["ranking"]["family_accounting"] == {
        "merged-sfk": {
            "indexed_keys": 20,
            "returned_key_hits": 20,
            "unique_values_reached": 20,
            "same_family_fanout_hits_suppressed": 0,
        }
    }
    assert result.trace["frozen_protocol"]["key_families"] == ["merged-sfk"]
    assert result.trace["hydration"]["derived_keys_delivered_to_reader"] is False
    assert source.trace == source_before


def test_merged_lane_is_deterministic_and_rejects_tampering() -> None:
    corpus, observation, source, _values, _order = _fixture()

    first = select_merged_lane_top20(source, corpus=corpus, observation=observation)
    second = select_merged_lane_top20(source, corpus=corpus, observation=observation)
    assert first == second
    assert first.trace_sha256 == second.trace_sha256

    drifted_trace = deepcopy(source.trace)
    drifted_trace["value_scores_sha256"] = "0" * 64
    with pytest.raises(MergedLaneSelectionError, match="value-score digest"):
        select_merged_lane_top20(
            replace(source, trace=drifted_trace),
            corpus=corpus,
            observation=observation,
        )

    drifted_trace = deepcopy(source.trace)
    drifted_trace["cell"] = RepresentationCell.RAW.value
    with pytest.raises(MergedLaneSelectionError, match="trace cell"):
        select_merged_lane_top20(
            replace(source, trace=drifted_trace),
            corpus=corpus,
            observation=observation,
        )

    drifted_trace = deepcopy(source.trace)
    drifted_trace["corpus"]["index_sha256"] = _digest("wrong-index")
    with pytest.raises(MergedLaneSelectionError, match="bindings differ"):
        select_merged_lane_top20(
            replace(source, trace=drifted_trace),
            corpus=corpus,
            observation=observation,
        )

    raw_observation = next(
        item for item in source.trace["observations"] if item["family"] == KeyFamily.RAW.value
    )
    assert raw_observation
    with pytest.raises(MergedLaneSelectionError, match="RankedFamilyObservation"):
        select_merged_lane_top20(source, corpus=corpus, observation=raw_observation)  # type: ignore[arg-type]
