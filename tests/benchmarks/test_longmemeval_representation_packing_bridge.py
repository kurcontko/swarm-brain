from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
from benchmarks.integrations.longmemeval_official_preflight import (
    DatasetRequirement,
    ExactTokenizerPin,
    freeze_pinned_preflight,
)
from benchmarks.integrations.longmemeval_representation import (
    EXTRACTOR_INPUT_FIELDS,
    CanonicalValue,
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
from benchmarks.integrations.longmemeval_representation.experiment import (
    RepresentationResult,
)
from benchmarks.integrations.longmemeval_representation.head_matched import (
    HEAD_MATCHED_VALUE_COUNT,
    HeadMatchedRepresentationError,
    head_match_representation_result,
)
from benchmarks.integrations.longmemeval_representation.packing_bridge import (
    RepresentationPackingBridgeError,
    pack_representation_result,
)
from benchmarks.integrations.longmemeval_turn_prompt import (
    LINEAR_TURN_SEPARATOR,
    PRIMARY_TOKEN_BUDGET,
    ExactTokenCountReceipt,
    TokenizerIdentity,
    TurnPromptPackingResult,
)
from benchmarks.integrations.longmemeval_turns import compile_dataset_bytes
from scripts._longmemeval_common import OFFICIAL_ANSWER_TEMPLATE


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _record(
    question_id: str,
    contents: list[str],
    *,
    question: str = "PRIVATE-QUESTION-MUST-NOT-ENTER-BRIDGE-TRACE",
    answer: str = "PRIVATE-GOLD-MUST-NOT-ENTER-PROMPT-OR-TRACE",
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question_type": "multi-session",
        "question": question,
        "answer": answer,
        "question_date": "2025/01/04 (Sat) 11:00",
        "haystack_session_ids": [f"session-{question_id}"],
        "haystack_dates": ["2025/01/03 (Fri) 09:07"],
        "haystack_sessions": [
            [
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": content,
                }
                for index, content in enumerate(contents)
            ]
        ],
        "answer_session_ids": [f"session-{question_id}"],
    }


def _source_bytes(records: list[dict[str, Any]]) -> bytes:
    return (
        json.dumps(records, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


class _ExactByteTokenizer:
    def __init__(self, identity: TokenizerIdentity) -> None:
        self._identity = identity
        self._request_id = 0

    @property
    def identity(self) -> TokenizerIdentity:
        return self._identity

    def count_prompt(self, prompt: str, *, query_sha256: str) -> ExactTokenCountReceipt:
        self._request_id += 1
        encoded = prompt.encode("utf-8")
        return ExactTokenCountReceipt(
            request_id=self._request_id,
            provider_request_id=f"fixture-exact-{self._request_id:04d}",
            tokenizer_identity_sha256=self.identity.identity_sha256,
            query_sha256=query_sha256,
            prompt_sha256=hashlib.sha256(encoded).hexdigest(),
            prompt_utf8_bytes=len(encoded),
            token_count=len(encoded),
        )


def _scorer() -> ScorerIdentity:
    return ScorerIdentity(
        producer="fixture-external-scorer",
        protocol="fixture-raw-rank-v1",
        model_id="fixture/ranker",
        model_revision="immutable-ranker-revision",
        model_artifact_sha256=_digest("ranker-model"),
        identity_artifact_sha256=_digest("ranker-identity"),
    )


def _representation_result(
    raw: bytes,
    *,
    question_id: str,
    ranking: tuple[int, ...],
) -> RepresentationResult:
    corpus = compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="synthetic-longmemeval.json",
    )
    values = compile_question_canonical_values(corpus, question_id=question_id)
    representation = RepresentationCorpus(
        projection_corpus=corpus,
        values=values,
        derived_keys=(),
        construction_receipts=(),
    )
    ranked = tuple(
        RankedKeyScore(key_id=raw_key_id(values[value_position]), raw_score=1000.0 - rank)
        for rank, value_position in enumerate(ranking, start=1)
    )
    observation = RankedFamilyObservation.create(
        family=KeyFamily.RAW,
        corpus=representation,
        query_sha256=_digest("fixed-query"),
        scorer=_scorer(),
        observation_artifact_sha256=_digest("raw-observation"),
        ranked_keys=ranked,
    )
    return evaluate_representation_cell(
        representation,
        cell=RepresentationCell.RAW,
        observations=(observation,),
    )


def _merged_representation_result(
    raw: bytes,
    *,
    question_id: str,
) -> RepresentationResult:
    corpus = compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="synthetic-longmemeval.json",
    )
    values = compile_question_canonical_values(corpus, question_id=question_id)
    extractor = ExtractorIdentity(
        producer="fixture-external-extractor",
        protocol="fixture-merged-sfk-json-v1",
        model_id="fixture/model",
        model_revision="immutable-revision",
        deployment_id="fixture-deployment",
        model_artifact_sha256=_digest("merged-model"),
        prompt_sha256=_digest("merged-prompt"),
        identity_artifact_sha256=_digest("merged-identity"),
    )
    keys: list[DerivedKey] = []
    receipts: list[ConstructionReceipt] = []
    for index, value in enumerate(values):
        receipt_id = opaque_navigation_id(prefix="receipt", material=f"receipt:{index}")
        key = DerivedKey.create(
            key_id=opaque_navigation_id(prefix="key", material=f"merged:{index}"),
            family=KeyFamily.MERGED_SFK,
            source=value,
            key_text=f"merged-key-{index:02d}",
            construction_receipt_id=receipt_id,
        )
        output_binding = derived_key_output_binding((key,))
        output_sha256 = hashlib.sha256(
            json.dumps(
                output_binding,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
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
    representation = RepresentationCorpus(
        projection_corpus=corpus,
        values=values,
        derived_keys=tuple(keys),
        construction_receipts=tuple(receipts),
    )

    def observation(family: KeyFamily, key_ids: tuple[str, ...]) -> RankedFamilyObservation:
        ranked = tuple(
            RankedKeyScore(key_id=key_id, raw_score=1000.0 - rank)
            for rank, key_id in enumerate(key_ids, start=1)
        )
        return RankedFamilyObservation.create(
            family=family,
            corpus=representation,
            query_sha256=_digest("fixed-query"),
            scorer=_scorer(),
            observation_artifact_sha256=_digest(f"observation:{family.value}"),
            ranked_keys=ranked,
        )

    raw = observation(
        KeyFamily.RAW,
        tuple(raw_key_id(value) for value in values[:20]),
    )
    merged = observation(
        KeyFamily.MERGED_SFK,
        tuple(key.key_id for key in keys[-20:]),
    )
    return evaluate_representation_cell(
        representation,
        cell=RepresentationCell.RAW_MERGED_SFK,
        observations=(raw, merged),
    )


def _fixture(
    contents: list[str] | None = None,
    *,
    ranking: tuple[int, ...] | None = None,
) -> tuple[
    bytes,
    dict[str, Any],
    RepresentationResult,
    Any,
    _ExactByteTokenizer,
]:
    record = _record("q-e6-pack", contents or ["source-zero", "source-one", "source-two"])
    raw = _source_bytes([record])
    pin = ExactTokenizerPin(
        model="fixture/exact-byte-tokenizer",
        revision="fixture-immutable-revision",
        artifact_sha256=_digest("tokenizer-artifact"),
        executable_sha256=_digest("tokenizer-executable"),
    )
    requirement = DatasetRequirement(
        name="Synthetic LongMemEval E6 packing rehearsal",
        source_label="synthetic-longmemeval.json",
        source_sha256=hashlib.sha256(raw).hexdigest(),
        question_count=1,
        official=False,
    )
    manifest = freeze_pinned_preflight(raw, dataset=requirement, tokenizer=pin)
    selected_ranking = ranking
    if selected_ranking is None:
        selected_ranking = (
            (2, 0, 1)
            if len(record["haystack_sessions"][0]) == 3
            else tuple(range(min(20, len(record["haystack_sessions"][0]))))
        )
    result = _representation_result(
        raw,
        question_id=record["question_id"],
        ranking=selected_ranking,
    )
    return raw, record, result, manifest, _ExactByteTokenizer(pin.identity)


def _pack(
    record: dict[str, Any],
    result: RepresentationResult,
    manifest: Any,
    tokenizer: _ExactByteTokenizer,
):
    return pack_representation_result(
        result,
        manifest=manifest,
        question_id=str(record["question_id"]),
        question=str(record["question"]),
        current_date=str(record["question_date"]),
        tokenizer=tokenizer,
    )


def test_bridge_preserves_hydrated_order_ids_and_exact_context_bytes() -> None:
    _raw, record, result, manifest, tokenizer = _fixture()

    bridge = _pack(record, result, manifest, tokenizer)

    expected_ids = [list(value.turn.turn_id.as_tuple()) for value in result.hydrated_values]
    expected_history = LINEAR_TURN_SEPARATOR.join(
        value.raw_value for value in result.hydrated_values
    )
    expected_history_bytes = expected_history.encode("utf-8")
    assert isinstance(bridge.packed, TurnPromptPackingResult)
    assert bridge.packed.trace["budget"]["token_budget"] == PRIMARY_TOKEN_BUDGET
    assert bridge.packed.trace["budget"]["is_primary"] is True
    assert bridge.packed.trace["candidate_blocks"] == [expected_ids]
    assert bridge.packed.trace["kept_ids"] == expected_ids
    assert bridge.prompt == OFFICIAL_ANSWER_TEMPLATE.format(
        expected_history,
        record["question_date"],
        record["question"],
    )
    assert bridge.trace["packing"]["final_context"] == {
        "sha256": hashlib.sha256(expected_history_bytes).hexdigest(),
        "utf8_bytes": len(expected_history_bytes),
    }
    assert [item["value_id"] for item in bridge.trace["representation"]["hydrated_values"]] == [
        value.value_id for value in result.hydrated_values
    ]
    assert bridge.trace["representation"]["trace_sha256"] == result.trace_sha256


def test_head_matched_result_is_deterministic_and_accepted_by_bridge() -> None:
    contents = [f"source-value-{index:02d}" for index in range(24)]
    raw, record, _raw_result, manifest, tokenizer = _fixture(contents)
    result = _merged_representation_result(raw, question_id=str(record["question_id"]))
    values_before = result.hydrated_values
    trace_before = deepcopy(result.trace)

    first = head_match_representation_result(result)
    second = head_match_representation_result(result)

    assert first == second
    assert first.trace_sha256 == second.trace_sha256
    assert len(first.hydrated_values) == HEAD_MATCHED_VALUE_COUNT
    assert first.hydrated_values == result.hydrated_values[:HEAD_MATCHED_VALUE_COUNT]
    assert first.trace["value_scores"] == result.trace["value_scores"][:HEAD_MATCHED_VALUE_COUNT]
    expected_scores = first.trace["value_scores"]
    assert (
        first.trace["value_scores_sha256"]
        == hashlib.sha256(
            json.dumps(
                expected_scores,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert first.trace["hydrated_value_ids"] == [value.value_id for value in first.hydrated_values]
    assert first.trace["hydrated_raw_value_hashes"] == [
        value.raw_value_sha256 for value in first.hydrated_values
    ]
    assert first.trace["hydrated_value_count"] == HEAD_MATCHED_VALUE_COUNT
    assert (
        first.trace["hydrated_value_pre_cap_count"] == result.trace["hydrated_value_pre_cap_count"]
    )
    assert first.trace["hydrated_value_cap"] == result.trace["hydrated_value_cap"]
    assert result.hydrated_values is values_before
    assert result.trace == trace_before

    bridge = _pack(record, first, manifest, tokenizer)
    assert bridge.packed.trace["candidate_blocks"] == [
        [list(value.turn.turn_id.as_tuple()) for value in first.hydrated_values]
    ]
    assert bridge.trace["representation"]["trace_sha256"] == first.trace_sha256


def test_head_matching_requires_twenty_bridge_valid_values() -> None:
    short_contents = [f"source-value-{index:02d}" for index in range(19)]
    _raw, _record_value, short, _manifest, _tokenizer = _fixture(short_contents)
    with pytest.raises(HeadMatchedRepresentationError, match="at least 20"):
        head_match_representation_result(short)

    exact_contents = [f"source-value-{index:02d}" for index in range(20)]
    _raw, _record_value, exact, _manifest, _tokenizer = _fixture(exact_contents)
    assert head_match_representation_result(exact) is exact

    contents = [f"source-value-{index:02d}" for index in range(21)]
    raw, record, _raw_result, _manifest, _tokenizer = _fixture(contents)
    result = _merged_representation_result(raw, question_id=str(record["question_id"]))
    drifted_trace = deepcopy(result.trace)
    drifted_trace["value_scores_sha256"] = "0" * 64
    drifted = replace(result, trace=drifted_trace)
    with pytest.raises(HeadMatchedRepresentationError, match="value-score digest"):
        head_match_representation_result(drifted)


def test_bridge_skips_oversized_value_and_continues_without_rewriting() -> None:
    _raw, record, result, manifest, tokenizer = _fixture(
        ["small-before", "small-after", "X" * 9000]
    )

    bridge = _pack(record, result, manifest, tokenizer)

    assert result.hydrated_values[0].turn.original_content == "X" * 9000
    assert bridge.packed.trace["kept_ids"] == [
        list(value.turn.turn_id.as_tuple()) for value in result.hydrated_values[1:]
    ]
    assert bridge.packed.trace["dropped_ids"] == [
        list(result.hydrated_values[0].turn.turn_id.as_tuple())
    ]
    assert bridge.packed.trace["oversized_ids"] == [
        list(result.hydrated_values[0].turn.turn_id.as_tuple())
    ]
    expected_history = LINEAR_TURN_SEPARATOR.join(
        value.raw_value for value in result.hydrated_values[1:]
    )
    assert expected_history in bridge.prompt
    assert "X" * 100 not in bridge.prompt
    assert bridge.packed.trace["final_prompt"]["tokens"] <= PRIMARY_TOKEN_BUDGET


def test_bridge_trace_is_content_free_and_declares_no_gold_consumption() -> None:
    _raw, record, result, manifest, tokenizer = _fixture()

    bridge = _pack(record, result, manifest, tokenizer)
    artifact = json.dumps(bridge.content_free_artifact(), ensure_ascii=False, sort_keys=True)

    assert record["question"] not in artifact
    assert record["answer"] not in artifact
    assert all(value.turn.original_content not in artifact for value in result.hydrated_values)
    assert bridge.trace["claims"]["gold_fields_consumed"] is False
    assert bridge.trace["claims"]["derived_navigation_keys_delivered_to_reader"] is False
    assert record["answer"] not in bridge.prompt


@pytest.mark.parametrize("field", ("question", "question_date"))
def test_bridge_rejects_question_or_date_drift(field: str) -> None:
    _raw, record, result, manifest, tokenizer = _fixture()
    mutated = dict(record)
    mutated[field] = f"{record[field]}-drift"

    with pytest.raises(RepresentationPackingBridgeError, match="preflight case"):
        _pack(mutated, result, manifest, tokenizer)


def test_bridge_rejects_tokenizer_identity_outside_preflight_pin() -> None:
    _raw, record, result, manifest, _tokenizer = _fixture()
    other = _ExactByteTokenizer(
        TokenizerIdentity(
            protocol=manifest.tokenizer.protocol,
            model=manifest.tokenizer.model,
            revision="different-immutable-revision",
            artifact_sha256=manifest.tokenizer.artifact_sha256,
        )
    )

    with pytest.raises(RepresentationPackingBridgeError, match="preflight pin"):
        _pack(record, result, manifest, other)


def test_bridge_rejects_representation_from_another_source_and_projection() -> None:
    _raw, record, _result, manifest, tokenizer = _fixture()
    other_record = _record("q-e6-pack", ["forged-other-source"])
    other_raw = _source_bytes([other_record])
    other_result = _representation_result(other_raw, question_id="q-e6-pack", ranking=(0,))

    with pytest.raises(RepresentationPackingBridgeError, match="source_artifact_sha256"):
        _pack(record, other_result, manifest, tokenizer)


def test_bridge_rejects_representation_trace_schema_or_score_order_drift() -> None:
    _raw, record, result, manifest, tokenizer = _fixture()
    extra_trace = deepcopy(result.trace)
    extra_trace["answer"] = "forbidden-covert-field"
    extra = replace(result, trace=extra_trace)

    with pytest.raises(RepresentationPackingBridgeError, match="frozen schema"):
        _pack(record, extra, manifest, tokenizer)

    score_trace = deepcopy(result.trace)
    score_trace["value_scores"][0], score_trace["value_scores"][1] = (
        score_trace["value_scores"][1],
        score_trace["value_scores"][0],
    )
    score_trace["value_scores_sha256"] = hashlib.sha256(
        json.dumps(
            score_trace["value_scores"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    score_drift = replace(result, trace=score_trace)

    with pytest.raises(RepresentationPackingBridgeError, match="value-score order"):
        _pack(record, score_drift, manifest, tokenizer)


def test_bridge_rejects_non_result_and_non_manifest() -> None:
    _raw, record, result, manifest, tokenizer = _fixture()

    with pytest.raises(RepresentationPackingBridgeError, match="RepresentationResult"):
        pack_representation_result(
            object(),  # type: ignore[arg-type]
            manifest=manifest,
            question_id=record["question_id"],
            question=record["question"],
            current_date=record["question_date"],
            tokenizer=tokenizer,
        )
    with pytest.raises(RepresentationPackingBridgeError, match="RunPreflightManifest"):
        _pack(record, result, object(), tokenizer)


def test_representation_result_input_remains_unmodified() -> None:
    _raw, record, result, manifest, tokenizer = _fixture()
    values_before: tuple[CanonicalValue, ...] = result.hydrated_values
    trace_before = deepcopy(result.trace)

    _pack(record, result, manifest, tokenizer)

    assert result.hydrated_values == values_before
    assert result.trace == trace_before
