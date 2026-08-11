from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from benchmarks.integrations.longmemeval_representation import (
    EXTRACTOR_INPUT_FIELDS,
    GRAPH_INPUT_FIELDS,
    KEY_FAMILY_DEPTH,
    PROTOCOL_VERSION,
    CanonicalValue,
    ConstructionReceipt,
    DerivedKey,
    EntityAdjacencyEdge,
    EntityAdjacencyGraph,
    ExtractorIdentity,
    GraphAccounting,
    GraphIdentity,
    KeyFamily,
    RankedFamilyObservation,
    RankedKeyScore,
    RepresentationCell,
    RepresentationCorpus,
    RepresentationError,
    ScorerIdentity,
    SimilarityAdjacencyEdge,
    SimilarityAdjacencyGraph,
    ValueProvenance,
    compile_question_canonical_values,
    derived_key_output_binding,
    evaluate_representation_cell,
    extraction_request_sha256,
    graph_request_sha256,
    opaque_navigation_id,
    question_value_binding_payload,
)
from benchmarks.integrations.longmemeval_turns import compile_dataset_bytes

_PROJECTION_CORPORA: dict[str, Any] = {}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _opaque(prefix: str, label: str) -> str:
    return opaque_navigation_id(prefix=prefix, material=label)


def _key_id(family: KeyFamily, value_index: int, key_index: int) -> str:
    return _opaque("key", f"{family.value}:{value_index}:{key_index}")


def _entity_endpoints(left: str, right: str) -> tuple[str, str]:
    first, second = sorted((_opaque("entity", left), _opaque("entity", right)))
    return first, second


def _record(
    question_id: str,
    contents: list[str],
    *,
    question: str = "PRIVATE-QUESTION-MUST-NOT-ENTER-TRACE",
    question_type: str = "multi-session",
    answer: str = "PRIVATE-GOLD-MUST-NOT-ENTER-TRACE",
    answer_sessions: list[str] | None = None,
) -> dict[str, Any]:
    session_id = f"session-{question_id}"
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question,
        "answer": answer,
        "question_date": "2025/01/04 (Sat) 11:00",
        "haystack_session_ids": [session_id],
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
        "answer_session_ids": answer_sessions or [session_id],
    }


def _compile_values(records: list[dict[str, Any]]) -> tuple[CanonicalValue, ...]:
    raw = (
        json.dumps(records, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    turns = compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="synthetic-longmemeval.json",
    )
    _PROJECTION_CORPORA[turns.projection_sha256] = turns
    question_ids = {record["question_id"] for record in records}
    return tuple(
        value
        for question_id in question_ids
        for value in compile_question_canonical_values(turns, question_id=str(question_id))
    )


def test_canonical_value_compilation_reuses_validated_projection_digest(monkeypatch) -> None:
    records = [_record("cache-case", ["first", "second", "third"])]
    raw = (
        json.dumps(records, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    turns = compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="synthetic-longmemeval.json",
    )
    expected_projection = turns.projection_sha256

    def fail_if_recomputed(_self) -> dict[str, object]:
        raise AssertionError("immutable projection payload was recomputed")

    monkeypatch.setattr(type(turns), "_projection_payload", fail_if_recomputed)

    first = compile_question_canonical_values(turns, question_id="cache-case")
    second = compile_question_canonical_values(turns, question_id="cache-case")
    assert first == second
    assert {value.projection_sha256 for value in first} == {expected_projection}


def _extractor(family: KeyFamily) -> ExtractorIdentity:
    return ExtractorIdentity(
        producer="fixture-external-extractor",
        protocol=f"fixture-{family.value}-json-v1",
        model_id="fixture/model",
        model_revision="immutable-revision",
        deployment_id="fixture-deployment",
        model_artifact_sha256=_digest(f"model:{family.value}"),
        prompt_sha256=_digest(f"prompt:{family.value}"),
        identity_artifact_sha256=_digest(f"identity:{family.value}"),
    )


KeySpec = Callable[[CanonicalValue, int], list[tuple[str, str | None]]]


def _construct_family(
    values: tuple[CanonicalValue, ...],
    family: KeyFamily,
    key_spec: KeySpec,
) -> tuple[tuple[DerivedKey, ...], tuple[ConstructionReceipt, ...]]:
    extractor = _extractor(family)
    all_keys: list[DerivedKey] = []
    receipts: list[ConstructionReceipt] = []
    for value_index, value in enumerate(values):
        receipt_id = _opaque("receipt", f"{family.value}:{value_index}")
        keys = tuple(
            DerivedKey.create(
                key_id=_key_id(family, value_index, key_index),
                family=family,
                source=value,
                key_text=text,
                construction_receipt_id=receipt_id,
                entity_id=(None if entity_id is None else _opaque("entity", entity_id)),
            )
            for key_index, (text, entity_id) in enumerate(key_spec(value, value_index))
        )
        request_sha256 = extraction_request_sha256(
            family=family,
            raw_value_sha256=value.raw_value_sha256,
            raw_value_utf8_bytes=value.raw_value_utf8_bytes,
            extractor=extractor,
        )
        receipts.append(
            ConstructionReceipt(
                receipt_id=receipt_id,
                family=family,
                source_value_id=value.value_id,
                question_id=value.question_id,
                source_artifact_sha256=value.source_artifact_sha256,
                projection_sha256=value.projection_sha256,
                source_version_sha256=value.source_version_sha256,
                raw_value_sha256=value.raw_value_sha256,
                raw_value_utf8_bytes=value.raw_value_utf8_bytes,
                extractor=extractor,
                construction_artifact_sha256=_digest(f"construction:{receipt_id}"),
                input_fields=EXTRACTOR_INPUT_FIELDS,
                source_input_sha256=value.raw_value_sha256,
                request_sha256=request_sha256,
                response_sha256=_digest(f"response:{receipt_id}"),
                output_key_ids=tuple(key.key_id for key in keys),
                output_keys_sha256=_digest_binding(keys),
                input_tokens=100 + value_index,
                output_tokens=10 + len(keys),
                latency_microseconds=1000 + value_index,
                cost_microusd=5 + len(keys),
                retry_count=value_index % 2,
                cache_hit=False,
                complete=True,
            )
        )
        all_keys.extend(keys)
    return tuple(all_keys), tuple(receipts)


def _digest_binding(keys: tuple[DerivedKey, ...]) -> str:
    payload = derived_key_output_binding(keys)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _corpus(
    values: tuple[CanonicalValue, ...],
    specs: dict[KeyFamily, KeySpec] | None = None,
) -> RepresentationCorpus:
    keys: list[DerivedKey] = []
    receipts: list[ConstructionReceipt] = []
    for family, spec in (specs or {}).items():
        family_keys, family_receipts = _construct_family(values, family, spec)
        keys.extend(family_keys)
        receipts.extend(family_receipts)
    return RepresentationCorpus(
        projection_corpus=_PROJECTION_CORPORA[values[0].projection_sha256],
        values=values,
        derived_keys=tuple(keys),
        construction_receipts=tuple(receipts),
    )


def _scorer(family: KeyFamily) -> ScorerIdentity:
    return ScorerIdentity(
        producer="fixture-external-scorer",
        protocol=f"fixture-{family.value}-rank-v1",
        model_id="fixture/ranker",
        model_revision="immutable-ranker-revision",
        model_artifact_sha256=_digest(f"ranker:{family.value}"),
        identity_artifact_sha256=_digest(f"ranker-identity:{family.value}"),
    )


def _observe(
    corpus: RepresentationCorpus,
    family: KeyFamily,
    *,
    ordered_key_ids: list[str] | None = None,
    equal_scores: bool = False,
) -> RankedFamilyObservation:
    keys = corpus.keys_for_family(family)
    default_ids = [key.key_id for key in sorted(keys, key=lambda item: item.key_id)]
    ids = (ordered_key_ids or default_ids)[:KEY_FAMILY_DEPTH]
    ranked = tuple(
        RankedKeyScore(
            key_id=key_id,
            raw_score=1.0 if equal_scores else 1000.0 - rank,
        )
        for rank, key_id in enumerate(ids, start=1)
    )
    if equal_scores:
        ranked = tuple(sorted(ranked, key=lambda item: item.key_id))
    return RankedFamilyObservation.create(
        family=family,
        corpus=corpus,
        query_sha256=_digest("fixed-query"),
        scorer=_scorer(family),
        observation_artifact_sha256=_digest(f"observation:{family.value}:{corpus.index_sha256}"),
        ranked_keys=ranked,
    )


def _graph_identity(kind: str) -> GraphIdentity:
    return GraphIdentity(
        producer="fixture-graph-builder",
        protocol=f"fixture-{kind}-v1",
        model_id="fixture/graph-model",
        model_revision="immutable-graph-revision",
        model_artifact_sha256=_digest(f"graph-model:{kind}"),
        prompt_sha256=_digest(f"graph-prompt:{kind}"),
        identity_artifact_sha256=_digest(f"graph-identity:{kind}"),
    )


def _graph_accounting() -> GraphAccounting:
    return GraphAccounting(
        input_tokens=20,
        output_tokens=5,
        latency_microseconds=100,
        cost_microusd=2,
    )


def test_r0_hydrates_exact_raw_values_and_trace_is_content_free() -> None:
    values = _compile_values([_record("q-r0", ["RAW-PRIVATE-ONE", "RAW-PRIVATE-TWO"])])
    corpus = _corpus(values)
    observation = _observe(corpus, KeyFamily.RAW)

    result = evaluate_representation_cell(
        corpus,
        cell=RepresentationCell.RAW,
        observations=(observation,),
    )

    assert result.hydrated_raw_values == tuple(value.raw_value for value in result.hydrated_values)
    assert {value.value_id for value in result.hydrated_values} == {
        value.value_id for value in values
    }
    assert result.trace["hydration"] == {
        "reader_evidence": "canonical-raw-value",
        "derived_keys_delivered_to_reader": False,
        "source_values_byte_identical": True,
    }
    assert result.trace["frozen_protocol"]["rrf_k"] == 60
    assert result.trace["promotion"]["cell_intrinsically_ineligible"] is False
    assert result.trace["construction_input_contract"] == {
        "gold_question_type_answer_or_judge_fields_allowed": False,
        "question_text_allowed_for_key_or_graph_construction": False,
        "request_material_digests_recomputed": True,
        "external_execution_identity_attested_not_verified": True,
    }
    trace = json.dumps(result.content_free_artifact(), ensure_ascii=False)
    assert "RAW-PRIVATE-ONE" not in trace
    assert "RAW-PRIVATE-TWO" not in trace
    assert "PRIVATE-QUESTION-MUST-NOT-ENTER-TRACE" not in trace
    assert "PRIVATE-GOLD-MUST-NOT-ENTER-TRACE" not in trace


def test_r1_requires_one_merged_key_and_adds_one_equal_family_contribution() -> None:
    values = _compile_values([_record("q-r1", ["one", "two"])])
    corpus = _corpus(
        values,
        {KeyFamily.MERGED_SFK: lambda _value, index: [(f"merged-{index}", None)]},
    )
    raw = _observe(corpus, KeyFamily.RAW)
    merged = _observe(corpus, KeyFamily.MERGED_SFK)

    result = evaluate_representation_cell(
        corpus,
        cell=RepresentationCell.RAW_MERGED_SFK,
        observations=(raw, merged),
    )

    first = result.trace["value_scores"][0]
    assert {item["family"] for item in first["family_contributions"]} == {
        "raw",
        "merged-sfk",
    }
    assert first["score"] == pytest.approx(
        sum(item["rrf_contribution"] for item in first["family_contributions"])
    )
    accounting = result.trace["construction_and_index_accounting"]
    assert accounting["construction_receipt_count"] == len(values)
    assert accounting["construction_accounting"]["input_tokens"] == 201
    assert accounting["orphan_keys"]["count"] == 0
    assert accounting["update_rate"]["updates"] == 0


def test_r2_same_family_fanout_contributes_only_best_rank() -> None:
    values = _compile_values([_record("q-r2", ["one", "two"])])
    corpus = _corpus(
        values,
        {
            KeyFamily.SUMMARY: lambda _value, index: [(f"summary-{index}", None)],
            KeyFamily.FACT: lambda _value, index: (
                [("fact-best", None), ("fact-extra", None)]
                if index == 0
                else [("fact-other", None)]
            ),
            KeyFamily.KEYWORD: lambda _value, index: [(f"keyword-{index}", None)],
        },
    )
    raw = _observe(corpus, KeyFamily.RAW)
    summary = _observe(corpus, KeyFamily.SUMMARY)
    fact_ids = [
        _key_id(KeyFamily.FACT, 0, 0),
        _key_id(KeyFamily.FACT, 1, 0),
        _key_id(KeyFamily.FACT, 0, 1),
    ]
    facts = _observe(corpus, KeyFamily.FACT, ordered_key_ids=fact_ids)
    keywords = _observe(corpus, KeyFamily.KEYWORD)

    result = evaluate_representation_cell(
        corpus,
        cell=RepresentationCell.RAW_SEPARATE_SFK,
        observations=(raw, summary, facts, keywords),
    )

    value_zero = next(
        item
        for item in result.trace["value_scores"]
        if item["value"]["value_id"] == values[0].value_id
    )
    fact = next(item for item in value_zero["family_contributions"] if item["family"] == "fact")
    assert fact["best_rank"] == 1
    assert fact["suppressed_same_family_hits"] == 1
    assert fact["rrf_contribution"] == pytest.approx(1.0 / 61.0)
    assert result.trace["ranking"]["family_accounting"]["fact"] == {
        "indexed_keys": 3,
        "returned_key_hits": 3,
        "unique_values_reached": 2,
        "same_family_fanout_hits_suppressed": 1,
    }


def test_rrf_tie_uses_best_prior_rank_then_canonical_value_id() -> None:
    values = _compile_values([_record("q-tie", ["one", "two"])])
    corpus = _corpus(
        values,
        {KeyFamily.MERGED_SFK: lambda _value, index: [(f"merged-{index}", None)]},
    )
    raw_ids = [key.key_id for key in corpus.keys_for_family(KeyFamily.RAW)]
    merged_ids = [key.key_id for key in corpus.keys_for_family(KeyFamily.MERGED_SFK)]
    raw = _observe(corpus, KeyFamily.RAW, ordered_key_ids=raw_ids)
    merged = _observe(corpus, KeyFamily.MERGED_SFK, ordered_key_ids=list(reversed(merged_ids)))

    result = evaluate_representation_cell(
        corpus,
        cell=RepresentationCell.RAW_MERGED_SFK,
        observations=(raw, merged),
    )

    expected = sorted(value.value_id for value in values)
    assert result.trace["hydrated_value_ids"] == expected


def test_r3_uses_only_abstraction_and_cues_but_hydrates_raw_values() -> None:
    values = _compile_values([_record("q-r3", ["raw-one", "raw-two"])])
    corpus = _corpus(
        values,
        {
            KeyFamily.PRIMARY_ABSTRACTION: lambda _value, index: [(f"abstraction-{index}", None)],
            KeyFamily.CUE_ANCHOR: lambda _value, index: [
                (f"cue-{index}-a", None),
                (f"cue-{index}-b", None),
            ],
        },
    )
    abstraction = _observe(corpus, KeyFamily.PRIMARY_ABSTRACTION)
    cues = _observe(corpus, KeyFamily.CUE_ANCHOR)

    result = evaluate_representation_cell(
        corpus,
        cell=RepresentationCell.ABSTRACTION_CUES,
        observations=(abstraction, cues),
    )

    assert result.trace["frozen_protocol"]["key_families"] == [
        "primary-abstraction",
        "cue-anchor",
    ]
    assert result.hydrated_raw_values
    assert all("raw" not in item["family"] for item in result.trace["observations"])
    with pytest.raises(RepresentationError, match="exact key-family order"):
        evaluate_representation_cell(
            corpus,
            cell=RepresentationCell.ABSTRACTION_CUES,
            observations=(_observe(corpus, KeyFamily.RAW), abstraction, cues),
        )


def test_r4_direct_entity_activation_fans_out_values_and_rejects_graph() -> None:
    values = _compile_values([_record("q-r4", ["one", "two", "three"])])

    def entities(_value: CanonicalValue, index: int) -> list[tuple[str, str | None]]:
        if index == 0:
            return [("shared description a", "entity-shared")]
        if index == 1:
            return [("shared description b", "entity-shared")]
        return [("separate", "entity-z")]

    corpus = _corpus(values, {KeyFamily.ENTITY_DESCRIPTION: entities})
    entity_keys = corpus.keys_for_family(KeyFamily.ENTITY_DESCRIPTION)
    observation = _observe(
        corpus,
        KeyFamily.ENTITY_DESCRIPTION,
        ordered_key_ids=[entity_keys[0].key_id, entity_keys[2].key_id, entity_keys[1].key_id],
    )
    result = evaluate_representation_cell(
        corpus,
        cell=RepresentationCell.ENTITY_DIRECT,
        observations=(observation,),
    )
    assert {value.value_id for value in result.hydrated_values} == {
        value.value_id for value in values
    }
    assert result.trace["ranking"]["expansion_hops"] == 0
    assert result.trace["ranking"]["adjacency_consulted"] is False

    graph = EntityAdjacencyGraph.create(
        corpus=corpus,
        identity=_graph_identity("entity"),
        graph_artifact_sha256=_digest("entity-graph"),
        response_sha256=_digest("entity-response"),
        accounting=_graph_accounting(),
        edges=(),
    )
    with pytest.raises(RepresentationError, match="cannot consume adjacency"):
        evaluate_representation_cell(
            corpus,
            cell=RepresentationCell.ENTITY_DIRECT,
            observations=(observation,),
            adjacency=graph,
        )


def _entity_expansion_fixture() -> tuple[
    RepresentationCorpus,
    tuple[CanonicalValue, ...],
    RankedFamilyObservation,
]:
    values = _compile_values([_record("q-r5", ["seed-value", "other-value", "target-value"])])

    def entities(_value: CanonicalValue, index: int) -> list[tuple[str, str | None]]:
        if index == 0:
            return [(f"seed-{item}", f"entity-seed-{item:02d}") for item in range(19)]
        if index == 1:
            return [("other", "entity-other")]
        return [("target", "entity-target")]

    corpus = _corpus(values, {KeyFamily.ENTITY_DESCRIPTION: entities})
    keys = list(corpus.keys_for_family(KeyFamily.ENTITY_DESCRIPTION))
    target_key = next(key for key in keys if key.entity_id == _opaque("entity", "entity-target"))
    top = [key.key_id for key in keys if key.key_id != target_key.key_id][:KEY_FAMILY_DEPTH]
    assert len(top) == KEY_FAMILY_DEPTH
    observation = _observe(
        corpus,
        KeyFamily.ENTITY_DESCRIPTION,
        ordered_key_ids=top,
    )
    return corpus, values, observation


def test_r5_expands_exactly_one_entity_hop_and_hydrates_target_once() -> None:
    corpus, values, observation = _entity_expansion_fixture()
    left, right = _entity_endpoints("entity-seed-00", "entity-target")
    secret_edge_id = "SECRET-EDGE-ID-CANARY"
    edge = EntityAdjacencyEdge(
        edge_id=_opaque("edge", secret_edge_id),
        left_entity_id=left,
        right_entity_id=right,
        evidence_values=tuple(
            sorted(
                (
                    ValueProvenance.from_value(values[0]),
                    ValueProvenance.from_value(values[2]),
                ),
                key=lambda item: item.value_id,
            )
        ),
        edge_artifact_sha256=_digest("entity-edge"),
    )
    graph = EntityAdjacencyGraph.create(
        corpus=corpus,
        identity=_graph_identity("entity"),
        graph_artifact_sha256=_digest("entity-graph"),
        response_sha256=_digest("entity-response"),
        accounting=_graph_accounting(),
        edges=(edge,),
    )

    direct = evaluate_representation_cell(
        corpus,
        cell=RepresentationCell.ENTITY_DIRECT,
        observations=(observation,),
    )
    result = evaluate_representation_cell(
        corpus,
        cell=RepresentationCell.ENTITY_ONE_HOP,
        observations=(observation,),
        adjacency=graph,
    )

    assert values[2].value_id not in {value.value_id for value in direct.hydrated_values}
    assert [value.value_id for value in result.hydrated_values].count(values[2].value_id) == 1
    target = next(
        item
        for item in result.trace["value_scores"]
        if item["value"]["value_id"] == values[2].value_id
    )
    assert target["expanded"] is True
    assert target["graph_support_count"] == 1
    assert target["best_prior_rank"] is None
    assert result.trace["ranking"]["expansion_hops"] == 1
    assert result.trace["ranking"]["recursive_expansion"] is False
    assert result.trace["ranking"]["traversed_edge_ids"] == [edge.edge_id]
    assert secret_edge_id not in json.dumps(result.content_free_artifact())


def test_r5_rejects_orphan_entity_and_stale_value_provenance_even_when_rehashed() -> None:
    corpus, values, observation = _entity_expansion_fixture()
    orphan_left, orphan_right = _entity_endpoints("entity-seed-00", "entity-unknown")
    orphan = EntityAdjacencyEdge(
        edge_id=_opaque("edge", "entity-seed-00:entity-unknown"),
        left_entity_id=orphan_left,
        right_entity_id=orphan_right,
        evidence_values=(ValueProvenance.from_value(values[0]),),
        edge_artifact_sha256=_digest("orphan-edge"),
    )
    orphan_graph = EntityAdjacencyGraph.create(
        corpus=corpus,
        identity=_graph_identity("entity"),
        graph_artifact_sha256=_digest("orphan-graph"),
        response_sha256=_digest("orphan-response"),
        accounting=_graph_accounting(),
        edges=(orphan,),
    )
    with pytest.raises(RepresentationError, match="orphan endpoint"):
        evaluate_representation_cell(
            corpus,
            cell=RepresentationCell.ENTITY_ONE_HOP,
            observations=(observation,),
            adjacency=orphan_graph,
        )

    stale_left, stale_right = _entity_endpoints("entity-seed-00", "entity-target")
    stale = EntityAdjacencyEdge(
        edge_id=_opaque("edge", "entity-seed-00:entity-target"),
        left_entity_id=stale_left,
        right_entity_id=stale_right,
        evidence_values=tuple(
            sorted(
                (
                    replace(
                        ValueProvenance.from_value(values[0]),
                        raw_value_sha256="f" * 64,
                    ),
                    ValueProvenance.from_value(values[2]),
                ),
                key=lambda item: item.value_id,
            )
        ),
        edge_artifact_sha256=_digest("stale-edge"),
    )
    stale_graph = EntityAdjacencyGraph.create(
        corpus=corpus,
        identity=_graph_identity("entity"),
        graph_artifact_sha256=_digest("stale-graph"),
        response_sha256=_digest("stale-response"),
        accounting=_graph_accounting(),
        edges=(stale,),
    )
    with pytest.raises(RepresentationError, match="raw value hash"):
        evaluate_representation_cell(
            corpus,
            cell=RepresentationCell.ENTITY_ONE_HOP,
            observations=(observation,),
            adjacency=stale_graph,
        )


def test_r5_rejects_digest_consistent_edge_with_unrelated_evidence_value() -> None:
    corpus, values, observation = _entity_expansion_fixture()
    left, right = _entity_endpoints("entity-seed-00", "entity-target")
    unrelated = EntityAdjacencyEdge(
        edge_id=_opaque("edge", "unrelated-evidence"),
        left_entity_id=left,
        right_entity_id=right,
        evidence_values=(ValueProvenance.from_value(values[1]),),
        edge_artifact_sha256=_digest("unrelated-evidence-edge"),
    )
    graph = EntityAdjacencyGraph.create(
        corpus=corpus,
        identity=_graph_identity("entity"),
        graph_artifact_sha256=_digest("unrelated-evidence-graph"),
        response_sha256=_digest("unrelated-evidence-response"),
        accounting=_graph_accounting(),
        edges=(unrelated,),
    )

    with pytest.raises(RepresentationError, match="unrelated to both"):
        evaluate_representation_cell(
            corpus,
            cell=RepresentationCell.ENTITY_ONE_HOP,
            observations=(observation,),
            adjacency=graph,
        )


def _similarity_fixture() -> tuple[
    RepresentationCorpus,
    tuple[CanonicalValue, ...],
    RankedFamilyObservation,
]:
    values = _compile_values([_record("q-rneg", [f"value-{index}" for index in range(21)])])
    corpus = _corpus(values)
    raw_keys = list(corpus.keys_for_family(KeyFamily.RAW))
    ordered = [key.key_id for key in raw_keys[:KEY_FAMILY_DEPTH]]
    observation = _observe(corpus, KeyFamily.RAW, ordered_key_ids=ordered)
    return corpus, values, observation


def test_rneg_one_hop_similarity_expansion_is_hard_nonpromotable() -> None:
    corpus, values, observation = _similarity_fixture()
    edge = SimilarityAdjacencyEdge(
        edge_id=_opaque("edge", "value-0:value-20"),
        left_value=ValueProvenance.from_value(values[0]),
        right_value=ValueProvenance.from_value(values[20]),
        similarity_score=0.9,
        edge_artifact_sha256=_digest("similarity-edge"),
    )
    graph = SimilarityAdjacencyGraph.create(
        corpus=corpus,
        identity=_graph_identity("similarity"),
        graph_artifact_sha256=_digest("similarity-graph"),
        response_sha256=_digest("similarity-response"),
        accounting=_graph_accounting(),
        edges=(edge,),
    )

    result = evaluate_representation_cell(
        corpus,
        cell=RepresentationCell.SIMILARITY_NEGATIVE,
        observations=(observation,),
        adjacency=graph,
    )

    assert values[20].value_id in {value.value_id for value in result.hydrated_values}
    assert result.trace["promotion"] == {
        "cell_intrinsically_ineligible": True,
        "reason": "registered negative control",
    }
    assert result.trace["ranking"]["expansion_hops"] == 1
    assert result.trace["ranking"]["similarity_edge_threshold"] == 0.8


def test_rneg_rejects_wrong_graph_and_stale_similarity_endpoint() -> None:
    corpus, values, observation = _similarity_fixture()
    entity_graph = EntityAdjacencyGraph.create(
        corpus=corpus,
        identity=_graph_identity("entity"),
        graph_artifact_sha256=_digest("empty-entity-graph"),
        response_sha256=_digest("empty-entity-response"),
        accounting=_graph_accounting(),
        edges=(),
    )
    with pytest.raises(RepresentationError, match="similarity-adjacency"):
        evaluate_representation_cell(
            corpus,
            cell=RepresentationCell.SIMILARITY_NEGATIVE,
            observations=(observation,),
            adjacency=entity_graph,
        )

    stale = SimilarityAdjacencyEdge(
        edge_id=_opaque("edge", "stale"),
        left_value=replace(
            ValueProvenance.from_value(values[0]),
            source_version_sha256="e" * 64,
        ),
        right_value=ValueProvenance.from_value(values[20]),
        similarity_score=0.9,
        edge_artifact_sha256=_digest("stale-similarity-edge"),
    )
    stale_graph = SimilarityAdjacencyGraph.create(
        corpus=corpus,
        identity=_graph_identity("similarity"),
        graph_artifact_sha256=_digest("stale-similarity-graph"),
        response_sha256=_digest("stale-similarity-response"),
        accounting=_graph_accounting(),
        edges=(stale,),
    )
    with pytest.raises(RepresentationError, match="source version"):
        evaluate_representation_cell(
            corpus,
            cell=RepresentationCell.SIMILARITY_NEGATIVE,
            observations=(observation,),
            adjacency=stale_graph,
        )


@pytest.mark.parametrize(
    "forbidden",
    ["question", "question_type", "answer", "gold_session_ids", "judge_label"],
)
def test_extractor_request_rejects_every_forbidden_input_field(forbidden: str) -> None:
    values = _compile_values([_record("q-inputs", ["source-only"])])
    _, receipts = _construct_family(
        values,
        KeyFamily.SUMMARY,
        lambda _value, _index: [("summary", None)],
    )

    with pytest.raises(RepresentationError, match="only the canonical source value"):
        replace(receipts[0], input_fields=("source_value", forbidden))


def test_extractor_request_digest_is_source_only_and_tamper_evident() -> None:
    values = _compile_values([_record("q-request", ["source-only"])])
    _, receipts = _construct_family(
        values,
        KeyFamily.SUMMARY,
        lambda _value, _index: [("summary", None)],
    )
    receipt = receipts[0]
    assert receipt.request_sha256 == extraction_request_sha256(
        family=receipt.family,
        raw_value_sha256=receipt.raw_value_sha256,
        raw_value_utf8_bytes=receipt.raw_value_utf8_bytes,
        extractor=receipt.extractor,
    )
    with pytest.raises(RepresentationError, match="source-only request material"):
        replace(receipt, request_sha256="f" * 64)
    with pytest.raises(RepresentationError, match="source-input digest"):
        replace(receipt, source_input_sha256="f" * 64)


@pytest.mark.parametrize(
    "forbidden",
    ["question", "question_type", "answer", "gold_session_ids", "judge_label"],
)
def test_graph_request_rejects_every_forbidden_input_field(forbidden: str) -> None:
    values = _compile_values([_record("q-graph-input", ["source-only"])])
    corpus = _corpus(values)
    graph = SimilarityAdjacencyGraph.create(
        corpus=corpus,
        identity=_graph_identity("similarity"),
        graph_artifact_sha256=_digest("graph-input-artifact"),
        response_sha256=_digest("graph-input-response"),
        accounting=_graph_accounting(),
        edges=(),
    )
    assert graph.input_fields == GRAPH_INPUT_FIELDS
    assert graph.request_sha256 == graph_request_sha256(
        graph_type="similarity-adjacency",
        navigation_index_sha256=corpus.navigation_index_sha256,
        identity=graph.identity,
    )
    with pytest.raises(RepresentationError, match="source-safe navigation index"):
        replace(graph, input_fields=("representation_index", forbidden))


def test_graph_request_digest_and_source_index_tamper_fail_closed() -> None:
    values = _compile_values([_record("q-graph-request", ["source-only"])])
    corpus = _corpus(values)
    graph = SimilarityAdjacencyGraph.create(
        corpus=corpus,
        identity=_graph_identity("similarity"),
        graph_artifact_sha256=_digest("graph-request-artifact"),
        response_sha256=_digest("graph-request-response"),
        accounting=_graph_accounting(),
        edges=(),
    )
    with pytest.raises(RepresentationError, match="source-safe navigation request material"):
        replace(graph, request_sha256="f" * 64)
    with pytest.raises(RepresentationError, match="source-safe navigation request material"):
        replace(graph, source_input_sha256="f" * 64)


def test_gold_and_question_metadata_changes_do_not_change_source_only_request_or_values() -> None:
    first_values = _compile_values(
        [
            _record(
                "q-gold-independent",
                ["identical source"],
                question="FIRST QUESTION",
                question_type="single-session-user",
                answer="FIRST ANSWER",
            )
        ]
    )
    second_values = _compile_values(
        [
            _record(
                "q-gold-independent",
                ["identical source"],
                question="SECOND QUESTION",
                question_type="knowledge-update",
                answer="SECOND ANSWER",
                answer_sessions=["different-gold-session"],
            )
        ]
    )
    _, first_receipts = _construct_family(
        first_values,
        KeyFamily.SUMMARY,
        lambda _value, _index: [("same summary", None)],
    )
    _, second_receipts = _construct_family(
        second_values,
        KeyFamily.SUMMARY,
        lambda _value, _index: [("same summary", None)],
    )

    assert first_values[0].raw_value == second_values[0].raw_value
    assert first_values[0].raw_value_sha256 == second_values[0].raw_value_sha256
    assert first_receipts[0].request_sha256 == second_receipts[0].request_sha256

    first_corpus = _corpus(
        first_values,
        {KeyFamily.SUMMARY: lambda _value, _index: [("same summary", None)]},
    )
    second_corpus = _corpus(
        second_values,
        {KeyFamily.SUMMARY: lambda _value, _index: [("same summary", None)]},
    )
    identity = _graph_identity("gold-independent")
    first_graph = SimilarityAdjacencyGraph.create(
        corpus=first_corpus,
        identity=identity,
        graph_artifact_sha256=_digest("gold-independent-graph-artifact"),
        response_sha256=_digest("gold-independent-graph-response"),
        accounting=_graph_accounting(),
        edges=(),
    )
    second_graph = SimilarityAdjacencyGraph.create(
        corpus=second_corpus,
        identity=identity,
        graph_artifact_sha256=_digest("gold-independent-graph-artifact"),
        response_sha256=_digest("gold-independent-graph-response"),
        accounting=_graph_accounting(),
        edges=(),
    )

    assert first_corpus.index_sha256 != second_corpus.index_sha256
    assert first_corpus.navigation_index_sha256 == second_corpus.navigation_index_sha256
    assert first_graph.source_input_sha256 == second_graph.source_input_sha256
    assert first_graph.request_sha256 == second_graph.request_sha256


def test_corpus_rejects_duplicate_cross_source_stale_and_output_tamper() -> None:
    values = _compile_values([_record("q-corpus", ["one", "two"])])
    keys, receipts = _construct_family(
        values,
        KeyFamily.SUMMARY,
        lambda _value, index: [(f"summary-{index}", None)],
    )
    with pytest.raises(RepresentationError, match="repeats a key ID"):
        RepresentationCorpus(
            projection_corpus=_PROJECTION_CORPORA[values[0].projection_sha256],
            values=values,
            derived_keys=(keys[0], keys[0], keys[1]),
            construction_receipts=receipts,
        )
    with pytest.raises(RepresentationError, match="source version"):
        RepresentationCorpus(
            projection_corpus=_PROJECTION_CORPORA[values[0].projection_sha256],
            values=values,
            derived_keys=(replace(keys[0], source_version_sha256="f" * 64), keys[1]),
            construction_receipts=receipts,
        )
    with pytest.raises(RepresentationError, match="output digest"):
        RepresentationCorpus(
            projection_corpus=_PROJECTION_CORPORA[values[0].projection_sha256],
            values=values,
            derived_keys=keys,
            construction_receipts=(
                replace(receipts[0], output_keys_sha256="f" * 64),
                receipts[1],
            ),
        )

    other = _compile_values([_record("q-other", ["other"])])
    with pytest.raises(RepresentationError, match="cross question boundaries"):
        RepresentationCorpus(
            projection_corpus=_PROJECTION_CORPORA[values[0].projection_sha256],
            values=(values[0], other[0]),
            derived_keys=(),
            construction_receipts=(),
        )


@pytest.mark.parametrize("selection", [slice(None, -1), slice(0, 2), slice(None, None, -1)])
def test_corpus_rejects_missing_tail_middle_and_reordered_f0_values(selection: slice) -> None:
    values = _compile_values([_record("q-complete", ["zero", "one", "two"])])
    selected = values[selection]
    if selection == slice(0, 2):
        selected = (values[0], values[2])

    with pytest.raises(RepresentationError, match="partial|order/content|exact authoritative"):
        RepresentationCorpus(
            projection_corpus=_PROJECTION_CORPORA[values[0].projection_sha256],
            values=selected,
            derived_keys=(),
            construction_receipts=(),
        )


def test_self_consistent_forged_subset_still_fails_authoritative_f0_slice() -> None:
    values = _compile_values([_record("q-forged-subset", ["zero", "one", "two"])])
    subset = values[:2]
    forged_digest = hashlib.sha256(
        json.dumps(
            question_value_binding_payload(tuple(value.turn for value in subset)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    forged = tuple(
        replace(
            value,
            question_value_count=len(subset),
            question_values_sha256=forged_digest,
        )
        for value in subset
    )

    with pytest.raises(RepresentationError, match="exact authoritative"):
        RepresentationCorpus(
            projection_corpus=_PROJECTION_CORPORA[values[0].projection_sha256],
            values=forged,
            derived_keys=(),
            construction_receipts=(),
        )


def test_missing_construction_and_partial_or_tampered_rankings_fail_closed() -> None:
    values = _compile_values([_record("q-partial", ["one", "two"])])
    corpus = _corpus(values)
    raw = _observe(corpus, KeyFamily.RAW)
    with pytest.raises(RepresentationError, match="one complete merged-sfk"):
        evaluate_representation_cell(
            corpus,
            cell=RepresentationCell.RAW_MERGED_SFK,
            observations=(raw, _observe(corpus, KeyFamily.MERGED_SFK)),
        )

    with pytest.raises(RepresentationError, match="partial ranking"):
        replace(raw, complete=False)
    with pytest.raises(RepresentationError, match="digest does not match"):
        replace(raw, observation_artifact_sha256=_digest("tampered-observation"))

    unknown_ranked = tuple(
        [
            replace(raw.ranked_keys[0], key_id=_opaque("key", "unknown-key")),
            *raw.ranked_keys[1:],
        ]
    )
    unknown = RankedFamilyObservation.create(
        family=KeyFamily.RAW,
        corpus=corpus,
        query_sha256=raw.query_sha256,
        scorer=raw.scorer,
        observation_artifact_sha256=_digest("unknown-observation"),
        ranked_keys=unknown_ranked,
    )
    with pytest.raises(RepresentationError, match="unknown key"):
        evaluate_representation_cell(
            corpus,
            cell=RepresentationCell.RAW,
            observations=(unknown,),
        )


def test_observation_from_another_source_is_rejected() -> None:
    first_values = _compile_values([_record("q-source-a", ["one"])])
    second_values = _compile_values([_record("q-source-b", ["two"])])
    first = _corpus(first_values)
    second = _corpus(second_values)

    with pytest.raises(RepresentationError, match="question boundary"):
        evaluate_representation_cell(
            first,
            cell=RepresentationCell.RAW,
            observations=(_observe(second, KeyFamily.RAW),),
        )


def test_graph_from_another_source_is_rejected_before_expansion() -> None:
    first_values = _compile_values([_record("q-graph-source", ["first source"])])
    second_values = _compile_values([_record("q-graph-source", ["second source"])])
    first = _corpus(first_values)
    second = _corpus(second_values)
    graph = SimilarityAdjacencyGraph.create(
        corpus=second,
        identity=_graph_identity("cross-source"),
        graph_artifact_sha256=_digest("cross-source-graph"),
        response_sha256=_digest("cross-source-response"),
        accounting=_graph_accounting(),
        edges=(),
    )

    with pytest.raises(RepresentationError, match="source boundary"):
        evaluate_representation_cell(
            first,
            cell=RepresentationCell.SIMILARITY_NEGATIVE,
            observations=(_observe(first, KeyFamily.RAW),),
            adjacency=graph,
        )


def test_cell_family_composition_and_merged_cardinality_fail_closed() -> None:
    values = _compile_values([_record("q-shape", ["one"])])
    keys, receipts = _construct_family(
        values,
        KeyFamily.MERGED_SFK,
        lambda _value, _index: [("merged-a", None), ("merged-b", None)],
    )
    corpus = RepresentationCorpus(
        projection_corpus=_PROJECTION_CORPORA[values[0].projection_sha256],
        values=values,
        derived_keys=keys,
        construction_receipts=receipts,
    )
    with pytest.raises(RepresentationError, match="exactly one"):
        evaluate_representation_cell(
            corpus,
            cell=RepresentationCell.RAW_MERGED_SFK,
            observations=(
                _observe(corpus, KeyFamily.RAW),
                _observe(corpus, KeyFamily.MERGED_SFK),
            ),
        )
    with pytest.raises(RepresentationError, match="exact key-family order"):
        evaluate_representation_cell(
            corpus,
            cell=RepresentationCell.RAW,
            observations=(
                _observe(corpus, KeyFamily.RAW),
                _observe(corpus, KeyFamily.MERGED_SFK),
            ),
        )


def test_graph_shape_rejects_reverse_duplicates_self_edges_and_low_similarity() -> None:
    values = _compile_values([_record("q-graph-shape", ["one", "two"])])
    left = ValueProvenance.from_value(values[0])
    right = ValueProvenance.from_value(values[1])
    with pytest.raises(RepresentationError, match="canonically ordered"):
        SimilarityAdjacencyEdge(
            edge_id=_opaque("edge", "reverse"),
            left_value=right,
            right_value=left,
            similarity_score=0.9,
            edge_artifact_sha256=_digest("reverse"),
        )
    with pytest.raises(RepresentationError, match="canonically ordered"):
        SimilarityAdjacencyEdge(
            edge_id=_opaque("edge", "self"),
            left_value=left,
            right_value=left,
            similarity_score=0.9,
            edge_artifact_sha256=_digest("self"),
        )
    with pytest.raises(RepresentationError, match="threshold"):
        SimilarityAdjacencyEdge(
            edge_id=_opaque("edge", "low"),
            left_value=left,
            right_value=right,
            similarity_score=0.79,
            edge_artifact_sha256=_digest("low"),
        )


def test_navigation_ids_are_opaque_and_secret_canaries_never_enter_artifacts() -> None:
    secret_key_text = "SECRET-KEY-TEXT-CANARY"
    secret_entity_name = "SECRET-ENTITY-NAME-CANARY"
    values = _compile_values([_record("q-opaque", ["source-value"])])
    corpus = _corpus(
        values,
        {
            KeyFamily.ENTITY_DESCRIPTION: lambda _value, _index: [
                (secret_key_text, secret_entity_name)
            ]
        },
    )
    result = evaluate_representation_cell(
        corpus,
        cell=RepresentationCell.ENTITY_DIRECT,
        observations=(_observe(corpus, KeyFamily.ENTITY_DESCRIPTION),),
    )
    serialized = json.dumps(result.content_free_artifact(), ensure_ascii=False)
    assert secret_key_text not in serialized
    assert secret_entity_name not in serialized
    assert all(
        key.entity_id is None or key.entity_id.startswith("entity:")
        for key in corpus.keys_for_family(KeyFamily.ENTITY_DESCRIPTION)
    )

    valid = corpus.derived_keys[0]
    with pytest.raises(RepresentationError, match="opaque key"):
        replace(valid, key_id="key:SECRET-KEY-ID-CANARY")
    with pytest.raises(RepresentationError, match="opaque entity"):
        replace(valid, entity_id="entity:SECRET-ENTITY-ID-CANARY")
    with pytest.raises(RepresentationError, match="opaque receipt"):
        replace(valid, construction_receipt_id="receipt:SECRET-RECEIPT-CANARY")


def test_protocol_is_explicitly_hypothetical_and_offline() -> None:
    values = _compile_values([_record("q-claims", ["one"])])
    corpus = _corpus(values)
    result = evaluate_representation_cell(
        corpus,
        cell=RepresentationCell.RAW,
        observations=(_observe(corpus, KeyFamily.RAW),),
    )
    assert result.trace["protocol_version"] == PROTOCOL_VERSION
    assert result.trace["paper_reproduction"] is False
    assert result.trace["sb_hypothesis"] == "SB-HMR-v1"
    assert result.trace["production_configuration"] is False
    assert result.trace["claims"]["executes_extractor_scorer_model_or_network"] is False
    assert result.trace["claims"]["quality_improvement"] is False
