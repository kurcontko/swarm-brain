from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
from benchmarks.integrations.longmemeval_turn_retrieval import (
    DENSE_LANE_DEPTH,
    DENSE_WEIGHT,
    E1A_PROTOCOL,
    E1A_PROTOCOL_VERSION,
    FUSED_CANDIDATE_CAP,
    LEXICAL_LANE_DEPTH,
    LEXICAL_WEIGHT,
    PRODUCTION_DENSE_SERVING_CAP,
    RRF_K,
    ExternalLaneIdentity,
    ImmutableArtifactIdentity,
    RankedLaneObservation,
    RankedTurnObservation,
    RetrievalLane,
    TurnRetrievalError,
    TurnRetrievalProtocol,
    evaluate_gold_session_recall,
    fuse_question_turns,
)
from benchmarks.integrations.longmemeval_turns import (
    LongMemEvalTurnId,
    compile_dataset_bytes,
)

QUERY_SENTINEL = "QUERY-TEXT-MUST-NOT-ENTER-TRACE Café"
TURN_SENTINEL = "TURN-TEXT-MUST-NOT-ENTER-TRACE"
ANSWER_SENTINEL = "ANSWER-TEXT-MUST-NOT-ENTER-TRACE"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _corpus(*, turns: int = 4):
    record = {
        "question_id": "q-e1a",
        "question_type": "multi-session",
        "question": QUERY_SENTINEL,
        "answer": ANSWER_SENTINEL,
        "question_date": "2025/02/01 (Sat) 12:00",
        "haystack_session_ids": [f"session-{index:03d}" for index in range(turns)],
        "haystack_dates": ["2025/01/31 (Fri) 09:00"] * turns,
        "haystack_sessions": [
            [{"role": "user", "content": f"{TURN_SENTINEL}-{index:03d}"}] for index in range(turns)
        ],
        "answer_session_ids": [f"session-{turns - 1:03d}"],
    }
    raw = (json.dumps([record], separators=(",", ":")) + "\n").encode()
    return compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="synthetic-e1a.json",
    )


def _identity(lane: RetrievalLane) -> ExternalLaneIdentity:
    return ExternalLaneIdentity(
        producer=f"offline-{lane.value}-producer",
        scorer=ImmutableArtifactIdentity(
            name=f"{lane.value}-scorer",
            revision="revision-1",
            artifact_sha256=_digest(f"{lane.value}-scorer"),
        ),
        projection=ImmutableArtifactIdentity(
            name=f"{lane.value}-projection",
            revision="projection-1",
            artifact_sha256=_digest(f"{lane.value}-projection"),
        ),
        observation_artifact_sha256=_digest(f"{lane.value}-observations"),
    )


def _lane(
    corpus,
    lane: RetrievalLane,
    positions: list[int],
    *,
    scores: list[float] | None = None,
    query_text: str = QUERY_SENTINEL,
    projection_sha256: str | None = None,
    requested_depth: int | None = None,
) -> RankedLaneObservation:
    values = scores or [1.0 / (index + 1) for index in range(len(positions))]
    return RankedLaneObservation(
        lane=lane,
        requested_depth=(
            requested_depth if requested_depth is not None else E1A_PROTOCOL.lane_depth(lane)
        ),
        query_sha256=hashlib.sha256(query_text.encode()).hexdigest(),
        turn_corpus_projection_sha256=(projection_sha256 or corpus.projection_sha256),
        identity=_identity(lane),
        candidates=tuple(
            RankedTurnObservation(
                turn_id=LongMemEvalTurnId("q-e1a", position, 0),
                raw_score=score,
            )
            for position, score in zip(positions, values, strict=True)
        ),
        examined_count=len(corpus.turns),
    )


def test_e1a_protocol_is_fixed_and_explicitly_not_production_parity() -> None:
    assert E1A_PROTOCOL.protocol_version == E1A_PROTOCOL_VERSION
    assert E1A_PROTOCOL.lexical_depth == LEXICAL_LANE_DEPTH == 128
    assert E1A_PROTOCOL.dense_depth == DENSE_LANE_DEPTH == 128
    assert E1A_PROTOCOL.lexical_weight == LEXICAL_WEIGHT == 3.0
    assert E1A_PROTOCOL.dense_weight == DENSE_WEIGHT == 4.0
    assert E1A_PROTOCOL.rrf_k == RRF_K == 60
    assert E1A_PROTOCOL.fused_candidate_cap == FUSED_CANDIDATE_CAP == 128
    assert PRODUCTION_DENSE_SERVING_CAP == 100

    metadata = E1A_PROTOCOL.as_dict()
    assert metadata["classification"] == "evaluation-only-turn-transfer"
    assert metadata["production_configuration"] is False
    assert metadata["transfer_notes"] == {
        "weights_mirror": "interactive-general-production-policy",
        "weights_match_interactive_general_policy": True,
        "lane_depths_are_evaluation_choices": True,
        "production_dense_serving_cap": 100,
        "dense_depth_exceeds_current_production_cap": True,
    }
    assert metadata["evidence_boundary"]["dense_vectors"] == ("externally-attested-unverified")
    assert metadata["evidence_boundary"]["dense_scores"] == ("externally-attested-unverified")
    assert metadata["evidence_boundary"]["qa_improvement_proven"] is False


def test_reserved_protocol_rejects_drift_but_new_version_can_register() -> None:
    with pytest.raises(TurnRetrievalError, match="reserved E1-A"):
        replace(E1A_PROTOCOL, dense_depth=127)

    held_out = TurnRetrievalProtocol(
        protocol_version="swarmbrain-longmemeval-turn-transfer-heldout-v2",
        protocol_name="held-out transfer protocol",
        cell="E1-A2",
        lexical_depth=64,
        dense_depth=64,
        lexical_weight=1.0,
        dense_weight=1.0,
        rrf_k=50,
        fused_candidate_cap=64,
    )
    assert held_out.dense_depth == 64


def test_weighted_rrf_is_exact_unique_and_content_free() -> None:
    corpus = _corpus()
    lexical = _lane(corpus, RetrievalLane.LEXICAL, [0, 1, 2], scores=[7.0, 4.0, -2.0])
    dense = _lane(corpus, RetrievalLane.DENSE, [1, 0, 3], scores=[0.91, 0.8, 0.1])

    result = fuse_question_turns(
        corpus,
        question_id="q-e1a",
        query_text=QUERY_SENTINEL,
        lexical=lexical,
        dense=dense,
    )

    assert [item.turn_id.as_tuple() for item in result.candidates] == [
        ("q-e1a", 1, 0),
        ("q-e1a", 0, 0),
        ("q-e1a", 3, 0),
        ("q-e1a", 2, 0),
    ]
    assert len({item.turn_id for item in result.candidates}) == 4
    assert result.candidates[0].raw_rrf == pytest.approx(3.0 / 62 + 4.0 / 61)
    assert result.candidates[1].raw_rrf == pytest.approx(3.0 / 61 + 4.0 / 62)
    assert result.candidates[2].raw_rrf == pytest.approx(4.0 / 63)
    assert result.candidates[3].raw_rrf == pytest.approx(3.0 / 63)

    trace = result.content_free_trace()
    encoded = json.dumps(trace, ensure_ascii=False, sort_keys=True)
    assert QUERY_SENTINEL not in encoded
    assert TURN_SENTINEL not in encoded
    assert ANSWER_SENTINEL not in encoded
    assert trace["query"]["sha256"] == hashlib.sha256(QUERY_SENTINEL.encode()).hexdigest()
    assert trace["query"]["query_text_stored"] is False
    assert len(trace["lanes"]) == 2
    assert all(len(lane["lane_trace_sha256"]) == 64 for lane in trace["lanes"])
    assert len(trace["fusion"]["pre_cap_order_sha256"]) == 64
    assert len(trace["fusion"]["post_cap_candidate_payloads_sha256"]) == 64
    assert trace["accounting"]["external_raw_scores"] == 6
    assert trace["accounting"]["cross_lane_overlap"] == 2
    assert trace["accounting"]["local_model_calls"] == 0
    assert trace["accounting"]["local_database_calls"] == 0
    assert trace["accounting"]["local_network_calls"] == 0
    assert trace["claims"]["dense_vectors_recomputed"] is False
    assert trace["claims"]["qa_improvement_proven"] is False
    assert trace["fusion"]["post_cap_candidates"][0]["candidate_payload_utf8"] == (
        result.candidates[0].turn.serialized_document_utf8.as_dict()
    )


def test_trace_export_is_stable_and_caller_mutation_cannot_change_result() -> None:
    corpus = _corpus()
    result = fuse_question_turns(
        corpus,
        question_id="q-e1a",
        query_text=QUERY_SENTINEL,
        lexical=_lane(corpus, RetrievalLane.LEXICAL, [0, 1]),
        dense=_lane(corpus, RetrievalLane.DENSE, [1, 0]),
    )
    digest = result.trace_sha256
    first = result.content_free_trace()
    first["fusion"]["post_cap_candidates"].clear()
    second = result.content_free_trace()

    assert result.trace_sha256 == digest
    assert len(second["fusion"]["post_cap_candidates"]) == 2
    assert second["trace_sha256"] == digest


def test_equal_rrf_scores_use_canonical_turn_id_tie_break() -> None:
    corpus = _corpus(turns=2)
    equal_weight_protocol = TurnRetrievalProtocol(
        protocol_version="swarmbrain-longmemeval-turn-transfer-tie-test-v1",
        protocol_name="tie behavior test",
        cell="test-only",
        lexical_depth=2,
        dense_depth=2,
        lexical_weight=1.0,
        dense_weight=1.0,
        rrf_k=60,
        fused_candidate_cap=2,
    )
    result = fuse_question_turns(
        corpus,
        question_id="q-e1a",
        query_text=QUERY_SENTINEL,
        lexical=_lane(
            corpus,
            RetrievalLane.LEXICAL,
            [1, 0],
            requested_depth=2,
        ),
        dense=_lane(
            corpus,
            RetrievalLane.DENSE,
            [0, 1],
            requested_depth=2,
        ),
        protocol=equal_weight_protocol,
    )

    assert result.candidates[0].raw_rrf == pytest.approx(result.candidates[1].raw_rrf)
    assert [item.turn_id.session_position for item in result.candidates] == [0, 1]


def test_cap_is_hard_and_gold_recall_is_strictly_post_hoc() -> None:
    corpus = _corpus(turns=256)
    # The gold session occurs only at the weakest lexical rank. Dense's larger
    # weight pushes it below the independently applied unique top-128 cap.
    lexical = _lane(corpus, RetrievalLane.LEXICAL, list(range(128, 256)))
    dense = _lane(corpus, RetrievalLane.DENSE, list(range(128)))
    result = fuse_question_turns(
        corpus,
        question_id="q-e1a",
        query_text=QUERY_SENTINEL,
        lexical=lexical,
        dense=dense,
    )
    trace_before = result.content_free_trace()
    evaluation = evaluate_gold_session_recall(
        result,
        gold_session_ids=("session-255",),
    )

    assert len(result.pre_cap_candidates) == 256
    assert len(result.candidates) == 128
    assert len({item.turn_id for item in result.candidates}) == 128
    assert trace_before["fusion"]["dropped_by_cap"] == 128
    assert evaluation.pre_cap.gold_session_recall == 1.0
    assert evaluation.pre_cap.any_gold_session_recalled is True
    assert evaluation.post_cap.gold_session_recall == 0.0
    assert evaluation.post_cap.any_gold_session_recalled is False
    assert evaluation.as_dict()["candidate_generation"] == {
        "gold_fields_used": False,
        "evaluation_is_post_hoc": True,
    }
    assert evaluation.as_dict()["qa_improvement_proven"] is False
    assert result.content_free_trace() == trace_before
    assert "gold_session_recall" not in trace_before["fusion"]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), True])
def test_non_finite_or_boolean_raw_scores_fail_closed(score: float) -> None:
    with pytest.raises(TurnRetrievalError, match="finite number"):
        RankedTurnObservation(
            turn_id=LongMemEvalTurnId("q-e1a", 0, 0),
            raw_score=score,
        )


def test_duplicate_unknown_cross_question_and_lane_depth_drift_fail_closed() -> None:
    corpus = _corpus()
    duplicate = RankedTurnObservation(LongMemEvalTurnId("q-e1a", 0, 0), 1.0)
    with pytest.raises(TurnRetrievalError, match="repeats"):
        RankedLaneObservation(
            lane=RetrievalLane.LEXICAL,
            requested_depth=128,
            query_sha256=_digest(QUERY_SENTINEL),
            turn_corpus_projection_sha256=corpus.projection_sha256,
            identity=_identity(RetrievalLane.LEXICAL),
            candidates=(duplicate, duplicate),
            examined_count=2,
        )

    valid_dense = _lane(corpus, RetrievalLane.DENSE, [0])
    unknown_lexical = RankedLaneObservation(
        lane=RetrievalLane.LEXICAL,
        requested_depth=128,
        query_sha256=_digest(QUERY_SENTINEL),
        turn_corpus_projection_sha256=corpus.projection_sha256,
        identity=_identity(RetrievalLane.LEXICAL),
        candidates=(RankedTurnObservation(LongMemEvalTurnId("different-question", 0, 0), 1.0),),
        examined_count=1,
    )
    with pytest.raises(TurnRetrievalError, match="unknown or cross-question"):
        fuse_question_turns(
            corpus,
            question_id="q-e1a",
            query_text=QUERY_SENTINEL,
            lexical=unknown_lexical,
            dense=valid_dense,
        )

    unknown_position = replace(
        unknown_lexical,
        candidates=(RankedTurnObservation(LongMemEvalTurnId("q-e1a", 999, 0), 1.0),),
    )
    with pytest.raises(TurnRetrievalError, match="unknown or cross-question"):
        fuse_question_turns(
            corpus,
            question_id="q-e1a",
            query_text=QUERY_SENTINEL,
            lexical=unknown_position,
            dense=valid_dense,
        )

    with pytest.raises(TurnRetrievalError, match="protocol hard depth"):
        fuse_question_turns(
            corpus,
            question_id="q-e1a",
            query_text=QUERY_SENTINEL,
            lexical=_lane(
                corpus,
                RetrievalLane.LEXICAL,
                [0],
                requested_depth=127,
            ),
            dense=valid_dense,
        )

    with pytest.raises(TurnRetrievalError, match="exact LongMemEvalTurnId"):
        RankedTurnObservation(turn_id='["q-e1a",0,0]', raw_score=1.0)  # type: ignore[arg-type]


def test_query_projection_identity_and_lane_bounds_fail_closed() -> None:
    corpus = _corpus()
    dense = _lane(corpus, RetrievalLane.DENSE, [0])
    with pytest.raises(TurnRetrievalError, match="different query"):
        fuse_question_turns(
            corpus,
            question_id="q-e1a",
            query_text=QUERY_SENTINEL,
            lexical=_lane(
                corpus,
                RetrievalLane.LEXICAL,
                [0],
                query_text="different query",
            ),
            dense=dense,
        )
    with pytest.raises(TurnRetrievalError, match="different turn projection"):
        fuse_question_turns(
            corpus,
            question_id="q-e1a",
            query_text=QUERY_SENTINEL,
            lexical=_lane(
                corpus,
                RetrievalLane.LEXICAL,
                [0],
                projection_sha256="0" * 64,
            ),
            dense=dense,
        )
    with pytest.raises(TurnRetrievalError, match="hard depth"):
        RankedLaneObservation(
            lane=RetrievalLane.LEXICAL,
            requested_depth=1,
            query_sha256=_digest(QUERY_SENTINEL),
            turn_corpus_projection_sha256=corpus.projection_sha256,
            identity=_identity(RetrievalLane.LEXICAL),
            candidates=(
                RankedTurnObservation(LongMemEvalTurnId("q-e1a", 0, 0), 1.0),
                RankedTurnObservation(LongMemEvalTurnId("q-e1a", 1, 0), 0.5),
            ),
            examined_count=2,
        )
    overexamined = replace(
        _lane(corpus, RetrievalLane.LEXICAL, [0]),
        examined_count=len(corpus.turns) + 1,
    )
    with pytest.raises(TurnRetrievalError, match="exceeds the question turn corpus"):
        fuse_question_turns(
            corpus,
            question_id="q-e1a",
            query_text=QUERY_SENTINEL,
            lexical=overexamined,
            dense=dense,
        )

    large_corpus = _corpus(turns=129)
    with pytest.raises(TurnRetrievalError, match="hard depth"):
        _lane(
            large_corpus,
            RetrievalLane.LEXICAL,
            list(range(129)),
        )


def test_gold_evaluator_rejects_unknown_or_duplicated_session_labels() -> None:
    corpus = _corpus()
    result = fuse_question_turns(
        corpus,
        question_id="q-e1a",
        query_text=QUERY_SENTINEL,
        lexical=_lane(corpus, RetrievalLane.LEXICAL, [0]),
        dense=_lane(corpus, RetrievalLane.DENSE, [1]),
    )
    with pytest.raises(TurnRetrievalError, match="duplicates"):
        evaluate_gold_session_recall(
            result,
            gold_session_ids=("session-000", "session-000"),
        )
    with pytest.raises(TurnRetrievalError, match="outside"):
        evaluate_gold_session_recall(result, gold_session_ids=("unknown-session",))


def test_identity_requires_immutable_component_and_observation_digests() -> None:
    with pytest.raises(TurnRetrievalError, match="SHA-256"):
        ImmutableArtifactIdentity(
            name="dense-model",
            revision="mutable-main",
            artifact_sha256="unverified",
        )
    with pytest.raises(TurnRetrievalError, match="SHA-256"):
        ExternalLaneIdentity(
            producer="external",
            scorer=ImmutableArtifactIdentity("scorer", "rev", "1" * 64),
            projection=ImmutableArtifactIdentity("projection", "rev", "2" * 64),
            observation_artifact_sha256="3" * 63,
        )
