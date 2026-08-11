from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import replace

import pytest
from benchmarks.integrations.longmemeval_e1 import (
    E1_POOL_CAP,
    E1_PROTOCOL,
    E1_PROTOCOL_VERSION,
    E1C_COLBERT_WEIGHT,
    E1C_CROSS_ENCODER_WEIGHT,
    E1C_RRF_K,
    E1D_CE_RELATIVE_THRESHOLD,
    E1D_FUSED_HEAD,
    E1Cell,
    E1SelectionError,
    ExternalScorerIdentity,
    PoolScoreObservation,
    ScoreChannel,
    TurnScoreObservation,
    bind_e1a_pool,
    select_e1b,
    select_e1c,
    select_e1d,
    stable_sigmoid,
    validate_e1b_result,
)
from benchmarks.integrations.longmemeval_turn_retrieval import (
    E1A_PROTOCOL,
    ExternalLaneIdentity,
    ImmutableArtifactIdentity,
    RankedLaneObservation,
    RankedTurnObservation,
    RetrievalLane,
    fuse_question_turns,
)
from benchmarks.integrations.longmemeval_turns import (
    LongMemEvalTurnId,
    compile_dataset_bytes,
)

QUERY_SENTINEL = "QUERY-TEXT-MUST-NOT-ENTER-E1-TRACE Café"
TURN_SENTINEL = "TURN-TEXT-MUST-NOT-ENTER-E1-TRACE"
ANSWER_SENTINEL = "ANSWER-TEXT-MUST-NOT-ENTER-E1-TRACE"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _corpus(*, turns: int = 4):
    record = {
        "question_id": "q-e1",
        "question_type": "multi-session",
        "question": QUERY_SENTINEL,
        "answer": ANSWER_SENTINEL,
        "question_date": "2025/02/01 (Sat) 12:00",
        "haystack_session_ids": [f"session-{position:03d}" for position in range(turns)],
        "haystack_dates": ["2025/01/31 (Fri) 09:00"] * turns,
        "haystack_sessions": [
            [{"role": "user", "content": f"{TURN_SENTINEL}-{position:03d}"}]
            for position in range(turns)
        ],
        "answer_session_ids": [f"session-{max(turns - 1, 0):03d}"],
    }
    raw = (json.dumps([record], separators=(",", ":")) + "\n").encode()
    return compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="synthetic-e1.json",
    )


def _lane_identity(lane: RetrievalLane) -> ExternalLaneIdentity:
    return ExternalLaneIdentity(
        producer=f"offline-{lane.value}",
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


def _source(
    *,
    turns: int = 4,
    lexical_positions: list[int] | None = None,
    dense_positions: list[int] | None = None,
):
    corpus = _corpus(turns=turns)
    lexical_positions = lexical_positions if lexical_positions is not None else list(range(turns))
    dense_positions = (
        dense_positions if dense_positions is not None else list(reversed(range(turns)))
    )
    query_sha256 = hashlib.sha256(QUERY_SENTINEL.encode()).hexdigest()

    def lane(kind: RetrievalLane, positions: list[int]) -> RankedLaneObservation:
        return RankedLaneObservation(
            lane=kind,
            requested_depth=E1A_PROTOCOL.lane_depth(kind),
            query_sha256=query_sha256,
            turn_corpus_projection_sha256=corpus.projection_sha256,
            identity=_lane_identity(kind),
            candidates=tuple(
                RankedTurnObservation(
                    turn_id=LongMemEvalTurnId("q-e1", position, 0),
                    raw_score=float(len(positions) - rank),
                )
                for rank, position in enumerate(positions)
            ),
            examined_count=turns,
        )

    source = fuse_question_turns(
        corpus,
        question_id="q-e1",
        query_text=QUERY_SENTINEL,
        lexical=lane(RetrievalLane.LEXICAL, lexical_positions),
        dense=lane(RetrievalLane.DENSE, dense_positions),
    )
    return corpus, source


def _scorer_identity(channel: ScoreChannel) -> ExternalScorerIdentity:
    return ExternalScorerIdentity(
        producer=f"offline-{channel.value}-runner",
        scorer=f"{channel.value}-scorer",
        model=f"pinned-{channel.value}-model",
        revision="immutable-revision-1",
        artifact_sha256=_digest(f"{channel.value}-model-artifact"),
        observation_artifact_sha256=_digest(f"{channel.value}-observation-artifact"),
    )


def _observation(
    source,
    channel: ScoreChannel,
    *,
    values_by_id: dict[LongMemEvalTurnId, float] | None = None,
    order: list[LongMemEvalTurnId] | None = None,
    **binding_overrides,
) -> PoolScoreObservation:
    binding = bind_e1a_pool(source).as_dict()
    binding.update(binding_overrides)
    order = order if order is not None else [candidate.turn_id for candidate in source.candidates]
    if values_by_id is None:
        values_by_id = {
            candidate.turn_id: float(len(source.candidates) - position)
            for position, candidate in enumerate(source.candidates)
        }
    return PoolScoreObservation(
        channel=channel,
        question_id=str(binding["question_id"]),
        query_sha256=str(binding["query_sha256"]),
        turn_corpus_projection_sha256=str(binding["turn_corpus_projection_sha256"]),
        e1a_trace_sha256=str(binding["e1a_trace_sha256"]),
        e1a_pool_sha256=str(binding["e1a_pool_sha256"]),
        pool_count=int(binding["pool_count"]),
        identity=_scorer_identity(channel),
        scores=tuple(
            TurnScoreObservation(turn_id=turn_id, raw_score=values_by_id[turn_id])
            for turn_id in order
        ),
    )


def test_reserved_v1_protocol_is_exact_and_rejects_drift() -> None:
    assert E1_PROTOCOL.protocol_version == E1_PROTOCOL_VERSION
    assert E1_PROTOCOL.source_pool_cap == E1_POOL_CAP == 128
    assert E1_PROTOCOL.rrf_k == E1C_RRF_K == 60
    assert E1_PROTOCOL.cross_encoder_weight == E1C_CROSS_ENCODER_WEIGHT == 0.7
    assert E1_PROTOCOL.colbert_weight == E1C_COLBERT_WEIGHT == 0.3
    assert E1_PROTOCOL.adaptive_fused_head == E1D_FUSED_HEAD == 60
    assert E1_PROTOCOL.adaptive_ce_relative_threshold == E1D_CE_RELATIVE_THRESHOLD == 0.03

    for change in (
        {"source_pool_cap": 127},
        {"rrf_k": 61},
        {"cross_encoder_weight": 0.6},
        {"colbert_weight": 0.4},
        {"adaptive_fused_head": 59},
        {"adaptive_ce_relative_threshold": 0.04},
        {"source_e1a_protocol_version": "different-E1-A"},
    ):
        with pytest.raises(E1SelectionError, match="reserved SmartSearch-shaped E1 v1"):
            replace(E1_PROTOCOL, **change)

    evidence = E1_PROTOCOL.as_dict()["evidence_boundary"]
    assert evidence["external_scorer_and_model_identity"] == "caller-attested-unverified"
    assert evidence["model_artifacts_reopened_or_hashed_by_selection"] is False
    assert evidence["gold_fields_used"] is False
    assert evidence["prompt_packing_executed"] is False
    assert evidence["reader_or_judge_executed"] is False
    assert evidence["qa_improvement_proven"] is False


@pytest.mark.parametrize(
    ("logit", "expected"),
    [(1e308, 1.0), (-1e308, 0.0), (0.0, 0.5), (-0.0, 0.5)],
)
def test_stable_sigmoid_handles_finite_extremes(logit: float, expected: float) -> None:
    assert stable_sigmoid(logit) == expected


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), True, 10**1000],
)
def test_score_observations_reject_non_finite_or_boolean_values(value: float | int) -> None:
    with pytest.raises(E1SelectionError, match="finite number"):
        TurnScoreObservation(turn_id=LongMemEvalTurnId("q-e1", 0, 0), raw_score=value)


def test_e1b_ranks_every_candidate_by_sigmoid_and_accepts_permuted_coverage() -> None:
    _, source = _source()
    ids = [candidate.turn_id for candidate in source.candidates]
    logits = {ids[0]: -2.0, ids[1]: 30.0, ids[2]: 0.0, ids[3]: -30.0}
    observation = _observation(
        source,
        ScoreChannel.CROSS_ENCODER_LOGIT,
        values_by_id=logits,
        order=list(reversed(ids)),
    )

    result = select_e1b(source, cross_encoder=observation)

    assert result.cell is E1Cell.CROSS_ENCODER
    assert len(result.candidates) == len(source.candidates)
    assert {candidate.turn_id for candidate in result.candidates} == set(ids)
    assert [candidate.cross_encoder_sigmoid for candidate in result.candidates] == sorted(
        (stable_sigmoid(value) for value in logits.values()), reverse=True
    )
    assert all(candidate.colbert_score is None for candidate in result.candidates)

    ordinary = _observation(
        source,
        ScoreChannel.CROSS_ENCODER_LOGIT,
        values_by_id=logits,
        order=ids,
    ).content_free_binding()
    permuted = observation.content_free_binding()
    assert ordinary["observed_order_sha256"] != permuted["observed_order_sha256"]
    assert ordinary["canonical_scores_sha256"] == permuted["canonical_scores_sha256"]


def test_cross_encoder_score_ties_use_prior_e1a_rank_before_canonical_id() -> None:
    _, source = _source(turns=2)
    # Frozen E1-A ranks session 1 before canonical-smaller session 0 because
    # the dense lane has the larger source weight.
    assert [candidate.turn_id.session_position for candidate in source.candidates] == [1, 0]
    tied = {candidate.turn_id: 7.0 for candidate in source.candidates}

    result = select_e1b(
        source,
        cross_encoder=_observation(
            source,
            ScoreChannel.CROSS_ENCODER_LOGIT,
            values_by_id=tied,
        ),
    )

    assert [candidate.turn_id.session_position for candidate in result.candidates] == [1, 0]

    saturated = {
        source.candidates[0].turn_id: 1e307,
        source.candidates[1].turn_id: 1e308,
    }
    saturated_result = select_e1b(
        source,
        cross_encoder=_observation(
            source,
            ScoreChannel.CROSS_ENCODER_LOGIT,
            values_by_id=saturated,
        ),
    )
    assert [candidate.cross_encoder_sigmoid for candidate in saturated_result.candidates] == [
        1.0,
        1.0,
    ]
    assert [candidate.turn_id.session_position for candidate in saturated_result.candidates] == [
        1,
        0,
    ]


def test_e1c_uses_exact_weighted_rrf_formula_and_lane_ranks() -> None:
    _, source = _source(turns=4)
    ids = [candidate.turn_id for candidate in source.candidates]
    ce_order = [ids[2], ids[0], ids[3], ids[1]]
    colbert_order = [ids[1], ids[3], ids[0], ids[2]]
    ce = {turn_id: float(10 - rank) for rank, turn_id in enumerate(ce_order, start=1)}
    cb = {turn_id: float(20 - rank) for rank, turn_id in enumerate(colbert_order, start=1)}

    result = select_e1c(
        source,
        cross_encoder=_observation(
            source,
            ScoreChannel.CROSS_ENCODER_LOGIT,
            values_by_id=ce,
        ),
        colbert=_observation(
            source,
            ScoreChannel.COLBERT_SCORE,
            values_by_id=cb,
        ),
    )

    ce_rank = {turn_id: rank for rank, turn_id in enumerate(ce_order, start=1)}
    cb_rank = {turn_id: rank for rank, turn_id in enumerate(colbert_order, start=1)}
    expected = {
        turn_id: 0.7 / (60 + ce_rank[turn_id]) + 0.3 / (60 + cb_rank[turn_id]) for turn_id in ids
    }
    assert [candidate.weighted_rrf for candidate in result.candidates] == sorted(
        expected.values(), reverse=True
    )
    for candidate in result.candidates:
        assert candidate.cross_encoder_rank == ce_rank[candidate.turn_id]
        assert candidate.colbert_rank == cb_rank[candidate.turn_id]
        assert candidate.weighted_rrf == expected[candidate.turn_id]
        assert candidate.e1c_rank == candidate.selected_rank


def test_equal_e1c_rrf_uses_prior_e1a_rank() -> None:
    _, source = _source(turns=75)
    pool = list(source.candidates)
    earlier = pool[0]
    later = pool[1]

    remaining = [candidate for candidate in pool if candidate not in (earlier, later)]
    ce_order = remaining[:2] + [later] + remaining[2:3] + [earlier] + remaining[3:]
    colbert_order = remaining[:56] + [earlier] + remaining[56:73] + [later]
    assert ce_order.index(later) + 1 == 3
    assert ce_order.index(earlier) + 1 == 5
    assert colbert_order.index(earlier) + 1 == 57
    assert colbert_order.index(later) + 1 == 75
    # Assign the rationally equal pair as (CE rank 3, CB rank 75) and
    # (CE rank 5, CB rank 57).
    # Keep logits inside the non-saturating binary64 range so sigmoid preserves
    # the intended CrossEncoder rank rather than legitimately tying at 1.0.
    ce = {candidate.turn_id: 4.0 - rank / 10.0 for rank, candidate in enumerate(ce_order, start=1)}
    cb = {
        candidate.turn_id: float(1000 - rank)
        for rank, candidate in enumerate(colbert_order, start=1)
    }

    result = select_e1c(
        source,
        cross_encoder=_observation(
            source,
            ScoreChannel.CROSS_ENCODER_LOGIT,
            values_by_id=ce,
        ),
        colbert=_observation(
            source,
            ScoreChannel.COLBERT_SCORE,
            values_by_id=cb,
        ),
    )
    by_id = {candidate.turn_id: candidate for candidate in result.candidates}
    assert by_id[later.turn_id].weighted_rrf == by_id[earlier.turn_id].weighted_rrf
    assert by_id[earlier.turn_id].selected_rank < by_id[later.turn_id].selected_rank


def test_e1d_retains_equality_and_rejects_just_below_threshold() -> None:
    _, source = _source(turns=4)
    ids = [candidate.turn_id for candidate in source.candidates]
    max_logit = math.log(0.25 / 0.75)
    equality_logit = math.log(0.0075 / (1.0 - 0.0075))
    below_logit = math.nextafter(equality_logit, -math.inf)
    logits = {
        ids[0]: max_logit,
        ids[1]: equality_logit,
        ids[2]: below_logit,
        ids[3]: -1e308,
    }
    colbert = {turn_id: float(10 - rank) for rank, turn_id in enumerate(ids)}

    result = select_e1d(
        source,
        cross_encoder=_observation(
            source,
            ScoreChannel.CROSS_ENCODER_LOGIT,
            values_by_id=logits,
        ),
        colbert=_observation(
            source,
            ScoreChannel.COLBERT_SCORE,
            values_by_id=colbert,
        ),
    )

    assert stable_sigmoid(max_logit) == 0.25
    assert stable_sigmoid(equality_logit) == 0.0075
    assert stable_sigmoid(below_logit) < 0.0075
    assert ids[0] in {candidate.turn_id for candidate in result.candidates}
    assert ids[1] in {candidate.turn_id for candidate in result.candidates}
    assert ids[2] not in {candidate.turn_id for candidate in result.candidates}
    trace = result.content_free_trace()
    assert trace["selection"]["absolute_cross_encoder_sigmoid_threshold"] == 0.0075
    assert trace["selection"]["threshold_denominator_scope"] == ("E1-C-preselection-head-only")
    assert trace["selection"]["keep_comparison"] == ">="
    equality = next(
        item
        for item in trace["selection"]["pre_threshold_head"]
        if item["turn_id"] == list(ids[1].as_tuple())
    )
    assert equality["retained_by_e1d"] is True


def test_e1d_threshold_denominator_is_the_preselected_head_not_the_full_pool() -> None:
    _, source = _source(turns=4)
    ids = [candidate.turn_id for candidate in source.candidates]
    # CE rank 1 is deliberately last in ColBERT. With a custom one-candidate
    # head, the E1-C winner is CE rank 2, so the global and head maxima differ.
    ce_order = [ids[0], ids[1], ids[2], ids[3]]
    cb_order = [ids[1], ids[2], ids[3], ids[0]]
    ce = {turn_id: float(10 - rank) / 10.0 for rank, turn_id in enumerate(ce_order, start=1)}
    cb = {turn_id: float(10 - rank) for rank, turn_id in enumerate(cb_order, start=1)}
    head_one = replace(
        E1_PROTOCOL,
        protocol_version="swarmbrain-longmemeval-smartsearch-shaped-e1-scope-test-v2",
        adaptive_fused_head=1,
    )

    result = select_e1d(
        source,
        cross_encoder=_observation(
            source,
            ScoreChannel.CROSS_ENCODER_LOGIT,
            values_by_id=ce,
        ),
        colbert=_observation(
            source,
            ScoreChannel.COLBERT_SCORE,
            values_by_id=cb,
        ),
        protocol=head_one,
    )
    trace = result.content_free_trace()
    winner = result.candidates[0]
    assert winner.turn_id != ids[0]
    assert trace["selection"]["maximum_cross_encoder_sigmoid_over_e1c_head"] == (
        winner.cross_encoder_sigmoid
    )
    assert trace["selection"]["maximum_cross_encoder_sigmoid_over_e1c_head"] < max(
        stable_sigmoid(value) for value in ce.values()
    )


def test_e1d_takes_exactly_e1c_top_60_before_thresholding() -> None:
    _, source = _source(
        turns=128,
        lexical_positions=list(range(128)),
        dense_positions=list(reversed(range(128))),
    )
    zero = {candidate.turn_id: 0.0 for candidate in source.candidates}
    ce = _observation(
        source,
        ScoreChannel.CROSS_ENCODER_LOGIT,
        values_by_id=zero,
    )
    cb = _observation(source, ScoreChannel.COLBERT_SCORE, values_by_id=zero)
    e1c = select_e1c(source, cross_encoder=ce, colbert=cb)
    e1d = select_e1d(source, cross_encoder=ce, colbert=cb)

    assert len(e1d.candidates) == 60
    assert [candidate.turn_id for candidate in e1d.candidates] == [
        candidate.turn_id for candidate in e1c.candidates[:60]
    ]
    assert e1d.content_free_trace()["selection"]["pre_threshold_head_count"] == 60


def test_missing_extra_and_duplicate_score_coverage_fail_closed() -> None:
    _, source = _source()
    binding = bind_e1a_pool(source)
    ids = [candidate.turn_id for candidate in source.candidates]

    with pytest.raises(E1SelectionError, match="repeats"):
        PoolScoreObservation(
            channel=ScoreChannel.CROSS_ENCODER_LOGIT,
            **binding.as_dict(),
            identity=_scorer_identity(ScoreChannel.CROSS_ENCODER_LOGIT),
            scores=(
                TurnScoreObservation(ids[0], 1.0),
                TurnScoreObservation(ids[0], 0.0),
                TurnScoreObservation(ids[2], -1.0),
                TurnScoreObservation(ids[3], -2.0),
            ),
        )

    missing = replace(
        _observation(source, ScoreChannel.CROSS_ENCODER_LOGIT),
        pool_count=len(ids) - 1,
        scores=_observation(source, ScoreChannel.CROSS_ENCODER_LOGIT).scores[:-1],
    )
    with pytest.raises(E1SelectionError, match="pool_count differs"):
        select_e1b(source, cross_encoder=missing)

    unknown = LongMemEvalTurnId("q-e1", 999, 0)
    ordinary = _observation(source, ScoreChannel.CROSS_ENCODER_LOGIT)
    extra = replace(
        ordinary,
        scores=ordinary.scores[:-1] + (TurnScoreObservation(unknown, 0.0),),
    )
    with pytest.raises(E1SelectionError, match="cover the fixed E1-A pool exactly once"):
        select_e1b(source, cross_encoder=extra)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("question_id", "different-question", "different question"),
        ("query_sha256", _digest("different-query"), "different query"),
        (
            "turn_corpus_projection_sha256",
            _digest("different-corpus"),
            "different turn corpus",
        ),
        ("e1a_trace_sha256", _digest("different-trace"), "different E1-A trace"),
        ("e1a_pool_sha256", _digest("different-pool"), "different E1-A pool"),
    ],
)
def test_question_query_corpus_trace_and_pool_binding_tampering_is_rejected(
    field: str,
    value: str,
    message: str,
) -> None:
    _, source = _source()
    observation = replace(
        _observation(source, ScoreChannel.CROSS_ENCODER_LOGIT),
        **{field: value},
    )
    with pytest.raises(E1SelectionError, match=message):
        select_e1b(source, cross_encoder=observation)


def test_pool_cap_is_exact_and_a_dropped_e1a_candidate_is_extra() -> None:
    _, source = _source(
        turns=130,
        lexical_positions=list(range(128)),
        dense_positions=list(range(2, 130)),
    )
    assert len(source.pre_cap_candidates) == 130
    assert len(source.candidates) == E1_POOL_CAP == 128
    observation = _observation(source, ScoreChannel.CROSS_ENCODER_LOGIT)
    assert len(select_e1b(source, cross_encoder=observation).candidates) == 128

    fixed_ids = {candidate.turn_id for candidate in source.candidates}
    dropped = next(
        candidate for candidate in source.pre_cap_candidates if candidate.turn_id not in fixed_ids
    )
    replaced = replace(
        observation,
        scores=observation.scores[:-1] + (TurnScoreObservation(dropped.turn_id, dropped.raw_rrf),),
    )
    with pytest.raises(E1SelectionError, match="cover the fixed E1-A pool exactly once"):
        select_e1b(source, cross_encoder=replaced)


def test_tampered_hydrated_source_order_is_rejected() -> None:
    _, source = _source()
    tampered = replace(source, candidates=tuple(reversed(source.candidates)))
    with pytest.raises(E1SelectionError, match="fused ranks must be contiguous"):
        bind_e1a_pool(tampered)


def test_downstream_replay_rejects_a_forged_e1b_result_and_trace() -> None:
    _, source = _source()
    observation = _observation(source, ScoreChannel.CROSS_ENCODER_LOGIT)
    result = select_e1b(source, cross_encoder=observation)
    forged_candidates = tuple(
        replace(candidate, selected_rank=rank)
        for rank, candidate in enumerate(reversed(result.candidates), start=1)
    )
    trace = result.content_free_trace()
    trace.pop("trace_sha256")
    bindings = [candidate.content_free_binding() for candidate in forged_candidates]
    trace["selection"]["output_candidates"] = bindings
    trace["selection"]["output_candidates_sha256"] = hashlib.sha256(
        json.dumps(
            bindings,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    forged = replace(
        result,
        candidates=forged_candidates,
        _trace_canonical_json=json.dumps(
            trace,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )

    with pytest.raises(E1SelectionError, match="deterministic replay"):
        validate_e1b_result(
            forged,
            source=source,
            cross_encoder=observation,
        )
    assert (
        validate_e1b_result(
            result,
            source=source,
            cross_encoder=observation,
        )
        is result
    )


def test_trace_is_content_free_immutable_and_explicit_about_missing_evidence() -> None:
    _, source = _source()
    ce = _observation(source, ScoreChannel.CROSS_ENCODER_LOGIT)
    cb = _observation(source, ScoreChannel.COLBERT_SCORE)
    result = select_e1c(source, cross_encoder=ce, colbert=cb)

    assert TURN_SENTINEL in result.turns[0].original_content
    encoded = json.dumps(result.content_free_trace(), ensure_ascii=False, sort_keys=True)
    assert QUERY_SENTINEL not in encoded
    assert TURN_SENTINEL not in encoded
    assert ANSWER_SENTINEL not in encoded
    trace = result.content_free_trace()
    assert trace["accounting"]["local_model_calls"] == 0
    assert trace["accounting"]["local_database_calls"] == 0
    assert trace["accounting"]["local_network_calls"] == 0
    assert trace["claims"] == {
        "external_scorer_model_revision_and_artifact_identity_verified": False,
        "external_observation_artifact_verified": False,
        "external_scores_recomputed": False,
        "gold_fields_present_in_selection_input": False,
        "prompt_packing_executed": False,
        "reader_or_judge_executed": False,
        "qa_improvement_proven": False,
        "SmartSearch_model_reproduction_proven": False,
    }
    assert all(
        observation["identity"]["identity_source"] == "caller-attested-unverified"
        for observation in trace["observations"]
    )
    original_digest = result.trace_sha256
    trace["selection"]["output_candidates"].clear()
    fresh = result.content_free_trace()
    assert result.trace_sha256 == original_digest
    assert len(fresh["selection"]["output_candidates"]) == len(source.candidates)
    assert fresh["trace_sha256"] == original_digest


def test_empty_fixed_pool_has_deterministic_empty_outputs() -> None:
    _, source = _source(turns=1, lexical_positions=[], dense_positions=[])
    assert source.candidates == ()
    ce = _observation(source, ScoreChannel.CROSS_ENCODER_LOGIT)
    cb = _observation(source, ScoreChannel.COLBERT_SCORE)

    for result in (
        select_e1b(source, cross_encoder=ce),
        select_e1c(source, cross_encoder=ce, colbert=cb),
        select_e1d(source, cross_encoder=ce, colbert=cb),
    ):
        assert result.candidates == ()
        assert result.turns == ()
    d_trace = select_e1d(source, cross_encoder=ce, colbert=cb).content_free_trace()
    assert d_trace["selection"]["maximum_cross_encoder_sigmoid_over_e1c_head"] is None
    assert d_trace["selection"]["absolute_cross_encoder_sigmoid_threshold"] is None


def test_public_selectors_accept_no_gold_prompt_reader_or_judge_input() -> None:
    for selector in (select_e1b, select_e1c, select_e1d):
        parameters = inspect.signature(selector).parameters
        assert "gold" not in parameters
        assert "answer" not in parameters
        assert "prompt" not in parameters
        assert "reader" not in parameters
        assert "judge" not in parameters
