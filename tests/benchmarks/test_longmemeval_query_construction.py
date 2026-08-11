from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest
from benchmarks.integrations.longmemeval_e1 import (
    ExternalScorerIdentity,
    PoolScoreObservation,
    ScoreChannel,
    TurnScoreObservation,
    bind_e1a_pool,
    select_e1b,
    select_e1c,
)
from benchmarks.integrations.longmemeval_official_preflight import (
    OFFICIAL_DATASET_REQUIREMENT,
    DatasetRequirement,
    ExactTokenizerPin,
    freeze_pinned_preflight,
)
from benchmarks.integrations.longmemeval_query_construction import (
    CONSTRUCTOR_SYSTEM_PROMPT_SHA256,
    CONSTRUCTOR_USER_PROMPT_SHA256,
    EXTRACTIVE_SEPARATOR,
    MAX_WINDOW_MESSAGES,
    PROVIDER_RESPONSE_PARSER,
    RECEIPT_REPLAY,
    RETRIEVED_TURNS,
    WINDOW_STRIDE,
    ConstructionCell,
    ConstructionDecision,
    ConstructorIdentity,
    DecisionOperation,
    QueryConstructionError,
    RetentionStyle,
    SourceSpan,
    WindowPosition,
    build_query_windows,
    build_retrieved_turn_pool,
    canonical_json_bytes,
    compile_query_construction,
    constructor_request_bytes,
    replay_window_construction_receipt,
    sha256_json,
    sha256_utf8,
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

QUESTION = "Which alpha facts were mentioned? PRIVATE-QUERY-SENTINEL"
ANSWER = "PRIVATE-GOLD-ANSWER-SENTINEL"
QUESTION_TYPE = "PRIVATE-QUESTION-TYPE-SENTINEL"
CURRENT_DATE = "2025/02/01 (Sat) 12:00"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _lane_identity(lane: RetrievalLane) -> ExternalLaneIdentity:
    return ExternalLaneIdentity(
        producer=f"fixture-{lane.value}",
        scorer=ImmutableArtifactIdentity(
            name=f"{lane.value}-scorer",
            revision="immutable-r1",
            artifact_sha256=_digest(f"lane-model:{lane.value}"),
        ),
        projection=ImmutableArtifactIdentity(
            name="turn-projection",
            revision="immutable-r1",
            artifact_sha256=_digest(f"projection:{lane.value}"),
        ),
        observation_artifact_sha256=_digest(f"lane-observation:{lane.value}"),
    )


def _source_bytes(
    turns: int = 128,
    *,
    question: str = QUESTION,
    current_date: str = CURRENT_DATE,
    answer: str = ANSWER,
    first_content: str | None = None,
) -> bytes:
    record = {
        "question_id": "q-e7",
        "question_type": QUESTION_TYPE,
        "question": question,
        "answer": answer,
        "question_date": current_date,
        "haystack_session_ids": ["session-e7"],
        "haystack_dates": ["2025/01/31 (Fri) 09:00"],
        "haystack_sessions": [
            [
                {
                    "role": "user" if position % 2 == 0 else "assistant",
                    "content": (
                        first_content
                        if position == 0 and first_content is not None
                        else f"message {position:02d} alpha beta PRIVATE-TURN-{position:02d}"
                    ),
                }
                for position in range(turns)
            ]
        ],
        "answer_session_ids": ["session-e7"],
    }
    return (json.dumps([record], separators=(",", ":")) + "\n").encode()


def _fixture(turns: int = 128):
    raw = _source_bytes(turns)
    corpus = compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="synthetic-e7.json",
    )
    query_sha256 = sha256_utf8(QUESTION)

    def lane(kind: RetrievalLane, order: list[int]) -> RankedLaneObservation:
        return RankedLaneObservation(
            lane=kind,
            requested_depth=E1A_PROTOCOL.lane_depth(kind),
            query_sha256=query_sha256,
            turn_corpus_projection_sha256=corpus.projection_sha256,
            identity=_lane_identity(kind),
            candidates=tuple(
                RankedTurnObservation(
                    turn_id=LongMemEvalTurnId("q-e7", 0, position),
                    raw_score=float(turns - rank),
                )
                for rank, position in enumerate(order)
            ),
            examined_count=turns,
        )

    e1a = fuse_question_turns(
        corpus,
        question_id="q-e7",
        query_text=QUESTION,
        lexical=lane(RetrievalLane.LEXICAL, list(range(turns))),
        dense=lane(RetrievalLane.DENSE, list(range(turns))),
    )
    binding = bind_e1a_pool(e1a)
    scorer = ExternalScorerIdentity(
        producer="fixture-cross-encoder-runner",
        scorer="fixture-cross-encoder",
        model="fixture/model",
        revision="immutable-r1",
        artifact_sha256=_digest("ce-model"),
        observation_artifact_sha256=_digest("ce-observation"),
    )
    ce = PoolScoreObservation(
        channel=ScoreChannel.CROSS_ENCODER_LOGIT,
        question_id=binding.question_id,
        query_sha256=binding.query_sha256,
        turn_corpus_projection_sha256=binding.turn_corpus_projection_sha256,
        e1a_trace_sha256=binding.e1a_trace_sha256,
        e1a_pool_sha256=binding.e1a_pool_sha256,
        pool_count=binding.pool_count,
        identity=scorer,
        scores=tuple(
            TurnScoreObservation(
                turn_id=candidate.turn_id,
                raw_score=float(turns - candidate.turn_id.turn_position),
            )
            for candidate in e1a.candidates
        ),
    )
    e1b = select_e1b(e1a, cross_encoder=ce)
    pool = build_retrieved_turn_pool(
        corpus,
        e1b,
        source=e1a,
        cross_encoder=ce,
        query=QUESTION,
    )
    preflight = freeze_pinned_preflight(
        raw,
        dataset=DatasetRequirement(
            name="Synthetic-E7",
            source_label="synthetic-e7.json",
            source_sha256=hashlib.sha256(raw).hexdigest(),
            question_count=1,
            official=False,
        ),
        tokenizer=ExactTokenizerPin(
            model="fixture-reader-tokenizer",
            revision="immutable-r1",
            artifact_sha256=_digest("reader-tokenizer"),
            executable_sha256=_digest("reader-tokenizer-executable"),
        ),
    )
    batch = build_query_windows(
        corpus,
        pool,
        preflight,
        source_bytes=raw,
        selection=e1b,
        e1_source=e1a,
        cross_encoder=ce,
        query=QUESTION,
        current_date=CURRENT_DATE,
    )
    return corpus, e1a, ce, e1b, pool, batch


def _identity() -> ConstructorIdentity:
    return ConstructorIdentity(
        producer="fixture-external-constructor",
        model="fixture/qwen-4b",
        revision="immutable-r1",
        deployment="fixture-deployment",
        model_artifact_sha256=_digest("constructor-model"),
        system_prompt_sha256=CONSTRUCTOR_SYSTEM_PROMPT_SHA256,
        user_prompt_sha256=CONSTRUCTOR_USER_PROMPT_SHA256,
        tokenizer_model="fixture/constructor-tokenizer",
        tokenizer_revision="immutable-r1",
        tokenizer_artifact_sha256=_digest("constructor-tokenizer"),
    )


def _drop(turn_id: LongMemEvalTurnId) -> ConstructionDecision:
    return ConstructionDecision(
        turn_id=turn_id,
        operation=DecisionOperation.DROP,
        style=RetentionStyle.DROP,
        compressed_content="",
        reason="not relevant",
        support_spans=(),
    )


def _verbatim(message) -> ConstructionDecision:
    raw = message.turn.original_content.encode("utf-8")
    return ConstructionDecision(
        turn_id=message.turn.turn_id,
        operation=DecisionOperation.KEEP,
        style=RetentionStyle.VERBATIM,
        compressed_content=message.turn.original_content,
        reason="direct evidence",
        support_spans=(
            SourceSpan(
                turn_id=message.turn.turn_id,
                start_byte=0,
                end_byte=len(raw),
            ),
        ),
    )


def _extractive(message) -> ConstructionDecision:
    raw = message.turn.original_content.encode("utf-8")
    first = raw.index(b"message")
    first_end = raw.index(b" alpha")
    second = raw.index(b"alpha")
    second_end = second + len(b"alpha")
    pieces = (
        SourceSpan(message.turn.turn_id, first, first_end),
        SourceSpan(message.turn.turn_id, second, second_end),
    )
    return ConstructionDecision(
        turn_id=message.turn.turn_id,
        operation=DecisionOperation.KEEP,
        style=RetentionStyle.EXTRACTIVE,
        compressed_content=EXTRACTIVE_SEPARATOR.join(
            raw[item.start_byte : item.end_byte].decode() for item in pieces
        ),
        reason="two exact relevant spans",
        support_spans=pieces,
    )


def _receipts(batch, *, keep_all: bool = True):
    identity = _identity()
    receipts = []
    for window in batch.windows:
        decisions = tuple(
            _verbatim(message) if keep_all else _drop(message.turn.turn_id)
            for message in window.messages
        )
        input_tokens = 100 + window.window_index
        output_tokens = 20
        raw_response = canonical_json_bytes(
            {
                "id": f"provider-request-{window.window_index}",
                "model": identity.model,
                "system_fingerprint": "fixture-system-fingerprint",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": canonical_json_bytes(
                                {
                                    "decisions": [
                                        decision.provider_payload() for decision in decisions
                                    ]
                                }
                            ).decode(),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
        )
        receipts.append(
            replay_window_construction_receipt(
                batch,
                window,
                raw_response=raw_response,
                request_id=f"local-request-{window.window_index}",
                identity=identity,
                latency_ms=10.0 + window.window_index,
                cost_usd=0.001,
            )
        )
    return tuple(receipts)


def _with_decisions(receipt, decisions):
    frozen = tuple(decisions)
    raw_response = json.loads(receipt.raw_response)
    raw_response["choices"][0]["message"]["content"] = canonical_json_bytes(
        {"decisions": [decision.provider_payload() for decision in frozen]}
    ).decode()
    return replace(
        receipt,
        decisions=frozen,
        raw_response=canonical_json_bytes(raw_response),
        response_sha256=sha256_json([decision.response_payload() for decision in frozen]),
    )


def test_exact_paper_window_shape_and_overlap() -> None:
    _, _, _, _, pool, batch = _fixture()

    assert len(pool.turns) == RETRIEVED_TURNS == 50
    assert len(batch.windows) == 8
    assert all(len(window.messages) <= MAX_WINDOW_MESSAGES == 8 for window in batch.windows)
    assert WINDOW_STRIDE == 7
    indexes = [
        [message.global_message_index for message in window.messages] for window in batch.windows
    ]
    assert indexes[0] == list(range(8))
    assert indexes[1] == list(range(7, 15))
    assert indexes[-1] == [49, 50, 51]
    assert all(indexes[position][-1] == indexes[position + 1][0] for position in range(7))
    assert all(
        message.position is WindowPosition.CORE
        for window in batch.windows[:-1]
        for message in window.messages
    )
    assert [message.position for message in batch.windows[-1].messages] == [
        WindowPosition.CORE,
        WindowPosition.NEXT_BRIDGE,
        WindowPosition.NEXT_BRIDGE,
    ]


def test_constructor_request_uses_only_query_date_and_source_messages() -> None:
    _, _, _, _, _, batch = _fixture()
    request = constructor_request_bytes(batch, batch.windows[0]).decode()

    assert QUESTION in request
    assert CURRENT_DATE in request
    assert "PRIVATE-TURN-00" in request
    assert ANSWER not in request
    assert QUESTION_TYPE not in request
    assert "answer_session_ids" not in request
    assert "question_id" not in request
    assert "has_answer" not in request


def test_raw_top50_control_is_chronological_and_constructor_free() -> None:
    _, _, _, _, _, batch = _fixture()
    result = compile_query_construction(batch, cell=ConstructionCell.RAW_TOP50)
    trace = result.content_free_trace()

    assert len(result.items) == 50
    assert [item.global_message_index for item in result.items] == list(range(50))
    assert trace["accounting"]["constructor_windows"] == 0
    assert trace["claims"]["all_output_byte_grounded"] is True
    assert "PRIVATE-TURN-00" in result.reader_context
    assert "PRIVATE-TURN-00" not in json.dumps(trace)


def test_grounded_construction_deduplicates_overlap_and_reports_window_latency() -> None:
    _, _, _, _, _, batch = _fixture()
    receipts = _receipts(batch)
    result = compile_query_construction(
        batch,
        cell=ConstructionCell.GROUNDED_QUERY_CONSTRUCTION,
        receipts=receipts,
    )
    trace = result.content_free_trace()

    assert len(result.items) == 52
    by_index = {item.global_message_index: item for item in result.items}
    assert by_index[7].appearances == 2
    assert by_index[7].keep_votes == 2
    assert by_index[7].selected_window_index == 0
    assert by_index[49].appearances == 2
    assert trace["accounting"]["constructor_windows"] == 8
    assert trace["accounting"]["constructor_usage_replayed_windows"] == 8
    assert trace["accounting"]["constructor_provider_response_bytes"] == sum(
        len(receipt.raw_response) for receipt in receipts
    )
    assert trace["accounting"]["constructor_latency_sum_ms"] == sum(
        10.0 + index for index in range(8)
    )
    assert trace["accounting"]["constructor_window_max_latency_ms"] == 17.0
    assert trace["accounting"]["constructor_batch_wall_clock_ms"] is None
    assert trace["claims"]["all_output_byte_grounded"] is True
    assert trace["claims"]["abstractive_faithfulness_proven"] is False
    assert trace["claims"]["raw_provider_response_replay_required"] is True
    assert trace["claims"]["raw_provider_response_replay_complete"] is True
    assert trace["claims"]["provider_usage_replay_complete"] is True
    provider_evidence = trace["constructor"]["receipts"][0]["provider_response"]
    assert provider_evidence["parser"] == PROVIDER_RESPONSE_PARSER
    assert provider_evidence["raw_sha256"] == hashlib.sha256(receipts[0].raw_response).hexdigest()
    assert provider_evidence["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }
    assert trace["constructor"]["receipts"][0]["replay"] == RECEIPT_REPLAY
    encoded = json.dumps(trace)
    assert QUESTION not in encoded
    assert ANSWER not in encoded
    assert "PRIVATE-TURN" not in encoded
    assert "local-request-0" not in encoded
    assert "provider-request-0" not in encoded
    assert "fixture-system-fingerprint" not in encoded


def test_exact_extractive_spans_are_accepted() -> None:
    _, _, _, _, _, batch = _fixture()
    receipts = list(_receipts(batch, keep_all=False))
    first = receipts[0]
    decisions = list(first.decisions)
    decisions[0] = _extractive(batch.windows[0].messages[0])
    receipts[0] = _with_decisions(first, decisions)

    result = compile_query_construction(
        batch,
        cell=ConstructionCell.GROUNDED_QUERY_CONSTRUCTION,
        receipts=tuple(receipts),
    )

    assert len(result.items) == 1
    assert result.items[0].compressed_content == f"message 00{EXTRACTIVE_SEPARATOR}alpha"
    assert result.items[0].style is RetentionStyle.EXTRACTIVE


def test_abstractive_output_is_analysis_only_and_rejected_by_grounded_cell() -> None:
    _, _, _, _, _, batch = _fixture()
    receipts = list(_receipts(batch, keep_all=False))
    first = receipts[0]
    message = batch.windows[0].messages[0]
    raw = message.turn.original_content.encode()
    abstract = ConstructionDecision(
        turn_id=message.turn.turn_id,
        operation=DecisionOperation.KEEP,
        style=RetentionStyle.ABSTRACTIVE,
        compressed_content="The user mentioned alpha.",
        reason="query-conditioned paraphrase",
        support_spans=(SourceSpan(message.turn.turn_id, 0, len(raw)),),
    )
    decisions = list(first.decisions)
    decisions[0] = abstract
    receipts[0] = _with_decisions(first, decisions)

    analysis = compile_query_construction(
        batch,
        cell=ConstructionCell.QUERY_CONSTRUCTION,
        receipts=tuple(receipts),
    )
    assert analysis.items[0].style is RetentionStyle.ABSTRACTIVE
    assert analysis.content_free_trace()["claims"]["all_output_byte_grounded"] is False

    with pytest.raises(QueryConstructionError, match="E7-C forbids"):
        compile_query_construction(
            batch,
            cell=ConstructionCell.GROUNDED_QUERY_CONSTRUCTION,
            receipts=tuple(receipts),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("request_digest", "request digest"),
        ("request_bytes", "byte count"),
        ("window_digest", "unknown window"),
        ("prompt", "system prompt"),
        ("duplicate_request", "request ID was reused"),
        ("duplicate_provider", "provider request ID was reused"),
        ("missing_receipt", "cover every window"),
    ],
)
def test_receipt_tampering_fails_closed(mutation: str, message: str) -> None:
    _, _, _, _, _, batch = _fixture()
    receipts = list(_receipts(batch))
    if mutation == "request_digest":
        receipts[0] = replace(receipts[0], request_sha256=_digest("wrong"))
    elif mutation == "request_bytes":
        receipts[0] = replace(receipts[0], request_utf8_bytes=receipts[0].request_utf8_bytes + 1)
    elif mutation == "window_digest":
        receipts[0] = replace(receipts[0], window_sha256=_digest("wrong-window"))
    elif mutation == "prompt":
        receipts[0] = replace(
            receipts[0],
            identity=replace(
                receipts[0].identity,
                system_prompt_sha256=_digest("wrong-prompt"),
            ),
        )
    elif mutation == "duplicate_request":
        receipts[1] = replace(receipts[1], request_id=receipts[0].request_id)
    elif mutation == "duplicate_provider":
        raw_response = json.loads(receipts[1].raw_response)
        raw_response["id"] = receipts[0].provider_request_id
        receipts[1] = replace(
            receipts[1],
            provider_request_id=receipts[0].provider_request_id,
            raw_response=canonical_json_bytes(raw_response),
        )
    else:
        receipts.pop()

    with pytest.raises(QueryConstructionError, match=message):
        compile_query_construction(
            batch,
            cell=ConstructionCell.GROUNDED_QUERY_CONSTRUCTION,
            receipts=tuple(receipts),
        )


def test_decision_and_span_tampering_fails_closed() -> None:
    _, _, _, _, _, batch = _fixture()
    receipts = list(_receipts(batch))
    first = receipts[0]
    message = batch.windows[0].messages[0]

    wrong_text = replace(first.decisions[0], compressed_content="not the cited bytes")
    receipts[0] = _with_decisions(first, (wrong_text, *first.decisions[1:]))
    with pytest.raises(QueryConstructionError, match="verbatim KEEP"):
        compile_query_construction(
            batch,
            cell=ConstructionCell.GROUNDED_QUERY_CONSTRUCTION,
            receipts=tuple(receipts),
        )

    bad_span = SourceSpan(message.turn.turn_id, 0, 10_000)
    bad = replace(
        first.decisions[0],
        support_spans=(bad_span,),
    )
    receipts[0] = _with_decisions(first, (bad, *first.decisions[1:]))
    with pytest.raises(QueryConstructionError, match="outside"):
        compile_query_construction(
            batch,
            cell=ConstructionCell.GROUNDED_QUERY_CONSTRUCTION,
            receipts=tuple(receipts),
        )


def test_normalized_response_digest_is_bound_to_parsed_decisions() -> None:
    _, _, _, _, _, batch = _fixture()
    receipt = _receipts(batch)[0]

    with pytest.raises(QueryConstructionError, match="normalized decisions"):
        replace(receipt, response_sha256=_digest("unrelated-response"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("model", "pinned constructor model"),
        ("provider_id", "differs from the raw provider response"),
        ("usage", "token accounting differs"),
        ("total", "token usage does not reconcile"),
        ("finish", "did not finish with stop"),
        ("decision", "decisions differ from replayed"),
        ("extra_content_field", "content fields differ"),
    ],
)
def test_raw_provider_response_replay_fails_closed(mutation: str, message: str) -> None:
    _, _, _, _, _, batch = _fixture()
    receipt = _receipts(batch)[0]
    payload = json.loads(receipt.raw_response)
    if mutation == "model":
        payload["model"] = "forged/model"
    elif mutation == "provider_id":
        payload["id"] = "forged-provider-request"
    elif mutation == "usage":
        payload["usage"]["prompt_tokens"] += 1
        payload["usage"]["total_tokens"] += 1
    elif mutation == "total":
        payload["usage"]["total_tokens"] += 1
    elif mutation == "finish":
        payload["choices"][0]["finish_reason"] = "length"
    else:
        content = json.loads(payload["choices"][0]["message"]["content"])
        if mutation == "decision":
            content["decisions"][0]["compressed_content"] = "forged but well-formed"
        else:
            content["forged_extra"] = False
        payload["choices"][0]["message"]["content"] = canonical_json_bytes(content).decode()

    with pytest.raises(QueryConstructionError, match=message):
        replace(receipt, raw_response=canonical_json_bytes(payload))


def test_raw_provider_response_rejects_duplicate_keys_and_parser_drift() -> None:
    _, _, _, _, _, batch = _fixture()
    receipt = _receipts(batch)[0]
    payload = json.loads(receipt.raw_response)
    content = payload["choices"][0]["message"]["content"]
    payload["choices"][0]["message"]["content"] = content.replace(
        '{"decisions":',
        '{"decisions":[],"decisions":',
        1,
    )

    with pytest.raises(QueryConstructionError, match="malformed JSON"):
        replace(receipt, raw_response=canonical_json_bytes(payload))
    with pytest.raises(QueryConstructionError, match="response parser drifted"):
        replace(receipt, response_parser="forged-parser")


def test_pool_requires_real_e1b_top50_and_matching_query() -> None:
    corpus, e1a, ce, _, _, _ = _fixture()
    colbert = PoolScoreObservation(
        channel=ScoreChannel.COLBERT_SCORE,
        question_id=ce.question_id,
        query_sha256=ce.query_sha256,
        turn_corpus_projection_sha256=ce.turn_corpus_projection_sha256,
        e1a_trace_sha256=ce.e1a_trace_sha256,
        e1a_pool_sha256=ce.e1a_pool_sha256,
        pool_count=ce.pool_count,
        identity=replace(ce.identity, scorer="colbert", model="colbert"),
        scores=ce.scores,
    )
    e1c = select_e1c(e1a, cross_encoder=ce, colbert=colbert)

    with pytest.raises(QueryConstructionError, match="E1-B"):
        build_retrieved_turn_pool(
            corpus,
            e1c,
            source=e1a,
            cross_encoder=ce,
            query=QUESTION,
        )
    with pytest.raises(QueryConstructionError, match="query"):
        build_retrieved_turn_pool(
            corpus,
            select_e1b(e1a, cross_encoder=ce),
            source=e1a,
            cross_encoder=ce,
            query="wrong",
        )


def test_pool_replays_e1b_from_source_evidence_before_accepting_result() -> None:
    corpus, e1a, ce, e1b, _, _ = _fixture()
    forged = replace(
        e1b,
        candidates=(e1b.candidates[1], e1b.candidates[0], *e1b.candidates[2:]),
    )

    with pytest.raises(QueryConstructionError, match="deterministic replay"):
        build_retrieved_turn_pool(
            corpus,
            forged,
            source=e1a,
            cross_encoder=ce,
            query=QUESTION,
        )


def test_windowing_requires_authoritative_preflight_date_and_source() -> None:
    corpus, e1a, ce, e1b, pool, batch = _fixture()

    with pytest.raises(QueryConstructionError, match="current_date"):
        build_query_windows(
            corpus,
            pool,
            batch.preflight,
            source_bytes=_source_bytes(),
            selection=e1b,
            e1_source=e1a,
            cross_encoder=ce,
            query=QUESTION,
            current_date="2025/02/02 (Sun) 12:00",
        )
    wrong_dataset = replace(
        batch.preflight.dataset,
        source_sha256=_digest("wrong-source"),
    )
    wrong_preflight = replace(batch.preflight, dataset=wrong_dataset)
    with pytest.raises(QueryConstructionError, match="exact preflight refreeze"):
        build_query_windows(
            corpus,
            pool,
            wrong_preflight,
            source_bytes=_source_bytes(),
            selection=e1b,
            e1_source=e1a,
            cross_encoder=ce,
            query=QUESTION,
            current_date=CURRENT_DATE,
        )


def test_windowing_rejects_dataclass_forged_query_and_date_bindings() -> None:
    corpus, e1a, ce, e1b, pool, batch = _fixture()
    source_bytes = _source_bytes()
    case = batch.preflight.cases[0]

    forged_query = "FORGED-QUERY-MUST-NOT-BECOME-AUTHORITATIVE"
    forged_query_bytes = forged_query.encode()
    forged_query_case = replace(
        case,
        question_sha256=hashlib.sha256(forged_query_bytes).hexdigest(),
        question_utf8_bytes=len(forged_query_bytes),
    )
    forged_query_preflight = replace(
        batch.preflight,
        cases=(forged_query_case,),
    )
    forged_query_pool = replace(pool, query_sha256=sha256_utf8(forged_query))
    with pytest.raises(QueryConstructionError, match="manifest rebuilt"):
        build_query_windows(
            corpus,
            forged_query_pool,
            forged_query_preflight,
            source_bytes=source_bytes,
            selection=e1b,
            e1_source=e1a,
            cross_encoder=ce,
            query=forged_query,
            current_date=CURRENT_DATE,
        )

    forged_date = "2025/02/02 (Sun) 12:00"
    forged_date_bytes = forged_date.encode()
    forged_date_case = replace(
        case,
        current_date_sha256=hashlib.sha256(forged_date_bytes).hexdigest(),
        current_date_utf8_bytes=len(forged_date_bytes),
    )
    forged_date_preflight = replace(
        batch.preflight,
        cases=(forged_date_case,),
    )
    with pytest.raises(QueryConstructionError, match="manifest rebuilt"):
        build_query_windows(
            corpus,
            pool,
            forged_date_preflight,
            source_bytes=source_bytes,
            selection=e1b,
            e1_source=e1a,
            cross_encoder=ce,
            query=QUESTION,
            current_date=forged_date,
        )


def test_official_corpus_identity_cannot_be_downgraded_to_synthetic_freezer() -> None:
    corpus, e1a, ce, e1b, pool, batch = _fixture()
    shadow_dataset = replace(OFFICIAL_DATASET_REQUIREMENT, official=False)
    question_ids = [f"shadow-q-{index}" for index in range(shadow_dataset.question_count)]
    shadow_cases = tuple(
        replace(
            batch.preflight.cases[0],
            case_index=index,
            question_id=question_id,
        )
        for index, question_id in enumerate(question_ids)
    )
    shadow_preflight = replace(
        batch.preflight,
        dataset=shadow_dataset,
        question_ids_sha256=sha256_json(question_ids),
        cases=shadow_cases,
    )

    with pytest.raises(QueryConstructionError, match="nonofficial downgrade"):
        build_query_windows(
            corpus,
            pool,
            shadow_preflight,
            source_bytes=_source_bytes(),
            selection=e1b,
            e1_source=e1a,
            cross_encoder=ce,
            query=QUESTION,
            current_date=CURRENT_DATE,
        )


def test_batch_rejects_dataclass_poisoned_window_date() -> None:
    _, _, _, _, _, batch = _fixture()
    poisoned = replace(
        batch.windows[0],
        current_date_sha256=_digest("poisoned-window-date"),
    )

    with pytest.raises(QueryConstructionError, match="query/current-date binding"):
        replace(batch, windows=(poisoned, *batch.windows[1:]))


def test_batch_refreeze_receipt_rejects_replaced_preflight_and_windows() -> None:
    _, _, _, _, _, batch = _fixture()
    forged_date = "2025/02/02 (Sun) 12:00"
    forged_date_bytes = forged_date.encode()
    forged_case = replace(
        batch.preflight.cases[0],
        current_date_sha256=hashlib.sha256(forged_date_bytes).hexdigest(),
        current_date_utf8_bytes=len(forged_date_bytes),
    )
    forged_preflight = replace(batch.preflight, cases=(forged_case,))
    forged_windows = tuple(
        replace(window, current_date_sha256=sha256_utf8(forged_date)) for window in batch.windows
    )

    with pytest.raises(QueryConstructionError, match="sealed builder authority"):
        replace(
            batch,
            preflight=forged_preflight,
            current_date=forged_date,
            windows=forged_windows,
        )


def test_batch_authority_rejects_answer_bearing_same_question_windows() -> None:
    _, _, _, _, _, batch = _fixture()
    poisoned_source = _source_bytes(first_content=ANSWER)
    poisoned_corpus = compile_dataset_bytes(
        poisoned_source,
        expected_sha256=hashlib.sha256(poisoned_source).hexdigest(),
        source_label="poisoned-synthetic-e7.json",
    )
    poisoned_message = replace(
        batch.windows[0].messages[0],
        turn=poisoned_corpus.turns[0],
    )
    poisoned_window = replace(
        batch.windows[0],
        messages=(poisoned_message, *batch.windows[0].messages[1:]),
    )
    poisoned_windows = (poisoned_window, *batch.windows[1:])
    trace = batch.content_free_trace()
    trace.pop("trace_sha256")
    bindings = [window.content_free_binding() for window in poisoned_windows]
    trace["windows"] = bindings
    trace["windows_sha256"] = sha256_json(bindings)

    with pytest.raises(QueryConstructionError, match="window set.*sealed builder authority"):
        replace(
            batch,
            windows=poisoned_windows,
            _trace_canonical_json=canonical_json_bytes(trace).decode(),
        )


@pytest.mark.parametrize("mutation", ["extra_top_level", "extra_accounting", "bool_alias"])
def test_batch_trace_requires_exact_fields_and_accounting(mutation: str) -> None:
    _, _, _, _, _, batch = _fixture()
    trace = batch.content_free_trace()
    trace.pop("trace_sha256")
    if mutation == "extra_top_level":
        trace["forged_extra"] = 0
        message = "trace fields"
    elif mutation == "extra_accounting":
        trace["accounting"]["forged_extra"] = 0
        message = "accounting"
    else:
        trace["accounting"]["local_model_calls"] = False
        message = "accounting"

    with pytest.raises(QueryConstructionError, match=message):
        replace(
            batch,
            _trace_canonical_json=canonical_json_bytes(trace).decode(),
        )


def test_gold_mutation_cannot_cross_source_refreeze_boundary() -> None:
    corpus, e1a, ce, e1b, pool, batch = _fixture()
    request = constructor_request_bytes(batch, batch.windows[0]).decode()
    trace = json.dumps(batch.content_free_trace())

    assert ANSWER not in request
    assert ANSWER not in trace

    mutated_source = _source_bytes(answer="FORGED-GOLD-SENTINEL-MUST-NOT-ENTER-E7")
    with pytest.raises(QueryConstructionError, match="exact preflight refreeze"):
        build_query_windows(
            corpus,
            pool,
            batch.preflight,
            source_bytes=mutated_source,
            selection=e1b,
            e1_source=e1a,
            cross_encoder=ce,
            query=QUESTION,
            current_date=CURRENT_DATE,
        )


def test_contracts_are_immutable_and_drop_cannot_smuggle_content() -> None:
    _, _, _, _, pool, _ = _fixture()
    with pytest.raises(FrozenInstanceError):
        pool.question_id = "changed"  # type: ignore[misc]
    with pytest.raises(QueryConstructionError, match="empty content"):
        ConstructionDecision(
            turn_id=pool.turns[0].turn_id,
            operation=DecisionOperation.DROP,
            style=RetentionStyle.DROP,
            compressed_content="smuggled",
            reason="irrelevant",
            support_spans=(),
        )
