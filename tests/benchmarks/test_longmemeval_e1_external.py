from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from benchmarks.integrations.longmemeval_turns import compile_dataset_bytes
from scripts._longmemeval_common import QWEN_QUERY_INSTRUCTION, select_questions
from scripts.run_longmemeval_e1_external import (
    ExternalE1Error,
    SelectedQuestion,
    _dense_observation_rows,
    _rank_dense_rows,
    _retrieval_metrics,
    canonical_json_bytes,
    load_json,
    padding_aware_batches,
    qwen_query_text,
    render_deepseek_chat_prompt,
    seal_artifact,
    selected_positions,
    validate_sealed_artifact,
    verify_provider_prompt_tokens,
    write_json,
)
from scripts.run_longmemeval_e1_external import lexical_scores as external_lexical_scores


def _record(*, answer: str = "gold") -> dict[str, object]:
    return {
        "question_id": "q-external",
        "question_type": "multi-session",
        "question": "alpha beta",
        "answer": answer,
        "question_date": "2025/02/01 (Sat) 12:00",
        "haystack_session_ids": ["old", "new", "irrelevant"],
        "haystack_dates": [
            "2025/01/29 (Wed) 09:00",
            "2025/01/31 (Fri) 09:00",
            "2025/01/30 (Thu) 09:00",
        ],
        "haystack_sessions": [
            [{"role": "user", "content": "the exact alpha beta phrase"}],
            [{"role": "user", "content": "BETA then ALPHA"}],
            [{"role": "assistant", "content": "nothing relevant"}],
        ],
        "answer_session_ids": ["new"],
    }


def _corpus(*, answer: str = "gold"):
    record = _record(answer=answer)
    raw = (json.dumps([record], separators=(",", ":")) + "\n").encode()
    corpus = compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="synthetic-external-e1.json",
    )
    return record, corpus


def test_sealed_artifact_round_trips_and_rejects_tampering(tmp_path) -> None:
    sealed = seal_artifact({"kind": "fixture", "score": 0.25})
    path = tmp_path / "sealed.json"
    write_json(path, sealed)

    assert load_json(path, sealed=True) == sealed
    assert validate_sealed_artifact(sealed) == sealed

    tampered = {**sealed, "score": 0.5}
    with pytest.raises(ExternalE1Error, match="digest"):
        validate_sealed_artifact(tampered)


def test_strict_json_loader_rejects_duplicate_fields(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_bytes(b'{"a":1,"a":2}\n')

    with pytest.raises(ExternalE1Error, match="duplicate"):
        load_json(path)


def test_seeded_positions_match_shared_selector() -> None:
    records = [{"question_id": str(index)} for index in range(25)]
    positions = selected_positions(len(records), sample=7, seed=20260807)

    assert [records[position] for position in positions] == select_questions(
        records,
        sample=7,
        seed=20260807,
    )
    assert positions == tuple(sorted(positions))
    assert len(set(positions)) == 7


def test_qwen_query_uses_the_frozen_production_instruction_exactly() -> None:
    query = "Where did the user go?"

    assert qwen_query_text(query) == (f"Instruct: {QWEN_QUERY_INSTRUCTION}\nQuery: {query}")
    assert canonical_json_bytes({"query": qwen_query_text(query)}) == (
        b'{"query":"Instruct: Given a coding-agent memory search query, retrieve relevant '
        b'memories\\nQuery: Where did the user go?"}'
    )


def test_deepseek_one_message_chat_surface_is_exact() -> None:
    prompt = "Reader prompt with unicode: Café"

    assert render_deepseek_chat_prompt(prompt) == (
        "<｜begin▁of▁sentence｜><｜User｜>Reader prompt with unicode: Café<｜Assistant｜></think>"
    )


def test_lexical_score_matches_overlap_substring_and_production_ties() -> None:
    _, corpus = _corpus()

    scored = external_lexical_scores("alpha beta", corpus.turns)

    assert [(turn.turn_id.session_position, score) for turn, score in scored] == [
        (0, 1.2),
        (1, 1.0),
    ]


def test_gold_answer_mutation_cannot_change_external_lexical_ranking() -> None:
    _, first = _corpus(answer="first private gold")
    _, second = _corpus(answer="different private gold")

    first_scores = [
        (turn.turn_id.as_tuple(), score)
        for turn, score in external_lexical_scores("alpha beta", first.turns)
    ]
    second_scores = [
        (turn.turn_id.as_tuple(), score)
        for turn, score in external_lexical_scores("alpha beta", second.turns)
    ]
    assert first_scores == second_scores


def test_dense_lane_clamps_cosine_and_uses_production_descending_id_tie() -> None:
    record, corpus = _corpus()
    question = SelectedQuestion(position=0, record=record, turns=corpus.turns)
    rows = _dense_observation_rows(
        question,
        cosine_scores=(-0.5, 0.4, 0.4),
        token_counts=(7, 8, 9),
        truncated=(False, False, True),
    )

    assert rows[0]["lane_score"] == 0.0
    assert [row["turn_id"][1] for row in _rank_dense_rows(rows)] == [2, 1, 0]
    assert rows[2]["truncated_right_at_8192"] is True


def test_padding_aware_batches_isolate_outliers_and_bound_attention_work() -> None:
    encoded = [
        (position, [position] * length, False)
        for position, length in enumerate([10] * 8 + [100, 2000, 5670])
    ]

    batches = padding_aware_batches(encoded, maximum_batch_size=8)

    assert [[len(item[1]) for item in batch] for batch in batches] == [
        [10] * 8,
        [100],
        [2000],
        [5670],
    ]
    attention_bound = padding_aware_batches(
        [(0, [0] * 4096, False), (1, [1] * 4096, False), (2, [2] * 4096, False)],
        maximum_batch_size=8,
    )
    assert [len(batch) for batch in attention_bound] == [2, 1]


def test_api_usage_must_equal_exact_local_prompt_count() -> None:
    result = SimpleNamespace(prompt_tokens=123)

    verify_provider_prompt_tokens(result, expected=123, label="fixture")
    with pytest.raises(ExternalE1Error, match="differs"):
        verify_provider_prompt_tokens(result, expected=124, label="fixture")


def test_gold_is_consumed_only_by_posthoc_retrieval_metrics() -> None:
    record, corpus = _corpus()
    question = SelectedQuestion(position=0, record=record, turns=corpus.turns)
    ranked = [list(turn.turn_id.as_tuple()) for turn in corpus.turns]
    kept = [list(corpus.turns[1].turn_id.as_tuple())]

    metrics = _retrieval_metrics(question, ranked_ids=ranked, kept_ids=kept)

    assert metrics == {
        "gold_session_positions": 1,
        "packed_gold_session_positions": 1,
        "any_gold_session_in_prompt": True,
        "all_gold_sessions_in_prompt": True,
        "answer_session_recall": 1.0,
        "candidate_mrr": 0.5,
    }


def test_seal_rejects_caller_supplied_digest() -> None:
    with pytest.raises(ExternalE1Error, match="cannot already carry"):
        seal_artifact({"artifact_sha256": "0" * 64})
