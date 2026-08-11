from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.integrations.longmemeval_turns import LongMemEvalTurnId
from scripts.run_longmemeval_e1_external import (
    DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION,
    DEEPSEEK_MODEL,
    DEEPSEEK_OUTPUT_USD_PER_MILLION,
    SelectedQuestion,
)
from scripts.run_longmemeval_e7_external import (
    ExternalE7Error,
    _constructor_chat_bytes,
    _constructor_chat_record,
    _e7_upper_bound_cost,
    _load_constructor_chat_records,
    _pack_cell,
    _qa_arm_order,
    _validate_constructor_chat_record,
    _validate_frozen_qa_result,
    main,
)
from scripts.run_longmemeval_qa import (
    chat_request_bytes,
    chat_result_from_raw_response,
)


def _question(*, position: int = 0) -> SelectedQuestion:
    return SelectedQuestion(
        position=position,
        record={
            "question_id": "q-e7-external",
            "question": "Which evidence matters?",
            "question_date": "2025/02/01 (Sat) 12:00",
        },
        turns=(),
    )


def _chat_result(
    *,
    prompt: str = "frozen constructor prompt",
    request_model: str = DEEPSEEK_MODEL,
    response_model: str = DEEPSEEK_MODEL,
    request_max_tokens: int = 4096,
    thinking_mode: str = "disabled",
    endpoint_url: str = "https://api.deepseek.com/v1/chat/completions",
    request_id: str | None = "fixture-request-id",
    attempts: int = 1,
):
    raw_request = chat_request_bytes(
        prompt=prompt,
        model=request_model,
        temperature=0.0,
        max_tokens=request_max_tokens,
        thinking_mode=thinking_mode,
    )
    raw_response = json.dumps(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "{}", "role": "assistant"},
                }
            ],
            "id": request_id,
            "model": response_model,
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 4,
                "total_tokens": 5,
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return chat_result_from_raw_response(
        raw_response,
        prompt=prompt,
        attempts=attempts,
        latency_ms=1.0,
        raw_request=raw_request,
        endpoint_url=endpoint_url,
    )


def test_constructor_chat_record_round_trips_exactly() -> None:
    question = _question()
    result = _chat_result()
    record = _constructor_chat_record(question, 0, result)

    assert _validate_constructor_chat_record(question, 0, record) == result


def test_frozen_qa_result_accepts_exact_reader_contract() -> None:
    _validate_frozen_qa_result(
        _chat_result(),
        expected_prompt="frozen constructor prompt",
        max_tokens=4096,
        label="fixture/reader",
    )


@pytest.mark.parametrize(
    ("result", "expected_prompt"),
    (
        (_chat_result(request_model="forged-model"), "frozen constructor prompt"),
        (_chat_result(response_model="forged-model"), "frozen constructor prompt"),
        (_chat_result(request_max_tokens=64), "frozen constructor prompt"),
        (_chat_result(thinking_mode="enabled"), "frozen constructor prompt"),
        (
            _chat_result(endpoint_url="https://fixture.invalid/v1/chat/completions"),
            "frozen constructor prompt",
        ),
        (_chat_result(request_id=None), "frozen constructor prompt"),
        (_chat_result(), "different prompt"),
    ),
)
def test_frozen_qa_result_rejects_identity_and_configuration_drift(
    result,
    expected_prompt: str,
) -> None:
    with pytest.raises(ExternalE7Error):
        _validate_frozen_qa_result(
            result,
            expected_prompt=expected_prompt,
            max_tokens=4096,
            label="fixture/reader",
        )


def test_retry_cost_prices_each_unseen_attempt_at_request_maximum() -> None:
    single = _chat_result(attempts=1)
    retried = _chat_result(attempts=2)
    unseen_attempt = (
        retried.prompt_tokens * DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION
        + retried.request.max_tokens * DEEPSEEK_OUTPUT_USD_PER_MILLION
    ) / 1_000_000.0

    assert _e7_upper_bound_cost(retried) == pytest.approx(
        _e7_upper_bound_cost(single) + unseen_attempt
    )


@pytest.mark.parametrize(
    "mutation",
    ("extra_nested", "digest_shadow", "bool_byte_count", "integer_latency"),
)
def test_constructor_chat_record_rejects_nested_schema_and_type_tampering(
    mutation: str,
) -> None:
    question = _question()
    record = deepcopy(_constructor_chat_record(question, 0, _chat_result()))
    if mutation == "extra_nested":
        record["provider_response"]["gold_sentinel"] = "must-not-enter"
    elif mutation == "digest_shadow":
        correct = record["provider_response"]["raw_sha256"]
        record["provider_response"]["raw_sha256"] = "0" * 64
        record["provider_response"]["sha256"] = correct
    elif mutation == "bool_byte_count":
        record["provider_request"]["raw_bytes"] = True
    else:
        record["transport"]["latency_ms"] = 1

    with pytest.raises(ExternalE7Error):
        _validate_constructor_chat_record(question, 0, record)


@pytest.mark.parametrize(
    "payload",
    (b'{"a":1,"a":2}\n', b'{"a":NaN}\n'),
)
def test_constructor_jsonl_loader_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "receipts.jsonl"
    path.write_bytes(payload)

    with pytest.raises(ExternalE7Error, match="malformed"):
        _load_constructor_chat_records(path)


def test_constructor_jsonl_loader_preserves_exact_canonical_record(
    tmp_path: Path,
) -> None:
    record = _constructor_chat_record(_question(), 0, _chat_result())
    path = tmp_path / "receipts.jsonl"
    path.write_bytes(_constructor_chat_bytes([record]))

    assert _load_constructor_chat_records(path) == (record,)


class _FakeReceipt:
    def __init__(self, token_count: int) -> None:
        self.token_count = token_count

    def content_free_binding(self) -> dict[str, int]:
        return {"token_count": self.token_count}


class _FakeTokenizer:
    identity = SimpleNamespace(as_dict=lambda: {"model": "fake"})

    def reset_receipts(self) -> None:
        return None

    def count_prompt(self, prompt: str, *, query_sha256: str) -> _FakeReceipt:
        assert len(query_sha256) == 64
        return _FakeReceipt(len(prompt))


class _FakeItem:
    def __init__(self, position: int, content: str) -> None:
        self.turn = SimpleNamespace(turn_id=LongMemEvalTurnId("q-e7-external", 0, position))
        self.compressed_content = content

    def content_free_binding(self) -> dict[str, object]:
        return {
            "turn_id": list(self.turn.turn_id.as_tuple()),
            "compressed_content_utf8": {
                "bytes": len(self.compressed_content.encode()),
                "sha256": "a" * 64,
            },
        }


def test_constructed_item_packer_skips_over_budget_and_continues(monkeypatch) -> None:
    items = (
        _FakeItem(0, "aa"),
        _FakeItem(1, "bbbb"),
        _FakeItem(2, "c"),
    )
    result = SimpleNamespace(items=items, trace_sha256="b" * 64)
    monkeypatch.setattr("scripts.run_longmemeval_e7_external.TOKEN_BUDGET", 5)
    monkeypatch.setattr(
        "scripts.run_longmemeval_e7_external._reader_prompt",
        lambda _question, selected: " ".join(item.compressed_content for item in selected),
    )

    trace, prompt = _pack_cell(
        _question(),
        cell="E7-C",
        result=result,
        tokenizer=_FakeTokenizer(),
    )

    assert prompt == "aa c"
    assert trace["final_prompt"]["tokens"] == 4
    assert trace["kept_turn_ids"] == [
        ["q-e7-external", 0, 0],
        ["q-e7-external", 0, 2],
    ]
    assert trace["dropped_turn_ids"] == [["q-e7-external", 0, 1]]
    assert [decision["accepted"] for decision in trace["decisions"]] == [
        True,
        False,
        True,
    ]


def test_e7_qa_order_rotates_and_report_refuses_partial_sample() -> None:
    assert _qa_arm_order(_question(position=0)) == ("E7-A", "E7-C")
    assert _qa_arm_order(_question(position=1)) == ("E7-C", "E7-A")

    with pytest.raises(SystemExit, match="complete frozen sample"):
        main(["report", "--limit", "1"])
