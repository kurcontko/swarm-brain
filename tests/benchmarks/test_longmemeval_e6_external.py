from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.integrations.longmemeval_representation.evidence import (
    DeepSeekR1PricingIdentity,
)
from run_longmemeval_qa import chat_request_bytes, chat_result_from_raw_response
from scripts import run_longmemeval_e6_external as runner
from scripts.run_longmemeval_e1_external import SelectedQuestion


def test_e6_v2_uses_a_fresh_protocol_and_output_namespace() -> None:
    assert runner.E6_RUN_PROTOCOL_VERSION.endswith("development-v2")
    assert runner.DEFAULT_E6_OUTPUT.name.endswith("pilot-v2")


def _question(position: int) -> SelectedQuestion:
    return SelectedQuestion(
        position=position,
        record={
            "question_id": "case",
            "question": "Question?",
            "question_type": "single-session-user",
            "answer": "Answer.",
            "question_date": "2025/01/01 (Wed) 00:00",
        },
        turns=(),
    )


def test_qa_arm_order_is_position_balanced() -> None:
    assert runner._qa_arm_order(_question(2)) == ("R0", "R1")
    assert runner._qa_arm_order(_question(3)) == ("R1", "R0")


def test_session_digest_is_exact_utf8_sha256() -> None:
    assert runner._session_digest("session-α") == hashlib.sha256("session-α".encode()).hexdigest()


def _pricing() -> DeepSeekR1PricingIdentity:
    return DeepSeekR1PricingIdentity(
        version="fixture-v1",
        artifact_sha256=hashlib.sha256(b"fixture-pricing").hexdigest(),
        cache_miss_input_microusd_per_million_tokens=140_000,
        output_microusd_per_million_tokens=280_000,
    )


def _context(tmp_path: Path):
    return SimpleNamespace(
        output_dir=tmp_path,
        manifest={"artifact_sha256": "0" * 64},
        pricing=_pricing(),
    )


def _chat_result(*, content: str = "ok", max_tokens: int = 16):
    prompt = "fixture prompt"
    raw_request = chat_request_bytes(
        prompt=prompt,
        model=runner.DEEPSEEK_MODEL,
        temperature=0.0,
        max_tokens=max_tokens,
        thinking_mode="disabled",
    )
    raw_response = json.dumps(
        {
            "id": "provider-fixture",
            "model": runner.DEEPSEEK_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "total_tokens": 6,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return chat_result_from_raw_response(
        raw_response,
        prompt=prompt,
        attempts=1,
        latency_ms=1.25,
        raw_request=raw_request,
        endpoint_url=runner.DEEPSEEK_DEPLOYMENT_ID,
    )


def test_external_journal_reservation_is_durable_and_replay_costed(tmp_path: Path) -> None:
    context = _context(tmp_path)
    result = _chat_result()
    route = "qa:fixture"
    paths = runner._call_journal_paths(context, namespace="qa", route=route)
    reserved = context.pricing.upper_bound_microusd(
        input_tokens=5,
        output_tokens=16,
        retry_count=runner.DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS - 1,
        request_max_tokens=16,
    )
    reservation = runner._reservation_artifact(
        context,
        namespace="qa",
        route=route,
        raw_request_sha256=result.raw_request_sha256,
        exact_prompt_tokens=5,
        request_max_tokens=16,
        reserved_microusd=reserved,
    )
    runner._write_reservation(context, paths, reservation)
    assert runner._external_journal_cost(context) == (reserved, 1)

    response = runner._call_response_artifact(
        context,
        reservation=reservation,
        result=result,
    )
    assert response == runner._raw_call_response_artifact(
        context,
        reservation=reservation,
        raw_request=result.raw_request,
        raw_response=result.raw_response,
        endpoint_url=result.endpoint_url,
        attempts=result.attempts,
        latency_ms=result.latency_ms,
        content_encoding="",
    )
    runner._write_call_response(paths, response)
    actual = runner._chat_cost_microusd(result, pricing=context.pricing)
    runner._write_settlement(
        paths,
        runner._settlement_artifact(
            context,
            reservation=reservation,
            evidence_sha256=response["artifact_sha256"],
            actual_microusd=actual,
        ),
    )
    assert runner._external_journal_cost(context) == (actual, 0)


def test_reservation_recomputes_worst_case_price(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(runner.ExternalE6Error, match="worst-case pricing"):
        runner._reservation_artifact(
            context,
            namespace="qa",
            route="qa:fixture",
            raw_request_sha256="1" * 64,
            exact_prompt_tokens=5,
            request_max_tokens=16,
            reserved_microusd=1,
        )


def _write_prompt_fixture(context, question: SelectedQuestion) -> None:
    runner.write_json(
        runner.e6_phase_path(context, "prompts", question),
        runner.seal_artifact(
            {
                "arms": {
                    "R0": {"prompt": "fixture prompt"},
                    "R1": {"prompt": "fixture prompt"},
                }
            }
        ),
    )


def _write_first_qa_journal(context, question: SelectedQuestion, *, response: bool) -> int:
    tokenizer = SimpleNamespace(exact_count=lambda _prompt: 5)
    paths, reservation, _, _, reserved = runner._expected_qa_reservation(
        context,
        question,
        cell="R0",
        role="reader",
        route_index=0,
        prompt="fixture prompt",
        max_tokens=runner.READER_MAX_TOKENS,
        tokenizer=tokenizer,
    )
    runner._write_reservation(context, paths, reservation)
    if response:
        result = _chat_result(max_tokens=runner.READER_MAX_TOKENS)
        runner._write_call_response(
            paths,
            runner._call_response_artifact(
                context,
                reservation=reservation,
                result=result,
            ),
        )
    return reserved


def test_qa_response_wal_recovers_receipt_and_settlement_without_call(tmp_path: Path) -> None:
    context = _context(tmp_path)
    question = _question(2)
    _write_prompt_fixture(context, question)
    _write_first_qa_journal(context, question, response=True)
    tokenizer = SimpleNamespace(exact_count=lambda _prompt: 5)

    receipts = runner._reconcile_qa_question_journals(
        context,
        question,
        tokenizer=tokenizer,
    )

    assert len(receipts) == 1
    assert receipts[0]["call_role"] == "reader"
    _, unresolved = runner._external_journal_cost(context)
    assert unresolved == 0


def test_qa_reservation_without_response_blocks_reissue(tmp_path: Path) -> None:
    context = _context(tmp_path)
    question = _question(2)
    _write_prompt_fixture(context, question)
    reserved = _write_first_qa_journal(context, question, response=False)
    tokenizer = SimpleNamespace(exact_count=lambda _prompt: 5)

    with pytest.raises(runner.ExternalE6Error, match="refusing to reissue"):
        runner._reconcile_qa_question_journals(
            context,
            question,
            tokenizer=tokenizer,
        )
    assert runner._external_journal_cost(context) == (reserved, 1)


def test_output_process_lock_rejects_second_owner(tmp_path: Path) -> None:
    with (
        runner._output_process_lock(tmp_path),
        pytest.raises(runner.ExternalE6Error, match="another E6 process"),
        runner._output_process_lock(tmp_path),
    ):
        raise AssertionError("unreachable")


def test_validate_qa_chat_rejects_empty_content() -> None:
    result = _chat_result(content="")
    tokenizer = SimpleNamespace(exact_count=lambda _prompt: 5)
    with pytest.raises(runner.ExternalE6Error, match="response content is empty"):
        runner._validate_qa_chat(
            result,
            expected_prompt="fixture prompt",
            expected_max_tokens=16,
            tokenizer=tokenizer,
            label="fixture",
        )


def test_qa_transport_checkpoints_and_returns_first_empty_2xx() -> None:
    empty = _chat_result(content="", max_tokens=16)

    class Response:
        status_code = 200
        headers: dict[str, str] = {}
        content = empty.raw_response

    class Client:
        calls = 0

        async def post(self, *_args, **_kwargs):
            self.calls += 1
            return Response()

        async def aclose(self) -> None:
            return None

    async def exercise() -> None:
        transport = Client()
        client = runner._QAFirst2xxChatClient(
            api_key="fixture",
            max_tokens=16,
            client=transport,
        )
        checkpoints: list[bytes] = []
        result = await client.complete(
            "fixture prompt",
            raw_request=empty.raw_request,
            on_first_2xx=lambda raw, *_rest: checkpoints.append(raw),
        )
        assert result.content == ""
        assert checkpoints == [empty.raw_response]
        assert transport.calls == 1

    asyncio.run(exercise())


def test_response_wal_retains_zero_byte_malformed_2xx(tmp_path: Path) -> None:
    context = _context(tmp_path)
    result = _chat_result(max_tokens=16)
    route = "qa:zero-byte"
    paths = runner._call_journal_paths(context, namespace="qa", route=route)
    reserved = context.pricing.upper_bound_microusd(
        input_tokens=5,
        output_tokens=16,
        retry_count=runner.DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS - 1,
        request_max_tokens=16,
    )
    reservation = runner._reservation_artifact(
        context,
        namespace="qa",
        route=route,
        raw_request_sha256=result.raw_request_sha256,
        exact_prompt_tokens=5,
        request_max_tokens=16,
        reserved_microusd=reserved,
    )
    runner._write_reservation(context, paths, reservation)
    runner._write_call_response(
        paths,
        runner._raw_call_response_artifact(
            context,
            reservation=reservation,
            raw_request=result.raw_request,
            raw_response=b"",
            endpoint_url=runner.DEEPSEEK_DEPLOYMENT_ID,
            attempts=1,
            latency_ms=1.0,
            content_encoding="identity",
        ),
    )
    assert runner.load_json(paths.response, sealed=True)["provider_response"]["raw_bytes"] == 0
    with pytest.raises(Exception, match="raw response must be non-empty bytes"):
        runner._replay_call_response(context, paths, reservation=reservation)
    assert runner._external_journal_cost(context) == (reserved, 1)


def test_merged_score_rows_enforce_order_token_and_cosine_bounds() -> None:
    keys = [
        SimpleNamespace(key_id="a", key_text_sha256="1" * 64),
        SimpleNamespace(key_id="b", key_text_sha256="2" * 64),
    ]
    corpus = SimpleNamespace(derived_keys=keys)
    rows = [
        {
            "key_id": "a",
            "key_text_sha256": "1" * 64,
            "input_tokens_after_truncation": 8192,
            "truncated_right_at_8192": True,
            "raw_cosine": 1.0,
        },
        {
            "key_id": "b",
            "key_text_sha256": "2" * 64,
            "input_tokens_after_truncation": 8,
            "truncated_right_at_8192": False,
            "raw_cosine": -1.0,
        },
    ]
    assert runner._validate_merged_score_rows(rows, corpus=corpus) == rows
    with pytest.raises(runner.ExternalE6Error, match="canonical derived-key order"):
        runner._validate_merged_score_rows(list(reversed(rows)), corpus=corpus)
    above_limit = [dict(rows[0], input_tokens_after_truncation=8193), rows[1]]
    with pytest.raises(runner.ExternalE6Error, match="accounting is invalid"):
        runner._validate_merged_score_rows(above_limit, corpus=corpus)
    bad_cosine = [dict(rows[0], raw_cosine=1.1), rows[1]]
    with pytest.raises(runner.ExternalE6Error, match="numeric tolerance"):
        runner._validate_merged_score_rows(bad_cosine, corpus=corpus)


def test_qwen_batch_plan_reconstruction_is_deterministic() -> None:
    assert runner._expected_qwen_batch_plan([1, 2, 100], maximum_batch_size=8) == [
        [0, 1],
        [2],
    ]


def test_tokenizer_receipt_namespace_is_arm_distinct(tmp_path: Path) -> None:
    context = _context(tmp_path)
    question = _question(2)
    r0 = runner._tokenizer_receipt_namespace(context, question, "R0")
    r1 = runner._tokenizer_receipt_namespace(context, question, "R1")
    assert r0["namespace_sha256"] != r1["namespace_sha256"]


def test_spend_ledger_reserves_settles_and_enforces_cap() -> None:
    async def exercise() -> None:
        ledger = runner._SpendLedger(spent_microusd=10, maximum_microusd=100)
        await ledger.reserve(50)
        assert ledger.reserved_microusd == 50
        await ledger.settle(reserved=50, actual=20)
        assert (ledger.spent_microusd, ledger.reserved_microusd) == (30, 0)
        with pytest.raises(runner.ExternalE6Error, match=r"exceed the \$6 cap"):
            await ledger.reserve(71)

    asyncio.run(exercise())


def test_limited_all_stops_before_qa_and_final_report(monkeypatch, capsys, tmp_path: Path) -> None:
    context = object()
    calls: list[str] = []
    monkeypatch.setattr(runner, "build_e6_context", lambda _args: context)
    monkeypatch.setattr(
        runner,
        "run_extraction_phase",
        lambda *_args, **_kwargs: calls.append("extract"),
    )
    monkeypatch.setattr(
        runner,
        "run_rank_phase",
        lambda *_args, **_kwargs: calls.append("rank"),
    )
    monkeypatch.setattr(
        runner,
        "run_pack_phase",
        lambda *_args, **_kwargs: calls.append("pack"),
    )

    def diagnose(*_args, **_kwargs):
        calls.append("diagnose")
        return {"artifact_sha256": "0" * 64}

    monkeypatch.setattr(runner, "build_diagnostic_report", diagnose)
    monkeypatch.setattr(
        runner,
        "run_qa_phase",
        lambda *_args, **_kwargs: calls.append("qa"),
    )
    monkeypatch.setattr(
        runner,
        "build_report",
        lambda *_args, **_kwargs: calls.append("report"),
    )
    assert runner.main(["all", "--limit", "1", "--output-dir", str(tmp_path)]) == 0
    assert calls == ["extract", "rank", "pack", "diagnose"]
    assert '"artifact_sha256"' in capsys.readouterr().out
