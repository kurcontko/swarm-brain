from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from benchmarks.integrations.longmemeval_turn_prompt import (
    CHAIN_BLOCK_SEPARATOR,
    CHAIN_HEADER_BODY_SEPARATOR,
    CHAIN_HEADER_TEMPLATE,
    CHAIN_TURN_SEPARATOR,
    PRIMARY_TOKEN_BUDGET,
    SUPPORTED_TOKEN_BUDGETS,
    TOKENIZER_PROTOCOL,
    ExactTokenCountReceipt,
    OrderedTurnBlocks,
    TokenizerIdentity,
    TurnPromptPackingError,
    pack_turn_prompt,
)
from benchmarks.integrations.longmemeval_turns import TurnProjection, compile_dataset_bytes
from scripts._longmemeval_common import EMPTY_CONTEXT_NOTE, OFFICIAL_ANSWER_TEMPLATE

QUESTION = "QUESTION-TEXT-MUST-NOT-ENTER-TRACE"
CURRENT_DATE = "2025/01/04 (Sat) 11:00"


def _record(
    question_id: str,
    contents: list[str],
    *,
    answer: str = "GOLD-ANSWER-MUST-NOT-AFFECT-PACKING",
    has_answer: bool = False,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question_type": "multi-session",
        "question": QUESTION,
        "answer": answer,
        "question_date": CURRENT_DATE,
        "haystack_session_ids": [f"session-{question_id}"],
        "haystack_dates": ["2025/01/03 (Fri) 09:07"],
        "haystack_sessions": [
            [
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": content,
                    "has_answer": has_answer,
                }
                for index, content in enumerate(contents)
            ]
        ],
        "answer_session_ids": [f"session-{question_id}"],
    }


def _compile(records: list[dict[str, Any]]):
    raw = (
        json.dumps(records, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    return compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="synthetic-longmemeval.json",
    )


ReceiptTransform = Callable[[ExactTokenCountReceipt, int, str], ExactTokenCountReceipt]
CountFunction = Callable[[str, int], int]


class FixtureExactReceiptIssuer:
    """Exact synthetic byte-tokenizer evidence; it performs no external call."""

    def __init__(
        self,
        *,
        count: CountFunction | None = None,
        transform: ReceiptTransform | None = None,
        identity_drift_after: int | None = None,
    ) -> None:
        self._identity = TokenizerIdentity(
            protocol=TOKENIZER_PROTOCOL,
            model="fixture/exact-utf8-byte-tokenizer",
            revision="fixture-revision-1",
            artifact_sha256="a" * 64,
        )
        self._drifted_identity = TokenizerIdentity(
            protocol=TOKENIZER_PROTOCOL,
            model="fixture/exact-utf8-byte-tokenizer",
            revision="fixture-revision-2",
            artifact_sha256="b" * 64,
        )
        self._count = count or (lambda prompt, _request: len(prompt.encode("utf-8")))
        self._transform = transform
        self._identity_drift_after = identity_drift_after
        self.requests: list[str] = []

    @property
    def identity(self) -> TokenizerIdentity:
        if (
            self._identity_drift_after is not None
            and len(self.requests) >= self._identity_drift_after
        ):
            return self._drifted_identity
        return self._identity

    def count_prompt(
        self,
        prompt: str,
        *,
        query_sha256: str,
    ) -> ExactTokenCountReceipt:
        self.requests.append(prompt)
        request_id = len(self.requests)
        receipt = ExactTokenCountReceipt(
            request_id=request_id,
            provider_request_id=f"fixture-{request_id}",
            tokenizer_identity_sha256=self._identity.identity_sha256,
            query_sha256=query_sha256,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            prompt_utf8_bytes=len(prompt.encode("utf-8")),
            token_count=self._count(prompt, request_id),
        )
        if self._transform is not None:
            return self._transform(receipt, request_id, prompt)
        return receipt


def _pack(
    turns: tuple[TurnProjection, ...],
    tokenizer: FixtureExactReceiptIssuer | None = None,
    *,
    budget: int = PRIMARY_TOKEN_BUDGET,
):
    return pack_turn_prompt(
        OrderedTurnBlocks.linear(turns),
        question_id=turns[0].turn_id.question_id if turns else "q-empty",
        question=QUESTION,
        current_date=CURRENT_DATE,
        token_budget=budget,
        tokenizer=tokenizer or FixtureExactReceiptIssuer(),
    )


def test_linear_pack_renders_exact_official_prompt_and_content_free_trace() -> None:
    corpus = _compile([_record("q-linear", ["FIRST-PRIVATE-TURN", "SECOND-PRIVATE-TURN"])])
    tokenizer = FixtureExactReceiptIssuer()

    result = _pack(corpus.turns, tokenizer)

    history = "\n\n".join(turn.serialized_text for turn in corpus.turns)
    assert result.prompt == OFFICIAL_ANSWER_TEMPLATE.format(history, CURRENT_DATE, QUESTION)
    assert result.trace["budget"] == {
        "token_budget": 8192,
        "supported_token_budgets": [4096, 8192, 16384],
        "primary_token_budget": 8192,
        "is_primary": True,
        "counted_surface": "complete-official-reader-prompt",
    }
    assert result.trace["kept_ids"] == [
        ["q-linear", 0, 0],
        ["q-linear", 0, 1],
    ]
    assert result.trace["dropped_ids"] == []
    assert result.trace["oversized_ids"] == []
    assert result.trace["final_prompt"]["tokens"] == len(result.prompt.encode("utf-8"))
    assert result.trace["tokenizer"] == tokenizer.identity.as_dict()
    assert result.trace["observation_accounting"] == {
        "requests": 6,
        "responses": 6,
        "unique_provider_request_ids": 6,
        "unique_prompt_digests": 4,
        "request_ids_strictly_monotone": True,
        "repeated_prompt_counts_reconciled": True,
    }

    serialized_trace = json.dumps(result.content_free_artifact(), ensure_ascii=False)
    assert QUESTION not in serialized_trace
    assert "FIRST-PRIVATE-TURN" not in serialized_trace
    assert "SECOND-PRIVATE-TURN" not in serialized_trace
    assert "GOLD-ANSWER-MUST-NOT-AFFECT-PACKING" not in serialized_trace
    assert result.trace_sha256 == result.content_free_artifact()["trace_sha256"]


def test_chain_blocks_bind_exact_headers_and_skip_oversized_turn_then_continue() -> None:
    corpus = _compile([_record("q-chain", ["small-first", "X" * 5000, "small-after-overflow"])])
    first, oversized, after = corpus.turns
    ordered = OrderedTurnBlocks.chain_blocks(((first, oversized), (after,)))

    result = pack_turn_prompt(
        ordered,
        question_id="q-chain",
        question=QUESTION,
        current_date=CURRENT_DATE,
        token_budget=4096,
        tokenizer=FixtureExactReceiptIssuer(),
    )

    block_one = (
        f"{CHAIN_HEADER_TEMPLATE.format(chain_number=1)}"
        f"{CHAIN_HEADER_BODY_SEPARATOR}{first.serialized_text}"
    )
    block_two = (
        f"{CHAIN_HEADER_TEMPLATE.format(chain_number=2)}"
        f"{CHAIN_HEADER_BODY_SEPARATOR}{after.serialized_text}"
    )
    history = CHAIN_BLOCK_SEPARATOR.join((block_one, block_two))
    assert result.prompt == OFFICIAL_ANSWER_TEMPLATE.format(history, CURRENT_DATE, QUESTION)
    assert oversized.serialized_text not in result.prompt
    assert result.trace["kept_ids"] == [["q-chain", 0, 0], ["q-chain", 0, 2]]
    assert result.trace["dropped_ids"] == [["q-chain", 0, 1]]
    assert result.trace["oversized_ids"] == [["q-chain", 0, 1]]
    assert result.trace["kept_ids_by_block"] == [
        [["q-chain", 0, 0]],
        [["q-chain", 0, 2]],
    ]
    assert result.trace["layout"]["chain_headers"][1]["literal"] == ("=== Evidence Chain 2 ===")
    assert result.trace["layout"]["chain_turn_separator"]["literal"] == (CHAIN_TURN_SEPARATOR)
    assert result.trace["final_prompt"]["within_budget"] is True


def test_accumulated_overflow_is_not_oversized_and_later_turn_can_still_fit() -> None:
    corpus = _compile(
        [
            _record(
                "q-greedy",
                ["UNIQUE-FIRST", "UNIQUE-MIDDLE", "UNIQUE-LAST"],
            )
        ]
    )

    def registered_fixture_counts(prompt: str, _request: int) -> int:
        contains_first = "UNIQUE-FIRST" in prompt
        contains_middle = "UNIQUE-MIDDLE" in prompt
        contains_last = "UNIQUE-LAST" in prompt
        if contains_first and contains_middle:
            return 5000
        if contains_first and contains_last:
            return 3500
        if contains_first:
            return 3000
        if contains_middle:
            return 1000
        if contains_last:
            return 500
        return 200

    result = _pack(
        corpus.turns,
        FixtureExactReceiptIssuer(count=registered_fixture_counts),
        budget=4096,
    )

    assert result.trace["kept_ids"] == [["q-greedy", 0, 0], ["q-greedy", 0, 2]]
    assert result.trace["dropped_ids"] == [["q-greedy", 0, 1]]
    assert result.trace["oversized_ids"] == []
    assert "UNIQUE-FIRST" in result.prompt
    assert "UNIQUE-MIDDLE" not in result.prompt
    assert "UNIQUE-LAST" in result.prompt


def test_empty_linear_input_uses_shared_empty_context_note_and_exact_recount() -> None:
    tokenizer = FixtureExactReceiptIssuer()

    result = pack_turn_prompt(
        OrderedTurnBlocks.linear(()),
        question_id="q-empty",
        question=QUESTION,
        current_date=CURRENT_DATE,
        tokenizer=tokenizer,
    )

    assert result.prompt == OFFICIAL_ANSWER_TEMPLATE.format(
        EMPTY_CONTEXT_NOTE,
        CURRENT_DATE,
        QUESTION,
    )
    assert result.trace["candidate_order"] == []
    assert [item["purpose"] for item in result.trace["exact_count_observations"]] == [
        "initial-empty-context",
        "final-independent-recount",
    ]
    assert len(tokenizer.requests) == 2


@pytest.mark.parametrize("budget", SUPPORTED_TOKEN_BUDGETS)
def test_only_frozen_budget_curve_values_are_accepted(budget: int) -> None:
    turns = _compile([_record("q-budget", ["evidence"])]).turns

    result = _pack(turns, budget=budget)

    assert result.trace["budget"]["token_budget"] == budget
    assert result.trace["budget"]["is_primary"] is (budget == PRIMARY_TOKEN_BUDGET)


@pytest.mark.parametrize("budget", [0, 2048, 4096.0, 8000, 32768, True])
def test_unregistered_budget_fails_closed(budget: object) -> None:
    turns = _compile([_record("q-budget-bad", ["evidence"])]).turns

    with pytest.raises(TurnPromptPackingError, match="token_budget must be one of"):
        _pack(turns, budget=budget)  # type: ignore[arg-type]


def test_duplicate_cross_question_and_unfrozen_inputs_fail_closed() -> None:
    corpus = _compile(
        [
            _record("q-one", ["first", "second"]),
            _record("q-two", ["third"]),
        ]
    )
    q_one = tuple(turn for turn in corpus.turns if turn.turn_id.question_id == "q-one")
    q_two = tuple(turn for turn in corpus.turns if turn.turn_id.question_id == "q-two")

    with pytest.raises(TurnPromptPackingError, match="repeat a turn ID"):
        _pack((q_one[0], q_one[0]))
    with pytest.raises(TurnPromptPackingError, match="requested question"):
        _pack((q_one[0], q_two[0]))
    with pytest.raises(TurnPromptPackingError, match="frozen in a tuple"):
        OrderedTurnBlocks.linear(list(q_one))  # type: ignore[arg-type]
    with pytest.raises(TurnPromptPackingError, match="every prompt block"):
        OrderedTurnBlocks.chain_blocks((list(q_one),))  # type: ignore[arg-type]


def test_question_date_validation_is_exact_and_weekday_checked() -> None:
    turn = _compile([_record("q-date", ["evidence"])]).turns[0]

    with pytest.raises(TurnPromptPackingError, match="weekday|canonically"):
        pack_turn_prompt(
            OrderedTurnBlocks.linear((turn,)),
            question_id="q-date",
            question=QUESTION,
            current_date="2025/01/04 (Sun) 11:00",
            tokenizer=FixtureExactReceiptIssuer(),
        )
    with pytest.raises(TurnPromptPackingError, match="exactly"):
        pack_turn_prompt(
            OrderedTurnBlocks.linear((turn,)),
            question_id="q-date",
            question=QUESTION,
            current_date=" 2025/01/04 (Sat) 11:00",
            tokenizer=FixtureExactReceiptIssuer(),
        )


def test_gold_only_source_changes_do_not_change_prompt_or_packing_decisions() -> None:
    first = _compile([_record("q-gold", ["same evidence"], answer="FIRST-GOLD", has_answer=False)])
    second = _compile([_record("q-gold", ["same evidence"], answer="SECOND-GOLD", has_answer=True)])
    assert first.turns[0].source_record != second.turns[0].source_record
    assert first.turns[0].source_turn != second.turns[0].source_turn

    first_result = _pack(first.turns)
    second_result = _pack(second.turns)

    assert first_result.prompt == second_result.prompt
    assert first_result.trace["kept_ids"] == second_result.trace["kept_ids"]
    assert first_result.trace["dropped_ids"] == second_result.trace["dropped_ids"]
    assert first_result.trace["decisions"] == second_result.trace["decisions"]
    assert first_result.trace["final_prompt"] == second_result.trace["final_prompt"]
    assert "FIRST-GOLD" not in first_result.prompt
    assert "SECOND-GOLD" not in second_result.prompt


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (
            lambda receipt, _request, _prompt: replace(
                receipt,
                tokenizer_identity_sha256="f" * 64,
            ),
            "identity",
        ),
        (
            lambda receipt, _request, _prompt: replace(
                receipt,
                query_sha256="f" * 64,
            ),
            "query digest",
        ),
        (
            lambda receipt, _request, _prompt: replace(
                receipt,
                prompt_sha256="f" * 64,
            ),
            "prompt digest",
        ),
        (
            lambda receipt, _request, _prompt: replace(
                receipt,
                prompt_utf8_bytes=receipt.prompt_utf8_bytes + 1,
            ),
            "byte count",
        ),
    ],
)
def test_mismatched_receipt_fields_fail_closed(
    transform: ReceiptTransform,
    message: str,
) -> None:
    turns = _compile([_record("q-receipt", ["evidence"])]).turns

    with pytest.raises(TurnPromptPackingError, match=message):
        _pack(turns, FixtureExactReceiptIssuer(transform=transform))


def test_identity_drift_nonmonotone_requests_and_provider_id_reuse_fail_closed() -> None:
    turns = _compile([_record("q-drift", ["first", "second"])]).turns

    with pytest.raises(TurnPromptPackingError, match="identity drifted"):
        _pack(turns, FixtureExactReceiptIssuer(identity_drift_after=1))

    def repeat_request_id(
        receipt: ExactTokenCountReceipt,
        request: int,
        _prompt: str,
    ) -> ExactTokenCountReceipt:
        return replace(receipt, request_id=1) if request == 2 else receipt

    with pytest.raises(TurnPromptPackingError, match="not strictly monotone"):
        _pack(turns, FixtureExactReceiptIssuer(transform=repeat_request_id))

    def reuse_provider_id(
        receipt: ExactTokenCountReceipt,
        request: int,
        _prompt: str,
    ) -> ExactTokenCountReceipt:
        return replace(receipt, provider_request_id="fixture-1") if request == 2 else receipt

    with pytest.raises(TurnPromptPackingError, match="was reused"):
        _pack(turns, FixtureExactReceiptIssuer(transform=reuse_provider_id))


def test_repeated_prompt_count_disagreement_fails_closed() -> None:
    turns = _compile([_record("q-counts", ["first", "second"])]).turns

    def disagree_on_first_duplicate(prompt: str, request: int) -> int:
        base = len(prompt.encode("utf-8"))
        return base + 1 if request == 3 else base

    with pytest.raises(TurnPromptPackingError, match="disagree on token count"):
        _pack(turns, FixtureExactReceiptIssuer(count=disagree_on_first_duplicate))


def test_lower_exact_count_after_concatenation_is_accepted_by_budget() -> None:
    turns = _compile([_record("q-retokenize", ["RETOKENIZE-FIRST", "RETOKENIZE-SECOND"])]).turns

    def boundary_retokenizing_count(prompt: str, _request: int) -> int:
        has_first = "RETOKENIZE-FIRST" in prompt
        has_second = "RETOKENIZE-SECOND" in prompt
        if has_first and has_second:
            return 2500
        if has_first:
            return 3000
        if has_second:
            return 1000
        return 200

    result = _pack(
        turns,
        FixtureExactReceiptIssuer(count=boundary_retokenizing_count),
        budget=4096,
    )

    proposal_counts = [
        observation["receipt"]["token_count"]
        for observation in result.trace["exact_count_observations"]
        if observation["purpose"] == "greedy-proposal"
    ]
    assert proposal_counts == [3000, 2500]
    assert result.trace["kept_ids"] == [["q-retokenize", 0, 0], ["q-retokenize", 0, 1]]
    assert result.trace["dropped_ids"] == []
    assert result.trace["final_prompt"]["tokens"] == 2500


def test_fixed_prompt_over_budget_fails_instead_of_emitting_invalid_result() -> None:
    turns = _compile([_record("q-base-budget", ["evidence"])]).turns

    with pytest.raises(TurnPromptPackingError, match="without turns already exceeds"):
        _pack(
            turns,
            FixtureExactReceiptIssuer(count=lambda _prompt, _request: 5000),
            budget=4096,
        )
