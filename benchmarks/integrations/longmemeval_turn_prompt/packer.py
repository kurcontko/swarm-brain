"""Exact greedy whole-turn packing for frozen LongMemEval E1/E2 orders.

This module renders prompts and validates externally produced exact-token
receipts.  It does not load a tokenizer, execute a model, access the network,
or mutate the production retrieval/runtime path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from scripts._longmemeval_common import EMPTY_CONTEXT_NOTE, OFFICIAL_ANSWER_TEMPLATE

from benchmarks.integrations.longmemeval_turns import LongMemEvalTurnId

from .contracts import (
    ARTIFACT_TYPE,
    CHAIN_BLOCK_SEPARATOR,
    CHAIN_HEADER_BODY_SEPARATOR,
    CHAIN_HEADER_TEMPLATE,
    CHAIN_TURN_SEPARATOR,
    EMPTY_CONTEXT_NOTE_SHA256,
    HISTORY_SERIALIZER_VERSION,
    LINEAR_TURN_SEPARATOR,
    OFFICIAL_ANSWER_TEMPLATE_SHA256,
    PRIMARY_TOKEN_BUDGET,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    SUPPORTED_TOKEN_BUDGETS,
    ExactCountObservation,
    ExactPromptTokenizer,
    ExactTokenCountReceipt,
    OrderedTurnBlocks,
    PackingDecision,
    PromptLayout,
    TokenizerIdentity,
    TurnPromptPackingError,
    TurnPromptPackingResult,
    checked_identifier,
    sha256_bytes,
    sha256_json,
    sha256_text,
)

_CURRENT_DATE_RE = re.compile(
    r"(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2}) "
    r"\((?P<weekday>Mon|Tue|Wed|Thu|Fri|Sat|Sun)\) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2})"
)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _validate_frozen_template_sources() -> None:
    if sha256_text(OFFICIAL_ANSWER_TEMPLATE) != OFFICIAL_ANSWER_TEMPLATE_SHA256:
        raise TurnPromptPackingError(
            "the shared official LongMemEval answer template changed; register a new protocol"
        )
    if sha256_text(EMPTY_CONTEXT_NOTE) != EMPTY_CONTEXT_NOTE_SHA256:
        raise TurnPromptPackingError(
            "the shared empty-context note changed; register a new prompt protocol"
        )


def _checked_current_date(value: Any) -> str:
    if not isinstance(value, str):
        raise TurnPromptPackingError("current_date must be text")
    match = _CURRENT_DATE_RE.fullmatch(value)
    if match is None:
        raise TurnPromptPackingError("current_date must use YYYY/MM/DD (Ddd) HH:MM exactly")
    try:
        parsed = datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            tzinfo=UTC,
        )
    except ValueError as exc:
        raise TurnPromptPackingError("current_date is not a valid calendar time") from exc
    weekday = _WEEKDAYS[parsed.weekday()]
    canonical = (
        f"{parsed.year:04d}/{parsed.month:02d}/{parsed.day:02d} "
        f"({weekday}) {parsed.hour:02d}:{parsed.minute:02d}"
    )
    if value != canonical:
        raise TurnPromptPackingError("current_date is not canonically serialized")
    return value


def _checked_question(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TurnPromptPackingError("question must be non-empty text")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise TurnPromptPackingError("question must be valid UTF-8 text") from exc
    return value


def _turn_id_payload(turn_id: LongMemEvalTurnId) -> list[str | int]:
    return list(turn_id.as_tuple())


def _validate_candidates(
    ordered: OrderedTurnBlocks,
    *,
    question_id: str,
) -> None:
    seen: set[LongMemEvalTurnId] = set()
    source_record = None
    for turn in ordered.flattened:
        if turn.turn_id.question_id != question_id:
            raise TurnPromptPackingError("every prompt turn must belong to the requested question")
        if turn.turn_id in seen:
            raise TurnPromptPackingError("prompt candidates cannot repeat a turn ID")
        seen.add(turn.turn_id)
        if source_record is None:
            source_record = turn.source_record
        elif turn.source_record != source_record:
            raise TurnPromptPackingError(
                "prompt candidates must share one immutable source-record binding"
            )


def _selected_history(
    ordered: OrderedTurnBlocks,
    selected_ids: frozenset[LongMemEvalTurnId],
) -> str:
    if not selected_ids:
        return EMPTY_CONTEXT_NOTE
    if ordered.layout is PromptLayout.LINEAR:
        return LINEAR_TURN_SEPARATOR.join(
            turn.serialized_text for turn in ordered.blocks[0] if turn.turn_id in selected_ids
        )

    rendered_blocks: list[str] = []
    for block_position, block in enumerate(ordered.blocks, start=1):
        selected = [turn.serialized_text for turn in block if turn.turn_id in selected_ids]
        if not selected:
            continue
        header = CHAIN_HEADER_TEMPLATE.format(chain_number=block_position)
        body = CHAIN_TURN_SEPARATOR.join(selected)
        rendered_blocks.append(f"{header}{CHAIN_HEADER_BODY_SEPARATOR}{body}")
    if not rendered_blocks:
        return EMPTY_CONTEXT_NOTE
    return CHAIN_BLOCK_SEPARATOR.join(rendered_blocks)


def _reader_prompt(
    ordered: OrderedTurnBlocks,
    selected_ids: tuple[LongMemEvalTurnId, ...],
    *,
    question: str,
    current_date: str,
) -> str:
    history = _selected_history(ordered, frozenset(selected_ids))
    return OFFICIAL_ANSWER_TEMPLATE.format(history, current_date, question)


@dataclass(slots=True)
class _ReceiptVerifier:
    tokenizer: ExactPromptTokenizer
    expected_identity: TokenizerIdentity
    query_sha256: str
    observations: list[ExactCountObservation]
    provider_request_ids: set[str]
    counts_by_prompt_sha256: dict[str, int]
    last_request_id: int | None = None

    @classmethod
    def create(
        cls,
        tokenizer: ExactPromptTokenizer,
        *,
        query_sha256: str,
    ) -> _ReceiptVerifier:
        try:
            identity = tokenizer.identity
        except Exception as exc:  # pragma: no cover - defensive boundary normalization
            raise TurnPromptPackingError("could not read tokenizer identity") from exc
        if not isinstance(identity, TokenizerIdentity):
            raise TurnPromptPackingError("exact prompt tokenizer must expose a TokenizerIdentity")
        return cls(
            tokenizer=tokenizer,
            expected_identity=identity,
            query_sha256=query_sha256,
            observations=[],
            provider_request_ids=set(),
            counts_by_prompt_sha256={},
        )

    def _assert_boundary_identity(self) -> None:
        try:
            current = self.tokenizer.identity
        except Exception as exc:  # pragma: no cover - defensive boundary normalization
            raise TurnPromptPackingError("could not re-read tokenizer identity") from exc
        if current != self.expected_identity:
            raise TurnPromptPackingError("tokenizer identity drifted during prompt packing")

    def observe(
        self,
        prompt: str,
        *,
        purpose: str,
        candidate_turn_id: LongMemEvalTurnId | None,
    ) -> ExactCountObservation:
        self._assert_boundary_identity()
        try:
            receipt = self.tokenizer.count_prompt(
                prompt,
                query_sha256=self.query_sha256,
            )
        except TurnPromptPackingError:
            raise
        except Exception as exc:
            raise TurnPromptPackingError("exact prompt tokenizer boundary failed") from exc
        self._assert_boundary_identity()
        if not isinstance(receipt, ExactTokenCountReceipt):
            raise TurnPromptPackingError("exact prompt tokenizer returned an invalid receipt type")

        encoded = prompt.encode("utf-8")
        prompt_sha256 = sha256_bytes(encoded)
        if receipt.tokenizer_identity_sha256 != self.expected_identity.identity_sha256:
            raise TurnPromptPackingError("tokenizer receipt identity does not match the boundary")
        if receipt.query_sha256 != self.query_sha256:
            raise TurnPromptPackingError("tokenizer receipt query digest does not match")
        if receipt.prompt_sha256 != prompt_sha256:
            raise TurnPromptPackingError("tokenizer receipt prompt digest does not match")
        if receipt.prompt_utf8_bytes != len(encoded):
            raise TurnPromptPackingError("tokenizer receipt prompt byte count does not match")
        if self.last_request_id is not None and receipt.request_id <= self.last_request_id:
            raise TurnPromptPackingError("tokenizer receipt request IDs are not strictly monotone")
        if receipt.provider_request_id in self.provider_request_ids:
            raise TurnPromptPackingError("tokenizer provider request ID was reused")

        prior_count = self.counts_by_prompt_sha256.get(prompt_sha256)
        if prior_count is not None and prior_count != receipt.token_count:
            raise TurnPromptPackingError(
                "repeated exact prompt observations disagree on token count"
            )
        self.counts_by_prompt_sha256[prompt_sha256] = receipt.token_count
        self.last_request_id = receipt.request_id
        self.provider_request_ids.add(receipt.provider_request_id)
        observation = ExactCountObservation(
            sequence=len(self.observations) + 1,
            purpose=purpose,
            candidate_turn_id=candidate_turn_id,
            receipt=receipt,
        )
        self.observations.append(observation)
        return observation


def _separator_binding(value: str) -> dict[str, Any]:
    encoded = value.encode("utf-8")
    return {"literal": value, "utf8_bytes": len(encoded), "sha256": sha256_bytes(encoded)}


def _layout_binding(ordered: OrderedTurnBlocks) -> dict[str, Any]:
    headers = []
    if ordered.layout is PromptLayout.CHAIN_BLOCKS:
        headers = [
            {
                "block_position": block_position,
                **_separator_binding(CHAIN_HEADER_TEMPLATE.format(chain_number=block_position)),
            }
            for block_position in range(1, len(ordered.blocks) + 1)
        ]
    return {
        "layout": ordered.layout.value,
        "history_serializer_version": HISTORY_SERIALIZER_VERSION,
        "linear_turn_separator": _separator_binding(LINEAR_TURN_SEPARATOR),
        "chain_header_template": CHAIN_HEADER_TEMPLATE,
        "chain_headers": headers,
        "chain_header_body_separator": _separator_binding(CHAIN_HEADER_BODY_SEPARATOR),
        "chain_turn_separator": _separator_binding(CHAIN_TURN_SEPARATOR),
        "chain_block_separator": _separator_binding(CHAIN_BLOCK_SEPARATOR),
        "empty_context_note": {
            "utf8_bytes": len(EMPTY_CONTEXT_NOTE.encode("utf-8")),
            "sha256": EMPTY_CONTEXT_NOTE_SHA256,
        },
    }


def _candidate_order_binding(ordered: OrderedTurnBlocks) -> list[dict[str, Any]]:
    return [
        {
            "candidate_position": candidate_position,
            "block_position": block_position,
            "position_in_block": position_in_block,
            "turn": turn.content_free_binding(),
        }
        for candidate_position, (block_position, position_in_block, turn) in enumerate(
            ordered.positions(),
            start=1,
        )
    ]


def _ids_by_block(
    ordered: OrderedTurnBlocks,
    selected_ids: set[LongMemEvalTurnId],
) -> list[list[list[str | int]]]:
    return [
        [_turn_id_payload(turn.turn_id) for turn in block if turn.turn_id in selected_ids]
        for block in ordered.blocks
    ]


def pack_turn_prompt(
    ordered: OrderedTurnBlocks,
    *,
    question_id: str,
    question: str,
    current_date: str,
    token_budget: int = PRIMARY_TOKEN_BUDGET,
    tokenizer: ExactPromptTokenizer,
) -> TurnPromptPackingResult:
    """Greedily pack indivisible turns and return one exact official reader prompt.

    Candidate and block order are caller-frozen.  A turn whose complete proposed
    prompt exceeds the budget is skipped and the scan continues.  A separate
    singleton observation proves whether a dropped turn was oversized even in
    an otherwise empty prompt.  The final prompt is independently recounted.
    """

    _validate_frozen_template_sources()
    if not isinstance(ordered, OrderedTurnBlocks):
        raise TurnPromptPackingError("ordered candidates must be OrderedTurnBlocks")
    question_id = checked_identifier(question_id, label="question_id")
    question = _checked_question(question)
    current_date = _checked_current_date(current_date)
    if type(token_budget) is not int or token_budget not in SUPPORTED_TOKEN_BUDGETS:
        raise TurnPromptPackingError(f"token_budget must be one of {SUPPORTED_TOKEN_BUDGETS}")
    _validate_candidates(ordered, question_id=question_id)

    query_sha256 = sha256_text(question)
    verifier = _ReceiptVerifier.create(tokenizer, query_sha256=query_sha256)
    initial_prompt = _reader_prompt(
        ordered,
        (),
        question=question,
        current_date=current_date,
    )
    initial = verifier.observe(
        initial_prompt,
        purpose="initial-empty-context",
        candidate_turn_id=None,
    )
    if initial.receipt.token_count > token_budget:
        raise TurnPromptPackingError(
            "the fixed official prompt without turns already exceeds the token budget"
        )

    selected: list[LongMemEvalTurnId] = []
    current = initial
    decisions: list[PackingDecision] = []
    for block_position, position_in_block, turn in ordered.positions():
        candidate_id = turn.turn_id
        singleton_prompt = _reader_prompt(
            ordered,
            (candidate_id,),
            question=question,
            current_date=current_date,
        )
        singleton = verifier.observe(
            singleton_prompt,
            purpose="candidate-alone",
            candidate_turn_id=candidate_id,
        )

        proposed_ids = (*selected, candidate_id)
        proposed_prompt = _reader_prompt(
            ordered,
            proposed_ids,
            question=question,
            current_date=current_date,
        )
        proposal = verifier.observe(
            proposed_prompt,
            purpose="greedy-proposal",
            candidate_turn_id=candidate_id,
        )

        accepted = proposal.receipt.token_count <= token_budget
        decisions.append(
            PackingDecision(
                candidate_turn_id=candidate_id,
                block_position=block_position,
                position_in_block=position_in_block,
                selected_before_ids=tuple(selected),
                proposed_ids=proposed_ids,
                singleton_observation_sequence=singleton.sequence,
                proposal_observation_sequence=proposal.sequence,
                accepted=accepted,
                oversized_alone=singleton.receipt.token_count > token_budget,
            )
        )
        if accepted:
            selected.append(candidate_id)
            current = proposal

    final_prompt = _reader_prompt(
        ordered,
        tuple(selected),
        question=question,
        current_date=current_date,
    )
    final = verifier.observe(
        final_prompt,
        purpose="final-independent-recount",
        candidate_turn_id=None,
    )
    if final.receipt.token_count != current.receipt.token_count:
        raise TurnPromptPackingError("final prompt token count differs from the accepted state")
    if final.receipt.token_count > token_budget:
        raise TurnPromptPackingError("final prompt exceeds its declared token budget")

    selected_set = set(selected)
    dropped = [decision.candidate_turn_id for decision in decisions if not decision.accepted]
    oversized = [
        decision.candidate_turn_id
        for decision in decisions
        if not decision.accepted and decision.oversized_alone
    ]
    candidate_order = _candidate_order_binding(ordered)
    decision_bindings = [decision.content_free_binding() for decision in decisions]
    observations = [item.content_free_binding() for item in verifier.observations]
    final_encoded = final_prompt.encode("utf-8")
    final_history = _selected_history(ordered, frozenset(selected))
    final_history_encoded = final_history.encode("utf-8")
    trace = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "classification": "evaluation-only-exact-turn-prompt-packing",
        "production_configuration": False,
        "packing_policy": {
            "method": "exact-complete-prompt-greedy-skip-and-continue",
            "whole_turns_indivisible": True,
            "candidate_order_mutated": False,
            "oversized_definition": "candidate-alone-complete-prompt-tokens>budget",
            "final_independent_recount": True,
        },
        "budget": {
            "token_budget": token_budget,
            "supported_token_budgets": list(SUPPORTED_TOKEN_BUDGETS),
            "primary_token_budget": PRIMARY_TOKEN_BUDGET,
            "is_primary": token_budget == PRIMARY_TOKEN_BUDGET,
            "counted_surface": "complete-official-reader-prompt",
        },
        "reader_prompt": {
            "template_source": "scripts._longmemeval_common.OFFICIAL_ANSWER_TEMPLATE",
            "template_sha256": OFFICIAL_ANSWER_TEMPLATE_SHA256,
            "template_utf8_bytes": len(OFFICIAL_ANSWER_TEMPLATE.encode("utf-8")),
            "history_placeholder": "frozen-turn-history",
        },
        "question_input": {
            "question_id": question_id,
            "query_sha256": query_sha256,
            "query_utf8_bytes": len(question.encode("utf-8")),
            "current_date_sha256": sha256_text(current_date),
            "current_date_utf8_bytes": len(current_date.encode("utf-8")),
            "combined_input_sha256": sha256_json(
                {"current_date": current_date, "question": question}
            ),
        },
        "tokenizer": verifier.expected_identity.as_dict(),
        "layout": _layout_binding(ordered),
        "candidate_order": candidate_order,
        "candidate_order_sha256": sha256_json(candidate_order),
        "candidate_blocks": [
            [_turn_id_payload(turn.turn_id) for turn in block] for block in ordered.blocks
        ],
        "kept_ids": [_turn_id_payload(turn_id) for turn_id in selected],
        "kept_ids_by_block": _ids_by_block(ordered, selected_set),
        "dropped_ids": [_turn_id_payload(turn_id) for turn_id in dropped],
        "oversized_ids": [_turn_id_payload(turn_id) for turn_id in oversized],
        "decisions": decision_bindings,
        "decisions_sha256": sha256_json(decision_bindings),
        "exact_count_observations": observations,
        "exact_count_observations_sha256": sha256_json(observations),
        "observation_accounting": {
            "requests": len(observations),
            "responses": len(observations),
            "unique_provider_request_ids": len(verifier.provider_request_ids),
            "unique_prompt_digests": len(verifier.counts_by_prompt_sha256),
            "request_ids_strictly_monotone": True,
            "repeated_prompt_counts_reconciled": True,
        },
        "final_prompt": {
            "sha256": sha256_bytes(final_encoded),
            "utf8_bytes": len(final_encoded),
            "tokens": final.receipt.token_count,
            "final_observation_sequence": final.sequence,
            "within_budget": final.receipt.token_count <= token_budget,
        },
        "final_history": {
            "sha256": sha256_bytes(final_history_encoded),
            "utf8_bytes": len(final_history_encoded),
        },
        "claims": {
            "token_counts_are_external_receipts": True,
            "token_counts_are_estimates": False,
            "external_tokenizer_boundary_invoked": True,
            "packer_loads_tokenizer_or_calls_reader_model": False,
            "tokenizer_artifact_reopened_by_packer": False,
            "gold_fields_consumed": False,
            "qa_improvement": False,
        },
    }
    return TurnPromptPackingResult(prompt=final_prompt, trace=trace)


__all__ = ["pack_turn_prompt"]
