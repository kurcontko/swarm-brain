"""Frozen contracts for source-bound LongMemEval query-time construction.

The package deliberately stops at an offline evidence boundary.  It builds
the exact LazyMem-shaped local windows and replays raw OpenAI-compatible
provider responses into KEEP/DROP decisions, but it never invokes a
constructor, tokenizer, reader, judge, database, or network service.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from benchmarks.integrations.longmemeval_turns import LongMemEvalTurnId, TurnProjection

ARTIFACT_TYPE = "swarmbrain-longmemeval-query-construction"
SCHEMA_VERSION = 2
PROTOCOL_VERSION = "swarmbrain-longmemeval-query-construction-e7-v2"

PAPER_URL = "https://arxiv.org/abs/2607.22690v2"
PAPER_REPOSITORY = "https://github.com/allacnobug/LazyMem"
PAPER_REPOSITORY_COMMIT = "af4109960aacb90d6dba994e9103a36a165cc380"
PAPER_WINDOW_PROTOCOL = "lazymem-v2-top50-radius2-window8-stride7"

RETRIEVED_TURNS = 50
CONTEXT_RADIUS = 2
MAX_WINDOW_MESSAGES = 8
WINDOW_STRIDE = 7
EXTRACTIVE_SEPARATOR = " … "
MAX_SUPPORT_SPANS_PER_KEEP = 32
MAX_COMPRESSED_UTF8_BYTES = 65_536
MAX_REASON_UTF8_BYTES = 4_096
MAX_RAW_PROVIDER_RESPONSE_BYTES = 1_048_576
EMPTY_MEMORY_CONTEXT = "(No kept memory was found.)"
READER_CONTEXT_SERIALIZER = "swarmbrain-e7-chronological-compressed-memory-v1"
PROVIDER_RESPONSE_PARSER = "openai-compatible-chat-completions-e7-v1"

CONSTRUCTOR_INPUT_FIELDS = (
    "query.text",
    "query.current_date",
    "window.messages.timestamp",
    "window.messages.role",
    "window.messages.content",
)
RECEIPT_AUTHENTICATION = "externally-attested-unsigned"
RECEIPT_REPLAY = "strict-raw-provider-response-and-usage-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class QueryConstructionError(ValueError):
    """Evidence cannot support the frozen query-time construction protocol."""


class ConstructionCell(StrEnum):
    """Frozen E7 ablation cells.

    E7-A is the broad raw-turn control.  E7-B accepts query-conditioned
    abstractive construction and therefore carries only caller-attested source
    support.  E7-C permits only byte-verifiable verbatim or extractive output.
    """

    RAW_TOP50 = "E7-A"
    QUERY_CONSTRUCTION = "E7-B"
    GROUNDED_QUERY_CONSTRUCTION = "E7-C"


class WindowPosition(StrEnum):
    PREVIOUS_BRIDGE = "previous_bridge"
    CORE = "core"
    NEXT_BRIDGE = "next_bridge"


class DecisionOperation(StrEnum):
    KEEP = "KEEP"
    DROP = "DROP"


class RetentionStyle(StrEnum):
    DROP = "drop"
    VERBATIM = "verbatim"
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise QueryConstructionError(f"value is not finite canonical UTF-8 JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_utf8(value: str) -> str:
    if not isinstance(value, str):
        raise QueryConstructionError("digest input must be text")
    try:
        return sha256_bytes(value.encode("utf-8"))
    except UnicodeError as exc:
        raise QueryConstructionError("digest input must be valid UTF-8") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise QueryConstructionError("provider response contains a duplicate JSON key")
        output[key] = value
    return output


def _reject_nonstandard_constant(value: str) -> None:
    del value
    raise QueryConstructionError("provider response contains a non-standard JSON constant")


def _strict_json(value: str, *, label: str) -> Any:
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (json.JSONDecodeError, QueryConstructionError, ValueError):
        raise QueryConstructionError(f"{label} is malformed JSON") from None


def _opaque_text_binding(value: str) -> dict[str, int | str]:
    """Bind caller-controlled text without copying it into a public trace."""

    encoded = value.encode("utf-8")
    return {"utf8_bytes": len(encoded), "sha256": sha256_bytes(encoded)}


def required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise QueryConstructionError(
            f"{label} must be non-empty text without surrounding whitespace"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise QueryConstructionError(f"{label} cannot contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise QueryConstructionError(f"{label} must be valid UTF-8") from exc
    return value


def checked_sha256(value: Any, *, label: str) -> str:
    text = required_text(value, label=label)
    if _SHA256_RE.fullmatch(text) is None:
        raise QueryConstructionError(f"{label} must be a lowercase hexadecimal SHA-256 digest")
    return text


def nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QueryConstructionError(f"{label} must be a non-negative integer")
    return value


def positive_int(value: Any, *, label: str) -> int:
    result = nonnegative_int(value, label=label)
    if result == 0:
        raise QueryConstructionError(f"{label} must be positive")
    return result


def nonnegative_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QueryConstructionError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise QueryConstructionError(f"{label} must be a finite non-negative number")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True, slots=True)
class RetrievedTurnPool:
    """Exact E1-B top-50 output used by the frozen E7 cells."""

    question_id: str
    query_sha256: str
    source_artifact_sha256: str
    projection_sha256: str
    selection_trace_sha256: str
    turns: tuple[TurnProjection, ...]

    def __post_init__(self) -> None:
        required_text(self.question_id, label="retrieved pool question_id")
        checked_sha256(self.query_sha256, label="retrieved pool query_sha256")
        checked_sha256(
            self.source_artifact_sha256,
            label="retrieved pool source_artifact_sha256",
        )
        checked_sha256(self.projection_sha256, label="retrieved pool projection_sha256")
        checked_sha256(
            self.selection_trace_sha256,
            label="retrieved pool selection_trace_sha256",
        )
        if not isinstance(self.turns, tuple):
            raise QueryConstructionError("retrieved pool turns must be frozen in a tuple")
        if len(self.turns) != RETRIEVED_TURNS:
            raise QueryConstructionError(
                f"E7 requires exactly the E1-B top {RETRIEVED_TURNS} turns"
            )
        seen: set[LongMemEvalTurnId] = set()
        for turn in self.turns:
            if not isinstance(turn, TurnProjection):
                raise QueryConstructionError("retrieved pool contains a non-turn value")
            if turn.turn_id.question_id != self.question_id:
                raise QueryConstructionError("retrieved pool contains a cross-question turn")
            if turn.turn_id in seen:
                raise QueryConstructionError("retrieved pool repeats a turn")
            seen.add(turn.turn_id)

    def content_free_binding(self) -> dict[str, Any]:
        ordered = [
            {
                "rank": rank,
                "turn_id": list(turn.turn_id.as_tuple()),
                "serialized_document_utf8": turn.serialized_document_utf8.as_dict(),
            }
            for rank, turn in enumerate(self.turns, start=1)
        ]
        return {
            "question_id": self.question_id,
            "query_sha256": self.query_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "projection_sha256": self.projection_sha256,
            "selection_cell": "E1-B",
            "selection_trace_sha256": self.selection_trace_sha256,
            "retrieved_turns": len(self.turns),
            "ordered_turns_sha256": sha256_json(ordered),
        }


@dataclass(frozen=True, slots=True)
class WindowMessage:
    turn: TurnProjection
    global_message_index: int
    retrieval_rank: int | None
    is_retrieved: bool
    position: WindowPosition

    def __post_init__(self) -> None:
        if not isinstance(self.turn, TurnProjection):
            raise QueryConstructionError("window message must contain a TurnProjection")
        nonnegative_int(self.global_message_index, label="global message index")
        if self.retrieval_rank is not None:
            rank = positive_int(self.retrieval_rank, label="retrieval rank")
            if rank > RETRIEVED_TURNS:
                raise QueryConstructionError("retrieval rank exceeds the frozen top-50 pool")
        if not isinstance(self.is_retrieved, bool):
            raise QueryConstructionError("window is_retrieved must be Boolean")
        if self.is_retrieved != (self.retrieval_rank is not None):
            raise QueryConstructionError("retrieval rank and is_retrieved disagree")
        if not isinstance(self.position, WindowPosition):
            raise QueryConstructionError("window position is not registered")

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "turn_id": list(self.turn.turn_id.as_tuple()),
            "global_message_index": self.global_message_index,
            "retrieval_rank": self.retrieval_rank,
            "is_retrieved": self.is_retrieved,
            "position": self.position.value,
            "source_turn": self.turn.source_turn.as_dict(),
            "original_content_utf8": self.turn.original_content_utf8.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class QueryWindow:
    question_id: str
    query_sha256: str
    current_date_sha256: str
    segment_index: int
    window_index: int
    segment_type: str
    messages: tuple[WindowMessage, ...]

    def __post_init__(self) -> None:
        required_text(self.question_id, label="window question_id")
        checked_sha256(self.query_sha256, label="window query_sha256")
        checked_sha256(self.current_date_sha256, label="window current_date_sha256")
        nonnegative_int(self.segment_index, label="window segment_index")
        nonnegative_int(self.window_index, label="window window_index")
        if self.segment_type not in {"singleton", "continuous", "continuous_sliced"}:
            raise QueryConstructionError("window segment_type is not registered")
        if not isinstance(self.messages, tuple) or not self.messages:
            raise QueryConstructionError("window messages must be a non-empty tuple")
        if len(self.messages) > MAX_WINDOW_MESSAGES:
            raise QueryConstructionError("window exceeds the frozen eight-message cap")
        if not any(message.position is WindowPosition.CORE for message in self.messages):
            raise QueryConstructionError("window must contain at least one core message")
        question_ids = {message.turn.turn_id.question_id for message in self.messages}
        if question_ids != {self.question_id}:
            raise QueryConstructionError("window contains a cross-question message")
        indexes = [message.global_message_index for message in self.messages]
        if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
            raise QueryConstructionError("window messages must be unique and chronological")

    @property
    def window_sha256(self) -> str:
        return sha256_json(self.content_free_binding())

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "query_sha256": self.query_sha256,
            "current_date_sha256": self.current_date_sha256,
            "segment_index": self.segment_index,
            "window_index": self.window_index,
            "segment_type": self.segment_type,
            "messages": [message.content_free_binding() for message in self.messages],
        }


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Half-open UTF-8 byte span into one immutable source message."""

    turn_id: LongMemEvalTurnId
    start_byte: int
    end_byte: int

    def __post_init__(self) -> None:
        if not isinstance(self.turn_id, LongMemEvalTurnId):
            raise QueryConstructionError("source span must use a LongMemEvalTurnId")
        start = nonnegative_int(self.start_byte, label="source span start_byte")
        end = positive_int(self.end_byte, label="source span end_byte")
        if end <= start:
            raise QueryConstructionError("source span end_byte must be greater than start_byte")

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_id": list(self.turn_id.as_tuple()),
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
        }


@dataclass(frozen=True, slots=True)
class ConstructionDecision:
    turn_id: LongMemEvalTurnId
    operation: DecisionOperation
    style: RetentionStyle
    compressed_content: str
    reason: str
    support_spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.turn_id, LongMemEvalTurnId):
            raise QueryConstructionError("construction decision must use a LongMemEvalTurnId")
        if not isinstance(self.operation, DecisionOperation):
            raise QueryConstructionError("construction decision operation is not registered")
        if not isinstance(self.style, RetentionStyle):
            raise QueryConstructionError("construction decision style is not registered")
        if not isinstance(self.compressed_content, str):
            raise QueryConstructionError("compressed_content must be text")
        try:
            self.compressed_content.encode("utf-8")
        except UnicodeError as exc:
            raise QueryConstructionError("compressed_content must be valid UTF-8") from exc
        required_text(self.reason, label="construction decision reason")
        if len(self.reason.encode("utf-8")) > MAX_REASON_UTF8_BYTES:
            raise QueryConstructionError("construction decision reason exceeds its byte cap")
        if not isinstance(self.support_spans, tuple):
            raise QueryConstructionError("support_spans must be frozen in a tuple")
        if any(not isinstance(span, SourceSpan) for span in self.support_spans):
            raise QueryConstructionError("support_spans contain an invalid span")
        if any(span.turn_id != self.turn_id for span in self.support_spans):
            raise QueryConstructionError("support span points to a different source turn")
        if self.operation is DecisionOperation.DROP:
            if self.style is not RetentionStyle.DROP:
                raise QueryConstructionError("DROP must use the drop retention style")
            if self.compressed_content or self.support_spans:
                raise QueryConstructionError("DROP must have empty content and no support spans")
        else:
            if self.style is RetentionStyle.DROP:
                raise QueryConstructionError("KEEP cannot use the drop retention style")
            if (
                not self.compressed_content
                or self.compressed_content != self.compressed_content.strip()
            ):
                raise QueryConstructionError(
                    "KEEP compressed_content must be non-empty without surrounding whitespace"
                )
            if not self.support_spans:
                raise QueryConstructionError("KEEP must cite at least one source byte span")
            if len(self.support_spans) > MAX_SUPPORT_SPANS_PER_KEEP:
                raise QueryConstructionError("KEEP exceeds the source-span cap")
            if len(self.compressed_content.encode("utf-8")) > MAX_COMPRESSED_UTF8_BYTES:
                raise QueryConstructionError("KEEP compressed_content exceeds its byte cap")

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "turn_id": list(self.turn_id.as_tuple()),
            "operation": self.operation.value,
            "style": self.style.value,
            "compressed_content_utf8": {
                "bytes": len(self.compressed_content.encode("utf-8")),
                "sha256": sha256_utf8(self.compressed_content),
            },
            "reason_utf8": {
                "bytes": len(self.reason.encode("utf-8")),
                "sha256": sha256_utf8(self.reason),
            },
            "support_spans": [span.as_dict() for span in self.support_spans],
        }

    def response_payload(self) -> dict[str, Any]:
        """Exact normalized content-bearing constructor response item."""

        return {
            "turn_id": list(self.turn_id.as_tuple()),
            "op": self.operation.value,
            "style": self.style.value,
            "compressed_content": self.compressed_content,
            "reason": self.reason,
            "support_spans": [span.as_dict() for span in self.support_spans],
        }

    def provider_payload(self) -> dict[str, Any]:
        """Exact response item visible to a constructor without benchmark IDs."""

        return {
            "op": self.operation.value,
            "style": self.style.value,
            "compressed_content": self.compressed_content,
            "reason": self.reason,
            "support_spans": [
                {"start_byte": span.start_byte, "end_byte": span.end_byte}
                for span in self.support_spans
            ],
        }


@dataclass(frozen=True, slots=True)
class ConstructorIdentity:
    producer: str
    model: str
    revision: str
    deployment: str
    model_artifact_sha256: str
    system_prompt_sha256: str
    user_prompt_sha256: str
    tokenizer_model: str
    tokenizer_revision: str
    tokenizer_artifact_sha256: str

    def __post_init__(self) -> None:
        required_text(self.producer, label="constructor producer")
        required_text(self.model, label="constructor model")
        required_text(self.revision, label="constructor revision")
        required_text(self.deployment, label="constructor deployment")
        checked_sha256(self.model_artifact_sha256, label="constructor model artifact")
        checked_sha256(self.system_prompt_sha256, label="constructor system prompt")
        checked_sha256(self.user_prompt_sha256, label="constructor user prompt")
        required_text(self.tokenizer_model, label="constructor tokenizer model")
        required_text(self.tokenizer_revision, label="constructor tokenizer revision")
        checked_sha256(self.tokenizer_artifact_sha256, label="constructor tokenizer artifact")

    @property
    def identity_sha256(self) -> str:
        return sha256_json(self.content_free_binding())

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "producer": _opaque_text_binding(self.producer),
            "model": _opaque_text_binding(self.model),
            "revision": _opaque_text_binding(self.revision),
            "deployment": _opaque_text_binding(self.deployment),
            "model_artifact_sha256": self.model_artifact_sha256,
            "system_prompt_sha256": self.system_prompt_sha256,
            "user_prompt_sha256": self.user_prompt_sha256,
            "tokenizer_model": _opaque_text_binding(self.tokenizer_model),
            "tokenizer_revision": _opaque_text_binding(self.tokenizer_revision),
            "tokenizer_artifact_sha256": self.tokenizer_artifact_sha256,
            "identity_source": "caller-attested-unverified",
        }


@dataclass(frozen=True, slots=True)
class ConstructionAccounting:
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float

    def __post_init__(self) -> None:
        nonnegative_int(self.input_tokens, label="constructor input_tokens")
        nonnegative_int(self.output_tokens, label="constructor output_tokens")
        nonnegative_number(self.latency_ms, label="constructor latency_ms")
        nonnegative_number(self.cost_usd, label="constructor cost_usd")

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "token_usage_source": RECEIPT_REPLAY,
            "latency_source": "caller-observed",
            "cost_source": "caller-attested",
        }


@dataclass(frozen=True, slots=True)
class ReplayedConstructorResponse:
    """Strict, content-bearing replay of one raw provider response."""

    response_model: str
    provider_request_id: str
    system_fingerprint: str | None
    finish_reason: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    decisions: tuple[ConstructionDecision, ...]


def _provider_text(value: Any, *, label: str, maximum_bytes: int = 512) -> str:
    text = required_text(value, label=label)
    if len(text.encode("utf-8")) > maximum_bytes:
        raise QueryConstructionError(f"{label} exceeds its byte cap")
    return text


def _usage_integer(usage: dict[str, Any], name: str) -> int:
    return nonnegative_int(usage.get(name), label=f"provider usage.{name}")


def _provider_decision(value: Any, turn_id: LongMemEvalTurnId) -> ConstructionDecision:
    if not isinstance(value, dict) or set(value) != {
        "op",
        "style",
        "compressed_content",
        "reason",
        "support_spans",
    }:
        raise QueryConstructionError("provider decision fields differ from the exact schema")
    try:
        operation = DecisionOperation(value["op"])
        style = RetentionStyle(value["style"])
    except (TypeError, ValueError):
        raise QueryConstructionError(
            "provider decision operation or style is not registered"
        ) from None
    raw_spans = value["support_spans"]
    if not isinstance(raw_spans, list):
        raise QueryConstructionError("provider decision support_spans must be an array")
    spans: list[SourceSpan] = []
    for raw_span in raw_spans:
        if not isinstance(raw_span, dict) or set(raw_span) != {"start_byte", "end_byte"}:
            raise QueryConstructionError(
                "provider support span fields differ from the exact schema"
            )
        spans.append(
            SourceSpan(
                turn_id=turn_id,
                start_byte=raw_span["start_byte"],
                end_byte=raw_span["end_byte"],
            )
        )
    return ConstructionDecision(
        turn_id=turn_id,
        operation=operation,
        style=style,
        compressed_content=value["compressed_content"],
        reason=value["reason"],
        support_spans=tuple(spans),
    )


def replay_constructor_provider_response(
    raw_response: bytes,
    *,
    turn_ids: tuple[LongMemEvalTurnId, ...],
) -> ReplayedConstructorResponse:
    """Replay an exact OpenAI-compatible response without trusting normalized fields."""

    if not isinstance(raw_response, bytes) or not raw_response:
        raise QueryConstructionError("raw provider response must be non-empty bytes")
    if len(raw_response) > MAX_RAW_PROVIDER_RESPONSE_BYTES:
        raise QueryConstructionError("raw provider response exceeds its byte cap")
    if not isinstance(turn_ids, tuple) or not turn_ids:
        raise QueryConstructionError("provider response replay requires frozen turn IDs")
    if any(not isinstance(turn_id, LongMemEvalTurnId) for turn_id in turn_ids):
        raise QueryConstructionError("provider response replay contains an invalid turn ID")
    try:
        response_text = raw_response.decode("utf-8")
    except UnicodeDecodeError:
        raise QueryConstructionError("raw provider response is not valid UTF-8") from None
    payload = _strict_json(response_text, label="raw provider response")
    if not isinstance(payload, dict):
        raise QueryConstructionError("raw provider response must be a JSON object")

    response_model = _provider_text(payload.get("model"), label="provider response model")
    provider_request_id = _provider_text(
        payload.get("id"),
        label="provider request id",
    )
    system_fingerprint = payload.get("system_fingerprint")
    if system_fingerprint is not None:
        system_fingerprint = _provider_text(
            system_fingerprint,
            label="provider system_fingerprint",
        )

    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise QueryConstructionError("provider response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise QueryConstructionError("provider response choice is malformed")
    if isinstance(choice.get("index"), bool) or choice.get("index") != 0:
        raise QueryConstructionError("provider response choice index must be integer zero")
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise QueryConstructionError("provider response did not finish with stop")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise QueryConstructionError("provider response assistant message is malformed")
    if message.get("refusal") is not None or message.get("tool_calls") not in (None, []):
        raise QueryConstructionError("provider response contains a refusal or tool call")
    content = message.get("content")
    if not isinstance(content, str):
        raise QueryConstructionError("provider response content must be text")
    normalized = _strict_json(content, label="provider response content")
    if not isinstance(normalized, dict) or set(normalized) != {"decisions"}:
        raise QueryConstructionError(
            "provider response content fields differ from the exact schema"
        )
    raw_decisions = normalized["decisions"]
    if not isinstance(raw_decisions, list) or len(raw_decisions) != len(turn_ids):
        raise QueryConstructionError(
            "provider response must contain one decision per window message"
        )
    decisions = tuple(
        _provider_decision(value, turn_id)
        for value, turn_id in zip(raw_decisions, turn_ids, strict=True)
    )

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise QueryConstructionError("provider response did not report token usage")
    input_tokens = _usage_integer(usage, "prompt_tokens")
    output_tokens = _usage_integer(usage, "completion_tokens")
    total_tokens = _usage_integer(usage, "total_tokens")
    if total_tokens != input_tokens + output_tokens:
        raise QueryConstructionError("provider response token usage does not reconcile")
    return ReplayedConstructorResponse(
        response_model=response_model,
        provider_request_id=provider_request_id,
        system_fingerprint=system_fingerprint,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        decisions=decisions,
    )


@dataclass(frozen=True, slots=True)
class WindowConstructionReceipt:
    question_id: str
    window_sha256: str
    request_sha256: str
    request_utf8_bytes: int
    response_parser: str
    raw_response: bytes = field(repr=False)
    response_sha256: str
    request_id: str
    provider_request_id: str
    input_fields: tuple[str, ...]
    identity: ConstructorIdentity
    accounting: ConstructionAccounting
    decisions: tuple[ConstructionDecision, ...]

    def __post_init__(self) -> None:
        required_text(self.question_id, label="constructor receipt question_id")
        checked_sha256(self.window_sha256, label="constructor receipt window_sha256")
        checked_sha256(self.request_sha256, label="constructor receipt request_sha256")
        positive_int(self.request_utf8_bytes, label="constructor receipt request_utf8_bytes")
        if self.response_parser != PROVIDER_RESPONSE_PARSER:
            raise QueryConstructionError("constructor receipt response parser drifted")
        checked_sha256(self.response_sha256, label="constructor receipt response_sha256")
        required_text(self.request_id, label="constructor receipt request_id")
        required_text(
            self.provider_request_id,
            label="constructor receipt provider_request_id",
        )
        if self.input_fields != CONSTRUCTOR_INPUT_FIELDS:
            raise QueryConstructionError(
                "constructor receipt must declare the exact source-safe input allowlist"
            )
        if not isinstance(self.identity, ConstructorIdentity):
            raise QueryConstructionError("constructor receipt identity is invalid")
        if not isinstance(self.accounting, ConstructionAccounting):
            raise QueryConstructionError("constructor receipt accounting is invalid")
        if not isinstance(self.decisions, tuple) or not self.decisions:
            raise QueryConstructionError("constructor decisions must be a non-empty tuple")
        if any(not isinstance(item, ConstructionDecision) for item in self.decisions):
            raise QueryConstructionError("constructor receipt contains an invalid decision")
        replayed = replay_constructor_provider_response(
            self.raw_response,
            turn_ids=tuple(item.turn_id for item in self.decisions),
        )
        if replayed.response_model != self.identity.model:
            raise QueryConstructionError(
                "provider response model differs from the pinned constructor model"
            )
        if replayed.provider_request_id != self.provider_request_id:
            raise QueryConstructionError(
                "provider request ID differs from the raw provider response"
            )
        if replayed.decisions != self.decisions:
            raise QueryConstructionError(
                "constructor decisions differ from replayed raw provider response"
            )
        if (
            replayed.input_tokens != self.accounting.input_tokens
            or replayed.output_tokens != self.accounting.output_tokens
        ):
            raise QueryConstructionError(
                "constructor token accounting differs from replayed provider usage"
            )
        normalized_response_sha256 = sha256_json(
            [item.response_payload() for item in self.decisions]
        )
        if self.response_sha256 != normalized_response_sha256:
            raise QueryConstructionError(
                "constructor response digest differs from normalized decisions"
            )

    @property
    def receipt_sha256(self) -> str:
        return sha256_json(self.content_free_binding())

    def content_free_binding(self) -> dict[str, Any]:
        replayed = replay_constructor_provider_response(
            self.raw_response,
            turn_ids=tuple(item.turn_id for item in self.decisions),
        )
        return {
            "question_id": self.question_id,
            "window_sha256": self.window_sha256,
            "request_sha256": self.request_sha256,
            "request_utf8_bytes": self.request_utf8_bytes,
            "response_sha256": self.response_sha256,
            "provider_response": {
                "parser": self.response_parser,
                "raw_utf8_bytes": len(self.raw_response),
                "raw_sha256": sha256_bytes(self.raw_response),
                "response_model_sha256": sha256_utf8(replayed.response_model),
                "finish_reason": replayed.finish_reason,
                "system_fingerprint_sha256": (
                    sha256_utf8(replayed.system_fingerprint)
                    if replayed.system_fingerprint is not None
                    else None
                ),
                "usage": {
                    "prompt_tokens": replayed.input_tokens,
                    "completion_tokens": replayed.output_tokens,
                    "total_tokens": replayed.total_tokens,
                },
                "usage_replayed": True,
            },
            "request_id_sha256": sha256_utf8(self.request_id),
            "provider_request_id_sha256": sha256_utf8(self.provider_request_id),
            "input_fields": list(self.input_fields),
            "identity": self.identity.content_free_binding(),
            "accounting": self.accounting.as_dict(),
            "decisions": [item.content_free_binding() for item in self.decisions],
            "authentication": RECEIPT_AUTHENTICATION,
            "replay": RECEIPT_REPLAY,
        }


@dataclass(frozen=True, slots=True)
class ConstructedMemoryItem:
    turn: TurnProjection
    global_message_index: int
    retrieval_rank: int | None
    compressed_content: str
    style: RetentionStyle
    support_spans: tuple[SourceSpan, ...]
    selected_window_index: int
    constructor_receipt_sha256: str
    keep_votes: int
    appearances: int

    def __post_init__(self) -> None:
        if not isinstance(self.turn, TurnProjection):
            raise QueryConstructionError("constructed item source is not a turn")
        nonnegative_int(self.global_message_index, label="constructed item global index")
        if self.retrieval_rank is not None:
            rank = positive_int(self.retrieval_rank, label="constructed item retrieval rank")
            if rank > RETRIEVED_TURNS:
                raise QueryConstructionError("constructed item retrieval rank exceeds top 50")
        if not isinstance(self.compressed_content, str) or not self.compressed_content:
            raise QueryConstructionError("constructed item content must be non-empty text")
        try:
            self.compressed_content.encode("utf-8")
        except UnicodeError as exc:
            raise QueryConstructionError("constructed item content must be valid UTF-8") from exc
        if self.style is RetentionStyle.DROP:
            raise QueryConstructionError("constructed item cannot use the drop style")
        if not self.support_spans:
            raise QueryConstructionError("constructed item must preserve source support")
        if any(span.turn_id != self.turn.turn_id for span in self.support_spans):
            raise QueryConstructionError("constructed item support is cross-turn")
        nonnegative_int(self.selected_window_index, label="selected window index")
        checked_sha256(
            self.constructor_receipt_sha256,
            label="constructed item constructor receipt",
        )
        positive_int(self.keep_votes, label="constructed item keep_votes")
        positive_int(self.appearances, label="constructed item appearances")
        if self.keep_votes > self.appearances:
            raise QueryConstructionError("constructed item keep votes exceed appearances")

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "turn_id": list(self.turn.turn_id.as_tuple()),
            "global_message_index": self.global_message_index,
            "retrieval_rank": self.retrieval_rank,
            "compressed_content_utf8": {
                "bytes": len(self.compressed_content.encode("utf-8")),
                "sha256": sha256_utf8(self.compressed_content),
            },
            "style": self.style.value,
            "support_spans": [span.as_dict() for span in self.support_spans],
            "selected_window_index": self.selected_window_index,
            "constructor_receipt_sha256": self.constructor_receipt_sha256,
            "keep_votes": self.keep_votes,
            "appearances": self.appearances,
        }


@dataclass(frozen=True, slots=True)
class QueryConstructionResult:
    cell: ConstructionCell
    items: tuple[ConstructedMemoryItem, ...]
    reader_context: str
    receipts: tuple[WindowConstructionReceipt, ...]
    _trace_canonical_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.cell, ConstructionCell):
            raise QueryConstructionError("result cell is not registered")
        if not isinstance(self.items, tuple):
            raise QueryConstructionError("result items must be frozen in a tuple")
        if any(not isinstance(item, ConstructedMemoryItem) for item in self.items):
            raise QueryConstructionError("result contains an invalid constructed item")
        if not isinstance(self.receipts, tuple) or any(
            not isinstance(receipt, WindowConstructionReceipt) for receipt in self.receipts
        ):
            raise QueryConstructionError("result receipts must be an immutable valid tuple")
        if self.cell is ConstructionCell.RAW_TOP50 and self.receipts:
            raise QueryConstructionError("E7-A result cannot contain constructor receipts")
        if self.cell is not ConstructionCell.RAW_TOP50 and not self.receipts:
            raise QueryConstructionError("constructed E7 result must contain receipts")
        indexes = [item.global_message_index for item in self.items]
        if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
            raise QueryConstructionError("result items must be unique and chronological")
        if not isinstance(self.reader_context, str):
            raise QueryConstructionError("reader_context must be text")
        try:
            parsed = json.loads(self._trace_canonical_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise QueryConstructionError("result trace is not canonical JSON") from exc
        if canonical_json_bytes(parsed).decode("utf-8") != self._trace_canonical_json:
            raise QueryConstructionError("result trace is not canonically serialized")
        if set(parsed) != {
            "artifact_type",
            "schema_version",
            "protocol_version",
            "cell",
            "paper_transfer",
            "window_batch",
            "constructor",
            "output",
            "accounting",
            "claims",
        }:
            raise QueryConstructionError("result trace fields differ from the exact schema")
        output = parsed.get("output", {})
        if (
            parsed.get("artifact_type") != ARTIFACT_TYPE
            or parsed.get("schema_version") != SCHEMA_VERSION
            or parsed.get("protocol_version") != PROTOCOL_VERSION
        ):
            raise QueryConstructionError("result protocol identity drifted")
        if parsed.get("paper_transfer") != {
            "source": PAPER_URL,
            "repository": PAPER_REPOSITORY,
            "repository_commit": PAPER_REPOSITORY_COMMIT,
            "window_protocol": PAPER_WINDOW_PROTOCOL,
            "paper_reproduction_claimed": False,
            "paper_duplicate_policy": "longest-compression-then-earliest-window",
        }:
            raise QueryConstructionError("result paper-transfer claim drifted")
        window_batch = parsed.get("window_batch")
        if not isinstance(window_batch, dict) or set(window_batch) != {
            "trace_sha256",
            "source",
            "windows",
        }:
            raise QueryConstructionError("result window-batch binding is invalid")
        checked_sha256(window_batch.get("trace_sha256"), label="result window-batch trace")
        window_count = positive_int(window_batch.get("windows"), label="result window count")
        source = window_batch.get("source")
        if not isinstance(source, dict) or set(source) != {
            "question_id",
            "query_sha256",
            "source_artifact_sha256",
            "projection_sha256",
            "selection_cell",
            "selection_trace_sha256",
            "retrieved_turns",
            "ordered_turns_sha256",
        }:
            raise QueryConstructionError("result source-pool binding is invalid")
        if (
            source.get("selection_cell") != "E1-B"
            or source.get("retrieved_turns") != RETRIEVED_TURNS
        ):
            raise QueryConstructionError("result source is not the frozen E1-B top 50")
        for digest_field in (
            "query_sha256",
            "source_artifact_sha256",
            "projection_sha256",
            "selection_trace_sha256",
            "ordered_turns_sha256",
        ):
            checked_sha256(
                source.get(digest_field),
                label=f"result source {digest_field}",
            )
        required_text(source.get("question_id"), label="result source question_id")
        question_ids = {item.turn.turn_id.question_id for item in self.items}
        question_ids.update(receipt.question_id for receipt in self.receipts)
        if question_ids and question_ids != {source["question_id"]}:
            raise QueryConstructionError("result contains a cross-question source binding")
        if self.receipts and window_count != len(self.receipts):
            raise QueryConstructionError("result receipt count differs from its window batch")
        if self.items and max(item.selected_window_index for item in self.items) >= window_count:
            raise QueryConstructionError("result item refers to an unknown source window")
        if self.cell is ConstructionCell.RAW_TOP50:
            if len(self.items) != RETRIEVED_TURNS:
                raise QueryConstructionError("E7-A result must contain the complete raw top 50")
            if sorted(
                item.retrieval_rank for item in self.items if item.retrieval_rank is not None
            ) != list(range(1, RETRIEVED_TURNS + 1)) or any(
                item.retrieval_rank is None for item in self.items
            ):
                raise QueryConstructionError("E7-A result retrieval ranks drifted")
            if any(
                item.style is not RetentionStyle.VERBATIM
                or item.compressed_content != item.turn.original_content
                for item in self.items
            ):
                raise QueryConstructionError("E7-A result is not raw verbatim source content")
        if not isinstance(output, dict) or set(output) != {
            "serializer",
            "items",
            "items_sha256",
            "item_count",
            "unique_source_turns",
            "reader_context_utf8_bytes",
            "reader_context_sha256",
        }:
            raise QueryConstructionError("result output fields differ from the exact schema")
        if output.get("serializer") != READER_CONTEXT_SERIALIZER:
            raise QueryConstructionError("result reader-context serializer drifted")
        if parsed.get("cell") != self.cell.value:
            raise QueryConstructionError("result cell differs from its trace")
        bindings = [item.content_free_binding() for item in self.items]
        if output.get("items") != bindings:
            raise QueryConstructionError("result items differ from their trace")
        if output.get("items_sha256") != sha256_json(bindings):
            raise QueryConstructionError("result item digest is inconsistent")
        if output.get("item_count") != len(self.items):
            raise QueryConstructionError("result item count is inconsistent")
        if output.get("unique_source_turns") != len({item.turn.turn_id for item in self.items}):
            raise QueryConstructionError("result unique-source count is inconsistent")
        expected_context = render_reader_context(self.items)
        if self.reader_context != expected_context:
            raise QueryConstructionError("reader context differs from result items")
        if output.get("reader_context_sha256") != sha256_utf8(self.reader_context):
            raise QueryConstructionError("reader context differs from its trace digest")
        if output.get("reader_context_utf8_bytes") != len(self.reader_context.encode("utf-8")):
            raise QueryConstructionError("reader context differs from its trace byte count")
        receipt_bindings = [receipt.content_free_binding() for receipt in self.receipts]
        constructor = parsed.get("constructor")
        if constructor != {
            "input_fields": list(CONSTRUCTOR_INPUT_FIELDS),
            "receipt_authentication": RECEIPT_AUTHENTICATION,
            "receipts": receipt_bindings,
            "receipts_sha256": sha256_json(receipt_bindings),
        }:
            raise QueryConstructionError("result constructor evidence drifted")
        identities = {receipt.identity.identity_sha256 for receipt in self.receipts}
        if len(identities) > 1:
            raise QueryConstructionError("result constructor identity drifted")
        decisions = [decision for receipt in self.receipts for decision in receipt.decisions]
        accounting = parsed.get("accounting")
        expected_accounting = {
            "constructor_identity_sha256": next(iter(identities), None),
            "constructor_windows": len(self.receipts),
            "constructor_input_tokens": sum(
                receipt.accounting.input_tokens for receipt in self.receipts
            ),
            "constructor_output_tokens": sum(
                receipt.accounting.output_tokens for receipt in self.receipts
            ),
            "constructor_provider_response_bytes": sum(
                len(receipt.raw_response) for receipt in self.receipts
            ),
            "constructor_usage_replayed_windows": len(self.receipts),
            "constructor_latency_sum_ms": sum(
                receipt.accounting.latency_ms for receipt in self.receipts
            ),
            "constructor_window_max_latency_ms": max(
                (receipt.accounting.latency_ms for receipt in self.receipts),
                default=0.0,
            ),
            "constructor_batch_wall_clock_ms": None if self.receipts else 0.0,
            "constructor_cost_usd": sum(receipt.accounting.cost_usd for receipt in self.receipts),
            "verbatim_keep_decisions": (
                sum(
                    decision.operation is DecisionOperation.KEEP
                    and decision.style is RetentionStyle.VERBATIM
                    for decision in decisions
                )
                if self.receipts
                else len(self.items)
            ),
            "extractive_keep_decisions": sum(
                decision.operation is DecisionOperation.KEEP
                and decision.style is RetentionStyle.EXTRACTIVE
                for decision in decisions
            ),
            "abstractive_keep_decisions": sum(
                decision.operation is DecisionOperation.KEEP
                and decision.style is RetentionStyle.ABSTRACTIVE
                for decision in decisions
            ),
            "compressed_content_utf8_bytes": sum(
                len(item.compressed_content.encode("utf-8")) for item in self.items
            ),
            "source_support_spans": sum(len(item.support_spans) for item in self.items),
        }
        if accounting != expected_accounting:
            raise QueryConstructionError("result accounting drifted")
        styles = {item.style for item in self.items}
        if parsed.get("claims") != {
            "source_turns_immutable": True,
            "every_keep_has_source_spans": True,
            "all_output_byte_grounded": RetentionStyle.ABSTRACTIVE not in styles,
            "abstractive_faithfulness_proven": False,
            "external_constructor_identity_verified": False,
            "external_receipts_authenticated": False,
            "raw_provider_response_replay_required": self.cell is not ConstructionCell.RAW_TOP50,
            "raw_provider_response_replay_complete": True,
            "provider_usage_replay_complete": True,
            "latency_and_cost_verified_from_provider": False,
            "parallel_execution_or_batch_wall_clock_proven": self.cell
            is ConstructionCell.RAW_TOP50,
            "exact_reader_prompt_packed": False,
            "reader_or_judge_executed": False,
            "qa_improvement_proven": False,
            "serving_eligibility_proven": False,
        }:
            raise QueryConstructionError("result claims drifted")

    @property
    def trace_sha256(self) -> str:
        return sha256_utf8(self._trace_canonical_json)

    def content_free_trace(self) -> dict[str, Any]:
        payload = json.loads(self._trace_canonical_json)
        return {**payload, "trace_sha256": self.trace_sha256}


def render_reader_context(items: tuple[ConstructedMemoryItem, ...]) -> str:
    """Render chronological E7 material before full reader-prompt packing."""

    if not isinstance(items, tuple):
        raise QueryConstructionError("reader context items must be frozen in a tuple")
    if not items:
        return EMPTY_MEMORY_CONTEXT
    return "\n".join(
        f"[{item.turn.parent_session_date}] {item.turn.role}: {item.compressed_content}"
        for item in items
    )


__all__ = [
    "ARTIFACT_TYPE",
    "CONSTRUCTOR_INPUT_FIELDS",
    "CONTEXT_RADIUS",
    "EMPTY_MEMORY_CONTEXT",
    "EXTRACTIVE_SEPARATOR",
    "MAX_WINDOW_MESSAGES",
    "MAX_COMPRESSED_UTF8_BYTES",
    "MAX_REASON_UTF8_BYTES",
    "MAX_RAW_PROVIDER_RESPONSE_BYTES",
    "MAX_SUPPORT_SPANS_PER_KEEP",
    "PAPER_REPOSITORY",
    "PAPER_REPOSITORY_COMMIT",
    "PAPER_URL",
    "PAPER_WINDOW_PROTOCOL",
    "PROTOCOL_VERSION",
    "PROVIDER_RESPONSE_PARSER",
    "RECEIPT_AUTHENTICATION",
    "RECEIPT_REPLAY",
    "READER_CONTEXT_SERIALIZER",
    "RETRIEVED_TURNS",
    "SCHEMA_VERSION",
    "WINDOW_STRIDE",
    "ConstructedMemoryItem",
    "ConstructionAccounting",
    "ConstructionCell",
    "ConstructionDecision",
    "ConstructorIdentity",
    "DecisionOperation",
    "QueryConstructionError",
    "QueryConstructionResult",
    "QueryWindow",
    "ReplayedConstructorResponse",
    "RetentionStyle",
    "RetrievedTurnPool",
    "SourceSpan",
    "WindowConstructionReceipt",
    "WindowMessage",
    "WindowPosition",
    "canonical_json_bytes",
    "checked_sha256",
    "nonnegative_int",
    "nonnegative_number",
    "positive_int",
    "replay_constructor_provider_response",
    "required_text",
    "render_reader_context",
    "sha256_bytes",
    "sha256_json",
    "sha256_utf8",
]
