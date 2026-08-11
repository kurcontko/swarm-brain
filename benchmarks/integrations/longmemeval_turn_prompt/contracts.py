"""Frozen contracts for exact, benchmark-only LongMemEval turn-prompt packing."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from benchmarks.integrations.longmemeval_turns import LongMemEvalTurnId, TurnProjection

ARTIFACT_TYPE = "swarmbrain-longmemeval-turn-prompt-packing-trace"
SCHEMA_VERSION = 1
PROTOCOL_VERSION = "swarmbrain-longmemeval-turn-prompt-pack-v1"
TOKENIZER_PROTOCOL = "swarmbrain-exact-tokenizer-jsonl-v1"

SUPPORTED_TOKEN_BUDGETS = (4096, 8192, 16384)
PRIMARY_TOKEN_BUDGET = 8192

LINEAR_TURN_SEPARATOR = "\n\n"
CHAIN_HEADER_TEMPLATE = "=== Evidence Chain {chain_number} ==="
CHAIN_HEADER_BODY_SEPARATOR = "\n"
CHAIN_TURN_SEPARATOR = "\n\n"
CHAIN_BLOCK_SEPARATOR = "\n\n"
HISTORY_SERIALIZER_VERSION = "longmemeval-turn-prompt-history-v1"

OFFICIAL_ANSWER_TEMPLATE_SHA256 = "e427ff913456e51a132ec865b1b5038d562bdc36890976943ad421cc9b365c9d"
EMPTY_CONTEXT_NOTE_SHA256 = "fd20ff537dab00b93f5246c94c66d5c4f49b9dde2159f43d6ae63bace2771fbc"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PROVIDER_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,256}")


class TurnPromptPackingError(ValueError):
    """A prompt input, tokenizer receipt, or derived packing state is invalid."""


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
        raise TurnPromptPackingError("trace values must be finite canonical UTF-8 JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    try:
        return sha256_bytes(value.encode("utf-8"))
    except UnicodeError as exc:
        raise TurnPromptPackingError("prompt inputs must be valid UTF-8 text") from exc


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def checked_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TurnPromptPackingError(f"{label} must be a lowercase hexadecimal SHA-256 digest")
    return value


def checked_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TurnPromptPackingError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise TurnPromptPackingError(f"{label} cannot have leading or trailing whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise TurnPromptPackingError(f"{label} cannot contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise TurnPromptPackingError(f"{label} must be valid UTF-8") from exc
    return value


def checked_positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TurnPromptPackingError(f"{label} must be a positive integer")
    return value


class PromptLayout(StrEnum):
    """The two registered caller-frozen evidence organizations."""

    LINEAR = "linear-e1"
    CHAIN_BLOCKS = "ordered-com-blocks"


@dataclass(frozen=True, slots=True)
class OrderedTurnBlocks:
    """Immutable turns in the exact order in which packing must consider them."""

    layout: PromptLayout
    blocks: tuple[tuple[TurnProjection, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layout, PromptLayout):
            raise TurnPromptPackingError("prompt layout must be a registered PromptLayout")
        if not isinstance(self.blocks, tuple):
            raise TurnPromptPackingError("prompt blocks must be frozen in a tuple")
        if self.layout is PromptLayout.LINEAR and len(self.blocks) != 1:
            raise TurnPromptPackingError("linear E1 input must contain exactly one block")
        if self.layout is PromptLayout.CHAIN_BLOCKS and not self.blocks:
            raise TurnPromptPackingError("ordered CoM input must contain at least one block")
        for block in self.blocks:
            if not isinstance(block, tuple):
                raise TurnPromptPackingError("every prompt block must be frozen in a tuple")
            if any(not isinstance(turn, TurnProjection) for turn in block):
                raise TurnPromptPackingError("every prompt candidate must be a TurnProjection")

    @classmethod
    def linear(cls, turns: tuple[TurnProjection, ...]) -> OrderedTurnBlocks:
        if not isinstance(turns, tuple):
            raise TurnPromptPackingError("linear turn order must be frozen in a tuple")
        return cls(layout=PromptLayout.LINEAR, blocks=(turns,))

    @classmethod
    def chain_blocks(
        cls,
        blocks: tuple[tuple[TurnProjection, ...], ...],
    ) -> OrderedTurnBlocks:
        return cls(layout=PromptLayout.CHAIN_BLOCKS, blocks=blocks)

    @property
    def flattened(self) -> tuple[TurnProjection, ...]:
        return tuple(turn for block in self.blocks for turn in block)

    def positions(self) -> tuple[tuple[int, int, TurnProjection], ...]:
        return tuple(
            (block_position, position_in_block, turn)
            for block_position, block in enumerate(self.blocks, start=1)
            for position_in_block, turn in enumerate(block, start=1)
        )


@dataclass(frozen=True, slots=True)
class TokenizerIdentity:
    """Immutable identity of the externally executed exact tokenizer."""

    protocol: str
    model: str
    revision: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        if self.protocol != TOKENIZER_PROTOCOL:
            raise TurnPromptPackingError(
                f"tokenizer protocol must be the frozen {TOKENIZER_PROTOCOL!r}"
            )
        checked_identifier(self.model, label="tokenizer model")
        checked_identifier(self.revision, label="tokenizer revision")
        checked_sha256(self.artifact_sha256, label="tokenizer artifact_sha256")

    def as_dict(self) -> dict[str, str]:
        return {
            "protocol": self.protocol,
            "model": self.model,
            "revision": self.revision,
            "artifact_sha256": self.artifact_sha256,
            "identity_sha256": self.identity_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return sha256_json(
            {
                "protocol": self.protocol,
                "tokenizer_artifact_sha256": self.artifact_sha256,
                "tokenizer_model": self.model,
                "tokenizer_revision": self.revision,
            }
        )


@dataclass(frozen=True, slots=True)
class ExactTokenCountReceipt:
    """Provider-observed exact count for one complete reader prompt."""

    request_id: int
    provider_request_id: str
    tokenizer_identity_sha256: str
    query_sha256: str
    prompt_sha256: str
    prompt_utf8_bytes: int
    token_count: int

    def __post_init__(self) -> None:
        checked_positive_integer(self.request_id, label="tokenizer receipt request_id")
        if (
            not isinstance(self.provider_request_id, str)
            or _PROVIDER_REQUEST_ID_RE.fullmatch(self.provider_request_id) is None
        ):
            raise TurnPromptPackingError("tokenizer provider_request_id is invalid")
        checked_sha256(
            self.tokenizer_identity_sha256,
            label="tokenizer receipt identity digest",
        )
        checked_sha256(self.query_sha256, label="tokenizer receipt query digest")
        checked_sha256(self.prompt_sha256, label="tokenizer receipt prompt digest")
        checked_positive_integer(
            self.prompt_utf8_bytes,
            label="tokenizer receipt prompt UTF-8 bytes",
        )
        checked_positive_integer(self.token_count, label="tokenizer receipt token_count")

    def content_free_binding(self) -> dict[str, int | str]:
        return {
            "request_id": self.request_id,
            "provider_request_id": self.provider_request_id,
            "tokenizer_identity_sha256": self.tokenizer_identity_sha256,
            "query_sha256": self.query_sha256,
            "prompt_sha256": self.prompt_sha256,
            "prompt_utf8_bytes": self.prompt_utf8_bytes,
            "token_count": self.token_count,
        }


class ExactPromptTokenizer(Protocol):
    """External exact-count boundary; implementations may call only local pinned code."""

    @property
    def identity(self) -> TokenizerIdentity: ...

    def count_prompt(
        self,
        prompt: str,
        *,
        query_sha256: str,
    ) -> ExactTokenCountReceipt: ...


@dataclass(frozen=True, slots=True)
class PackingDecision:
    candidate_turn_id: LongMemEvalTurnId
    block_position: int
    position_in_block: int
    selected_before_ids: tuple[LongMemEvalTurnId, ...]
    proposed_ids: tuple[LongMemEvalTurnId, ...]
    singleton_observation_sequence: int
    proposal_observation_sequence: int
    accepted: bool
    oversized_alone: bool

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "candidate_turn_id": list(self.candidate_turn_id.as_tuple()),
            "block_position": self.block_position,
            "position_in_block": self.position_in_block,
            "selected_before_ids": [list(item.as_tuple()) for item in self.selected_before_ids],
            "proposed_ids": [list(item.as_tuple()) for item in self.proposed_ids],
            "singleton_observation_sequence": self.singleton_observation_sequence,
            "proposal_observation_sequence": self.proposal_observation_sequence,
            "accepted": self.accepted,
            "oversized_alone": self.oversized_alone,
        }


@dataclass(frozen=True, slots=True)
class ExactCountObservation:
    sequence: int
    purpose: str
    candidate_turn_id: LongMemEvalTurnId | None
    receipt: ExactTokenCountReceipt

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "purpose": self.purpose,
            "candidate_turn_id": (
                None if self.candidate_turn_id is None else list(self.candidate_turn_id.as_tuple())
            ),
            "receipt": self.receipt.content_free_binding(),
        }


@dataclass(frozen=True, slots=True)
class TurnPromptPackingResult:
    """The only returned field containing benchmark question or turn text is ``prompt``."""

    prompt: str
    trace: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise TurnPromptPackingError("packed reader prompt must be non-empty text")
        if not isinstance(self.trace, dict):
            raise TurnPromptPackingError("packed reader trace must be a dictionary")
        final = self.trace.get("final_prompt")
        if not isinstance(final, dict):
            raise TurnPromptPackingError("packing trace is missing final prompt evidence")
        encoded = self.prompt.encode("utf-8")
        if final.get("sha256") != sha256_bytes(encoded):
            raise TurnPromptPackingError("returned prompt differs from its trace digest")
        if final.get("utf8_bytes") != len(encoded):
            raise TurnPromptPackingError("returned prompt differs from its trace byte count")

    @property
    def trace_sha256(self) -> str:
        return sha256_json(self.trace)

    def content_free_artifact(self) -> dict[str, Any]:
        return {**self.trace, "trace_sha256": self.trace_sha256}


__all__ = [
    "ARTIFACT_TYPE",
    "CHAIN_BLOCK_SEPARATOR",
    "CHAIN_HEADER_BODY_SEPARATOR",
    "CHAIN_HEADER_TEMPLATE",
    "CHAIN_TURN_SEPARATOR",
    "EMPTY_CONTEXT_NOTE_SHA256",
    "ExactCountObservation",
    "ExactPromptTokenizer",
    "ExactTokenCountReceipt",
    "HISTORY_SERIALIZER_VERSION",
    "LINEAR_TURN_SEPARATOR",
    "OFFICIAL_ANSWER_TEMPLATE_SHA256",
    "OrderedTurnBlocks",
    "PRIMARY_TOKEN_BUDGET",
    "PROTOCOL_VERSION",
    "PackingDecision",
    "PromptLayout",
    "SCHEMA_VERSION",
    "SUPPORTED_TOKEN_BUDGETS",
    "TOKENIZER_PROTOCOL",
    "TokenizerIdentity",
    "TurnPromptPackingError",
    "TurnPromptPackingResult",
    "canonical_json_bytes",
    "checked_identifier",
    "checked_positive_integer",
    "checked_sha256",
    "sha256_bytes",
    "sha256_json",
    "sha256_text",
]
