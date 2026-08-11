"""Frozen, benchmark-only contracts for the Chain-of-Memory E2 cells.

The contracts deliberately consume similarities instead of producing them.
Embedding execution, model identity, and the evidence artifact that contains
the similarities live outside this pure/offline module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from benchmarks.integrations.longmemeval_turns import (
    TURN_SERIALIZER_VERSION,
    LongMemEvalTurnId,
    TurnProjection,
)

ARTIFACT_TYPE = "swarmbrain-chain-of-memory-organization-trace"
SCHEMA_VERSION = 1
PROTOCOL_VERSION = "swarmbrain-longmemeval-chain-of-memory-e2-v1"

K = 20
L = 3
BETA = 0.5
CHAIN_CONTEXT_SEPARATOR = "\n\n"
CHAIN_CONTEXT_SERIALIZER_VERSION = f"{TURN_SERIALIZER_VERSION}-double-newline-concatenated-chain-v1"
MAX_CHAIN_LENGTH = K
MAX_CHAIN_DECISIONS = K - 1
MAX_TOTAL_DECISIONS = L * MAX_CHAIN_DECISIONS
MAX_CONTEXT_SCORE_EVALUATIONS = L * (K * (K - 1) // 2)
MAX_PARITY_RENDERED_TURNS = K * L

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ChainOfMemoryError(ValueError):
    """The supplied E2 input or a derived decision violates the frozen contract."""


class E2Cell(StrEnum):
    """Preregistered E2 organization cells from the frozen research protocol."""

    RETRIEVAL_ORDER = "E2-A"
    QUERY_ONLY_APT = "E2-B"
    PRODUCT_NO_APT = "E2-C"
    PRODUCT_APT = "E2-D"
    PRODUCT_APT_DEDUP = "E2-E"


@dataclass(frozen=True, slots=True)
class CellProtocol:
    cell: E2Cell
    evolves_chains: bool
    score_mode: str
    adaptive_path_truncation: bool
    deduplicate_cross_chain_rendering: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell.value,
            "evolves_chains": self.evolves_chains,
            "score_mode": self.score_mode,
            "adaptive_path_truncation": self.adaptive_path_truncation,
            "deduplicate_cross_chain_rendering": self.deduplicate_cross_chain_rendering,
        }


CELL_PROTOCOLS = MappingProxyType(
    {
        E2Cell.RETRIEVAL_ORDER: CellProtocol(
            cell=E2Cell.RETRIEVAL_ORDER,
            evolves_chains=False,
            score_mode="retrieval_order",
            adaptive_path_truncation=False,
            deduplicate_cross_chain_rendering=False,
        ),
        E2Cell.QUERY_ONLY_APT: CellProtocol(
            cell=E2Cell.QUERY_ONLY_APT,
            evolves_chains=True,
            score_mode="query_only",
            adaptive_path_truncation=True,
            deduplicate_cross_chain_rendering=False,
        ),
        E2Cell.PRODUCT_NO_APT: CellProtocol(
            cell=E2Cell.PRODUCT_NO_APT,
            evolves_chains=True,
            score_mode="product",
            adaptive_path_truncation=False,
            deduplicate_cross_chain_rendering=False,
        ),
        E2Cell.PRODUCT_APT: CellProtocol(
            cell=E2Cell.PRODUCT_APT,
            evolves_chains=True,
            score_mode="product",
            adaptive_path_truncation=True,
            deduplicate_cross_chain_rendering=False,
        ),
        E2Cell.PRODUCT_APT_DEDUP: CellProtocol(
            cell=E2Cell.PRODUCT_APT_DEDUP,
            evolves_chains=True,
            score_mode="product",
            adaptive_path_truncation=True,
            deduplicate_cross_chain_rendering=True,
        ),
    }
)


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
        raise ChainOfMemoryError("trace value must be finite canonical UTF-8 JSON") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _required_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ChainOfMemoryError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ChainOfMemoryError(f"{label} cannot have leading or trailing whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ChainOfMemoryError(f"{label} cannot contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ChainOfMemoryError(f"{label} must be valid UTF-8") from exc
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ChainOfMemoryError(f"{label} must be a lowercase hexadecimal SHA-256 digest")
    return value


def normalized_cosine(value: Any, *, label: str) -> float:
    """Validate cosine evidence from normalized vectors in the closed [-1, 1] range."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChainOfMemoryError(f"{label} must be a finite normalized cosine in [-1, 1]")
    result = float(value)
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ChainOfMemoryError(f"{label} must be a finite normalized cosine in [-1, 1]")
    return 0.0 if result == 0.0 else result


def finite_score(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChainOfMemoryError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ChainOfMemoryError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def turn_id_payload(turn_id: LongMemEvalTurnId) -> list[str | int]:
    return list(turn_id.as_tuple())


def chain_context_text(chain: tuple[TurnProjection, ...]) -> str:
    if not chain or len(chain) > MAX_CHAIN_LENGTH:
        raise ChainOfMemoryError(f"chain context must contain 1..{MAX_CHAIN_LENGTH} turns")
    return CHAIN_CONTEXT_SEPARATOR.join(turn.serialized_text for turn in chain)


def chain_context_sha256(chain: tuple[TurnProjection, ...]) -> str:
    return hashlib.sha256(chain_context_text(chain).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExternalSimilarityEvidence:
    """Caller-supplied binding to the artifact that produced similarity values."""

    producer: str
    model_id: str
    model_revision: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _required_identifier(self.producer, label="similarity producer")
        _required_identifier(self.model_id, label="similarity model_id")
        _required_identifier(self.model_revision, label="similarity model_revision")
        _sha256(self.artifact_sha256, label="similarity artifact_sha256")

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "artifact_sha256": self.artifact_sha256,
            "execution_boundary": "external-to-organizer",
            "verified_by_organizer": False,
        }


@dataclass(frozen=True, slots=True)
class ChainCandidate:
    """One fixed retrieval-head candidate plus its external query cosine."""

    turn: TurnProjection
    query_cosine: float

    def __post_init__(self) -> None:
        if not isinstance(self.turn, TurnProjection):
            raise ChainOfMemoryError("chain candidate must contain a TurnProjection")
        object.__setattr__(
            self,
            "query_cosine",
            normalized_cosine(self.query_cosine, label="candidate query_cosine"),
        )

    @property
    def turn_id(self) -> LongMemEvalTurnId:
        return self.turn.turn_id


@dataclass(frozen=True, slots=True)
class ContextCosine:
    """Precomputed candidate-to-concatenated-chain cosine for one exact prefix."""

    chain_turn_ids: tuple[LongMemEvalTurnId, ...]
    candidate_turn_id: LongMemEvalTurnId
    chain_context_sha256: str
    candidate_document_sha256: str
    cosine: float

    def __post_init__(self) -> None:
        if not 1 <= len(self.chain_turn_ids) <= MAX_CHAIN_LENGTH:
            raise ChainOfMemoryError(
                f"context cosine chain must contain 1..{MAX_CHAIN_LENGTH} turn IDs"
            )
        if any(not isinstance(turn_id, LongMemEvalTurnId) for turn_id in self.chain_turn_ids):
            raise ChainOfMemoryError("context cosine chain IDs must be LongMemEvalTurnId values")
        if not isinstance(self.candidate_turn_id, LongMemEvalTurnId):
            raise ChainOfMemoryError(
                "context cosine candidate ID must be a LongMemEvalTurnId value"
            )
        if len(set(self.chain_turn_ids)) != len(self.chain_turn_ids):
            raise ChainOfMemoryError("context cosine chain cannot repeat a turn ID")
        if self.candidate_turn_id in self.chain_turn_ids:
            raise ChainOfMemoryError("context cosine candidate is already in its chain")
        question_ids = {
            turn_id.question_id for turn_id in (*self.chain_turn_ids, self.candidate_turn_id)
        }
        if len(question_ids) != 1:
            raise ChainOfMemoryError("context cosine IDs must belong to one question")
        _sha256(self.chain_context_sha256, label="chain context digest")
        _sha256(self.candidate_document_sha256, label="candidate document digest")
        object.__setattr__(
            self,
            "cosine",
            normalized_cosine(self.cosine, label="context cosine"),
        )

    @classmethod
    def from_turns(
        cls,
        chain: tuple[TurnProjection, ...],
        candidate: TurnProjection,
        cosine: float,
    ) -> ContextCosine:
        return cls(
            chain_turn_ids=tuple(turn.turn_id for turn in chain),
            candidate_turn_id=candidate.turn_id,
            chain_context_sha256=chain_context_sha256(chain),
            candidate_document_sha256=candidate.serialized_document_utf8.sha256,
            cosine=cosine,
        )

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "chain_turn_ids": [turn_id_payload(turn_id) for turn_id in self.chain_turn_ids],
            "candidate_turn_id": turn_id_payload(self.candidate_turn_id),
            "chain_context_sha256": self.chain_context_sha256,
            "candidate_document_sha256": self.candidate_document_sha256,
            "cosine": self.cosine,
        }


@dataclass(frozen=True, slots=True)
class PrecomputedContextSimilarities:
    """Immutable exact-prefix lookup for externally computed context cosines."""

    entries: tuple[ContextCosine, ...]
    _by_key: Any = field(init=False, repr=False, compare=False)

    def __init__(self, entries: Iterable[ContextCosine]) -> None:
        values = tuple(entries)
        if len(values) > MAX_CONTEXT_SCORE_EVALUATIONS:
            raise ChainOfMemoryError(
                "precomputed context table exceeds the hard score-evidence bound"
            )
        lookup: dict[tuple[tuple[LongMemEvalTurnId, ...], LongMemEvalTurnId], ContextCosine] = {}
        for entry in values:
            if not isinstance(entry, ContextCosine):
                raise ChainOfMemoryError("precomputed context entries must be ContextCosine")
            key = (entry.chain_turn_ids, entry.candidate_turn_id)
            if key in lookup:
                raise ChainOfMemoryError("precomputed context table repeats an exact lookup key")
            lookup[key] = entry
        object.__setattr__(self, "entries", values)
        object.__setattr__(self, "_by_key", MappingProxyType(lookup))

    @property
    def table_sha256(self) -> str:
        ordered = sorted(
            (entry.content_free_binding() for entry in self.entries),
            key=lambda item: canonical_json_bytes(item),
        )
        return sha256_json(ordered)

    def lookup(
        self,
        chain: tuple[TurnProjection, ...],
        candidate: TurnProjection,
    ) -> float:
        key = (tuple(turn.turn_id for turn in chain), candidate.turn_id)
        entry = self._by_key.get(key)
        if entry is None:
            raise ChainOfMemoryError(
                "precomputed context table is missing the exact chain-prefix/candidate pair"
            )
        if entry.chain_context_sha256 != chain_context_sha256(chain):
            raise ChainOfMemoryError("precomputed context digest differs from the exact chain text")
        if entry.candidate_document_sha256 != candidate.serialized_document_utf8.sha256:
            raise ChainOfMemoryError(
                "precomputed candidate digest differs from the exact turn document"
            )
        return entry.cosine


ContextSimilarityCallback = Callable[[str, str], float]
ContextSimilaritySource = PrecomputedContextSimilarities | ContextSimilarityCallback


@dataclass(frozen=True, slots=True)
class CandidateScoreTrace:
    candidate_turn_id: LongMemEvalTurnId
    retrieval_rank: int
    query_cosine: float
    context_cosine: float | None
    gate_score: float

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "candidate_turn_id": turn_id_payload(self.candidate_turn_id),
            "retrieval_rank": self.retrieval_rank,
            "query_cosine": self.query_cosine,
            "context_cosine": self.context_cosine,
            "gate_score": self.gate_score,
        }


@dataclass(frozen=True, slots=True)
class ChainDecision:
    anchor_turn_id: LongMemEvalTurnId
    iteration: int
    previous_appended_score: float
    threshold: float | None
    scorecard: tuple[CandidateScoreTrace, ...]
    best_candidate_turn_id: LongMemEvalTurnId
    best_score: float
    appended: bool
    reason: str

    def content_free_binding(self) -> dict[str, Any]:
        scorecard = [score.content_free_binding() for score in self.scorecard]
        return {
            "anchor_turn_id": turn_id_payload(self.anchor_turn_id),
            "iteration": self.iteration,
            "previous_appended_score": self.previous_appended_score,
            "threshold": self.threshold,
            "scorecard": scorecard,
            "scorecard_sha256": sha256_json(scorecard),
            "best_candidate_turn_id": turn_id_payload(self.best_candidate_turn_id),
            "best_score": self.best_score,
            "appended": self.appended,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class EvidenceChain:
    anchor: ChainCandidate
    turns: tuple[ChainCandidate, ...]
    marginal_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.turns or self.turns[0].turn_id != self.anchor.turn_id:
            raise ChainOfMemoryError("evidence chain must begin with its anchor")
        if len(self.turns) > MAX_CHAIN_LENGTH:
            raise ChainOfMemoryError("evidence chain exceeds the hard K bound")
        if len({candidate.turn_id for candidate in self.turns}) != len(self.turns):
            raise ChainOfMemoryError("a single evidence chain cannot repeat a turn")
        if len(self.turns) != len(self.marginal_scores):
            raise ChainOfMemoryError("chain turns and marginal scores must align")
        for score in self.marginal_scores:
            finite_score(score, label="chain marginal score")

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "anchor_turn_id": turn_id_payload(self.anchor.turn_id),
            "anchor_query_cosine": self.anchor.query_cosine,
            "turn_ids": [turn_id_payload(candidate.turn_id) for candidate in self.turns],
            "marginal_scores": list(self.marginal_scores),
        }


__all__ = [
    "ARTIFACT_TYPE",
    "BETA",
    "CELL_PROTOCOLS",
    "CHAIN_CONTEXT_SEPARATOR",
    "CHAIN_CONTEXT_SERIALIZER_VERSION",
    "K",
    "L",
    "MAX_CHAIN_DECISIONS",
    "MAX_CHAIN_LENGTH",
    "MAX_CONTEXT_SCORE_EVALUATIONS",
    "MAX_PARITY_RENDERED_TURNS",
    "MAX_TOTAL_DECISIONS",
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "CandidateScoreTrace",
    "CellProtocol",
    "ChainCandidate",
    "ChainDecision",
    "ChainOfMemoryError",
    "ContextCosine",
    "ContextSimilarityCallback",
    "ContextSimilaritySource",
    "E2Cell",
    "EvidenceChain",
    "ExternalSimilarityEvidence",
    "PrecomputedContextSimilarities",
    "canonical_json_bytes",
    "chain_context_sha256",
    "chain_context_text",
    "finite_score",
    "normalized_cosine",
    "sha256_json",
    "turn_id_payload",
]
