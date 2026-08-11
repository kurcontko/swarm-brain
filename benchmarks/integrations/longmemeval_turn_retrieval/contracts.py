"""Frozen contracts for the offline LongMemEval turn-retrieval head.

This package is an evaluation boundary, not another production retriever.  It
accepts ranked observations produced elsewhere and deterministically fuses
them.  In particular, it never computes lexical scores, embeddings, or dense
similarities itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from benchmarks.integrations.longmemeval_turns import LongMemEvalTurnId

ARTIFACT_TYPE = "swarmbrain-longmemeval-turn-retrieval-trace"
SCHEMA_VERSION = 1
E1A_PROTOCOL_VERSION = "swarmbrain-longmemeval-turn-transfer-e1a-v1"
E1A_PROTOCOL_NAME = "Swarm Brain LongMemEval turn-transfer E1-A"

LEXICAL_LANE_DEPTH = 128
DENSE_LANE_DEPTH = 128
LEXICAL_WEIGHT = 3.0
DENSE_WEIGHT = 4.0
RRF_K = 60
FUSED_CANDIDATE_CAP = 128
PRODUCTION_DENSE_SERVING_CAP = 100

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_E1A_SIGNATURE = (
    E1A_PROTOCOL_NAME,
    "E1-A",
    LEXICAL_LANE_DEPTH,
    DENSE_LANE_DEPTH,
    LEXICAL_WEIGHT,
    DENSE_WEIGHT,
    RRF_K,
    FUSED_CANDIDATE_CAP,
)


class TurnRetrievalError(ValueError):
    """Input evidence cannot support a deterministic E1-A trace."""


class RetrievalLane(StrEnum):
    LEXICAL = "lexical"
    DENSE = "dense"


def canonical_json_bytes(value: Any) -> bytes:
    """Encode finite JSON without normalizing Unicode or numeric values."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TurnRetrievalError("trace value must be finite canonical UTF-8 JSON") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TurnRetrievalError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise TurnRetrievalError(f"{label} cannot have leading or trailing whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise TurnRetrievalError(f"{label} cannot contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise TurnRetrievalError(f"{label} must be valid UTF-8") from exc
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TurnRetrievalError(f"{label} must be a lowercase hexadecimal SHA-256 digest")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TurnRetrievalError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TurnRetrievalError(f"{label} must be a non-negative integer")
    return value


def finite_score(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TurnRetrievalError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TurnRetrievalError(f"{label} must be a finite number")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True, slots=True)
class ImmutableArtifactIdentity:
    """Caller-attested immutable identity for a scorer or its projection."""

    name: str
    revision: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.name, label="component name")
        _identifier(self.revision, label="component revision")
        _sha256(self.artifact_sha256, label="component artifact_sha256")

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "revision": self.revision,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExternalLaneIdentity:
    """Immutable external producer, scorer, projection, and observation binding.

    The fusion layer validates the spelling and digests but cannot prove that
    the external process actually loaded those artifacts.  The trace therefore
    labels this identity as attested rather than verified.
    """

    producer: str
    scorer: ImmutableArtifactIdentity
    projection: ImmutableArtifactIdentity
    observation_artifact_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.producer, label="lane producer")
        if not isinstance(self.scorer, ImmutableArtifactIdentity):
            raise TurnRetrievalError("lane scorer must have an immutable artifact identity")
        if not isinstance(self.projection, ImmutableArtifactIdentity):
            raise TurnRetrievalError("lane projection must have an immutable artifact identity")
        _sha256(
            self.observation_artifact_sha256,
            label="lane observation_artifact_sha256",
        )

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "scorer": self.scorer.as_dict(),
            "projection": self.projection.as_dict(),
            "observation_artifact_sha256": self.observation_artifact_sha256,
            "identity_source": "externally-attested-unverified",
            "artifacts_reopened_by_fusion": False,
        }


@dataclass(frozen=True, slots=True)
class RankedTurnObservation:
    """One exact turn ID and its external lane score; tuple order is rank."""

    turn_id: LongMemEvalTurnId
    raw_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.turn_id, LongMemEvalTurnId):
            raise TurnRetrievalError("ranked observation must use an exact LongMemEvalTurnId")
        object.__setattr__(
            self,
            "raw_score",
            finite_score(self.raw_score, label="ranked observation raw_score"),
        )

    def content_free_binding(self, *, rank: int) -> dict[str, Any]:
        return {
            "turn_id": list(self.turn_id.as_tuple()),
            "rank": rank,
            "raw_score": self.raw_score,
        }


@dataclass(frozen=True, slots=True)
class RankedLaneObservation:
    """One bounded ranked lane produced outside this offline package."""

    lane: RetrievalLane
    requested_depth: int
    query_sha256: str
    turn_corpus_projection_sha256: str
    identity: ExternalLaneIdentity
    candidates: tuple[RankedTurnObservation, ...]
    examined_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.lane, RetrievalLane):
            raise TurnRetrievalError("ranked observation lane must be lexical or dense")
        _positive_int(self.requested_depth, label="lane requested_depth")
        _sha256(self.query_sha256, label="lane query_sha256")
        _sha256(
            self.turn_corpus_projection_sha256,
            label="lane turn_corpus_projection_sha256",
        )
        if not isinstance(self.identity, ExternalLaneIdentity):
            raise TurnRetrievalError("lane identity must be ExternalLaneIdentity")
        if not isinstance(self.candidates, tuple):
            raise TurnRetrievalError("lane candidates must be an immutable tuple")
        if len(self.candidates) > self.requested_depth:
            raise TurnRetrievalError("lane candidates exceed the requested hard depth")
        seen: set[LongMemEvalTurnId] = set()
        for candidate in self.candidates:
            if not isinstance(candidate, RankedTurnObservation):
                raise TurnRetrievalError("lane candidates must be RankedTurnObservation values")
            if candidate.turn_id in seen:
                raise TurnRetrievalError("lane repeats a turn candidate")
            seen.add(candidate.turn_id)
        _nonnegative_int(self.examined_count, label="lane examined_count")
        if self.examined_count < len(self.candidates):
            raise TurnRetrievalError("lane examined_count cannot be smaller than returned count")


@dataclass(frozen=True, slots=True)
class TurnRetrievalProtocol:
    """Registered two-lane transfer protocol.

    The E1-A version is reserved: constructing it with any changed parameter
    fails closed.  A later held-out study can intentionally register different
    values under a new version instead of silently mutating E1-A.
    """

    protocol_version: str
    protocol_name: str
    cell: str
    lexical_depth: int
    dense_depth: int
    lexical_weight: float
    dense_weight: float
    rrf_k: int
    fused_candidate_cap: int

    def __post_init__(self) -> None:
        _identifier(self.protocol_version, label="protocol_version")
        _identifier(self.protocol_name, label="protocol_name")
        _identifier(self.cell, label="protocol cell")
        _positive_int(self.lexical_depth, label="lexical depth")
        _positive_int(self.dense_depth, label="dense depth")
        object.__setattr__(
            self,
            "lexical_weight",
            finite_score(self.lexical_weight, label="lexical weight"),
        )
        object.__setattr__(
            self,
            "dense_weight",
            finite_score(self.dense_weight, label="dense weight"),
        )
        if self.lexical_weight <= 0.0 or self.dense_weight <= 0.0:
            raise TurnRetrievalError("lane weights must be positive")
        _positive_int(self.rrf_k, label="RRF k")
        _positive_int(self.fused_candidate_cap, label="fused candidate cap")
        signature = (
            self.protocol_name,
            self.cell,
            self.lexical_depth,
            self.dense_depth,
            self.lexical_weight,
            self.dense_weight,
            self.rrf_k,
            self.fused_candidate_cap,
        )
        if self.protocol_version == E1A_PROTOCOL_VERSION and signature != _E1A_SIGNATURE:
            raise TurnRetrievalError(
                "the reserved E1-A protocol version cannot be used with changed parameters"
            )

    def lane_depth(self, lane: RetrievalLane) -> int:
        if lane is RetrievalLane.LEXICAL:
            return self.lexical_depth
        if lane is RetrievalLane.DENSE:
            return self.dense_depth
        raise TurnRetrievalError("protocol does not define the requested lane")

    def lane_weight(self, lane: RetrievalLane) -> float:
        if lane is RetrievalLane.LEXICAL:
            return self.lexical_weight
        if lane is RetrievalLane.DENSE:
            return self.dense_weight
        raise TurnRetrievalError("protocol does not define the requested lane")

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "protocol_name": self.protocol_name,
            "cell": self.cell,
            "classification": "evaluation-only-turn-transfer",
            "production_configuration": False,
            "lanes": {
                RetrievalLane.LEXICAL.value: {
                    "hard_depth": self.lexical_depth,
                    "weight": self.lexical_weight,
                },
                RetrievalLane.DENSE.value: {
                    "hard_depth": self.dense_depth,
                    "weight": self.dense_weight,
                },
            },
            "fusion": {
                "method": "weighted-reciprocal-rank-fusion",
                "rrf_k": self.rrf_k,
                "unique_candidate_cap": self.fused_candidate_cap,
                "tie_break": "canonical-turn-id-ascending",
            },
            "transfer_notes": {
                "weights_mirror": (
                    "interactive-general-production-policy"
                    if (self.lexical_weight, self.dense_weight) == (LEXICAL_WEIGHT, DENSE_WEIGHT)
                    else None
                ),
                "weights_match_interactive_general_policy": (
                    (self.lexical_weight, self.dense_weight) == (LEXICAL_WEIGHT, DENSE_WEIGHT)
                ),
                "lane_depths_are_evaluation_choices": True,
                "production_dense_serving_cap": PRODUCTION_DENSE_SERVING_CAP,
                "dense_depth_exceeds_current_production_cap": (
                    self.dense_depth > PRODUCTION_DENSE_SERVING_CAP
                ),
            },
            "evidence_boundary": {
                "lexical_scores": "externally-attested-unverified",
                "dense_vectors": "externally-attested-unverified",
                "dense_scores": "externally-attested-unverified",
                "model_database_network_calls_by_fusion": 0,
                "gold_fields_used_for_candidate_generation": False,
                "reader_or_judge_executed": False,
                "qa_improvement_proven": False,
            },
        }


E1A_PROTOCOL = TurnRetrievalProtocol(
    protocol_version=E1A_PROTOCOL_VERSION,
    protocol_name=E1A_PROTOCOL_NAME,
    cell="E1-A",
    lexical_depth=LEXICAL_LANE_DEPTH,
    dense_depth=DENSE_LANE_DEPTH,
    lexical_weight=LEXICAL_WEIGHT,
    dense_weight=DENSE_WEIGHT,
    rrf_k=RRF_K,
    fused_candidate_cap=FUSED_CANDIDATE_CAP,
)


__all__ = [
    "ARTIFACT_TYPE",
    "DENSE_LANE_DEPTH",
    "DENSE_WEIGHT",
    "E1A_PROTOCOL",
    "E1A_PROTOCOL_NAME",
    "E1A_PROTOCOL_VERSION",
    "FUSED_CANDIDATE_CAP",
    "ImmutableArtifactIdentity",
    "LEXICAL_LANE_DEPTH",
    "LEXICAL_WEIGHT",
    "PRODUCTION_DENSE_SERVING_CAP",
    "RRF_K",
    "RankedLaneObservation",
    "RankedTurnObservation",
    "RetrievalLane",
    "SCHEMA_VERSION",
    "TurnRetrievalError",
    "TurnRetrievalProtocol",
    "ExternalLaneIdentity",
    "canonical_json_bytes",
    "finite_score",
    "sha256_json",
]
