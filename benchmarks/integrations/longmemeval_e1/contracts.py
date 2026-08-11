"""Frozen contracts for the offline LongMemEval E1 selection cells.

The contracts deliberately carry only immutable IDs, numeric observations,
and digests.  They do not execute or verify the external scorers that produced
those observations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from benchmarks.integrations.longmemeval_turn_retrieval import (
    E1A_PROTOCOL_VERSION,
    FUSED_CANDIDATE_CAP,
)
from benchmarks.integrations.longmemeval_turns import LongMemEvalTurnId

ARTIFACT_TYPE = "swarmbrain-longmemeval-e1-selection-trace"
SCHEMA_VERSION = 1

E1_PROTOCOL_NAME = "Swarm Brain LongMemEval SmartSearch-shaped E1 selection"
E1_PROTOCOL_VERSION = "swarmbrain-longmemeval-smartsearch-shaped-e1-v1"

E1_POOL_CAP = FUSED_CANDIDATE_CAP
E1C_RRF_K = 60
E1C_CROSS_ENCODER_WEIGHT = 0.7
E1C_COLBERT_WEIGHT = 0.3
E1D_FUSED_HEAD = 60
E1D_CE_RELATIVE_THRESHOLD = 0.03

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RESERVED_V1_SIGNATURE = (
    E1_PROTOCOL_NAME,
    E1A_PROTOCOL_VERSION,
    E1_POOL_CAP,
    E1C_RRF_K,
    E1C_CROSS_ENCODER_WEIGHT,
    E1C_COLBERT_WEIGHT,
    E1D_FUSED_HEAD,
    E1D_CE_RELATIVE_THRESHOLD,
)


class E1SelectionError(ValueError):
    """The supplied evidence cannot support a deterministic E1 result."""


class E1Cell(StrEnum):
    CROSS_ENCODER = "E1-B"
    CROSS_ENCODER_COLBERT = "E1-C"
    ADAPTIVE_THRESHOLD = "E1-D"


class ScoreChannel(StrEnum):
    CROSS_ENCODER_LOGIT = "cross-encoder-logit"
    COLBERT_SCORE = "colbert-score"


def canonical_json_bytes(value: Any) -> bytes:
    """Encode finite canonical JSON without normalizing Unicode."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise E1SelectionError("E1 trace values must be finite canonical UTF-8 JSON") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise E1SelectionError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise E1SelectionError(f"{label} cannot have leading or trailing whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise E1SelectionError(f"{label} cannot contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise E1SelectionError(f"{label} must be valid UTF-8") from exc
    return value


def checked_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise E1SelectionError(f"{label} must be a lowercase hexadecimal SHA-256 digest")
    return value


def finite_score(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise E1SelectionError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise E1SelectionError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise E1SelectionError(f"{label} must be a finite number")
    return 0.0 if result == 0.0 else result


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise E1SelectionError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise E1SelectionError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class E1SelectionProtocol:
    """Registered E1 protocol; the public v1 identifier is drift-proof."""

    protocol_version: str
    protocol_name: str
    source_e1a_protocol_version: str
    source_pool_cap: int
    rrf_k: int
    cross_encoder_weight: float
    colbert_weight: float
    adaptive_fused_head: int
    adaptive_ce_relative_threshold: float

    def __post_init__(self) -> None:
        _identifier(self.protocol_version, label="E1 protocol_version")
        _identifier(self.protocol_name, label="E1 protocol_name")
        _identifier(
            self.source_e1a_protocol_version,
            label="E1 source_e1a_protocol_version",
        )
        _positive_int(self.source_pool_cap, label="E1 source pool cap")
        _positive_int(self.rrf_k, label="E1 RRF k")
        object.__setattr__(
            self,
            "cross_encoder_weight",
            finite_score(self.cross_encoder_weight, label="E1 CrossEncoder weight"),
        )
        object.__setattr__(
            self,
            "colbert_weight",
            finite_score(self.colbert_weight, label="E1 ColBERT weight"),
        )
        if self.cross_encoder_weight <= 0.0 or self.colbert_weight <= 0.0:
            raise E1SelectionError("E1 fusion weights must be positive")
        _positive_int(self.adaptive_fused_head, label="E1-D fused head")
        if self.adaptive_fused_head > self.source_pool_cap:
            raise E1SelectionError("E1-D fused head cannot exceed the fixed source pool cap")
        object.__setattr__(
            self,
            "adaptive_ce_relative_threshold",
            finite_score(
                self.adaptive_ce_relative_threshold,
                label="E1-D CE relative threshold",
            ),
        )
        if not 0.0 <= self.adaptive_ce_relative_threshold <= 1.0:
            raise E1SelectionError("E1-D CE relative threshold must be within [0, 1]")
        signature = (
            self.protocol_name,
            self.source_e1a_protocol_version,
            self.source_pool_cap,
            self.rrf_k,
            self.cross_encoder_weight,
            self.colbert_weight,
            self.adaptive_fused_head,
            self.adaptive_ce_relative_threshold,
        )
        if self.protocol_version == E1_PROTOCOL_VERSION and signature != _RESERVED_V1_SIGNATURE:
            raise E1SelectionError(
                "the reserved SmartSearch-shaped E1 v1 protocol cannot be used with changed parameters"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "protocol_name": self.protocol_name,
            "classification": "evaluation-only-paper-shaped-selection",
            "production_configuration": False,
            "source": {
                "cell": "E1-A",
                "protocol_version": self.source_e1a_protocol_version,
                "fixed_unique_pool_cap": self.source_pool_cap,
            },
            "cells": {
                E1Cell.CROSS_ENCODER.value: {
                    "method": "sigmoid-cross-encoder-logit-rank",
                    "coverage": "every-fixed-E1-A-pool-candidate",
                },
                E1Cell.CROSS_ENCODER_COLBERT.value: {
                    "method": "weighted-reciprocal-rank-fusion",
                    "rrf_k": self.rrf_k,
                    "weights": {
                        ScoreChannel.CROSS_ENCODER_LOGIT.value: self.cross_encoder_weight,
                        ScoreChannel.COLBERT_SCORE.value: self.colbert_weight,
                    },
                },
                E1Cell.ADAPTIVE_THRESHOLD.value: {
                    "source": "E1-C",
                    "fused_head": self.adaptive_fused_head,
                    "operation_order": ["take-E1-C-fused-head", "compute-head-max", "threshold"],
                    "keep_rule": (
                        "cross_encoder_sigmoid >= relative_threshold * "
                        "max_cross_encoder_sigmoid_within_E1-C-head"
                    ),
                    "relative_threshold": self.adaptive_ce_relative_threshold,
                    "threshold_denominator_scope": "E1-C-preselection-head-only",
                },
            },
            "tie_break": ["prior-E1-A-fused-rank", "canonical-turn-id"],
            "evidence_boundary": {
                "external_scorer_and_model_identity": "caller-attested-unverified",
                "model_artifacts_reopened_or_hashed_by_selection": False,
                "scores_recomputed_by_selection": False,
                "model_network_database_calls_by_selection": 0,
                "gold_fields_used": False,
                "prompt_packing_executed": False,
                "reader_or_judge_executed": False,
                "qa_improvement_proven": False,
            },
        }


E1_PROTOCOL = E1SelectionProtocol(
    protocol_version=E1_PROTOCOL_VERSION,
    protocol_name=E1_PROTOCOL_NAME,
    source_e1a_protocol_version=E1A_PROTOCOL_VERSION,
    source_pool_cap=E1_POOL_CAP,
    rrf_k=E1C_RRF_K,
    cross_encoder_weight=E1C_CROSS_ENCODER_WEIGHT,
    colbert_weight=E1C_COLBERT_WEIGHT,
    adaptive_fused_head=E1D_FUSED_HEAD,
    adaptive_ce_relative_threshold=E1D_CE_RELATIVE_THRESHOLD,
)


@dataclass(frozen=True, slots=True)
class ExternalScorerIdentity:
    """Caller-attested scorer/model identity; no field implies verification."""

    producer: str
    scorer: str
    model: str
    revision: str
    artifact_sha256: str
    observation_artifact_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.producer, label="score producer")
        _identifier(self.scorer, label="scorer")
        _identifier(self.model, label="model")
        _identifier(self.revision, label="model revision")
        checked_sha256(self.artifact_sha256, label="model artifact_sha256")
        checked_sha256(
            self.observation_artifact_sha256,
            label="observation artifact_sha256",
        )

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "producer": self.producer,
            "scorer": self.scorer,
            "model": self.model,
            "revision": self.revision,
            "artifact_sha256": self.artifact_sha256,
            "observation_artifact_sha256": self.observation_artifact_sha256,
            "identity_source": "caller-attested-unverified",
            "model_artifact_reopened_by_selection": False,
            "model_executed_by_selection": False,
        }


@dataclass(frozen=True, slots=True)
class TurnScoreObservation:
    turn_id: LongMemEvalTurnId
    raw_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.turn_id, LongMemEvalTurnId):
            raise E1SelectionError("score observation must use an exact LongMemEvalTurnId")
        object.__setattr__(
            self,
            "raw_score",
            finite_score(self.raw_score, label="turn raw score"),
        )

    def content_free_binding(self, *, observation_position: int) -> dict[str, Any]:
        return {
            "turn_id": list(self.turn_id.as_tuple()),
            "observation_position": observation_position,
            "raw_score": self.raw_score,
        }


@dataclass(frozen=True, slots=True)
class PoolScoreObservation:
    """Exact one-score-per-candidate evidence for a fixed E1-A pool."""

    channel: ScoreChannel
    question_id: str
    query_sha256: str
    turn_corpus_projection_sha256: str
    e1a_trace_sha256: str
    e1a_pool_sha256: str
    pool_count: int
    identity: ExternalScorerIdentity
    scores: tuple[TurnScoreObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.channel, ScoreChannel):
            raise E1SelectionError("score channel must be CrossEncoder-logit or ColBERT-score")
        _identifier(self.question_id, label="score question_id")
        checked_sha256(self.query_sha256, label="score query_sha256")
        checked_sha256(
            self.turn_corpus_projection_sha256,
            label="score turn_corpus_projection_sha256",
        )
        checked_sha256(self.e1a_trace_sha256, label="score E1-A trace_sha256")
        checked_sha256(self.e1a_pool_sha256, label="score E1-A pool_sha256")
        _nonnegative_int(self.pool_count, label="score pool_count")
        if not isinstance(self.identity, ExternalScorerIdentity):
            raise E1SelectionError("score identity must be ExternalScorerIdentity")
        if not isinstance(self.scores, tuple):
            raise E1SelectionError("scores must be an immutable tuple")
        if len(self.scores) != self.pool_count:
            raise E1SelectionError("score count must equal its attested pool_count")
        seen: set[LongMemEvalTurnId] = set()
        for item in self.scores:
            if not isinstance(item, TurnScoreObservation):
                raise E1SelectionError("scores must contain TurnScoreObservation values")
            if item.turn_id in seen:
                raise E1SelectionError("score observation repeats a turn candidate")
            seen.add(item.turn_id)

    def content_free_binding(self) -> dict[str, Any]:
        observed = [
            item.content_free_binding(observation_position=position)
            for position, item in enumerate(self.scores, start=1)
        ]
        canonical = sorted(
            ({"turn_id": item["turn_id"], "raw_score": item["raw_score"]} for item in observed),
            key=lambda item: tuple(item["turn_id"]),
        )
        return {
            "channel": self.channel.value,
            "question_id": self.question_id,
            "query_sha256": self.query_sha256,
            "turn_corpus_projection_sha256": self.turn_corpus_projection_sha256,
            "e1a_trace_sha256": self.e1a_trace_sha256,
            "e1a_pool_sha256": self.e1a_pool_sha256,
            "pool_count": self.pool_count,
            "identity": self.identity.content_free_binding(),
            "observed_order_sha256": sha256_json(observed),
            "canonical_scores_sha256": sha256_json(canonical),
            "input_observation_rows_stored_in_trace": False,
        }


__all__ = [
    "ARTIFACT_TYPE",
    "E1C_COLBERT_WEIGHT",
    "E1C_CROSS_ENCODER_WEIGHT",
    "E1C_RRF_K",
    "E1D_CE_RELATIVE_THRESHOLD",
    "E1D_FUSED_HEAD",
    "E1Cell",
    "E1_POOL_CAP",
    "E1_PROTOCOL",
    "E1_PROTOCOL_NAME",
    "E1_PROTOCOL_VERSION",
    "E1SelectionError",
    "E1SelectionProtocol",
    "ExternalScorerIdentity",
    "PoolScoreObservation",
    "SCHEMA_VERSION",
    "ScoreChannel",
    "TurnScoreObservation",
    "canonical_json_bytes",
    "checked_sha256",
    "finite_score",
    "sha256_json",
]
