"""Frozen, pure/offline contracts for the E6/SB-HMR-v1 representation cells."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from benchmarks.integrations.longmemeval_turns import (
    LongMemEvalTurnId,
    TurnProjection,
    TurnProjectionCorpus,
)

ARTIFACT_TYPE = "swarmbrain-longmemeval-harmonic-multikey-representation-trace"
SCHEMA_VERSION = 1
PROTOCOL_VERSION = "E6/SB-HMR-v1"
RRF_K = 60
KEY_FAMILY_DEPTH = 20
MAX_VALUES = 16384
MAX_HYDRATED_VALUES = 128
MAX_KEYS_PER_VALUE_PER_FAMILY = 32
MAX_TOTAL_DERIVED_KEYS = 524288
MAX_ADJACENCY_EDGES = 262144
MAX_EXPANSION_HOPS = 1
MAX_NEIGHBORS_PER_NODE = 20
SIMILARITY_EDGE_THRESHOLD = 0.8
EXTRACTOR_INPUT_FIELDS = ("source_value",)
GRAPH_INPUT_FIELDS = ("source_safe_navigation_index",)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OPAQUE_ID_RE = re.compile(r"(?P<prefix>raw|key|entity|receipt|edge):[0-9a-f]{64}")


class RepresentationError(ValueError):
    """Representation evidence is incomplete, inconsistent, or outside SB-HMR-v1."""


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
        raise RepresentationError("representation evidence must be finite UTF-8 JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RepresentationError(f"{label} must be a lowercase hexadecimal SHA-256 digest")
    return value


def _identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RepresentationError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise RepresentationError(f"{label} cannot have leading or trailing whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise RepresentationError(f"{label} cannot contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise RepresentationError(f"{label} must be valid UTF-8") from exc
    return value


def _opaque_id(value: Any, *, prefix: str | tuple[str, ...], label: str) -> str:
    if not isinstance(value, str):
        raise RepresentationError(f"{label} must be a typed opaque SHA-256 ID")
    match = _OPAQUE_ID_RE.fullmatch(value)
    allowed = (prefix,) if isinstance(prefix, str) else prefix
    if match is None or match.group("prefix") not in allowed:
        expected = "/".join(allowed)
        raise RepresentationError(f"{label} must use opaque {expected}:<sha256> form")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RepresentationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RepresentationError(f"{label} must be a non-negative integer")
    return value


def _finite_score(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RepresentationError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise RepresentationError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def _unit_score(value: Any, *, label: str) -> float:
    result = _finite_score(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise RepresentationError(f"{label} must be in [0, 1]")
    return result


def turn_id_payload(turn_id: LongMemEvalTurnId) -> list[str | int]:
    return list(turn_id.as_tuple())


def opaque_navigation_id(*, prefix: str, material: Any) -> str:
    """Create a typed local ID without exposing caller-controlled navigation text."""

    if prefix not in {"key", "entity", "receipt", "edge"}:
        raise RepresentationError("navigation ID prefix must be key/entity/receipt/edge")
    return f"{prefix}:{sha256_json({'prefix': prefix, 'material': material})}"


class RepresentationCell(StrEnum):
    RAW = "R0"
    RAW_MERGED_SFK = "R1"
    RAW_SEPARATE_SFK = "R2"
    ABSTRACTION_CUES = "R3"
    ENTITY_DIRECT = "R4"
    ENTITY_ONE_HOP = "R5"
    SIMILARITY_NEGATIVE = "R-neg"


class KeyFamily(StrEnum):
    RAW = "raw"
    MERGED_SFK = "merged-sfk"
    SUMMARY = "summary"
    FACT = "fact"
    KEYWORD = "keyword"
    PRIMARY_ABSTRACTION = "primary-abstraction"
    CUE_ANCHOR = "cue-anchor"
    ENTITY_DESCRIPTION = "entity-description"


CELL_KEY_FAMILIES = {
    RepresentationCell.RAW: (KeyFamily.RAW,),
    RepresentationCell.RAW_MERGED_SFK: (KeyFamily.RAW, KeyFamily.MERGED_SFK),
    RepresentationCell.RAW_SEPARATE_SFK: (
        KeyFamily.RAW,
        KeyFamily.SUMMARY,
        KeyFamily.FACT,
        KeyFamily.KEYWORD,
    ),
    RepresentationCell.ABSTRACTION_CUES: (
        KeyFamily.PRIMARY_ABSTRACTION,
        KeyFamily.CUE_ANCHOR,
    ),
    RepresentationCell.ENTITY_DIRECT: (KeyFamily.ENTITY_DESCRIPTION,),
    RepresentationCell.ENTITY_ONE_HOP: (KeyFamily.ENTITY_DESCRIPTION,),
    RepresentationCell.SIMILARITY_NEGATIVE: (KeyFamily.RAW,),
}


@dataclass(frozen=True, slots=True)
class CanonicalValue:
    """One immutable F0 turn used as both source and hydrated reader value."""

    turn: TurnProjection
    source_artifact_sha256: str
    projection_sha256: str
    question_value_count: int
    question_values_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.turn, TurnProjection):
            raise RepresentationError("canonical value must contain a TurnProjection")
        _sha256(self.source_artifact_sha256, label="canonical value source artifact")
        _sha256(self.projection_sha256, label="canonical value projection")
        _positive_int(self.question_value_count, label="canonical question value count")
        _sha256(self.question_values_sha256, label="canonical question values")

    @property
    def value_id(self) -> str:
        return self.turn.turn_id.canonical_id

    @property
    def question_id(self) -> str:
        return self.turn.turn_id.question_id

    @property
    def source_id(self) -> str:
        return self.turn.turn_id.canonical_id

    @property
    def source_version_sha256(self) -> str:
        return self.turn.source_turn.sha256

    @property
    def raw_value_sha256(self) -> str:
        return self.turn.serialized_document_utf8.sha256

    @property
    def raw_value_utf8_bytes(self) -> int:
        return self.turn.serialized_document_utf8.bytes

    @property
    def raw_value(self) -> str:
        return self.turn.serialized_text

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "value_id": self.value_id,
            "turn_id": turn_id_payload(self.turn.turn_id),
            "question_id": self.question_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "projection_sha256": self.projection_sha256,
            "question_value_count": self.question_value_count,
            "question_values_sha256": self.question_values_sha256,
            "source_id": self.source_id,
            "source_version_sha256": self.source_version_sha256,
            "source_record_sha256": self.turn.source_record.sha256,
            "raw_value": {
                "utf8_bytes": self.raw_value_utf8_bytes,
                "sha256": self.raw_value_sha256,
            },
        }


def question_value_binding_payload(turns: tuple[TurnProjection, ...]) -> list[dict[str, Any]]:
    return [
        {
            "turn_id": turn_id_payload(turn.turn_id),
            "source_version_sha256": turn.source_turn.sha256,
            "raw_value_sha256": turn.serialized_document_utf8.sha256,
            "raw_value_utf8_bytes": turn.serialized_document_utf8.bytes,
        }
        for turn in turns
    ]


def compile_question_canonical_values(
    corpus: TurnProjectionCorpus,
    *,
    question_id: str,
) -> tuple[CanonicalValue, ...]:
    """Project the complete authoritative F0 question corpus into E6 values."""

    if not isinstance(corpus, TurnProjectionCorpus):
        raise RepresentationError("canonical values require TurnProjectionCorpus")
    _identifier(question_id, label="canonical value question_id")
    turns = tuple(turn for turn in corpus.turns if turn.turn_id.question_id == question_id)
    question = next(
        (item for item in corpus.questions if item.question_id == question_id),
        None,
    )
    if question is None or not turns:
        raise RepresentationError("question ID is absent from the F0 turn corpus")
    if len(turns) != question.turns:
        raise RepresentationError("F0 question binding and turn count disagree")
    digest = sha256_json(question_value_binding_payload(turns))
    return tuple(
        CanonicalValue(
            turn=turn,
            source_artifact_sha256=corpus.source_artifact.sha256,
            projection_sha256=corpus.projection_sha256,
            question_value_count=len(turns),
            question_values_sha256=digest,
        )
        for turn in turns
    )


@dataclass(frozen=True, slots=True)
class ExtractorIdentity:
    """Caller-attested immutable identity for one derived-key extractor."""

    producer: str
    protocol: str
    model_id: str
    model_revision: str
    deployment_id: str
    model_artifact_sha256: str
    prompt_sha256: str
    identity_artifact_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.producer, label="extractor producer")
        _identifier(self.protocol, label="extractor protocol")
        _identifier(self.model_id, label="extractor model_id")
        _identifier(self.model_revision, label="extractor model_revision")
        _identifier(self.deployment_id, label="extractor deployment_id")
        _sha256(self.model_artifact_sha256, label="extractor model artifact")
        _sha256(self.prompt_sha256, label="extractor prompt")
        _sha256(self.identity_artifact_sha256, label="extractor identity artifact")

    @property
    def identity_sha256(self) -> str:
        return sha256_json(self.binding_without_digest())

    def binding_without_digest(self) -> dict[str, str]:
        return {
            "producer": self.producer,
            "protocol": self.protocol,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "deployment_id": self.deployment_id,
            "model_artifact_sha256": self.model_artifact_sha256,
            "prompt_sha256": self.prompt_sha256,
            "identity_artifact_sha256": self.identity_artifact_sha256,
            "identity_source": "externally-attested-unverified",
        }

    def content_free_binding(self) -> dict[str, str]:
        return {**self.binding_without_digest(), "identity_sha256": self.identity_sha256}


@dataclass(frozen=True, slots=True)
class DerivedKey:
    """One exact derived navigation key; its text never enters a trace."""

    key_id: str
    family: KeyFamily
    source_value_id: str
    question_id: str
    source_artifact_sha256: str
    projection_sha256: str
    source_version_sha256: str
    raw_value_sha256: str
    key_text: str
    key_text_sha256: str
    key_text_utf8_bytes: int
    construction_receipt_id: str
    entity_id: str | None = None

    def __post_init__(self) -> None:
        _opaque_id(self.key_id, prefix="key", label="derived key_id")
        if not isinstance(self.family, KeyFamily) or self.family is KeyFamily.RAW:
            raise RepresentationError("DerivedKey family must be a non-raw KeyFamily")
        _identifier(self.source_value_id, label="derived key source_value_id")
        _identifier(self.question_id, label="derived key question_id")
        _sha256(self.source_artifact_sha256, label="derived key source artifact")
        _sha256(self.projection_sha256, label="derived key projection")
        _sha256(self.source_version_sha256, label="derived key source version")
        _sha256(self.raw_value_sha256, label="derived key raw value")
        if not isinstance(self.key_text, str) or not self.key_text:
            raise RepresentationError("derived key text must be non-empty")
        try:
            encoded = self.key_text.encode("utf-8")
        except UnicodeError as exc:
            raise RepresentationError("derived key text must be valid UTF-8") from exc
        if self.key_text_sha256 != sha256_bytes(encoded):
            raise RepresentationError("derived key text digest does not match exact text")
        if self.key_text_utf8_bytes != len(encoded):
            raise RepresentationError("derived key byte count does not match exact text")
        _opaque_id(
            self.construction_receipt_id,
            prefix="receipt",
            label="derived key construction receipt",
        )
        if self.family is KeyFamily.ENTITY_DESCRIPTION:
            _opaque_id(
                self.entity_id,
                prefix="entity",
                label="entity-description key entity_id",
            )
        elif self.entity_id is not None:
            raise RepresentationError("only entity-description keys may carry entity_id")

    @classmethod
    def create(
        cls,
        *,
        key_id: str,
        family: KeyFamily,
        source: CanonicalValue,
        key_text: str,
        construction_receipt_id: str,
        entity_id: str | None = None,
    ) -> DerivedKey:
        encoded = key_text.encode("utf-8")
        return cls(
            key_id=key_id,
            family=family,
            source_value_id=source.value_id,
            question_id=source.question_id,
            source_artifact_sha256=source.source_artifact_sha256,
            projection_sha256=source.projection_sha256,
            source_version_sha256=source.source_version_sha256,
            raw_value_sha256=source.raw_value_sha256,
            key_text=key_text,
            key_text_sha256=sha256_bytes(encoded),
            key_text_utf8_bytes=len(encoded),
            construction_receipt_id=construction_receipt_id,
            entity_id=entity_id,
        )

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "family": self.family.value,
            "source_value_id": self.source_value_id,
            "question_id": self.question_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "projection_sha256": self.projection_sha256,
            "source_version_sha256": self.source_version_sha256,
            "raw_value_sha256": self.raw_value_sha256,
            "key_text": {
                "utf8_bytes": self.key_text_utf8_bytes,
                "sha256": self.key_text_sha256,
            },
            "construction_receipt_id": self.construction_receipt_id,
            "entity_id": self.entity_id,
        }


def derived_key_output_binding(keys: tuple[DerivedKey, ...]) -> list[dict[str, Any]]:
    return [
        {
            "key_id": key.key_id,
            "family": key.family.value,
            "key_text_sha256": key.key_text_sha256,
            "key_text_utf8_bytes": key.key_text_utf8_bytes,
            "entity_id": key.entity_id,
        }
        for key in keys
    ]


@dataclass(frozen=True, slots=True)
class ConstructionReceipt:
    """Complete external construction accounting for one value/family invocation."""

    receipt_id: str
    family: KeyFamily
    source_value_id: str
    question_id: str
    source_artifact_sha256: str
    projection_sha256: str
    source_version_sha256: str
    raw_value_sha256: str
    raw_value_utf8_bytes: int
    extractor: ExtractorIdentity
    construction_artifact_sha256: str
    input_fields: tuple[str, ...]
    source_input_sha256: str
    request_sha256: str
    response_sha256: str
    output_key_ids: tuple[str, ...]
    output_keys_sha256: str
    input_tokens: int
    output_tokens: int
    latency_microseconds: int
    cost_microusd: int
    retry_count: int = 0
    cache_hit: bool = False
    complete: bool = True

    def __post_init__(self) -> None:
        _opaque_id(
            self.receipt_id,
            prefix="receipt",
            label="construction receipt_id",
        )
        if not isinstance(self.family, KeyFamily) or self.family is KeyFamily.RAW:
            raise RepresentationError("construction receipt family must be derived")
        _identifier(self.source_value_id, label="construction source_value_id")
        _identifier(self.question_id, label="construction question_id")
        _sha256(self.source_artifact_sha256, label="construction source artifact")
        _sha256(self.projection_sha256, label="construction projection")
        _sha256(self.source_version_sha256, label="construction source version")
        _sha256(self.raw_value_sha256, label="construction raw value")
        _positive_int(self.raw_value_utf8_bytes, label="construction raw value bytes")
        if not isinstance(self.extractor, ExtractorIdentity):
            raise RepresentationError("construction receipt requires ExtractorIdentity")
        _sha256(self.construction_artifact_sha256, label="construction artifact")
        if self.input_fields != EXTRACTOR_INPUT_FIELDS:
            raise RepresentationError(
                "extractor input fields must contain only the canonical source value"
            )
        if self.source_input_sha256 != self.raw_value_sha256:
            raise RepresentationError(
                "extractor source-input digest must equal the immutable raw value digest"
            )
        if self.request_sha256 != extraction_request_sha256(
            family=self.family,
            raw_value_sha256=self.raw_value_sha256,
            raw_value_utf8_bytes=self.raw_value_utf8_bytes,
            extractor=self.extractor,
            input_fields=self.input_fields,
        ):
            raise RepresentationError(
                "construction request digest does not match source-only request material"
            )
        _sha256(self.response_sha256, label="construction response")
        if not isinstance(self.output_key_ids, tuple):
            raise RepresentationError("construction output IDs must be an immutable tuple")
        for key_id in self.output_key_ids:
            _opaque_id(key_id, prefix="key", label="construction output key_id")
        if len(set(self.output_key_ids)) != len(self.output_key_ids):
            raise RepresentationError("construction receipt repeats an output key ID")
        _sha256(self.output_keys_sha256, label="construction output key binding")
        _nonnegative_int(self.input_tokens, label="construction input_tokens")
        _nonnegative_int(self.output_tokens, label="construction output_tokens")
        _nonnegative_int(self.latency_microseconds, label="construction latency")
        _nonnegative_int(self.cost_microusd, label="construction cost")
        _nonnegative_int(self.retry_count, label="construction retry_count")
        if not isinstance(self.cache_hit, bool):
            raise RepresentationError("construction cache_hit must be boolean")
        if self.complete is not True:
            raise RepresentationError("partial construction receipts are forbidden")

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "family": self.family.value,
            "source_value_id": self.source_value_id,
            "question_id": self.question_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "projection_sha256": self.projection_sha256,
            "source_version_sha256": self.source_version_sha256,
            "raw_value_sha256": self.raw_value_sha256,
            "raw_value_utf8_bytes": self.raw_value_utf8_bytes,
            "extractor": self.extractor.content_free_binding(),
            "construction_artifact_sha256": self.construction_artifact_sha256,
            "input_contract": {
                "fields": list(self.input_fields),
                "question_id_is_routing_metadata_not_extractor_input": True,
                "source_input_sha256": self.source_input_sha256,
                "forbidden_fields": [
                    "question",
                    "question_type",
                    "answer",
                    "gold_session_ids",
                    "judge_label",
                ],
            },
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "output_key_ids": list(self.output_key_ids),
            "output_keys_sha256": self.output_keys_sha256,
            "accounting": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "latency_microseconds": self.latency_microseconds,
                "cost_microusd": self.cost_microusd,
                "retry_count": self.retry_count,
                "cache_hit": self.cache_hit,
            },
            "complete": self.complete,
        }


def extraction_request_sha256(
    *,
    family: KeyFamily,
    raw_value_sha256: str,
    raw_value_utf8_bytes: int,
    extractor: ExtractorIdentity,
    input_fields: tuple[str, ...] = EXTRACTOR_INPUT_FIELDS,
) -> str:
    """Digest the only request material permitted by the frozen extraction seam."""

    if not isinstance(family, KeyFamily) or family is KeyFamily.RAW:
        raise RepresentationError("extraction request family must be derived")
    _sha256(raw_value_sha256, label="extraction request raw value")
    _positive_int(raw_value_utf8_bytes, label="extraction request raw value bytes")
    if not isinstance(extractor, ExtractorIdentity):
        raise RepresentationError("extraction request requires ExtractorIdentity")
    if input_fields != EXTRACTOR_INPUT_FIELDS:
        raise RepresentationError("extraction request input allowlist is not source-only")
    return sha256_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "family": family.value,
            "input_fields": list(input_fields),
            "source_value": {
                "sha256": raw_value_sha256,
                "utf8_bytes": raw_value_utf8_bytes,
            },
            "extractor_identity_sha256": extractor.identity_sha256,
            "extractor_prompt_sha256": extractor.prompt_sha256,
        }
    )


def raw_key_id(value: CanonicalValue) -> str:
    return f"raw:{sha256_json({'value_id': value.value_id, 'sha256': value.raw_value_sha256})}"


@dataclass(frozen=True, slots=True)
class IndexedKeyBinding:
    """Content-free index view shared by raw and derived key rankings."""

    key_id: str
    family: KeyFamily
    source_value_id: str
    question_id: str
    source_artifact_sha256: str
    projection_sha256: str
    source_version_sha256: str
    raw_value_sha256: str
    key_text_sha256: str
    key_text_utf8_bytes: int
    entity_id: str | None
    construction_receipt_id: str | None

    def __post_init__(self) -> None:
        _opaque_id(self.key_id, prefix=("raw", "key"), label="indexed key_id")
        if not isinstance(self.family, KeyFamily):
            raise RepresentationError("indexed key family must be KeyFamily")
        _identifier(self.source_value_id, label="indexed key source_value_id")
        _identifier(self.question_id, label="indexed key question_id")
        _sha256(self.source_artifact_sha256, label="indexed key source artifact")
        _sha256(self.projection_sha256, label="indexed key projection")
        _sha256(self.source_version_sha256, label="indexed key source version")
        _sha256(self.raw_value_sha256, label="indexed key raw value")
        _sha256(self.key_text_sha256, label="indexed key text")
        _positive_int(self.key_text_utf8_bytes, label="indexed key UTF-8 bytes")
        if self.family is KeyFamily.ENTITY_DESCRIPTION:
            _opaque_id(self.entity_id, prefix="entity", label="indexed entity_id")
        elif self.entity_id is not None:
            raise RepresentationError("only entity-description indexed keys carry entity_id")
        if self.family is KeyFamily.RAW:
            if self.construction_receipt_id is not None:
                raise RepresentationError("raw indexed keys cannot have construction receipts")
        else:
            _opaque_id(
                self.construction_receipt_id,
                prefix="receipt",
                label="indexed key construction receipt",
            )

    @classmethod
    def from_value(cls, value: CanonicalValue) -> IndexedKeyBinding:
        return cls(
            key_id=raw_key_id(value),
            family=KeyFamily.RAW,
            source_value_id=value.value_id,
            question_id=value.question_id,
            source_artifact_sha256=value.source_artifact_sha256,
            projection_sha256=value.projection_sha256,
            source_version_sha256=value.source_version_sha256,
            raw_value_sha256=value.raw_value_sha256,
            key_text_sha256=value.raw_value_sha256,
            key_text_utf8_bytes=value.raw_value_utf8_bytes,
            entity_id=None,
            construction_receipt_id=None,
        )

    @classmethod
    def from_derived(cls, key: DerivedKey) -> IndexedKeyBinding:
        return cls(
            key_id=key.key_id,
            family=key.family,
            source_value_id=key.source_value_id,
            question_id=key.question_id,
            source_artifact_sha256=key.source_artifact_sha256,
            projection_sha256=key.projection_sha256,
            source_version_sha256=key.source_version_sha256,
            raw_value_sha256=key.raw_value_sha256,
            key_text_sha256=key.key_text_sha256,
            key_text_utf8_bytes=key.key_text_utf8_bytes,
            entity_id=key.entity_id,
            construction_receipt_id=key.construction_receipt_id,
        )

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "family": self.family.value,
            "source_value_id": self.source_value_id,
            "question_id": self.question_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "projection_sha256": self.projection_sha256,
            "source_version_sha256": self.source_version_sha256,
            "raw_value_sha256": self.raw_value_sha256,
            "key_text_sha256": self.key_text_sha256,
            "key_text_utf8_bytes": self.key_text_utf8_bytes,
            "entity_id": self.entity_id,
            "construction_receipt_id": self.construction_receipt_id,
        }


@dataclass(frozen=True, slots=True)
class RepresentationCorpus:
    """One bounded question-local source value and derived-navigation corpus."""

    projection_corpus: TurnProjectionCorpus
    values: tuple[CanonicalValue, ...]
    derived_keys: tuple[DerivedKey, ...]
    construction_receipts: tuple[ConstructionReceipt, ...]
    _all_keys_cache: tuple[IndexedKeyBinding, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _index_sha256_cache: str = field(init=False, repr=False, compare=False)
    _navigation_index_sha256_cache: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.projection_corpus, TurnProjectionCorpus):
            raise RepresentationError(
                "representation corpus requires the authoritative F0 TurnProjectionCorpus"
            )
        if not isinstance(self.values, tuple) or not self.values:
            raise RepresentationError("representation values must be a non-empty tuple")
        if len(self.values) > MAX_VALUES:
            raise RepresentationError(f"representation corpus exceeds {MAX_VALUES} values")
        if any(not isinstance(value, CanonicalValue) for value in self.values):
            raise RepresentationError("every representation value must be CanonicalValue")
        if not isinstance(self.derived_keys, tuple):
            raise RepresentationError("derived keys must be an immutable tuple")
        if len(self.derived_keys) > MAX_TOTAL_DERIVED_KEYS:
            raise RepresentationError("representation corpus exceeds the derived-key bound")
        if any(not isinstance(key, DerivedKey) for key in self.derived_keys):
            raise RepresentationError("every derived key must be DerivedKey")
        if not isinstance(self.construction_receipts, tuple):
            raise RepresentationError("construction receipts must be an immutable tuple")
        if any(
            not isinstance(receipt, ConstructionReceipt) for receipt in self.construction_receipts
        ):
            raise RepresentationError("every construction receipt must be ConstructionReceipt")

        value_by_id: dict[str, CanonicalValue] = {}
        question_ids: set[str] = set()
        source_artifacts: set[str] = set()
        projections: set[str] = set()
        question_value_counts: set[int] = set()
        question_value_digests: set[str] = set()
        for value in self.values:
            if value.value_id in value_by_id:
                raise RepresentationError("representation corpus repeats a canonical value ID")
            value_by_id[value.value_id] = value
            question_ids.add(value.question_id)
            source_artifacts.add(value.source_artifact_sha256)
            projections.add(value.projection_sha256)
            question_value_counts.add(value.question_value_count)
            question_value_digests.add(value.question_values_sha256)
        if len(question_ids) != 1:
            raise RepresentationError("representation values cross question boundaries")
        if len(source_artifacts) != 1 or len(projections) != 1:
            raise RepresentationError("representation values cross source/projection boundaries")
        if next(iter(source_artifacts)) != self.projection_corpus.source_artifact.sha256:
            raise RepresentationError(
                "representation values do not match the authoritative F0 source artifact"
            )
        if next(iter(projections)) != self.projection_corpus.projection_sha256:
            raise RepresentationError(
                "representation values do not match the authoritative F0 projection"
            )
        if len(question_value_counts) != 1 or len(question_value_digests) != 1:
            raise RepresentationError(
                "representation values disagree on the authoritative question corpus binding"
            )
        expected_count = next(iter(question_value_counts))
        expected_digest = next(iter(question_value_digests))
        if len(self.values) != expected_count:
            raise RepresentationError(
                "representation corpus is a partial question-local F0 value set"
            )
        turns = tuple(value.turn for value in self.values)
        if sha256_json(question_value_binding_payload(turns)) != expected_digest:
            raise RepresentationError(
                "representation value order/content differs from the authoritative F0 binding"
            )
        authoritative_turns = tuple(
            turn
            for turn in self.projection_corpus.turns
            if turn.turn_id.question_id == self.values[0].question_id
        )
        authoritative_question = next(
            (
                question
                for question in self.projection_corpus.questions
                if question.question_id == self.values[0].question_id
            ),
            None,
        )
        if (
            authoritative_question is None
            or len(authoritative_turns) != authoritative_question.turns
        ):
            raise RepresentationError(
                "authoritative F0 question binding is missing or inconsistent"
            )
        if turns != authoritative_turns:
            raise RepresentationError(
                "representation corpus is not the exact authoritative question-local F0 slice"
            )

        receipt_by_id: dict[str, ConstructionReceipt] = {}
        receipt_by_value_family: set[tuple[str, KeyFamily]] = set()
        for receipt in self.construction_receipts:
            if receipt.receipt_id in receipt_by_id:
                raise RepresentationError("representation corpus repeats a construction receipt")
            source = value_by_id.get(receipt.source_value_id)
            if source is None:
                raise RepresentationError("construction receipt refers to an unknown value")
            _validate_source_reference(
                source,
                question_id=receipt.question_id,
                source_artifact_sha256=receipt.source_artifact_sha256,
                projection_sha256=receipt.projection_sha256,
                source_version_sha256=receipt.source_version_sha256,
                raw_value_sha256=receipt.raw_value_sha256,
                label="construction receipt",
            )
            if receipt.raw_value_utf8_bytes != source.raw_value_utf8_bytes:
                raise RepresentationError(
                    "construction receipt raw value byte count is stale or tampered"
                )
            key = (receipt.source_value_id, receipt.family)
            if key in receipt_by_value_family:
                raise RepresentationError(
                    "representation corpus repeats a value/family construction receipt"
                )
            receipt_by_value_family.add(key)
            receipt_by_id[receipt.receipt_id] = receipt

        key_by_id: dict[str, DerivedKey] = {}
        raw_key_ids = {raw_key_id(value) for value in self.values}
        family_value_counts: dict[tuple[KeyFamily, str], int] = {}
        keys_by_receipt: dict[str, list[DerivedKey]] = {
            receipt_id: [] for receipt_id in receipt_by_id
        }
        for key in self.derived_keys:
            if key.key_id in key_by_id or key.key_id in raw_key_ids:
                raise RepresentationError("representation corpus repeats a key ID")
            source = value_by_id.get(key.source_value_id)
            if source is None:
                raise RepresentationError("derived key refers to an unknown source value")
            _validate_source_reference(
                source,
                question_id=key.question_id,
                source_artifact_sha256=key.source_artifact_sha256,
                projection_sha256=key.projection_sha256,
                source_version_sha256=key.source_version_sha256,
                raw_value_sha256=key.raw_value_sha256,
                label="derived key",
            )
            receipt = receipt_by_id.get(key.construction_receipt_id)
            if receipt is None:
                raise RepresentationError("derived key has no complete construction receipt")
            if receipt.family is not key.family or receipt.source_value_id != key.source_value_id:
                raise RepresentationError("derived key and construction receipt disagree")
            family_value = (key.family, key.source_value_id)
            family_value_counts[family_value] = family_value_counts.get(family_value, 0) + 1
            if family_value_counts[family_value] > MAX_KEYS_PER_VALUE_PER_FAMILY:
                raise RepresentationError("derived key fan-out exceeds the frozen family bound")
            key_by_id[key.key_id] = key
            keys_by_receipt[receipt.receipt_id].append(key)

        for receipt in self.construction_receipts:
            keys = tuple(keys_by_receipt[receipt.receipt_id])
            if tuple(key.key_id for key in keys) != receipt.output_key_ids:
                raise RepresentationError(
                    "construction receipt output IDs do not match exact derived keys"
                )
            if sha256_json(derived_key_output_binding(keys)) != receipt.output_keys_sha256:
                raise RepresentationError(
                    "construction receipt output digest does not match exact derived keys"
                )
        raw_keys = tuple(IndexedKeyBinding.from_value(value) for value in self.values)
        derived_keys = tuple(IndexedKeyBinding.from_derived(key) for key in self.derived_keys)
        all_keys = (*raw_keys, *derived_keys)
        object.__setattr__(self, "_all_keys_cache", all_keys)
        object.__setattr__(
            self,
            "_index_sha256_cache",
            sha256_json(
                {
                    "values": [value.content_free_binding() for value in self.values],
                    "keys": [key.content_free_binding() for key in all_keys],
                    "construction_receipts": [
                        receipt.content_free_binding() for receipt in self.construction_receipts
                    ],
                }
            ),
        )
        object.__setattr__(
            self,
            "_navigation_index_sha256_cache",
            sha256_json(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "classification": "source-only-navigation-index",
                    "values": [
                        {
                            "value_id": value.value_id,
                            "source_version_sha256": value.source_version_sha256,
                            "raw_value_sha256": value.raw_value_sha256,
                            "raw_value_utf8_bytes": value.raw_value_utf8_bytes,
                        }
                        for value in self.values
                    ],
                    "keys": [
                        {
                            "key_id": key.key_id,
                            "family": key.family.value,
                            "source_value_id": key.source_value_id,
                            "source_version_sha256": key.source_version_sha256,
                            "raw_value_sha256": key.raw_value_sha256,
                            "key_text_sha256": key.key_text_sha256,
                            "key_text_utf8_bytes": key.key_text_utf8_bytes,
                            "entity_id": key.entity_id,
                        }
                        for key in all_keys
                    ],
                }
            ),
        )

    @property
    def question_id(self) -> str:
        return self.values[0].question_id

    @property
    def source_artifact_sha256(self) -> str:
        return self.values[0].source_artifact_sha256

    @property
    def projection_sha256(self) -> str:
        return self.values[0].projection_sha256

    @property
    def all_keys(self) -> tuple[IndexedKeyBinding, ...]:
        return self._all_keys_cache

    def values_by_id(self) -> dict[str, CanonicalValue]:
        return {value.value_id: value for value in self.values}

    def keys_by_id(self) -> dict[str, IndexedKeyBinding]:
        return {key.key_id: key for key in self.all_keys}

    def keys_for_family(self, family: KeyFamily) -> tuple[IndexedKeyBinding, ...]:
        return tuple(key for key in self.all_keys if key.family is family)

    def receipts_for_family(self, family: KeyFamily) -> tuple[ConstructionReceipt, ...]:
        return tuple(receipt for receipt in self.construction_receipts if receipt.family is family)

    @property
    def index_sha256(self) -> str:
        return self._index_sha256_cache

    @property
    def navigation_index_sha256(self) -> str:
        """Gold-insensitive value/key material permitted as graph-construction input."""

        return self._navigation_index_sha256_cache


def _validate_source_reference(
    source: CanonicalValue,
    *,
    question_id: str,
    source_artifact_sha256: str,
    projection_sha256: str,
    source_version_sha256: str,
    raw_value_sha256: str,
    label: str,
) -> None:
    if question_id != source.question_id:
        raise RepresentationError(f"{label} crosses the question boundary")
    if source_artifact_sha256 != source.source_artifact_sha256:
        raise RepresentationError(f"{label} crosses the source-artifact boundary")
    if projection_sha256 != source.projection_sha256:
        raise RepresentationError(f"{label} crosses the projection boundary")
    if source_version_sha256 != source.source_version_sha256:
        raise RepresentationError(f"{label} source version is stale or tampered")
    if raw_value_sha256 != source.raw_value_sha256:
        raise RepresentationError(f"{label} raw value hash is stale or tampered")


@dataclass(frozen=True, slots=True)
class ScorerIdentity:
    producer: str
    protocol: str
    model_id: str
    model_revision: str
    model_artifact_sha256: str
    identity_artifact_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.producer, label="scorer producer")
        _identifier(self.protocol, label="scorer protocol")
        _identifier(self.model_id, label="scorer model_id")
        _identifier(self.model_revision, label="scorer model_revision")
        _sha256(self.model_artifact_sha256, label="scorer model artifact")
        _sha256(self.identity_artifact_sha256, label="scorer identity artifact")

    @property
    def identity_sha256(self) -> str:
        return sha256_json(self.binding_without_digest())

    def binding_without_digest(self) -> dict[str, str]:
        return {
            "producer": self.producer,
            "protocol": self.protocol,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_artifact_sha256": self.model_artifact_sha256,
            "identity_artifact_sha256": self.identity_artifact_sha256,
            "identity_source": "externally-attested-unverified",
        }

    def content_free_binding(self) -> dict[str, str]:
        return {**self.binding_without_digest(), "identity_sha256": self.identity_sha256}


@dataclass(frozen=True, slots=True)
class RankedKeyScore:
    key_id: str
    raw_score: float

    def __post_init__(self) -> None:
        _opaque_id(self.key_id, prefix=("raw", "key"), label="ranked key_id")
        object.__setattr__(self, "raw_score", _finite_score(self.raw_score, label="key score"))

    def content_free_binding(self, *, rank: int) -> dict[str, Any]:
        return {"key_id": self.key_id, "rank": rank, "raw_score": self.raw_score}


@dataclass(frozen=True, slots=True)
class RankedFamilyObservation:
    """Complete, externally attested top-depth ranking for one exact key family."""

    family: KeyFamily
    question_id: str
    source_artifact_sha256: str
    projection_sha256: str
    index_sha256: str
    query_sha256: str
    scorer: ScorerIdentity
    observation_artifact_sha256: str
    requested_depth: int
    indexed_key_count: int
    examined_key_count: int
    ranked_keys: tuple[RankedKeyScore, ...]
    observation_sha256: str
    complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.family, KeyFamily):
            raise RepresentationError("ranking family must be KeyFamily")
        _identifier(self.question_id, label="ranking question_id")
        _sha256(self.source_artifact_sha256, label="ranking source artifact")
        _sha256(self.projection_sha256, label="ranking projection")
        _sha256(self.index_sha256, label="ranking index")
        _sha256(self.query_sha256, label="ranking query")
        if not isinstance(self.scorer, ScorerIdentity):
            raise RepresentationError("ranking observation requires ScorerIdentity")
        _sha256(self.observation_artifact_sha256, label="ranking observation artifact")
        if self.requested_depth != KEY_FAMILY_DEPTH:
            raise RepresentationError(
                f"ranking requested_depth must be frozen at {KEY_FAMILY_DEPTH}"
            )
        _nonnegative_int(self.indexed_key_count, label="ranking indexed_key_count")
        _nonnegative_int(self.examined_key_count, label="ranking examined_key_count")
        if self.examined_key_count != self.indexed_key_count:
            raise RepresentationError("partial key-family ranking observations are forbidden")
        if not isinstance(self.ranked_keys, tuple):
            raise RepresentationError("ranked keys must be an immutable tuple")
        if len(self.ranked_keys) != min(self.requested_depth, self.indexed_key_count):
            raise RepresentationError("ranking does not contain the exact frozen top depth")
        if any(not isinstance(item, RankedKeyScore) for item in self.ranked_keys):
            raise RepresentationError("ranking entries must be RankedKeyScore")
        ids = [item.key_id for item in self.ranked_keys]
        if len(set(ids)) != len(ids):
            raise RepresentationError("ranking repeats a key ID")
        expected_order = sorted(self.ranked_keys, key=lambda item: (-item.raw_score, item.key_id))
        if list(self.ranked_keys) != expected_order:
            raise RepresentationError("ranking order must be score-descending then key-ID")
        if self.complete is not True:
            raise RepresentationError("partial ranking receipts are forbidden")
        _sha256(self.observation_sha256, label="ranking observation digest")
        if self.observation_sha256 != sha256_json(self.payload_without_digest()):
            raise RepresentationError("ranking observation digest does not match its payload")

    @classmethod
    def create(
        cls,
        *,
        family: KeyFamily,
        corpus: RepresentationCorpus,
        query_sha256: str,
        scorer: ScorerIdentity,
        observation_artifact_sha256: str,
        ranked_keys: tuple[RankedKeyScore, ...],
    ) -> RankedFamilyObservation:
        indexed_count = len(corpus.keys_for_family(family))
        values = {
            "family": family.value,
            "question_id": corpus.question_id,
            "source_artifact_sha256": corpus.source_artifact_sha256,
            "projection_sha256": corpus.projection_sha256,
            "index_sha256": corpus.index_sha256,
            "query_sha256": query_sha256,
            "scorer": scorer.content_free_binding(),
            "observation_artifact_sha256": observation_artifact_sha256,
            "requested_depth": KEY_FAMILY_DEPTH,
            "indexed_key_count": indexed_count,
            "examined_key_count": indexed_count,
            "ranked_keys": [
                item.content_free_binding(rank=rank)
                for rank, item in enumerate(ranked_keys, start=1)
            ],
            "complete": True,
        }
        return cls(
            family=family,
            question_id=corpus.question_id,
            source_artifact_sha256=corpus.source_artifact_sha256,
            projection_sha256=corpus.projection_sha256,
            index_sha256=corpus.index_sha256,
            query_sha256=query_sha256,
            scorer=scorer,
            observation_artifact_sha256=observation_artifact_sha256,
            requested_depth=KEY_FAMILY_DEPTH,
            indexed_key_count=indexed_count,
            examined_key_count=indexed_count,
            ranked_keys=ranked_keys,
            observation_sha256=sha256_json(values),
            complete=True,
        )

    def payload_without_digest(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "question_id": self.question_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "projection_sha256": self.projection_sha256,
            "index_sha256": self.index_sha256,
            "query_sha256": self.query_sha256,
            "scorer": self.scorer.content_free_binding(),
            "observation_artifact_sha256": self.observation_artifact_sha256,
            "requested_depth": self.requested_depth,
            "indexed_key_count": self.indexed_key_count,
            "examined_key_count": self.examined_key_count,
            "ranked_keys": [
                item.content_free_binding(rank=rank)
                for rank, item in enumerate(self.ranked_keys, start=1)
            ],
            "complete": self.complete,
        }

    def content_free_binding(self) -> dict[str, Any]:
        return {**self.payload_without_digest(), "observation_sha256": self.observation_sha256}


@dataclass(frozen=True, slots=True)
class ValueProvenance:
    value_id: str
    source_version_sha256: str
    raw_value_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.value_id, label="edge provenance value_id")
        _sha256(self.source_version_sha256, label="edge provenance source version")
        _sha256(self.raw_value_sha256, label="edge provenance raw value")

    @classmethod
    def from_value(cls, value: CanonicalValue) -> ValueProvenance:
        return cls(
            value_id=value.value_id,
            source_version_sha256=value.source_version_sha256,
            raw_value_sha256=value.raw_value_sha256,
        )

    def content_free_binding(self) -> dict[str, str]:
        return {
            "value_id": self.value_id,
            "source_version_sha256": self.source_version_sha256,
            "raw_value_sha256": self.raw_value_sha256,
        }


@dataclass(frozen=True, slots=True)
class GraphIdentity:
    producer: str
    protocol: str
    model_id: str
    model_revision: str
    model_artifact_sha256: str
    prompt_sha256: str
    identity_artifact_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.producer, label="graph producer")
        _identifier(self.protocol, label="graph protocol")
        _identifier(self.model_id, label="graph model_id")
        _identifier(self.model_revision, label="graph model_revision")
        _sha256(self.model_artifact_sha256, label="graph model artifact")
        _sha256(self.prompt_sha256, label="graph prompt")
        _sha256(self.identity_artifact_sha256, label="graph identity artifact")

    @property
    def identity_sha256(self) -> str:
        return sha256_json(self.binding_without_digest())

    def binding_without_digest(self) -> dict[str, str]:
        return {
            "producer": self.producer,
            "protocol": self.protocol,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_artifact_sha256": self.model_artifact_sha256,
            "prompt_sha256": self.prompt_sha256,
            "identity_artifact_sha256": self.identity_artifact_sha256,
            "identity_source": "externally-attested-unverified",
        }

    def content_free_binding(self) -> dict[str, str]:
        return {**self.binding_without_digest(), "identity_sha256": self.identity_sha256}


@dataclass(frozen=True, slots=True)
class GraphAccounting:
    input_tokens: int
    output_tokens: int
    latency_microseconds: int
    cost_microusd: int
    retry_count: int = 0
    cache_hit: bool = False

    def __post_init__(self) -> None:
        _nonnegative_int(self.input_tokens, label="graph input_tokens")
        _nonnegative_int(self.output_tokens, label="graph output_tokens")
        _nonnegative_int(self.latency_microseconds, label="graph latency")
        _nonnegative_int(self.cost_microusd, label="graph cost")
        _nonnegative_int(self.retry_count, label="graph retry_count")
        if not isinstance(self.cache_hit, bool):
            raise RepresentationError("graph cache_hit must be boolean")

    def content_free_binding(self) -> dict[str, int | bool]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_microseconds": self.latency_microseconds,
            "cost_microusd": self.cost_microusd,
            "retry_count": self.retry_count,
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True, slots=True)
class EntityAdjacencyEdge:
    edge_id: str
    left_entity_id: str
    right_entity_id: str
    evidence_values: tuple[ValueProvenance, ...]
    edge_artifact_sha256: str

    def __post_init__(self) -> None:
        _opaque_id(self.edge_id, prefix="edge", label="entity edge_id")
        _opaque_id(
            self.left_entity_id,
            prefix="entity",
            label="entity edge left endpoint",
        )
        _opaque_id(
            self.right_entity_id,
            prefix="entity",
            label="entity edge right endpoint",
        )
        if self.left_entity_id >= self.right_entity_id:
            raise RepresentationError(
                "entity adjacency endpoints must be distinct and canonically ordered"
            )
        if not isinstance(self.evidence_values, tuple) or not self.evidence_values:
            raise RepresentationError("entity adjacency requires immutable value provenance")
        if any(not isinstance(item, ValueProvenance) for item in self.evidence_values):
            raise RepresentationError("entity edge provenance must use ValueProvenance")
        ids = [item.value_id for item in self.evidence_values]
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            raise RepresentationError(
                "entity edge provenance values must be unique and canonically ordered"
            )
        _sha256(self.edge_artifact_sha256, label="entity edge artifact")

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "edge_type": "entity-adjacency",
            "left_entity_id": self.left_entity_id,
            "right_entity_id": self.right_entity_id,
            "evidence_values": [item.content_free_binding() for item in self.evidence_values],
            "edge_artifact_sha256": self.edge_artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class EntityAdjacencyGraph:
    question_id: str
    source_artifact_sha256: str
    projection_sha256: str
    index_sha256: str
    identity: GraphIdentity
    graph_artifact_sha256: str
    input_fields: tuple[str, ...]
    source_input_sha256: str
    request_sha256: str
    response_sha256: str
    accounting: GraphAccounting
    edges: tuple[EntityAdjacencyEdge, ...]
    graph_sha256: str
    complete: bool = True

    def __post_init__(self) -> None:
        _validate_graph_envelope(self)
        if any(not isinstance(edge, EntityAdjacencyEdge) for edge in self.edges):
            raise RepresentationError("entity graph edges must be EntityAdjacencyEdge")
        pairs = [(edge.left_entity_id, edge.right_entity_id) for edge in self.edges]
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(set(pairs)) != len(pairs) or len(set(edge_ids)) != len(edge_ids):
            raise RepresentationError("entity graph repeats an edge or endpoint pair")
        if list(self.edges) != sorted(
            self.edges,
            key=lambda edge: (edge.left_entity_id, edge.right_entity_id, edge.edge_id),
        ):
            raise RepresentationError("entity graph edges must be canonically ordered")
        _validate_degree_bound(pairs, label="entity graph")
        if self.graph_sha256 != sha256_json(self.payload_without_digest()):
            raise RepresentationError("entity graph digest does not match its payload")

    @classmethod
    def create(
        cls,
        *,
        corpus: RepresentationCorpus,
        identity: GraphIdentity,
        graph_artifact_sha256: str,
        response_sha256: str,
        accounting: GraphAccounting,
        edges: tuple[EntityAdjacencyEdge, ...],
    ) -> EntityAdjacencyGraph:
        base = _graph_payload(
            graph_type="entity-adjacency",
            corpus=corpus,
            identity=identity,
            graph_artifact_sha256=graph_artifact_sha256,
            input_fields=GRAPH_INPUT_FIELDS,
            source_input_sha256=corpus.navigation_index_sha256,
            request_sha256=graph_request_sha256(
                graph_type="entity-adjacency",
                navigation_index_sha256=corpus.navigation_index_sha256,
                identity=identity,
            ),
            response_sha256=response_sha256,
            accounting=accounting,
            edges=[edge.content_free_binding() for edge in edges],
        )
        return cls(
            question_id=corpus.question_id,
            source_artifact_sha256=corpus.source_artifact_sha256,
            projection_sha256=corpus.projection_sha256,
            index_sha256=corpus.index_sha256,
            identity=identity,
            graph_artifact_sha256=graph_artifact_sha256,
            input_fields=GRAPH_INPUT_FIELDS,
            source_input_sha256=corpus.navigation_index_sha256,
            request_sha256=graph_request_sha256(
                graph_type="entity-adjacency",
                navigation_index_sha256=corpus.navigation_index_sha256,
                identity=identity,
            ),
            response_sha256=response_sha256,
            accounting=accounting,
            edges=edges,
            graph_sha256=sha256_json(base),
            complete=True,
        )

    def payload_without_digest(self) -> dict[str, Any]:
        return _graph_payload_from_instance(
            graph_type="entity-adjacency",
            graph=self,
            edges=[edge.content_free_binding() for edge in self.edges],
        )

    def content_free_binding(self) -> dict[str, Any]:
        return {**self.payload_without_digest(), "graph_sha256": self.graph_sha256}


@dataclass(frozen=True, slots=True)
class SimilarityAdjacencyEdge:
    edge_id: str
    left_value: ValueProvenance
    right_value: ValueProvenance
    similarity_score: float
    edge_artifact_sha256: str

    def __post_init__(self) -> None:
        _opaque_id(self.edge_id, prefix="edge", label="similarity edge_id")
        if not isinstance(self.left_value, ValueProvenance) or not isinstance(
            self.right_value, ValueProvenance
        ):
            raise RepresentationError("similarity edge endpoints require ValueProvenance")
        if self.left_value.value_id >= self.right_value.value_id:
            raise RepresentationError(
                "similarity endpoints must be distinct and canonically ordered"
            )
        object.__setattr__(
            self,
            "similarity_score",
            _unit_score(self.similarity_score, label="similarity edge score"),
        )
        if self.similarity_score < SIMILARITY_EDGE_THRESHOLD:
            raise RepresentationError(
                f"similarity edge score must meet the frozen {SIMILARITY_EDGE_THRESHOLD} threshold"
            )
        _sha256(self.edge_artifact_sha256, label="similarity edge artifact")

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "edge_type": "similarity-adjacency",
            "left_value": self.left_value.content_free_binding(),
            "right_value": self.right_value.content_free_binding(),
            "similarity_score": self.similarity_score,
            "edge_artifact_sha256": self.edge_artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class SimilarityAdjacencyGraph:
    question_id: str
    source_artifact_sha256: str
    projection_sha256: str
    index_sha256: str
    identity: GraphIdentity
    graph_artifact_sha256: str
    input_fields: tuple[str, ...]
    source_input_sha256: str
    request_sha256: str
    response_sha256: str
    accounting: GraphAccounting
    edges: tuple[SimilarityAdjacencyEdge, ...]
    graph_sha256: str
    complete: bool = True

    def __post_init__(self) -> None:
        _validate_graph_envelope(self)
        if any(not isinstance(edge, SimilarityAdjacencyEdge) for edge in self.edges):
            raise RepresentationError("similarity graph edges must be SimilarityAdjacencyEdge")
        pairs = [(edge.left_value.value_id, edge.right_value.value_id) for edge in self.edges]
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(set(pairs)) != len(pairs) or len(set(edge_ids)) != len(edge_ids):
            raise RepresentationError("similarity graph repeats an edge or endpoint pair")
        if list(self.edges) != sorted(
            self.edges,
            key=lambda edge: (
                edge.left_value.value_id,
                edge.right_value.value_id,
                edge.edge_id,
            ),
        ):
            raise RepresentationError("similarity graph edges must be canonically ordered")
        _validate_degree_bound(pairs, label="similarity graph")
        if self.graph_sha256 != sha256_json(self.payload_without_digest()):
            raise RepresentationError("similarity graph digest does not match its payload")

    @classmethod
    def create(
        cls,
        *,
        corpus: RepresentationCorpus,
        identity: GraphIdentity,
        graph_artifact_sha256: str,
        response_sha256: str,
        accounting: GraphAccounting,
        edges: tuple[SimilarityAdjacencyEdge, ...],
    ) -> SimilarityAdjacencyGraph:
        base = _graph_payload(
            graph_type="similarity-adjacency",
            corpus=corpus,
            identity=identity,
            graph_artifact_sha256=graph_artifact_sha256,
            input_fields=GRAPH_INPUT_FIELDS,
            source_input_sha256=corpus.navigation_index_sha256,
            request_sha256=graph_request_sha256(
                graph_type="similarity-adjacency",
                navigation_index_sha256=corpus.navigation_index_sha256,
                identity=identity,
            ),
            response_sha256=response_sha256,
            accounting=accounting,
            edges=[edge.content_free_binding() for edge in edges],
        )
        return cls(
            question_id=corpus.question_id,
            source_artifact_sha256=corpus.source_artifact_sha256,
            projection_sha256=corpus.projection_sha256,
            index_sha256=corpus.index_sha256,
            identity=identity,
            graph_artifact_sha256=graph_artifact_sha256,
            input_fields=GRAPH_INPUT_FIELDS,
            source_input_sha256=corpus.navigation_index_sha256,
            request_sha256=graph_request_sha256(
                graph_type="similarity-adjacency",
                navigation_index_sha256=corpus.navigation_index_sha256,
                identity=identity,
            ),
            response_sha256=response_sha256,
            accounting=accounting,
            edges=edges,
            graph_sha256=sha256_json(base),
            complete=True,
        )

    def payload_without_digest(self) -> dict[str, Any]:
        return _graph_payload_from_instance(
            graph_type="similarity-adjacency",
            graph=self,
            edges=[edge.content_free_binding() for edge in self.edges],
        )

    def content_free_binding(self) -> dict[str, Any]:
        return {**self.payload_without_digest(), "graph_sha256": self.graph_sha256}


Graph = EntityAdjacencyGraph | SimilarityAdjacencyGraph


def _validate_graph_envelope(graph: Graph) -> None:
    _identifier(graph.question_id, label="graph question_id")
    _sha256(graph.source_artifact_sha256, label="graph source artifact")
    _sha256(graph.projection_sha256, label="graph projection")
    _sha256(graph.index_sha256, label="graph index")
    if not isinstance(graph.identity, GraphIdentity):
        raise RepresentationError("graph requires GraphIdentity")
    _sha256(graph.graph_artifact_sha256, label="graph artifact")
    if graph.input_fields != GRAPH_INPUT_FIELDS:
        raise RepresentationError(
            "graph input fields must contain only the source-safe navigation index"
        )
    _sha256(graph.source_input_sha256, label="graph source-safe navigation index")
    graph_type = (
        "entity-adjacency" if isinstance(graph, EntityAdjacencyGraph) else "similarity-adjacency"
    )
    if graph.request_sha256 != graph_request_sha256(
        graph_type=graph_type,
        navigation_index_sha256=graph.source_input_sha256,
        identity=graph.identity,
        input_fields=graph.input_fields,
    ):
        raise RepresentationError(
            "graph request digest does not match source-safe navigation request material"
        )
    _sha256(graph.response_sha256, label="graph response")
    if not isinstance(graph.accounting, GraphAccounting):
        raise RepresentationError("graph requires GraphAccounting")
    if not isinstance(graph.edges, tuple):
        raise RepresentationError("graph edges must be an immutable tuple")
    if len(graph.edges) > MAX_ADJACENCY_EDGES:
        raise RepresentationError("graph exceeds the frozen adjacency edge bound")
    _sha256(graph.graph_sha256, label="graph payload")
    if graph.complete is not True:
        raise RepresentationError("partial adjacency graphs are forbidden")


def _validate_degree_bound(pairs: list[tuple[str, str]], *, label: str) -> None:
    degree: dict[str, int] = {}
    for left, right in pairs:
        degree[left] = degree.get(left, 0) + 1
        degree[right] = degree.get(right, 0) + 1
    if any(count > MAX_NEIGHBORS_PER_NODE for count in degree.values()):
        raise RepresentationError(f"{label} exceeds the frozen per-node neighbor bound")


def _graph_payload(
    *,
    graph_type: str,
    corpus: RepresentationCorpus,
    identity: GraphIdentity,
    graph_artifact_sha256: str,
    input_fields: tuple[str, ...],
    source_input_sha256: str,
    request_sha256: str,
    response_sha256: str,
    accounting: GraphAccounting,
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "graph_type": graph_type,
        "question_id": corpus.question_id,
        "source_artifact_sha256": corpus.source_artifact_sha256,
        "projection_sha256": corpus.projection_sha256,
        "index_sha256": corpus.index_sha256,
        "identity": identity.content_free_binding(),
        "graph_artifact_sha256": graph_artifact_sha256,
        "input_contract": {
            "fields": list(input_fields),
            "question_id_is_routing_metadata_not_graph_input": True,
            "source_input_sha256": source_input_sha256,
            "forbidden_fields": [
                "question",
                "question_type",
                "answer",
                "gold_session_ids",
                "judge_label",
            ],
        },
        "request_sha256": request_sha256,
        "response_sha256": response_sha256,
        "accounting": accounting.content_free_binding(),
        "edges": edges,
        "complete": True,
    }


def _graph_payload_from_instance(
    *,
    graph_type: str,
    graph: Graph,
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "graph_type": graph_type,
        "question_id": graph.question_id,
        "source_artifact_sha256": graph.source_artifact_sha256,
        "projection_sha256": graph.projection_sha256,
        "index_sha256": graph.index_sha256,
        "identity": graph.identity.content_free_binding(),
        "graph_artifact_sha256": graph.graph_artifact_sha256,
        "input_contract": {
            "fields": list(graph.input_fields),
            "question_id_is_routing_metadata_not_graph_input": True,
            "source_input_sha256": graph.source_input_sha256,
            "forbidden_fields": [
                "question",
                "question_type",
                "answer",
                "gold_session_ids",
                "judge_label",
            ],
        },
        "request_sha256": graph.request_sha256,
        "response_sha256": graph.response_sha256,
        "accounting": graph.accounting.content_free_binding(),
        "edges": edges,
        "complete": graph.complete,
    }


def graph_request_sha256(
    *,
    graph_type: str,
    navigation_index_sha256: str,
    identity: GraphIdentity,
    input_fields: tuple[str, ...] = GRAPH_INPUT_FIELDS,
) -> str:
    """Digest the only material allowed to construct a representation graph."""

    if graph_type not in {"entity-adjacency", "similarity-adjacency"}:
        raise RepresentationError("graph request type is unsupported")
    _sha256(
        navigation_index_sha256,
        label="graph request source-safe navigation index",
    )
    if not isinstance(identity, GraphIdentity):
        raise RepresentationError("graph request requires GraphIdentity")
    if input_fields != GRAPH_INPUT_FIELDS:
        raise RepresentationError(
            "graph request input allowlist is not source-safe-navigation-index-only"
        )
    return sha256_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "graph_type": graph_type,
            "input_fields": list(input_fields),
            "source_safe_navigation_index_sha256": navigation_index_sha256,
            "graph_identity_sha256": identity.identity_sha256,
            "graph_prompt_sha256": identity.prompt_sha256,
        }
    )
