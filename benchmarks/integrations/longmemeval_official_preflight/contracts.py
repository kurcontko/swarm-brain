"""Frozen contracts for offline LongMemEval-S reader-run admission."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from benchmarks.integrations.longmemeval_turn_prompt import (
    ARTIFACT_TYPE as PROMPT_ARTIFACT_TYPE,
)
from benchmarks.integrations.longmemeval_turn_prompt import (
    OFFICIAL_ANSWER_TEMPLATE_SHA256,
    PRIMARY_TOKEN_BUDGET,
    TOKENIZER_PROTOCOL,
    TokenizerIdentity,
)
from benchmarks.integrations.longmemeval_turn_prompt import (
    PROTOCOL_VERSION as PROMPT_PROTOCOL_VERSION,
)
from benchmarks.integrations.longmemeval_turn_prompt import (
    SCHEMA_VERSION as PROMPT_SCHEMA_VERSION,
)
from benchmarks.integrations.longmemeval_turns import OFFICIAL_LONGMEMEVAL_S_SHA256

ARTIFACT_TYPE = "swarmbrain-longmemeval-official-preflight"
SCHEMA_VERSION = 1
PROTOCOL_VERSION = "swarmbrain-longmemeval-official-preflight-v1"
READY_ARTIFACT_TYPE = "swarmbrain-longmemeval-official-packed-run-ready"

OFFICIAL_DATASET_NAME = "LongMemEval-S"
OFFICIAL_SOURCE_LABEL = "longmemeval_s_cleaned.json"
OFFICIAL_QUESTION_COUNT = 500

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class LongMemEvalOfficialPreflightError(ValueError):
    """A planned or packed run violates the frozen admission contract."""


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
        raise LongMemEvalOfficialPreflightError(
            "preflight material must be finite canonical UTF-8 JSON"
        ) from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_text(value: str) -> str:
    try:
        return sha256_bytes(value.encode("utf-8"))
    except UnicodeError as exc:
        raise LongMemEvalOfficialPreflightError("preflight text must be valid UTF-8") from exc


def checked_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise LongMemEvalOfficialPreflightError(
            f"{label} must be a lowercase hexadecimal SHA-256 digest"
        )
    return value


def checked_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise LongMemEvalOfficialPreflightError(
            f"{label} must be non-empty text without surrounding whitespace"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise LongMemEvalOfficialPreflightError(f"{label} cannot contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise LongMemEvalOfficialPreflightError(f"{label} must be valid UTF-8") from exc
    return value


def checked_integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LongMemEvalOfficialPreflightError(f"{label} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class DatasetRequirement:
    """Operator-frozen corpus identity; official mode cannot be weakened."""

    name: str
    source_label: str
    source_sha256: str
    question_count: int
    official: bool

    def __post_init__(self) -> None:
        checked_text(self.name, label="dataset name")
        checked_text(self.source_label, label="dataset source label")
        checked_sha256(self.source_sha256, label="dataset source SHA-256")
        checked_integer(self.question_count, label="dataset question count", minimum=1)
        if not isinstance(self.official, bool):
            raise LongMemEvalOfficialPreflightError("dataset official flag must be boolean")
        if self.official and (
            self.name,
            self.source_label,
            self.source_sha256,
            self.question_count,
        ) != (
            OFFICIAL_DATASET_NAME,
            OFFICIAL_SOURCE_LABEL,
            OFFICIAL_LONGMEMEVAL_S_SHA256,
            OFFICIAL_QUESTION_COUNT,
        ):
            raise LongMemEvalOfficialPreflightError(
                "official mode requires the exact cleaned LongMemEval-S corpus and 500 cases"
            )

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_label": self.source_label,
            "source_sha256": self.source_sha256,
            "question_count": self.question_count,
            "official": self.official,
        }


OFFICIAL_DATASET_REQUIREMENT = DatasetRequirement(
    name=OFFICIAL_DATASET_NAME,
    source_label=OFFICIAL_SOURCE_LABEL,
    source_sha256=OFFICIAL_LONGMEMEVAL_S_SHA256,
    question_count=OFFICIAL_QUESTION_COUNT,
    official=True,
)


@dataclass(frozen=True, slots=True)
class ExactTokenizerPin:
    """Precommitted tokenizer and local executable bytes for the whole run."""

    model: str
    revision: str
    artifact_sha256: str
    executable_sha256: str
    protocol: str = TOKENIZER_PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol != TOKENIZER_PROTOCOL:
            raise LongMemEvalOfficialPreflightError(
                f"tokenizer protocol must be the exact {TOKENIZER_PROTOCOL!r}"
            )
        checked_text(self.model, label="tokenizer model")
        checked_text(self.revision, label="tokenizer revision")
        checked_sha256(self.artifact_sha256, label="tokenizer artifact SHA-256")
        checked_sha256(self.executable_sha256, label="tokenizer executable SHA-256")

    @property
    def identity(self) -> TokenizerIdentity:
        return TokenizerIdentity(
            protocol=self.protocol,
            model=self.model,
            revision=self.revision,
            artifact_sha256=self.artifact_sha256,
        )

    def content_free_binding(self) -> dict[str, str]:
        return {
            **self.identity.as_dict(),
            "executable_sha256": self.executable_sha256,
        }


@dataclass(frozen=True, slots=True)
class DatasetCaseBinding:
    """Content-free source binding needed to admit one packed prompt."""

    case_index: int
    question_id: str
    source_record_sha256: str
    source_record_utf8_bytes: int
    question_sha256: str
    question_utf8_bytes: int
    current_date_sha256: str
    current_date_utf8_bytes: int

    def __post_init__(self) -> None:
        checked_integer(self.case_index, label="case index")
        checked_text(self.question_id, label="case question_id")
        checked_sha256(self.source_record_sha256, label="case source-record SHA-256")
        checked_integer(
            self.source_record_utf8_bytes,
            label="case source-record bytes",
            minimum=1,
        )
        checked_sha256(self.question_sha256, label="case question SHA-256")
        checked_integer(self.question_utf8_bytes, label="case question bytes", minimum=1)
        checked_sha256(self.current_date_sha256, label="case current-date SHA-256")
        checked_integer(
            self.current_date_utf8_bytes,
            label="case current-date bytes",
            minimum=1,
        )

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "case_index": self.case_index,
            "question_id": self.question_id,
            "source_record": {
                "sha256": self.source_record_sha256,
                "utf8_bytes": self.source_record_utf8_bytes,
            },
            "question": {
                "sha256": self.question_sha256,
                "utf8_bytes": self.question_utf8_bytes,
            },
            "current_date": {
                "sha256": self.current_date_sha256,
                "utf8_bytes": self.current_date_utf8_bytes,
            },
        }


@dataclass(frozen=True, slots=True)
class RunPreflightManifest:
    """Immutable plan that must be frozen before external prompt counting."""

    dataset: DatasetRequirement
    source_artifact_utf8_bytes: int
    parsed_records_sha256: str
    parsed_records_utf8_bytes: int
    projection_sha256: str
    question_ids_sha256: str
    cases: tuple[DatasetCaseBinding, ...]
    tokenizer: ExactTokenizerPin

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, DatasetRequirement):
            raise LongMemEvalOfficialPreflightError(
                "preflight dataset must be a DatasetRequirement"
            )
        checked_integer(
            self.source_artifact_utf8_bytes,
            label="source artifact bytes",
            minimum=1,
        )
        checked_sha256(self.parsed_records_sha256, label="parsed-records SHA-256")
        checked_integer(
            self.parsed_records_utf8_bytes,
            label="parsed-records bytes",
            minimum=1,
        )
        checked_sha256(self.projection_sha256, label="turn projection SHA-256")
        checked_sha256(self.question_ids_sha256, label="question IDs SHA-256")
        if not isinstance(self.cases, tuple) or any(
            not isinstance(case, DatasetCaseBinding) for case in self.cases
        ):
            raise LongMemEvalOfficialPreflightError(
                "preflight cases must be an immutable tuple of DatasetCaseBinding"
            )
        if len(self.cases) != self.dataset.question_count:
            raise LongMemEvalOfficialPreflightError(
                "preflight case coverage does not equal the pinned dataset question count"
            )
        if [case.case_index for case in self.cases] != list(range(len(self.cases))):
            raise LongMemEvalOfficialPreflightError(
                "preflight cases must preserve contiguous source order"
            )
        question_ids = [case.question_id for case in self.cases]
        if len(set(question_ids)) != len(question_ids):
            raise LongMemEvalOfficialPreflightError("preflight repeats a question ID")
        if self.question_ids_sha256 != sha256_json(question_ids):
            raise LongMemEvalOfficialPreflightError(
                "preflight question-ID digest does not match exact ordered coverage"
            )
        if not isinstance(self.tokenizer, ExactTokenizerPin):
            raise LongMemEvalOfficialPreflightError(
                "preflight tokenizer must be an ExactTokenizerPin"
            )

    @property
    def manifest_sha256(self) -> str:
        return sha256_json(self.content_free_binding())

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "artifact_type": ARTIFACT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "classification": (
                "official-full-500-reader-preflight"
                if self.dataset.official
                else "nonofficial-pinned-reader-preflight"
            ),
            "dataset": {
                **self.dataset.content_free_binding(),
                "source_artifact_utf8_bytes": self.source_artifact_utf8_bytes,
                "parsed_records_sha256": self.parsed_records_sha256,
                "parsed_records_utf8_bytes": self.parsed_records_utf8_bytes,
                "turn_projection_sha256": self.projection_sha256,
                "question_ids_sha256": self.question_ids_sha256,
            },
            "prompt": {
                "artifact_type": PROMPT_ARTIFACT_TYPE,
                "schema_version": PROMPT_SCHEMA_VERSION,
                "protocol_version": PROMPT_PROTOCOL_VERSION,
                "template_sha256": OFFICIAL_ANSWER_TEMPLATE_SHA256,
                "token_budget": PRIMARY_TOKEN_BUDGET,
                "counted_surface": "complete-official-reader-prompt",
                "whole_turns_indivisible": True,
                "oversized_policy": "skip-and-continue",
            },
            "tokenizer": self.tokenizer.content_free_binding(),
            "cases": [case.content_free_binding() for case in self.cases],
            "case_bindings_sha256": sha256_json(
                [case.content_free_binding() for case in self.cases]
            ),
            "ready_for_external_calls": False,
        }

    def content_free_artifact(self) -> dict[str, Any]:
        return {**self.content_free_binding(), "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True, slots=True)
class PreparedRunReceipt:
    """Content-free proof that every packed prompt matches one frozen plan."""

    preflight_manifest_sha256: str
    packed_case_count: int
    question_ids_sha256: str
    prompt_trace_sha256s: tuple[str, ...]
    prompt_trace_set_sha256: str
    tokenizer_evidence_sha256: str
    exact_count_observation_count: int
    exact_count_prompt_utf8_bytes: int

    def __post_init__(self) -> None:
        checked_sha256(self.preflight_manifest_sha256, label="ready preflight manifest")
        checked_integer(self.packed_case_count, label="ready packed case count", minimum=1)
        checked_sha256(self.question_ids_sha256, label="ready question IDs")
        if (
            not isinstance(self.prompt_trace_sha256s, tuple)
            or len(self.prompt_trace_sha256s) != self.packed_case_count
        ):
            raise LongMemEvalOfficialPreflightError(
                "ready prompt trace coverage does not match packed cases"
            )
        for digest in self.prompt_trace_sha256s:
            checked_sha256(digest, label="ready prompt trace")
        if len(set(self.prompt_trace_sha256s)) != len(self.prompt_trace_sha256s):
            raise LongMemEvalOfficialPreflightError("ready receipt repeats a prompt trace digest")
        checked_sha256(self.prompt_trace_set_sha256, label="ready prompt trace set")
        if self.prompt_trace_set_sha256 != sha256_json(list(self.prompt_trace_sha256s)):
            raise LongMemEvalOfficialPreflightError(
                "ready prompt trace-set digest does not match exact ordered traces"
            )
        checked_sha256(self.tokenizer_evidence_sha256, label="ready tokenizer evidence")
        checked_integer(
            self.exact_count_observation_count,
            label="ready exact-count observations",
            minimum=1,
        )
        checked_integer(
            self.exact_count_prompt_utf8_bytes,
            label="ready exact-count prompt bytes",
            minimum=1,
        )

    @property
    def receipt_sha256(self) -> str:
        return sha256_json(self.content_free_binding())

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "artifact_type": READY_ARTIFACT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "preflight_manifest_sha256": self.preflight_manifest_sha256,
            "packed_case_count": self.packed_case_count,
            "question_ids_sha256": self.question_ids_sha256,
            "prompt_trace_sha256s": list(self.prompt_trace_sha256s),
            "prompt_trace_set_sha256": self.prompt_trace_set_sha256,
            "tokenizer_evidence_sha256": self.tokenizer_evidence_sha256,
            "exact_count_observation_count": self.exact_count_observation_count,
            "exact_count_prompt_utf8_bytes": self.exact_count_prompt_utf8_bytes,
            "ready_for_reader_calls": True,
        }

    def content_free_artifact(self) -> dict[str, Any]:
        return {**self.content_free_binding(), "receipt_sha256": self.receipt_sha256}


__all__ = [
    "ARTIFACT_TYPE",
    "DatasetCaseBinding",
    "DatasetRequirement",
    "ExactTokenizerPin",
    "LongMemEvalOfficialPreflightError",
    "OFFICIAL_DATASET_NAME",
    "OFFICIAL_DATASET_REQUIREMENT",
    "OFFICIAL_QUESTION_COUNT",
    "OFFICIAL_SOURCE_LABEL",
    "PROTOCOL_VERSION",
    "PreparedRunReceipt",
    "READY_ARTIFACT_TYPE",
    "RunPreflightManifest",
    "SCHEMA_VERSION",
    "canonical_json_bytes",
    "checked_integer",
    "checked_sha256",
    "checked_text",
    "sha256_bytes",
    "sha256_json",
    "sha256_text",
]
