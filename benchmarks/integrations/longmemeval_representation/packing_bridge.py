"""Fail-closed E6 hydration bridge to the exact LongMemEval reader packer.

The representation experiment ranks canonical raw F0 turns.  This module is
the deliberately small boundary that passes those turns, in their hydrated
order and without rewriting their bytes, to the existing exact 8,192-token
official reader-prompt packer.  It does not read gold fields, execute a reader,
or make a network call.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from scripts._longmemeval_common import EMPTY_CONTEXT_NOTE, OFFICIAL_ANSWER_TEMPLATE

from benchmarks.integrations.longmemeval_official_preflight import (
    RunPreflightManifest,
)
from benchmarks.integrations.longmemeval_turn_prompt import (
    LINEAR_TURN_SEPARATOR,
    PRIMARY_TOKEN_BUDGET,
    ExactPromptTokenizer,
    OrderedTurnBlocks,
    TurnPromptPackingError,
    TurnPromptPackingResult,
    pack_turn_prompt,
)

from .contracts import (
    ARTIFACT_TYPE as REPRESENTATION_ARTIFACT_TYPE,
)
from .contracts import (
    MAX_HYDRATED_VALUES,
    CanonicalValue,
    RepresentationError,
    sha256_json,
)
from .contracts import (
    PROTOCOL_VERSION as REPRESENTATION_PROTOCOL_VERSION,
)
from .contracts import (
    SCHEMA_VERSION as REPRESENTATION_SCHEMA_VERSION,
)
from .experiment import RepresentationResult

ARTIFACT_TYPE = "swarmbrain-longmemeval-e6-representation-packing-bridge"
SCHEMA_VERSION = 1
PROTOCOL_VERSION = "swarmbrain-longmemeval-e6-representation-packing-bridge-v1"

_REPRESENTATION_TRACE_FIELDS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "protocol_version",
        "cell",
        "classification",
        "production_configuration",
        "paper_reproduction",
        "sb_hypothesis",
        "frozen_protocol",
        "promotion",
        "corpus",
        "observations",
        "observations_sha256",
        "ranking",
        "graph",
        "value_scores",
        "value_scores_sha256",
        "key_level_returned_count",
        "hydrated_value_pre_cap_count",
        "hydrated_value_cap",
        "hydrated_value_ids",
        "hydrated_raw_value_hashes",
        "hydrated_value_count",
        "hydration",
        "construction_and_index_accounting",
        "construction_input_contract",
        "claims",
    }
)


class RepresentationPackingBridgeError(RepresentationError):
    """An E6 result cannot be admitted to exact reader-prompt packing."""


def _utf8_bytes(value: str, *, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise RepresentationPackingBridgeError(f"{label} must be non-empty text")
    try:
        return value.encode("utf-8")
    except UnicodeError as exc:
        raise RepresentationPackingBridgeError(f"{label} must be valid UTF-8") from exc


def _exact_mapping(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RepresentationPackingBridgeError(f"{label} fields differ from the frozen schema")
    return value


def _turn_id_payload(value: CanonicalValue) -> list[str | int]:
    return list(value.turn.turn_id.as_tuple())


def _validate_representation_trace(result: RepresentationResult) -> Mapping[str, Any]:
    trace = _exact_mapping(
        result.trace,
        _REPRESENTATION_TRACE_FIELDS,
        label="representation trace",
    )
    expected_header = {
        "artifact_type": REPRESENTATION_ARTIFACT_TYPE,
        "schema_version": REPRESENTATION_SCHEMA_VERSION,
        "protocol_version": REPRESENTATION_PROTOCOL_VERSION,
        "cell": result.cell.value,
        "classification": "benchmark-only-source-preserving-representation-control",
        "production_configuration": False,
        "paper_reproduction": False,
        "sb_hypothesis": "SB-HMR-v1",
    }
    for field, expected in expected_header.items():
        if type(trace.get(field)) is not type(expected) or trace.get(field) != expected:
            raise RepresentationPackingBridgeError(f"representation trace {field} drifted")

    values = result.hydrated_values
    value_ids = [value.value_id for value in values]
    raw_hashes = [value.raw_value_sha256 for value in values]
    expected_hydration = {
        "reader_evidence": "canonical-raw-value",
        "derived_keys_delivered_to_reader": False,
        "source_values_byte_identical": True,
    }
    exact_values = {
        "hydrated_value_ids": value_ids,
        "hydrated_raw_value_hashes": raw_hashes,
        "hydrated_value_count": len(values),
        "hydrated_value_cap": MAX_HYDRATED_VALUES,
        "hydration": expected_hydration,
    }
    for field, expected in exact_values.items():
        if type(trace.get(field)) is not type(expected) or trace.get(field) != expected:
            raise RepresentationPackingBridgeError(f"representation trace {field} drifted")

    scores = trace.get("value_scores")
    if not isinstance(scores, list) or len(scores) != len(values):
        raise RepresentationPackingBridgeError(
            "representation value-score order does not cover every hydrated value"
        )
    for rank, (score, value) in enumerate(zip(scores, values, strict=True), start=1):
        if (
            not isinstance(score, Mapping)
            or score.get("rank") != rank
            or score.get("value") != value.content_free_binding()
        ):
            raise RepresentationPackingBridgeError(
                "representation value-score order differs from hydrated values"
            )
    if trace.get("value_scores_sha256") != sha256_json(scores):
        raise RepresentationPackingBridgeError("representation value-score digest drifted")

    expected_input_contract = {
        "gold_question_type_answer_or_judge_fields_allowed": False,
        "question_text_allowed_for_key_or_graph_construction": False,
        "request_material_digests_recomputed": True,
        "external_execution_identity_attested_not_verified": True,
    }
    if trace.get("construction_input_contract") != expected_input_contract:
        raise RepresentationPackingBridgeError("representation construction input contract drifted")
    expected_claims = {
        "question_query_consumed_by_ranking": True,
        "question_id_is_local_routing_metadata_only": True,
        "executes_extractor_scorer_model_or_network": False,
        "external_identities_verified_by_this_module": False,
        "quality_improvement": False,
        "serving_change": False,
    }
    if trace.get("claims") != expected_claims:
        raise RepresentationPackingBridgeError("representation claims drifted")
    return trace


def _validate_source_and_case(
    result: RepresentationResult,
    *,
    manifest: RunPreflightManifest,
    question_id: str,
    question: str,
    current_date: str,
) -> Any:
    if not isinstance(manifest, RunPreflightManifest):
        raise RepresentationPackingBridgeError("preflight must be a RunPreflightManifest")
    if not isinstance(question_id, str) or not question_id or question_id != question_id.strip():
        raise RepresentationPackingBridgeError("question_id must be non-empty canonical text")
    question_bytes = _utf8_bytes(question, label="question")
    current_date_bytes = _utf8_bytes(current_date, label="current_date")
    case = next((item for item in manifest.cases if item.question_id == question_id), None)
    if case is None:
        raise RepresentationPackingBridgeError("question_id is absent from the preflight manifest")
    expected_question = (len(question_bytes), hashlib.sha256(question_bytes).hexdigest())
    if expected_question != (case.question_utf8_bytes, case.question_sha256):
        raise RepresentationPackingBridgeError("question bytes differ from the preflight case")
    expected_date = (len(current_date_bytes), hashlib.sha256(current_date_bytes).hexdigest())
    if expected_date != (case.current_date_utf8_bytes, case.current_date_sha256):
        raise RepresentationPackingBridgeError("current_date bytes differ from the preflight case")

    trace = _validate_representation_trace(result)
    corpus = trace.get("corpus")
    if not isinstance(corpus, Mapping):
        raise RepresentationPackingBridgeError("representation trace lacks a corpus binding")
    expected_corpus = {
        "question_id": question_id,
        "source_artifact_sha256": manifest.dataset.source_sha256,
        "projection_sha256": manifest.projection_sha256,
    }
    for field, expected in expected_corpus.items():
        if type(corpus.get(field)) is not type(expected) or corpus.get(field) != expected:
            raise RepresentationPackingBridgeError(f"representation corpus {field} drifted")

    for value in result.hydrated_values:
        if not isinstance(value, CanonicalValue):
            raise RepresentationPackingBridgeError("hydration contains a non-canonical value")
        if value.question_id != question_id:
            raise RepresentationPackingBridgeError("hydration crosses the preflight question")
        if value.source_artifact_sha256 != manifest.dataset.source_sha256:
            raise RepresentationPackingBridgeError(
                "hydration crosses the preflight source artifact"
            )
        if value.projection_sha256 != manifest.projection_sha256:
            raise RepresentationPackingBridgeError("hydration crosses the preflight projection")
        if (
            value.turn.source_record.sha256 != case.source_record_sha256
            or value.turn.source_record.bytes != case.source_record_utf8_bytes
        ):
            raise RepresentationPackingBridgeError("hydration source-record bytes drifted")
    return case


def _hydrated_bindings(values: tuple[CanonicalValue, ...]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_position": position,
            "value_id": value.value_id,
            "turn_id": _turn_id_payload(value),
            "source_version_sha256": value.source_version_sha256,
            "raw_value_sha256": value.raw_value_sha256,
            "raw_value_utf8_bytes": value.raw_value_utf8_bytes,
        }
        for position, value in enumerate(values, start=1)
    ]


@dataclass(frozen=True, slots=True)
class RepresentationPromptPackingResult:
    """Exact packed prompt plus a content-free E6-to-packer binding."""

    packed: TurnPromptPackingResult
    trace: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.packed, TurnPromptPackingResult):
            raise RepresentationPackingBridgeError("packed must be TurnPromptPackingResult")
        if not isinstance(self.trace, dict):
            raise RepresentationPackingBridgeError("bridge trace must be a dictionary")
        packing = self.trace.get("packing")
        if not isinstance(packing, Mapping):
            raise RepresentationPackingBridgeError("bridge trace lacks packing evidence")
        if packing.get("trace_sha256") != self.packed.trace_sha256:
            raise RepresentationPackingBridgeError("bridge trace does not bind the exact packing")
        if packing.get("final_prompt") != self.packed.trace.get("final_prompt"):
            raise RepresentationPackingBridgeError("bridge final-prompt binding drifted")
        if packing.get("final_context") != self.packed.trace.get("final_history"):
            raise RepresentationPackingBridgeError("bridge final-context binding drifted")

    @property
    def prompt(self) -> str:
        return self.packed.prompt

    @property
    def trace_sha256(self) -> str:
        return sha256_json(self.trace)

    def content_free_artifact(self) -> dict[str, Any]:
        return {**self.trace, "trace_sha256": self.trace_sha256}


def pack_representation_result(
    result: RepresentationResult,
    *,
    manifest: RunPreflightManifest,
    question_id: str,
    question: str,
    current_date: str,
    tokenizer: ExactPromptTokenizer,
) -> RepresentationPromptPackingResult:
    """Pack one E6 result into an exact primary-budget official reader prompt.

    ``result.hydrated_values`` is the sole evidence input.  Its canonical turns
    are passed to the established linear whole-turn packer without sorting,
    text normalization, truncation, or conversion of derived navigation keys.
    The returned ``packed`` value can be passed unchanged to the run-level
    ``validate_prepared_run`` preflight admission boundary.
    """

    if not isinstance(result, RepresentationResult):
        raise RepresentationPackingBridgeError("result must be RepresentationResult")
    case = _validate_source_and_case(
        result,
        manifest=manifest,
        question_id=question_id,
        question=question,
        current_date=current_date,
    )
    try:
        tokenizer_identity = tokenizer.identity
    except Exception as exc:  # pragma: no cover - defensive boundary normalization
        raise RepresentationPackingBridgeError("could not read exact tokenizer identity") from exc
    if tokenizer_identity != manifest.tokenizer.identity:
        raise RepresentationPackingBridgeError("tokenizer identity differs from the preflight pin")

    values = result.hydrated_values
    turns = tuple(value.turn for value in values)
    try:
        packed = pack_turn_prompt(
            OrderedTurnBlocks.linear(turns),
            question_id=question_id,
            question=question,
            current_date=current_date,
            token_budget=PRIMARY_TOKEN_BUDGET,
            tokenizer=tokenizer,
        )
    except TurnPromptPackingError as exc:
        raise RepresentationPackingBridgeError(f"exact prompt packing failed: {exc}") from exc

    expected_candidate_ids = [_turn_id_payload(value) for value in values]
    if packed.trace.get("candidate_blocks") != [expected_candidate_ids]:
        raise RepresentationPackingBridgeError("packer mutated the hydrated candidate order")
    candidate_order = packed.trace.get("candidate_order")
    if (
        not isinstance(candidate_order, list)
        or [
            item.get("turn", {}).get("turn_id") if isinstance(item, Mapping) else None
            for item in candidate_order
        ]
        != expected_candidate_ids
    ):
        raise RepresentationPackingBridgeError("packer candidate evidence lost hydrated IDs")

    kept_ids = packed.trace.get("kept_ids")
    if not isinstance(kept_ids, list):
        raise RepresentationPackingBridgeError("packer returned malformed kept IDs")
    kept_keys = {tuple(item) for item in kept_ids if isinstance(item, list)}
    if len(kept_keys) != len(kept_ids):
        raise RepresentationPackingBridgeError("packer returned malformed or duplicate kept IDs")
    expected_kept = [value for value in values if tuple(value.turn.turn_id.as_tuple()) in kept_keys]
    if [_turn_id_payload(value) for value in expected_kept] != kept_ids:
        raise RepresentationPackingBridgeError("packer changed kept-value relative order")
    history = (
        LINEAR_TURN_SEPARATOR.join(value.raw_value for value in expected_kept)
        if expected_kept
        else EMPTY_CONTEXT_NOTE
    )
    history_bytes = history.encode("utf-8")
    expected_history = {
        "sha256": hashlib.sha256(history_bytes).hexdigest(),
        "utf8_bytes": len(history_bytes),
    }
    if packed.trace.get("final_history") != expected_history:
        raise RepresentationPackingBridgeError("packed context bytes differ from hydrated values")
    if packed.prompt != OFFICIAL_ANSWER_TEMPLATE.format(history, current_date, question):
        raise RepresentationPackingBridgeError("packed prompt rewrote the exact hydrated context")
    final = packed.trace.get("final_prompt")
    if not isinstance(final, Mapping) or final.get("tokens") is None:
        raise RepresentationPackingBridgeError("packed prompt lacks an exact final token receipt")
    if final.get("tokens") > PRIMARY_TOKEN_BUDGET or final.get("within_budget") is not True:
        raise RepresentationPackingBridgeError("packed prompt exceeds the exact primary budget")

    hydrated = _hydrated_bindings(values)
    trace = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "classification": "evaluation-only-content-free-e6-packing-bridge",
        "production_configuration": False,
        "preflight": {
            "manifest_sha256": manifest.manifest_sha256,
            "case_index": case.case_index,
            "question_id": case.question_id,
            "source_artifact_sha256": manifest.dataset.source_sha256,
            "projection_sha256": manifest.projection_sha256,
            "source_record_sha256": case.source_record_sha256,
            "tokenizer_identity_sha256": manifest.tokenizer.identity.identity_sha256,
        },
        "representation": {
            "cell": result.cell.value,
            "trace_sha256": result.trace_sha256,
            "hydrated_value_count": len(values),
            "hydrated_values": hydrated,
            "hydrated_value_order_sha256": sha256_json(hydrated),
        },
        "packing": {
            "trace_sha256": packed.trace_sha256,
            "token_budget": PRIMARY_TOKEN_BUDGET,
            "candidate_ids": expected_candidate_ids,
            "kept_ids": kept_ids,
            "dropped_ids": packed.trace.get("dropped_ids"),
            "final_context": expected_history,
            "final_prompt": dict(final),
        },
        "claims": {
            "gold_fields_consumed": False,
            "derived_navigation_keys_delivered_to_reader": False,
            "hydrated_source_value_bytes_rewritten": False,
            "hydrated_candidate_order_mutated": False,
            "complete_official_reader_prompt_counted": True,
            "exact_primary_8192_token_budget": True,
            "reader_or_judge_executed": False,
            "production_policy_changed": False,
        },
    }
    return RepresentationPromptPackingResult(packed=packed, trace=trace)


__all__ = [
    "ARTIFACT_TYPE",
    "PROTOCOL_VERSION",
    "RepresentationPackingBridgeError",
    "RepresentationPromptPackingResult",
    "SCHEMA_VERSION",
    "pack_representation_result",
]
