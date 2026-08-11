"""Frozen contracts for offline paired LongMemEval selection QA evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

RUN_SCHEMA_VERSION = 1
CASE_SCHEMA_VERSION = 2
REPORT_SCHEMA_VERSION = 2

RUN_ARTIFACT_TYPE = "swarmbrain-longmemeval-selection-qa-run"
CASE_ARTIFACT_TYPE = "swarmbrain-longmemeval-selection-qa-case"
REPORT_ARTIFACT_TYPE = "swarmbrain-longmemeval-selection-qa-report"
PROTOCOL_VERSION = "swarmbrain-longmemeval-selection-qa-paired-v2"

BASELINE_ARM = "baseline"
CANDIDATE_ARM = "candidate"
ARMS = (BASELINE_ARM, CANDIDATE_ARM)

PRIMARY_PROMPT_BUDGET = 8_192
BOOTSTRAP_METHOD = "stratified-percentile-paired-question-bootstrap-v1"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_809
BOOTSTRAP_CONFIDENCE = 0.95
MAX_TYPE_REGRESSION = 0.02

RECEIPT_AUTHENTICATION = "externally-attested-unsigned"
RECEIPT_ENVELOPE_SCHEMA = "swarmbrain-selection-receipt-envelope-v1"

HELDOUT_PREREGISTRATION_SCHEMA_VERSION = 1
HELDOUT_PREREGISTRATION_ARTIFACT_TYPE = "swarmbrain-selection-heldout-preregistration"
HELDOUT_CONFIRMATION_SCHEMA_VERSION = 1
HELDOUT_CONFIRMATION_ARTIFACT_TYPE = "swarmbrain-selection-heldout-confirmation"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

E1_A_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e1-a-input-v1"
E1_B_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e1-b-input-v1"
E1_C_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e1-c-input-v1"
E1_D_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e1-d-input-v1"
E2_A_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e2-a-input-v1"
E2_B_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e2-b-input-v1"
E2_C_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e2-c-input-v1"
E2_D_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e2-d-input-v1"
E2_E_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e2-e-input-v1"
E6_R0_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e6-r0-input-v1"
E6_R1_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e6-r1-input-v1"
E6_R2_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e6-r2-input-v1"
E6_R3_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e6-r3-input-v1"
E6_R4_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e6-r4-input-v1"
E6_R5_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e6-r5-input-v1"
E6_R_NEG_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e6-r-neg-input-v1"
E7_A_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e7-a-input-v1"
E7_B_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e7-b-input-v1"
E7_C_INPUT_PROFILE = "swarmbrain-longmemeval-selection-e7-c-input-v1"

COMMON_SELECTION_INPUT_FIELDS = (
    "routing.question_id",
    "query.question",
    "corpus.turn.canonical_id",
    "corpus.turn.serialized_document",
)
E1_A_SELECTION_INPUT_FIELDS = COMMON_SELECTION_INPUT_FIELDS + (
    "retrieval.lexical.raw_score.attested",
    "retrieval.dense.raw_score.attested",
)
E1_B_SELECTION_INPUT_FIELDS = COMMON_SELECTION_INPUT_FIELDS + (
    "ranking.e1_a.rrf_score.derived",
    "reranker.cross_encoder.logit.attested",
)
E1_C_SELECTION_INPUT_FIELDS = E1_B_SELECTION_INPUT_FIELDS + ("reranker.colbert.raw_score.attested",)
E1_D_SELECTION_INPUT_FIELDS = E1_C_SELECTION_INPUT_FIELDS + (
    "policy.cross_encoder.relative_threshold.deterministic_metadata",
)
E2_BASE_SELECTION_INPUT_FIELDS = COMMON_SELECTION_INPUT_FIELDS + (
    "ranking.e1_a.rrf_score.derived",
    "embedding.query_turn.cosine.attested",
    "policy.e2.fixed_e1_a_top20.deterministic_metadata",
)
E2_A_SELECTION_INPUT_FIELDS = E2_BASE_SELECTION_INPUT_FIELDS
E2_B_SELECTION_INPUT_FIELDS = E2_BASE_SELECTION_INPUT_FIELDS + (
    "policy.e2.query_only_adaptive_path_truncation.deterministic_metadata",
)
E2_PRODUCT_SELECTION_INPUT_FIELDS = E2_BASE_SELECTION_INPUT_FIELDS + (
    "embedding.chain_context_turn.cosine.attested",
    "policy.e2.multiplicative_gate.deterministic_metadata",
)
E2_C_SELECTION_INPUT_FIELDS = E2_PRODUCT_SELECTION_INPUT_FIELDS
E2_D_SELECTION_INPUT_FIELDS = E2_PRODUCT_SELECTION_INPUT_FIELDS + (
    "policy.e2.adaptive_path_truncation.deterministic_metadata",
)
E2_E_SELECTION_INPUT_FIELDS = E2_D_SELECTION_INPUT_FIELDS + (
    "policy.e2.cross_chain_render_dedup.deterministic_metadata",
)

E6_RAW_RANKING_INPUT_FIELDS = ("ranking.representation.raw.raw_score.attested",)
E6_MERGED_SFK_INPUT_FIELDS = (
    "construction.representation.merged_sfk.source_only_receipt.attested",
    "representation.merged_sfk.derived_key.attested",
    "ranking.representation.merged_sfk.raw_score.attested",
)
E6_SUMMARY_INPUT_FIELDS = (
    "construction.representation.summary.source_only_receipt.attested",
    "representation.summary.derived_key.attested",
    "ranking.representation.summary.raw_score.attested",
)
E6_FACT_INPUT_FIELDS = (
    "construction.representation.fact.source_only_receipt.attested",
    "representation.fact.derived_key.attested",
    "ranking.representation.fact.raw_score.attested",
)
E6_KEYWORD_INPUT_FIELDS = (
    "construction.representation.keyword.source_only_receipt.attested",
    "representation.keyword.derived_key.attested",
    "ranking.representation.keyword.raw_score.attested",
)
E6_PRIMARY_ABSTRACTION_INPUT_FIELDS = (
    "construction.representation.primary_abstraction.source_only_receipt.attested",
    "representation.primary_abstraction.derived_key.attested",
    "ranking.representation.primary_abstraction.raw_score.attested",
)
E6_CUE_ANCHOR_INPUT_FIELDS = (
    "construction.representation.cue_anchor.source_only_receipt.attested",
    "representation.cue_anchor.derived_key.attested",
    "ranking.representation.cue_anchor.raw_score.attested",
)
E6_ENTITY_DESCRIPTION_INPUT_FIELDS = (
    "construction.representation.entity_description.source_only_receipt.attested",
    "representation.entity_description.derived_key.attested",
    "ranking.representation.entity_description.raw_score.attested",
)
E6_R0_SELECTION_INPUT_FIELDS = COMMON_SELECTION_INPUT_FIELDS + E6_RAW_RANKING_INPUT_FIELDS
E6_R1_SELECTION_INPUT_FIELDS = E6_R0_SELECTION_INPUT_FIELDS + E6_MERGED_SFK_INPUT_FIELDS
E6_R2_SELECTION_INPUT_FIELDS = (
    E6_R0_SELECTION_INPUT_FIELDS
    + E6_SUMMARY_INPUT_FIELDS
    + E6_FACT_INPUT_FIELDS
    + E6_KEYWORD_INPUT_FIELDS
)
E6_R3_SELECTION_INPUT_FIELDS = (
    COMMON_SELECTION_INPUT_FIELDS + E6_PRIMARY_ABSTRACTION_INPUT_FIELDS + E6_CUE_ANCHOR_INPUT_FIELDS
)
E6_R4_SELECTION_INPUT_FIELDS = COMMON_SELECTION_INPUT_FIELDS + E6_ENTITY_DESCRIPTION_INPUT_FIELDS
E6_R5_SELECTION_INPUT_FIELDS = E6_R4_SELECTION_INPUT_FIELDS + (
    "construction.representation.entity_adjacency.source_safe_receipt.attested",
    "representation.entity_adjacency.source_safe_graph.attested",
    "policy.representation.one_hop_entity_activation.deterministic_metadata",
)
E6_R_NEG_SELECTION_INPUT_FIELDS = E6_R0_SELECTION_INPUT_FIELDS + (
    "construction.representation.similarity_adjacency.source_safe_receipt.attested",
    "representation.similarity_adjacency.source_safe_graph.attested",
    "representation.similarity_adjacency.edge_score.attested",
    "policy.representation.one_hop_similarity_activation.deterministic_metadata",
)
E7_BASE_SELECTION_INPUT_FIELDS = COMMON_SELECTION_INPUT_FIELDS + (
    "ranking.e1_b.top50.selection_trace_sha256.derived",
    "preflight.official.manifest_sha256.derived",
    "construction.e7.window_batch.trace_sha256.derived",
)
E7_OUTPUT_SELECTION_INPUT_FIELDS = (
    "construction.e7.normalized_constructor_receipts_sha256.derived",
    "construction.e7.reader_context_sha256.derived",
    "construction.e7.grounded_vs_abstractive_claim.derived",
)
E7_A_SELECTION_INPUT_FIELDS = (
    E7_BASE_SELECTION_INPUT_FIELDS
    + ("policy.e7.constructor_disabled.deterministic_metadata",)
    + E7_OUTPUT_SELECTION_INPUT_FIELDS
)
E7_B_SELECTION_INPUT_FIELDS = (
    E7_BASE_SELECTION_INPUT_FIELDS
    + ("policy.e7.abstractive_output_permitted.deterministic_metadata",)
    + E7_OUTPUT_SELECTION_INPUT_FIELDS
)
E7_C_SELECTION_INPUT_FIELDS = (
    E7_BASE_SELECTION_INPUT_FIELDS
    + ("policy.e7.abstractive_output_forbidden.deterministic_metadata",)
    + E7_OUTPUT_SELECTION_INPUT_FIELDS
)
SELECTION_INPUT_PROFILES = {
    E1_A_INPUT_PROFILE: E1_A_SELECTION_INPUT_FIELDS,
    E1_B_INPUT_PROFILE: E1_B_SELECTION_INPUT_FIELDS,
    E1_C_INPUT_PROFILE: E1_C_SELECTION_INPUT_FIELDS,
    E1_D_INPUT_PROFILE: E1_D_SELECTION_INPUT_FIELDS,
    E2_A_INPUT_PROFILE: E2_A_SELECTION_INPUT_FIELDS,
    E2_B_INPUT_PROFILE: E2_B_SELECTION_INPUT_FIELDS,
    E2_C_INPUT_PROFILE: E2_C_SELECTION_INPUT_FIELDS,
    E2_D_INPUT_PROFILE: E2_D_SELECTION_INPUT_FIELDS,
    E2_E_INPUT_PROFILE: E2_E_SELECTION_INPUT_FIELDS,
    E6_R0_INPUT_PROFILE: E6_R0_SELECTION_INPUT_FIELDS,
    E6_R1_INPUT_PROFILE: E6_R1_SELECTION_INPUT_FIELDS,
    E6_R2_INPUT_PROFILE: E6_R2_SELECTION_INPUT_FIELDS,
    E6_R3_INPUT_PROFILE: E6_R3_SELECTION_INPUT_FIELDS,
    E6_R4_INPUT_PROFILE: E6_R4_SELECTION_INPUT_FIELDS,
    E6_R5_INPUT_PROFILE: E6_R5_SELECTION_INPUT_FIELDS,
    E6_R_NEG_INPUT_PROFILE: E6_R_NEG_SELECTION_INPUT_FIELDS,
    E7_A_INPUT_PROFILE: E7_A_SELECTION_INPUT_FIELDS,
    E7_B_INPUT_PROFILE: E7_B_SELECTION_INPUT_FIELDS,
    E7_C_INPUT_PROFILE: E7_C_SELECTION_INPUT_FIELDS,
}

E7_PROTOCOL_EVIDENCE_FIELDS = (
    "source_e1_b_selection_trace_sha256",
    "preflight_manifest_sha256",
    "window_batch_trace_sha256",
    "normalized_constructor_receipts_sha256",
    "normalized_constructor_receipt_count",
    "constructor_receipt_authentication",
    "reader_context_sha256",
    "grounding_claim",
    "authenticated_faithfulness_evidence_sha256",
)
E7_RAW_GROUNDING_CLAIM = "raw-source-byte-grounded"
E7_BYTE_GROUNDED_CLAIM = "byte-grounded-no-abstractive-output"
E7_ABSTRACTIVE_GROUNDING_CLAIM = "caller-attested-abstractive-output-faithfulness-unproven"
E7_NO_CONSTRUCTOR_RECEIPTS = "not-applicable-zero-constructor-receipts"
INTRINSICALLY_INELIGIBLE_CANDIDATE_PROFILES = (
    E6_R_NEG_INPUT_PROFILE,
    E7_B_INPUT_PROFILE,
)


class LongMemEvalSelectionEvidenceError(ValueError):
    """Evidence cannot support the frozen paired selection QA report."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode finite JSON deterministically without changing Unicode text."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LongMemEvalSelectionEvidenceError(
            f"value is not finite canonical UTF-8 JSON: {exc}"
        ) from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_utf8(value: str) -> str:
    try:
        return sha256_bytes(value.encode("utf-8"))
    except UnicodeError as exc:
        raise LongMemEvalSelectionEvidenceError("text must be valid UTF-8") from exc


def required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must be a non-empty string without surrounding whitespace"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise LongMemEvalSelectionEvidenceError(f"{label} cannot contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise LongMemEvalSelectionEvidenceError(f"{label} must be valid UTF-8") from exc
    return value


def sha256_text(value: Any, *, label: str) -> str:
    text = required_text(value, label=label)
    if _SHA256_RE.fullmatch(text) is None:
        raise LongMemEvalSelectionEvidenceError(
            f"{label} must be a lowercase hexadecimal SHA-256 digest"
        )
    return text


def integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LongMemEvalSelectionEvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def finite_number(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LongMemEvalSelectionEvidenceError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise LongMemEvalSelectionEvidenceError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise LongMemEvalSelectionEvidenceError(f"{label} must be >= {minimum}")
    return 0.0 if result == 0.0 else result


def selection_input_profile_digest(profile: str) -> str:
    """Digest one registered ordered input allowlist and its version name."""

    checked = required_text(profile, label="selection input profile")
    fields = SELECTION_INPUT_PROFILES.get(checked)
    if fields is None:
        raise LongMemEvalSelectionEvidenceError(
            f"selection input profile {checked!r} is not registered"
        )
    return sha256_json({"profile": checked, "ordered_fields": list(fields)})


def receipt_envelope_sha256(
    *,
    case_index: int,
    question_id: str,
    arm: str,
    stage: str,
    identities: dict[str, str],
    request_id: str,
    provider_request_id: str,
    request_sha256: str,
    response_sha256: str,
    outcome: dict[str, Any],
    accounting: dict[str, int | float],
) -> str:
    """Bind every content-free field in one externally attested call receipt."""

    integer(case_index, label="receipt envelope case_index")
    required_text(question_id, label="receipt envelope question_id")
    if arm not in ARMS:
        raise LongMemEvalSelectionEvidenceError("receipt envelope arm is not registered")
    if stage not in {"reader", "judge"}:
        raise LongMemEvalSelectionEvidenceError("receipt envelope stage must be reader or judge")
    if not isinstance(identities, dict) or not identities:
        raise LongMemEvalSelectionEvidenceError("receipt envelope identities must be non-empty")
    for name, digest in identities.items():
        required_text(name, label="receipt envelope identity role")
        sha256_text(digest, label=f"receipt envelope identity {name}")
    required_text(request_id, label="receipt envelope request_id")
    required_text(provider_request_id, label="receipt envelope provider_request_id")
    sha256_text(request_sha256, label="receipt envelope request_sha256")
    sha256_text(response_sha256, label="receipt envelope response_sha256")
    if not isinstance(outcome, dict) or not outcome:
        raise LongMemEvalSelectionEvidenceError("receipt envelope outcome must be non-empty")
    if not isinstance(accounting, dict) or not accounting:
        raise LongMemEvalSelectionEvidenceError("receipt envelope accounting must be non-empty")
    return sha256_json(
        {
            "schema": RECEIPT_ENVELOPE_SCHEMA,
            "case_index": case_index,
            "question_id": question_id,
            "arm": arm,
            "stage": stage,
            "identities": identities,
            "request_id": request_id,
            "provider_request_id": provider_request_id,
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
            "outcome": outcome,
            "accounting": accounting,
        }
    )


def fixed_protocol() -> dict[str, Any]:
    """Return the immutable comparison and promotion design."""

    return {
        "paired": True,
        "arms": list(ARMS),
        "coverage": "complete-official-cleaned-dataset",
        "primary_prompt_budget_tokens": PRIMARY_PROMPT_BUDGET,
        "prompt_budget_includes_full_prompt": True,
        "whole_turns_indivisible": True,
        "oversized_turn_policy": "skip-and-continue",
        "shared_reader_judge_tokenizer_serializer": True,
        "selection_input_profiles": {
            profile: {
                "ordered_fields": list(fields),
                "profile_sha256": selection_input_profile_digest(profile),
            }
            for profile, fields in SELECTION_INPUT_PROFILES.items()
        },
        "receipt_envelope_schema": RECEIPT_ENVELOPE_SCHEMA,
        "construction_plus_query_cost": ("embedding_cost_plus_reranker_cost_plus_constructor_cost"),
        "reader_cost": "reported-separately-and-included-in-total-cost",
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "strata": "question_type",
            "unit": "question",
            "paired": True,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": BOOTSTRAP_CONFIDENCE,
        },
        "promotion": {
            "qa_ci_lower_strictly_above_zero": True,
            "maximum_question_type_regression": MAX_TYPE_REGRESSION,
            "any_gold_noninferior": True,
            "all_metrics_reported": True,
            "pareto_dimensions": [
                "qa_accuracy:higher",
                "p95_prompt_tokens:lower",
                "p95_operational_latency_ms:lower",
                "construction_plus_query_cost_usd:lower",
            ],
            "structurally_preregistered_byte_distinct_heldout_directional_qa_required": True,
            "canonical_official_primary_required": True,
            "intrinsically_ineligible_candidate_profiles": list(
                INTRINSICALLY_INELIGIBLE_CANDIDATE_PROFILES
            ),
            "e7_b_requires_authenticated_abstractive_faithfulness": True,
            "authenticated_faithfulness_verifier_available_in_v2": False,
            "serving_promotion_eligible_from_offline_compiler": False,
        },
    }


__all__ = [
    "ARMS",
    "BASELINE_ARM",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_METHOD",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CANDIDATE_ARM",
    "CASE_ARTIFACT_TYPE",
    "CASE_SCHEMA_VERSION",
    "COMMON_SELECTION_INPUT_FIELDS",
    "E1_A_INPUT_PROFILE",
    "E1_A_SELECTION_INPUT_FIELDS",
    "E1_B_INPUT_PROFILE",
    "E1_B_SELECTION_INPUT_FIELDS",
    "E1_C_INPUT_PROFILE",
    "E1_C_SELECTION_INPUT_FIELDS",
    "E1_D_INPUT_PROFILE",
    "E1_D_SELECTION_INPUT_FIELDS",
    "E2_A_INPUT_PROFILE",
    "E2_A_SELECTION_INPUT_FIELDS",
    "E2_B_INPUT_PROFILE",
    "E2_B_SELECTION_INPUT_FIELDS",
    "E2_C_INPUT_PROFILE",
    "E2_C_SELECTION_INPUT_FIELDS",
    "E2_D_INPUT_PROFILE",
    "E2_D_SELECTION_INPUT_FIELDS",
    "E2_E_INPUT_PROFILE",
    "E2_E_SELECTION_INPUT_FIELDS",
    "E6_R0_INPUT_PROFILE",
    "E6_R0_SELECTION_INPUT_FIELDS",
    "E6_R1_INPUT_PROFILE",
    "E6_R1_SELECTION_INPUT_FIELDS",
    "E6_R2_INPUT_PROFILE",
    "E6_R2_SELECTION_INPUT_FIELDS",
    "E6_R3_INPUT_PROFILE",
    "E6_R3_SELECTION_INPUT_FIELDS",
    "E6_R4_INPUT_PROFILE",
    "E6_R4_SELECTION_INPUT_FIELDS",
    "E6_R5_INPUT_PROFILE",
    "E6_R5_SELECTION_INPUT_FIELDS",
    "E6_R_NEG_INPUT_PROFILE",
    "E6_R_NEG_SELECTION_INPUT_FIELDS",
    "E7_ABSTRACTIVE_GROUNDING_CLAIM",
    "E7_A_INPUT_PROFILE",
    "E7_A_SELECTION_INPUT_FIELDS",
    "E7_B_INPUT_PROFILE",
    "E7_B_SELECTION_INPUT_FIELDS",
    "E7_BYTE_GROUNDED_CLAIM",
    "E7_C_INPUT_PROFILE",
    "E7_C_SELECTION_INPUT_FIELDS",
    "E7_NO_CONSTRUCTOR_RECEIPTS",
    "E7_PROTOCOL_EVIDENCE_FIELDS",
    "E7_RAW_GROUNDING_CLAIM",
    "HELDOUT_CONFIRMATION_ARTIFACT_TYPE",
    "HELDOUT_CONFIRMATION_SCHEMA_VERSION",
    "HELDOUT_PREREGISTRATION_ARTIFACT_TYPE",
    "HELDOUT_PREREGISTRATION_SCHEMA_VERSION",
    "LongMemEvalSelectionEvidenceError",
    "INTRINSICALLY_INELIGIBLE_CANDIDATE_PROFILES",
    "MAX_TYPE_REGRESSION",
    "PRIMARY_PROMPT_BUDGET",
    "PROTOCOL_VERSION",
    "RECEIPT_AUTHENTICATION",
    "RECEIPT_ENVELOPE_SCHEMA",
    "REPORT_ARTIFACT_TYPE",
    "REPORT_SCHEMA_VERSION",
    "RUN_ARTIFACT_TYPE",
    "RUN_SCHEMA_VERSION",
    "SELECTION_INPUT_PROFILES",
    "canonical_json_bytes",
    "finite_number",
    "fixed_protocol",
    "integer",
    "required_text",
    "receipt_envelope_sha256",
    "selection_input_profile_digest",
    "sha256_bytes",
    "sha256_json",
    "sha256_text",
    "sha256_utf8",
]
