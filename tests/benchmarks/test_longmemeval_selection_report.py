from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from benchmarks.integrations.longmemeval_selection_report import (
    BASELINE_ARM,
    CANDIDATE_ARM,
    CASE_ARTIFACT_TYPE,
    CASE_SCHEMA_VERSION,
    E1_A_INPUT_PROFILE,
    E1_B_INPUT_PROFILE,
    E1_C_INPUT_PROFILE,
    E1_D_INPUT_PROFILE,
    E2_A_INPUT_PROFILE,
    E2_B_INPUT_PROFILE,
    E2_C_INPUT_PROFILE,
    E2_D_INPUT_PROFILE,
    E2_E_INPUT_PROFILE,
    E6_R0_INPUT_PROFILE,
    E6_R1_INPUT_PROFILE,
    E6_R2_INPUT_PROFILE,
    E6_R3_INPUT_PROFILE,
    E6_R4_INPUT_PROFILE,
    E6_R5_INPUT_PROFILE,
    E6_R_NEG_INPUT_PROFILE,
    E7_A_INPUT_PROFILE,
    E7_ABSTRACTIVE_GROUNDING_CLAIM,
    E7_B_INPUT_PROFILE,
    E7_BYTE_GROUNDED_CLAIM,
    E7_C_INPUT_PROFILE,
    E7_NO_CONSTRUCTOR_RECEIPTS,
    E7_RAW_GROUNDING_CLAIM,
    HELDOUT_CONFIRMATION_ARTIFACT_TYPE,
    HELDOUT_CONFIRMATION_SCHEMA_VERSION,
    HELDOUT_PREREGISTRATION_ARTIFACT_TYPE,
    HELDOUT_PREREGISTRATION_SCHEMA_VERSION,
    PRIMARY_PROMPT_BUDGET,
    PROTOCOL_VERSION,
    RECEIPT_AUTHENTICATION,
    REPORT_ARTIFACT_TYPE,
    SELECTION_INPUT_PROFILES,
    LongMemEvalSelectionEvidenceError,
    build_run_manifest,
    compile_longmemeval_selection_report,
    receipt_envelope_sha256,
    selection_input_profile_digest,
    sha256_json,
    sha256_utf8,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTION_TYPES = ("temporal-reasoning", "knowledge-update")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _binding(path: Path, root: Path, *, count_field: str, count: int) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        count_field: count,
    }


def _file_binding(path: Path, root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _record(index: int) -> dict[str, Any]:
    question_id = f"question-{index:03d}"
    session_ids = [f"session-{index:03d}-a", f"session-{index:03d}-b"]
    return {
        "question_id": question_id,
        "question_type": QUESTION_TYPES[index // 4],
        "question": f"SECRET QUESTION {index} Żółć",
        "question_date": "2026/08/09 (Sun) 12:00",
        "answer": f"SECRET ANSWER {index}",
        "answer_session_ids": [session_ids[0]],
        "haystack_session_ids": session_ids,
        "haystack_dates": [
            "2026/08/01 (Sat) 00:00",
            "2026/08/02 (Sun) 00:00",
        ],
        "haystack_sessions": [
            [
                {"role": "user", "content": f"SECRET CONTEXT {index}/a"},
                {"role": "assistant", "content": "stored a"},
            ],
            [
                {"role": "user", "content": f"SECRET CONTEXT {index}/b"},
                {"role": "assistant", "content": "stored b"},
            ],
        ],
    }


def _identities() -> dict[str, dict[str, str]]:
    return {
        "baseline_cell": {
            "name": "fixed-fusion-session-baseline",
            "version": "fixture-v1",
            "artifact_sha256": _digest("baseline-cell"),
            "selection_input_profile": E1_A_INPUT_PROFILE,
            "selection_input_profile_sha256": selection_input_profile_digest(E1_A_INPUT_PROFILE),
        },
        "candidate_cell": {
            "name": "turn-selection-candidate",
            "version": "fixture-v1",
            "artifact_sha256": _digest("candidate-cell"),
            "selection_input_profile": E1_B_INPUT_PROFILE,
            "selection_input_profile_sha256": selection_input_profile_digest(E1_B_INPUT_PROFILE),
        },
        "prompt_serializer": {
            "name": "fixture-exact-prompt-serializer",
            "version": "v1",
            "artifact_sha256": _digest("serializer"),
            "tokenizer": "fixture-tokenizer",
            "tokenizer_revision": "revision-1",
            "tokenizer_artifact_sha256": _digest("tokenizer"),
        },
        "reader_identity": {
            "provider": "fixture-provider",
            "model": "fixture-reader",
            "revision": "reader-revision-1",
            "deployment_sha256": _digest("reader-deployment"),
            "tokenizer": "fixture-tokenizer",
            "tokenizer_revision": "revision-1",
            "tokenizer_artifact_sha256": _digest("tokenizer"),
            "prompt_template_sha256": _digest("reader-template"),
        },
        "judge_identity": {
            "provider": "fixture-provider",
            "model": "fixture-judge",
            "revision": "judge-revision-1",
            "deployment_sha256": _digest("judge-deployment"),
            "tokenizer": "fixture-judge-tokenizer",
            "tokenizer_revision": "judge-tokenizer-revision-1",
            "tokenizer_artifact_sha256": _digest("judge-tokenizer"),
            "prompt_template_sha256": _digest("judge-template"),
        },
    }


def _accounting(
    index: int,
    arm: str,
    *,
    reader_cost_usd: float = 0.01,
    constructor_calls: int = 0,
) -> dict[str, Any]:
    candidate_extra = 1 if arm == CANDIDATE_ARM else 0
    return {
        "embedding": {
            "calls": 1,
            "ingestion_tokens": 100 + index,
            "query_tokens": 10,
            "cost_usd": 0.001,
            "latency_ms": 2.0,
        },
        "reranker": {
            "calls": candidate_extra,
            "input_tokens": 20 * candidate_extra,
            "output_tokens": 0,
            "cost_usd": 0.001 * candidate_extra,
            "latency_ms": 1.0 * candidate_extra,
        },
        "constructor": {
            "calls": constructor_calls,
            "input_tokens": 30 * constructor_calls,
            "output_tokens": 5 * constructor_calls,
            "cost_usd": 0.002 * constructor_calls,
            "latency_ms": 3.0 if constructor_calls else 0.0,
        },
        "reader": {
            "calls": 1,
            "input_tokens": 120 + index,
            "output_tokens": 12,
            "cost_usd": reader_cost_usd,
            "latency_ms": 20.0 + candidate_extra,
        },
        "judge": {
            "calls": 1,
            "input_tokens": 80,
            "output_tokens": 1,
            "cost_usd": 0.002,
            "latency_ms": 5.0,
        },
    }


def _selection_protocol_evidence(
    index: int,
    arm: str,
    profile: str,
    *,
    deliver_gold: bool,
) -> tuple[str, str, dict[str, Any]]:
    input_artifact_sha256 = _digest(f"selection-input/{index}/{arm}")
    context_sha256 = _digest(f"context/{index}/{arm}/{deliver_gold}")
    if profile not in {E7_A_INPUT_PROFILE, E7_B_INPUT_PROFILE, E7_C_INPUT_PROFILE}:
        return input_artifact_sha256, context_sha256, {}

    source_trace_sha256 = _digest(f"e1-b-top50/{index}/{arm}")
    if profile == E7_A_INPUT_PROFILE:
        receipt_count = 0
        receipt_authentication = E7_NO_CONSTRUCTOR_RECEIPTS
        receipts_sha256 = sha256_json([])
        grounding_claim = E7_RAW_GROUNDING_CLAIM
    else:
        receipt_count = 2
        receipt_authentication = RECEIPT_AUTHENTICATION
        receipts_sha256 = _digest(f"normalized-constructor-receipts/{index}/{arm}")
        grounding_claim = (
            E7_ABSTRACTIVE_GROUNDING_CLAIM
            if profile == E7_B_INPUT_PROFILE
            else E7_BYTE_GROUNDED_CLAIM
        )
    context_sha256 = _digest(f"e7-reader-context/{index}/{arm}/{deliver_gold}")
    return (
        source_trace_sha256,
        context_sha256,
        {
            "source_e1_b_selection_trace_sha256": source_trace_sha256,
            "preflight_manifest_sha256": _digest("e7-run-preflight-manifest"),
            "window_batch_trace_sha256": _digest(f"e7-window-batch/{index}/{arm}"),
            "normalized_constructor_receipts_sha256": receipts_sha256,
            "normalized_constructor_receipt_count": receipt_count,
            "constructor_receipt_authentication": receipt_authentication,
            "reader_context_sha256": context_sha256,
            "grounding_claim": grounding_claim,
            "authenticated_faithfulness_evidence_sha256": None,
        },
    )


def _arm(
    index: int,
    arm: str,
    record: dict[str, Any],
    identities: dict[str, dict[str, str]],
    *,
    correct: bool,
    deliver_gold: bool = True,
    reader_cost_usd: float = 0.01,
) -> dict[str, Any]:
    question_sha256 = sha256_utf8(record["question"])
    question_date_sha256 = sha256_utf8(record["question_date"])
    reference_answer_sha256 = sha256_json(record["answer"])
    corpus_sha256 = sha256_json(
        {
            "haystack_session_ids": record["haystack_session_ids"],
            "haystack_dates": record["haystack_dates"],
            "haystack_sessions": record["haystack_sessions"],
        }
    )
    pool_ids = list(record["haystack_session_ids"])
    pool_sha256 = sha256_json({"unit": "session", "candidate_ids": pool_ids})
    delivered = pool_ids if deliver_gold else [pool_ids[1]]
    cell_identity = identities[f"{arm}_cell"]
    cell_identity_sha256 = sha256_json(cell_identity)
    input_profile = cell_identity["selection_input_profile"]
    input_artifact_sha256, context_sha256, protocol_evidence = _selection_protocol_evidence(
        index,
        arm,
        input_profile,
        deliver_gold=deliver_gold,
    )
    serializer_sha256 = sha256_json(identities["prompt_serializer"])
    reader_identity_sha256 = sha256_json(identities["reader_identity"])
    judge_identity_sha256 = sha256_json(identities["judge_identity"])
    reader_material = sha256_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "serializer_identity_sha256": serializer_sha256,
            "reader_identity_sha256": reader_identity_sha256,
            "budget_tokens": PRIMARY_PROMPT_BUDGET,
            "question_sha256": question_sha256,
            "question_date_sha256": question_date_sha256,
            "candidate_pool_sha256": pool_sha256,
            "delivered_candidate_ids": delivered,
            "context_sha256": context_sha256,
        }
    )
    answer_sha256 = _digest(f"model-answer/{index}/{arm}")
    judge_material = sha256_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "judge_identity_sha256": judge_identity_sha256,
            "question_sha256": question_sha256,
            "reference_answer_sha256": reference_answer_sha256,
            "model_answer_sha256": answer_sha256,
        }
    )
    accounting = _accounting(
        index,
        arm,
        reader_cost_usd=reader_cost_usd,
        constructor_calls=int(protocol_evidence.get("normalized_constructor_receipt_count", 0)),
    )
    prompt_tokens = 100 + index + (1 if arm == CANDIDATE_ARM else 0)
    reader_prompt_sha256 = _digest(f"reader-prompt/{index}/{arm}")
    tokenized_prompt_sha256 = _digest(f"tokenized-prompt/{index}/{arm}")
    reader_request_id = f"reader-client-{index}-{arm}"
    reader_provider_id = f"reader-provider-{index}-{arm}"
    reader_request_sha256 = _digest(f"reader-request/{index}/{arm}")
    reader_response_sha256 = _digest(f"reader-response/{index}/{arm}")
    reader_envelope_sha256 = receipt_envelope_sha256(
        case_index=index,
        question_id=record["question_id"],
        arm=arm,
        stage="reader",
        identities={
            "cell": cell_identity_sha256,
            "prompt_serializer": serializer_sha256,
            "reader": reader_identity_sha256,
        },
        request_id=reader_request_id,
        provider_request_id=reader_provider_id,
        request_sha256=reader_request_sha256,
        response_sha256=reader_response_sha256,
        outcome={
            "prompt_material_sha256": reader_material,
            "prompt_sha256": reader_prompt_sha256,
            "tokenized_prompt_sha256": tokenized_prompt_sha256,
            "prompt_tokens": prompt_tokens,
            "answer_sha256": answer_sha256,
        },
        accounting=accounting["reader"],
    )
    judge_prompt_sha256 = _digest(f"judge-prompt/{index}/{arm}")
    judge_response_sha256 = _digest(f"judge-response/{index}/{arm}")
    judge_request_id = f"judge-client-{index}-{arm}"
    judge_provider_id = f"judge-provider-{index}-{arm}"
    judge_request_sha256 = _digest(f"judge-request/{index}/{arm}")
    judge_envelope_sha256 = receipt_envelope_sha256(
        case_index=index,
        question_id=record["question_id"],
        arm=arm,
        stage="judge",
        identities={
            "cell": cell_identity_sha256,
            "judge": judge_identity_sha256,
            "prompt_serializer": serializer_sha256,
            "reader": reader_identity_sha256,
        },
        request_id=judge_request_id,
        provider_request_id=judge_provider_id,
        request_sha256=judge_request_sha256,
        response_sha256=judge_response_sha256,
        outcome={
            "prompt_material_sha256": judge_material,
            "prompt_sha256": judge_prompt_sha256,
            "reference_answer_sha256": reference_answer_sha256,
            "model_answer_sha256": answer_sha256,
            "label": correct,
        },
        accounting=accounting["judge"],
    )
    return {
        "arm": arm,
        "cell_identity_sha256": cell_identity_sha256,
        "selection": {
            "query_sha256": question_sha256,
            "source_corpus_sha256": corpus_sha256,
            "input_artifact_sha256": input_artifact_sha256,
            "trace_sha256": _digest(f"selection-trace/{index}/{arm}"),
            "input_profile": input_profile,
            "input_profile_sha256": cell_identity["selection_input_profile_sha256"],
            "input_field_names": list(SELECTION_INPUT_PROFILES[input_profile]),
            "gold_fields_used": False,
            "protocol_evidence": protocol_evidence,
            "protocol_evidence_sha256": sha256_json(protocol_evidence),
        },
        "candidate_pool": {
            "unit": "session",
            "candidate_ids": pool_ids,
            "candidate_pool_sha256": pool_sha256,
        },
        "delivered_context": {
            "candidate_ids": delivered,
            "context_sha256": context_sha256,
        },
        "reader": {
            "identity_sha256": reader_identity_sha256,
            "serializer_identity_sha256": serializer_sha256,
            "budget_tokens": PRIMARY_PROMPT_BUDGET,
            "prompt_material_sha256": reader_material,
            "prompt_sha256": reader_prompt_sha256,
            "tokenized_prompt_sha256": tokenized_prompt_sha256,
            "prompt_tokens": prompt_tokens,
            "answer_sha256": answer_sha256,
            "request_id": reader_request_id,
            "provider_request_id": reader_provider_id,
            "request_sha256": reader_request_sha256,
            "response_sha256": reader_response_sha256,
            "receipt_envelope_sha256": reader_envelope_sha256,
            "receipt_authentication": RECEIPT_AUTHENTICATION,
        },
        "judge": {
            "identity_sha256": judge_identity_sha256,
            "prompt_material_sha256": judge_material,
            "prompt_sha256": judge_prompt_sha256,
            "response_sha256": judge_response_sha256,
            "request_id": judge_request_id,
            "provider_request_id": judge_provider_id,
            "request_sha256": judge_request_sha256,
            "reference_answer_sha256": reference_answer_sha256,
            "model_answer_sha256": answer_sha256,
            "label": correct,
            "receipt_envelope_sha256": judge_envelope_sha256,
            "receipt_authentication": RECEIPT_AUTHENTICATION,
        },
        "accounting": accounting,
    }


def _row(
    index: int,
    record: dict[str, Any],
    identities: dict[str, dict[str, str]],
) -> dict[str, Any]:
    # Each stratum is homogeneous, making a truly stratified bootstrap CI constant at 0.5.
    candidate_correct = record["question_type"] == QUESTION_TYPES[0]
    corpus_sha256 = sha256_json(
        {
            "haystack_session_ids": record["haystack_session_ids"],
            "haystack_dates": record["haystack_dates"],
            "haystack_sessions": record["haystack_sessions"],
        }
    )
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "artifact_type": CASE_ARTIFACT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "case_index": index,
        "question_id": record["question_id"],
        "question_type": record["question_type"],
        "dataset_record_sha256": sha256_json(record),
        "question_sha256": sha256_utf8(record["question"]),
        "question_date_sha256": sha256_utf8(record["question_date"]),
        "reference_answer_sha256": sha256_json(record["answer"]),
        "source_corpus_sha256": corpus_sha256,
        BASELINE_ARM: _arm(index, BASELINE_ARM, record, identities, correct=False),
        CANDIDATE_ARM: _arm(
            index,
            CANDIDATE_ARM,
            record,
            identities,
            correct=candidate_correct,
        ),
    }


@pytest.fixture
def evidence(tmp_path: Path) -> dict[str, Any]:
    records = [_record(index) for index in range(8)]
    identities = _identities()
    rows = [_row(index, record, identities) for index, record in enumerate(records)]
    dataset_path = tmp_path / "dataset.json"
    evidence_path = tmp_path / "evidence.jsonl"
    run_path = tmp_path / "run.json"
    _write_json(dataset_path, records)
    _write_jsonl(evidence_path, rows)
    run = build_run_manifest(
        created_at_utc="2026-08-09T12:00:00+00:00",
        experiment_id="fixture-selection-e1",
        dataset_path="dataset.json",
        evidence_path="evidence.jsonl",
        artifact_root=tmp_path,
        **identities,
    )
    _write_json(run_path, run)
    return {
        "root": tmp_path,
        "records": records,
        "identities": identities,
        "rows": rows,
        "dataset": dataset_path,
        "evidence": evidence_path,
        "run_path": run_path,
        "run": run,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    }


def _rebind(evidence: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    _write_jsonl(evidence["evidence"], rows)
    run = deepcopy(evidence["run"])
    run["evidence"] = _binding(
        evidence["evidence"], evidence["root"], count_field="rows", count=len(rows)
    )
    _write_json(evidence["run_path"], run)


def _replace_dataset(evidence: dict[str, Any], records: list[dict[str, Any]]) -> None:
    rows = [_row(index, record, evidence["identities"]) for index, record in enumerate(records)]
    _write_json(evidence["dataset"], records)
    _write_jsonl(evidence["evidence"], rows)
    run = build_run_manifest(
        created_at_utc="2026-08-09T12:00:00+00:00",
        experiment_id="fixture-selection-e1",
        dataset_path="dataset.json",
        evidence_path="evidence.jsonl",
        artifact_root=evidence["root"],
        **evidence["identities"],
    )
    _write_json(evidence["run_path"], run)
    evidence.update(
        {
            "records": records,
            "rows": rows,
            "run": run,
            "dataset_sha256": hashlib.sha256(evidence["dataset"].read_bytes()).hexdigest(),
        }
    )


def _configure_candidate_profile(evidence: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    identities = deepcopy(evidence["identities"])
    identities["candidate_cell"]["selection_input_profile"] = profile
    identities["candidate_cell"]["selection_input_profile_sha256"] = selection_input_profile_digest(
        profile
    )
    rows = [_row(index, record, identities) for index, record in enumerate(evidence["records"])]
    _write_jsonl(evidence["evidence"], rows)
    run = build_run_manifest(
        created_at_utc="2026-08-09T12:00:00+00:00",
        experiment_id=f"fixture-{profile}",
        dataset_path="dataset.json",
        evidence_path="evidence.jsonl",
        artifact_root=evidence["root"],
        **identities,
    )
    _write_json(evidence["run_path"], run)
    evidence.update({"identities": identities, "rows": rows, "run": run})
    return rows


def _compile(
    evidence: dict[str, Any],
    *,
    output: str = "report.json",
    heldout: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    return compile_longmemeval_selection_report(
        "run.json",
        output,
        artifact_root=evidence["root"],
        code_root=REPO_ROOT,
        heldout_confirmation_path=heldout,
        expected_dataset_sha256=evidence["dataset_sha256"],
        expected_question_count=len(evidence["records"]),
        overwrite=overwrite,
    )


def test_compiles_stratified_paired_qa_and_refuses_promotion_without_heldout(
    evidence: dict[str, Any],
) -> None:
    report = _compile(evidence)

    qa = report["metrics"]["qa"]
    assert qa["paired_delta"]["delta"] == pytest.approx(0.5)
    assert qa["paired_delta"]["ci_low"] == pytest.approx(0.5)
    assert qa["paired_delta"]["ci_high"] == pytest.approx(0.5)
    assert qa["bootstrap"] == {
        "method": "stratified-percentile-paired-question-bootstrap-v1",
        "strata": "question_type",
        "unit": "question",
        "paired": True,
        "resamples": 10_000,
        "seed": 20_260_809,
        "confidence": 0.95,
    }
    assert report["promotion"]["primary_track_passed"] is True
    assert report["promotion"]["canonical_official_primary_passed"] is False
    assert report["promotion"]["structural_offline_policy_eligible"] is False
    assert report["promotion"]["serving_promotion_eligible"] is False
    assert report["promotion"]["verdict"] == "refused-noncanonical-primary-dataset"
    assert report["validation"]["receipt_envelope_digests"] == 32
    assert report["validation"]["receipt_envelope_digests_globally_unique"] is True


def test_e1_input_profiles_are_exact_ordered_semantic_allowlists() -> None:
    common = [
        "routing.question_id",
        "query.question",
        "corpus.turn.canonical_id",
        "corpus.turn.serialized_document",
    ]
    assert list(SELECTION_INPUT_PROFILES[E1_A_INPUT_PROFILE]) == common + [
        "retrieval.lexical.raw_score.attested",
        "retrieval.dense.raw_score.attested",
    ]
    assert list(SELECTION_INPUT_PROFILES[E1_B_INPUT_PROFILE]) == common + [
        "ranking.e1_a.rrf_score.derived",
        "reranker.cross_encoder.logit.attested",
    ]
    assert list(SELECTION_INPUT_PROFILES[E1_C_INPUT_PROFILE]) == list(
        SELECTION_INPUT_PROFILES[E1_B_INPUT_PROFILE]
    ) + ["reranker.colbert.raw_score.attested"]
    assert list(SELECTION_INPUT_PROFILES[E1_D_INPUT_PROFILE]) == list(
        SELECTION_INPUT_PROFILES[E1_C_INPUT_PROFILE]
    ) + ["policy.cross_encoder.relative_threshold.deterministic_metadata"]


def test_e2_input_profiles_are_exact_ordered_semantic_allowlists() -> None:
    common = list(SELECTION_INPUT_PROFILES[E1_A_INPUT_PROFILE][:4])
    base = common + [
        "ranking.e1_a.rrf_score.derived",
        "embedding.query_turn.cosine.attested",
        "policy.e2.fixed_e1_a_top20.deterministic_metadata",
    ]
    product = base + [
        "embedding.chain_context_turn.cosine.attested",
        "policy.e2.multiplicative_gate.deterministic_metadata",
    ]
    assert list(SELECTION_INPUT_PROFILES[E2_A_INPUT_PROFILE]) == base
    assert list(SELECTION_INPUT_PROFILES[E2_B_INPUT_PROFILE]) == base + [
        "policy.e2.query_only_adaptive_path_truncation.deterministic_metadata"
    ]
    assert list(SELECTION_INPUT_PROFILES[E2_C_INPUT_PROFILE]) == product
    assert list(SELECTION_INPUT_PROFILES[E2_D_INPUT_PROFILE]) == product + [
        "policy.e2.adaptive_path_truncation.deterministic_metadata"
    ]
    assert list(SELECTION_INPUT_PROFILES[E2_E_INPUT_PROFILE]) == product + [
        "policy.e2.adaptive_path_truncation.deterministic_metadata",
        "policy.e2.cross_chain_render_dedup.deterministic_metadata",
    ]


def test_e6_input_profiles_are_exact_ordered_semantic_allowlists() -> None:
    common = list(SELECTION_INPUT_PROFILES[E1_A_INPUT_PROFILE][:4])
    raw = ["ranking.representation.raw.raw_score.attested"]

    def derived(family: str) -> list[str]:
        return [
            f"construction.representation.{family}.source_only_receipt.attested",
            f"representation.{family}.derived_key.attested",
            f"ranking.representation.{family}.raw_score.attested",
        ]

    assert list(SELECTION_INPUT_PROFILES[E6_R0_INPUT_PROFILE]) == common + raw
    assert list(SELECTION_INPUT_PROFILES[E6_R1_INPUT_PROFILE]) == (
        common + raw + derived("merged_sfk")
    )
    assert list(SELECTION_INPUT_PROFILES[E6_R2_INPUT_PROFILE]) == (
        common + raw + derived("summary") + derived("fact") + derived("keyword")
    )
    assert list(SELECTION_INPUT_PROFILES[E6_R3_INPUT_PROFILE]) == (
        common + derived("primary_abstraction") + derived("cue_anchor")
    )
    assert list(SELECTION_INPUT_PROFILES[E6_R4_INPUT_PROFILE]) == (
        common + derived("entity_description")
    )
    assert list(SELECTION_INPUT_PROFILES[E6_R5_INPUT_PROFILE]) == (
        common
        + derived("entity_description")
        + [
            "construction.representation.entity_adjacency.source_safe_receipt.attested",
            "representation.entity_adjacency.source_safe_graph.attested",
            "policy.representation.one_hop_entity_activation.deterministic_metadata",
        ]
    )
    assert list(SELECTION_INPUT_PROFILES[E6_R_NEG_INPUT_PROFILE]) == (
        common
        + raw
        + [
            "construction.representation.similarity_adjacency.source_safe_receipt.attested",
            "representation.similarity_adjacency.source_safe_graph.attested",
            "representation.similarity_adjacency.edge_score.attested",
            "policy.representation.one_hop_similarity_activation.deterministic_metadata",
        ]
    )

    e6_profiles = (
        E6_R0_INPUT_PROFILE,
        E6_R1_INPUT_PROFILE,
        E6_R2_INPUT_PROFILE,
        E6_R3_INPUT_PROFILE,
        E6_R4_INPUT_PROFILE,
        E6_R5_INPUT_PROFILE,
        E6_R_NEG_INPUT_PROFILE,
    )
    assert len({selection_input_profile_digest(profile) for profile in e6_profiles}) == len(
        e6_profiles
    )


def test_e7_input_profiles_bind_exact_construction_evidence_roles() -> None:
    common = list(SELECTION_INPUT_PROFILES[E1_A_INPUT_PROFILE][:4])
    base = common + [
        "ranking.e1_b.top50.selection_trace_sha256.derived",
        "preflight.official.manifest_sha256.derived",
        "construction.e7.window_batch.trace_sha256.derived",
    ]
    output = [
        "construction.e7.normalized_constructor_receipts_sha256.derived",
        "construction.e7.reader_context_sha256.derived",
        "construction.e7.grounded_vs_abstractive_claim.derived",
    ]
    assert (
        list(SELECTION_INPUT_PROFILES[E7_A_INPUT_PROFILE])
        == base + ["policy.e7.constructor_disabled.deterministic_metadata"] + output
    )
    assert (
        list(SELECTION_INPUT_PROFILES[E7_B_INPUT_PROFILE])
        == base + ["policy.e7.abstractive_output_permitted.deterministic_metadata"] + output
    )
    assert (
        list(SELECTION_INPUT_PROFILES[E7_C_INPUT_PROFILE])
        == base + ["policy.e7.abstractive_output_forbidden.deterministic_metadata"] + output
    )


def test_all_e1_e2_e6_e7_profile_names_digests_and_fields_are_unique() -> None:
    assert len(SELECTION_INPUT_PROFILES) == 19
    digests = [selection_input_profile_digest(profile) for profile in SELECTION_INPUT_PROFILES]
    assert len(digests) == len(set(digests))
    for fields in SELECTION_INPUT_PROFILES.values():
        assert len(fields) == len(set(fields))


@pytest.mark.parametrize(
    "profile",
    [
        E2_A_INPUT_PROFILE,
        E2_B_INPUT_PROFILE,
        E2_C_INPUT_PROFILE,
        E2_D_INPUT_PROFILE,
        E2_E_INPUT_PROFILE,
        E6_R0_INPUT_PROFILE,
        E6_R1_INPUT_PROFILE,
        E6_R2_INPUT_PROFILE,
        E6_R3_INPUT_PROFILE,
        E6_R4_INPUT_PROFILE,
        E6_R5_INPUT_PROFILE,
        E6_R_NEG_INPUT_PROFILE,
        E7_A_INPUT_PROFILE,
        E7_B_INPUT_PROFILE,
        E7_C_INPUT_PROFILE,
    ],
)
def test_manifest_admits_every_registered_e2_e6_and_e7_profile(
    evidence: dict[str, Any], profile: str
) -> None:
    identities = deepcopy(evidence["identities"])
    identities["candidate_cell"]["selection_input_profile"] = profile
    identities["candidate_cell"]["selection_input_profile_sha256"] = selection_input_profile_digest(
        profile
    )

    run = build_run_manifest(
        created_at_utc="2026-08-09T12:00:00+00:00",
        experiment_id=f"manifest-{profile}",
        dataset_path="dataset.json",
        evidence_path="evidence.jsonl",
        artifact_root=evidence["root"],
        **identities,
    )

    assert run["candidate_cell"]["selection_input_profile"] == profile
    assert run["protocol"]["selection_input_profiles"][profile] == {
        "ordered_fields": list(SELECTION_INPUT_PROFILES[profile]),
        "profile_sha256": selection_input_profile_digest(profile),
    }


@pytest.mark.parametrize(
    ("profile", "intrinsically_eligible"),
    [
        (E2_E_INPUT_PROFILE, True),
        (E6_R5_INPUT_PROFILE, True),
        (E6_R_NEG_INPUT_PROFILE, False),
        (E7_A_INPUT_PROFILE, True),
        (E7_B_INPUT_PROFILE, False),
        (E7_C_INPUT_PROFILE, True),
    ],
)
def test_compiler_admits_e2_e6_and_e7_profiles_without_e1_masquerading(
    evidence: dict[str, Any],
    profile: str,
    intrinsically_eligible: bool,
) -> None:
    _configure_candidate_profile(evidence, profile)

    report = _compile(evidence, output=f"{profile}.json")

    assert report["identities"]["candidate_cell"]["selection_input_profile"] == profile
    assert (
        report["promotion"]["candidate_intrinsic_eligibility"]["passed"] is intrinsically_eligible
    )
    if not intrinsically_eligible:
        assert report["promotion"]["structural_offline_policy_eligible"] is False
    intrinsic = report["promotion"]["candidate_intrinsic_eligibility"]
    if profile == E7_B_INPUT_PROFILE:
        assert intrinsic["authenticated_abstractive_faithfulness_required"] is True
        assert intrinsic["authenticated_abstractive_faithfulness_verified"] is False
    if profile == E7_C_INPUT_PROFILE:
        assert intrinsic["authenticated_abstractive_faithfulness_required"] is False


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("protocol_digest", "does not bind the exact protocol evidence"),
        ("missing_window_batch", "must contain exactly"),
        ("malformed_preflight", "must be a lowercase hexadecimal SHA-256 digest"),
        ("drifted_preflight", "one shared preflight manifest"),
        ("malformed_receipts", "must be a lowercase hexadecimal SHA-256 digest"),
        ("source_trace", "differs from selection.input_artifact_sha256"),
        ("reader_context", "differs from delivered context"),
        ("receipt_count", "differs from normalized E7 receipts"),
        ("receipt_authentication", "receipts must remain"),
        ("abstractive_claim", "byte-grounded E7 grounding claim"),
    ],
)
def test_e7_c_rejects_unbound_or_semantically_incompatible_construction_evidence(
    evidence: dict[str, Any], mutation: str, match: str
) -> None:
    rows = deepcopy(_configure_candidate_profile(evidence, E7_C_INPUT_PROFILE))
    selection = rows[0][CANDIDATE_ARM]["selection"]
    protocol_evidence = selection["protocol_evidence"]
    if mutation == "protocol_digest":
        selection["protocol_evidence_sha256"] = _digest("forged-e7-protocol-evidence")
    elif mutation == "missing_window_batch":
        protocol_evidence.pop("window_batch_trace_sha256")
    elif mutation == "malformed_preflight":
        protocol_evidence["preflight_manifest_sha256"] = "not-a-digest"
    elif mutation == "drifted_preflight":
        protocol_evidence["preflight_manifest_sha256"] = _digest("other-run-preflight")
    elif mutation == "malformed_receipts":
        protocol_evidence["normalized_constructor_receipts_sha256"] = "not-a-digest"
    elif mutation == "source_trace":
        protocol_evidence["source_e1_b_selection_trace_sha256"] = _digest(
            "other-e1-b-selection-trace"
        )
    elif mutation == "reader_context":
        protocol_evidence["reader_context_sha256"] = _digest("other-reader-context")
    elif mutation == "receipt_count":
        protocol_evidence["normalized_constructor_receipt_count"] += 1
    elif mutation == "receipt_authentication":
        protocol_evidence["constructor_receipt_authentication"] = "authenticated"
    else:
        protocol_evidence["grounding_claim"] = E7_ABSTRACTIVE_GROUNDING_CLAIM
    if mutation != "protocol_digest":
        selection["protocol_evidence_sha256"] = sha256_json(protocol_evidence)
    _rebind(evidence, rows)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match=match):
        _compile(evidence, output=f"e7-c-{mutation}.json")


@pytest.mark.parametrize(
    "field",
    [
        "source_e1_b_selection_trace_sha256",
        "window_batch_trace_sha256",
        "normalized_constructor_receipts_sha256",
    ],
)
def test_e7_rejects_question_bound_construction_digest_replay(
    evidence: dict[str, Any], field: str
) -> None:
    rows = deepcopy(_configure_candidate_profile(evidence, E7_C_INPUT_PROFILE))
    first = rows[0][CANDIDATE_ARM]["selection"]["protocol_evidence"]
    second_selection = rows[1][CANDIDATE_ARM]["selection"]
    second = second_selection["protocol_evidence"]
    second[field] = first[field]
    if field == "source_e1_b_selection_trace_sha256":
        second_selection["input_artifact_sha256"] = first[field]
    second_selection["protocol_evidence_sha256"] = sha256_json(second)
    _rebind(evidence, rows)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match="across questions"):
        _compile(evidence, output=f"e7-replayed-{field}.json")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("receipt_count", "E7-A must bind zero normalized constructor receipts"),
        ("receipt_digest", "E7-A constructor-receipt digest must bind the empty list"),
        ("grounding_claim", "E7-A must claim raw source-byte grounding"),
    ],
)
def test_e7_a_rejects_constructor_or_grounding_masquerades(
    evidence: dict[str, Any], mutation: str, match: str
) -> None:
    rows = deepcopy(_configure_candidate_profile(evidence, E7_A_INPUT_PROFILE))
    selection = rows[0][CANDIDATE_ARM]["selection"]
    protocol_evidence = selection["protocol_evidence"]
    if mutation == "receipt_count":
        protocol_evidence["normalized_constructor_receipt_count"] = 1
    elif mutation == "receipt_digest":
        protocol_evidence["normalized_constructor_receipts_sha256"] = _digest(
            "non-empty-constructor-receipts"
        )
    else:
        protocol_evidence["grounding_claim"] = E7_BYTE_GROUNDED_CLAIM
    selection["protocol_evidence_sha256"] = sha256_json(protocol_evidence)
    _rebind(evidence, rows)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match=match):
        _compile(evidence, output=f"e7-a-{mutation}.json")


def test_e7_b_cannot_self_attest_faithfulness_or_evade_profile_ineligibility(
    evidence: dict[str, Any],
) -> None:
    rows = deepcopy(_configure_candidate_profile(evidence, E7_B_INPUT_PROFILE))
    selection = rows[0][CANDIDATE_ARM]["selection"]
    protocol_evidence = selection["protocol_evidence"]
    protocol_evidence["authenticated_faithfulness_evidence_sha256"] = _digest(
        "caller-claimed-faithfulness"
    )
    selection["protocol_evidence_sha256"] = sha256_json(protocol_evidence)
    _rebind(evidence, rows)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match="cannot authenticate"):
        _compile(evidence, output="e7-b-fake-faithfulness.json")

    rows = deepcopy(_configure_candidate_profile(evidence, E7_B_INPUT_PROFILE))
    for row in rows:
        protocol_evidence = row[CANDIDATE_ARM]["selection"]["protocol_evidence"]
        protocol_evidence["grounding_claim"] = E7_BYTE_GROUNDED_CLAIM
        row[CANDIDATE_ARM]["selection"]["protocol_evidence_sha256"] = sha256_json(protocol_evidence)
    _rebind(evidence, rows)
    report = _compile(evidence, output="e7-b-byte-grounded.json")

    assert report["promotion"]["candidate_intrinsic_eligibility"]["passed"] is False
    assert report["promotion"]["structural_offline_policy_eligible"] is False


def test_non_e7_profile_cannot_smuggle_e7_protocol_evidence(evidence: dict[str, Any]) -> None:
    rows = deepcopy(evidence["rows"])
    selection = rows[0][CANDIDATE_ARM]["selection"]
    selection["protocol_evidence"] = {"reader_context_sha256": _digest("smuggled-context")}
    selection["protocol_evidence_sha256"] = sha256_json(selection["protocol_evidence"])
    _rebind(evidence, rows)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match="only for frozen E7 profiles"):
        _compile(evidence, output="non-e7-smuggled-evidence.json")


def test_e7_constructor_accounting_is_in_latency_cost_and_pareto_axes(
    evidence: dict[str, Any],
) -> None:
    _configure_candidate_profile(evidence, E7_C_INPUT_PROFILE)
    report = _compile(evidence, output="e7-constructor-accounting.json")
    candidate = report["metrics"]["efficiency"]["candidate"]

    assert candidate["accounting_totals"]["constructor"]["calls"] == 16
    assert candidate["accounting_totals"]["constructor"]["input_tokens"] == 480
    assert candidate["accounting_totals"]["constructor"]["output_tokens"] == 80
    assert candidate["accounting_totals"]["constructor"]["cost_usd"] == pytest.approx(0.032)
    assert candidate["construction_plus_query_cost_usd"]["total"] == pytest.approx(0.048)
    assert candidate["operational_latency_ms"]["p95"] == pytest.approx(27.0)
    assert report["validation"]["construction_plus_query_cost_formula"] == (
        "embedding_cost_plus_reranker_cost_plus_constructor_cost"
    )


def test_reports_exact_context_efficiency_and_zero_margin_noninferiority(
    evidence: dict[str, Any],
) -> None:
    report = _compile(evidence, output="metrics.json")

    context = report["metrics"]["gold_context"]
    assert context["baseline"]["any_gold_in_context"] == 1.0
    assert context["candidate"]["all_gold_in_context"] == 1.0
    assert context["candidate"]["answer_session_mrr"] == 1.0
    gate = report["promotion"]["primary_track_gates"]["any_gold_in_context_noninferior"]
    assert gate["noninferiority_margin"] == 0.0
    assert gate["passed"] is True
    efficiency = report["metrics"]["efficiency"]
    assert efficiency["baseline"]["prompt_tokens"]["total"] == sum(range(100, 108))
    assert efficiency["candidate"]["accounting_totals"]["reranker"]["calls"] == 8
    assert efficiency["baseline"]["accounting_totals"]["embedding"]["ingestion_tokens"] == sum(
        range(100, 108)
    )


def test_reader_cost_cannot_change_construction_query_cost_or_pareto_axis(
    evidence: dict[str, Any],
) -> None:
    before = _compile(evidence, output="before-reader-cost.json")
    rows = deepcopy(evidence["rows"])
    for index, record in enumerate(evidence["records"]):
        rows[index][CANDIDATE_ARM] = _arm(
            index,
            CANDIDATE_ARM,
            record,
            evidence["identities"],
            correct=rows[index][CANDIDATE_ARM]["judge"]["label"],
            reader_cost_usd=999.0,
        )
    _rebind(evidence, rows)
    after = _compile(evidence, output="after-reader-cost.json")

    before_efficiency = before["metrics"]["efficiency"]["candidate"]
    after_efficiency = after["metrics"]["efficiency"]["candidate"]
    assert (
        after_efficiency["construction_plus_query_cost_usd"]
        == before_efficiency["construction_plus_query_cost_usd"]
    )
    assert after_efficiency["reader_cost_usd"]["total"] == pytest.approx(8 * 999.0)
    assert (
        after_efficiency["total_cost_usd"]["total"] > before_efficiency["total_cost_usd"]["total"]
    )
    assert (
        after["promotion"]["primary_track_gates"]["pareto_non_dominated"]
        == before["promotion"]["primary_track_gates"]["pareto_non_dominated"]
    )


def test_report_is_content_free(evidence: dict[str, Any]) -> None:
    _compile(evidence, output="content-free.json")
    raw = (evidence["root"] / "content-free.json").read_text(encoding="utf-8")

    assert "SECRET QUESTION" not in raw
    assert "SECRET ANSWER" not in raw
    assert "SECRET CONTEXT" not in raw
    assert '"content_text_fields_omitted": true' in raw
    assert '"arbitrary_identifier_metadata_content_free_proven": false' in raw


def test_rejects_artifact_tampering_without_rebinding(evidence: dict[str, Any]) -> None:
    with evidence["evidence"].open("a", encoding="utf-8") as stream:
        stream.write(" ")

    with pytest.raises(LongMemEvalSelectionEvidenceError, match="bytes does not match|sha256"):
        _compile(evidence)


def test_manifest_builder_rejects_absolute_input_paths(evidence: dict[str, Any]) -> None:
    with pytest.raises(LongMemEvalSelectionEvidenceError, match="repository-local relative"):
        build_run_manifest(
            created_at_utc="2026-08-09T12:00:00+00:00",
            experiment_id="absolute-path-attack",
            dataset_path=evidence["dataset"],
            evidence_path="evidence.jsonl",
            artifact_root=evidence["root"],
            **evidence["identities"],
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("shared_artifact", "artifact_sha256 values must differ"),
        ("profile_digest", "does not bind its exact ordered allowlist"),
        ("unknown_profile", "not a frozen E1/E2/E6/E7 profile"),
        ("tokenizer_tuple", "must exactly equal the reader tokenizer tuple"),
    ],
)
def test_run_rejects_shared_cell_artifact_profile_digest_or_tokenizer_drift(
    evidence: dict[str, Any], mutation: str, match: str
) -> None:
    run = deepcopy(evidence["run"])
    if mutation == "shared_artifact":
        run["candidate_cell"]["artifact_sha256"] = run["baseline_cell"]["artifact_sha256"]
    elif mutation == "profile_digest":
        run["candidate_cell"]["selection_input_profile_sha256"] = _digest("wrong-profile")
    elif mutation == "unknown_profile":
        run["candidate_cell"]["selection_input_profile"] = "unregistered-profile"
    else:
        run["prompt_serializer"]["tokenizer_revision"] = "other-tokenizer-revision"
    _write_json(evidence["run_path"], run)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match=match):
        _compile(evidence)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "reordered"])
def test_rejects_incomplete_duplicate_or_reordered_pairs(
    evidence: dict[str, Any], mode: str
) -> None:
    rows = deepcopy(evidence["rows"])
    if mode == "missing":
        rows.pop()
    elif mode == "duplicate":
        rows[1] = deepcopy(rows[0])
    else:
        rows[0], rows[1] = rows[1], rows[0]
    _rebind(evidence, rows)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match="coverage|dataset/order"):
        _compile(evidence)


def test_rejects_altered_shared_query_or_corpus(evidence: dict[str, Any]) -> None:
    rows = deepcopy(evidence["rows"])
    rows[0][CANDIDATE_ARM]["selection"]["source_corpus_sha256"] = _digest("other-corpus")
    _rebind(evidence, rows)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match="shared question corpus"):
        _compile(evidence)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("gold_flag", "gold_fields_used must be exactly false"),
        ("identity", "identity_sha256 must be"),
        ("answer_hash", "model_answer_sha256 must be"),
        ("token_budget", "exceeds the full 8192-token"),
        ("receipt_replay", "reuses a receipt request ID"),
    ],
)
def test_rejects_leakage_identity_answer_budget_and_receipt_attacks(
    evidence: dict[str, Any], mutation: str, match: str
) -> None:
    rows = deepcopy(evidence["rows"])
    candidate = rows[0][CANDIDATE_ARM]
    if mutation == "gold_flag":
        candidate["selection"]["gold_fields_used"] = True
    elif mutation == "identity":
        candidate["reader"]["identity_sha256"] = _digest("drifted-reader")
    elif mutation == "answer_hash":
        candidate["judge"]["model_answer_sha256"] = _digest("other-answer")
    elif mutation == "token_budget":
        candidate["reader"]["prompt_tokens"] = PRIMARY_PROMPT_BUDGET + 1
    else:
        candidate["reader"]["request_id"] = rows[0][BASELINE_ARM]["reader"]["request_id"]
    _rebind(evidence, rows)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match=match):
        _compile(evidence)


@pytest.mark.parametrize(
    "bypass_field",
    [
        "ground_truth",
        "oracle_output",
        "target_response",
        "referenceanswer",
        "labels",
        "answer_session_ids",
    ],
)
def test_exact_profile_allowlist_rejects_gold_or_oracle_name_bypasses(
    evidence: dict[str, Any], bypass_field: str
) -> None:
    rows = deepcopy(evidence["rows"])
    rows[0][CANDIDATE_ARM]["selection"]["input_field_names"].append(bypass_field)
    _rebind(evidence, rows)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match="exactly equal.*allowlist"):
        _compile(evidence)


def test_exact_profile_allowlist_rejects_order_and_cross_profile_substitution(
    evidence: dict[str, Any],
) -> None:
    rows = deepcopy(evidence["rows"])
    rows[0][BASELINE_ARM]["selection"]["input_field_names"].reverse()
    _rebind(evidence, rows)
    with pytest.raises(LongMemEvalSelectionEvidenceError, match="exactly equal.*allowlist"):
        _compile(evidence)

    rows = deepcopy(evidence["rows"])
    selection = rows[0][CANDIDATE_ARM]["selection"]
    selection["input_profile"] = E1_C_INPUT_PROFILE
    selection["input_profile_sha256"] = selection_input_profile_digest(E1_C_INPUT_PROFILE)
    selection["input_field_names"] = list(SELECTION_INPUT_PROFILES[E1_C_INPUT_PROFILE])
    _rebind(evidence, rows)
    with pytest.raises(LongMemEvalSelectionEvidenceError, match="differs from the arm's cell"):
        _compile(evidence, output="cross-profile.json")


def test_cross_namespace_request_id_replay_is_rejected(evidence: dict[str, Any]) -> None:
    rows = deepcopy(evidence["rows"])
    rows[0][CANDIDATE_ARM]["reader"]["request_id"] = rows[0][BASELINE_ARM]["reader"][
        "provider_request_id"
    ]
    _rebind(evidence, rows)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match="reuses a receipt request ID"):
        _compile(evidence)


def test_receipt_envelope_digest_tampering_is_rejected(evidence: dict[str, Any]) -> None:
    rows = deepcopy(evidence["rows"])
    rows[0][CANDIDATE_ARM]["reader"]["receipt_envelope_sha256"] = _digest("forged-envelope")
    _rebind(evidence, rows)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match="complete reader receipt"):
        _compile(evidence)


def test_rejects_nonfinite_accounting_even_when_manifest_is_rebound(
    evidence: dict[str, Any],
) -> None:
    text = evidence["evidence"].read_text(encoding="utf-8")
    text = text.replace('"latency_ms": 2.0', '"latency_ms": NaN', 1)
    evidence["evidence"].write_text(text, encoding="utf-8")
    run = deepcopy(evidence["run"])
    run["evidence"] = _binding(evidence["evidence"], evidence["root"], count_field="rows", count=8)
    _write_json(evidence["run_path"], run)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match="non-finite JSON"):
        _compile(evidence)


@pytest.mark.parametrize("line_ending", ["\r\n", "\r"])
def test_rejects_crlf_and_bare_cr_jsonl_even_when_manifest_is_rebound(
    evidence: dict[str, Any], line_ending: str
) -> None:
    text = evidence["evidence"].read_text(encoding="utf-8")
    evidence["evidence"].write_bytes(text.replace("\n", line_ending, 1).encode("utf-8"))
    run = deepcopy(evidence["run"])
    run["evidence"] = _binding(evidence["evidence"], evidence["root"], count_field="rows", count=8)
    _write_json(evidence["run_path"], run)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match="byte-exact LF"):
        _compile(evidence)


def test_type_regression_gate_uses_raw_counts_and_two_point_boundary(
    evidence: dict[str, Any],
) -> None:
    rows = deepcopy(evidence["rows"])
    # Type 2 baseline 4/4, candidate 3/4: a raw one-question, 25 pp net regression.
    for index in range(4, 8):
        record = evidence["records"][index]
        rows[index][BASELINE_ARM] = _arm(
            index, BASELINE_ARM, record, evidence["identities"], correct=True
        )
        rows[index][CANDIDATE_ARM] = _arm(
            index, CANDIDATE_ARM, record, evidence["identities"], correct=index != 4
        )
    _rebind(evidence, rows)
    report = _compile(evidence, output="type-regression.json")

    gate = report["promotion"]["primary_track_gates"][
        "no_question_type_regression_over_two_percentage_points"
    ]
    assert gate["passed"] is False
    assert gate["comparison_uses_raw_correct_counts"] is True
    assert gate["violations"] == [
        {
            "question_type": "knowledge-update",
            "questions": 4,
            "baseline_correct": 4,
            "candidate_correct": 3,
            "net_regression_count": 1,
            "regression_percentage_points": 25.0,
        }
    ]


def test_any_gold_regression_fails_exact_zero_margin(evidence: dict[str, Any]) -> None:
    rows = deepcopy(evidence["rows"])
    for index, record in enumerate(evidence["records"]):
        rows[index][CANDIDATE_ARM] = _arm(
            index,
            CANDIDATE_ARM,
            record,
            evidence["identities"],
            correct=rows[index][CANDIDATE_ARM]["judge"]["label"],
            deliver_gold=False,
        )
    _rebind(evidence, rows)
    report = _compile(evidence, output="gold-regression.json")

    gate = report["promotion"]["primary_track_gates"]["any_gold_in_context_noninferior"]
    assert gate == {
        "passed": False,
        "noninferiority_margin": 0.0,
        "baseline_any_gold_questions": 8,
        "candidate_any_gold_questions": 0,
    }


def test_internal_abs_token_does_not_mark_an_answerable_question_abstention(
    evidence: dict[str, Any],
) -> None:
    records = deepcopy(evidence["records"])
    records[0]["question_id"] = "question_abs_internal-000"
    _replace_dataset(evidence, records)
    report = _compile(evidence, output="internal-abs.json")

    assert report["metrics"]["gold_context"]["questions"] == 8
    assert (
        report["promotion"]["primary_track_gates"]["all_required_metrics_reported"]["passed"]
        is True
    )


def test_baseline_weakly_dominates_only_with_one_strict_advantage(
    evidence: dict[str, Any],
) -> None:
    rows = deepcopy(evidence["rows"])
    for index, record in enumerate(evidence["records"]):
        rows[index][CANDIDATE_ARM] = _arm(
            index,
            CANDIDATE_ARM,
            record,
            evidence["identities"],
            correct=rows[index][BASELINE_ARM]["judge"]["label"],
        )
    _rebind(evidence, rows)
    report = _compile(evidence, output="dominated.json")

    gate = report["promotion"]["primary_track_gates"]["pareto_non_dominated"]
    assert gate["baseline_dominates_candidate"] is True
    assert gate["passed"] is False


def test_force_overwrite_is_type_checked_atomic_and_cleans_temporary_files(
    evidence: dict[str, Any],
) -> None:
    _compile(evidence, output="replaceable.json")
    replaced = _compile(evidence, output="replaceable.json", overwrite=True)
    stored = json.loads((evidence["root"] / "replaceable.json").read_text(encoding="utf-8"))

    assert stored["artifact_type"] == REPORT_ARTIFACT_TYPE
    assert stored == replaced
    assert list(evidence["root"].glob(".replaceable.json.*.tmp")) == []


def test_force_refuses_foreign_output_and_hardlinked_input(evidence: dict[str, Any]) -> None:
    _write_json(evidence["root"] / "foreign.json", {"artifact_type": "other-report"})
    with pytest.raises(LongMemEvalSelectionEvidenceError, match="can replace only"):
        _compile(evidence, output="foreign.json", overwrite=True)

    os.link(evidence["evidence"], evidence["root"] / "hardlink.json")
    with pytest.raises(LongMemEvalSelectionEvidenceError, match="share an inode"):
        _compile(evidence, output="hardlink.json", overwrite=True)


def _make_heldout(evidence: dict[str, Any]) -> str:
    root = evidence["root"]
    records = [
        {
            "question_id": f"heldout-{index}",
            "question_type": QUESTION_TYPES[index % 2],
            "opaque_input_sha256": _digest(f"heldout-input/{index}"),
        }
        for index in range(4)
    ]
    dataset_path = root / "heldout-dataset.json"
    _write_json(dataset_path, records)
    dataset_binding = _binding(dataset_path, root, count_field="questions", count=4)
    identities = evidence["identities"]
    preregistration = {
        "schema_version": HELDOUT_PREREGISTRATION_SCHEMA_VERSION,
        "artifact_type": HELDOUT_PREREGISTRATION_ARTIFACT_TYPE,
        "preregistration_id": "heldout-prereg-1",
        "registered_at_utc": "2026-08-01T00:00:00+00:00",
        "primary_dataset_sha256": evidence["dataset_sha256"],
        "heldout_dataset": dataset_binding,
        "protocol_version": PROTOCOL_VERSION,
        "baseline_cell_sha256": sha256_json(identities["baseline_cell"]),
        "candidate_cell_sha256": sha256_json(identities["candidate_cell"]),
        "prompt_serializer_sha256": sha256_json(identities["prompt_serializer"]),
        "reader_identity_sha256": sha256_json(identities["reader_identity"]),
        "judge_identity_sha256": sha256_json(identities["judge_identity"]),
        "evidence_schema": "paired-binary-labels-with-receipt-digests-v1",
    }
    prereg_path = root / "heldout-preregistration.json"
    _write_json(prereg_path, preregistration)
    rows = [
        {
            "schema_version": 1,
            "artifact_type": "swarmbrain-selection-heldout-paired-label",
            "protocol_version": PROTOCOL_VERSION,
            "case_index": index,
            "question_id": record["question_id"],
            "question_type": record["question_type"],
            "dataset_record_sha256": sha256_json(record),
            "baseline_cell_sha256": preregistration["baseline_cell_sha256"],
            "candidate_cell_sha256": preregistration["candidate_cell_sha256"],
            "prompt_serializer_sha256": preregistration["prompt_serializer_sha256"],
            "reader_identity_sha256": preregistration["reader_identity_sha256"],
            "judge_identity_sha256": preregistration["judge_identity_sha256"],
            "baseline_label": False,
            "candidate_label": True,
            "baseline_receipt_sha256": _digest(f"heldout-baseline-receipt/{index}"),
            "candidate_receipt_sha256": _digest(f"heldout-candidate-receipt/{index}"),
        }
        for index, record in enumerate(records)
    ]
    heldout_evidence_path = root / "heldout-evidence.jsonl"
    _write_jsonl(heldout_evidence_path, rows)
    confirmation = {
        "schema_version": HELDOUT_CONFIRMATION_SCHEMA_VERSION,
        "artifact_type": HELDOUT_CONFIRMATION_ARTIFACT_TYPE,
        "completed_at_utc": "2026-08-08T00:00:00+00:00",
        "preregistration": _file_binding(prereg_path, root),
        "dataset": dataset_binding,
        "evidence": _binding(heldout_evidence_path, root, count_field="rows", count=len(rows)),
        "receipt_authentication": RECEIPT_AUTHENTICATION,
    }
    confirmation_path = root / "heldout-confirmation.json"
    _write_json(confirmation_path, confirmation)
    return confirmation_path.relative_to(root).as_posix()


def test_structural_heldout_rows_never_unlock_noncanonical_or_serving_promotion(
    evidence: dict[str, Any],
) -> None:
    heldout = _make_heldout(evidence)
    report = _compile(evidence, output="promoted.json", heldout=heldout)

    confirmation = report["promotion"]["heldout_structural_evidence"]
    assert confirmation["structurally_complete_and_distinct"] is True
    assert confirmation["labels_recomputed_from_complete_paired_rows"] is True
    assert confirmation["directional_qa_benefit"] is True
    assert confirmation["qa_delta"] == 1.0
    assert confirmation["authentication"] == RECEIPT_AUTHENTICATION
    assert confirmation["evidence_authenticity_verified"] is False
    assert confirmation["independence_claim_authenticated"] is False
    assert report["promotion"]["structural_heldout_gate_passed"] is True
    assert report["promotion"]["structural_offline_policy_eligible"] is False
    assert report["promotion"]["serving_promotion_eligible"] is False
    assert "not semantic independence" in report["promotion"]["structural_eligibility_scope"]


def test_rejects_self_attested_or_tampered_heldout_confirmation(
    evidence: dict[str, Any],
) -> None:
    heldout = _make_heldout(evidence)
    path = evidence["root"] / heldout
    confirmation = json.loads(path.read_text(encoding="utf-8"))
    confirmation["directional_qa_benefit"] = True
    _write_json(path, confirmation)

    with pytest.raises(LongMemEvalSelectionEvidenceError, match="fields differ"):
        _compile(evidence, output="forged.json", heldout=heldout)


def test_rejects_heldout_evidence_tampering_after_confirmation(
    evidence: dict[str, Any],
) -> None:
    heldout = _make_heldout(evidence)
    with (evidence["root"] / "heldout-evidence.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(" ")

    with pytest.raises(LongMemEvalSelectionEvidenceError, match="bytes does not match|sha256"):
        _compile(evidence, output="tampered-heldout.json", heldout=heldout)
