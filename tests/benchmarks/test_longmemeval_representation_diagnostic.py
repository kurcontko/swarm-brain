from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from benchmarks.integrations.longmemeval_representation.contracts import (
    ARTIFACT_TYPE as REPRESENTATION_ARTIFACT_TYPE,
)
from benchmarks.integrations.longmemeval_representation.contracts import (
    MAX_HYDRATED_VALUES,
    canonical_json_bytes,
    sha256_json,
)
from benchmarks.integrations.longmemeval_representation.contracts import (
    PROTOCOL_VERSION as REPRESENTATION_PROTOCOL_VERSION,
)
from benchmarks.integrations.longmemeval_representation.contracts import (
    SCHEMA_VERSION as REPRESENTATION_SCHEMA_VERSION,
)
from benchmarks.integrations.longmemeval_representation.diagnostic import (
    CASE_ARTIFACT_TYPE,
    CASE_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    R0,
    R1,
    RepresentationDiagnosticError,
    compile_r0_r1_diagnostic,
    load_sealed_case_input,
    seal_case_input,
)


def _digest(label: str) -> str:
    return sha256_json({"label": label})


def _observation(family: str, question_id: str) -> dict[str, Any]:
    payload = {
        "family": family,
        "question_id": question_id,
        "source_artifact_sha256": _digest("source"),
        "projection_sha256": _digest("projection"),
        "index_sha256": _digest(f"index:{family}"),
        "query_sha256": _digest(f"query:{question_id}"),
        "scorer": {"identity_sha256": _digest(f"scorer:{family}")},
        "observation_artifact_sha256": _digest(f"observation:{family}"),
        "requested_depth": 20,
        "indexed_key_count": 3,
        "examined_key_count": 3,
        "ranked_keys": [],
        "complete": True,
    }
    return {**payload, "observation_sha256": sha256_json(payload)}


def _construction(cell: str, value_ids: list[str]) -> dict[str, Any]:
    r1 = cell == R1
    derived = [
        {"value_id": value_id, "key_counts": {"raw": 1, "merged-sfk": int(r1)}}
        for value_id in value_ids
    ]
    construction = {
        "input_tokens": 90 if r1 else 0,
        "output_tokens": 15 if r1 else 0,
        "latency_microseconds": 300 if r1 else 0,
        "cost_microusd": 20 if r1 else 0,
        "retry_count": 1 if r1 else 0,
        "cache_hits": 0,
    }
    return {
        "canonical_value_count": len(value_ids),
        "canonical_value_utf8_bytes": 60,
        "active_indexed_key_count": len(value_ids) * (2 if r1 else 1),
        "active_indexed_key_utf8_bytes": 75 if r1 else 60,
        "derived_key_count": len(value_ids) if r1 else 0,
        "derived_key_utf8_bytes": 15 if r1 else 0,
        "derived_objects_per_source": derived,
        "derived_objects_per_source_sha256": sha256_json(derived),
        "construction_receipt_count": len(value_ids) if r1 else 0,
        "construction_receipts_sha256": _digest(f"receipts:{cell}"),
        "extractor_identities": ([{"identity_sha256": _digest("extractor")}] if r1 else []),
        "construction_artifact_sha256s": ([_digest("construction")] if r1 else []),
        "construction_accounting": construction,
        "duplicate_key_text": {"count": 0, "denominator": len(value_ids) if r1 else 0},
        "orphan_keys": {"count": 0, "denominator": len(value_ids) * (2 if r1 else 1)},
        "update_rate": {
            "updates": 0,
            "construction_receipts": len(value_ids) if r1 else 0,
            "classification": "static-representation-control-not-consolidation",
        },
        "index_token_count": None,
        "index_token_count_status": "not-inferred",
    }


def _representation(
    cell: str,
    *,
    question_id: str,
    value_ids: list[str],
) -> dict[str, Any]:
    observations = [
        _observation(family, question_id)
        for family in (["raw"] if cell == R0 else ["raw", "merged-sfk"])
    ]
    value_scores = [
        {
            "rank": rank,
            "value": {
                "value_id": value_id,
                "raw_value": {
                    "utf8_bytes": 10 * rank,
                    "sha256": _digest(f"raw:{value_id}"),
                },
            },
            "score": 1.0 / rank,
        }
        for rank, value_id in enumerate(value_ids, start=1)
    ]
    trace = {
        "artifact_type": REPRESENTATION_ARTIFACT_TYPE,
        "schema_version": REPRESENTATION_SCHEMA_VERSION,
        "protocol_version": REPRESENTATION_PROTOCOL_VERSION,
        "cell": cell,
        "classification": "benchmark-only-source-preserving-representation-control",
        "production_configuration": False,
        "paper_reproduction": False,
        "sb_hypothesis": "SB-HMR-v1",
        "frozen_protocol": {"key_families": ["raw"] if cell == R0 else ["raw", "merged-sfk"]},
        "promotion": {
            "cell_intrinsically_ineligible": False,
            "reason": "quality eligibility requires downstream held-out paired evidence",
        },
        "corpus": {
            "question_id": question_id,
            "source_artifact_sha256": _digest("source"),
            "projection_sha256": _digest("projection"),
            "index_sha256": _digest(f"index:{cell}"),
            "navigation_index_sha256": _digest(f"navigation:{cell}"),
            "navigation_index_classification": "source-only-navigation-index",
            "canonical_value_count": len(value_ids),
            "canonical_value_order_sha256": _digest("canonical-order"),
            "complete_question_local_corpus_precedes_retrieval": True,
        },
        "observations": observations,
        "observations_sha256": sha256_json(observations),
        "ranking": {"method": "fixture"},
        "graph": None,
        "value_scores": value_scores,
        "value_scores_sha256": sha256_json(value_scores),
        "key_level_returned_count": len(value_ids),
        "hydrated_value_pre_cap_count": len(value_ids),
        "hydrated_value_cap": MAX_HYDRATED_VALUES,
        "hydrated_value_ids": value_ids,
        "hydrated_raw_value_hashes": [
            value_score["value"]["raw_value"]["sha256"] for value_score in value_scores
        ],
        "hydrated_value_count": len(value_ids),
        "hydration": {
            "reader_evidence": "canonical-raw-value",
            "derived_keys_delivered_to_reader": False,
            "source_values_byte_identical": True,
        },
        "construction_and_index_accounting": _construction(cell, value_ids),
        "construction_input_contract": {"gold_question_type_answer_or_judge_fields_allowed": False},
        "claims": {
            "question_query_consumed_by_ranking": True,
            "quality_improvement": False,
            "serving_change": False,
        },
    }
    return {**trace, "trace_sha256": sha256_json(trace)}


def _query_accounting(cell: str, *, latency: int, cost: int) -> dict[str, Any]:
    stages = [
        {
            "name": "family-scoring",
            "calls": 2 if cell == R1 else 1,
            "input_tokens": 40 if cell == R1 else 20,
            "output_tokens": 4 if cell == R1 else 2,
            "latency_microseconds": latency,
            "cost_microusd": cost,
            "retry_count": 0,
            "cache_hits": 0,
        }
    ]
    return {
        "complete": True,
        "source": "externally-attested-unverified",
        "stages": stages,
        "stages_sha256": sha256_json(stages),
    }


def _arm(
    cell: str,
    *,
    question_id: str,
    value_ids: list[str],
    sessions: list[str],
    prompt_ids: list[str],
    prompt_tokens: int,
    query_latency: int,
    query_cost: int,
    qa: bool | None = None,
) -> dict[str, Any]:
    return {
        "cell": cell,
        "representation": _representation(cell, question_id=question_id, value_ids=value_ids),
        "context": {
            "candidate_session_sha256s": sessions,
            "prompt_value_ids": prompt_ids,
            "prompt_tokens": prompt_tokens,
            "prompt_sha256": _digest(f"prompt:{cell}:{question_id}"),
            "tokenizer_artifact_sha256": _digest("tokenizer"),
            "tokenizer_receipt_sha256": _digest(f"tokenizer-receipt:{cell}:{question_id}"),
        },
        "query_accounting": _query_accounting(
            cell,
            latency=query_latency,
            cost=query_cost,
        ),
        "qa": (
            None
            if qa is None
            else {
                "correct": qa,
                "reader_receipt_sha256": _digest(f"reader:{cell}:{question_id}"),
                "judge_receipt_sha256": _digest(f"judge:{cell}:{question_id}"),
            }
        ),
    }


def _case_payload(
    *,
    r1_sessions: list[str] | None = None,
    r1_prompt_ids: list[str] | None = None,
    r1_prompt_tokens: int = 450,
    r1_query_latency: int = 600,
    r1_query_cost: int = 90,
    qa: tuple[bool, bool] | None = None,
) -> dict[str, Any]:
    question_id = "q-e6-diagnostic"
    value_ids = ["value-0", "value-1", "value-2"]
    gold_a = _digest("gold-a")
    gold_b = _digest("gold-b")
    other = _digest("other")
    r0_sessions = [gold_a, other, gold_b]
    candidate_sessions = r1_sessions or [gold_a, gold_b, other]
    qa_r0, qa_r1 = (None, None) if qa is None else qa
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "artifact_type": CASE_ARTIFACT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "case_index": 0,
        "question_id": question_id,
        "question_type": "multi-session",
        "gold_session_sha256s": sorted([gold_a, gold_b]),
        "arms": {
            R0: _arm(
                R0,
                question_id=question_id,
                value_ids=value_ids,
                sessions=r0_sessions,
                prompt_ids=["value-0", "value-1"],
                prompt_tokens=500,
                query_latency=500,
                query_cost=100,
                qa=qa_r0,
            ),
            R1: _arm(
                R1,
                question_id=question_id,
                value_ids=value_ids,
                sessions=candidate_sessions,
                prompt_ids=r1_prompt_ids or ["value-0", "value-1"],
                prompt_tokens=r1_prompt_tokens,
                query_latency=r1_query_latency,
                query_cost=r1_query_cost,
                qa=qa_r1,
            ),
        },
    }


def _write_case(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    sealed = seal_case_input(payload)
    path.write_bytes(canonical_json_bytes(sealed) + b"\n")
    return sealed


def test_compiler_reopens_cases_and_reconciles_full_accounting(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    sealed = _write_case(case_path, _case_payload())

    assert load_sealed_case_input(case_path) == sealed
    report = compile_r0_r1_diagnostic([case_path])

    assert report["source"]["all_case_files_reopened"] is True
    assert report["context_quality"]["R1_minus_R0"][
        "prompt_answer_session_recall"
    ] == pytest.approx(0.5)
    r1 = report["efficiency"]["arms"][R1]
    assert r1["construction_accounting_totals"]["cost_microusd"] == 20
    assert r1["query_accounting_totals"]["cost_microusd"] == 90
    assert r1["construction_plus_query_accounting_totals"]["cost_microusd"] == 110
    assert report["early_stop"]["triggered"] is False
    assert report["decision"]["eligible_for_serving_promotion"] is False
    assert report["claims"]["official_longmemeval_score"] is False


def test_gold_regression_triggers_preregistered_early_stop(tmp_path: Path) -> None:
    gold_a = _digest("gold-a")
    other = _digest("other")
    path = tmp_path / "regression.json"
    _write_case(
        path,
        _case_payload(
            r1_sessions=[other, gold_a, other],
            r1_prompt_ids=["value-0"],
        ),
    )

    report = compile_r0_r1_diagnostic([path])

    assert report["early_stop"]["gold_noninferiority"]["passed"] is False
    assert "R1-fails-zero-margin-gold-context-noninferiority" in report["early_stop"]["reasons"]
    assert report["early_stop"]["continue_to_optional_qa"] is False


def test_equal_context_and_higher_r1_cost_is_r0_pareto_dominance(tmp_path: Path) -> None:
    gold_a = _digest("gold-a")
    gold_b = _digest("gold-b")
    other = _digest("other")
    path = tmp_path / "dominated.json"
    _write_case(
        path,
        _case_payload(
            r1_sessions=[gold_a, other, gold_b],
            r1_prompt_ids=["value-0", "value-1"],
            r1_prompt_tokens=600,
            r1_query_latency=900,
            r1_query_cost=150,
        ),
    )

    report = compile_r0_r1_diagnostic([path])

    assert report["early_stop"]["gold_noninferiority"]["passed"] is True
    assert report["early_stop"]["R0_pareto_dominates_R1"]["value"] is True
    assert report["early_stop"]["triggered"] is True


def test_optional_qa_is_paired_descriptive_and_never_promotional(tmp_path: Path) -> None:
    path = tmp_path / "qa.json"
    _write_case(path, _case_payload(qa=(False, True)))

    report = compile_r0_r1_diagnostic([path])

    assert report["qa"]["available"] is True
    assert report["qa"]["paired_accuracy_delta_R1_minus_R0"] == 1.0
    assert report["qa"]["promotion_use"] == "forbidden-development-diagnostic-only"
    assert report["decision"]["qa_can_override_early_stop"] is False
    assert report["decision"]["eligible_for_composition"] is False


def test_unpaired_qa_and_tampered_seals_fail_closed(tmp_path: Path) -> None:
    unpaired = _case_payload()
    unpaired["arms"][R0]["qa"] = {
        "correct": True,
        "reader_receipt_sha256": _digest("reader"),
        "judge_receipt_sha256": _digest("judge"),
    }
    with pytest.raises(RepresentationDiagnosticError, match="both arms or neither"):
        seal_case_input(unpaired)

    path = tmp_path / "tampered.json"
    sealed = _write_case(path, _case_payload())
    tampered = deepcopy(sealed)
    tampered["arms"][R1]["context"]["prompt_tokens"] += 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RepresentationDiagnosticError, match="seal"):
        compile_r0_r1_diagnostic([path])


def test_trace_tampering_is_rejected_even_when_outer_case_is_resealed() -> None:
    payload = _case_payload()
    trace = payload["arms"][R1]["representation"]
    trace["hydrated_value_ids"] = list(reversed(trace["hydrated_value_ids"]))
    case_without_seal = deepcopy(payload)

    with pytest.raises(RepresentationDiagnosticError, match="trace seal"):
        seal_case_input(case_without_seal)
