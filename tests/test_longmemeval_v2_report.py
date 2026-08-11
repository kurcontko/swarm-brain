from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from build_longmemeval_v2_report import (
    EXPECTED_JUDGE_MODEL,
    EXPECTED_QUESTIONS,
    EXPECTED_READER_MODEL,
    PINNED_REPOSITORY_COMMIT,
    SIDECAR_SCHEMA_VERSION,
    SOTA_EMBEDDING_MODEL,
    SOTA_EMBEDDING_PROVIDER,
    SOTA_EMBEDDING_QUERY_INSTRUCTION_SHA256,
    TRACE_DIGEST_METADATA_KEY,
    LongMemEvalV2EvidenceError,
    _aggregate_expected,
    _canonical_sha256,
    _expected_invocation_id,
    _lafs_summary,
    _LafsPoint,
    _metric_overview,
    _package_hashes,
    _percentile_index,
    _protocol_sha256,
    _tree_sha256,
    build_report,
    load_tier_evidence,
    main,
)
from evaluate_sota_readiness import evaluate_manifest

QUESTION_TYPES = (
    "static-environment",
    "static-environment-abs",
    "dynamic-environment",
    "dynamic-environment-abs",
    "procedure",
    "procedure-abs",
    "errors-gotchas",
)
CATEGORY_BY_TYPE = {
    "static-environment": "static",
    "static-environment-abs": "static-abs",
    "dynamic-environment": "dynamic",
    "dynamic-environment-abs": "dynamic-abs",
    "procedure": "procedure",
    "procedure-abs": "procedure-abs",
    "errors-gotchas": "gotchas",
}
DATASET_REVISION = "fixture-caller-pinned-revision"
DATASET_MANIFEST_SHA256 = "1" * 64
EMBEDDING_REVISION = "fixture-public-embedding-revision"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, payload: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in payload),
        encoding="utf-8",
    )


def _run_args(domain: str) -> dict[str, object]:
    return {
        "domain": domain,
        "questions_path": f"/runs/{domain}/runtime_inputs/questions.json",
        "haystack_path": f"/runs/{domain}/runtime_inputs/haystack.json",
        "trajectories_path": "/data/trajectories.jsonl",
        "memory_config_path": f"/runs/{domain}/runtime_inputs/memory_config.json",
        "output_dir": f"/runs/{domain}",
        "save_memory": False,
        "skip_evaluation": False,
        "load_memory_dir": None,
        "model": EXPECTED_READER_MODEL,
        "base_url": "http://reader.example/v1",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_file": None,
        "max_completion_tokens": 20_000,
        "memory_context_max_tokens": 200_000,
        "prompt_build_max_workers": 1,
        "shuffle_questions_seed": 17,
        "reader_max_concurrent_requests": 16,
        "timeout_seconds": 43_200.0,
        "reasoning_effort": None,
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": None,
        "top_k": 20,
        "repetition_penalty": None,
        "reader_enable_thinking": True,
        "evaluator_model": EXPECTED_JUDGE_MODEL,
        "evaluator_base_url": "https://api.openai.com/v1",
        "evaluator_api_key_env": "OPENAI_API_KEY",
        "evaluator_api_key_file": None,
        "evaluator_reasoning_effort": "medium",
        "evaluator_max_completion_tokens": 4096,
        "evaluator_timeout_seconds": 43_200.0,
        "started_at_utc": f"2026-08-09T00:00:0{0 if domain == 'web' else 1}+00:00",
    }


def _question(global_index: int, domain: str) -> dict[str, object]:
    question_type = QUESTION_TYPES[global_index % len(QUESTION_TYPES)]
    return {
        "id": f"q{global_index:03d}",
        "domain": domain,
        "question_type": question_type,
        "question": f"What happened in environment item {global_index}?",
        "answer": f"answer-{global_index}",
        "eval_function": "exact_match",
    }


def _record(question: dict[str, object], global_index: int) -> dict[str, object]:
    question_type = str(question["question_type"])
    correct = global_index % 5 != 0
    query_duration = 0.010 + (global_index % 7) * 0.001
    return {
        "question_id": question["id"],
        "question_type": question_type,
        "category": CATEGORY_BY_TYPE[question_type],
        "is_abstention_problem": question_type.endswith("-abs"),
        "eval_function": question["eval_function"],
        "answer_gold": question["answer"],
        "response_raw": f"reader-response-{global_index}",
        "is_unknown": False,
        "score": 1.0 if correct else 0.0,
        "score_bool": correct,
        "usage": {
            "prompt_tokens": 100 + global_index % 11,
            "completion_tokens": 10 + global_index % 3,
            "total_tokens": 110 + global_index % 11 + global_index % 3,
        },
        "memory_query_duration_seconds": query_duration,
        "memory_post_query_duration_seconds": 0.001,
        "memory_context_original_token_count": 90,
        "memory_context_token_count": 80,
        "memory_context_was_truncated": False,
    }


def _domain_indices(domain: str) -> range:
    return range(0, 226) if domain == "web" else range(226, EXPECTED_QUESTIONS)


def _opaque_memory_id(*parts: str) -> str:
    return "mem_" + hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def _operation_trace(
    record: dict[str, object], *, tier: str, point_name: str
) -> list[dict[str, object]]:
    question_id = str(record["question_id"])
    seed_id = _opaque_memory_id(tier, point_name, question_id, "seed")
    return [
        {
            "sequence": 0,
            "invocation_id": _expected_invocation_id(
                tier=tier,
                point_name=point_name,
                question_id=question_id,
                sequence=0,
                operation="recall_memory",
            ),
            "operation": "recall_memory",
            "success": True,
            "depth": 0,
            "seed_memory_ids": [],
            "result_memory_ids": [seed_id],
            "delivered_tokens": 0,
            "latency_ms": 2.0,
        },
        {
            "sequence": 1,
            "invocation_id": _expected_invocation_id(
                tier=tier,
                point_name=point_name,
                question_id=question_id,
                sequence=1,
                operation="read_expand_memory",
            ),
            "operation": "read_expand_memory",
            "success": True,
            "depth": 1,
            "seed_memory_ids": [seed_id],
            "result_memory_ids": [
                seed_id,
                _opaque_memory_id(tier, point_name, question_id, "expanded"),
            ],
            "delivered_tokens": record["memory_context_token_count"],
            "latency_ms": 3.0,
        },
    ]


def _embedding_evidence(*, inserted_memories: int = 2) -> dict[str, object]:
    return {
        "retrieval_mode": "openai_hybrid",
        "sota_capable": True,
        "provider": SOTA_EMBEDDING_PROVIDER,
        "model": SOTA_EMBEDDING_MODEL,
        "model_revision": EMBEDDING_REVISION,
        "dimensions": 4_096,
        "response_model_requirement": SOTA_EMBEDDING_MODEL,
        "query_instruction_sha256": SOTA_EMBEDDING_QUERY_INSTRUCTION_SHA256,
        "inserted_memories": inserted_memories,
        "embedding_work_completed": inserted_memories,
        "call_accounting": {
            "source": "provider-observed",
            "document_inputs": inserted_memories,
            "document_batch_calls": inserted_memories,
            "document_successful_http_calls": inserted_memories,
            "document_http_attempts": inserted_memories,
            "query_calls": 1,
            "query_successful_http_calls": 1,
            "query_http_attempts": 1,
        },
        "exact_response_model_verified": True,
        "deterministic_fallback_used": False,
    }


def _write_run(
    point_dir: Path, domain: str, *, tier: str, point_name: str
) -> tuple[list[dict[str, object]], dict[str, object]]:
    run_dir = point_dir / domain
    questions = [_question(index, domain) for index in _domain_indices(domain)]
    records = [
        _record(question, index)
        for index, question in zip(_domain_indices(domain), questions, strict=True)
    ]
    for record in records:
        operations = _operation_trace(record, tier=tier, point_name=point_name)
        embedding = _embedding_evidence()
        record["memory_post_query_metadata"] = {
            TRACE_DIGEST_METADATA_KEY: _canonical_sha256(
                {"operations": operations, "embedding": embedding}
            )
        }
    run_args = _run_args(domain)
    memory_config: dict[str, object] = {
        "memory_type": "swarmbrain",
        "memory_params": {
            "activation": "event-triggered",
            "exploration": "search-read-expand",
            "limit": 8,
            "dataset_revision": DATASET_REVISION,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        },
    }
    _write_json(run_dir / "run_args.json", run_args)
    _write_json(run_dir / "runtime_inputs/questions.json", questions)
    _write_json(
        run_dir / "runtime_inputs/haystack.json",
        {str(question["id"]): [f"trajectory-{question['id']}"] for question in questions},
    )
    _write_json(run_dir / "runtime_inputs/memory_config.json", memory_config)
    _write_jsonl(run_dir / "per_question.jsonl", records)
    aggregated = _aggregate_expected(records)
    aggregated["completed_at_utc"] = "2026-08-09T00:10:00+00:00"
    aggregated["shared_haystack"] = False
    _write_json(run_dir / "aggregated_metrics.json", aggregated)
    return records, run_args


def _query_rows(
    records_by_domain: dict[str, list[dict[str, object]]],
    *,
    tier: str,
    point_name: str,
) -> list[dict[str, object]]:
    rows = [
        {
            "question_id": record["question_id"],
            "domain": domain,
            "query_tokens": record["memory_context_token_count"],
            "query_latency_ms": float(record["memory_query_duration_seconds"]) * 1000.0,
            "query_failed": False,
            "unanswered": False,
            "operations": _operation_trace(record, tier=tier, point_name=point_name),
            "embedding": _embedding_evidence(),
        }
        for domain, records in records_by_domain.items()
        for record in records
    ]
    return list(reversed(rows))


def _query_summary(rows: list[dict[str, object]]) -> dict[str, int | float]:
    tokens = [int(row["query_tokens"]) for row in rows]
    latencies = [float(row["query_latency_ms"]) for row in rows]
    return {
        "questions": len(rows),
        "query_token_observations": len(rows),
        "query_tokens_total": sum(tokens),
        "query_tokens_mean": sum(tokens) / len(tokens),
        "query_latency_observations": len(rows),
        "query_latency_total_ms": sum(latencies),
        "query_latency_mean_ms": sum(latencies) / len(latencies),
        "query_latency_p95_ms": _percentile_index(latencies, 0.95),
        "query_failures": 0,
        "unanswered_questions": 0,
    }


def _build_tier(tmp_path: Path, tier: str) -> tuple[Path, Path]:
    submission_name = f"fixture-{tier}"
    package = tmp_path / submission_name
    point_name = "balanced"
    point_dir = package / "operating_points" / point_name
    records_by_domain: dict[str, list[dict[str, object]]] = {}
    run_args_by_domain: dict[str, dict[str, object]] = {}
    for domain in ("web", "enterprise"):
        records, run_args = _write_run(point_dir, domain, tier=tier, point_name=point_name)
        records_by_domain[domain] = records
        run_args_by_domain[domain] = run_args
    combined_records = records_by_domain["web"] + records_by_domain["enterprise"]
    metric = _metric_overview(combined_records)
    _write_json(point_dir / "metric_overview.json", metric)
    _write_json(
        point_dir / "operating_point_metadata.json",
        {
            "submission_name": submission_name,
            "operating_point_name": point_name,
            "method": "swarmbrain",
            "tier": tier,
            "generated_at_utc": "2026-08-09T01:00:00+00:00",
            "runs": {
                domain: {
                    "source_run_dir": f"/runs/{domain}",
                    "domain": domain,
                    "question_count": len(records_by_domain[domain]),
                    "question_type_counts": {},
                    "model": EXPECTED_READER_MODEL,
                    "evaluator_model": EXPECTED_JUDGE_MODEL,
                }
                for domain in ("web", "enterprise")
            },
            "included_run_files": [
                "aggregated_metrics.json",
                "per_question.jsonl",
                "run_args.json",
                "runtime_inputs/",
            ],
        },
    )
    point = _LafsPoint(
        point_name,
        metric["overall_full_set"] * 100.0,
        metric["memory_query_avg_seconds"],
    )
    _write_json(package / "SYSTEM_DESCRIPTION.md", {"fixture": "system description"})
    (package / "code_file.py").write_text("# immutable fixture code\n", encoding="utf-8")
    _write_json(
        package / "submission_overview.json",
        {
            "submission_name": submission_name,
            "method": "swarmbrain",
            "tier": tier,
            "generated_at_utc": "2026-08-09T01:00:01+00:00",
            "archive_name": f"{submission_name}.tar.gz",
            "system_description_file": "SYSTEM_DESCRIPTION.md",
            "code_file": "code_file.py",
            "lafs": _lafs_summary(tier, [point]),
            "operating_points": [
                {
                    "name": point_name,
                    "metric_overview_file": (f"operating_points/{point_name}/metric_overview.json"),
                    "overall_full_set": metric["overall_full_set"],
                    "memory_query_avg_seconds": metric["memory_query_avg_seconds"],
                    "lafs_accuracy_percentage_points": metric["overall_full_set"] * 100.0,
                    "lafs_latency_seconds": metric["memory_query_avg_seconds"],
                }
            ],
        },
    )
    memory_config = json.loads(
        (point_dir / "web/runtime_inputs/memory_config.json").read_text(encoding="utf-8")
    )
    protocol_sha256 = _protocol_sha256(run_args_by_domain["web"], memory_config, "swarmbrain")
    query_rows = _query_rows(records_by_domain, tier=tier, point_name=point_name)
    package_files = _package_hashes(package)
    sidecar = tmp_path / f"{tier}-swarm-evidence.json"
    _write_json(
        sidecar,
        {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "benchmark_repository_commit": PINNED_REPOSITORY_COMMIT,
            "tier": tier,
            "method": "swarmbrain",
            "reader_model": EXPECTED_READER_MODEL,
            "judge_model": EXPECTED_JUDGE_MODEL,
            "dataset_revision": DATASET_REVISION,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "dataset_identity_source": "caller-pinned-run-ledger",
            "iterative_search_read_expand": True,
            "semantic_embedding_proof": True,
            "package": {
                "tree_sha256": _tree_sha256(package_files),
                "files_sha256": package_files,
            },
            "operating_points": [
                {
                    "name": point_name,
                    "protocol_sha256": protocol_sha256,
                    "summary": _query_summary(query_rows),
                    "queries": query_rows,
                }
            ],
        },
    )
    return package, sidecar


def _refresh_package_hashes(package: Path, sidecar: Path) -> None:
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    package_files = _package_hashes(package)
    payload["package"] = {
        "tree_sha256": _tree_sha256(package_files),
        "files_sha256": package_files,
    }
    _write_json(sidecar, payload)


def test_compiler_builds_manifest_schema_and_is_path_independent(tmp_path: Path) -> None:
    small_package, small_sidecar = _build_tier(tmp_path / "first", "small")
    medium_package, medium_sidecar = _build_tier(tmp_path / "first", "medium")

    small = load_tier_evidence(small_package, small_sidecar, expected_tier="small")
    medium = load_tier_evidence(medium_package, medium_sidecar, expected_tier="medium")
    report = build_report(small, medium)

    assert report["benchmark"]["repository_commit"] == PINNED_REPOSITORY_COMMIT
    assert report["dataset"]["questions"] == EXPECTED_QUESTIONS
    assert report["dataset"]["revision"] == DATASET_REVISION
    assert report["dataset"]["manifest_sha256"] == DATASET_MANIFEST_SHA256
    assert report["evaluation"] == {
        "method": "swarmbrain",
        "fixed_reader": True,
        "reader_model": EXPECTED_READER_MODEL,
        "judge_model": EXPECTED_JUDGE_MODEL,
        "iterative_search_read_expand": True,
        "iterative_search_read_expand_declaration_role": "secondary-metadata",
        "retrieval_control_flow": "fixed_two_stage_recall_then_expand",
        "adaptive_multi_round_retrieval": False,
        "primary_agentic_evidence": "per-query-content-free-operation-traces",
        "semantic_embedding_required": True,
        "primary_embedding_evidence": ("per-query-content-free-provider-and-http-accounting"),
        "operation_trace_binding": (
            f"official-per_question.memory_post_query_metadata.{TRACE_DIGEST_METADATA_KEY}"
        ),
        "protocol_sha256": report["evaluation"]["protocol_sha256"],
    }
    assert report["tiers"]["small"]["accuracy"] > 0.749
    assert report["tiers"]["small"]["lafs_gain"] > 0
    assert report["tiers"]["medium"]["accuracy"] > 0.701
    assert report["tiers"]["medium"]["lafs_gain"] > 0
    assert report["failures"] == {"query_failures": 0, "unanswered_questions": 0}
    semantic = report["semantic_embedding_proof"]
    assert semantic["provider"] == SOTA_EMBEDDING_PROVIDER
    assert semantic["model"] == SOTA_EMBEDDING_MODEL
    assert semantic["dimensions"] == 4_096
    assert semantic["inserted_memory_embedding_coverage"] == 1.0
    assert semantic["query_embedding_call_coverage"] == 1.0
    assert semantic["deterministic_fallback_used"] is False
    assert report["latency"]["observations"] == EXPECTED_QUESTIONS * 2
    assert report["tokens"]["observations"] == EXPECTED_QUESTIONS * 2
    for tier in ("small", "medium"):
        proof = report["tiers"][tier]["agentic_proof"]
        assert proof["queries"] == EXPECTED_QUESTIONS
        assert proof["search_then_read_expand_queries"] == EXPECTED_QUESTIONS
        assert proof["all_queries_proved"] is True
        assert proof["all_traces_package_bound"] is True
        assert proof["package_bound_trace_queries"] == EXPECTED_QUESTIONS
        assert proof["token_reconciliation_exact"] is True
        assert proof["operation_latency_bounded"] is True
        assert proof["invocation_ids_unique"] is True
        assert proof["max_depth"] == 1
    assert report["agentic_proof"]["queries"] == EXPECTED_QUESTIONS * 2
    assert len(report["evidence"]["small"]["package_files_sha256"]) >= 16

    repository_root = Path(__file__).resolve().parents[1]
    frozen_manifest = json.loads(
        (repository_root / "benchmarks/sota/manifest.json").read_text(encoding="utf-8")
    )
    gate = next(item for item in frozen_manifest["gates"] if item["id"] == "longmemeval_v2_agentic")
    gate.pop("compiler_replay", None)
    gate["artifact"] = "compiled-report.json"
    _write_json(tmp_path / "compiled-report.json", report)
    _write_json(
        tmp_path / "proof-manifest.json",
        {
            "schema_version": 2,
            "claim": "fixture LongMemEval-V2 proof",
            "frozen_at": frozen_manifest["frozen_at"],
            "claim_scope": {
                "claim_sha256": hashlib.sha256(b"fixture LongMemEval-V2 proof").hexdigest(),
                "coverage_policy": "every-dimension-covered-by-required-gate",
                "dimensions": gate["claim_dimensions"],
            },
            "gates": [gate],
        },
    )
    readiness = evaluate_manifest(tmp_path / "proof-manifest.json", repo_root=tmp_path)
    assert readiness.ready

    copied_root = tmp_path / "copied"
    copied_root.mkdir()
    copied_small = shutil.copytree(small_package, copied_root / small_package.name)
    copied_medium = shutil.copytree(medium_package, copied_root / medium_package.name)
    copied_small_sidecar = copied_root / small_sidecar.name
    copied_medium_sidecar = copied_root / medium_sidecar.name
    shutil.copy2(small_sidecar, copied_small_sidecar)
    shutil.copy2(medium_sidecar, copied_medium_sidecar)
    copied_report = build_report(
        load_tier_evidence(copied_small, copied_small_sidecar, expected_tier="small"),
        load_tier_evidence(copied_medium, copied_medium_sidecar, expected_tier="medium"),
    )
    assert json.dumps(copied_report, sort_keys=True) == json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)


def test_compiler_rejects_nonexact_model_and_duplicate_coverage(tmp_path: Path) -> None:
    package, sidecar = _build_tier(tmp_path / "model", "small")
    run_args_path = package / "operating_points/balanced/web/run_args.json"
    run_args = json.loads(run_args_path.read_text(encoding="utf-8"))
    run_args["model"] = f"{EXPECTED_READER_MODEL}-quantized"
    _write_json(run_args_path, run_args)
    _refresh_package_hashes(package, sidecar)

    with pytest.raises(LongMemEvalV2EvidenceError, match="exact fixed reader"):
        load_tier_evidence(package, sidecar, expected_tier="small")

    duplicate_package, duplicate_sidecar = _build_tier(tmp_path / "duplicate", "small")
    questions_path = (
        duplicate_package / "operating_points/balanced/web/runtime_inputs/questions.json"
    )
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    questions[1]["id"] = questions[0]["id"]
    _write_json(questions_path, questions)
    _refresh_package_hashes(duplicate_package, duplicate_sidecar)

    with pytest.raises(LongMemEvalV2EvidenceError, match="duplicate id"):
        load_tier_evidence(duplicate_package, duplicate_sidecar, expected_tier="small")


def test_compiler_recomputes_metrics_and_positive_official_lafs(tmp_path: Path) -> None:
    package, sidecar = _build_tier(tmp_path / "metric", "small")
    metrics_path = package / "operating_points/balanced/web/aggregated_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["overall"]["overall_full_set"] = 1.0
    _write_json(metrics_path, metrics)
    _refresh_package_hashes(package, sidecar)

    with pytest.raises(LongMemEvalV2EvidenceError, match="inconsistent"):
        load_tier_evidence(package, sidecar, expected_tier="small")

    lafs_package, lafs_sidecar = _build_tier(tmp_path / "lafs", "small")
    overview_path = lafs_package / "submission_overview.json"
    overview = json.loads(overview_path.read_text(encoding="utf-8"))
    overview["lafs"]["lafs_gain"] = 0.0
    _write_json(overview_path, overview)
    _refresh_package_hashes(lafs_package, lafs_sidecar)

    with pytest.raises(LongMemEvalV2EvidenceError, match="lafs_gain must be positive"):
        load_tier_evidence(lafs_package, lafs_sidecar, expected_tier="small")


def test_compiler_rejects_false_iterative_claim_and_bad_query_accounting(
    tmp_path: Path,
) -> None:
    package, sidecar = _build_tier(tmp_path / "iterative", "small")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["iterative_search_read_expand"] = False
    _write_json(sidecar, payload)

    with pytest.raises(LongMemEvalV2EvidenceError, match="must be true"):
        load_tier_evidence(package, sidecar, expected_tier="small")

    accounting_package, accounting_sidecar = _build_tier(tmp_path / "accounting", "small")
    accounting = json.loads(accounting_sidecar.read_text(encoding="utf-8"))
    accounting["operating_points"][0]["queries"][0]["query_latency_ms"] += 1
    _write_json(accounting_sidecar, accounting)

    with pytest.raises(LongMemEvalV2EvidenceError, match="query_latency_ms is inconsistent"):
        load_tier_evidence(accounting_package, accounting_sidecar, expected_tier="small")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("model", "model must equal pinned value"),
        ("document_coverage", "document provider calls do not reconcile"),
        ("query_coverage", "one successful query embedding HTTP call"),
        ("fallback", "deterministic_fallback_used must equal pinned value"),
        ("revision", "model_revision must be a non-empty string"),
    ],
)
def test_compiler_rejects_unproved_semantic_embedding_evidence(
    tmp_path: Path, case: str, message: str
) -> None:
    package, sidecar = _build_tier(tmp_path / f"embedding-{case}", "small")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    embedding = payload["operating_points"][0]["queries"][0]["embedding"]
    if case == "model":
        embedding["model"] = "Qwen/Qwen3-Embedding-0.6B"
    elif case == "document_coverage":
        embedding["call_accounting"]["document_inputs"] -= 1
    elif case == "query_coverage":
        embedding["call_accounting"]["query_successful_http_calls"] = 0
    elif case == "fallback":
        embedding["deterministic_fallback_used"] = True
    elif case == "revision":
        embedding["model_revision"] = ""
    _write_json(sidecar, payload)

    with pytest.raises(LongMemEvalV2EvidenceError, match=message):
        load_tier_evidence(package, sidecar, expected_tier="small")


def test_compiler_rejects_dataset_identity_not_bound_to_runtime_config(
    tmp_path: Path,
) -> None:
    package, sidecar = _build_tier(tmp_path / "dataset-identity", "small")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["dataset_manifest_sha256"] = "2" * 64
    _write_json(sidecar, payload)

    with pytest.raises(LongMemEvalV2EvidenceError, match="caller-pinned sidecar dataset identity"):
        load_tier_evidence(package, sidecar, expected_tier="small")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "content-free query evidence fields"),
        ("empty", "must contain between 2 and 16 operations"),
        ("sequence_gap", "sequence numbers must be contiguous"),
        ("empty_invocation", "invocation_id must be a non-empty string"),
        ("nondeterministic_invocation", "nondeterministic invocation_id"),
        ("depth", "depth exceeds 2"),
        ("duplicate_memory_id", "unique opaque memory IDs"),
        ("unlinked_seed", "seeds did not come from one preceding"),
        ("token_mismatch", "delivered-token total does not equal query_tokens"),
        ("latency", "latency_ms exceeds official query latency"),
        ("content", "only content-free trace fields"),
        ("no_successful_expand", "must prove successful recall_memory"),
    ],
)
def test_compiler_rejects_malformed_or_unproved_per_query_operation_traces(
    tmp_path: Path, case: str, message: str
) -> None:
    package, sidecar = _build_tier(tmp_path / case, "small")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    query = payload["operating_points"][0]["queries"][0]
    operations = query["operations"]
    if case == "missing":
        query.pop("operations")
    elif case == "empty":
        query["operations"] = []
    elif case == "sequence_gap":
        operations[1]["sequence"] = 2
    elif case == "empty_invocation":
        operations[0]["invocation_id"] = ""
    elif case == "nondeterministic_invocation":
        operations[0]["invocation_id"] = "inv_" + "0" * 64
    elif case == "depth":
        operations[1]["depth"] = 3
    elif case == "duplicate_memory_id":
        operations[1]["result_memory_ids"] = [
            operations[1]["result_memory_ids"][0],
            operations[1]["result_memory_ids"][0],
        ]
    elif case == "unlinked_seed":
        operations[1]["seed_memory_ids"] = ["mem_" + "f" * 64]
    elif case == "token_mismatch":
        operations[1]["delivered_tokens"] += 1
    elif case == "latency":
        operations[0]["latency_ms"] = query["query_latency_ms"] + 1
    elif case == "content":
        operations[0]["raw_content"] = "must never appear in proof traces"
    elif case == "no_successful_expand":
        operations[0]["delivered_tokens"] = query["query_tokens"]
        operations[1]["success"] = False
        operations[1]["result_memory_ids"] = []
        operations[1]["delivered_tokens"] = 0
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(case)
    _write_json(sidecar, payload)

    with pytest.raises(LongMemEvalV2EvidenceError, match=message):
        load_tier_evidence(package, sidecar, expected_tier="small")


def test_compiler_binds_each_sidecar_trace_to_official_post_query_metadata(
    tmp_path: Path,
) -> None:
    package, sidecar = _build_tier(tmp_path / "trace-binding", "small")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["operating_points"][0]["queries"][0]["operations"][0]["latency_ms"] = 2.5
    _write_json(sidecar, payload)

    with pytest.raises(
        LongMemEvalV2EvidenceError, match="canonical query-proof SHA does not match"
    ):
        load_tier_evidence(package, sidecar, expected_tier="small")


def test_compiler_rejects_tampered_file_hash_and_cross_tier_protocol_mix(
    tmp_path: Path,
) -> None:
    package, sidecar = _build_tier(tmp_path / "hash", "small")
    (package / "code_file.py").write_text("# tampered\n", encoding="utf-8")

    with pytest.raises(LongMemEvalV2EvidenceError, match="evidence hashes differ"):
        load_tier_evidence(package, sidecar, expected_tier="small")

    small_package, small_sidecar = _build_tier(tmp_path / "mixed", "small")
    medium_package, medium_sidecar = _build_tier(tmp_path / "mixed", "medium")
    medium_payload = json.loads(medium_sidecar.read_text(encoding="utf-8"))
    for domain in ("web", "enterprise"):
        args_path = medium_package / f"operating_points/balanced/{domain}/run_args.json"
        args = json.loads(args_path.read_text(encoding="utf-8"))
        args["temperature"] = 0.1
        _write_json(args_path, args)
    memory_config = json.loads(
        (
            medium_package / "operating_points/balanced/web/runtime_inputs/memory_config.json"
        ).read_text(encoding="utf-8")
    )
    changed_args = json.loads(
        (medium_package / "operating_points/balanced/web/run_args.json").read_text(encoding="utf-8")
    )
    medium_payload["operating_points"][0]["protocol_sha256"] = _protocol_sha256(
        changed_args, memory_config, "swarmbrain"
    )
    _write_json(medium_sidecar, medium_payload)
    _refresh_package_hashes(medium_package, medium_sidecar)

    small = load_tier_evidence(small_package, small_sidecar, expected_tier="small")
    medium = load_tier_evidence(medium_package, medium_sidecar, expected_tier="medium")
    with pytest.raises(LongMemEvalV2EvidenceError, match="different operating-point protocols"):
        build_report(small, medium)


def test_cli_requires_explicit_fresh_output_and_writes_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small_package, small_sidecar = _build_tier(tmp_path / "cli", "small")
    medium_package, medium_sidecar = _build_tier(tmp_path / "cli", "medium")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_longmemeval_v2_report.py",
            "--small-package",
            str(small_package),
            "--small-sidecar",
            str(small_sidecar),
            "--medium-package",
            str(medium_package),
            "--medium-sidecar",
            str(medium_sidecar),
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dataset"]["questions"] == EXPECTED_QUESTIONS
    assert main() == 2
