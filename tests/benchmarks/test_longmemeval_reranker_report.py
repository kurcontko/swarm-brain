from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest
from _longmemeval_common import retrieve_question
from benchmarks.integrations.longmemeval_reranker import (
    LongMemEvalRerankerEvidenceError,
    build_run_manifest,
    build_trace_row,
    compile_longmemeval_reranker_report,
)
from benchmarks.integrations.longmemeval_reranker.evidence import (
    build_core_request,
    candidate_core_input,
    canonical_policy,
)

from swarmbrain.domain.reranking import (
    LearnedRerankerComponent,
    LearnedRerankerIdentity,
    LearnedRerankTrace,
    LearnedRerankUsage,
    learned_reranker_model_bundle_payload,
    learned_reranker_tokenizer_bundle_payload,
    rerank_sha256_json,
    rerank_sha256_text,
)
from swarmbrain.retrieval.learned_reranking import (
    build_learned_rerank_result,
    canonical_memory_rerank_input,
    request_usage_dimensions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTION_TYPES = (
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
    "single-session-user",
)


def _identity() -> LearnedRerankerIdentity:
    component = LearnedRerankerComponent(
        role="cross-encoder",
        model="fixture-qwen3-reranker",
        revision="1" * 40,
        model_artifact_sha256="2" * 64,
        tokenizer_revision="3" * 40,
        tokenizer_artifact_sha256="4" * 64,
        weight=1.0,
    )
    components = (component,)
    return LearnedRerankerIdentity(
        provider="fixture-local-jsonl",
        model="fixture-qwen3-control",
        revision="5" * 40,
        components=components,
        model_artifact_sha256=rerank_sha256_json(learned_reranker_model_bundle_payload(components)),
        tokenizer_artifact_sha256=rerank_sha256_json(
            learned_reranker_tokenizer_bundle_payload(components)
        ),
        deployment_manifest_sha256="6" * 64,
        adapter_artifact_sha256="7" * 64,
        runtime_environment_sha256="8" * 64,
        protocol_revision="fixture-jsonl-v1",
    )


def _record(index: int) -> dict[str, Any]:
    case_id = f"q{index:03d}" + ("_abs" if index == 0 else "")
    session_ids = [f"session-{index:03d}-{position:02d}" for position in range(12)]
    return {
        "question_id": case_id,
        "question_type": QUESTION_TYPES[index % len(QUESTION_TYPES)],
        "question": f"Which session contains answer {index}? Żółć",
        "question_date": "2026/08/09 (Sun) 12:00",
        "answer": f"answer-{index}",
        "answer_session_ids": [session_ids[11 if index % 3 == 0 else 0]],
        "haystack_session_ids": session_ids,
        "haystack_dates": ["2026/08/01 (Sat) 00:00"] * len(session_ids),
        "haystack_sessions": [
            [
                {"role": "user", "content": f"question context {index}/{position}"},
                {"role": "assistant", "content": f"stored answer {index}/{position}"},
            ]
            for position in range(len(session_ids))
        ],
    }


def _source_case(record: dict[str, Any]) -> dict[str, Any]:
    fused = [
        f"{position:03d}:{session_id}"
        for position, session_id in enumerate(record["haystack_session_ids"])
    ]
    answers = set(record["answer_session_ids"])
    relevant = [
        candidate_id
        for candidate_id, session_id in zip(fused, record["haystack_session_ids"], strict=True)
        if session_id in answers
    ]
    return {
        "case_id": record["question_id"],
        "category": record["question_type"],
        "abstention_question": str(record["question_id"]).endswith("_abs"),
        "relevant_ids": relevant,
        "haystack_sessions": len(fused),
        "degraded_lanes": [],
        "rankings": {"fused": fused, "final": fused[:10]},
    }


def _provider_ids(case_id: str, count: int) -> list[str]:
    return [
        str(uuid5(NAMESPACE_URL, f"fixture/{case_id}/memory/{position}"))
        for position in range(count)
    ]


def _trace_row(
    index: int,
    record: dict[str, Any],
    source_case: dict[str, Any],
    identity: LearnedRerankerIdentity,
    *,
    request_id: str | None = None,
    provider_request_id: str | None = None,
    usage_delta: dict[str, int] | None = None,
) -> dict[str, Any]:
    policy = canonical_policy(identity)
    evaluation_ids = list(source_case["rankings"]["fused"])
    provider_ids = _provider_ids(str(record["question_id"]), len(evaluation_ids))
    resolved_request_id = request_id or str(uuid5(NAMESPACE_URL, f"fixture/request/{index}"))
    request = build_core_request(
        record,
        evaluation_ids,
        provider_ids,
        policy,
        request_id=resolved_request_id,
    )
    gold_evaluation_id = source_case["relevant_ids"][0]
    gold_position = evaluation_ids.index(gold_evaluation_id)
    if index % 3 == 0:
        scores = [0.9 - position * 0.01 for position in range(len(provider_ids))]
        scores[gold_position] = 1.0
    elif index % 3 == 1:
        scores = [0.9 - position * 0.01 for position in range(len(provider_ids))]
        scores[gold_position] = 0.0
    else:
        scores = [1.0 - position * 0.01 for position in range(len(provider_ids))]
    dimensions = request_usage_dimensions(request)
    if usage_delta:
        for field, delta in usage_delta.items():
            dimensions[field] += delta
    input_tokens = 200 + index
    usage = LearnedRerankUsage(
        provider_reported=True,
        **dimensions,
        input_tokens=input_tokens,
        output_tokens=0,
        total_tokens=input_tokens,
        tokenized_input_sha256=rerank_sha256_text(f"tokenized-input-{index}"),
    )
    result = build_learned_rerank_result(
        request,
        scores=scores,
        usage=usage,
        provider_request_id=provider_request_id or f"provider-{index}",
    )
    source_rank = {candidate_id: rank for rank, candidate_id in enumerate(provider_ids)}
    score_by_id = dict(zip(provider_ids, scores, strict=True))
    output_ids = tuple(
        sorted(provider_ids, key=lambda item: (-score_by_id[item], source_rank[item]))
    )
    trace = LearnedRerankTrace(
        policy=policy,
        identity=identity,
        attempted=True,
        applied=True,
        degraded=False,
        serializer_revision=request.serializer_revision,
        request_id=request.request_id,
        provider_request_id=result.receipt.provider_request_id,
        request_sha256=request.request_sha256,
        query_sha256=request.query_sha256,
        candidate_pool_sha256=request.candidate_pool_sha256,
        candidate_document_sha256={
            candidate.candidate_id: candidate.document_sha256 for candidate in request.candidates
        },
        candidate_temporal_sha256={
            candidate.candidate_id: candidate.temporal_sha256 for candidate in request.candidates
        },
        input_ids=tuple(provider_ids),
        output_ids=output_ids,
        scores=result.scores,
        usage=usage,
        response_sha256=result.receipt.response_sha256,
        latency_ms=1.0 + index,
    )
    return build_trace_row(
        case_index=index,
        record=record,
        source_case=source_case,
        policy=policy,
        learned_trace=trace,
    )


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


@pytest.fixture
def evidence(tmp_path: Path) -> dict[str, Any]:
    records = [_record(index) for index in range(8)]
    source_cases = [_source_case(record) for record in records]
    dataset_path = tmp_path / "dataset.json"
    source_path = tmp_path / "source.json"
    traces_path = tmp_path / "traces.jsonl"
    run_path = tmp_path / "run.json"
    _write_json(dataset_path, records)
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    _write_json(
        source_path,
        {
            "artifact_type": "swarmbrain-retrieval-eval-run",
            "schema_version": 2,
            "protocol_version": "swarmbrain-longmemeval-retrieval-v2",
            "track": "longmemeval-s",
            "granularity": "one memory per haystack session",
            "dataset": {
                "name": "LongMemEval-S",
                "source": "fixture",
                "sha256": dataset_sha256,
                "total_questions": len(records),
                "evaluated_questions": len(records),
                "sample_seed": None,
            },
            "recall_limit": 10,
            "saved_ranking_depth": 50,
            "cases": source_cases,
        },
    )
    identity = _identity()
    rows = [
        _trace_row(index, record, source_case, identity)
        for index, (record, source_case) in enumerate(zip(records, source_cases, strict=True))
    ]
    _write_jsonl(traces_path, rows)
    run = build_run_manifest(
        created_at_utc="2026-08-09T12:00:00+00:00",
        dataset_sha256=dataset_sha256,
        question_count=len(records),
        source_retrieval_path=source_path,
        traces_path=traces_path,
        identity=identity,
        policy=canonical_policy(identity),
        artifact_root=tmp_path,
        code_root=REPO_ROOT,
        trace_rows=rows,
    )
    _write_json(run_path, run)
    return {
        "root": tmp_path,
        "records": records,
        "source_cases": source_cases,
        "identity": identity,
        "dataset_sha256": dataset_sha256,
        "dataset": dataset_path,
        "source": source_path,
        "traces": traces_path,
        "run_path": run_path,
        "run": run,
        "rows": rows,
    }


def _compile(evidence: dict[str, Any], output: str = "report.json") -> dict[str, Any]:
    return compile_longmemeval_reranker_report(
        "run.json",
        evidence["dataset"],
        output,
        artifact_root=evidence["root"],
        code_root=REPO_ROOT,
        expected_dataset_sha256=evidence["dataset_sha256"],
        expected_question_count=len(evidence["records"]),
        require_publishable_source=False,
        require_current_source_implementation=False,
    )


def _rebind_traces(evidence: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    _write_jsonl(evidence["traces"], rows)
    raw = evidence["traces"].read_bytes()
    run = deepcopy(evidence["run"])
    run["traces_artifact"].update(
        {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows)}
    )
    _write_json(evidence["run_path"], run)


def test_compiler_replays_core_receipts_and_paired_slices(evidence: dict[str, Any]) -> None:
    report = _compile(evidence)

    assert report["benchmark"]["canonical_official_dataset"] is False
    assert report["coverage"]["questions"] == 8
    assert report["validation"]["core_request_digests_recomputed"] is True
    assert report["validation"]["core_response_receipts_recomputed"] is True
    assert report["validation"]["pass_threshold_encoded"] is False
    assert set(report["paired_metrics"]) == {
        "overall",
        "temporal",
        "conflict",
        "multi_session",
    }
    overall = report["paired_metrics"]["overall"]["k=10"]
    assert overall["bootstrap"] == {
        "method": "percentile-paired-question-bootstrap-v1",
        "unit": "question",
        "paired": True,
        "resamples": 10_000,
        "seed": 20_260_809,
        "confidence": 0.95,
    }
    assert overall["paired_delta"]["recall_at_k"]["improved_questions"] > 0
    assert (
        report["paired_metrics"]["conflict"]["k=5"]["paired_delta"]["mrr_at_k"][
            "regressed_questions"
        ]
        > 0
    )


@pytest.mark.asyncio
async def test_dataset_reconstruction_matches_real_longmemeval_memory_projection(
    evidence: dict[str, Any],
) -> None:
    record = evidence["records"][0]
    policy = canonical_policy(evidence["identity"])
    retrieved = await retrieve_question(record, limit=10, use_dense=False)
    by_id = retrieved.by_memory_id

    for hit in retrieved.execution.bundle.hits:
        session = by_id[hit.memory.memory_id]
        assert candidate_core_input(record, session.position, policy) == (
            canonical_memory_rerank_input(hit.memory, policy)
        )


def test_compiler_rejects_self_attested_candidate_text(evidence: dict[str, Any]) -> None:
    rows = deepcopy(evidence["rows"])
    rows[0]["candidate_documents"][0]["document_sha256"] = "f" * 64
    _rebind_traces(evidence, rows)

    with pytest.raises(LongMemEvalRerankerEvidenceError, match="candidate_documents"):
        _compile(evidence)


def test_compiler_rejects_omitted_or_duplicate_output(evidence: dict[str, Any]) -> None:
    rows = deepcopy(evidence["rows"])
    output_ids = rows[0]["learned"]["trace"]["output_ids"]
    output_ids[-1] = output_ids[0]
    _rebind_traces(evidence, rows)

    with pytest.raises(LongMemEvalRerankerEvidenceError, match="valid core trace"):
        _compile(evidence)


def test_compiler_rejects_added_output_candidate(evidence: dict[str, Any]) -> None:
    rows = deepcopy(evidence["rows"])
    rows[0]["learned"]["trace"]["output_ids"].append(str(uuid5(NAMESPACE_URL, "added")))
    _rebind_traces(evidence, rows)

    with pytest.raises(LongMemEvalRerankerEvidenceError, match="valid core trace"):
        _compile(evidence)


def test_compiler_rejects_nonfinite_or_out_of_range_score(evidence: dict[str, Any]) -> None:
    rows = deepcopy(evidence["rows"])
    rows[0]["learned"]["trace"]["scores"][0]["score"] = 1.01
    _rebind_traces(evidence, rows)

    with pytest.raises(LongMemEvalRerankerEvidenceError, match="valid core trace"):
        _compile(evidence)


def test_compiler_rejects_reused_request_ids_even_with_valid_digests(
    evidence: dict[str, Any],
) -> None:
    rows = deepcopy(evidence["rows"])
    reused = rows[0]["learned"]["trace"]["request_id"]
    rows[1] = _trace_row(
        1,
        evidence["records"][1],
        evidence["source_cases"][1],
        evidence["identity"],
        request_id=reused,
    )
    _rebind_traces(evidence, rows)

    with pytest.raises(LongMemEvalRerankerEvidenceError, match="reuses a request ID"):
        _compile(evidence)


def test_compiler_rejects_reused_provider_request_ids_with_valid_receipts(
    evidence: dict[str, Any],
) -> None:
    rows = deepcopy(evidence["rows"])
    rows[1] = _trace_row(
        1,
        evidence["records"][1],
        evidence["source_cases"][1],
        evidence["identity"],
        provider_request_id="provider-0",
    )
    _rebind_traces(evidence, rows)

    with pytest.raises(LongMemEvalRerankerEvidenceError, match="reuses a provider request ID"):
        _compile(evidence)


def test_compiler_rejects_response_digest_or_identity_drift(evidence: dict[str, Any]) -> None:
    rows = deepcopy(evidence["rows"])
    rows[0]["learned"]["trace"]["response_sha256"] = "f" * 64
    _rebind_traces(evidence, rows)

    with pytest.raises(LongMemEvalRerankerEvidenceError, match="response receipt"):
        _compile(evidence)


def test_compiler_rejects_response_model_revision_drift(evidence: dict[str, Any]) -> None:
    rows = deepcopy(evidence["rows"])
    rows[0]["learned"]["trace"]["identity"]["revision"] = "9" * 40
    _rebind_traces(evidence, rows)

    with pytest.raises(LongMemEvalRerankerEvidenceError, match="valid core trace"):
        _compile(evidence)


def test_compiler_rejects_k_or_tokenizer_input_change_between_arms(
    evidence: dict[str, Any],
) -> None:
    rows = deepcopy(evidence["rows"])
    rows[0]["baseline"]["input"]["k_values"] = [5]
    rows[1]["learned"]["input"]["tokenized_input_sha256"] = "f" * 64
    _rebind_traces(evidence, rows)

    with pytest.raises(LongMemEvalRerankerEvidenceError, match="changes query"):
        _compile(evidence)


def test_compiler_rejects_response_bound_but_wrong_byte_accounting(
    evidence: dict[str, Any],
) -> None:
    rows = deepcopy(evidence["rows"])
    rows[0] = _trace_row(
        0,
        evidence["records"][0],
        evidence["source_cases"][0],
        evidence["identity"],
        usage_delta={"document_bytes": 1},
    )
    _rebind_traces(evidence, rows)

    with pytest.raises(LongMemEvalRerankerEvidenceError, match="usage.document_bytes"):
        _compile(evidence)
