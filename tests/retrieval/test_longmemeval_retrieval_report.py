from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import run_retrieval_eval as retrieval_eval
from build_longmemeval_retrieval_report import (
    RetrievalReportError,
    _repo_input,
    _strict_object,
    compile_report,
)
from evaluate_sota_readiness import _SAFE_REPLAY_COMPILERS

REPO_ROOT = Path(__file__).resolve().parents[2]


def _canonical_payload() -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for index in range(500):
        abstention = index >= 470
        suffix = "_abs" if abstention else ""
        case_id = f"case-{index:03d}{suffix}"
        relevant_id = f"000:gold-{index:03d}{suffix}"
        cases.append(
            {
                "case_id": case_id,
                "category": "single-session-user",
                "abstention_question": abstention,
                "relevant_ids": [relevant_id],
                "haystack_sessions": 414 if index == 0 else 47,
                "wall_ms": 1.0,
                "lane_latency_ms": {"dense": 0.5},
                "degraded_lanes": [],
                "final_relevance": [1.0],
                "final_tokens": [100],
                "rankings": {"dense": [relevant_id], "final": [relevant_id]},
                "temporal_routing": None,
            }
        )
    return {
        "artifact_type": retrieval_eval.RUN_ARTIFACT_TYPE,
        "schema_version": retrieval_eval.RETRIEVAL_ARTIFACT_SCHEMA_VERSION,
        "protocol_version": retrieval_eval.RETRIEVAL_PROTOCOL_VERSION,
        "implementation": retrieval_eval.retrieval_implementation_fingerprint(),
        "track": "longmemeval-s",
        "dataset": {
            "name": "LongMemEval-S",
            "source": (
                "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
                "resolve/main/longmemeval_s_cleaned.json"
            ),
            "sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
            "total_questions": 500,
            "evaluated_questions": 500,
            "sample_seed": None,
        },
        "granularity": "one memory per haystack session",
        "recall_limit": 10,
        "saved_ranking_depth": 50,
        "dense_lane_enabled": True,
        "temporal_query_routing": {
            "enabled": False,
            "parser": None,
            "session_valid_from": "LongMemEval haystack_dates normalized to UTC",
        },
        "embedding": {
            "provider": "OpenAICompatibleEmbeddingProvider",
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "dimensions": 1024,
            "response_model_requirement": "Qwen/Qwen3-Embedding-0.6B",
            "query_instruction_sha256": (
                "a695bbf99f6e2c59bbedb4ca2b397a995afbe92114c2d965a84acfac4253727f"
            ),
        },
        "embedding_call_accounting": {
            "source": "provider-observed",
            "document_inputs": 23_867,
            "document_batch_calls": 500,
            "query_calls": 500,
            "successful_http_calls": 1000,
            "http_attempts": 1000,
        },
        "context_token_accounting": retrieval_eval.context_token_accounting(),
        "cases": cases,
    }


def _write_run(root: Path, payload: dict[str, object]) -> Path:
    relative = Path("evidence") / "run.json"
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return relative


def test_offline_compiler_rebuilds_and_hash_binds_canonical_report(tmp_path: Path) -> None:
    relative = _write_run(tmp_path, _canonical_payload())
    output = tmp_path / "report.json"

    report = compile_report(relative, output, repo_root=tmp_path)

    raw = (tmp_path / relative).read_bytes()
    assert report["run_artifact"]["bytes"] == len(raw)
    assert report["run_artifact"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["overall"]["k=10"]["final"]["cases"] == 500
    assert report["by_abstention"]["k=10"]["answerable_questions"]["final"]["cases"] == 470
    assert report["by_abstention"]["k=10"]["abstention_questions"]["final"]["cases"] == 30
    assert report["execution"]["context_token_accounting"] == (
        retrieval_eval.context_token_accounting()
    )
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_compiler_rejects_strict_json_and_unsafe_input_paths(tmp_path: Path) -> None:
    with pytest.raises(RetrievalReportError, match="duplicate JSON field"):
        _strict_object(b'{"schema_version":2,"schema_version":2}', label="fixture")
    with pytest.raises(RetrievalReportError, match="non-finite"):
        _strict_object(b'{"value":NaN}', label="fixture")
    with pytest.raises(RetrievalReportError, match="repository-local"):
        _repo_input(Path("../run.json"), repo_root=tmp_path)

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "run.json"
    link.symlink_to(target)
    with pytest.raises(RetrievalReportError, match="symbolic links"):
        _repo_input(Path("run.json"), repo_root=tmp_path)


def test_compiler_rejects_self_consistent_but_noncurrent_implementation(tmp_path: Path) -> None:
    payload = _canonical_payload()
    implementation = payload["implementation"]
    assert isinstance(implementation, dict)
    files = dict(implementation["files"])
    files["scripts/run_retrieval_eval.py"] = "0" * 64
    implementation["files"] = files
    implementation["tree_sha256"] = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    relative = _write_run(tmp_path, payload)

    with pytest.raises(RetrievalReportError, match="does not match current tree"):
        compile_report(relative, tmp_path / "report.json", repo_root=tmp_path)


def test_sota_gate_replays_canonical_run_but_requires_an_exact_tokenizer() -> None:
    assert _SAFE_REPLAY_COMPILERS["scripts/build_longmemeval_retrieval_report.py"] == {"--run"}
    manifest = json.loads((REPO_ROOT / "benchmarks" / "sota" / "manifest.json").read_text())
    gate = next(item for item in manifest["gates"] if item["id"] == "longmemeval_retrieval_context")
    assert gate["compiler_replay"] == {
        "compiler": "scripts/build_longmemeval_retrieval_report.py",
        "arguments": [
            "--run",
            "benchmarks/retrieval/longmemeval-s-memory-openai-run.json",
        ],
    }
    checks = {(item["pointer"], item["operator"]): item for item in gate["checks"]}
    exact = checks[("/execution/context_token_accounting/exact_model_tokenizer", "eq")]
    assert exact["expected"] is True
    assert retrieval_eval.context_token_accounting()["exact_model_tokenizer"] is False
    assert checks[("/by_abstention/k=10/answerable_questions/final/cases", "eq")]["expected"] == 470
    assert checks[("/by_abstention/k=10/abstention_questions/final/cases", "eq")]["expected"] == 30
