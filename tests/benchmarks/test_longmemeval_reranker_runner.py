from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import run_longmemeval_reranker_ab as runner
from _longmemeval_common import retrieve_question
from benchmarks.integrations.longmemeval_reranker import (
    compile_longmemeval_reranker_report,
)

from swarmbrain.adapters.reranking.local_jsonl import LOCAL_RERANKER_MANIFEST_SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "fixture/local-reranker"
REVISION = "a" * 40


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _record(index: int) -> dict[str, Any]:
    session_ids = [f"session-{index}-{position}" for position in range(12)]
    categories = ("temporal-reasoning", "knowledge-update", "multi-session")
    return {
        "question_id": f"runner-{index}",
        "question_type": categories[index % len(categories)],
        "question": f"Which shared answer belongs to fixture {index}?",
        "question_date": "2026/08/09 (Sun) 12:00",
        "answer": f"fixture {index}",
        "answer_session_ids": [session_ids[-1]],
        "haystack_session_ids": session_ids,
        "haystack_dates": ["2026/08/01 (Sat) 00:00"] * len(session_ids),
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": f"shared answer fixture {index} session {position}",
                },
                {
                    "role": "assistant",
                    "content": f"fixture detail {index}-{position}",
                },
            ]
            for position in range(len(session_ids))
        ],
    }


async def _write_dataset_and_source(
    root: Path,
    *,
    question_count: int = 3,
) -> tuple[Path, Path, str, list[dict[str, Any]]]:
    records = [_record(index) for index in range(question_count)]
    dataset = root / "dataset.json"
    _write_json(dataset, records)
    digest = _sha256(dataset.read_bytes())
    cases: list[dict[str, Any]] = []
    for record in records:
        retrieved = await retrieve_question(record, limit=10, use_dense=False)
        fused = [
            retrieved.key_by_memory_id[item.canonical_id]
            for item in retrieved.execution.trace.fused_candidates
        ]
        assert len(fused) == len(record["haystack_session_ids"])
        answers = set(record["answer_session_ids"])
        relevant = [
            f"{position:03d}:{session_id}"
            for position, session_id in enumerate(record["haystack_session_ids"])
            if session_id in answers
        ]
        cases.append(
            {
                "case_id": record["question_id"],
                "category": record["question_type"],
                "abstention_question": False,
                "relevant_ids": relevant,
                "haystack_sessions": len(record["haystack_session_ids"]),
                "degraded_lanes": [],
                "rankings": {"fused": fused, "final": fused[:10]},
            }
        )
    source = root / "source.json"
    _write_json(
        source,
        {
            "artifact_type": "swarmbrain-retrieval-eval-run",
            "schema_version": 2,
            "protocol_version": "swarmbrain-longmemeval-retrieval-v2",
            "track": "longmemeval-s",
            "granularity": "one memory per haystack session",
            "dataset": {
                "name": "LongMemEval-S",
                "source": "fixture",
                "sha256": digest,
                "total_questions": len(records),
                "evaluated_questions": len(records),
                "sample_seed": None,
            },
            "recall_limit": 10,
            "saved_ranking_depth": 50,
            "cases": cases,
        },
    )
    return dataset, source, digest, records


def _write_deployment(root: Path) -> dict[str, Any]:
    executable = root / "fake-reranker.py"
    source = f"""#!{sys.executable}
import hashlib
import json
import sys
import uuid

def canonical(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))

manifest = sys.argv[sys.argv.index("--manifest") + 1]
for line in sys.stdin:
    with open(manifest + ".calls", "a", encoding="utf-8") as marker:
        marker.write("1\\n")
    raw = line.rstrip("\\n")
    request = json.loads(raw)
    candidates = request["candidates"]
    scores = [
        {{"candidate_id": item["candidate_id"], "score": (index + 1) / len(candidates)}}
        for index, item in enumerate(candidates)
    ]
    query = request["query"]
    input_tokens = max(1, len(query.split()) + sum(len(item["document"].split()) for item in candidates))
    usage = {{
        "provider_reported": True,
        "candidate_count": len(candidates),
        "query_characters": len(query),
        "document_characters": sum(len(item["document"]) for item in candidates),
        "temporal_characters": sum(len(item["temporal_context"]) for item in candidates),
        "query_bytes": len(query.encode("utf-8")),
        "document_bytes": sum(len(item["document"].encode("utf-8")) for item in candidates),
        "temporal_bytes": sum(len(item["temporal_context"].encode("utf-8")) for item in candidates),
        "request_bytes": len(raw.encode("utf-8")),
        "input_tokens": input_tokens,
        "output_tokens": 0,
        "total_tokens": input_tokens,
        "tokenized_input_sha256": hashlib.sha256((request["request_sha256"] + ":tokens").encode()).hexdigest(),
    }}
    receipt = {{
        "identity": request["identity"],
        "request_sha256": request["request_sha256"],
        "provider_request_id": str(uuid.uuid4()),
        "usage": usage,
    }}
    response = {{
        "protocol_schema": "swarmbrain.learned-rerank.response.v1",
        "scores": scores,
        "receipt": receipt,
    }}
    receipt["response_sha256"] = hashlib.sha256(canonical(response).encode()).hexdigest()
    print(canonical(response), flush=True)
"""
    executable.write_text(source, encoding="utf-8")
    executable.chmod(0o700)
    model = root / "model.bin"
    tokenizer = root / "tokenizer.json"
    model.write_bytes(b"synthetic-not-a-model")
    tokenizer.write_bytes(b'{"synthetic":true}')
    manifest = root / "deployment.json"
    manifest_payload = {
        "schema": LOCAL_RERANKER_MANIFEST_SCHEMA,
        "model": MODEL,
        "revision": REVISION,
        "components": [
            {
                "role": "cross_encoder",
                "model": MODEL,
                "revision": REVISION,
                "model_artifacts": [
                    {
                        "path": model.name,
                        "sha256": _sha256(model.read_bytes()),
                        "size_bytes": model.stat().st_size,
                    }
                ],
                "tokenizer_revision": REVISION,
                "tokenizer_artifacts": [
                    {
                        "path": tokenizer.name,
                        "sha256": _sha256(tokenizer.read_bytes()),
                        "size_bytes": tokenizer.stat().st_size,
                    }
                ],
                "weight": 1.0,
            }
        ],
    }
    manifest_raw = json.dumps(
        manifest_payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest.write_bytes(manifest_raw)
    return {
        "executable": executable,
        "executable_sha256": _sha256(source.encode("utf-8")),
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_raw),
        "calls": Path(str(manifest) + ".calls"),
    }


def _run_kwargs(
    root: Path,
    dataset: Path,
    digest: str,
    records: list[dict[str, Any]],
    deployment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_path": dataset,
        "source_retrieval_path": Path("source.json"),
        "traces_path": Path("traces.jsonl"),
        "run_path": Path("run.json"),
        "executable_path": deployment["executable"],
        "executable_sha256": deployment["executable_sha256"],
        "deployment_manifest_path": deployment["manifest"],
        "deployment_manifest_sha256": deployment["manifest_sha256"],
        "required_model": MODEL,
        "required_revision": REVISION,
        "artifact_root": root,
        "code_root": REPO_ROOT,
        "expected_dataset_sha256": digest,
        "expected_question_count": len(records),
        "require_publishable_source": False,
        "require_current_source_implementation": False,
        "use_dense": False,
    }


@pytest.mark.asyncio
async def test_runner_produces_compilable_raw_evidence_without_dense_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _source, digest, records = await _write_dataset_and_source(tmp_path)
    deployment = _write_deployment(tmp_path)

    def forbidden_embedding_configuration(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no embedding provider may be configured in this test")

    monkeypatch.setattr(runner, "configure_embeddings", forbidden_embedding_configuration)
    manifest = await runner.run_longmemeval_reranker_ab(
        **_run_kwargs(tmp_path, dataset, digest, records, deployment)
    )

    assert manifest["rerank_policy"]["alpha"] == 1.0
    assert manifest["rerank_policy"]["window"] == 50
    assert manifest["traces_artifact"]["rows"] == len(records)
    assert "src/swarmbrain/application/memory_service.py" in manifest["implementation"]["files"]
    assert "src/swarmbrain/retrieval/temporal_query.py" in manifest["implementation"]["files"]
    assert deployment["calls"].read_text(encoding="utf-8").splitlines() == ["1", "1", "1"]
    report = compile_longmemeval_reranker_report(
        "run.json",
        dataset,
        "report.json",
        artifact_root=tmp_path,
        code_root=REPO_ROOT,
        expected_dataset_sha256=digest,
        expected_question_count=len(records),
        require_publishable_source=False,
        require_current_source_implementation=False,
    )
    assert report["coverage"]["questions"] == len(records)
    assert not list(tmp_path.glob(".traces.jsonl.*"))
    assert not list(tmp_path.glob(".run.json.*"))


@pytest.mark.asyncio
async def test_runner_resumes_only_after_strict_prefix_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _source, digest, records = await _write_dataset_and_source(
        tmp_path,
        question_count=3,
    )
    deployment = _write_deployment(tmp_path)
    kwargs = _run_kwargs(tmp_path, dataset, digest, records, deployment)
    real_retrieve = runner.retrieve_question
    calls = 0

    async def interrupt_before_second(*args: object, **kwargs: object) -> Any:
        nonlocal calls
        if calls == 1:
            raise RuntimeError("synthetic interruption")
        calls += 1
        return await real_retrieve(*args, **kwargs)

    monkeypatch.setattr(runner, "retrieve_question", interrupt_before_second)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        await runner.run_longmemeval_reranker_ab(**kwargs)
    trace_path = tmp_path / "traces.jsonl"
    assert trace_path.read_bytes().endswith(b"\n")
    assert len(trace_path.read_text(encoding="utf-8").splitlines()) == 1
    assert not (tmp_path / "run.json").exists()

    monkeypatch.setattr(runner, "retrieve_question", real_retrieve)
    manifest = await runner.run_longmemeval_reranker_ab(**kwargs)
    assert manifest["traces_artifact"]["rows"] == 3
    assert len(trace_path.read_text(encoding="utf-8").splitlines()) == 3
    assert len(deployment["calls"].read_text(encoding="utf-8").splitlines()) == 3


@pytest.mark.asyncio
async def test_runner_rejects_live_fused_order_drift_before_accepting_trace(
    tmp_path: Path,
) -> None:
    dataset, source, digest, records = await _write_dataset_and_source(
        tmp_path,
        question_count=1,
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    fused = payload["cases"][0]["rankings"]["fused"]
    fused[0], fused[1] = fused[1], fused[0]
    payload["cases"][0]["rankings"]["final"] = fused[:10]
    _write_json(source, payload)
    deployment = _write_deployment(tmp_path)

    with pytest.raises(runner.LongMemEvalRerankerRunError, match="pre-learned fused"):
        await runner.run_longmemeval_reranker_ab(
            **_run_kwargs(tmp_path, dataset, digest, records, deployment)
        )

    assert not (tmp_path / "traces.jsonl").exists()
    assert not (tmp_path / "run.json").exists()
    assert deployment["calls"].read_text(encoding="utf-8").splitlines() == ["1"]


@pytest.mark.asyncio
async def test_runner_rejects_nonterminated_resume_prefix_without_launching_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _source, digest, records = await _write_dataset_and_source(
        tmp_path,
        question_count=2,
    )
    deployment = _write_deployment(tmp_path)
    kwargs = _run_kwargs(tmp_path, dataset, digest, records, deployment)
    real_retrieve = runner.retrieve_question
    calls = 0

    async def interrupt_before_second(*args: object, **kwargs: object) -> Any:
        nonlocal calls
        if calls == 1:
            raise RuntimeError("synthetic interruption")
        calls += 1
        return await real_retrieve(*args, **kwargs)

    monkeypatch.setattr(runner, "retrieve_question", interrupt_before_second)
    with pytest.raises(RuntimeError):
        await runner.run_longmemeval_reranker_ab(**kwargs)
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes(trace_path.read_bytes().removesuffix(b"\n"))
    launches_before = deployment["calls"].read_text(encoding="utf-8")
    monkeypatch.setattr(runner, "retrieve_question", real_retrieve)

    with pytest.raises(runner.LongMemEvalRerankerRunError, match="newline terminated"):
        await runner.run_longmemeval_reranker_ab(**kwargs)

    assert deployment["calls"].read_text(encoding="utf-8") == launches_before


@pytest.mark.asyncio
async def test_runner_rejects_crlf_resume_prefix_without_new_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _source, digest, records = await _write_dataset_and_source(
        tmp_path,
        question_count=2,
    )
    deployment = _write_deployment(tmp_path)
    kwargs = _run_kwargs(tmp_path, dataset, digest, records, deployment)
    real_retrieve = runner.retrieve_question
    calls = 0

    async def interrupt_before_second(*args: object, **kwargs: object) -> Any:
        nonlocal calls
        if calls == 1:
            raise RuntimeError("synthetic interruption")
        calls += 1
        return await real_retrieve(*args, **kwargs)

    monkeypatch.setattr(runner, "retrieve_question", interrupt_before_second)
    with pytest.raises(RuntimeError):
        await runner.run_longmemeval_reranker_ab(**kwargs)
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_bytes(trace_path.read_bytes().replace(b"\n", b"\r\n"))
    launches_before = deployment["calls"].read_text(encoding="utf-8")
    monkeypatch.setattr(runner, "retrieve_question", real_retrieve)

    with pytest.raises(runner.LongMemEvalRerankerRunError, match="LF-only"):
        await runner.run_longmemeval_reranker_ab(**kwargs)

    assert deployment["calls"].read_text(encoding="utf-8") == launches_before


@pytest.mark.asyncio
async def test_runner_configures_no_embedding_client_when_local_pins_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _source, digest, records = await _write_dataset_and_source(
        tmp_path,
        question_count=1,
    )
    deployment = _write_deployment(tmp_path)
    kwargs = _run_kwargs(tmp_path, dataset, digest, records, deployment)
    kwargs.update(
        {
            "use_dense": True,
            "embeddings_base_url": "http://embedding.invalid/v1",
            "embeddings_model": "fixture/embedding",
        }
    )
    embedding_configured = False

    def fail_local_pins(**_kwargs: object) -> None:
        raise ValueError("synthetic local pin failure")

    def record_embedding_configuration(*_args: object, **_kwargs: object) -> None:
        nonlocal embedding_configured
        embedding_configured = True

    monkeypatch.setattr(runner, "LocalJsonlLearnedReranker", fail_local_pins)
    monkeypatch.setattr(runner, "configure_embeddings", record_embedding_configuration)

    with pytest.raises(ValueError, match="synthetic local pin failure"):
        await runner.run_longmemeval_reranker_ab(**kwargs)

    assert embedding_configured is False
    assert not (tmp_path / "traces.jsonl").exists()
    assert not (tmp_path / "run.json").exists()


@pytest.mark.asyncio
async def test_runner_rejects_implementation_drift_before_manifest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _source, digest, records = await _write_dataset_and_source(
        tmp_path,
        question_count=1,
    )
    deployment = _write_deployment(tmp_path)
    real_build_manifest = runner.build_run_manifest

    def build_drifted_manifest(*args: object, **kwargs: object) -> dict[str, Any]:
        manifest = real_build_manifest(*args, **kwargs)
        manifest["implementation"] = {
            "tree_sha256": "0" * 64,
            "files": {},
        }
        return manifest

    monkeypatch.setattr(runner, "build_run_manifest", build_drifted_manifest)

    with pytest.raises(runner.LongMemEvalRerankerRunError, match="implementation changed"):
        await runner.run_longmemeval_reranker_ab(
            **_run_kwargs(tmp_path, dataset, digest, records, deployment)
        )

    assert (tmp_path / "traces.jsonl").exists()
    assert not (tmp_path / "run.json").exists()
