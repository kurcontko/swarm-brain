from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import benchmarks.integrations.mem2act.report as mem2act_report_module
import pytest
from benchmarks.integrations.mem2act.contracts import (
    CorpusSession,
    DatasetFingerprint,
    IngestionResult,
    Mem2ActContractError,
    Mem2ActDataset,
    Mem2ActTask,
    OracleMemory,
    PublicConversationSession,
    ReaderRequest,
    ReaderResult,
    RetrievalResult,
    RetrievedMemory,
    ToolPrediction,
)
from benchmarks.integrations.mem2act.dataset import (
    OFFICIAL_MEM2ACT_SPEC,
    DatasetSpec,
    canonical_json,
    load_mem2act_dataset,
    sha256_file,
)
from benchmarks.integrations.mem2act.metrics import (
    paired_bootstrap,
    parse_tool_prediction,
    score_prediction,
)
from benchmarks.integrations.mem2act.provenance import implementation_fingerprint
from benchmarks.integrations.mem2act.report import (
    Mem2ActReportError,
    compile_mem2act_report,
)
from benchmarks.integrations.mem2act.runner import (
    BenchmarkConfig,
    Mem2ActEvaluator,
    write_benchmark_outputs,
)
from benchmarks.integrations.mem2act.runtime_bridge import (
    RuntimeMemoryBridge,
    benchmark_actor,
    build_openai_semantic_in_memory_bridge,
    build_public_in_memory_bridge,
)
from evaluate_sota_readiness import evaluate_manifest

from swarmbrain.adapters.embeddings import DeterministicEmbeddingProvider
from swarmbrain.application.runtime import build_in_memory_runtime

OFFICIAL_CHECKOUT = Path("/private/tmp/swarmbrain-mem2act")


def _readable_git_checkout(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def test_mem2act_implementation_fingerprint_covers_runtime_and_lockfiles() -> None:
    files = implementation_fingerprint()["files_sha256"]

    assert "pyproject.toml" in files
    assert "uv.lock" in files
    assert "src/swarmbrain/application/retrieval_service.py" in files
    assert "src/swarmbrain/adapters/memory/in_memory.py" in files


def _schema(name: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Call {name}",
        "parameters": {
            "type": "dict",
            "properties": properties,
            "required": list(properties),
        },
        "required": None,
    }


def _dataset() -> Mem2ActDataset:
    schemas = (
        _schema("find_weather", {"city": {"type": "string"}}),
        _schema(
            "order_item",
            {"item": {"type": "string"}, "quantity": {"type": "int"}},
        ),
    )
    catalog = tuple(
        {
            "schema_id": hashlib.sha256(canonical_json(schema).encode()).hexdigest(),
            "schema": schema,
        }
        for schema in schemas
    )
    catalog_hash = hashlib.sha256(canonical_json(catalog).encode()).hexdigest()
    tasks = (
        Mem2ActTask(
            qa_id="qa_001",
            query="What is the weather there?",
            source_conversation_ids=("source_weather",),
            oracle_memories=(
                OracleMemory(
                    attribute="Travel destination",
                    fact="The user is visiting Warsaw.",
                    source_text="I will be in Warsaw next week.",
                ),
            ),
            gold_tool_name="find_weather",
            gold_arguments={"city": "Warsaw"},
            target_tool_schema=schemas[0],
            complexity_level="L1",
        ),
        Mem2ActTask(
            qa_id="qa_002",
            query="Order my usual amount of that drink.",
            source_conversation_ids=("source_order",),
            oracle_memories=(
                OracleMemory(
                    attribute="Drink order",
                    fact="The user's usual order is two teas.",
                    source_text="I normally order two teas.",
                ),
            ),
            gold_tool_name="order_item",
            gold_arguments={"item": "tea", "quantity": 2},
            target_tool_schema=schemas[1],
            complexity_level="L2",
        ),
    )
    sessions = (
        CorpusSession(
            session_id="session_00001",
            original_conversation_ids=("source_weather", "distractor_1"),
            turns=(
                {
                    "role": "user",
                    "content": "I will be in Warsaw next week.",
                    "source_id": "source_weather",
                },
                {"role": "assistant", "content": "Noted.", "source_id": "source_weather"},
                {"role": "user", "content": "I like jazz.", "source_id": "distractor_1"},
            ),
            turn_count=2,
            token_count=20,
        ),
        CorpusSession(
            session_id="session_00002",
            original_conversation_ids=("source_order",),
            turns=(
                {
                    "role": "user",
                    "content": "I normally order two teas.",
                    "source_id": "source_order",
                },
            ),
            turn_count=1,
            token_count=8,
        ),
    )
    return Mem2ActDataset(
        tasks=tasks,
        sessions=sessions,
        tool_catalog=catalog,
        tool_catalog_sha256=catalog_hash,
        fingerprint=DatasetFingerprint(
            repo_commit="fixture",
            files_sha256={"qa": "a" * 64, "sessions": "b" * 64},
            task_count=2,
            session_count=2,
            unresolved_source_ids=(),
        ),
    )


class _FakeMemoryBridge:
    def __init__(self) -> None:
        self.ingested: tuple[PublicConversationSession, ...] = ()
        self.queries: list[tuple[str, int, int | None]] = []
        self.closed = False

    async def ingest(self, sessions: tuple[PublicConversationSession, ...]) -> IngestionResult:
        self.ingested = sessions
        return IngestionResult(memory_count=len(sessions), latency_ms=3.0)

    async def retrieve(
        self, query: str, *, limit: int, token_budget: int | None
    ) -> RetrievalResult:
        self.queries.append((query, limit, token_budget))
        return RetrievalResult(
            memories=(
                RetrievedMemory(
                    memory_id=f"memory-{len(self.queries)}",
                    content=f"retrieved evidence for: {query}",
                    score=0.75,
                    reasons=("lexical",),
                ),
            ),
            latency_ms=2.0,
            total_candidates=2,
            truncated=False,
        )

    async def close(self) -> None:
        self.closed = True


class _FakeReader:
    def __init__(self) -> None:
        self.requests: list[ReaderRequest] = []

    async def select_tool(self, request: ReaderRequest) -> ReaderResult:
        self.requests.append(request)
        if not request.memory_contexts:
            prediction = (
                {"name": "find_weather", "arguments": {"city": "Krakow"}}
                if "weather" in request.query
                else {"name": "find_weather", "arguments": {}}
            )
        elif request.memory_contexts[0].startswith("retrieved evidence"):
            prediction = (
                {"name": "find_weather", "arguments": {"city": "Warsaw"}}
                if "weather" in request.query
                else {
                    "name": "order_item",
                    "arguments": {"item": "tea", "quantity": 2, "note": "hot"},
                }
            )
        else:
            prediction = (
                {"name": "find_weather", "arguments": {"city": "Warsaw"}}
                if "weather" in request.query
                else {"name": "order_item", "arguments": {"item": "tea", "quantity": 2}}
            )
        return ReaderResult(
            raw_prediction=json.dumps(prediction),
            model="fake-reader-v1",
            prompt_tokens=11,
            completion_tokens=5,
            latency_ms=1.25,
            metadata={"provider_request_id": f"fake-{len(self.requests)}"},
        )


@pytest.mark.skipif(
    not _readable_git_checkout(OFFICIAL_CHECKOUT),
    reason="official checkout is not present or not readable",
)
def test_official_release_loads_all_400_tasks_and_429_sessions() -> None:
    dataset = load_mem2act_dataset(OFFICIAL_CHECKOUT)
    assert len(dataset.tasks) == 400
    assert len(dataset.sessions) == 429
    assert dataset.fingerprint.repo_commit == OFFICIAL_MEM2ACT_SPEC.repo_commit
    assert set(dataset.fingerprint.unresolved_source_ids) == set(
        OFFICIAL_MEM2ACT_SPEC.allowed_unresolved_source_ids
    )
    assert dataset.tool_catalog_sha256


def test_strict_loader_rejects_fingerprint_drift(tmp_path: Path) -> None:
    _, spec = _write_loader_fixture(tmp_path)
    assert len(load_mem2act_dataset(tmp_path, spec=spec, verify_git=False).tasks) == 2
    qa_path = tmp_path / spec.qa_path
    qa_path.write_text(qa_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(Mem2ActContractError, match="fingerprint mismatch"):
        load_mem2act_dataset(tmp_path, spec=spec, verify_git=False)


def test_strict_loader_rejects_duplicate_task_ids_even_with_a_matching_hash(
    tmp_path: Path,
) -> None:
    rows, spec = _write_loader_fixture(tmp_path)
    rows[1]["qa_id"] = "qa_001"
    qa_path = tmp_path / spec.qa_path
    _write_jsonl(qa_path, rows)
    duplicate_spec = DatasetSpec(
        repo_commit=None,
        qa_path=spec.qa_path,
        conversation_path=spec.conversation_path,
        statistics_path=spec.statistics_path,
        files_sha256={**spec.files_sha256, spec.qa_path: sha256_file(qa_path)},
        task_count=2,
        session_count=2,
    )
    with pytest.raises(Mem2ActContractError, match="duplicate Mem2ActBench task ID"):
        load_mem2act_dataset(tmp_path, spec=duplicate_spec, verify_git=False)


def test_prediction_parser_and_exact_scoring_are_type_sensitive() -> None:
    prediction = parse_tool_prediction('{"name":"toggle","arguments":{"enabled":true}}')
    exact = score_prediction(prediction, gold_tool_name="toggle", gold_arguments={"enabled": True})
    assert exact.exact_tool_and_arguments == 1
    assert exact.parameter_f1 == 1.0

    wrong_type = score_prediction(
        ToolPrediction(name="toggle", arguments={"enabled": 1}),
        gold_tool_name="toggle",
        gold_arguments={"enabled": True},
    )
    assert wrong_type.exact_tool_and_arguments == 0
    assert wrong_type.slot_accuracy == 0.0

    with pytest.raises(Mem2ActContractError, match="exactly the keys"):
        parse_tool_prediction('{"name":"toggle","arguments":{},"reason":"because"}')
    with pytest.raises(Mem2ActContractError, match="duplicate prediction key"):
        parse_tool_prediction('{"name":"a","name":"b","arguments":{}}')


async def test_three_arm_runner_fences_labels_and_emits_gate_compatible_report(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    memory = _FakeMemoryBridge()
    reader = _FakeReader()
    result = await Mem2ActEvaluator(
        dataset,
        memory,
        reader,
        config=BenchmarkConfig(bootstrap_resamples=200, bootstrap_seed=19),
    ).run()

    assert memory.ingested == tuple(session.public_view() for session in dataset.sessions)
    assert all("source_id" not in turn for session in memory.ingested for turn in session.turns)
    assert all(not hasattr(session, "original_conversation_ids") for session in memory.ingested)
    assert memory.queries == [
        (dataset.tasks[0].query, 5, 8_192),
        (dataset.tasks[1].query, 5, 8_192),
    ]  # exactly one frozen query-only recall per task
    assert len(result.records) == 12
    assert len(reader.requests) == 12
    assert set(asdict(reader.requests[0])) == {
        "condition",
        "query",
        "memory_contexts",
        "tool_catalog",
    }
    target_requests = [
        request for request in reader.requests if request.condition == "target_tool_given"
    ]
    catalog_requests = [
        request for request in reader.requests if request.condition == "full_catalog"
    ]
    assert all(len(request.tool_catalog) == 1 for request in target_requests)
    assert all(request.tool_catalog == dataset.tool_catalog for request in catalog_requests)
    for task in dataset.tasks:
        swarm_records = [
            record
            for record in result.records
            if record.qa_id == task.qa_id and record.arm == "swarm"
        ]
        assert len(swarm_records) == 2
        assert swarm_records[0].memory_contexts == swarm_records[1].memory_contexts
        assert swarm_records[0].retrieved_memory_ids == swarm_records[1].retrieved_memory_ids

    no_memory, swarm, oracle = reader.requests[:3]
    assert no_memory.memory_contexts == ()
    assert swarm.memory_contexts == ("retrieved evidence for: What is the weather there?",)
    assert "source_weather" not in "\n".join(oracle.memory_contexts)
    assert "find_weather" not in "\n".join(oracle.memory_contexts)

    report = result.report
    assert report["evaluation"]["oracle_arm"] is True
    assert report["evaluation"]["no_memory_arm"] is True
    assert report["evaluation"]["complete_400_task_protocol"] is False
    assert report["evaluation"]["primary_parameter_condition"] == "target_tool_given"
    assert report["scoring"]["official_evaluator_released"] is False
    assert "strict reimplementation" in report["scoring"]["implementation"]
    assert report["paper_references"]["table_4_hybrid_at_5_parameter_f1"]["role"].endswith(
        "not the SOTA frontier"
    )
    assert report["paper_references"]["table_3_a_mem_qwen2_5_72b_parameter_f1"]["value"] == 0.3593
    assert report["memory"]["parameter_f1"] > report["no_memory"]["parameter_f1"]
    assert report["memory"]["parameter_condition"] == "target_tool_given"
    assert report["memory"]["tool_selection_condition"] == "full_catalog"
    assert report["oracle"]["exact_tool_and_arguments"] == 1.0
    interval = report["comparison"]["memory_vs_no_memory_ci95"]
    assert set(interval) == {"metric", "delta", "lower", "upper"}
    assert report["arms"]["swarm"]["tokens"]["total"] == 32

    primary_records = [
        record for record in result.records if record.condition == "target_tool_given"
    ]
    repeated = paired_bootstrap(
        primary_records,
        arm_pairs=(("swarm", "no_memory"),),
        resamples=200,
        seed=19,
    )
    assert (
        repeated["pairs"]["swarm-minus-no_memory"]
        == report["paired_bootstrap"]["pairs"]["swarm-minus-no_memory"]
    )

    artifact = tmp_path / "mem2act-report.json"
    artifact.write_text(json.dumps(report), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "claim": "fixture",
                "frozen_at": "2026-08-09",
                "claim_scope": {
                    "claim_sha256": hashlib.sha256(b"fixture").hexdigest(),
                    "coverage_policy": "every-dimension-covered-by-required-gate",
                    "dimensions": ["memory-to-action"],
                },
                "gates": [
                    {
                        "id": "memory_to_action",
                        "title": "fixture Mem2Act gate",
                        "required": True,
                        "claim_dimensions": ["memory-to-action"],
                        "artifact": artifact.name,
                        "checks": [
                            {
                                "pointer": "/evaluation/oracle_arm",
                                "operator": "eq",
                                "expected": True,
                            },
                            {
                                "pointer": "/evaluation/no_memory_arm",
                                "operator": "eq",
                                "expected": True,
                            },
                            {
                                "pointer": "/memory/parameter_f1",
                                "operator": "gt",
                                "expected": 0.307,
                            },
                            {
                                "pointer": "/memory/exact_tool_and_arguments",
                                "operator": "exists",
                            },
                            {"pointer": "/memory/slot_accuracy", "operator": "exists"},
                            {
                                "pointer": "/comparison/memory_vs_no_memory_ci95/lower",
                                "operator": "gt",
                                "expected": 0,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    readiness = evaluate_manifest(manifest, repo_root=tmp_path)
    assert readiness.ready


class _MalformedReader:
    async def select_tool(self, request: ReaderRequest) -> ReaderResult:
        del request
        return ReaderResult(
            raw_prediction="not-json",
            model="fake-malformed",
            prompt_tokens=7,
            completion_tokens=2,
            latency_ms=0.5,
        )


async def test_parse_failures_keep_raw_prediction_usage_and_zero_score(tmp_path: Path) -> None:
    result = await Mem2ActEvaluator(
        _dataset(),
        _FakeMemoryBridge(),
        _MalformedReader(),
        config=BenchmarkConfig(task_limit=1, bootstrap_resamples=10),
    ).run()
    assert len(result.records) == 6
    assert all(record.raw_prediction == "not-json" for record in result.records)
    assert all(
        record.failure and record.failure.stage == "prediction_parse" for record in result.records
    )
    assert all(
        record.prompt_tokens == 7 and record.completion_tokens == 2 for record in result.records
    )
    assert all(record.metrics.exact_tool_and_arguments == 0 for record in result.records)

    paths = write_benchmark_outputs(result, tmp_path / "mem2act-fixture")
    assert paths.predictions.name == "mem2act-fixture-predictions.jsonl"
    assert paths.run.name == "mem2act-fixture-run.json"
    assert paths.report.name == "mem2act-fixture-report.json"
    prediction_rows = [
        json.loads(line) for line in paths.predictions.read_text(encoding="utf-8").splitlines()
    ]
    assert prediction_rows[0]["raw_prediction"] == "not-json"
    assert prediction_rows[0]["failure"]["stage"] == "prediction_parse"
    report = json.loads(paths.report.read_text(encoding="utf-8"))
    assert report["predictions_artifact"]["sha256"] == sha256_file(paths.predictions)
    assert report["predictions_artifact"]["rows"] == 6
    assert report["run_artifact"]["sha256"] == sha256_file(paths.run)
    assert report["provenance"] == {
        "compiled_offline": True,
        "raw_predictions_reparsed": True,
        "stored_metrics_recomputed": True,
        "current_tree_verified": True,
        "official_dataset_reconstructed": False,
        "canonical_reader_requests_recomputed": False,
        "implementation": report["provenance"]["implementation"],
    }
    assert report["scoring"]["official_evaluator_released"] is False
    with pytest.raises(FileExistsError):
        write_benchmark_outputs(result, tmp_path / "mem2act-fixture")


async def test_offline_compiler_recomputes_report_from_bound_raw_rows(tmp_path: Path) -> None:
    result = await Mem2ActEvaluator(
        _dataset(),
        _FakeMemoryBridge(),
        _FakeReader(),
        config=BenchmarkConfig(task_limit=1, bootstrap_resamples=20, bootstrap_seed=7),
    ).run()
    # The in-memory report is not an authority for derived metrics.
    result.report["memory"]["parameter_f1"] = 999.0
    paths = write_benchmark_outputs(result, tmp_path / "offline")
    compiled = json.loads(paths.report.read_text(encoding="utf-8"))
    assert compiled["memory"]["parameter_f1"] != 999.0
    assert compiled["provenance"]["stored_metrics_recomputed"] is True
    assert compiled["scoring"]["official_evaluator_released"] is False

    repeated_path = tmp_path / "offline-repeated-report.json"
    repeated = compile_mem2act_report(
        paths.run.relative_to(tmp_path.parent),
        repeated_path,
        artifact_root=tmp_path.parent,
        enforce_repository_local=True,
    )
    assert canonical_json(repeated) == canonical_json(compiled)
    assert repeated_path.read_bytes() == paths.report.read_bytes()


async def test_official_compilation_reconstructs_tasks_and_catalog_from_raw_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    _, spec = _write_loader_fixture(dataset_root)
    dataset = load_mem2act_dataset(dataset_root, spec=spec, verify_git=False)
    monkeypatch.setattr(mem2act_report_module, "OFFICIAL_MEM2ACT_SPEC", spec)
    monkeypatch.setattr(mem2act_report_module, "MEM2ACT_REPO_COMMIT", "unversioned-fixture")
    monkeypatch.setattr(mem2act_report_module, "KNOWN_TOOL_NAME_REPAIRS", {})
    result = await Mem2ActEvaluator(
        dataset,
        _FakeMemoryBridge(),
        _FakeReader(),
        config=BenchmarkConfig(task_limit=1, bootstrap_resamples=10),
    ).run()
    paths = write_benchmark_outputs(
        result,
        tmp_path / "official-fixture",
        dataset_dir=dataset_root,
    )
    report = json.loads(paths.report.read_text(encoding="utf-8"))
    assert report["evaluation"]["official_dataset_verified"] is True
    assert report["evaluation"]["canonical_reader_protocol_verified"] is False
    assert report["evaluation"]["complete_400_task_protocol"] is False

    with pytest.raises(Mem2ActReportError, match="requires --dataset-dir"):
        compile_mem2act_report(
            paths.run,
            tmp_path / "missing-dataset-report.json",
            artifact_root=tmp_path,
            enforce_repository_local=False,
        )

    rows = [json.loads(line) for line in paths.predictions.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["query"] = f"synthetic easy query for {row['qa_id']}"
    paths.predictions.write_text(
        "".join(f"{canonical_json(row)}\n" for row in rows),
        encoding="utf-8",
    )
    run = json.loads(paths.run.read_text(encoding="utf-8"))
    run["predictions_artifact"]["bytes"] = paths.predictions.stat().st_size
    run["predictions_artifact"]["sha256"] = sha256_file(paths.predictions)
    paths.run.write_text(
        json.dumps(run, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Mem2ActReportError, match="differs from the pinned raw dataset"):
        compile_mem2act_report(
            paths.run,
            tmp_path / "synthetic-task-report.json",
            dataset_dir=dataset_root,
            artifact_root=tmp_path,
            enforce_repository_local=False,
        )


async def test_offline_compiler_rejects_hash_and_recomputed_metric_drift(
    tmp_path: Path,
) -> None:
    result = await Mem2ActEvaluator(
        _dataset(),
        _FakeMemoryBridge(),
        _FakeReader(),
        config=BenchmarkConfig(task_limit=1, bootstrap_resamples=10),
    ).run()
    paths = write_benchmark_outputs(result, tmp_path / "tamper")
    paths.predictions.write_text(
        paths.predictions.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(Mem2ActReportError, match="byte count differs"):
        compile_mem2act_report(
            paths.run,
            tmp_path / "hash-report.json",
            artifact_root=tmp_path,
            enforce_repository_local=False,
        )

    rows = [
        json.loads(line)
        for line in paths.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["raw_prediction"] = '{"name":"find_weather","arguments":{"city":"Gdansk"}}'
    paths.predictions.write_text(
        "".join(f"{canonical_json(row)}\n" for row in rows), encoding="utf-8"
    )
    run = json.loads(paths.run.read_text(encoding="utf-8"))
    run["predictions_artifact"]["bytes"] = paths.predictions.stat().st_size
    run["predictions_artifact"]["sha256"] = sha256_file(paths.predictions)
    paths.run.write_text(
        json.dumps(run, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Mem2ActReportError, match="parsed_prediction differs"):
        compile_mem2act_report(
            paths.run,
            tmp_path / "metric-report.json",
            artifact_root=tmp_path,
            enforce_repository_local=False,
        )


async def test_offline_compiler_rejects_unbound_swarm_contexts(tmp_path: Path) -> None:
    result = await Mem2ActEvaluator(
        _dataset(),
        _FakeMemoryBridge(),
        _FakeReader(),
        config=BenchmarkConfig(task_limit=1, bootstrap_resamples=10),
    ).run()
    paths = write_benchmark_outputs(result, tmp_path / "unbound-context")
    rows = [json.loads(line) for line in paths.predictions.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["arm"] == "swarm":
            row["memory_contexts"].append("context with no corresponding retrieval trace")
    paths.predictions.write_text(
        "".join(f"{canonical_json(row)}\n" for row in rows), encoding="utf-8"
    )
    run = json.loads(paths.run.read_text(encoding="utf-8"))
    run["predictions_artifact"]["bytes"] = paths.predictions.stat().st_size
    run["predictions_artifact"]["sha256"] = sha256_file(paths.predictions)
    paths.run.write_text(
        json.dumps(run, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(Mem2ActReportError, match="one-to-one"):
        compile_mem2act_report(
            paths.run,
            tmp_path / "unbound-context-recompiled.json",
            artifact_root=tmp_path,
            enforce_repository_local=False,
        )


async def test_offline_compiler_rejects_current_tree_and_unsafe_paths(tmp_path: Path) -> None:
    result = await Mem2ActEvaluator(
        _dataset(),
        _FakeMemoryBridge(),
        _FakeReader(),
        config=BenchmarkConfig(task_limit=1, bootstrap_resamples=10),
    ).run()
    paths = write_benchmark_outputs(result, tmp_path / "strict")

    output_directory = tmp_path / "real-output"
    output_directory.mkdir()
    (tmp_path / "linked-output").symlink_to(output_directory, target_is_directory=True)
    with pytest.raises(Mem2ActReportError, match="symbolic links"):
        compile_mem2act_report(
            paths.run,
            Path("linked-output/report.json"),
            artifact_root=tmp_path,
            enforce_repository_local=False,
        )

    run = json.loads(paths.run.read_text(encoding="utf-8"))
    run["implementation"]["tree_sha256"] = "0" * 64
    paths.run.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(Mem2ActReportError, match="current tree"):
        compile_mem2act_report(
            paths.run,
            tmp_path / "tree-report.json",
            artifact_root=tmp_path,
            enforce_repository_local=False,
        )

    # CLI-style compilation accepts repository-local relative input only.
    with pytest.raises(Mem2ActReportError, match="relative path"):
        compile_mem2act_report(
            paths.run,
            Path("unused-report.json"),
            artifact_root=tmp_path,
            enforce_repository_local=True,
        )

    paths.run.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(Mem2ActReportError, match="duplicate JSON object key"):
        compile_mem2act_report(
            paths.run,
            tmp_path / "duplicate-report.json",
            artifact_root=tmp_path,
            enforce_repository_local=False,
        )


async def test_public_runtime_bridge_ingests_and_recalls_without_private_store_access() -> None:
    bridge = await build_public_in_memory_bridge(seed="pytest-mem2act")
    try:
        session = _dataset().sessions[0].public_view()
        ingested = await bridge.ingest((session,))
        assert ingested.memory_count == 1
        result = await bridge.retrieve("Warsaw", limit=1, token_budget=2_000)
        assert len(result.memories) == 1
        assert "Warsaw" in result.memories[0].content
        assert result.memories[0].memory_id
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_runtime_bridge_drains_all_configured_embedding_work_before_recall() -> None:
    runtime = build_in_memory_runtime(
        "mem2act-semantic-test-secret",
        embeddings=DeterministicEmbeddingProvider(dimensions=1024),
    )
    await runtime.start()
    bridge = RuntimeMemoryBridge(
        runtime,
        benchmark_actor("pytest-mem2act-semantic"),
        owns_runtime=True,
        drain_embeddings=True,
        embedding_revision="deterministic-test-revision",
        embedding_protocol={"name": "fixture-semantic-v1"},
    )
    try:
        session = _dataset().sessions[0].public_view()
        ingested = await bridge.ingest((session,))
        assert ingested.metadata["embedding_work_completed"] == 1
        assert ingested.metadata["embedding_revision"] == "deterministic-test-revision"
        result = await bridge.retrieve("Warsaw", limit=1, token_budget=2_000)
        assert result.memories
        evidence = bridge.evidence_metadata()
        assert evidence["embedding_work_completed"] == 1
        assert evidence["embedding_dimensions"] == 1024
    finally:
        await bridge.close()


@pytest.mark.asyncio
async def test_semantic_bridge_factory_fails_closed_without_endpoint_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MEM2ACT_EMBEDDINGS_BASE_URL", raising=False)
    monkeypatch.delenv("MEM2ACT_EMBEDDINGS_REVISION", raising=False)

    with pytest.raises(Mem2ActContractError, match="MEM2ACT_EMBEDDINGS_BASE_URL"):
        await build_openai_semantic_in_memory_bridge()


def _write_loader_fixture(root: Path) -> tuple[list[dict[str, Any]], DatasetSpec]:
    dataset = _dataset()
    rows: list[dict[str, Any]] = []
    for task in dataset.tasks:
        rows.append(
            {
                "qa_id": task.qa_id,
                "source_conversation_ids": list(task.source_conversation_ids),
                "evolution_chain": [
                    {
                        "attribute": memory.attribute,
                        "source_id": task.source_conversation_ids[index],
                        "fact": memory.fact,
                        "source_text": memory.source_text,
                    }
                    for index, memory in enumerate(task.oracle_memories)
                ],
                "query": task.query,
                "tool_call": {
                    "name": task.gold_tool_name,
                    "arguments": task.gold_arguments,
                    "grounding_info": {},
                },
                "target_tool_schema": task.target_tool_schema,
                "complexity_metadata": {"level": task.complexity_level},
            }
        )
    session_rows = [
        {
            "session_id": session.session_id,
            "original_conversation_ids": list(session.original_conversation_ids),
            "turns": list(session.turns),
            "turn_count": session.turn_count,
            "has_tool_calls": False,
            "token_count": session.token_count,
        }
        for session in dataset.sessions
    ]
    qa_relative = "fixture/qa.jsonl"
    sessions_relative = "fixture/sessions.jsonl"
    stats_relative = "fixture/statistics.json"
    qa_path = root / qa_relative
    sessions_path = root / sessions_relative
    stats_path = root / stats_relative
    qa_path.parent.mkdir(parents=True)
    _write_jsonl(qa_path, rows)
    _write_jsonl(sessions_path, session_rows)
    stats_path.write_text(json.dumps({"total_sessions": 2, "total_qa": 2}), encoding="utf-8")
    spec = DatasetSpec(
        repo_commit=None,
        qa_path=qa_relative,
        conversation_path=sessions_relative,
        statistics_path=stats_relative,
        files_sha256={
            qa_relative: sha256_file(qa_path),
            sessions_relative: sha256_file(sessions_path),
            stats_relative: sha256_file(stats_path),
        },
        task_count=2,
        session_count=2,
    )
    return rows, spec


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(f"{canonical_json(row)}\n" for row in rows), encoding="utf-8")
