from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from benchmarks.integrations.longmemeval_v2.adapter import (
    BridgeLifecycle,
    SwarmQueryAdapter,
    TraceJournal,
    bind_official_prompt_row,
    build_official_memory_class,
)
from benchmarks.integrations.longmemeval_v2.contracts import (
    PINNED_REPOSITORY_COMMIT,
    SOTA_EMBEDDING_MODEL,
    SOTA_EMBEDDING_PROVIDER,
    SOTA_EMBEDDING_QUERY_INSTRUCTION,
    SOTA_EMBEDDING_QUERY_INSTRUCTION_SHA256,
    AdapterConfig,
    BoundQueryEvidence,
    EmbeddingRuntimeEvidence,
    LongMemEvalV2AdapterError,
    ReadExpandMemoryResult,
    RecallMemoryResult,
)
from benchmarks.integrations.longmemeval_v2.evidence import (
    EvidenceLedger,
    build_operation_sidecar,
    canonical_sha256,
    expected_invocation_id,
    ledger_payload,
    write_ledger,
    write_operation_sidecar,
)
from benchmarks.integrations.longmemeval_v2.runner import (
    dataset_manifest,
    dry_run,
    preflight_official_environment,
)
from benchmarks.integrations.longmemeval_v2.runtime_bridge import (
    LOCAL_RUNTIME_BRIDGE_FACTORY,
    LocalRuntimeBridge,
    build_local_runtime_bridge,
)
from build_longmemeval_v2_report import (
    _expected_invocation_id,
    _validate_operation_trace,
)


def _config(**changes: Any) -> AdapterConfig:
    values: dict[str, Any] = {
        "tier": "small",
        "operating_point": "balanced",
        "dataset_revision": "dataset-revision-fixture",
        "dataset_manifest_sha256": "1" * 64,
        "bridge_factory": "fixture:factory",
        "bridge_params": {},
    }
    values.update(changes)
    return AdapterConfig(**values)


def _sota_embedding(*, inserted_memories: int = 1) -> EmbeddingRuntimeEvidence:
    return EmbeddingRuntimeEvidence(
        retrieval_mode="openai_hybrid",
        sota_capable=True,
        provider=SOTA_EMBEDDING_PROVIDER,
        model=SOTA_EMBEDDING_MODEL,
        model_revision="fixture-public-revision",
        dimensions=4_096,
        response_model_requirement=SOTA_EMBEDDING_MODEL,
        query_instruction_sha256=SOTA_EMBEDDING_QUERY_INSTRUCTION_SHA256,
        inserted_memories=inserted_memories,
        embedding_work_completed=inserted_memories,
        call_accounting_source="provider-observed",
        document_inputs=inserted_memories,
        document_batch_calls=inserted_memories,
        document_successful_http_calls=inserted_memories,
        document_http_attempts=inserted_memories,
        query_calls=1,
        query_successful_http_calls=1,
        query_http_attempts=1,
        exact_response_model_verified=True,
        deterministic_fallback_used=False,
    )


class RecordingBridge:
    def __init__(self, *, empty_recall: bool = False) -> None:
        self.empty_recall = empty_recall
        self.calls: list[tuple[str, Any]] = []
        self.close_count = 0

    def insert_trajectory(self, trajectory: Any) -> None:
        self.calls.append(("insert_trajectory", dict(trajectory)))

    def recall_memory(self, query: str, *, limit: int) -> RecallMemoryResult:
        self.calls.append(("recall_memory", {"query": query, "limit": limit}))
        return RecallMemoryResult(() if self.empty_recall else ("raw-seed", "raw-second"))

    def read_expand_memory(
        self,
        query: str,
        *,
        memory_ids: tuple[str, ...],
        max_depth: int,
        max_fanout: int,
        token_budget: int,
    ) -> ReadExpandMemoryResult:
        self.calls.append(
            (
                "read_expand_memory",
                {
                    "query": query,
                    "memory_ids": memory_ids,
                    "max_depth": max_depth,
                    "max_fanout": max_fanout,
                    "token_budget": token_budget,
                },
            )
        )
        return ReadExpandMemoryResult(
            ("raw-seed", "raw-neighbor"),
            "Exact context from the canonical read-expand operation.",
        )

    def close(self) -> None:
        self.close_count += 1

    def embedding_evidence(self) -> EmbeddingRuntimeEvidence:
        return _sota_embedding()


def _bound_query(
    *, bridge: RecordingBridge | None = None
) -> tuple[dict[str, Any], BoundQueryEvidence, RecordingBridge]:
    bridge = bridge or RecordingBridge()
    journal = TraceJournal()
    ledger = EvidenceLedger()
    adapter = SwarmQueryAdapter(bridge, journal, _config())
    adapter.insert({"id": "trajectory-public"})
    memory_context = adapter.query(
        "Where is the deployment-specific control?",
        opaque_invocation_id="opaque-official-handle",
    )
    row = {
        "question_id": "stable-question-id",
        "query_invocation_id": "opaque-official-handle",
        "memory_context": memory_context,
        "memory_query_duration_seconds": 1.0,
        "memory_context_original_token_count": 17,
        "memory_context_token_count": 17,
        "memory_context_was_truncated": False,
        "memory_post_query_metadata": adapter.post_query_metadata("opaque-official-handle"),
    }
    bound = bind_official_prompt_row(
        row,
        journal=journal,
        ledger=ledger,
        tier="small",
        operating_point="balanced",
        domain="web",
    )
    return bound, ledger.snapshot()[0], bridge


def test_adapter_routes_query_only_through_recall_then_exact_read_expand() -> None:
    bound, evidence, bridge = _bound_query()

    assert [name for name, _ in bridge.calls] == [
        "insert_trajectory",
        "recall_memory",
        "read_expand_memory",
    ]
    assert bridge.calls[1][1] == {
        "query": "Where is the deployment-specific control?",
        "limit": 8,
    }
    assert bridge.calls[2][1]["memory_ids"] == ("raw-seed", "raw-second")
    assert bridge.calls[2][1]["max_depth"] == 1
    assert bridge.calls[2][1]["token_budget"] == 16_384
    assert "stable-question-id" not in json.dumps(bridge.calls, sort_keys=True)
    assert "opaque-official-handle" not in json.dumps(evidence.operations, sort_keys=True)
    assert "raw-seed" not in json.dumps(evidence.operations, sort_keys=True)

    operations, trace_sha256 = _validate_operation_trace(
        list(evidence.operations),
        tier="small",
        point_name="balanced",
        question_id=evidence.question_id,
        query_tokens=evidence.query_tokens,
        query_latency_ms=evidence.query_latency_ms,
        official_record=bound,
        seen_invocation_ids=set(),
        label="fixture",
        embedding=evidence.embedding,
    )
    assert len(operations) == 2
    assert trace_sha256 == evidence.trace_sha256
    assert evidence.operations[0]["invocation_id"] == _expected_invocation_id(
        tier="small",
        point_name="balanced",
        question_id="stable-question-id",
        sequence=0,
        operation="recall_memory",
    )


def test_adapter_fails_closed_without_a_search_seed_or_on_revision_drift() -> None:
    adapter = SwarmQueryAdapter(RecordingBridge(empty_recall=True), TraceJournal(), _config())
    with pytest.raises(LongMemEvalV2AdapterError, match="returned no seed"):
        adapter.query("public query", opaque_invocation_id="opaque")

    params = _config().memory_params()
    params["benchmark_repository_commit"] = "0" * 40
    with pytest.raises(LongMemEvalV2AdapterError, match="benchmark_repository_commit"):
        AdapterConfig.from_memory_params(params)

    params = _config().memory_params()
    params["reader_model"] = "Qwen/Qwen3.5-9B-quantized"
    with pytest.raises(LongMemEvalV2AdapterError, match="reader_model"):
        AdapterConfig.from_memory_params(params)


def test_adapter_rejects_inline_bridge_secrets_but_allows_env_indirection() -> None:
    config = _config(
        bridge_params={
            "base_url": "https://swarmbrain.internal/v1",
            "api_key_env": "SWARMBRAIN_API_KEY",
            "transport": {
                "client_secret_env_var": "SWARMBRAIN_CLIENT_SECRET",
                "retry_delays": [0.1, 0.5],
            },
        }
    )
    assert config.bridge_params["api_key_env"] == "SWARMBRAIN_API_KEY"

    inline_secret = "do-not-serialize-this-value"
    with pytest.raises(
        LongMemEvalV2AdapterError, match=r"bridge_params\.transport\.api_key"
    ) as exc:
        _config(bridge_params={"transport": {"api_key": inline_secret}})
    assert inline_secret not in str(exc.value)

    with pytest.raises(LongMemEvalV2AdapterError, match="xApiKey"):
        _config(bridge_params={"headers": {"xApiKey": inline_secret}})

    with pytest.raises(LongMemEvalV2AdapterError, match="appears to contain inline secret"):
        _config(bridge_params={"headers": {"X-Custom": "Bearer opaque-token"}})

    with pytest.raises(LongMemEvalV2AdapterError, match="must name an environment variable"):
        _config(bridge_params={"api_key_env": "not an env name"})

    with pytest.raises(LongMemEvalV2AdapterError, match="appears to contain inline secret"):
        _config(bridge_params={"base_url": "https://host/v1?access_token=inline-secret"})


def test_trace_binder_rejects_tampered_post_query_metadata() -> None:
    bridge = RecordingBridge()
    journal = TraceJournal()
    adapter = SwarmQueryAdapter(bridge, journal, _config())
    context = adapter.query("public query", opaque_invocation_id="opaque")
    row = {
        "question_id": "qid",
        "query_invocation_id": "opaque",
        "memory_context": context,
        "memory_query_duration_seconds": 1.0,
        "memory_context_token_count": 4,
        "memory_post_query_metadata": {
            "swarmbrain_raw_operation_trace_sha256": "0" * 64,
        },
    }
    with pytest.raises(LongMemEvalV2AdapterError, match="raw trace digest"):
        bind_official_prompt_row(
            row,
            journal=journal,
            ledger=EvidenceLedger(),
            tier="small",
            operating_point="balanced",
            domain="web",
        )


def test_dynamic_official_backend_uses_only_opaque_query_context() -> None:
    registry: dict[str, type[Any]] = {}

    class DummyOfficialMemory:
        def __init__(self, memory_params: dict[str, object]) -> None:
            self.memory_params = memory_params
            self._context: dict[str, str] = {}

        def get_query_context(self) -> dict[str, str]:
            return dict(self._context)

    bridge = RecordingBridge()

    def register(memory_class: type[Any]) -> type[Any]:
        registry[memory_class.memory_type] = memory_class
        return memory_class

    memory_class = build_official_memory_class(
        memory_base=DummyOfficialMemory,
        register_memory=register,
        bridge_factory=lambda _: bridge,
        journal=TraceJournal(),
    )
    memory = memory_class(_config().memory_params())
    memory._context = {"query_invocation_id": "opaque-only"}
    memory.insert({"id": "trajectory"})
    result = memory.query("public text", query_image="/public/question.png")
    metadata = memory.post_query_hook(
        query="public text",
        query_image="/public/question.png",
        memory_context=result,
    )

    assert registry["swarmbrain"] is memory_class
    assert result[0]["value"].startswith("Exact context")
    assert set(metadata) == {"swarmbrain_raw_operation_trace_sha256"}
    assert "question_id" not in json.dumps(bridge.calls)


def test_dynamic_backend_closes_per_question_and_shared_bridges_deterministically() -> None:
    class DummyOfficialMemory:
        def __init__(self, memory_params: dict[str, object]) -> None:
            self.memory_params = memory_params
            self._context = {"query_invocation_id": "opaque"}

        def get_query_context(self) -> dict[str, str]:
            return dict(self._context)

    lifecycle = BridgeLifecycle()
    per_question = RecordingBridge()
    memory_class = build_official_memory_class(
        memory_base=DummyOfficialMemory,
        register_memory=lambda value: value,
        bridge_factory=lambda _: per_question,
        journal=TraceJournal(),
        lifecycle=lifecycle,
        close_after_query=True,
    )
    memory = memory_class(_config().memory_params())
    context = memory.query("public query")
    memory.post_query_hook(query="public query", query_image=None, memory_context=context)
    assert per_question.close_count == 1
    assert lifecycle.active() == 0
    memory.close()
    assert per_question.close_count == 1

    shared_lifecycle = BridgeLifecycle()
    shared = RecordingBridge()
    shared_class = build_official_memory_class(
        memory_base=DummyOfficialMemory,
        register_memory=lambda value: value,
        bridge_factory=lambda _: shared,
        journal=TraceJournal(),
        lifecycle=shared_lifecycle,
        close_after_query=False,
    )
    shared_memory = shared_class(_config().memory_params())
    shared_context = shared_memory.query("public query")
    shared_memory.post_query_hook(
        query="public query",
        query_image=None,
        memory_context=shared_context,
    )
    assert shared.close_count == 0
    assert shared_lifecycle.active() == 1
    shared_lifecycle.close_all()
    assert shared.close_count == 1


def _public_trajectory(
    *,
    trajectory_id: str = "trajectory-public-1",
    answer_text: str = "The deployment-specific control is Export Orders.",
) -> dict[str, Any]:
    return {
        "id": trajectory_id,
        "domain": "web",
        "goal": "Find the deployment-specific control in the orders interface",
        "start_url": "https://shop.example.test/admin/orders",
        "outcome": "The control was found and opened successfully.",
        "states": [
            {
                "url": "https://shop.example.test/admin/orders",
                "action": None,
                "thought": "Inspect the available controls.",
                "accessibility_tree": "Orders page with Filters and Search controls.",
                "screenshot": "screenshots/trajectory-public-1/0000.png",
            },
            {
                "url": "https://shop.example.test/admin/orders",
                "action": "click Export Orders",
                "thought": "The export menu is now visible.",
                "accessibility_tree": answer_text,
                "screenshot": "screenshots/trajectory-public-1/0001.png",
            },
        ],
    }


def _local_config(**bridge_changes: Any) -> AdapterConfig:
    bridge_params: dict[str, Any] = {
        "backend": "in_memory",
        "retrieval_mode": "deterministic_hybrid",
        "embedding_dimensions": 64,
        "chunk_chars": 1_024,
    }
    if (
        bridge_changes.get("retrieval_mode") == "lexical"
        and "embedding_dimensions" not in bridge_changes
    ):
        bridge_params.pop("embedding_dimensions")
    bridge_params.update(bridge_changes)
    return _config(
        bridge_factory=LOCAL_RUNTIME_BRIDGE_FACTORY,
        bridge_params=bridge_params,
    )


def _openai_config(**bridge_changes: Any) -> AdapterConfig:
    params: dict[str, Any] = {
        "backend": "in_memory",
        "retrieval_mode": "openai_hybrid",
        "chunk_chars": 1_024,
        "embedding_base_url": "https://embeddings.example.test/v1",
        "embedding_api_key_env": "LME_V2_TEST_EMBEDDING_KEY",
        "embedding_model_revision": "fixture-public-revision",
    }
    params.update(bridge_changes)
    return _config(
        bridge_factory=LOCAL_RUNTIME_BRIDGE_FACTORY,
        bridge_params=params,
    )


class _FakeEmbeddingResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeEmbeddingClient:
    def __init__(self, *, wrong_query_model: bool = False) -> None:
        self.wrong_query_model = wrong_query_model
        self.requests: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> _FakeEmbeddingResponse:
        request = {"url": url, **kwargs}
        self.requests.append(request)
        payload = kwargs["json"]
        inputs = payload["input"]
        is_query = bool(inputs and inputs[0].startswith("Instruct: "))
        response_model = (
            "wrong/revision" if is_query and self.wrong_query_model else payload["model"]
        )
        vector = [1.0, *([0.0] * 4_095)]
        return _FakeEmbeddingResponse(
            {
                "model": response_model,
                "data": [{"index": index, "embedding": vector} for index, _ in enumerate(inputs)],
            }
        )


def test_bundled_local_bridge_runs_canonical_sync_runtime_and_closes() -> None:
    bridge = build_local_runtime_bridge(_local_config())
    assert isinstance(bridge, LocalRuntimeBridge)
    bridge.insert_trajectory(_public_trajectory())

    recalled = bridge.recall_memory("Where is the Export Orders control?", limit=4)
    assert recalled.memory_ids
    expanded = bridge.read_expand_memory(
        "Where is the Export Orders control?",
        memory_ids=recalled.memory_ids,
        max_depth=1,
        max_fanout=4,
        token_budget=4_096,
    )

    assert expanded.memory_ids
    assert "Export Orders" in expanded.context
    assert "visibility=task" in expanded.context
    assert bridge.runtime_backend == "memory"
    assert set(bridge.scope_ids) == {
        "tenant_id",
        "project_id",
        "repository_id",
        "swarm_id",
        "run_id",
        "agent_id",
        "task_id",
    }
    assert all(str(UUID(value)) == value for value in bridge.scope_ids.values())
    development_evidence = bridge.embedding_evidence()
    assert development_evidence.sota_capable is False
    assert development_evidence.retrieval_mode == "deterministic_hybrid"

    bridge.close()
    assert bridge.closed is True
    bridge.close()
    with pytest.raises(LongMemEvalV2AdapterError, match="bridge is closed"):
        bridge.recall_memory("Export Orders", limit=4)


def test_openai_hybrid_reconciles_provider_calls_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LME_V2_TEST_EMBEDDING_KEY", "fixture-key-not-persisted")
    client = _FakeEmbeddingClient()
    bridge = build_local_runtime_bridge(_openai_config(), openai_client=client)
    try:
        bridge.insert_trajectory(_public_trajectory())
        recalled = bridge.recall_memory("Where is the Export Orders control?", limit=4)
        expanded = bridge.read_expand_memory(
            "Where is the Export Orders control?",
            memory_ids=recalled.memory_ids,
            max_depth=1,
            max_fanout=4,
            token_budget=4_096,
        )
        evidence = bridge.embedding_evidence()
    finally:
        bridge.close()

    assert expanded.memory_ids
    assert evidence.sota_capable is True
    assert evidence.model == SOTA_EMBEDDING_MODEL
    assert evidence.dimensions == 4_096
    assert evidence.embedding_work_completed == evidence.inserted_memories
    assert evidence.document_inputs == evidence.inserted_memories
    assert evidence.document_successful_http_calls == evidence.inserted_memories
    assert evidence.query_calls == evidence.query_successful_http_calls == 1
    assert evidence.exact_response_model_verified is True
    assert evidence.deterministic_fallback_used is False
    query_input = client.requests[-1]["json"]["input"][0]
    assert query_input.startswith(f"Instruct: {SOTA_EMBEDDING_QUERY_INSTRUCTION}\nQuery: ")
    serialized = json.dumps(evidence.as_json(), sort_keys=True)
    assert "fixture-key-not-persisted" not in serialized
    assert "embeddings.example.test" not in serialized


def test_openai_hybrid_rejects_lexical_fallback_and_unsafe_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LME_V2_TEST_EMBEDDING_KEY", "fixture-key")
    bridge = build_local_runtime_bridge(
        _openai_config(), openai_client=_FakeEmbeddingClient(wrong_query_model=True)
    )
    try:
        bridge.insert_trajectory(_public_trajectory())
        with pytest.raises(LongMemEvalV2AdapterError, match="fallback is forbidden"):
            bridge.recall_memory("Where is Export Orders?", limit=4)
    finally:
        bridge.close()

    with pytest.raises(LongMemEvalV2AdapterError, match="without credentials, query, or fragment"):
        build_local_runtime_bridge(_openai_config(embedding_base_url="https://example.test/v1?x=1"))
    with pytest.raises(LongMemEvalV2AdapterError, match="must name an environment variable"):
        build_local_runtime_bridge(_openai_config(embedding_api_key_env="not-valid"))


def test_bundled_local_bridge_scope_is_deterministic_and_instances_are_isolated() -> None:
    first = build_local_runtime_bridge(_local_config(retrieval_mode="lexical"))
    second = build_local_runtime_bridge(_local_config(retrieval_mode="lexical"))
    changed = build_local_runtime_bridge(_local_config(retrieval_mode="lexical"))
    try:
        first.insert_trajectory(_public_trajectory())
        second.insert_trajectory(_public_trajectory())
        changed.insert_trajectory(
            _public_trajectory(
                trajectory_id="trajectory-public-2",
                answer_text="A different isolated trajectory fact.",
            )
        )
        assert first.recall_memory("Export Orders", limit=4).memory_ids
        assert second.recall_memory("Export Orders", limit=4).memory_ids
        assert changed.recall_memory("isolated trajectory", limit=4).memory_ids
        assert first.scope_ids == second.scope_ids
        assert first.scope_ids != changed.scope_ids
    finally:
        first.close()
        second.close()
        changed.close()

    lifecycle = BridgeLifecycle()
    official_first = build_local_runtime_bridge(_local_config(retrieval_mode="lexical"))
    official_second = build_local_runtime_bridge(_local_config(retrieval_mode="lexical"))
    lifecycle.register(official_first)
    lifecycle.register(official_second)
    try:
        official_first.insert_trajectory(_public_trajectory())
        official_second.insert_trajectory(_public_trajectory())
        assert official_first.scope_ids != official_second.scope_ids
    finally:
        lifecycle.close_all()


def test_bundled_local_bridge_fails_closed_on_backend_or_config_drift() -> None:
    with pytest.raises(LongMemEvalV2AdapterError, match="backend must be exactly"):
        build_local_runtime_bridge(_local_config(backend="http"))
    with pytest.raises(LongMemEvalV2AdapterError, match="unsupported field"):
        build_local_runtime_bridge(_local_config(api_key_env="SWARMBRAIN_API_KEY"))
    with pytest.raises(LongMemEvalV2AdapterError, match="embedding settings cannot"):
        build_local_runtime_bridge(_local_config(retrieval_mode="lexical", embedding_dimensions=64))

    bridge = build_local_runtime_bridge(_local_config(retrieval_mode="lexical"))
    try:
        invalid = _public_trajectory()
        invalid["states"] = []
        with pytest.raises(LongMemEvalV2AdapterError, match="non-empty list"):
            bridge.insert_trajectory(invalid)
    finally:
        bridge.close()


def test_dry_run_is_explicitly_non_claiming_and_makes_no_model_calls() -> None:
    result = dry_run()
    assert result["model_api_calls"] == 0
    assert result["question_id_visible_to_bridge"] is False
    assert result["trace_bound_in_official_metadata"] is True
    assert "not leaderboard or SOTA evidence" in result["claim_status"]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_dataset_manifest_pins_all_451_question_and_haystack_ids(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    questions = [
        {
            "id": f"q{index:03d}",
            "domain": "web" if index < 226 else "enterprise",
            "question": f"question {index}",
        }
        for index in range(451)
    ]
    trajectories = [
        {"id": "web-trajectory", "domain": "web", "states": []},
        {"id": "enterprise-trajectory", "domain": "enterprise", "states": []},
    ]
    haystack = {
        row["id"]: ["web-trajectory" if row["domain"] == "web" else "enterprise-trajectory"]
        for row in questions
    }
    _write_jsonl(data_root / "questions.jsonl", questions)
    _write_jsonl(data_root / "trajectories.jsonl", trajectories)
    haystack_path = data_root / "haystacks/lme_v2_small.json"
    haystack_path.parent.mkdir(parents=True)
    haystack_path.write_text(json.dumps(haystack), encoding="utf-8")

    first = dataset_manifest(data_root, tier="small", dataset_revision="revision-a")
    second = dataset_manifest(data_root, tier="small", dataset_revision="revision-a")
    assert first == second
    assert first["questions"] == 451
    assert len(first["manifest_sha256"]) == 64

    questions[0]["question"] = "changed"
    _write_jsonl(data_root / "questions.jsonl", questions)
    changed = dataset_manifest(data_root, tier="small", dataset_revision="revision-a")
    assert changed["manifest_sha256"] != first["manifest_sha256"]


def _operation(question_id: str, *, point: str) -> tuple[dict[str, Any], ...]:
    memory_id = "mem_" + canonical_sha256({"question_id": question_id})
    return (
        {
            "sequence": 0,
            "invocation_id": expected_invocation_id(
                tier="small",
                operating_point=point,
                question_id=question_id,
                sequence=0,
                operation="recall_memory",
            ),
            "operation": "recall_memory",
            "success": True,
            "depth": 0,
            "seed_memory_ids": [],
            "result_memory_ids": [memory_id],
            "delivered_tokens": 0,
            "latency_ms": 1.0,
        },
        {
            "sequence": 1,
            "invocation_id": expected_invocation_id(
                tier="small",
                operating_point=point,
                question_id=question_id,
                sequence=1,
                operation="read_expand_memory",
            ),
            "operation": "read_expand_memory",
            "success": True,
            "depth": 1,
            "seed_memory_ids": [memory_id],
            "result_memory_ids": [memory_id],
            "delivered_tokens": 1,
            "latency_ms": 1.0,
        },
    )


def test_sidecar_builder_reconciles_complete_package_ledgers_and_strict_write_gate(
    tmp_path: Path,
) -> None:
    package = tmp_path / "fixture-small"
    point = "balanced"
    overview = {
        "submission_name": package.name,
        "method": "swarmbrain",
        "tier": "small",
        "operating_points": [{"name": point}],
    }
    package.mkdir()
    (package / "submission_overview.json").write_text(json.dumps(overview), encoding="utf-8")
    ledgers: list[Path] = []
    config = _config(operating_point=point)
    for domain, indices in (
        ("web", range(226)),
        ("enterprise", range(226, 451)),
    ):
        run_dir = package / "operating_points" / point / domain
        runtime_dir = run_dir / "runtime_inputs"
        runtime_dir.mkdir(parents=True)
        run_args = {
            "domain": domain,
            "model": "Qwen/Qwen3.5-9B",
            "evaluator_model": "gpt-5.2",
            "output_dir": f"/{domain}",
        }
        memory_config = {
            "memory_type": "swarmbrain",
            "memory_params": config.memory_params(),
        }
        (run_dir / "run_args.json").write_text(json.dumps(run_args), encoding="utf-8")
        (runtime_dir / "memory_config.json").write_text(json.dumps(memory_config), encoding="utf-8")
        records: list[dict[str, Any]] = []
        ledger = EvidenceLedger()
        for index in indices:
            question_id = f"q{index:03d}"
            operations = _operation(question_id, point=point)
            embedding = _sota_embedding()
            trace_sha256 = canonical_sha256(
                {"operations": operations, "embedding": embedding.as_json()}
            )
            ledger.record(
                BoundQueryEvidence(
                    question_id=question_id,
                    domain=domain,
                    query_tokens=1,
                    query_latency_ms=10.0,
                    operations=operations,
                    embedding=embedding,
                    trace_sha256=trace_sha256,
                )
            )
            records.append(
                {
                    "question_id": question_id,
                    "memory_context_token_count": 1,
                    "memory_query_duration_seconds": 0.01,
                    "memory_post_query_metadata": {
                        "swarmbrain_operation_trace_sha256": trace_sha256
                    },
                    "response_raw": "answer",
                }
            )
        _write_jsonl(run_dir / "per_question.jsonl", records)
        ledger_path = tmp_path / f"{domain}.ledger.json"
        write_ledger(
            ledger_path,
            ledger_payload(
                ledger,
                tier="small",
                operating_point=point,
                domain=domain,
                dataset_revision=config.dataset_revision,
                dataset_manifest_sha256=config.dataset_manifest_sha256,
                run_args=run_args,
                memory_config=memory_config,
            ),
        )
        ledgers.append(ledger_path)

    sidecar = build_operation_sidecar(package, ledgers)
    assert sidecar["schema_version"] == 3
    assert sidecar["benchmark_repository_commit"] == PINNED_REPOSITORY_COMMIT
    assert sidecar["operating_points"][0]["summary"]["questions"] == 451
    assert sidecar["operating_points"][0]["summary"]["query_tokens_total"] == 451

    output = tmp_path / "strict-sidecar.json"
    with pytest.raises(LongMemEvalV2AdapterError, match="strict compiler rejected"):
        write_operation_sidecar(package, ledgers, output)
    assert not output.exists()

    web_run_dir = package / "operating_points" / point / "web"
    run_args_path = web_run_dir / "run_args.json"
    original_run_args = json.loads(run_args_path.read_text(encoding="utf-8"))
    changed_run_args = {**original_run_args, "temperature": 0.7}
    run_args_path.write_text(json.dumps(changed_run_args), encoding="utf-8")
    with pytest.raises(LongMemEvalV2AdapterError, match="ledger protocol digest differs"):
        build_operation_sidecar(package, ledgers)
    run_args_path.write_text(json.dumps(original_run_args), encoding="utf-8")

    memory_config_path = web_run_dir / "runtime_inputs/memory_config.json"
    original_memory_config = json.loads(memory_config_path.read_text(encoding="utf-8"))
    changed_memory_config = json.loads(json.dumps(original_memory_config))
    changed_memory_config["memory_params"]["bridge_factory"] = "tampered:factory"
    memory_config_path.write_text(json.dumps(changed_memory_config), encoding="utf-8")
    with pytest.raises(LongMemEvalV2AdapterError, match="ledger memory config digest differs"):
        build_operation_sidecar(package, ledgers)

    changed_memory_config["memory_params"]["adapter_revision"] = "tampered-revision"
    memory_config_path.write_text(json.dumps(changed_memory_config), encoding="utf-8")
    with pytest.raises(LongMemEvalV2AdapterError, match="adapter_revision"):
        build_operation_sidecar(package, ledgers)


def test_pinned_clone_preflight_reports_exact_missing_data_blocker() -> None:
    repository = Path("/private/tmp/swarmbrain-lme-v2")
    if not repository.is_dir():
        pytest.skip("pinned LongMemEval-V2 checkout is not available")
    result = preflight_official_environment(
        repository,
        repository / "data/longmemeval-v2",
        tier="small",
        dataset_revision="not-downloaded",
        expected_dataset_manifest_sha256=None,
    )
    assert result.repository_commit == PINNED_REPOSITORY_COMMIT
    assert result.repository_clean is True
    assert result.ready is False
    assert {Path(path).name for path in result.missing_paths} == {
        "questions.jsonl",
        "trajectories.jsonl",
        "lme_v2_small.json",
    }
