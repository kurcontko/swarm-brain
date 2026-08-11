"""Official LongMemEval-V2 memory backend over Swarm search/read-expand."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from time import perf_counter
from typing import Any

from .contracts import (
    ADAPTER_REVISION,
    MEMORY_TYPE,
    RAW_TRACE_DIGEST_METADATA_KEY,
    TRACE_DIGEST_METADATA_KEY,
    AdapterConfig,
    EmbeddingRuntimeEvidence,
    LongMemEvalV2AdapterError,
    RawOperation,
    RawQueryTrace,
    ReadExpandMemoryResult,
    RecallMemoryResult,
    SwarmOperationBridge,
)
from .evidence import EvidenceLedger, bind_query_trace, raw_trace_sha256

BridgeFactory = Callable[[AdapterConfig], SwarmOperationBridge]


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongMemEvalV2AdapterError(f"{label} must be a non-empty string")
    return value.strip()


class TraceJournal:
    """Keep run-local operation IDs away from persisted official artifacts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._traces: dict[str, RawQueryTrace] = {}

    def record(self, trace: RawQueryTrace) -> None:
        with self._lock:
            if trace.opaque_invocation_id in self._traces:
                raise LongMemEvalV2AdapterError(
                    "the official harness reused an opaque query invocation ID"
                )
            self._traces[trace.opaque_invocation_id] = trace

    def get(self, opaque_invocation_id: str) -> RawQueryTrace:
        with self._lock:
            trace = self._traces.get(opaque_invocation_id)
        if trace is None:
            raise LongMemEvalV2AdapterError("no operation trace exists for this query invocation")
        return trace

    def consume(self, opaque_invocation_id: str) -> RawQueryTrace:
        with self._lock:
            trace = self._traces.pop(opaque_invocation_id, None)
        if trace is None:
            raise LongMemEvalV2AdapterError(
                "operation trace was missing or already bound to an official row"
            )
        return trace

    def pending(self) -> int:
        with self._lock:
            return len(self._traces)


class BridgeLifecycle:
    """Own and deterministically close bridges the official harness creates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bridges: dict[int, SwarmOperationBridge] = {}
        self._next_instance_ordinal = 0

    def register(self, bridge: SwarmOperationBridge) -> None:
        with self._lock:
            key = id(bridge)
            if key in self._bridges:
                raise LongMemEvalV2AdapterError("bridge factory returned a reused bridge instance")
            ordinal = self._next_instance_ordinal
            self._next_instance_ordinal += 1
            self._bridges[key] = bridge
        binder = getattr(bridge, "bind_official_instance_ordinal", None)
        if not callable(binder):
            return
        try:
            result = binder(ordinal)
            if inspect.isawaitable(result):
                result.close() if inspect.iscoroutine(result) else None
                raise LongMemEvalV2AdapterError("Swarm bridge instance binding must be synchronous")
        except BaseException as exc:
            with self._lock:
                self._bridges.pop(key, None)
            with suppress(BaseException):
                self._close_one(bridge)
            raise LongMemEvalV2AdapterError(
                "Swarm bridge rejected deterministic instance binding"
            ) from exc

    @staticmethod
    def _close_one(bridge: SwarmOperationBridge) -> None:
        result = bridge.close()
        if inspect.isawaitable(result):
            result.close() if inspect.iscoroutine(result) else None
            raise LongMemEvalV2AdapterError("Swarm bridge close must be synchronous")

    def close(self, bridge: SwarmOperationBridge) -> None:
        with self._lock:
            registered = self._bridges.pop(id(bridge), None)
        if registered is None:
            return
        self._close_one(registered)

    def close_all(self) -> None:
        with self._lock:
            bridges = tuple(reversed(self._bridges.values()))
            self._bridges.clear()
        failed_types: list[str] = []
        first_error: BaseException | None = None
        for bridge in bridges:
            try:
                self._close_one(bridge)
            except BaseException as exc:
                failed_types.append(type(bridge).__name__)
                first_error = first_error or exc
        if first_error is not None:
            failed = ", ".join(failed_types)
            raise LongMemEvalV2AdapterError(
                f"one or more Swarm bridges failed to close cleanly: {failed}"
            ) from first_error

    def active(self) -> int:
        with self._lock:
            return len(self._bridges)


class SwarmQueryAdapter:
    """Translate one official query into exactly two canonical Swarm operations."""

    def __init__(
        self,
        bridge: SwarmOperationBridge,
        journal: TraceJournal,
        config: AdapterConfig,
    ) -> None:
        self.bridge = bridge
        self.journal = journal
        self.config = config

    def insert(self, trajectory: Mapping[str, Any]) -> None:
        if not isinstance(trajectory, Mapping):
            raise LongMemEvalV2AdapterError("official trajectory must be an object")
        result = self.bridge.insert_trajectory(trajectory)
        if inspect.isawaitable(result):
            result.close() if inspect.iscoroutine(result) else None
            raise LongMemEvalV2AdapterError("Swarm bridge operations must be synchronous")

    def query(
        self,
        query: str,
        *,
        opaque_invocation_id: str,
        query_image: str | None = None,
    ) -> list[dict[str, str]]:
        query = _required_text(query, "query")
        opaque_invocation_id = _required_text(opaque_invocation_id, "opaque_invocation_id")
        if query_image is not None:
            _required_text(query_image, "query_image")

        recall_started = perf_counter()
        recall = self.bridge.recall_memory(query, limit=self.config.recall_limit)
        recall_latency_ms = (perf_counter() - recall_started) * 1000.0
        if inspect.isawaitable(recall):
            recall.close() if inspect.iscoroutine(recall) else None
            raise LongMemEvalV2AdapterError("Swarm bridge operations must be synchronous")
        if not isinstance(recall, RecallMemoryResult):
            raise LongMemEvalV2AdapterError("recall_memory must return RecallMemoryResult")
        if not recall.memory_ids:
            raise LongMemEvalV2AdapterError(
                "recall_memory returned no seed; iterative evidence cannot be claimed"
            )
        if len(recall.memory_ids) > self.config.recall_limit:
            raise LongMemEvalV2AdapterError("recall_memory exceeded its requested limit")
        seeds = recall.memory_ids[:8]

        expand_started = perf_counter()
        expanded = self.bridge.read_expand_memory(
            query,
            memory_ids=seeds,
            max_depth=self.config.max_depth,
            max_fanout=self.config.max_fanout,
            token_budget=self.config.token_budget,
        )
        expand_latency_ms = (perf_counter() - expand_started) * 1000.0
        if inspect.isawaitable(expanded):
            expanded.close() if inspect.iscoroutine(expanded) else None
            raise LongMemEvalV2AdapterError("Swarm bridge operations must be synchronous")
        if not isinstance(expanded, ReadExpandMemoryResult):
            raise LongMemEvalV2AdapterError("read_expand_memory must return ReadExpandMemoryResult")

        embedding = self.bridge.embedding_evidence()
        if inspect.isawaitable(embedding):
            embedding.close() if inspect.iscoroutine(embedding) else None
            raise LongMemEvalV2AdapterError("Swarm bridge operations must be synchronous")
        if not isinstance(embedding, EmbeddingRuntimeEvidence):
            raise LongMemEvalV2AdapterError(
                "embedding_evidence must return EmbeddingRuntimeEvidence"
            )

        trace = RawQueryTrace(
            opaque_invocation_id=opaque_invocation_id,
            operations=(
                RawOperation(
                    sequence=0,
                    operation="recall_memory",
                    depth=0,
                    seed_memory_ids=(),
                    result_memory_ids=recall.memory_ids,
                    latency_ms=recall_latency_ms,
                ),
                RawOperation(
                    sequence=1,
                    operation="read_expand_memory",
                    depth=self.config.max_depth,
                    seed_memory_ids=seeds,
                    result_memory_ids=expanded.memory_ids,
                    latency_ms=expand_latency_ms,
                ),
            ),
            embedding=embedding,
        )
        self.journal.record(trace)
        return [{"type": "text", "value": expanded.context}]

    def post_query_metadata(self, opaque_invocation_id: str) -> dict[str, str]:
        trace = self.journal.get(_required_text(opaque_invocation_id, "opaque_invocation_id"))
        return {RAW_TRACE_DIGEST_METADATA_KEY: raw_trace_sha256(trace)}


def build_official_memory_class(
    *,
    memory_base: type[Any],
    register_memory: Callable[[type[Any]], type[Any]],
    bridge_factory: BridgeFactory,
    journal: TraceJournal,
    lifecycle: BridgeLifecycle | None = None,
    close_after_query: bool = False,
) -> type[Any]:
    """Create and register a backend without modifying the pinned checkout."""

    bridge_lifecycle = lifecycle or BridgeLifecycle()

    class OfficialSwarmBrainMemory(memory_base):
        memory_type = MEMORY_TYPE

        def __init__(self, memory_params: dict[str, object]) -> None:
            super().__init__(memory_params)
            config = AdapterConfig.from_memory_params(memory_params)
            bridge = bridge_factory(config)
            if not callable(getattr(bridge, "close", None)):
                raise LongMemEvalV2AdapterError("bridge factory result does not implement close")
            bridge_lifecycle.register(bridge)
            for method in (
                "insert_trajectory",
                "recall_memory",
                "read_expand_memory",
                "embedding_evidence",
            ):
                if not callable(getattr(bridge, method, None)):
                    bridge_lifecycle.close(bridge)
                    raise LongMemEvalV2AdapterError(
                        f"bridge factory result does not implement {method}"
                    )
            self._swarm_bridge = bridge
            self._swarm_adapter = SwarmQueryAdapter(bridge, journal, config)

        def insert(self, trajectory: dict[str, object]) -> None:
            self._swarm_adapter.insert(trajectory)

        def query(self, query: str, query_image: str | None = None) -> list[dict[str, str]]:
            query_context = self.get_query_context()
            opaque_id = _required_text(
                query_context.get("query_invocation_id"),
                "official query_context.query_invocation_id",
            )
            return self._swarm_adapter.query(
                query,
                opaque_invocation_id=opaque_id,
                query_image=query_image,
            )

        def post_query_hook(
            self,
            *,
            query: str,
            query_image: str | None,
            memory_context: list[dict[str, str]],
        ) -> dict[str, object]:
            del query, query_image, memory_context
            query_context = self.get_query_context()
            opaque_id = _required_text(
                query_context.get("query_invocation_id"),
                "official query_context.query_invocation_id",
            )
            metadata = self._swarm_adapter.post_query_metadata(opaque_id)
            if close_after_query:
                self.close()
            return metadata

        def close(self) -> None:
            bridge_lifecycle.close(self._swarm_bridge)

    OfficialSwarmBrainMemory.__name__ = "OfficialSwarmBrainMemory"
    OfficialSwarmBrainMemory.__qualname__ = "OfficialSwarmBrainMemory"
    return register_memory(OfficialSwarmBrainMemory)


def bind_official_prompt_row(
    row: Mapping[str, Any],
    *,
    journal: TraceJournal,
    ledger: EvidenceLedger,
    tier: str,
    operating_point: str,
    domain: str,
) -> dict[str, Any]:
    """Bind, redact, and return a copy of one official prompt row."""

    opaque_id = _required_text(row.get("query_invocation_id"), "query_invocation_id")
    trace = journal.consume(opaque_id)
    evidence = bind_query_trace(
        trace,
        row,
        tier=tier,
        operating_point=operating_point,
        domain=domain,
    )
    ledger.record(evidence)
    bound = dict(row)
    # The official package keeps this metadata but omits query_invocation_id.
    # Rebuild from an allowlist so no backend-specific content can leak.
    bound["memory_post_query_metadata"] = {
        TRACE_DIGEST_METADATA_KEY: evidence.trace_sha256,
        "swarmbrain_adapter_revision": ADAPTER_REVISION,
    }
    return bound


@contextmanager
def official_prompt_row_binding(
    harness_module: Any,
    *,
    journal: TraceJournal,
    ledger: EvidenceLedger,
    tier: str,
    operating_point: str,
    domain: str,
) -> Iterator[None]:
    """Patch only the in-process prompt-row boundary, then restore it."""

    original = getattr(harness_module, "build_prompt_row", None)
    if not callable(original):
        raise LongMemEvalV2AdapterError("official harness has no build_prompt_row function")

    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        row = original(*args, **kwargs)
        if not isinstance(row, dict):
            raise LongMemEvalV2AdapterError("official build_prompt_row returned a non-object")
        return bind_official_prompt_row(
            row,
            journal=journal,
            ledger=ledger,
            tier=tier,
            operating_point=operating_point,
            domain=domain,
        )

    harness_module.build_prompt_row = wrapped
    try:
        yield
    finally:
        harness_module.build_prompt_row = original


__all__ = [
    "BridgeLifecycle",
    "BridgeFactory",
    "SwarmQueryAdapter",
    "TraceJournal",
    "bind_official_prompt_row",
    "build_official_memory_class",
    "official_prompt_row_binding",
]
