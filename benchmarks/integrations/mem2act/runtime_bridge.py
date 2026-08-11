"""Mem2ActBench bridge over Swarm Brain's public application runtime."""

from __future__ import annotations

import hashlib
import inspect
import os
import re
from time import perf_counter
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from swarmbrain.adapters.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from swarmbrain.application.memory_policy import memory_content_text
from swarmbrain.application.runtime import SwarmBrainRuntime, build_in_memory_runtime
from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.memory import MemoryState, RecallQuery, RememberCommand, Visibility

from .contracts import (
    IngestionResult,
    Mem2ActContractError,
    PublicConversationSession,
    RetrievalResult,
    RetrievedMemory,
)


class RuntimeMemoryBridge:
    """Store and recall the fixed corpus through canonical application services."""

    def __init__(
        self,
        runtime: SwarmBrainRuntime,
        actor: ActorContext,
        *,
        owns_runtime: bool = False,
        drain_embeddings: bool = False,
        embedding_revision: str | None = None,
        embedding_protocol: dict[str, Any] | None = None,
    ) -> None:
        if drain_embeddings and runtime.embeddings is None:
            raise Mem2ActContractError("drain_embeddings requires a runtime embedding provider")
        self.runtime = runtime
        self.actor = actor
        self.owns_runtime = owns_runtime
        self.drain_embeddings = drain_embeddings
        self.embedding_revision = embedding_revision
        self.embedding_protocol = dict(embedding_protocol or {})
        self._ingested = False
        self._closed = False
        self._embedding_work_completed = 0

    async def ingest(self, sessions: tuple[PublicConversationSession, ...]) -> IngestionResult:
        if self._closed:
            raise Mem2ActContractError("memory bridge is closed")
        if self._ingested:
            raise Mem2ActContractError("Mem2ActBench corpus may only be ingested once per bridge")
        if not sessions:
            raise Mem2ActContractError("Mem2ActBench corpus must not be empty")
        started = perf_counter()
        memory_ids: list[str] = []
        for position, session in enumerate(sessions):
            # The content is an allowlist: only the published conversation
            # session and its stable session ID. QA/construction source IDs,
            # evolution chains, tool labels, target arguments, and complexity
            # annotations never cross this boundary.
            result = await self.runtime.memory.publish(
                self.actor,
                RememberCommand(
                    idempotency_key=f"mem2act-corpus:{position:03d}:{session.session_id}",
                    kind="observation",
                    content={
                        "session_id": session.session_id,
                        "turns": list(session.turns),
                    },
                    desired_state=MemoryState.TENTATIVE,
                    visibility=Visibility.RUN,
                    title=f"Mem2ActBench conversation {session.session_id}",
                    tags=("mem2actbench", "conversation-session"),
                    confidence=1.0,
                    metadata={
                        "benchmark": "Mem2ActBench",
                        "corpus_position": position,
                        "session_id": session.session_id,
                    },
                ),
            )
            if result.memory is None:
                raise Mem2ActContractError(
                    f"Swarm Brain did not store Mem2ActBench session {session.session_id}"
                )
            memory_ids.append(result.memory.memory_id)
        if self.drain_embeddings:
            self._embedding_work_completed = await self._drain_embedding_work(
                expected=len(memory_ids)
            )
        self._ingested = True
        return IngestionResult(
            memory_count=len(memory_ids),
            latency_ms=(perf_counter() - started) * 1000.0,
            metadata={
                "memory_ids_sha256": _ordered_ids_sha256(memory_ids),
                "runtime_backend": self.runtime.backend.value,
                "embedding_provider": (
                    None
                    if self.runtime.embeddings is None
                    else type(self.runtime.embeddings).__name__
                ),
                "embedding_model": (
                    None if self.runtime.embeddings is None else self.runtime.embeddings.model_name
                ),
                "embedding_revision": self.embedding_revision,
                "embedding_work_completed": self._embedding_work_completed,
                "embedding_protocol": self.embedding_protocol,
                "embedding_call_accounting_after_ingest": self._embedding_call_accounting(),
            },
        )

    async def retrieve(
        self,
        query: str,
        *,
        limit: int,
        token_budget: int | None,
    ) -> RetrievalResult:
        if self._closed:
            raise Mem2ActContractError("memory bridge is closed")
        if not self._ingested:
            raise Mem2ActContractError("memory bridge must ingest the corpus before retrieval")
        if not isinstance(query, str) or not query.strip():
            raise Mem2ActContractError("retrieval query must be a non-empty string")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise Mem2ActContractError("retrieval limit must be an integer in [1, 100]")
        if token_budget is not None and (
            not isinstance(token_budget, int) or isinstance(token_budget, bool) or token_budget < 1
        ):
            raise Mem2ActContractError("retrieval token_budget must be a positive integer")

        started = perf_counter()
        bundle = await self.runtime.memory.recall(
            self.actor,
            RecallQuery(
                # Query-only retrieval is the central leakage fence.  The
                # target tool/schema/arguments and evidence source IDs are not
                # available to this method or encoded into the query.
                text=query,
                visibilities=frozenset({Visibility.RUN}),
                states=frozenset({MemoryState.TENTATIVE, MemoryState.CONFIRMED}),
                include_evidence=False,
                include_lineage=False,
                limit=limit,
            ),
            token_budget=token_budget,
        )
        latency_ms = (perf_counter() - started) * 1000.0
        memories = tuple(
            RetrievedMemory(
                memory_id=hit.memory.memory_id,
                content=memory_content_text(hit.memory.content),
                score=hit.score,
                reasons=tuple(hit.reasons),
            )
            for hit in bundle.hits
        )
        return RetrievalResult(
            memories=memories,
            latency_ms=latency_ms,
            total_candidates=bundle.total_candidates,
            truncated=bundle.truncated,
            metadata={"embedding_call_accounting": self._embedding_call_accounting()},
        )

    def evidence_metadata(self) -> dict[str, Any]:
        """Return content-free, gate-facing runtime and embedding evidence."""

        provider = self.runtime.embeddings
        return {
            "runtime_backend": self.runtime.backend.value,
            "embedding_provider": None if provider is None else type(provider).__name__,
            "embedding_model": None if provider is None else provider.model_name,
            "embedding_dimensions": None if provider is None else provider.dimensions,
            "embedding_revision": self.embedding_revision,
            "embedding_work_completed": self._embedding_work_completed,
            "embedding_protocol": self.embedding_protocol,
            "embedding_call_accounting": self._embedding_call_accounting(),
        }

    async def _drain_embedding_work(self, *, expected: int) -> int:
        worker = self.runtime.embedding_worker(retry_delay_seconds=1)
        completed = 0
        while completed < expected:
            batch = await worker.run_once(
                "mem2actbench-embedding-worker",
                limit=min(100, expected - completed),
                lease_seconds=300,
            )
            if not batch:
                break
            completed += len(batch)
        if completed != expected:
            raise Mem2ActContractError(
                f"embedding work completed {completed}/{expected}; semantic corpus is incomplete"
            )
        return completed

    def _embedding_call_accounting(self) -> dict[str, int] | None:
        provider = self.runtime.embeddings
        accounting = getattr(provider, "call_accounting", None)
        return dict(accounting) if isinstance(accounting, dict) else None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.owns_runtime:
            await self.runtime.close()
            provider_close = getattr(self.runtime.embeddings, "close", None)
            if callable(provider_close):
                result = provider_close()
                if inspect.isawaitable(result):
                    await result


def benchmark_actor(seed: str = "official-v1") -> ActorContext:
    def identifier(name: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"swarmbrain/mem2act/{seed}/{name}"))

    return ActorContext(
        tenant_id=identifier("tenant"),
        project_id=identifier("project"),
        repository_id=identifier("repository"),
        swarm_id=identifier("swarm"),
        run_id=identifier("run"),
        agent_id=identifier("agent"),
        harness="mem2actbench-external",
        harness_version="1",
        provider="benchmark",
        model="none",
        capabilities=frozenset(
            {
                Capability.MEMORY_PUBLISH.value,
                Capability.MEMORY_RECALL.value,
            }
        ),
        metadata={"benchmark": "Mem2ActBench", "protocol": "400-task-official-v1"},
    )


async def build_public_in_memory_bridge(*, seed: str = "official-v1") -> RuntimeMemoryBridge:
    """Runnable default using only the public in-memory composition root."""

    runtime = build_in_memory_runtime("mem2actbench-local-runtime-secret")
    await runtime.start()
    return RuntimeMemoryBridge(runtime, benchmark_actor(seed), owns_runtime=True)


MEM2ACT_EMBEDDING_QUERY_INSTRUCTION = (
    "Retrieve conversation memories that contain the user's current preferences, "
    "constraints, and task state needed to ground a tool call"
)


async def build_openai_semantic_in_memory_bridge() -> RuntimeMemoryBridge:
    """Canonical SOTA-run bridge configured only through environment indirection."""

    base_url = _required_env("MEM2ACT_EMBEDDINGS_BASE_URL")
    model = os.getenv("MEM2ACT_EMBEDDINGS_MODEL", "Qwen/Qwen3-Embedding-0.6B").strip()
    if not model:
        raise Mem2ActContractError("MEM2ACT_EMBEDDINGS_MODEL must be configured")
    revision = _required_env("MEM2ACT_EMBEDDINGS_REVISION")
    api_key_env = os.getenv("MEM2ACT_EMBEDDINGS_API_KEY_ENV", "MEM2ACT_EMBEDDINGS_API_KEY").strip()
    if api_key_env and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env) is None:
        raise Mem2ActContractError(
            "MEM2ACT_EMBEDDINGS_API_KEY_ENV must name an environment variable"
        )
    api_key = os.getenv(api_key_env) if api_key_env else None
    provider = OpenAICompatibleEmbeddingProvider(
        base_url=base_url,
        model_id=model,
        dimensions=1024,
        api_key=api_key,
        required_response_model=model,
        query_instruction=MEM2ACT_EMBEDDING_QUERY_INSTRUCTION,
    )
    runtime = build_in_memory_runtime(
        "mem2actbench-local-semantic-runtime-secret",
        embeddings=provider,
    )
    await runtime.start()
    return RuntimeMemoryBridge(
        runtime,
        benchmark_actor("official-semantic-v1"),
        owns_runtime=True,
        drain_embeddings=True,
        embedding_revision=revision,
        embedding_protocol={
            "name": "mem2act-openai-semantic-v1",
            "api_key_env": api_key_env or None,
            "response_model_requirement": model,
            "query_instruction_sha256": hashlib.sha256(
                MEM2ACT_EMBEDDING_QUERY_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
            "document_projection": "canonical memory content",
            "corpus_embedding_work": "durable queue drained and completion-count reconciled",
        },
    )


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise Mem2ActContractError(f"{name} must be configured")
    return value


def _ordered_ids_sha256(memory_ids: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(memory_ids).encode("utf-8")).hexdigest()


__all__ = [
    "RuntimeMemoryBridge",
    "benchmark_actor",
    "build_openai_semantic_in_memory_bridge",
    "build_public_in_memory_bridge",
]
