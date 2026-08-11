"""Canonical Swarm Brain runtime behind MemoryArena's three memory operations."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any, TypeAlias
from uuid import NAMESPACE_URL, uuid5

from swarmbrain.adapters.embeddings import DeterministicEmbeddingProvider
from swarmbrain.adapters.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from swarmbrain.application.memory_policy import memory_content_text
from swarmbrain.application.retrieval_service import RetrievalExecution
from swarmbrain.application.runtime import SwarmBrainRuntime, build_in_memory_runtime
from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.memory import MemoryState, RecallQuery, RememberCommand, Visibility
from swarmbrain.domain.retrieval import RetrievalPurpose, RetrievalSignal
from swarmbrain.retrieval.packing import estimate_tokens

from .contracts import (
    DETERMINISTIC_EMBEDDING_MODE,
    SEMANTIC_EMBEDDING_MODE,
    SEMANTIC_EMBEDDING_MODEL_ID,
    SEMANTIC_EMBEDDING_PROVIDER,
    SEMANTIC_EMBEDDING_QUERY_INSTRUCTION,
    SEMANTIC_EMBEDDING_QUERY_INSTRUCTION_SHA256,
    BridgeConfig,
    MemoryArenaContractError,
    MemoryArenaNotInitialized,
    MemoryArenaSystemMismatch,
    MemoryArenaUnsupportedSystem,
    canonical_json,
    sha256_json,
    sha256_text,
)
from .evidence import (
    ContentFreeEvidenceLedger,
    OperationEvidence,
    assert_content_free,
    elapsed_ms,
    empty_ids_sha256,
    invocation_id,
    ordered_ids_sha256,
)

RuntimeFactory: TypeAlias = Callable[
    [str, BridgeConfig, Any],
    SwarmBrainRuntime | Awaitable[SwarmBrainRuntime],
]


@dataclass(slots=True)
class _ScopeState:
    scope_sha256: str
    runtime: SwarmBrainRuntime
    actor: ActorContext
    memory_system_name: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    memory_count: int = 0
    embedding_work_completed: int = 0
    closed: bool = False


class MemoryArenaRuntimeBridge:
    """Async implementation of the official MemoryArena memory seam.

    One independently composed runtime is allocated per official ``user_id``.
    The pinned runners use that value as a task/group identifier; consequently
    no memory, vector index, work queue, or authenticated actor is shared across
    task groups.  Re-initialization atomically replaces the old runtime.
    """

    def __init__(
        self,
        *,
        config: BridgeConfig | None = None,
        runtime_factory: RuntimeFactory | None = None,
        openai_client: Any | None = None,
    ) -> None:
        self.config = config or BridgeConfig()
        self._embedding_provider = _build_embedding_provider(
            self.config,
            openai_client=openai_client,
        )
        self._runtime_factory = runtime_factory or _build_runtime
        self._scopes: dict[str, _ScopeState] = {}
        self._registry_lock = asyncio.Lock()
        self._ledger = ContentFreeEvidenceLedger()
        self._successful_nonempty_adds = 0
        self._embedding_work_completed = 0
        self._document_embedding_failures = 0
        self._dense_queries_required = 0
        self._dense_queries_completed = 0
        self._dense_fallbacks = 0
        self._provider_closed = False
        self._closed = False

    async def initialize(self, user_id: str, memory_system_name: str) -> dict[str, Any]:
        user_id = self._bounded_string(user_id, "user_id", self.config.max_user_id_chars)
        memory_system_name = self._bounded_string(memory_system_name, "memory_system_name", 128)
        scope_sha256 = _scope_sha256(user_id, memory_system_name)
        request = {"user_id": user_id, "memory_system_name": memory_system_name}
        started = perf_counter_ns()
        before = 0
        if memory_system_name != self.config.memory_system_name:
            error = MemoryArenaUnsupportedSystem(f"Unsupported memory_system: {memory_system_name}")
            self._record(
                operation="initialize",
                scope_sha256=scope_sha256,
                request=request,
                started_ns=started,
                error=error,
            )
            raise error
        self._require_open()

        runtime: SwarmBrainRuntime | None = None
        try:
            built = self._runtime_factory(
                scope_sha256,
                self.config,
                self._embedding_provider,
            )
            runtime = await built if inspect.isawaitable(built) else built
            if not isinstance(runtime, SwarmBrainRuntime):
                raise MemoryArenaContractError("runtime factory returned an invalid runtime")
            if runtime.embeddings is None or runtime.work_queue is None:
                raise MemoryArenaContractError(
                    "MemoryArena runtime requires embeddings and durable embedding work"
                )
            if runtime.embeddings is not self._embedding_provider:
                raise MemoryArenaContractError(
                    "MemoryArena runtime must use the bridge-owned embedding provider"
                )
            await runtime.start()
            state = _ScopeState(
                scope_sha256=scope_sha256,
                runtime=runtime,
                actor=_actor_for_scope(scope_sha256),
                memory_system_name=memory_system_name,
            )
            async with self._registry_lock:
                self._require_open()
                previous = self._scopes.get(user_id)
                before = 0 if previous is None else previous.memory_count
                self._scopes[user_id] = state
            if previous is not None:
                await self._close_state(previous)
        except BaseException as exc:
            if runtime is not None and all(
                item.runtime is not runtime for item in self._scopes.values()
            ):
                await _close_runtime(runtime)
            self._record(
                operation="initialize",
                scope_sha256=scope_sha256,
                request=request,
                started_ns=started,
                memory_count_before=before,
                error=exc,
            )
            raise

        response = {
            "status": "ok",
            "user_id": user_id,
            "memory_system_name": memory_system_name,
        }
        self._record(
            operation="initialize",
            scope_sha256=scope_sha256,
            request=request,
            response=response,
            started_ns=started,
            memory_count_before=before,
        )
        return response

    async def add(self, user_id: str, memory_system_name: str, chunk: str) -> dict[str, Any]:
        user_id = self._bounded_string(user_id, "user_id", self.config.max_user_id_chars)
        memory_system_name = self._bounded_string(memory_system_name, "memory_system_name", 128)
        chunk = self._bounded_string(chunk, "chunk", self.config.max_chunk_chars)
        request = {
            "user_id": user_id,
            "memory_system_name": memory_system_name,
            "chunk": chunk,
        }
        started = perf_counter_ns()
        state: _ScopeState | None = None
        before = 0
        embedded = 0
        selected_ids: tuple[str, ...] = ()
        semantic_embedding_attempted = False
        try:
            state = await self._state_for(user_id, memory_system_name)
            async with state.lock:
                self._require_state_open(state)
                before = state.memory_count
                if chunk.strip():
                    result = await state.runtime.memory.publish(
                        state.actor,
                        RememberCommand(
                            idempotency_key=(
                                f"memoryarena-add:{state.memory_count}:{sha256_text(chunk)[:48]}"
                            ),
                            kind="observation",
                            content=chunk,
                            desired_state=MemoryState.TENTATIVE,
                            visibility=Visibility.RUN,
                            title="MemoryArena interaction chunk",
                            tags=("memoryarena", "official-memory-api"),
                            confidence=1.0,
                            metadata={
                                "benchmark": "MemoryArena",
                                "scope_sha256": state.scope_sha256,
                                "ordinal": state.memory_count,
                            },
                        ),
                    )
                    if result.memory is None:
                        raise MemoryArenaContractError(
                            "canonical runtime did not append the MemoryArena chunk"
                        )
                    selected_ids = (result.memory.memory_id,)
                    semantic_embedding_attempted = (
                        self.config.embedding_mode == SEMANTIC_EMBEDDING_MODE
                    )
                    embedded = await _drain_embedding_work(state, expected=1)
                    state.memory_count += 1
                    state.embedding_work_completed += embedded
                    self._successful_nonempty_adds += 1
                    self._embedding_work_completed += embedded
        except BaseException as exc:
            if semantic_embedding_attempted:
                self._document_embedding_failures += 1
            self._record(
                operation="add",
                scope_sha256=(
                    _scope_sha256(user_id, memory_system_name)
                    if state is None
                    else state.scope_sha256
                ),
                request=request,
                started_ns=started,
                memory_count_before=before,
                memory_count_after=before,
                embedding_work_completed=embedded,
                selected_memory_ids=selected_ids,
                error=exc,
            )
            raise

        assert state is not None
        response = {
            "status": "ok",
            "user_id": user_id,
            "response": {
                "stored": bool(selected_ids),
                "memory_sha256": (None if not selected_ids else sha256_text(selected_ids[0])),
                "embedding_work_completed": embedded,
            },
        }
        self._record(
            operation="add",
            scope_sha256=state.scope_sha256,
            request=request,
            response=response,
            started_ns=started,
            memory_count_before=before,
            memory_count_after=state.memory_count,
            embedding_work_completed=embedded,
            selected_memory_ids=selected_ids,
        )
        return response

    async def wrap_user_prompt(
        self,
        user_id: str,
        memory_system_name: str,
        question: str,
    ) -> dict[str, Any]:
        user_id = self._bounded_string(user_id, "user_id", self.config.max_user_id_chars)
        memory_system_name = self._bounded_string(memory_system_name, "memory_system_name", 128)
        question = self._bounded_string(question, "question", self.config.max_question_chars)
        request = {
            "user_id": user_id,
            "memory_system_name": memory_system_name,
            "question": question,
        }
        started = perf_counter_ns()
        state: _ScopeState | None = None
        selected_ids: tuple[str, ...] = ()
        dropped_count = 0
        context_tokens = 0
        before = 0
        dense_required = False
        dense_completed = False
        dense_fallback = False
        try:
            state = await self._state_for(user_id, memory_system_name)
            async with state.lock:
                self._require_state_open(state)
                before = state.memory_count
                hits = ()
                if question.strip() and state.memory_count:
                    dense_required = self.config.embedding_mode == SEMANTIC_EMBEDDING_MODE
                    if dense_required:
                        self._dense_queries_required += 1
                    recalled = await state.runtime.memory.recall_for_activation(
                        state.actor,
                        RecallQuery(
                            text=question,
                            visibilities=frozenset({Visibility.RUN}),
                            states=frozenset({MemoryState.TENTATIVE, MemoryState.CONFIRMED}),
                            include_evidence=False,
                            include_lineage=False,
                            limit=self.config.recall_limit,
                        ),
                        purpose=RetrievalPurpose.INTERACTIVE_RECALL,
                        seed_memory_ids=(),
                        token_budget=self.config.memory_context_token_budget,
                    )
                    if not isinstance(recalled, RetrievalExecution):
                        raise MemoryArenaContractError(
                            "canonical retrieval did not return packing evidence"
                        )
                    if dense_required:
                        try:
                            _require_available_dense_lane(recalled)
                        except MemoryArenaContractError:
                            self._dense_fallbacks += 1
                            dense_fallback = True
                            raise
                        self._dense_queries_completed += 1
                        dense_completed = True
                    hits = recalled.bundle.hits
                    selected_ids = tuple(hit.memory.memory_id for hit in hits)
                    if recalled.trace.packing is not None:
                        dropped_count = len(recalled.trace.packing.dropped_ids)
                chunks = tuple(memory_content_text(hit.memory.content) for hit in hits)
                context = "\n".join(f"<memory>{chunk}</memory>" for chunk in chunks)
                if not context:
                    context = "None"
                context_tokens = estimate_tokens(context)
                prompt = f"<memory_context>\n{context}\n</memory_context>\nUser: {question}"
        except BaseException as exc:
            if dense_required and not dense_completed and not dense_fallback:
                self._dense_fallbacks += 1
                dense_fallback = True
            self._record(
                operation="wrap_user_prompt",
                scope_sha256=(
                    _scope_sha256(user_id, memory_system_name)
                    if state is None
                    else state.scope_sha256
                ),
                request=request,
                started_ns=started,
                memory_count_before=before,
                memory_count_after=before,
                selected_memory_ids=selected_ids,
                dropped_memory_count=dropped_count,
                context_estimated_tokens=context_tokens,
                dense_required=dense_required,
                dense_completed=dense_completed,
                dense_fallback=dense_fallback,
                error=exc,
            )
            raise

        assert state is not None
        response = {"status": "ok", "user_id": user_id, "prompt": prompt}
        self._record(
            operation="wrap_user_prompt",
            scope_sha256=state.scope_sha256,
            request=request,
            response=response,
            started_ns=started,
            memory_count_before=before,
            memory_count_after=before,
            selected_memory_ids=selected_ids,
            dropped_memory_count=dropped_count,
            context_estimated_tokens=context_tokens,
            dense_required=dense_required,
            dense_completed=dense_completed,
            dense_fallback=dense_fallback,
        )
        return response

    async def cleanup(self, user_id: str) -> bool:
        """Release one task scope; repeated cleanup is deliberately a no-op."""

        user_id = self._bounded_string(user_id, "user_id", self.config.max_user_id_chars)
        started = perf_counter_ns()
        async with self._registry_lock:
            state = self._scopes.pop(user_id, None)
        scope_sha256 = sha256_json({"memoryarena_user_scope": user_id})
        before = 0 if state is None else state.memory_count
        if state is not None:
            scope_sha256 = state.scope_sha256
            await self._close_state(state)
        self._record(
            operation="cleanup",
            scope_sha256=scope_sha256,
            request={"cleanup_scope": user_id},
            response={"cleaned": state is not None},
            started_ns=started,
            memory_count_before=before,
        )
        return state is not None

    async def close(self) -> None:
        if self._closed and self._provider_closed:
            return
        states: tuple[_ScopeState, ...] = ()
        if not self._closed:
            self._closed = True
            async with self._registry_lock:
                states = tuple(self._scopes.values())
                self._scopes.clear()
        first_error: BaseException | None = None
        for state in states:
            try:
                await self._close_state(state)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if not self._provider_closed:
            try:
                await _close_embedding_provider(self._embedding_provider)
                self._provider_closed = True
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def evidence(self) -> dict[str, Any]:
        payload = self._ledger.export()
        embedding = self._embedding_execution_evidence()
        payload["embedding_execution"] = embedding
        payload["embedding_execution_sha256"] = sha256_json(embedding)
        assert_content_free(payload)
        return payload

    async def active_scope_count(self) -> int:
        async with self._registry_lock:
            return len(self._scopes)

    async def _state_for(self, user_id: str, memory_system_name: str) -> _ScopeState:
        self._require_open()
        async with self._registry_lock:
            state = self._scopes.get(user_id)
        if state is None:
            raise MemoryArenaNotInitialized("User not initialized")
        if state.memory_system_name != memory_system_name:
            raise MemoryArenaSystemMismatch("Mismatched memory_system for user")
        return state

    async def _close_state(self, state: _ScopeState) -> None:
        async with state.lock:
            if state.closed:
                return
            state.closed = True
            await _close_runtime(state.runtime)

    def _embedding_execution_evidence(self) -> dict[str, Any]:
        semantic = self.config.embedding_mode == SEMANTIC_EMBEDDING_MODE
        accounting = self._provider_accounting()
        provider_identity_verified = (
            semantic
            and isinstance(self._embedding_provider, OpenAICompatibleEmbeddingProvider)
            and self._embedding_provider.model_name == SEMANTIC_EMBEDDING_MODEL_ID
            and self._embedding_provider.dimensions == self.config.embedding_dimensions
            and self._embedding_provider.required_response_model
            == self.config.embedding_response_model
        )
        document_coverage = (
            self._document_embedding_failures == 0
            and self._successful_nonempty_adds == self._embedding_work_completed
            and self._successful_nonempty_adds == accounting["document_inputs"]
            and self._successful_nonempty_adds == accounting["document_batch_calls"]
        )
        query_coverage = (
            self._dense_fallbacks == 0
            and self._dense_queries_required == self._dense_queries_completed
            and self._dense_queries_required == accounting["query_calls"]
        )
        expected_successful_calls = accounting["document_batch_calls"] + accounting["query_calls"]
        provider_accounting_complete = (
            accounting["successful_http_calls"] == expected_successful_calls
            and accounting["http_attempts"] >= accounting["successful_http_calls"]
        )
        exact_response_model_verified = (
            provider_identity_verified
            and expected_successful_calls > 0
            and provider_accounting_complete
        )
        # The OpenAI-compatible embeddings response identifies only the served
        # model alias.  It does not attest the immutable source/weights revision
        # supplied by the operator.  A revision-shaped CLI value is therefore a
        # declaration, not execution proof.  Keep this false until a trusted
        # deployment attestation (or an equivalently verifiable local model
        # manifest bound to the endpoint) is implemented and compiler-verified.
        model_revision_binding_verified = False
        full_coverage = document_coverage and query_coverage and provider_accounting_complete
        publishable = (
            semantic
            and self._successful_nonempty_adds > 0
            and self._dense_queries_required > 0
            and provider_identity_verified
            and exact_response_model_verified
            and model_revision_binding_verified
            and full_coverage
        )
        configuration_payload = {
            "mode": self.config.embedding_mode,
            "base_url": self.config.embedding_base_url,
            "api_key_env": self.config.embedding_api_key_env,
            "model_id": self.config.embedding_model_id,
            "model_revision": self.config.embedding_model_revision,
            "dimensions": self.config.embedding_dimensions,
            "response_model": self.config.embedding_response_model,
            "query_instruction_sha256": (
                SEMANTIC_EMBEDDING_QUERY_INSTRUCTION_SHA256 if semantic else None
            ),
        }
        return {
            "mode": self.config.embedding_mode,
            "publishable": publishable,
            "provider": (
                SEMANTIC_EMBEDDING_PROVIDER if semantic else "DeterministicEmbeddingProvider"
            ),
            "model_id": (
                self.config.embedding_model_id if semantic else self._embedding_provider.model_name
            ),
            "model_revision": self.config.embedding_model_revision if semantic else None,
            "model_revision_source": ("operator-declared-unverified" if semantic else "none"),
            "provider_attested_model_revision": None,
            "deployment_manifest_bound_revision": False,
            "model_revision_binding_verified": model_revision_binding_verified,
            "immutable_model_revision": model_revision_binding_verified,
            "dimensions": self.config.embedding_dimensions,
            "required_response_model": (self.config.embedding_response_model if semantic else None),
            "exact_response_model_verified": exact_response_model_verified,
            "query_instruction_sha256": (
                SEMANTIC_EMBEDDING_QUERY_INSTRUCTION_SHA256 if semantic else None
            ),
            "provider_configuration_sha256": sha256_json(configuration_payload),
            "credential_source": ("environment-variable-name-indirection" if semantic else "none"),
            "call_accounting": {
                "source": "provider-observed" if semantic else "not-provider-observed",
                **accounting,
            },
            "coverage": {
                "successful_nonempty_adds": self._successful_nonempty_adds,
                "embedding_work_completed": self._embedding_work_completed,
                "document_embedding_failures": self._document_embedding_failures,
                "dense_queries_required": self._dense_queries_required,
                "dense_queries_completed": self._dense_queries_completed,
                "dense_fallbacks": self._dense_fallbacks,
                "full_document_coverage": document_coverage,
                "full_query_coverage": query_coverage,
                "provider_accounting_complete": provider_accounting_complete,
                "full_coverage": full_coverage,
                "zero_fallback": self._dense_fallbacks == 0,
            },
            "provider_lifecycle": {
                "owner": "bridge",
                "closed": self._provider_closed,
            },
            "publishability_blockers": (
                ["embedding_model_revision_not_attested_or_deployment_manifest_bound"]
                if semantic
                else ["deterministic_embedding_mode"]
            ),
        }

    def _provider_accounting(self) -> dict[str, int]:
        if self.config.embedding_mode == DETERMINISTIC_EMBEDDING_MODE:
            return _zero_provider_accounting()
        if not isinstance(self._embedding_provider, OpenAICompatibleEmbeddingProvider):
            raise MemoryArenaContractError(
                "semantic mode does not own OpenAICompatibleEmbeddingProvider"
            )
        accounting = self._embedding_provider.call_accounting
        expected = _zero_provider_accounting()
        if set(accounting) != set(expected) or any(
            type(value) is not int or value < 0 for value in accounting.values()
        ):
            raise MemoryArenaContractError("embedding provider returned malformed accounting")
        return dict(accounting)

    def _require_open(self) -> None:
        if self._closed:
            raise MemoryArenaContractError("MemoryArena runtime bridge is closed")

    @staticmethod
    def _require_state_open(state: _ScopeState) -> None:
        if state.closed:
            raise MemoryArenaNotInitialized("User not initialized")

    @staticmethod
    def _bounded_string(value: Any, label: str, maximum: int) -> str:
        if not isinstance(value, str):
            raise MemoryArenaContractError(f"{label} must be a string")
        if len(value) > maximum:
            raise MemoryArenaContractError(f"{label} exceeds {maximum} characters")
        return value

    def _record(
        self,
        *,
        operation: str,
        scope_sha256: str,
        request: dict[str, Any],
        started_ns: int,
        response: dict[str, Any] | None = None,
        memory_count_before: int = 0,
        memory_count_after: int = 0,
        selected_memory_ids: tuple[str, ...] = (),
        dropped_memory_count: int = 0,
        embedding_work_completed: int = 0,
        context_estimated_tokens: int = 0,
        dense_required: bool = False,
        dense_completed: bool = False,
        dense_fallback: bool = False,
        error: BaseException | None = None,
    ) -> None:
        sequence = self._ledger.next_sequence()
        encoded_request = canonical_json(request).encode("utf-8")
        event = OperationEvidence(
            sequence=sequence,
            invocation_id=invocation_id(sequence, operation, scope_sha256),
            operation=operation,
            scope_sha256=scope_sha256,
            request_sha256=sha256_text(encoded_request.decode("utf-8")),
            request_bytes=len(encoded_request),
            response_sha256=None if response is None else sha256_json(response),
            success=error is None,
            error_code=None if error is None else type(error).__name__,
            memory_count_before=memory_count_before,
            memory_count_after=memory_count_after,
            selected_memory_ids_sha256=(
                empty_ids_sha256()
                if not selected_memory_ids
                else ordered_ids_sha256(selected_memory_ids)
            ),
            selected_memory_count=len(selected_memory_ids),
            dropped_memory_count=dropped_memory_count,
            embedding_work_completed=embedding_work_completed,
            context_estimated_tokens=context_estimated_tokens,
            dense_required=dense_required,
            dense_completed=dense_completed,
            dense_fallback=dense_fallback,
            latency_ms=elapsed_ms(started_ns),
        )
        self._ledger.append(event)


def _build_embedding_provider(
    config: BridgeConfig,
    *,
    openai_client: Any | None,
) -> DeterministicEmbeddingProvider | OpenAICompatibleEmbeddingProvider:
    if config.embedding_mode == DETERMINISTIC_EMBEDDING_MODE:
        if openai_client is not None:
            raise MemoryArenaContractError(
                "openai_client can only be supplied in openai_semantic mode"
            )
        return DeterministicEmbeddingProvider(
            dimensions=config.embedding_dimensions,
            model_name="memoryarena-deterministic-local-v1",
        )

    assert config.embedding_api_key_env is not None
    assert config.embedding_base_url is not None
    assert config.embedding_model_id is not None
    assert config.embedding_response_model is not None
    api_key = os.getenv(config.embedding_api_key_env, "").strip()
    if not api_key:
        raise MemoryArenaContractError(
            "embedding_api_key_env resolves to an unset or empty variable"
        )
    return OpenAICompatibleEmbeddingProvider(
        base_url=config.embedding_base_url,
        model_id=config.embedding_model_id,
        dimensions=config.embedding_dimensions,
        api_key=api_key,
        required_response_model=config.embedding_response_model,
        query_instruction=SEMANTIC_EMBEDDING_QUERY_INSTRUCTION,
        client=openai_client,
    )


async def _build_runtime(
    scope_sha256: str,
    config: BridgeConfig,
    provider: Any,
) -> SwarmBrainRuntime:
    return build_in_memory_runtime(
        f"memoryarena-local-{scope_sha256}",
        embeddings=provider,
    )


def _actor_for_scope(scope_sha256: str) -> ActorContext:
    def identifier(label: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"swarmbrain/memoryarena/{scope_sha256}/{label}"))

    return ActorContext(
        tenant_id=identifier("tenant"),
        project_id=identifier("project"),
        repository_id=identifier("repository"),
        swarm_id=identifier("swarm"),
        run_id=identifier("run"),
        agent_id=identifier("agent"),
        harness="memoryarena-official-api",
        harness_version="1",
        provider="benchmark",
        model="none",
        capabilities=frozenset(
            {
                Capability.MEMORY_PUBLISH.value,
                Capability.MEMORY_RECALL.value,
            }
        ),
        metadata={
            "benchmark": "MemoryArena",
            "protocol": "official-memory-api",
            "scope_sha256": scope_sha256,
        },
    )


def _scope_sha256(user_id: str, memory_system_name: str) -> str:
    return sha256_json(
        {
            "memory_system_name": memory_system_name,
            "protocol": "swarmbrain-memoryarena-scope-v1",
            "user_id": user_id,
        }
    )


async def _drain_embedding_work(state: _ScopeState, *, expected: int) -> int:
    worker = state.runtime.embedding_worker(retry_delay_seconds=1)
    completed = 0
    while completed < expected:
        batch = await worker.run_once(
            f"memoryarena-embedding-{state.scope_sha256[:16]}",
            limit=expected - completed,
            lease_seconds=300,
        )
        if not batch:
            break
        completed += len(batch)
    if completed != expected:
        raise MemoryArenaContractError(
            f"embedding work completed {completed}/{expected}; recall would be incomplete"
        )
    return completed


def _require_available_dense_lane(execution: RetrievalExecution) -> None:
    batches = tuple(
        batch for batch in execution.trace.batches if batch.lane is RetrievalSignal.DENSE
    )
    if (
        RetrievalSignal.DENSE not in execution.trace.plan.signal_lanes
        or len(batches) != 1
        or batches[0].degraded
        or RetrievalSignal.DENSE in execution.trace.degraded_lanes
    ):
        raise MemoryArenaContractError(
            "publishable semantic retrieval requires one available dense lane; "
            "lexical fallback is forbidden"
        )


def _zero_provider_accounting() -> dict[str, int]:
    return {
        "document_inputs": 0,
        "document_batch_calls": 0,
        "query_calls": 0,
        "successful_http_calls": 0,
        "http_attempts": 0,
    }


async def _close_runtime(runtime: SwarmBrainRuntime) -> None:
    await runtime.close()


async def _close_embedding_provider(provider: Any) -> None:
    provider_close = getattr(provider, "close", None)
    if callable(provider_close):
        result = provider_close()
        if inspect.isawaitable(result):
            await result


__all__ = ["MemoryArenaRuntimeBridge", "RuntimeFactory"]
