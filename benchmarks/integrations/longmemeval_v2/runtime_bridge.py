"""Concrete LongMemEval-V2 bridge over the canonical local Swarm runtime.

The official benchmark boundary is synchronous, while Swarm Brain's
application services are asynchronous.  This module owns one event-loop thread
per official memory instance and keeps every async object on that thread.  A
fresh in-memory composition root, run scope, task, and lease are created for
each ordered haystack; no state is shared between questions.

Only the local in-memory state backend is supported here.  It is the real
application runtime (coordination, memory policy, retrieval planner, graph
expansion, and durable embedding worker).  The SOTA-capable mode projects
through a hardened OpenAI-compatible embedding endpoint; lexical and
deterministic modes remain explicit development modes.  Unsupported or
misspelled configuration is rejected rather than silently falling back.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import secrets
import threading
from collections.abc import Awaitable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

from swarmbrain.adapters.embeddings import DeterministicEmbeddingProvider
from swarmbrain.adapters.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from swarmbrain.application.runtime import SwarmBrainRuntime, build_in_memory_runtime
from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.exploration import ReadExpandMemoryRequest
from swarmbrain.domain.memory import (
    MemoryKind,
    MemoryState,
    RecallQuery,
    RememberCommand,
    Visibility,
)
from swarmbrain.domain.tasks import (
    ClaimTaskCommand,
    ClaimTaskResult,
    CompleteTaskCommand,
    ReleaseTaskCommand,
    Task,
    TaskOutcome,
    TaskStatus,
)

from .contracts import (
    ADAPTER_REVISION,
    SOTA_EMBEDDING_DIMENSIONS,
    SOTA_EMBEDDING_MODEL,
    SOTA_EMBEDDING_PROVIDER,
    SOTA_EMBEDDING_QUERY_INSTRUCTION,
    SOTA_EMBEDDING_QUERY_INSTRUCTION_SHA256,
    AdapterConfig,
    EmbeddingRuntimeEvidence,
    LongMemEvalV2AdapterError,
    ReadExpandMemoryResult,
    RecallMemoryResult,
)

LOCAL_RUNTIME_BRIDGE_FACTORY = (
    "benchmarks.integrations.longmemeval_v2.runtime_bridge:build_local_runtime_bridge"
)

_SCOPE_PROTOCOL = "swarmbrain/longmemeval-v2/local-runtime-v1"
_CLOCK_START = datetime(2026, 8, 9, 0, 0, tzinfo=UTC)
_DEFAULT_CHUNK_CHARS = 6_000
_DEFAULT_EMBEDDING_DIMENSIONS = SOTA_EMBEDDING_DIMENSIONS
_MAX_PUBLIC_FIELD_CHARS = 65_536
_ENVIRONMENT_VARIABLE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RUNTIME_CAPABILITIES = frozenset(
    {
        Capability.RUN_JOIN.value,
        Capability.TASK_CLAIM.value,
        Capability.TASK_COMPLETE.value,
        Capability.TASK_RELEASE.value,
        Capability.MEMORY_PUBLISH.value,
        Capability.MEMORY_RECALL.value,
        Capability.MEMORY_CONFIRM.value,
    }
)

_T = TypeVar("_T")


def _configuration_error(message: str) -> LongMemEvalV2AdapterError:
    return LongMemEvalV2AdapterError(f"local runtime bridge configuration {message}")


def _bounded_integer(
    params: Mapping[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = params.get(name, default)
    if type(value) is not int or not minimum <= value <= maximum:
        raise _configuration_error(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _bounded_float(
    params: Mapping[str, Any],
    name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = params.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= float(value) <= maximum
    ):
        raise _configuration_error(f"{name} must be finite and in [{minimum}, {maximum}]")
    return float(value)


@dataclass(frozen=True, slots=True)
class LocalRuntimeBridgeSettings:
    """Strict public configuration for the local state backend."""

    retrieval_mode: str
    chunk_chars: int
    embedding_dimensions: int
    dense_min_similarity: float
    recall_min_score: float
    embedding_base_url: str | None
    embedding_api_key_env: str | None
    embedding_model: str | None
    embedding_model_revision: str | None
    embedding_response_model: str | None

    @classmethod
    def from_adapter_config(cls, config: AdapterConfig) -> LocalRuntimeBridgeSettings:
        params = dict(config.bridge_params)
        allowed = {
            "backend",
            "retrieval_mode",
            "chunk_chars",
            "embedding_dimensions",
            "dense_min_similarity",
            "recall_min_score",
            "embedding_base_url",
            "embedding_api_key_env",
            "embedding_model",
            "embedding_model_revision",
            "embedding_response_model",
        }
        unexpected = sorted(set(params) - allowed)
        if unexpected:
            names = ", ".join(unexpected)
            raise _configuration_error(f"contains unsupported field(s): {names}")

        backend = params.get("backend", "in_memory")
        if backend != "in_memory":
            raise _configuration_error("backend must be exactly 'in_memory'")
        retrieval_mode = params.get("retrieval_mode", "deterministic_hybrid")
        if retrieval_mode not in {"lexical", "deterministic_hybrid", "openai_hybrid"}:
            raise _configuration_error(
                "retrieval_mode must be 'lexical', 'deterministic_hybrid', or 'openai_hybrid'"
            )
        openai_fields = {
            "embedding_base_url",
            "embedding_api_key_env",
            "embedding_model",
            "embedding_model_revision",
            "embedding_response_model",
        }
        if retrieval_mode == "lexical" and (
            "embedding_dimensions" in params or "dense_min_similarity" in params
        ):
            raise _configuration_error(
                "embedding settings cannot be supplied when retrieval_mode is 'lexical'"
            )
        if retrieval_mode != "openai_hybrid" and set(params) & openai_fields:
            raise _configuration_error(
                "OpenAI embedding settings require retrieval_mode 'openai_hybrid'"
            )

        embedding_base_url: str | None = None
        embedding_api_key_env: str | None = None
        embedding_model: str | None = None
        embedding_model_revision: str | None = None
        embedding_response_model: str | None = None
        if retrieval_mode == "openai_hybrid":
            embedding_base_url = _validated_embedding_base_url(params.get("embedding_base_url"))
            embedding_api_key_env = _required_setting(
                params.get("embedding_api_key_env"), "embedding_api_key_env"
            )
            if _ENVIRONMENT_VARIABLE_RE.fullmatch(embedding_api_key_env) is None:
                raise _configuration_error(
                    "embedding_api_key_env must name an environment variable"
                )
            embedding_model = params.get("embedding_model", SOTA_EMBEDDING_MODEL)
            if embedding_model != SOTA_EMBEDDING_MODEL:
                raise _configuration_error(
                    f"embedding_model must be exactly {SOTA_EMBEDDING_MODEL!r}"
                )
            embedding_model_revision = _required_setting(
                params.get("embedding_model_revision"), "embedding_model_revision"
            )
            if len(embedding_model_revision) > 255:
                raise _configuration_error("embedding_model_revision cannot exceed 255 characters")
            embedding_response_model = params.get("embedding_response_model", SOTA_EMBEDDING_MODEL)
            if embedding_response_model != SOTA_EMBEDDING_MODEL:
                raise _configuration_error(
                    f"embedding_response_model must be exactly {SOTA_EMBEDDING_MODEL!r}"
                )

        dimensions = _bounded_integer(
            params,
            "embedding_dimensions",
            default=_DEFAULT_EMBEDDING_DIMENSIONS,
            minimum=32,
            maximum=4_096,
        )
        return cls(
            retrieval_mode=retrieval_mode,
            chunk_chars=_bounded_integer(
                params,
                "chunk_chars",
                default=_DEFAULT_CHUNK_CHARS,
                minimum=1_024,
                maximum=32_768,
            ),
            embedding_dimensions=dimensions,
            dense_min_similarity=_bounded_float(
                params,
                "dense_min_similarity",
                default=0.0,
                minimum=0.0,
                maximum=1.0,
            ),
            recall_min_score=_bounded_float(
                params,
                "recall_min_score",
                default=0.0,
                minimum=0.0,
                maximum=1.0,
            ),
            embedding_base_url=embedding_base_url,
            embedding_api_key_env=embedding_api_key_env,
            embedding_model=embedding_model,
            embedding_model_revision=embedding_model_revision,
            embedding_response_model=embedding_response_model,
        )


def _required_setting(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _configuration_error(f"{name} must be a non-empty public string")
    return value.strip()


def _validated_embedding_base_url(value: Any) -> str:
    base_url = _required_setting(value, "embedding_base_url").rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise _configuration_error(
            "embedding_base_url must be an http(s) URL without credentials, query, or fragment"
        )
    return base_url


def _required_public_text(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongMemEvalV2AdapterError(f"{label} must be a non-empty string")
    text = value.strip()
    if len(text) > maximum:
        raise LongMemEvalV2AdapterError(f"{label} exceeds {maximum} characters")
    return text


def _optional_public_text(value: Any, label: str, *, maximum: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LongMemEvalV2AdapterError(f"{label} must be a string or null")
    text = value.strip()
    if len(text) > maximum:
        raise LongMemEvalV2AdapterError(f"{label} exceeds {maximum} characters")
    return text or None


@dataclass(frozen=True, slots=True)
class _PublicState:
    position: int
    url: str
    action: str | None
    thought: str | None
    accessibility_tree: str

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "url": self.url,
            "action": self.action,
            "thought": self.thought,
            "accessibility_tree": self.accessibility_tree,
        }


@dataclass(frozen=True, slots=True)
class _PublicTrajectory:
    trajectory_id: str
    domain: str
    goal: str
    start_url: str
    outcome: str | None
    states: tuple[_PublicState, ...]

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "id": self.trajectory_id,
            "domain": self.domain,
            "goal": self.goal,
            "start_url": self.start_url,
            "outcome": self.outcome,
            "states": [state.fingerprint_payload() for state in self.states],
        }


def _parse_trajectory(value: Mapping[str, Any]) -> _PublicTrajectory:
    trajectory_id = _required_public_text(
        value.get("id"),
        "trajectory.id",
        maximum=255,
    )
    domain = _required_public_text(
        value.get("domain"),
        f"trajectory {trajectory_id!r} domain",
        maximum=32,
    )
    if domain not in {"web", "enterprise"}:
        raise LongMemEvalV2AdapterError(
            f"trajectory {trajectory_id!r} domain must be web or enterprise"
        )
    goal = _required_public_text(
        value.get("goal"),
        f"trajectory {trajectory_id!r} goal",
        maximum=_MAX_PUBLIC_FIELD_CHARS,
    )
    start_url = _required_public_text(
        value.get("start_url"),
        f"trajectory {trajectory_id!r} start_url",
        maximum=8_192,
    )
    outcome = _optional_public_text(
        value.get("outcome"),
        f"trajectory {trajectory_id!r} outcome",
        maximum=_MAX_PUBLIC_FIELD_CHARS,
    )
    raw_states = value.get("states")
    if not isinstance(raw_states, list) or not raw_states:
        raise LongMemEvalV2AdapterError(
            f"trajectory {trajectory_id!r} states must be a non-empty list"
        )

    states: list[_PublicState] = []
    for position, raw_state in enumerate(raw_states):
        if not isinstance(raw_state, Mapping):
            raise LongMemEvalV2AdapterError(
                f"trajectory {trajectory_id!r} state {position} must be an object"
            )
        url = _required_public_text(
            raw_state.get("url"),
            f"trajectory {trajectory_id!r} state {position} url",
            maximum=8_192,
        )
        action = _optional_public_text(
            raw_state.get("action"),
            f"trajectory {trajectory_id!r} state {position} action",
            maximum=_MAX_PUBLIC_FIELD_CHARS,
        )
        thought_value = (
            raw_state.get("thought") if "thought" in raw_state else raw_state.get("thoughts")
        )
        thought = _optional_public_text(
            thought_value,
            f"trajectory {trajectory_id!r} state {position} thought",
            maximum=_MAX_PUBLIC_FIELD_CHARS,
        )
        tree_value = (
            raw_state.get("accessibility_tree")
            if "accessibility_tree" in raw_state
            else raw_state.get("text")
        )
        if not isinstance(tree_value, str):
            raise LongMemEvalV2AdapterError(
                f"trajectory {trajectory_id!r} state {position} accessibility tree must be a string"
            )
        states.append(
            _PublicState(
                position=position,
                url=url,
                action=action,
                thought=thought,
                accessibility_tree=tree_value.strip() or "(empty accessibility tree)",
            )
        )
    return _PublicTrajectory(
        trajectory_id=trajectory_id,
        domain=domain,
        goal=goal,
        start_url=start_url,
        outcome=outcome,
        states=tuple(states),
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_text(text: str, limit: int) -> tuple[str, ...]:
    if len(text) <= limit:
        return (text,)
    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            parts.append(remaining.strip())
            break
        cutoff = remaining.rfind("\n", 0, limit + 1)
        if cutoff < limit // 2:
            cutoff = limit
        part = remaining[:cutoff].strip()
        if part:
            parts.append(part)
        remaining = remaining[cutoff:].lstrip()
    return tuple(parts) or ("(empty)",)


def _chunk_document(label: str, body: str, *, limit: int) -> tuple[str, ...]:
    # Reserve enough room for the stable label and segment counters.  IDs are
    # bounded during parsing, so this remains positive for every valid config.
    payload_limit = limit - len(label) - 64
    if payload_limit < 256:
        raise LongMemEvalV2AdapterError(
            "local runtime bridge chunk_chars is too small for trajectory labels"
        )
    parts = _split_text(body, payload_limit)
    total = len(parts)
    documents = tuple(
        f"{label}; segment {index}/{total}\n{part}" for index, part in enumerate(parts, start=1)
    )
    if any(len(document) > limit for document in documents):
        raise LongMemEvalV2AdapterError(
            "local runtime bridge could not satisfy its configured chunk bound"
        )
    return documents


def _trajectory_documents(
    trajectory: _PublicTrajectory,
    *,
    chunk_chars: int,
) -> tuple[tuple[MemoryKind, str, str, dict[str, Any]], ...]:
    outcome = trajectory.outcome or "(outcome not recorded)"
    actions = [
        f"- destination state {state.position}: {state.action or '(no action recorded)'}"
        for state in trajectory.states
    ]
    summary_body = "\n".join(
        (
            "Recorded UI task trajectory summary",
            f"Goal: {trajectory.goal}",
            f"Outcome: {outcome}",
            f"Start URL: {trajectory.start_url}",
            f"Recorded states: {len(trajectory.states)}",
            "Actions are attached to their destination states.",
            "Action sequence:",
            *actions,
        )
    )
    documents: list[tuple[MemoryKind, str, str, dict[str, Any]]] = []
    summary_chunks = _chunk_document(
        f"trajectory {trajectory.trajectory_id} summary",
        summary_body,
        limit=chunk_chars,
    )
    for chunk_position, content in enumerate(summary_chunks):
        documents.append(
            (
                MemoryKind.PROCEDURE,
                f"LongMemEval-V2 trajectory {trajectory.trajectory_id} summary",
                content,
                {
                    "document_type": "trajectory_summary",
                    "state_position": None,
                    "chunk_position": chunk_position,
                    "chunk_count": len(summary_chunks),
                },
            )
        )

    for state in trajectory.states:
        state_body = "\n".join(
            (
                "Recorded UI trajectory state",
                f"Goal: {trajectory.goal}",
                f"Outcome: {outcome}",
                f"Start URL: {trajectory.start_url}",
                f"State position: {state.position}",
                f"Current URL: {state.url}",
                (
                    "Action stored on this destination state: "
                    f"{state.action or '(no action recorded)'}"
                ),
                f"Agent thought: {state.thought or '(no thought recorded)'}",
                "Accessibility tree:",
                state.accessibility_tree,
            )
        )
        state_chunks = _chunk_document(
            f"trajectory {trajectory.trajectory_id} state {state.position}",
            state_body,
            limit=chunk_chars,
        )
        for chunk_position, content in enumerate(state_chunks):
            documents.append(
                (
                    MemoryKind.OBSERVATION,
                    f"LongMemEval-V2 trajectory {trajectory.trajectory_id} state {state.position}",
                    content,
                    {
                        "document_type": "trajectory_state",
                        "state_position": state.position,
                        "chunk_position": chunk_position,
                        "chunk_count": len(state_chunks),
                    },
                )
            )
    return tuple(documents)


class _SteppingClock:
    def __init__(self) -> None:
        self.value = _CLOCK_START

    def __call__(self) -> datetime:
        return self.value

    def step(self) -> None:
        self.value += timedelta(seconds=1)


class _AsyncRuntimeThread:
    """Own an asyncio loop and every runtime object created on that loop."""

    def __init__(self, *, name: str) -> None:
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._startup_error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._main, name=name, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise LongMemEvalV2AdapterError("local Swarm runtime event loop did not start")
        if self._startup_error is not None or self._loop is None:
            raise LongMemEvalV2AdapterError("local Swarm runtime event loop failed to start")

    def _main(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()

    def run(self, awaitable: Awaitable[_T]) -> _T:
        loop = self._loop
        if self._closed or loop is None:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise LongMemEvalV2AdapterError("local Swarm runtime event loop is closed")
        if threading.current_thread() is self._thread:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise LongMemEvalV2AdapterError(
                "synchronous bridge operation cannot run on its owned event-loop thread"
            )
        try:
            future = asyncio.run_coroutine_threadsafe(awaitable, loop)
        except BaseException:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise LongMemEvalV2AdapterError(
                "local Swarm runtime event-loop thread did not stop cleanly"
            )


@dataclass(frozen=True, slots=True)
class _RuntimeState:
    runtime: SwarmBrainRuntime
    actor: ActorContext
    claim: ClaimTaskResult
    task_id: str
    scope_ids: dict[str, str]
    scope_sha256: str
    clock: _SteppingClock


def _scope_identifier(scope_sha256: str, name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{_SCOPE_PROTOCOL}/{scope_sha256}/{name}"))


class LocalRuntimeBridge:
    """Synchronous, question-isolated bridge backed by ``SwarmBrainRuntime``."""

    def __init__(
        self,
        config: AdapterConfig,
        settings: LocalRuntimeBridgeSettings,
        *,
        openai_client: Any | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self._openai_client = openai_client
        self._lock = threading.RLock()
        self._trajectory_ids: set[str] = set()
        self._trajectory_count = 0
        self._memory_ids: list[str] = []
        self._domain: str | None = None
        self._instance_ordinal: int | None = None
        self._host: _AsyncRuntimeThread | None = None
        self._state: _RuntimeState | None = None
        self._closed = False
        self._successful_expansions = 0
        self._embedding_work_completed = 0
        self._recall_completed = False
        self._document_accounting = self._zero_accounting()
        self._query_accounting = self._zero_accounting()

    @property
    def scope_ids(self) -> dict[str, str]:
        """Return content-free deterministic IDs after the first insertion."""

        with self._lock:
            return {} if self._state is None else dict(self._state.scope_ids)

    @property
    def runtime_backend(self) -> str | None:
        with self._lock:
            return None if self._state is None else self._state.runtime.backend.value

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise LongMemEvalV2AdapterError("local Swarm runtime bridge is closed")

    def bind_official_instance_ordinal(self, ordinal: int) -> None:
        """Bind a content-free ordinal allocated by the run-owned lifecycle.

        The runner fixes question order and prompt construction to one worker,
        making this ordinal deterministic without exposing a question ID.
        Direct callers that do not use the official lifecycle derive scope from
        the first trajectory alone.
        """

        with self._lock:
            self._require_open()
            if type(ordinal) is not int or ordinal < 0:
                raise LongMemEvalV2AdapterError(
                    "official bridge instance ordinal must be a non-negative integer"
                )
            if self._instance_ordinal is not None or self._state is not None:
                raise LongMemEvalV2AdapterError(
                    "official bridge instance ordinal may only be bound once before ingestion"
                )
            self._instance_ordinal = ordinal

    def insert_trajectory(self, trajectory: Mapping[str, Any]) -> None:
        with self._lock:
            self._require_open()
            if self._recall_completed or self._successful_expansions:
                raise LongMemEvalV2AdapterError(
                    "local Swarm runtime cannot ingest after query execution starts"
                )
            parsed = _parse_trajectory(trajectory)
            if parsed.trajectory_id in self._trajectory_ids:
                raise LongMemEvalV2AdapterError(
                    f"trajectory {parsed.trajectory_id!r} was inserted more than once"
                )
            if self._domain is not None and parsed.domain != self._domain:
                raise LongMemEvalV2AdapterError(
                    "one local runtime bridge cannot mix web and enterprise trajectories"
                )
            self._domain = parsed.domain
            state = self._state or self._initialize_runtime(parsed)
            assert self._host is not None
            try:
                memory_ids, embedded = self._host.run(
                    self._ingest_trajectory_async(
                        state,
                        parsed,
                        trajectory_position=self._trajectory_count,
                    )
                )
            except BaseException as exc:
                with suppress(BaseException):
                    self.close()
                raise LongMemEvalV2AdapterError("local Swarm trajectory ingestion failed") from exc
            self._trajectory_ids.add(parsed.trajectory_id)
            self._trajectory_count += 1
            self._memory_ids.extend(memory_ids)
            self._embedding_work_completed += embedded

    def _scope_payload(self, first_trajectory: _PublicTrajectory) -> dict[str, Any]:
        return {
            "scope_protocol": _SCOPE_PROTOCOL,
            "adapter_revision": ADAPTER_REVISION,
            "tier": self.config.tier,
            "operating_point": self.config.operating_point,
            "dataset_revision": self.config.dataset_revision,
            "dataset_manifest_sha256": self.config.dataset_manifest_sha256,
            "bridge_settings": {
                "retrieval_mode": self.settings.retrieval_mode,
                "chunk_chars": self.settings.chunk_chars,
                "embedding_dimensions": self.settings.embedding_dimensions,
                "dense_min_similarity": self.settings.dense_min_similarity,
                "recall_min_score": self.settings.recall_min_score,
                "embedding_model": self.settings.embedding_model,
                "embedding_model_revision": self.settings.embedding_model_revision,
                "embedding_response_model": self.settings.embedding_response_model,
                "query_instruction_sha256": (
                    SOTA_EMBEDDING_QUERY_INSTRUCTION_SHA256
                    if self.settings.retrieval_mode == "openai_hybrid"
                    else None
                ),
            },
            "instance_ordinal": self._instance_ordinal,
            "domain": first_trajectory.domain,
            "first_trajectory_sha256": _canonical_sha256(first_trajectory.fingerprint_payload()),
        }

    def _initialize_runtime(self, first_trajectory: _PublicTrajectory) -> _RuntimeState:
        self._require_open()
        if self._state is not None:
            return self._state
        scope_sha256 = _canonical_sha256(self._scope_payload(first_trajectory))
        host = _AsyncRuntimeThread(name=f"lme-v2-runtime-{scope_sha256[:12]}")
        self._host = host
        try:
            state = host.run(self._initialize_async(scope_sha256))
        except BaseException as exc:
            self._closed = True
            with suppress(BaseException):
                host.close()
            raise LongMemEvalV2AdapterError("local Swarm runtime initialization failed") from exc
        self._state = state
        return state

    def _require_initialized(self) -> _RuntimeState:
        self._require_open()
        if self._state is None or not self._memory_ids:
            raise LongMemEvalV2AdapterError(
                "local Swarm runtime requires at least one inserted trajectory before recall"
            )
        return self._state

    async def _initialize_async(self, scope_sha256: str) -> _RuntimeState:
        clock = _SteppingClock()
        embeddings: Any | None
        if self.settings.retrieval_mode == "deterministic_hybrid":
            embeddings = DeterministicEmbeddingProvider(
                dimensions=self.settings.embedding_dimensions,
                model_name=(
                    f"longmemeval-v2-deterministic-{self.settings.embedding_dimensions}-v1"
                ),
            )
        elif self.settings.retrieval_mode == "openai_hybrid":
            assert self.settings.embedding_base_url is not None
            assert self.settings.embedding_api_key_env is not None
            assert self.settings.embedding_model is not None
            assert self.settings.embedding_response_model is not None
            api_key = os.getenv(self.settings.embedding_api_key_env, "").strip()
            if not api_key:
                raise _configuration_error(
                    "embedding_api_key_env resolves to an unset or empty variable"
                )
            embeddings = OpenAICompatibleEmbeddingProvider(
                base_url=self.settings.embedding_base_url,
                model_id=self.settings.embedding_model,
                dimensions=self.settings.embedding_dimensions,
                api_key=api_key,
                required_response_model=self.settings.embedding_response_model,
                query_instruction=SOTA_EMBEDDING_QUERY_INSTRUCTION,
                client=self._openai_client,
            )
        else:
            embeddings = None
        runtime = build_in_memory_runtime(
            secrets.token_urlsafe(32),
            embeddings=embeddings,
            dense_min_similarity=self.settings.dense_min_similarity,
            clock=clock,
        )
        try:
            await runtime.start()
            scope_ids = {
                name: _scope_identifier(scope_sha256, name)
                for name in (
                    "tenant_id",
                    "project_id",
                    "repository_id",
                    "swarm_id",
                    "run_id",
                )
            }
            actor = ActorContext(
                **scope_ids,
                agent_id=_scope_identifier(scope_sha256, "agent_id"),
                harness="longmemeval-v2-official",
                harness_version=ADAPTER_REVISION,
                provider="local",
                model="none",
                capabilities=_RUNTIME_CAPABILITIES,
                metadata={
                    "benchmark": "LongMemEval-V2",
                    "scope_protocol": _SCOPE_PROTOCOL,
                    "scope_sha256": scope_sha256,
                    "instance_ordinal": self._instance_ordinal,
                    "domain": self._domain,
                },
            )
            await runtime.coordination.join(actor)
            task_id = _scope_identifier(scope_sha256, "task_id")
            task = Task(
                **scope_ids,
                task_id=task_id,
                title="LongMemEval-V2 isolated memory retrieval",
                description=(
                    "Ingest one ordered public trajectory haystack and execute "
                    "lease-bound recall/read-expand."
                ),
                status=TaskStatus.READY,
                tags=("longmemeval-v2", "benchmark"),
                created_by_agent_id=actor.agent_id,
                created_at=clock(),
                updated_at=clock(),
                metadata={
                    "benchmark": "LongMemEval-V2",
                    "scope_sha256": scope_sha256,
                    "instance_ordinal": self._instance_ordinal,
                    "domain": self._domain,
                },
            )
            await runtime.coordination.add_task(task)
            # Claim with least privilege so task bootstrap cannot perform an
            # untraced recall.  The same authenticated identity receives its
            # MEMORY_RECALL capability only for the two measured operations.
            claim_actor = actor.model_copy(
                update={
                    "capabilities": frozenset(
                        capability
                        for capability in actor.capabilities
                        if capability != Capability.MEMORY_RECALL.value
                    )
                }
            )
            claim = await runtime.coordination.claim(
                claim_actor,
                ClaimTaskCommand(
                    idempotency_key=f"lme-v2-claim:{scope_sha256[:32]}",
                    task_id=task_id,
                    lease_seconds=3_600,
                ),
            )
            return _RuntimeState(
                runtime=runtime,
                actor=actor,
                claim=claim,
                task_id=task_id,
                scope_ids={**scope_ids, "agent_id": actor.agent_id, "task_id": task_id},
                scope_sha256=scope_sha256,
                clock=clock,
            )
        except BaseException:
            await runtime.close()
            raise

    async def _ingest_trajectory_async(
        self,
        state: _RuntimeState,
        trajectory: _PublicTrajectory,
        *,
        trajectory_position: int,
    ) -> tuple[tuple[str, ...], int]:
        documents = _trajectory_documents(
            trajectory,
            chunk_chars=self.settings.chunk_chars,
        )
        memory_ids: list[str] = []
        previous_memory_id: str | None = None
        for document_offset, (kind, title, content, document_metadata) in enumerate(documents):
            state.clock.step()
            document_position = len(self._memory_ids) + document_offset
            result = await state.runtime.memory.publish(
                state.actor,
                RememberCommand(
                    idempotency_key=(f"lme-v2:{state.scope_sha256[:32]}:{document_position:06d}"),
                    kind=kind,
                    content=content,
                    desired_state=MemoryState.CONFIRMED,
                    visibility=Visibility.TASK,
                    task_id=state.task_id,
                    title=title,
                    tags=("longmemeval-v2", trajectory.domain, "ui-trajectory"),
                    confidence=1.0,
                    related_memory_ids=(
                        (previous_memory_id,) if previous_memory_id is not None else ()
                    ),
                    metadata={
                        "benchmark": "LongMemEval-V2",
                        "domain_lane": "execution_history",
                        "trajectory_id": trajectory.trajectory_id,
                        "trajectory_position": trajectory_position,
                        **document_metadata,
                    },
                ),
            )
            if result.memory is None:
                raise RuntimeError("canonical memory publish returned no memory")
            previous_memory_id = result.memory.memory_id
            memory_ids.append(result.memory.memory_id)

        if not memory_ids:
            raise RuntimeError("trajectory ingestion produced no canonical memories")
        embedded = 0
        accounting_before = self._provider_accounting(state)
        if state.runtime.embeddings is not None:
            worker = state.runtime.embedding_worker(retry_delay_seconds=1)
            while embedded < len(memory_ids):
                completed = await worker.run_once(
                    f"lme-v2-embed-{state.scope_sha256[:16]}",
                    limit=min(100, len(memory_ids) - embedded),
                    lease_seconds=3_600,
                )
                if not completed:
                    raise RuntimeError("embedding worker did not drain")
                embedded += len(completed)
            if embedded != len(memory_ids):
                raise RuntimeError("embedding coverage differs from memory count")
        accounting_after = self._provider_accounting(state)
        if self.settings.retrieval_mode == "openai_hybrid":
            delta = self._accounting_delta(accounting_before, accounting_after)
            expected = len(memory_ids)
            if (
                delta["document_inputs"] != expected
                or delta["document_batch_calls"] != expected
                or delta["query_calls"] != 0
                or delta["successful_http_calls"] != expected
                or delta["http_attempts"] < expected
            ):
                raise RuntimeError(
                    "provider document embedding accounting does not reconcile with memories"
                )
            self._document_accounting = self._accounting_add(self._document_accounting, delta)
        return tuple(memory_ids), embedded

    def recall_memory(self, query: str, *, limit: int) -> RecallMemoryResult:
        if not isinstance(query, str) or not query.strip():
            raise LongMemEvalV2AdapterError("recall_memory query must be a non-empty string")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise LongMemEvalV2AdapterError("recall_memory limit must be an integer in [1, 100]")
        with self._lock:
            state = self._require_initialized()
            if self._recall_completed:
                raise LongMemEvalV2AdapterError(
                    "local Swarm runtime allows exactly one recall per question scope"
                )
            assert self._host is not None
            accounting_before = self._provider_accounting(state)
            try:
                bundle = self._host.run(
                    state.runtime.memory.recall(
                        state.actor,
                        RecallQuery(
                            text=query.strip(),
                            task_id=state.task_id,
                            visibilities=frozenset({Visibility.TASK}),
                            states=frozenset({MemoryState.CONFIRMED}),
                            include_evidence=False,
                            include_lineage=False,
                            min_score=self.settings.recall_min_score,
                            limit=limit,
                        ),
                    )
                )
            except BaseException as exc:
                raise LongMemEvalV2AdapterError("local Swarm recall_memory failed") from exc
            accounting_after = self._provider_accounting(state)
            if self.settings.retrieval_mode == "openai_hybrid":
                delta = self._accounting_delta(accounting_before, accounting_after)
                if (
                    delta["document_inputs"] != 0
                    or delta["document_batch_calls"] != 0
                    or delta["query_calls"] != 1
                    or delta["successful_http_calls"] != 1
                    or delta["http_attempts"] < 1
                ):
                    raise LongMemEvalV2AdapterError(
                        "OpenAI query embedding did not complete exactly once; "
                        "lexical fallback is forbidden for SOTA runs"
                    )
                self._query_accounting = delta
            self._recall_completed = True
            return RecallMemoryResult(tuple(hit.memory.memory_id for hit in bundle.hits))

    def read_expand_memory(
        self,
        query: str,
        *,
        memory_ids: tuple[str, ...],
        max_depth: int,
        max_fanout: int,
        token_budget: int,
    ) -> ReadExpandMemoryResult:
        if not isinstance(query, str) or not query.strip():
            raise LongMemEvalV2AdapterError("read_expand_memory query must be a non-empty string")
        with self._lock:
            state = self._require_initialized()
            if not self._recall_completed:
                raise LongMemEvalV2AdapterError(
                    "read_expand_memory requires the preceding successful recall_memory"
                )
            if self._successful_expansions:
                raise LongMemEvalV2AdapterError(
                    "local Swarm runtime allows exactly one read-expand per question scope"
                )
            assert self._host is not None
            accounting_before = self._provider_accounting(state)
            try:
                expanded = self._host.run(
                    state.runtime.coordination.read_expand_memory(
                        state.actor,
                        ReadExpandMemoryRequest(
                            task_id=state.task_id,
                            lease_id=state.claim.lease.lease_id,
                            query_text=query.strip(),
                            memory_ids=memory_ids,
                            max_depth=max_depth,
                            max_fanout=max_fanout,
                            token_budget=token_budget,
                            include_evidence=False,
                        ),
                    )
                )
            except BaseException as exc:
                raise LongMemEvalV2AdapterError("local Swarm read_expand_memory failed") from exc
            accounting_after = self._provider_accounting(state)
            if accounting_after != accounting_before:
                raise LongMemEvalV2AdapterError(
                    "read_expand_memory unexpectedly changed embedding provider accounting"
                )
            self._successful_expansions += 1
            return ReadExpandMemoryResult(expanded.memory_ids, expanded.context)

    @staticmethod
    def _zero_accounting() -> dict[str, int]:
        return {
            "document_inputs": 0,
            "document_batch_calls": 0,
            "query_calls": 0,
            "successful_http_calls": 0,
            "http_attempts": 0,
        }

    @classmethod
    def _provider_accounting(cls, state: _RuntimeState) -> dict[str, int]:
        provider = state.runtime.embeddings
        if not isinstance(provider, OpenAICompatibleEmbeddingProvider):
            return cls._zero_accounting()
        raw = provider.call_accounting
        if set(raw) != set(cls._zero_accounting()) or any(
            type(value) is not int or value < 0 for value in raw.values()
        ):
            raise RuntimeError("embedding provider returned malformed call accounting")
        return dict(raw)

    @staticmethod
    def _accounting_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
        delta = {name: after[name] - before[name] for name in before}
        if any(value < 0 for value in delta.values()):
            raise RuntimeError("embedding provider call accounting moved backwards")
        return delta

    @staticmethod
    def _accounting_add(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, int]:
        return {name: left[name] + right[name] for name in left}

    def embedding_evidence(self) -> EmbeddingRuntimeEvidence:
        """Return fail-closed provider proof for the completed question scope."""

        with self._lock:
            state = self._require_initialized()
            if not self._recall_completed or self._successful_expansions != 1:
                raise LongMemEvalV2AdapterError(
                    "embedding evidence requires one completed recall/read-expand query"
                )
            inserted = len(self._memory_ids)
            if self.settings.retrieval_mode == "openai_hybrid":
                provider = state.runtime.embeddings
                if not isinstance(provider, OpenAICompatibleEmbeddingProvider):
                    raise LongMemEvalV2AdapterError(
                        "openai_hybrid runtime does not own the hardened provider"
                    )
                if (
                    provider.model_name != SOTA_EMBEDDING_MODEL
                    or provider.dimensions != self.settings.embedding_dimensions
                    or provider.required_response_model != SOTA_EMBEDDING_MODEL
                ):
                    raise LongMemEvalV2AdapterError(
                        "openai_hybrid provider identity differs from the pinned protocol"
                    )
                total = provider.call_accounting
                expected_total = self._accounting_add(
                    self._document_accounting, self._query_accounting
                )
                if total != expected_total:
                    raise LongMemEvalV2AdapterError(
                        "embedding provider totals differ from document/query accounting"
                    )
                return EmbeddingRuntimeEvidence(
                    retrieval_mode="openai_hybrid",
                    sota_capable=True,
                    provider=SOTA_EMBEDDING_PROVIDER,
                    model=SOTA_EMBEDDING_MODEL,
                    model_revision=self.settings.embedding_model_revision,
                    dimensions=self.settings.embedding_dimensions,
                    response_model_requirement=SOTA_EMBEDDING_MODEL,
                    query_instruction_sha256=SOTA_EMBEDDING_QUERY_INSTRUCTION_SHA256,
                    inserted_memories=inserted,
                    embedding_work_completed=self._embedding_work_completed,
                    call_accounting_source="provider-observed",
                    document_inputs=self._document_accounting["document_inputs"],
                    document_batch_calls=self._document_accounting["document_batch_calls"],
                    document_successful_http_calls=self._document_accounting[
                        "successful_http_calls"
                    ],
                    document_http_attempts=self._document_accounting["http_attempts"],
                    query_calls=self._query_accounting["query_calls"],
                    query_successful_http_calls=self._query_accounting["successful_http_calls"],
                    query_http_attempts=self._query_accounting["http_attempts"],
                    exact_response_model_verified=True,
                    deterministic_fallback_used=False,
                )

            deterministic = self.settings.retrieval_mode == "deterministic_hybrid"
            return EmbeddingRuntimeEvidence(
                retrieval_mode=self.settings.retrieval_mode,
                sota_capable=False,
                provider=("DeterministicEmbeddingProvider" if deterministic else None),
                model=(
                    f"longmemeval-v2-deterministic-{self.settings.embedding_dimensions}-v1"
                    if deterministic
                    else None
                ),
                model_revision=None,
                dimensions=self.settings.embedding_dimensions if deterministic else None,
                response_model_requirement=None,
                query_instruction_sha256=None,
                inserted_memories=inserted,
                embedding_work_completed=self._embedding_work_completed,
                call_accounting_source="bridge-observed-development-mode",
                document_inputs=inserted if deterministic else 0,
                document_batch_calls=inserted if deterministic else 0,
                document_successful_http_calls=0,
                document_http_attempts=0,
                query_calls=1 if deterministic else 0,
                query_successful_http_calls=0,
                query_http_attempts=0,
                exact_response_model_verified=False,
                deterministic_fallback_used=False,
            )

    async def _shutdown_async(self, state: _RuntimeState) -> None:
        lifecycle_error: BaseException | None = None
        outcome = TaskOutcome.SUCCEEDED if self._successful_expansions else TaskOutcome.FAILED
        try:
            await state.runtime.coordination.complete(
                state.actor,
                CompleteTaskCommand(
                    idempotency_key=f"lme-v2-complete:{state.scope_sha256[:32]}",
                    task_id=state.task_id,
                    lease_id=state.claim.lease.lease_id,
                    expected_task_version=state.claim.task.version,
                    expected_lease_version=state.claim.lease.version,
                    outcome=outcome,
                    summary=(
                        "LongMemEval-V2 bridge completed canonical read-expand"
                        if self._successful_expansions
                        else "LongMemEval-V2 bridge closed before canonical read-expand completed"
                    ),
                ),
            )
        except BaseException as exc:
            lifecycle_error = exc
            with suppress(BaseException):
                await state.runtime.coordination.release(
                    state.actor,
                    ReleaseTaskCommand(
                        idempotency_key=f"lme-v2-release:{state.scope_sha256[:32]}",
                        task_id=state.task_id,
                        lease_id=state.claim.lease.lease_id,
                        expected_task_version=state.claim.task.version,
                        expected_lease_version=state.claim.lease.version,
                        reason="LongMemEval-V2 bridge cleanup fallback",
                    ),
                )
        try:
            await state.runtime.close()
        except BaseException as exc:
            lifecycle_error = lifecycle_error or exc
        provider_close = getattr(state.runtime.embeddings, "close", None)
        if callable(provider_close):
            try:
                result = provider_close()
                if isinstance(result, Awaitable):
                    await result
            except BaseException as exc:
                lifecycle_error = lifecycle_error or exc
        if lifecycle_error is not None:
            raise RuntimeError("canonical runtime cleanup failed") from lifecycle_error

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            state = self._state
            host = self._host
            cleanup_error: BaseException | None = None
            if state is not None and host is not None:
                try:
                    host.run(self._shutdown_async(state))
                except BaseException as exc:
                    cleanup_error = exc
            if host is not None:
                try:
                    host.close()
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
            self._state = None
            self._host = None
            if cleanup_error is not None:
                raise LongMemEvalV2AdapterError(
                    "local Swarm runtime bridge did not close cleanly"
                ) from cleanup_error


def build_local_runtime_bridge(
    config: AdapterConfig, *, openai_client: Any | None = None
) -> LocalRuntimeBridge:
    """Build the bundled local-state bridge for an official memory instance.

    ``openai_client`` is an injection seam for protocol tests.  The official
    harness calls the factory with only ``config`` and therefore constructs the
    hardened provider's own transport.
    """

    if not isinstance(config, AdapterConfig):
        raise LongMemEvalV2AdapterError("local runtime bridge factory requires AdapterConfig")
    settings = LocalRuntimeBridgeSettings.from_adapter_config(config)
    if openai_client is not None and settings.retrieval_mode != "openai_hybrid":
        raise LongMemEvalV2AdapterError(
            "openai_client is only valid with retrieval_mode 'openai_hybrid'"
        )
    return LocalRuntimeBridge(config, settings, openai_client=openai_client)


__all__ = [
    "LOCAL_RUNTIME_BRIDGE_FACTORY",
    "LocalRuntimeBridge",
    "LocalRuntimeBridgeSettings",
    "build_local_runtime_bridge",
]
