"""Narrow Swarm Brain gateways used by the GateMem external harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

import httpx

from swarmbrain.application.runtime import SwarmBrainRuntime
from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.memory import MemoryState, RecallQuery, RememberCommand, Visibility

from .contracts import GateMemContractError, PrincipalScope


@dataclass(frozen=True, slots=True)
class MemoryWrite:
    idempotency_key: str
    content: Any
    title: str
    tags: tuple[str, ...]
    metadata: dict[str, Any]
    supersedes_memory_id: str | None = None


@dataclass(frozen=True, slots=True)
class PublishedMemory:
    memory_id: str
    version: int
    content: Any
    metadata: dict[str, Any]
    state: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class RecallRequest:
    text: str
    limit: int
    min_score: float = 0.0


@dataclass(frozen=True, slots=True)
class RecalledMemory:
    memory_id: str
    version: int
    content: Any
    metadata: dict[str, Any]
    state: str
    score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecallResponse:
    memories: tuple[RecalledMemory, ...]
    total_candidates: int
    truncated: bool
    latency_ms: float


class MemoryGateway(Protocol):
    """The only persistence operations the GateMem harness can perform."""

    async def publish(self, scope: PrincipalScope, write: MemoryWrite) -> PublishedMemory: ...

    async def recall(self, scope: PrincipalScope, request: RecallRequest) -> RecallResponse: ...


def _actor(scope: PrincipalScope) -> ActorContext:
    return ActorContext(
        tenant_id=scope.tenant_id,
        project_id=scope.project_id,
        repository_id=scope.repository_id,
        swarm_id=scope.swarm_id,
        run_id=scope.run_id,
        agent_id=scope.agent_id,
        harness="gatemem-external",
        harness_version="1",
        provider="benchmark",
        model="none",
        capabilities=frozenset({Capability.MEMORY_PUBLISH.value, Capability.MEMORY_RECALL.value}),
        metadata={
            "benchmark": "GateMem",
            "domain": scope.domain,
            "episode_id": scope.episode_id,
            "principal_id": scope.principal_id,
            "principal_role": scope.principal_role,
        },
    )


class RuntimeMemoryGateway:
    """Call the canonical application service without bypassing actor fencing."""

    def __init__(self, runtime: SwarmBrainRuntime) -> None:
        self.runtime = runtime

    async def publish(self, scope: PrincipalScope, write: MemoryWrite) -> PublishedMemory:
        started = perf_counter()
        result = await self.runtime.memory.publish(
            _actor(scope),
            RememberCommand(
                idempotency_key=write.idempotency_key,
                kind="observation",
                content=write.content,
                desired_state=MemoryState.TENTATIVE,
                visibility=Visibility.RUN,
                title=write.title,
                tags=write.tags,
                confidence=1.0,
                supersedes_memory_id=write.supersedes_memory_id,
                metadata=write.metadata,
            ),
        )
        latency_ms = (perf_counter() - started) * 1000.0
        if result.memory is None:
            raise GateMemContractError(
                f"Swarm Brain publish returned {result.operation.value!r} without a memory"
            )
        _validate_domain_scope(scope, result.memory.model_dump(mode="json"))
        return PublishedMemory(
            memory_id=result.memory.memory_id,
            version=result.memory.version,
            content=result.memory.content,
            metadata=dict(result.memory.metadata),
            state=result.memory.state.value,
            latency_ms=latency_ms,
        )

    async def recall(self, scope: PrincipalScope, request: RecallRequest) -> RecallResponse:
        started = perf_counter()
        bundle = await self.runtime.memory.recall(
            _actor(scope),
            RecallQuery(
                text=request.text,
                visibilities=frozenset({Visibility.RUN}),
                states=frozenset({MemoryState.TENTATIVE, MemoryState.CONFIRMED}),
                min_score=request.min_score,
                limit=request.limit,
                include_evidence=False,
                include_lineage=False,
            ),
        )
        latency_ms = (perf_counter() - started) * 1000.0
        memories: list[RecalledMemory] = []
        for hit in bundle.hits:
            _validate_domain_scope(scope, hit.memory.model_dump(mode="json"))
            memories.append(
                RecalledMemory(
                    memory_id=hit.memory.memory_id,
                    version=hit.memory.version,
                    content=hit.memory.content,
                    metadata=dict(hit.memory.metadata),
                    state=hit.memory.state.value,
                    score=hit.score,
                    reasons=tuple(hit.reasons),
                )
            )
        return RecallResponse(
            memories=tuple(memories),
            total_candidates=bundle.total_candidates,
            truncated=bundle.truncated,
            latency_ms=latency_ms,
        )


class PrincipalTokenProvider(Protocol):
    async def token_for(self, scope: PrincipalScope) -> str: ...


@dataclass(slots=True)
class StaticTokenProvider:
    """Tokens keyed by ``episode_id::principal_id``; never serialized to output."""

    tokens: dict[str, str]

    async def token_for(self, scope: PrincipalScope) -> str:
        token = self.tokens.get(scope.key)
        if not token:
            raise GateMemContractError(
                f"no bearer token configured for principal scope {scope.key}"
            )
        return token


@dataclass(slots=True)
class HttpMemoryGateway:
    """Use only the authenticated public ``/v1/memories`` API."""

    base_url: str
    tokens: PrincipalTokenProvider
    client: httpx.AsyncClient = field(default_factory=httpx.AsyncClient)
    _owns_client: bool = True

    @classmethod
    def with_client(
        cls,
        *,
        base_url: str,
        tokens: PrincipalTokenProvider,
        client: httpx.AsyncClient,
    ) -> HttpMemoryGateway:
        return cls(base_url=base_url, tokens=tokens, client=client, _owns_client=False)

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def publish(self, scope: PrincipalScope, write: MemoryWrite) -> PublishedMemory:
        token = await self.tokens.token_for(scope)
        body: dict[str, Any] = {
            "kind": "observation",
            "content": write.content,
            "desired_state": "tentative",
            "visibility": "run",
            "title": write.title,
            "tags": list(write.tags),
            "confidence": 1.0,
            "metadata": write.metadata,
        }
        if write.supersedes_memory_id is not None:
            body["supersedes_memory_id"] = write.supersedes_memory_id
        started = perf_counter()
        response = await self.client.post(
            f"{self.base_url.rstrip('/')}/v1/memories",
            headers={
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": write.idempotency_key,
            },
            json=body,
        )
        latency_ms = (perf_counter() - started) * 1000.0
        payload = _response_json(response, operation="publish")
        memory = payload.get("memory")
        if not isinstance(memory, dict):
            raise GateMemContractError("Swarm Brain publish response is missing memory")
        _validate_domain_scope(scope, memory)
        return PublishedMemory(
            memory_id=_response_text(memory, "memory_id"),
            version=_response_int(memory, "version"),
            content=memory.get("content"),
            metadata=_response_object(memory, "metadata"),
            state=_response_text(memory, "state"),
            latency_ms=latency_ms,
        )

    async def recall(self, scope: PrincipalScope, request: RecallRequest) -> RecallResponse:
        token = await self.tokens.token_for(scope)
        started = perf_counter()
        response = await self.client.post(
            f"{self.base_url.rstrip('/')}/v1/memories:recall",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "text": request.text,
                "visibilities": ["run"],
                "states": ["tentative", "confirmed"],
                "min_score": request.min_score,
                "limit": request.limit,
                "include_evidence": False,
                "include_lineage": False,
            },
        )
        latency_ms = (perf_counter() - started) * 1000.0
        payload = _response_json(response, operation="recall")
        hits = payload.get("hits")
        if not isinstance(hits, list):
            raise GateMemContractError("Swarm Brain recall response hits must be a list")
        recalled: list[RecalledMemory] = []
        for hit in hits:
            if not isinstance(hit, dict) or not isinstance(hit.get("memory"), dict):
                raise GateMemContractError("Swarm Brain recall hit is malformed")
            memory = hit["memory"]
            _validate_domain_scope(scope, memory)
            reasons = hit.get("reasons") or []
            if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
                raise GateMemContractError("Swarm Brain recall reasons must be strings")
            score = hit.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise GateMemContractError("Swarm Brain recall score must be numeric")
            recalled.append(
                RecalledMemory(
                    memory_id=_response_text(memory, "memory_id"),
                    version=_response_int(memory, "version"),
                    content=memory.get("content"),
                    metadata=_response_object(memory, "metadata"),
                    state=_response_text(memory, "state"),
                    score=float(score),
                    reasons=tuple(reasons),
                )
            )
        total_candidates = payload.get("total_candidates", 0)
        truncated = payload.get("truncated", False)
        if not isinstance(total_candidates, int) or isinstance(total_candidates, bool):
            raise GateMemContractError("Swarm Brain total_candidates must be an integer")
        if not isinstance(truncated, bool):
            raise GateMemContractError("Swarm Brain truncated must be boolean")
        return RecallResponse(
            memories=tuple(recalled),
            total_candidates=total_candidates,
            truncated=truncated,
            latency_ms=latency_ms,
        )


def _response_json(response: httpx.Response, *, operation: str) -> dict[str, Any]:
    if response.status_code >= 400:
        raise GateMemContractError(
            f"Swarm Brain {operation} failed with HTTP {response.status_code}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise GateMemContractError(f"Swarm Brain {operation} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise GateMemContractError(f"Swarm Brain {operation} response must be an object")
    return payload


def _response_text(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise GateMemContractError(f"Swarm Brain response {field_name} must be text")
    return value


def _response_int(payload: dict[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise GateMemContractError(f"Swarm Brain response {field_name} must be a positive integer")
    return value


def _response_object(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name) or {}
    if not isinstance(value, dict):
        raise GateMemContractError(f"Swarm Brain response {field_name} must be an object")
    return dict(value)


def _validate_domain_scope(scope: PrincipalScope, memory: dict[str, Any]) -> None:
    expected = {
        "tenant_id": scope.tenant_id,
        "project_id": scope.project_id,
        "repository_id": scope.repository_id,
        "swarm_id": scope.swarm_id,
        "run_id": scope.run_id,
        "author_agent_id": scope.agent_id,
    }
    mismatches = [field for field, value in expected.items() if memory.get(field) != value]
    if mismatches:
        raise GateMemContractError(
            f"Swarm Brain returned memory outside principal scope: {sorted(mismatches)}"
        )


__all__ = [
    "HttpMemoryGateway",
    "MemoryGateway",
    "MemoryWrite",
    "PrincipalTokenProvider",
    "PublishedMemory",
    "RecallRequest",
    "RecallResponse",
    "RecalledMemory",
    "RuntimeMemoryGateway",
    "StaticTokenProvider",
]
