"""Composition root shared by HTTP and local tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from swarmbrain.adapters.auth import RunTokenCodec
from swarmbrain.adapters.embeddings import DeterministicEmbeddingProvider
from swarmbrain.adapters.extraction.in_memory import InMemoryWorkStore
from swarmbrain.adapters.memory import InMemoryKernel
from swarmbrain.config import ApiSettings, BackendKind, EmbeddingsKind
from swarmbrain.ports.embeddings import EmbeddingIndex, EmbeddingProvider
from swarmbrain.ports.work_queue import WorkQueueStore
from swarmbrain.workers.durable import LeasedWorkWorker
from swarmbrain.workers.embedding import EmbedMemoryHandler

from .conflict_service import ConflictService
from .coordination import CoordinationService
from .evidence_service import EvidenceService
from .memory_policy import ConservativeMemoryPolicy
from .memory_service import MemoryService
from .work import DurableWorkService


class RuntimeLifecycle(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def health(self) -> None: ...


class _CockroachResource(Protocol):
    async def start(self, *, verify_schema: bool = True) -> None: ...

    async def close(self) -> None: ...

    async def health(self) -> None: ...


class _InMemoryLifecycle:
    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def health(self) -> None:
        return None


class _CockroachLifecycle:
    def __init__(self, database: _CockroachResource) -> None:
        self._database = database

    async def start(self) -> None:
        await self._database.start(verify_schema=True)

    async def close(self) -> None:
        await self._database.close()

    async def health(self) -> None:
        await self._database.health()


@dataclass(frozen=True, slots=True)
class SwarmBrainRuntime:
    backend: BackendKind
    coordination: CoordinationService
    memory: MemoryService
    conflicts: ConflictService
    evidence: EvidenceService
    tokens: RunTokenCodec
    kernel: InMemoryKernel | None = None
    work: DurableWorkService | None = None
    work_queue: WorkQueueStore | None = None
    embeddings: EmbeddingProvider | None = None
    embedding_index: EmbeddingIndex | None = None
    _lifecycle: RuntimeLifecycle = field(default_factory=_InMemoryLifecycle, repr=False)

    async def start(self) -> None:
        """Open runtime resources and verify prerequisites without applying DDL."""

        await self._lifecycle.start()

    async def close(self) -> None:
        await self._lifecycle.close()

    async def ready(self) -> bool:
        try:
            await self._lifecycle.health()
        except Exception:
            return False
        return True

    def embedding_worker(self, *, retry_delay_seconds: int = 30) -> LeasedWorkWorker:
        """Compose the leased worker that turns EMBED_MEMORY items into vectors."""

        if self.embeddings is None or self.embedding_index is None or self.work_queue is None:
            raise RuntimeError("embeddings are not configured for this runtime")
        return LeasedWorkWorker(
            self.work_queue,
            [EmbedMemoryHandler(self.embeddings, self.embedding_index)],
            retry_delay_seconds=retry_delay_seconds,
        )


def build_in_memory_runtime(
    token_secret: str,
    *,
    embeddings: EmbeddingProvider | None = None,
) -> SwarmBrainRuntime:
    kernel = InMemoryKernel()
    policy = ConservativeMemoryPolicy()
    work_queue = InMemoryWorkStore()
    work = DurableWorkService(work_queue)
    embedding_index = kernel if embeddings is not None else None
    memory = MemoryService(
        kernel,
        policy,
        review_store=kernel,
        embeddings=embeddings,
        embedding_index=embedding_index,
        work=work if embeddings is not None else None,
    )
    coordination = CoordinationService(kernel, memory_service=memory)
    return SwarmBrainRuntime(
        backend=BackendKind.MEMORY,
        coordination=coordination,
        memory=memory,
        conflicts=ConflictService(kernel),
        evidence=EvidenceService(kernel),
        tokens=RunTokenCodec(token_secret),
        kernel=kernel,
        work=work,
        work_queue=work_queue,
        embeddings=embeddings,
        embedding_index=embedding_index,
    )


def build_runtime(settings: ApiSettings) -> SwarmBrainRuntime:
    """Compose the selected backend without opening network resources."""

    if settings.backend is BackendKind.MEMORY:
        return build_in_memory_runtime(
            settings.token_secret,
            embeddings=_build_embedding_provider(settings),
        )
    return _build_cockroach_runtime(settings)


def _build_embedding_provider(settings: ApiSettings) -> EmbeddingProvider | None:
    if settings.embeddings is EmbeddingsKind.NONE:
        return None
    if settings.embeddings is EmbeddingsKind.DETERMINISTIC:
        return DeterministicEmbeddingProvider(
            dimensions=settings.embeddings_dimensions,
            model_name=settings.embeddings_model or "deterministic-v0",
        )
    # Lazy so the optional AWS SDK is only touched when Bedrock is selected.
    from swarmbrain.adapters.embeddings.bedrock import BedrockEmbeddingProvider

    return BedrockEmbeddingProvider(
        model_id=settings.embeddings_model or "amazon.titan-embed-text-v2:0",
        dimensions=settings.embeddings_dimensions,
        region_name=settings.aws_region,
    )


def _build_cockroach_runtime(settings: ApiSettings) -> SwarmBrainRuntime:
    # Keep the optional driver and durable repositories out of memory-only imports.
    from swarmbrain.adapters.cockroach.coordination import CockroachCoordinationStore
    from swarmbrain.adapters.cockroach.database import CockroachDatabase
    from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
    from swarmbrain.adapters.cockroach.work_store import CockroachWorkStore

    assert settings.database_url is not None
    database = CockroachDatabase(
        settings.database_url,
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
    )
    coordination_store = CockroachCoordinationStore(database)
    memory_store = CockroachMemoryStore(database)
    policy = ConservativeMemoryPolicy()
    work_queue = CockroachWorkStore(database)
    work = DurableWorkService(work_queue)
    embeddings = _build_embedding_provider(settings)
    embedding_index = memory_store if embeddings is not None else None
    memory = MemoryService(
        memory_store,
        policy,
        review_store=memory_store,
        embeddings=embeddings,
        embedding_index=embedding_index,
        work=work if embeddings is not None else None,
    )
    coordination = CoordinationService(coordination_store, memory_service=memory)
    return SwarmBrainRuntime(
        backend=BackendKind.COCKROACH,
        coordination=coordination,
        memory=memory,
        conflicts=ConflictService(memory_store),
        evidence=EvidenceService(memory_store),
        tokens=RunTokenCodec(settings.token_secret),
        work=work,
        work_queue=work_queue,
        embeddings=embeddings,
        embedding_index=embedding_index,
        _lifecycle=_CockroachLifecycle(database),
    )


__all__ = ["SwarmBrainRuntime", "build_in_memory_runtime", "build_runtime"]
