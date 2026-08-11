"""Composition root shared by HTTP and local tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from swarmbrain.adapters.auth import RunTokenCodec
from swarmbrain.adapters.consolidation import SafeDeterministicConsolidator
from swarmbrain.adapters.embeddings import DeterministicEmbeddingProvider
from swarmbrain.adapters.extraction import DefaultRuleExtractor, InMemoryWorkStore
from swarmbrain.adapters.memory import (
    InMemoryKernel,
    in_memory_hybrid_retrieval_gateways,
    in_memory_retrieval_gateways,
)
from swarmbrain.config import (
    ApiSettings,
    BackendKind,
    EmbeddingsKind,
    ExtractionProviderKind,
)
from swarmbrain.domain.evidence import SourceTrust
from swarmbrain.ports.consolidation import ConsolidationProvider
from swarmbrain.ports.coordination_store import CoordinationStore
from swarmbrain.ports.embeddings import EmbeddingIndex, EmbeddingProvider
from swarmbrain.ports.extraction import ExtractionProvider, SourceIngestStore
from swarmbrain.ports.work_queue import WorkQueueStore
from swarmbrain.workers import ConsolidationWorker, ExtractionWorker, LeasedWorkWorker
from swarmbrain.workers.embedding import EmbedMemoryHandler

from .conflict_service import ConflictService
from .consolidation import ConsolidationObserver, ConsolidationService
from .coordination import CoordinationService
from .evidence_service import EvidenceService
from .extraction import ExtractionService
from .memory_policy import ConservativeMemoryPolicy
from .memory_service import MemoryService
from .retrieval_service import RetrievalService
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
class IngestionProfile:
    """Operator-owned extraction controls, deliberately absent from HTTP bodies."""

    trust: SourceTrust = SourceTrust.UNKNOWN
    use_provider: bool = False
    chunk_chars: int = 65_536
    priority: int = 0


@dataclass(frozen=True, slots=True)
class SwarmBrainRuntime:
    backend: BackendKind
    coordination: CoordinationService
    memory: MemoryService
    conflicts: ConflictService
    evidence: EvidenceService
    tokens: RunTokenCodec
    coordination_store: CoordinationStore | None = None
    kernel: InMemoryKernel | None = None
    extraction: ExtractionService | None = None
    ingestion_profile: IngestionProfile = field(default_factory=IngestionProfile)
    work: DurableWorkService | None = None
    work_queue: WorkQueueStore | None = None
    source_store: SourceIngestStore | None = None
    embeddings: EmbeddingProvider | None = None
    embedding_index: EmbeddingIndex | None = None
    consolidation: ConsolidationService | None = None
    consolidation_observer: ConsolidationObserver | None = None
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

    def extraction_worker(self, *, retry_delay_seconds: int = 30) -> ExtractionWorker:
        if self.extraction is None or self.source_store is None or self.work_queue is None:
            raise RuntimeError("durable extraction is not configured for this runtime")
        return ExtractionWorker(
            self.work_queue,
            self.source_store,
            self.extraction,
            retry_delay_seconds=retry_delay_seconds,
        )

    def embedding_worker(self, *, retry_delay_seconds: int = 30) -> LeasedWorkWorker:
        if self.embeddings is None or self.embedding_index is None or self.work_queue is None:
            raise RuntimeError("embeddings are not configured for this runtime")
        return LeasedWorkWorker(
            self.work_queue,
            [EmbedMemoryHandler(self.embeddings)],
            retry_delay_seconds=retry_delay_seconds,
        )

    def consolidation_worker(self, *, retry_delay_seconds: int = 30) -> ConsolidationWorker:
        if self.consolidation is None or self.work_queue is None:
            raise RuntimeError("memory consolidation is not configured for this runtime")
        return ConsolidationWorker(
            self.work_queue,
            self.consolidation,
            retry_delay_seconds=retry_delay_seconds,
        )


def build_in_memory_runtime(
    token_secret: str,
    *,
    embeddings: EmbeddingProvider | None = None,
    dense_min_similarity: float = 0.0,
    clock: Callable[[], datetime] | None = None,
    consolidation_enabled: bool = False,
    consolidation_provider: ConsolidationProvider | None = None,
    consolidation_use_provider: bool = False,
    consolidation_max_memories: int = 12,
    consolidation_max_actions: int = 4,
    consolidation_max_input_bytes: int = 65_536,
) -> SwarmBrainRuntime:
    kernel = InMemoryKernel() if clock is None else InMemoryKernel(clock=clock)
    policy = ConservativeMemoryPolicy()
    work_queue = InMemoryWorkStore() if clock is None else InMemoryWorkStore(clock=clock)
    work = DurableWorkService(work_queue)
    embedding_index = work_queue if embeddings is not None else None
    gateways = (
        in_memory_retrieval_gateways(kernel)
        if embedding_index is None
        else in_memory_hybrid_retrieval_gateways(
            kernel,
            embedding_index,
            dense_min_similarity=dense_min_similarity,
        )
    )
    retrieval = RetrievalService(gateways, kernel)
    observer = ConsolidationObserver(
        kernel,
        kernel,
        work,
        enabled=consolidation_enabled,
        use_provider=consolidation_use_provider,
        max_memories=consolidation_max_memories,
        max_actions=consolidation_max_actions,
        max_input_bytes=consolidation_max_input_bytes,
    )
    memory = MemoryService(
        kernel,
        policy,
        review_store=kernel,
        embeddings=embeddings,
        embedding_index=embedding_index,
        work=work if embeddings is not None else None,
        retrieval=retrieval,
        canonical_reader=kernel,
        write_observer=observer if consolidation_enabled else None,
    )
    consolidation = ConsolidationService(
        memory,
        kernel,
        SafeDeterministicConsolidator(),
        provider=consolidation_provider,
    )
    coordination = CoordinationService(kernel, memory_service=memory)
    return SwarmBrainRuntime(
        backend=BackendKind.MEMORY,
        coordination=coordination,
        memory=memory,
        conflicts=ConflictService(kernel),
        evidence=EvidenceService(kernel),
        tokens=RunTokenCodec(token_secret),
        coordination_store=kernel,
        kernel=kernel,
        work=work,
        work_queue=work_queue,
        embeddings=embeddings,
        embedding_index=embedding_index,
        consolidation=consolidation if consolidation_enabled else None,
        consolidation_observer=observer if consolidation_enabled else None,
    )


def build_runtime(settings: ApiSettings) -> SwarmBrainRuntime:
    """Compose the selected backend without opening network resources."""

    if settings.backend is BackendKind.MEMORY:
        return build_in_memory_runtime(
            settings.token_secret,
            embeddings=_build_embedding_provider(settings),
            dense_min_similarity=settings.retrieval_dense_min_similarity,
            consolidation_enabled=settings.consolidation_enabled,
            consolidation_provider=_build_consolidation_provider(settings),
            consolidation_use_provider=settings.consolidation_use_provider,
            consolidation_max_memories=settings.consolidation_max_memories,
            consolidation_max_actions=settings.consolidation_max_actions,
            consolidation_max_input_bytes=settings.consolidation_max_input_bytes,
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
    if settings.embeddings is EmbeddingsKind.OPENAI:
        from swarmbrain.adapters.embeddings.openai_compatible import (
            OpenAICompatibleEmbeddingProvider,
        )

        assert settings.embeddings_base_url is not None
        return OpenAICompatibleEmbeddingProvider(
            base_url=settings.embeddings_base_url,
            model_id=settings.embeddings_model or "Qwen/Qwen3-Embedding-0.6B",
            dimensions=settings.embeddings_dimensions,
            api_key=settings.embeddings_api_key,
        )
    from swarmbrain.adapters.embeddings.bedrock import BedrockEmbeddingProvider

    return BedrockEmbeddingProvider(
        model_id=settings.embeddings_model or "amazon.titan-embed-text-v2:0",
        dimensions=settings.embeddings_dimensions,
        region_name=settings.aws_region,
    )


def _build_extraction_provider(settings: ApiSettings) -> ExtractionProvider | None:
    """Compose provider configuration without opening a client or network connection."""

    if settings.extraction_provider is ExtractionProviderKind.NONE:
        return None
    from swarmbrain.adapters.extraction.openai_compatible import (
        OpenAICompatibleExtractionProvider,
    )

    assert settings.extraction_base_url is not None
    assert settings.extraction_model is not None
    return OpenAICompatibleExtractionProvider(
        base_url=settings.extraction_base_url,
        model_id=settings.extraction_model,
        revision=settings.extraction_revision,
        api_key=settings.extraction_api_key,
        timeout_seconds=settings.extraction_timeout_seconds,
        max_output_tokens=settings.extraction_max_output_tokens,
    )


def _build_consolidation_provider(settings: ApiSettings) -> ConsolidationProvider | None:
    """Reuse the separately authenticated model profile with a stricter prompt."""

    if not settings.consolidation_use_provider:
        return None
    from swarmbrain.adapters.consolidation import OpenAICompatibleConsolidationProvider

    assert settings.extraction_base_url is not None
    assert settings.extraction_model is not None
    return OpenAICompatibleConsolidationProvider(
        base_url=settings.extraction_base_url,
        model_id=settings.extraction_model,
        revision=settings.extraction_revision,
        api_key=settings.extraction_api_key,
        timeout_seconds=settings.extraction_timeout_seconds,
        max_output_tokens=settings.extraction_max_output_tokens,
        max_input_bytes=settings.consolidation_max_input_bytes,
    )


def _build_cockroach_runtime(settings: ApiSettings) -> SwarmBrainRuntime:
    # Keep the optional driver and durable repositories out of memory-only imports.
    from swarmbrain.adapters.cockroach.coordination import CockroachCoordinationStore
    from swarmbrain.adapters.cockroach.database import CockroachDatabase
    from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
    from swarmbrain.adapters.cockroach.retrieval import (
        cockroach_hybrid_retrieval_gateways,
        cockroach_retrieval_gateways,
    )
    from swarmbrain.adapters.cockroach.work_store import CockroachWorkStore

    assert settings.database_url is not None
    database = CockroachDatabase(
        settings.database_url,
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
    )
    coordination_store = CockroachCoordinationStore(database)
    memory_store = CockroachMemoryStore(database)
    embeddings = _build_embedding_provider(settings)
    work_queue = CockroachWorkStore(database)
    work = DurableWorkService(work_queue)
    embedding_index = memory_store if embeddings is not None else None
    gateways = (
        cockroach_retrieval_gateways(database)
        if embeddings is None
        else cockroach_hybrid_retrieval_gateways(
            database,
            dense_min_similarity=settings.retrieval_dense_min_similarity,
            dense_beam_size=settings.retrieval_dense_ann_beam_size,
        )
    )
    retrieval = RetrievalService(gateways, memory_store)
    policy = ConservativeMemoryPolicy()
    observer = ConsolidationObserver(
        memory_store,
        memory_store,
        work,
        enabled=settings.consolidation_enabled,
        use_provider=settings.consolidation_use_provider,
        max_memories=settings.consolidation_max_memories,
        max_actions=settings.consolidation_max_actions,
        max_input_bytes=settings.consolidation_max_input_bytes,
    )
    memory = MemoryService(
        memory_store,
        policy,
        review_store=memory_store,
        embeddings=embeddings,
        embedding_index=embedding_index,
        work=work if embeddings is not None else None,
        retrieval=retrieval,
        canonical_reader=memory_store,
        write_observer=observer if settings.consolidation_enabled else None,
    )
    consolidation = ConsolidationService(
        memory,
        memory_store,
        SafeDeterministicConsolidator(),
        provider=_build_consolidation_provider(settings),
    )
    extraction = ExtractionService(
        work_queue,
        DefaultRuleExtractor(),
        provider=_build_extraction_provider(settings),
        embedding_model=embeddings.model_name if embeddings is not None else None,
    )
    coordination = CoordinationService(coordination_store, memory_service=memory)
    return SwarmBrainRuntime(
        backend=BackendKind.COCKROACH,
        coordination=coordination,
        memory=memory,
        conflicts=ConflictService(memory_store),
        evidence=EvidenceService(memory_store),
        tokens=RunTokenCodec(settings.token_secret),
        coordination_store=coordination_store,
        extraction=extraction,
        ingestion_profile=IngestionProfile(
            trust=settings.ingest_trust,
            use_provider=settings.ingest_use_provider,
            chunk_chars=settings.ingest_chunk_chars,
            priority=settings.ingest_priority,
        ),
        work=work,
        work_queue=work_queue,
        source_store=work_queue,
        embeddings=embeddings,
        embedding_index=embedding_index,
        consolidation=consolidation if settings.consolidation_enabled else None,
        consolidation_observer=observer if settings.consolidation_enabled else None,
        _lifecycle=_CockroachLifecycle(database),
    )


__all__ = [
    "IngestionProfile",
    "SwarmBrainRuntime",
    "build_in_memory_runtime",
    "build_runtime",
]
