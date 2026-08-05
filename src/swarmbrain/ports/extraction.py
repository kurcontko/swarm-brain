"""Narrow boundaries for raw ingestion and untrusted candidate generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.extraction import (
    ExtractionCandidate,
    ExtractionInput,
    PreparedSourceIngest,
    ProviderDescriptor,
    SourceIngestResult,
)
from swarmbrain.domain.work import SourceExtractionStatus, WorkLease

ProviderCandidate = Mapping[str, object] | ExtractionCandidate


@runtime_checkable
class SourceIngestStore(Protocol):
    """Atomically preserve raw source/chunks and enqueue its extraction work."""

    async def ingest_source(
        self,
        actor: ActorContext,
        prepared: PreparedSourceIngest,
    ) -> SourceIngestResult: ...

    async def load_extraction_input(self, lease: WorkLease) -> ExtractionInput: ...

    async def get_extraction_status(
        self,
        actor: ActorContext,
        source_id: str,
    ) -> SourceExtractionStatus: ...


@runtime_checkable
class DeterministicExtractor(Protocol):
    name: str
    revision: str

    async def extract(self, request: ExtractionInput) -> Sequence[ExtractionCandidate]: ...


@runtime_checkable
class ExtractionProvider(Protocol):
    """Remote/model boundary. Implementations are invoked outside transactions."""

    @property
    def descriptor(self) -> ProviderDescriptor: ...

    async def extract(self, request: ExtractionInput) -> Sequence[ProviderCandidate]: ...


__all__ = [
    "DeterministicExtractor",
    "ExtractionProvider",
    "ProviderCandidate",
    "SourceIngestStore",
]
