"""Provider boundary for evidence-gated memory reflection."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from swarmbrain.domain.consolidation import (
    ConsolidationProposal,
    ConsolidationWorkPayload,
)
from swarmbrain.domain.extraction import ProviderDescriptor


@runtime_checkable
class ConsolidationProvider(Protocol):
    @property
    def descriptor(self) -> ProviderDescriptor: ...

    async def reflect(
        self,
        request: ConsolidationWorkPayload,
    ) -> tuple[ConsolidationProposal, ...]: ...


@runtime_checkable
class DeterministicConsolidator(Protocol):
    name: str
    revision: str

    async def reflect(
        self,
        request: ConsolidationWorkPayload,
    ) -> tuple[ConsolidationProposal, ...]: ...


__all__ = ["ConsolidationProvider", "DeterministicConsolidator"]
