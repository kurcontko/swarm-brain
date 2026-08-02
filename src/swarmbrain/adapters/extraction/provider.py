"""Lazy optional-provider boundary with no import-time credential requirement."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from swarmbrain.domain.extraction import ExtractionInput, ProviderDescriptor
from swarmbrain.ports.extraction import ExtractionProvider, ProviderCandidate


class ProviderUnavailable(RuntimeError):
    """The configured optional provider could not be instantiated."""


class LazyExtractionProvider:
    """Instantiate an optional SDK/provider only on the first selected provider route."""

    def __init__(
        self,
        descriptor: ProviderDescriptor,
        factory: Callable[[], ExtractionProvider],
    ) -> None:
        self._descriptor = descriptor
        self._factory = factory
        self._provider: ExtractionProvider | None = None

    @property
    def descriptor(self) -> ProviderDescriptor:
        return self._descriptor

    async def extract(self, request: ExtractionInput) -> Sequence[ProviderCandidate]:
        if self._provider is None:
            try:
                self._provider = self._factory()
            except Exception as exc:
                raise ProviderUnavailable("optional extraction provider is unavailable") from exc
        return await self._provider.extract(request)


__all__ = ["LazyExtractionProvider", "ProviderUnavailable"]
