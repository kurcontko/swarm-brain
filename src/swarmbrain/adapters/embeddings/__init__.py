"""Optional embedding providers behind the provider-neutral port."""

from .deterministic import DeterministicEmbeddingProvider

__all__ = ["DeterministicEmbeddingProvider"]
