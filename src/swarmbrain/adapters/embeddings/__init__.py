"""Embedding provider adapters; heavy SDKs stay behind lazy boundaries."""

from .deterministic import DeterministicEmbeddingProvider

__all__ = ["DeterministicEmbeddingProvider"]
