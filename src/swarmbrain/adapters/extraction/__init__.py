"""Local and lazy-provider extraction adapters."""

from .coding import CodingRuleExtractor
from .in_memory import InMemoryWorkStore
from .provider import LazyExtractionProvider, ProviderUnavailable

__all__ = [
    "CodingRuleExtractor",
    "InMemoryWorkStore",
    "LazyExtractionProvider",
    "ProviderUnavailable",
]
