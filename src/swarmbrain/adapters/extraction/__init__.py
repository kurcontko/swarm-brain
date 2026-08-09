"""Local and lazy-provider extraction adapters."""

from .coding import CodingRuleExtractor
from .in_memory import InMemoryWorkStore
from .openai_compatible import (
    OpenAICompatibleExtractionProvider,
    OpenAICompatibleExtractionUnavailable,
)
from .provider import LazyExtractionProvider, ProviderUnavailable
from .structured import (
    STRUCTURED_MEMORY_SOURCE_KIND,
    DefaultRuleExtractor,
    StructuredMemoryEnvelope,
)

__all__ = [
    "CodingRuleExtractor",
    "DefaultRuleExtractor",
    "InMemoryWorkStore",
    "LazyExtractionProvider",
    "OpenAICompatibleExtractionProvider",
    "OpenAICompatibleExtractionUnavailable",
    "ProviderUnavailable",
    "STRUCTURED_MEMORY_SOURCE_KIND",
    "StructuredMemoryEnvelope",
]
