"""Local deterministic adapters used for development and contract tests."""

from .in_memory import InMemoryKernel
from .retrieval import (
    InMemoryDenseRetrievalGateway,
    InMemoryGraphRetrievalGateway,
    InMemoryRetrievalGateway,
    in_memory_hybrid_retrieval_gateways,
    in_memory_retrieval_gateways,
)

__all__ = [
    "InMemoryDenseRetrievalGateway",
    "InMemoryGraphRetrievalGateway",
    "InMemoryKernel",
    "InMemoryRetrievalGateway",
    "in_memory_hybrid_retrieval_gateways",
    "in_memory_retrieval_gateways",
]
