"""Purpose-aware retrieval planning, projection, and fusion primitives."""

from .fusion import RRF_K, weighted_rrf
from .planner import RetrievalPlanner, parse_query_identifiers
from .projection import (
    RETRIEVAL_PROJECTION_ID,
    domain_lane,
    exact_terms,
    lookup_text,
    projection_scope_key,
    search_text,
    trigram_similarity,
)

__all__ = [
    "RETRIEVAL_PROJECTION_ID",
    "RRF_K",
    "RetrievalPlanner",
    "domain_lane",
    "exact_terms",
    "lookup_text",
    "parse_query_identifiers",
    "projection_scope_key",
    "search_text",
    "trigram_similarity",
    "weighted_rrf",
]
