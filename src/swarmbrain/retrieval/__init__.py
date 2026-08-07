"""Purpose-aware retrieval planning, projection, and fusion primitives."""

from .dense import (
    DENSE_PROJECTION_ID,
    DENSE_PROJECTION_REVISION,
    DENSE_VECTOR_DIMENSIONS,
    dense_projection_signature,
)
from .fusion import RRF_K, weighted_rrf
from .graph import (
    GRAPH_LINK_TYPES,
    GRAPH_PROJECTION_ID,
    GRAPH_PROJECTION_VERSION,
    GRAPH_RELATION_WEIGHTS,
)
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
from .relevance import (
    RELEVANCE_VERSION,
    RelevanceQuery,
    candidate_relevance,
    relevance_query,
)

__all__ = [
    "RELEVANCE_VERSION",
    "DENSE_PROJECTION_ID",
    "DENSE_PROJECTION_REVISION",
    "DENSE_VECTOR_DIMENSIONS",
    "GRAPH_LINK_TYPES",
    "GRAPH_PROJECTION_ID",
    "GRAPH_PROJECTION_VERSION",
    "GRAPH_RELATION_WEIGHTS",
    "RETRIEVAL_PROJECTION_ID",
    "RRF_K",
    "RelevanceQuery",
    "RetrievalPlanner",
    "candidate_relevance",
    "domain_lane",
    "dense_projection_signature",
    "exact_terms",
    "lookup_text",
    "parse_query_identifiers",
    "projection_scope_key",
    "relevance_query",
    "search_text",
    "trigram_similarity",
    "weighted_rrf",
]
