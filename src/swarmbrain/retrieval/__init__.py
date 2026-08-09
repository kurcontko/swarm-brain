"""Purpose-aware retrieval planning, projection, and fusion primitives."""

from .dense import (
    DENSE_PROJECTION_ID,
    DENSE_PROJECTION_REVISION,
    DENSE_VECTOR_DIMENSIONS,
    dense_projection_signature,
)
from .fusion import RRF_K, relevance_reranked, weighted_rrf
from .graph import (
    GRAPH_LINK_TYPES,
    GRAPH_PROJECTION_ID,
    GRAPH_PROJECTION_VERSION,
    GRAPH_RELATION_WEIGHTS,
)
from .packing import (
    AnswerInContextMetrics,
    PackedBundle,
    answer_in_context,
    estimate_tokens,
    pack_to_budget,
    render_recall_hit,
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
    "AnswerInContextMetrics",
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
    "PackedBundle",
    "RetrievalPlanner",
    "answer_in_context",
    "candidate_relevance",
    "domain_lane",
    "dense_projection_signature",
    "estimate_tokens",
    "exact_terms",
    "lookup_text",
    "pack_to_budget",
    "parse_query_identifiers",
    "projection_scope_key",
    "relevance_query",
    "relevance_reranked",
    "render_recall_hit",
    "search_text",
    "trigram_similarity",
    "weighted_rrf",
]
