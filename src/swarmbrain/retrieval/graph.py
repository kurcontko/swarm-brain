"""Bounded, query-aware spreading activation over canonical memory links."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from swarmbrain.domain.common import JsonObject, MemoryContent
from swarmbrain.domain.retrieval import FusedCandidate, RetrievalPlan

from .projection import MAX_QUERY_CHARS, MAX_QUERY_TOKENS, normalize_term, search_text

GRAPH_PROJECTION_ID = "memory-links-v1"
GRAPH_PROJECTION_VERSION = "bounded-spreading-activation-v1"
GRAPH_STEP_DECAY = 0.85
GRAPH_QUERY_GATE_FLOOR = 0.60
GRAPH_FANOUT_SCAN_MULTIPLIER = 4

# Only server-known relations enter synchronous retrieval. Custom semantic
# labels remain valid lineage metadata but cannot unexpectedly create a recall
# path until a planner/gateway release assigns them an explicit weight.
GRAPH_RELATION_WEIGHTS: Mapping[str, float] = {
    "supports": 1.00,
    "supersedes": 0.98,
    "derived_from": 0.92,
    "merged_from": 0.90,
    "contradicts": 0.88,
    "duplicate_of": 0.82,
    "related_to": 0.75,
}
GRAPH_LINK_TYPES = frozenset(GRAPH_RELATION_WEIGHTS)

_TOKEN = re.compile(r"\w+", flags=re.UNICODE)
_SYMMETRIC_RELATIONS = frozenset({"contradicts", "duplicate_of", "related_to"})


@dataclass(frozen=True, slots=True)
class GraphTraversalState:
    node_id: str
    activation: float
    depth: int
    seed_id: str
    node_path: tuple[str, ...]
    trace_path: tuple[str, ...]
    relation_path: tuple[str, ...]


def select_graph_seeds(
    plan: RetrievalPlan,
    fused: Sequence[FusedCandidate],
) -> tuple[tuple[str, float], ...]:
    """Prefer explicit seeds, then take the strongest direct fused hits."""

    if plan.graph_seed_limit <= 0:
        return ()
    selected: dict[str, float] = {}
    for memory_id in plan.seed_memory_ids:
        selected.setdefault(memory_id, 1.0)
        if len(selected) >= plan.graph_seed_limit:
            return tuple(selected.items())
    for candidate in fused:
        selected.setdefault(candidate.canonical_id, candidate.normalized_score)
        if len(selected) >= plan.graph_seed_limit:
            break
    return tuple(selected.items())


def graph_query_gate(
    query_text: str,
    *,
    title: str | None,
    content: MemoryContent,
    tags: tuple[str, ...],
    metadata: JsonObject | None,
) -> float:
    """Return a bounded lexical relevance gate without suppressing linked context."""

    query_tokens = tuple(
        dict.fromkeys(_TOKEN.findall(normalize_term(query_text[:MAX_QUERY_CHARS])))
    )[:MAX_QUERY_TOKENS]
    if not query_tokens:
        return 1.0
    projected = normalize_term(
        search_text(title=title, content=content, tags=tags, metadata=metadata)
    )
    target_tokens = frozenset(_TOKEN.findall(projected))
    overlap = sum(token in target_tokens for token in query_tokens) / len(query_tokens)
    return GRAPH_QUERY_GATE_FLOOR + (1.0 - GRAPH_QUERY_GATE_FLOOR) * overlap


def graph_step_activation(
    parent_activation: float,
    relation: str,
    *,
    reverse: bool,
    query_gate: float,
) -> float:
    """Apply relation, direction, hop, and query-aware decay for one edge."""

    relation_weight = GRAPH_RELATION_WEIGHTS.get(relation, 0.0)
    direction_weight = 1.0 if not reverse or relation in _SYMMETRIC_RELATIONS else 0.95
    return parent_activation * relation_weight * direction_weight * GRAPH_STEP_DECAY * query_gate


def graph_state_is_better(
    candidate: GraphTraversalState,
    current: GraphTraversalState | None,
) -> bool:
    if current is None:
        return True
    candidate_key = (-candidate.activation, candidate.depth, candidate.trace_path)
    current_key = (-current.activation, current.depth, current.trace_path)
    return candidate_key < current_key


__all__ = [
    "GRAPH_LINK_TYPES",
    "GRAPH_FANOUT_SCAN_MULTIPLIER",
    "GRAPH_PROJECTION_ID",
    "GRAPH_PROJECTION_VERSION",
    "GRAPH_QUERY_GATE_FLOOR",
    "GRAPH_RELATION_WEIGHTS",
    "GRAPH_STEP_DECAY",
    "GraphTraversalState",
    "graph_query_gate",
    "graph_state_is_better",
    "graph_step_activation",
    "select_graph_seeds",
]
