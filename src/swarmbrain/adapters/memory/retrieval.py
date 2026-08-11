"""Deterministic reference retrieval lanes over the local in-memory kernel."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from swarmbrain.application.memory_policy import memory_text_sha256
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.memory import Memory, MemoryLink, RecallQuery
from swarmbrain.domain.retrieval import (
    Candidate,
    CandidateBatch,
    DenseQuery,
    FusedCandidate,
    RetrievalPlan,
    RetrievalPurpose,
    RetrievalSignal,
)
from swarmbrain.ports.embeddings import EmbeddingIndex
from swarmbrain.retrieval.dense import DENSE_PROJECTION_ID, dense_projection_signature
from swarmbrain.retrieval.graph import (
    GRAPH_FANOUT_SCAN_MULTIPLIER,
    GRAPH_PROJECTION_ID,
    GRAPH_PROJECTION_VERSION,
    GRAPH_RELATION_WEIGHTS,
    GraphTraversalState,
    graph_query_gate,
    graph_state_is_better,
    graph_step_activation,
    select_graph_seeds,
)
from swarmbrain.retrieval.planner import (
    OCCURRENCE_TEMPORAL_PROJECTION_ID,
    OCCURRENCE_TEMPORAL_PROJECTION_VERSION,
    TEMPORAL_PROJECTION_ID,
    TEMPORAL_PROJECTION_VERSION,
    temporal_query_target,
    temporal_valid_from_score,
)
from swarmbrain.retrieval.projection import (
    FUZZY_SIMILARITY_THRESHOLD,
    MAX_EXACT_TERM,
    MAX_QUERY_CHARS,
    MAX_QUERY_TOKENS,
    RETRIEVAL_PROJECTION_ID,
    domain_lane,
    exact_terms,
    lookup_text,
    normalize_term,
    search_text,
    trigram_similarity,
)

from .in_memory import InMemoryKernel

_TOKENS = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class _AdjacentMemoryLink:
    link: MemoryLink
    other_id: str
    reverse: bool


class InMemoryRetrievalGateway:
    def __init__(self, kernel: InMemoryKernel, signal: RetrievalSignal) -> None:
        if signal not in {
            RetrievalSignal.EXACT,
            RetrievalSignal.LEXICAL,
            RetrievalSignal.FUZZY,
        }:
            raise ValueError(f"unsupported in-memory retrieval signal: {signal}")
        self.kernel = kernel
        self._signal = signal

    @property
    def signal(self) -> RetrievalSignal:
        return self._signal

    async def retrieve(
        self,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
        dense_query: DenseQuery | None = None,
    ) -> CandidateBatch:
        del dense_query
        started = perf_counter()
        memories = await self.kernel.recallable_memories(actor, query)
        if self.signal is RetrievalSignal.EXACT:
            scored = self._exact(memories, plan, query)
        elif self.signal is RetrievalSignal.LEXICAL:
            scored = self._lexical(memories, plan, query)
        else:
            scored = self._fuzzy(memories, query)
        scored.sort(
            key=lambda item: (-item[0], -item[1].recorded_from.timestamp(), item[1].memory_id)
        )
        budget = plan.lane_budgets[self.signal.value]
        candidates = tuple(
            self._candidate(memory, rank, raw_score, reasons)
            for rank, (raw_score, memory, reasons) in enumerate(scored[:budget], start=1)
        )
        return CandidateBatch(
            lane=self.signal,
            candidates=candidates,
            examined_count=len(memories),
            latency_ms=(perf_counter() - started) * 1000.0,
            truncated=len(scored) > budget,
            projection_watermark=RETRIEVAL_PROJECTION_ID,
        )

    @staticmethod
    def _exact(
        memories: tuple[Memory, ...],
        plan: RetrievalPlan,
        query: RecallQuery,
    ) -> list[tuple[float, Memory, tuple[str, ...]]]:
        query_term = normalize_term(query.text[:MAX_QUERY_CHARS])
        seeds = frozenset(plan.seed_memory_ids)
        scored: list[tuple[float, Memory, tuple[str, ...]]] = []
        for memory in memories:
            reasons: list[str] = []
            if memory.memory_id in seeds:
                reasons.append("exact_seed_id")
            terms = exact_terms(
                memory_id=memory.memory_id,
                content_sha256=memory_text_sha256(memory.content),
                title=memory.title,
                content=memory.content,
                tags=memory.tags,
                metadata=memory.metadata,
            )
            if len(query_term) <= MAX_EXACT_TERM:
                reasons.extend(f"exact_{term.kind}" for term in terms if term.value == query_term)
            if reasons:
                scored.append((1.0, memory, tuple(dict.fromkeys(reasons))))
        return scored

    @staticmethod
    def _lexical(
        memories: tuple[Memory, ...],
        plan: RetrievalPlan,
        query: RecallQuery,
    ) -> list[tuple[float, Memory, tuple[str, ...]]]:
        normalized_query = normalize_term(query.text[:MAX_QUERY_CHARS])
        query_tokens = set(_TOKENS.findall(normalized_query)[:MAX_QUERY_TOKENS])
        if not query_tokens:
            return []
        scored: list[tuple[float, Memory, tuple[str, ...]]] = []
        for memory in memories:
            haystack = search_text(
                title=memory.title,
                content=memory.content,
                tags=memory.tags,
                metadata=memory.metadata,
            )
            normalized_haystack = normalize_term(haystack)
            memory_tokens = set(_TOKENS.findall(normalized_haystack))
            overlap = len(query_tokens & memory_tokens) / len(query_tokens)
            substring = normalized_query in normalized_haystack
            score = overlap + (0.2 if substring else 0.0)
            if score <= 0.0:
                continue
            if plan.purpose is RetrievalPurpose.TASK_BOOTSTRAP and domain_lane(
                memory.kind, memory.metadata
            ) in {"handoff", "playbook"}:
                score *= 1.1
            scored.append((score, memory, ("lexical_overlap",)))
        return scored

    @staticmethod
    def _fuzzy(
        memories: tuple[Memory, ...],
        query: RecallQuery,
    ) -> list[tuple[float, Memory, tuple[str, ...]]]:
        query_term = normalize_term(query.text[:MAX_QUERY_CHARS])
        if len(query_term) < 3 or len(query_term) > 256:
            return []
        scored: list[tuple[float, Memory, tuple[str, ...]]] = []
        for memory in memories:
            projected_lookup = lookup_text(
                memory_id=memory.memory_id,
                content_sha256=memory_text_sha256(memory.content),
                title=memory.title,
                content=memory.content,
                tags=memory.tags,
                metadata=memory.metadata,
            )
            score = trigram_similarity(projected_lookup, query_term)
            if score >= FUZZY_SIMILARITY_THRESHOLD:
                scored.append((score, memory, ("trigram_similarity",)))
        return scored

    def _candidate(
        self,
        memory: Memory,
        rank: int,
        raw_score: float,
        reasons: tuple[str, ...],
    ) -> Candidate:
        return Candidate(
            resource_type="memory",
            resource_id=memory.memory_id,
            resource_version=memory.version,
            canonical_id=memory.memory_id,
            domain_lane=domain_lane(memory.kind, memory.metadata),
            signal=self.signal,
            rank=rank,
            raw_score=raw_score,
            projection_id=RETRIEVAL_PROJECTION_ID,
            projection_version="in-memory-v1",
            reasons=reasons,
            evidence_ids=tuple(reference.evidence_id for reference in memory.evidence),
        )


class InMemoryTemporalRetrievalGateway:
    """Explicit temporal lane over canonically eligible memories."""

    signal = RetrievalSignal.TEMPORAL

    def __init__(self, kernel: InMemoryKernel) -> None:
        self.kernel = kernel

    async def retrieve(
        self,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
        dense_query: DenseQuery | None = None,
    ) -> CandidateBatch:
        del dense_query
        started = perf_counter()
        target_with_kind = temporal_query_target(query)
        if target_with_kind is None:
            return CandidateBatch(
                lane=self.signal,
                examined_count=0,
                latency_ms=(perf_counter() - started) * 1000.0,
                projection_watermark=TEMPORAL_PROJECTION_VERSION,
            )

        target, target_kind = target_with_kind
        memories = await self.kernel.recallable_memories(actor, query)
        occurrence_prior = query.occurrence_time_prior_from is not None
        if occurrence_prior:
            scored = [
                (temporal_valid_from_score(memory.occurred_at, target), memory)
                for memory in memories
                if memory.occurred_at is not None
            ]
            scored.sort(key=lambda item: (-item[0], item[1].occurred_at, item[1].memory_id))
            projection_id = OCCURRENCE_TEMPORAL_PROJECTION_ID
            projection_version = OCCURRENCE_TEMPORAL_PROJECTION_VERSION
            reason = "temporal_occurrence_distance"
        else:
            scored = [
                (temporal_valid_from_score(memory.valid_from, target), memory)
                for memory in memories
            ]
            scored.sort(key=lambda item: (-item[0], item[1].valid_from, item[1].memory_id))
            projection_id = TEMPORAL_PROJECTION_ID
            projection_version = TEMPORAL_PROJECTION_VERSION
            reason = "temporal_valid_from_distance"
        budget = plan.lane_budgets[self.signal.value]
        candidates = tuple(
            Candidate(
                resource_type="memory",
                resource_id=memory.memory_id,
                resource_version=memory.version,
                canonical_id=memory.memory_id,
                domain_lane=domain_lane(memory.kind, memory.metadata),
                signal=self.signal,
                rank=rank,
                raw_score=raw_score,
                projection_id=projection_id,
                projection_version=projection_version,
                reasons=(
                    reason,
                    f"temporal_target:{target_kind}",
                ),
                evidence_ids=tuple(reference.evidence_id for reference in memory.evidence),
            )
            for rank, (raw_score, memory) in enumerate(scored[:budget], start=1)
        )
        return CandidateBatch(
            lane=self.signal,
            candidates=candidates,
            examined_count=len(memories),
            latency_ms=(perf_counter() - started) * 1000.0,
            truncated=len(scored) > budget,
            projection_watermark=projection_version,
        )


class InMemoryDenseRetrievalGateway:
    """Exact reference dense lane over a scope-filterable embedding index."""

    signal = RetrievalSignal.DENSE

    def __init__(
        self,
        kernel: InMemoryKernel,
        embedding_index: EmbeddingIndex,
        *,
        min_similarity: float = 0.0,
    ) -> None:
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("dense min_similarity must be between 0 and 1")
        self.kernel = kernel
        self.embedding_index = embedding_index
        self.min_similarity = min_similarity

    async def retrieve(
        self,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
        dense_query: DenseQuery | None = None,
    ) -> CandidateBatch:
        started = perf_counter()
        if dense_query is None or dense_query.values is None:
            return CandidateBatch(
                lane=self.signal,
                examined_count=0,
                latency_ms=(perf_counter() - started) * 1000.0,
                degraded=True,
                degradation_reason=(
                    dense_query.unavailable_reason
                    if dense_query is not None
                    else "query_embedding_unavailable"
                ),
            )

        memories = await self.kernel.recallable_memories(actor, query)
        by_id = {memory.memory_id: memory for memory in memories}
        budget = plan.lane_budgets[self.signal.value]
        matches = await self.embedding_index.search_embeddings(
            actor,
            dense_query.values,
            model=dense_query.model,
            limit=budget,
            min_score=self.min_similarity,
            candidate_ids=tuple(by_id),
        )
        signature = dense_projection_signature(dense_query.model, dense_query.dimensions)
        candidates = tuple(
            Candidate(
                resource_type="memory",
                resource_id=match.memory_id,
                resource_version=by_id[match.memory_id].version,
                canonical_id=match.memory_id,
                domain_lane=domain_lane(
                    by_id[match.memory_id].kind,
                    by_id[match.memory_id].metadata,
                ),
                signal=self.signal,
                rank=rank,
                raw_score=match.score,
                projection_id=DENSE_PROJECTION_ID,
                projection_version=signature,
                reasons=("semantic_match", "dense_cosine_exact"),
                evidence_ids=tuple(
                    reference.evidence_id for reference in by_id[match.memory_id].evidence
                ),
            )
            for rank, match in enumerate(matches, start=1)
            if match.memory_id in by_id
        )
        return CandidateBatch(
            lane=self.signal,
            candidates=candidates,
            examined_count=len(memories),
            latency_ms=(perf_counter() - started) * 1000.0,
            truncated=len(matches) >= budget,
            projection_watermark=signature,
        )


class InMemoryGraphRetrievalGateway:
    """Bounded second-stage spreading activation over canonical memory links."""

    signal = RetrievalSignal.GRAPH

    def __init__(self, kernel: InMemoryKernel) -> None:
        self.kernel = kernel

    async def expand(
        self,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
        seeds: Sequence[FusedCandidate],
        dense_query: DenseQuery | None = None,
    ) -> CandidateBatch:
        del dense_query
        started = perf_counter()
        selected_seeds = select_graph_seeds(plan, seeds)
        if (
            plan.max_graph_hops == 0
            or plan.graph_seed_limit == 0
            or plan.graph_max_fanout == 0
            or plan.graph_edge_budget == 0
            or not selected_seeds
        ):
            return CandidateBatch(
                lane=self.signal,
                examined_count=0,
                latency_ms=(perf_counter() - started) * 1000.0,
                projection_watermark=GRAPH_PROJECTION_VERSION,
            )

        memories, links = await self.kernel.recallable_memory_graph(actor, query)
        by_id = {memory.memory_id: memory for memory in memories}
        allowed_relations = frozenset(plan.graph_link_types) & frozenset(GRAPH_RELATION_WEIGHTS)
        eligible_seeds = tuple(
            (memory_id, score) for memory_id, score in selected_seeds if memory_id in by_id
        )
        if not eligible_seeds or not allowed_relations:
            return CandidateBatch(
                lane=self.signal,
                examined_count=0,
                latency_ms=(perf_counter() - started) * 1000.0,
                projection_watermark=GRAPH_PROJECTION_VERSION,
            )

        adjacency: dict[str, list[_AdjacentMemoryLink]] = defaultdict(list)
        for link in links:
            relation = str(link.kind)
            if relation not in allowed_relations:
                continue
            adjacency[link.source_memory_id].append(
                _AdjacentMemoryLink(link=link, other_id=link.target_memory_id, reverse=False)
            )
            adjacency[link.target_memory_id].append(
                _AdjacentMemoryLink(link=link, other_id=link.source_memory_id, reverse=True)
            )
        for incident in adjacency.values():
            incident.sort(
                key=lambda item: (
                    -GRAPH_RELATION_WEIGHTS[str(item.link.kind)],
                    -item.link.created_at.timestamp(),
                    item.link.link_id,
                    item.other_id,
                )
            )

        seed_ids = frozenset(memory_id for memory_id, _ in eligible_seeds)
        frontier = tuple(
            GraphTraversalState(
                node_id=memory_id,
                activation=score,
                depth=0,
                seed_id=memory_id,
                node_path=(memory_id,),
                trace_path=(f"memory:{memory_id}",),
                relation_path=(),
            )
            for memory_id, score in eligible_seeds
        )
        best: dict[str, GraphTraversalState] = {state.node_id: state for state in frontier}
        examined_count = 0
        truncated = False
        traversed_at = []

        for _hop in range(1, plan.max_graph_hops + 1):
            next_frontier: dict[str, GraphTraversalState] = {}
            for state in frontier:
                if examined_count >= plan.graph_edge_budget:
                    truncated = True
                    break
                incident = tuple(
                    edge
                    for edge in adjacency.get(state.node_id, ())
                    if edge.other_id not in state.node_path
                )
                remaining = plan.graph_edge_budget - examined_count
                scan_cap = max(
                    plan.graph_max_fanout + 1,
                    plan.graph_max_fanout * GRAPH_FANOUT_SCAN_MULTIPLIER,
                )
                looked_at = incident[: min(scan_cap, remaining)]
                examined_count += len(looked_at)
                unique_edges: list[_AdjacentMemoryLink] = []
                seen_neighbors: set[str] = set()
                for edge in looked_at:
                    if edge.other_id in seen_neighbors:
                        continue
                    seen_neighbors.add(edge.other_id)
                    unique_edges.append(edge)
                if len(unique_edges) > plan.graph_max_fanout or len(incident) > len(looked_at):
                    truncated = True

                traversed_at.extend(edge.link.created_at for edge in looked_at)
                for edge in unique_edges[: plan.graph_max_fanout]:
                    relation = str(edge.link.kind)
                    target = by_id[edge.other_id]
                    activation = graph_step_activation(
                        state.activation,
                        relation,
                        reverse=edge.reverse,
                        query_gate=graph_query_gate(
                            query.text,
                            title=target.title,
                            content=target.content,
                            tags=target.tags,
                            metadata=target.metadata,
                        ),
                    )
                    direction = "in" if edge.reverse else "out"
                    candidate = GraphTraversalState(
                        node_id=edge.other_id,
                        activation=activation,
                        depth=state.depth + 1,
                        seed_id=state.seed_id,
                        node_path=(*state.node_path, edge.other_id),
                        trace_path=(
                            *state.trace_path,
                            f"edge:{edge.link.link_id}:{direction}:{relation}",
                            f"memory:{edge.other_id}",
                        ),
                        relation_path=(*state.relation_path, relation),
                    )
                    if not graph_state_is_better(candidate, best.get(candidate.node_id)):
                        continue
                    best[candidate.node_id] = candidate
                    current_next = next_frontier.get(candidate.node_id)
                    if graph_state_is_better(candidate, current_next):
                        next_frontier[candidate.node_id] = candidate
            if not next_frontier or examined_count >= plan.graph_edge_budget:
                truncated = truncated or (
                    bool(next_frontier) and examined_count >= plan.graph_edge_budget
                )
                break
            frontier = tuple(
                sorted(
                    next_frontier.values(),
                    key=lambda state: (-state.activation, state.depth, state.trace_path),
                )
            )

        ordered = sorted(
            (
                state
                for memory_id, state in best.items()
                if memory_id not in seed_ids and state.depth > 0
            ),
            key=lambda state: (
                -state.activation,
                state.depth,
                -by_id[state.node_id].recorded_from.timestamp(),
                state.node_id,
            ),
        )
        budget = plan.lane_budgets[self.signal.value]
        candidates = tuple(
            Candidate(
                resource_type="memory",
                resource_id=state.node_id,
                resource_version=by_id[state.node_id].version,
                canonical_id=state.node_id,
                domain_lane=domain_lane(
                    by_id[state.node_id].kind,
                    by_id[state.node_id].metadata,
                ),
                signal=self.signal,
                rank=rank,
                raw_score=state.activation,
                projection_id=GRAPH_PROJECTION_ID,
                projection_version=GRAPH_PROJECTION_VERSION,
                reasons=(
                    "graph_expand",
                    "graph_query_gated",
                    f"graph_hops:{state.depth}",
                    f"graph_path:{'>'.join(state.relation_path)}",
                ),
                evidence_ids=tuple(
                    reference.evidence_id for reference in by_id[state.node_id].evidence
                ),
                path=state.trace_path,
            )
            for rank, state in enumerate(ordered[:budget], start=1)
        )
        watermark = max(traversed_at).isoformat() if traversed_at else GRAPH_PROJECTION_VERSION
        return CandidateBatch(
            lane=self.signal,
            candidates=candidates,
            examined_count=examined_count,
            latency_ms=(perf_counter() - started) * 1000.0,
            truncated=truncated or len(ordered) > budget,
            projection_watermark=watermark,
        )


def in_memory_retrieval_gateways(
    kernel: InMemoryKernel,
) -> tuple[
    InMemoryRetrievalGateway | InMemoryTemporalRetrievalGateway | InMemoryGraphRetrievalGateway,
    ...,
]:
    primary = tuple(
        InMemoryRetrievalGateway(kernel, signal)
        for signal in (
            RetrievalSignal.EXACT,
            RetrievalSignal.LEXICAL,
            RetrievalSignal.FUZZY,
        )
    )
    return (
        *primary,
        InMemoryTemporalRetrievalGateway(kernel),
        InMemoryGraphRetrievalGateway(kernel),
    )


def in_memory_hybrid_retrieval_gateways(
    kernel: InMemoryKernel,
    embedding_index: EmbeddingIndex,
    *,
    dense_min_similarity: float = 0.0,
) -> tuple[
    InMemoryRetrievalGateway
    | InMemoryTemporalRetrievalGateway
    | InMemoryDenseRetrievalGateway
    | InMemoryGraphRetrievalGateway,
    ...,
]:
    return (
        *(
            gateway
            for gateway in in_memory_retrieval_gateways(kernel)
            if gateway.signal is not RetrievalSignal.GRAPH
        ),
        InMemoryDenseRetrievalGateway(
            kernel,
            embedding_index,
            min_similarity=dense_min_similarity,
        ),
        InMemoryGraphRetrievalGateway(kernel),
    )


__all__ = [
    "InMemoryDenseRetrievalGateway",
    "InMemoryGraphRetrievalGateway",
    "InMemoryRetrievalGateway",
    "InMemoryTemporalRetrievalGateway",
    "in_memory_hybrid_retrieval_gateways",
    "in_memory_retrieval_gateways",
]
