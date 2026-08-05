"""Deterministic reference retrieval lanes over the local in-memory kernel."""

from __future__ import annotations

import re
from time import perf_counter

from swarmbrain.application.memory_policy import memory_text_sha256
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.memory import Memory, RecallQuery
from swarmbrain.domain.retrieval import (
    Candidate,
    CandidateBatch,
    RetrievalPlan,
    RetrievalPurpose,
    RetrievalSignal,
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
    ) -> CandidateBatch:
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


def in_memory_retrieval_gateways(
    kernel: InMemoryKernel,
) -> tuple[InMemoryRetrievalGateway, ...]:
    return tuple(
        InMemoryRetrievalGateway(kernel, signal)
        for signal in (
            RetrievalSignal.EXACT,
            RetrievalSignal.LEXICAL,
            RetrievalSignal.FUZZY,
        )
    )


__all__ = ["InMemoryRetrievalGateway", "in_memory_retrieval_gateways"]
