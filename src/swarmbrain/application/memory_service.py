from __future__ import annotations

import hashlib
from collections.abc import Sequence
from contextlib import suppress
from typing import Protocol

from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.common import MemoryId
from swarmbrain.domain.evidence import RejectSourceCommand, SourceRejectionResult
from swarmbrain.domain.exploration import ReadExpandMemoryRequest, ReadExpandMemoryResult
from swarmbrain.domain.memory import (
    Memory,
    MemoryLineage,
    MemoryReviewDecision,
    MemoryState,
    RecallBundle,
    RecallHit,
    RecallQuery,
    RememberCommand,
    RememberResult,
    ReviewMemoryCommand,
    ReviewMemoryResult,
)
from swarmbrain.domain.retrieval import DenseQuery, RetrievalPurpose, RetrievalSignal
from swarmbrain.domain.work import EnqueueWorkCommand, WorkKind
from swarmbrain.ports.embeddings import EmbeddingIndex, EmbeddingProvider
from swarmbrain.ports.memory_store import MemoryOperationPolicy, MemoryReviewStore, MemoryStore
from swarmbrain.ports.retrieval import CanonicalMemoryReader

from .capabilities import require_capability
from .memory_policy import memory_content_text
from .retrieval_service import RetrievalExecution, RetrievalService
from .work import DurableWorkService

SEMANTIC_MATCH_REASON = "semantic_match"


class MemoryWriteObserver(Protocol):
    async def observe(self, actor: ActorContext, memory: Memory) -> object: ...


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        policy: MemoryOperationPolicy,
        *,
        review_store: MemoryReviewStore | None = None,
        embeddings: EmbeddingProvider | None = None,
        embedding_index: EmbeddingIndex | None = None,
        work: DurableWorkService | None = None,
        retrieval: RetrievalService | None = None,
        canonical_reader: CanonicalMemoryReader | None = None,
        write_observer: MemoryWriteObserver | None = None,
    ) -> None:
        self.store = store
        self.policy = policy
        self.review_store = review_store
        self.embeddings = embeddings
        self.embedding_index = embedding_index
        self.work = work
        self.retrieval = retrieval
        self.canonical_reader = canonical_reader or (
            store if isinstance(store, CanonicalMemoryReader) else None
        )
        self.write_observer = write_observer

    async def publish(self, actor: ActorContext, command: RememberCommand) -> RememberResult:
        require_capability(actor, Capability.MEMORY_PUBLISH)
        if command.desired_state is MemoryState.CONFIRMED:
            require_capability(actor, Capability.MEMORY_CONFIRM)
        if command.desired_state is MemoryState.REFUTED:
            require_capability(actor, Capability.MEMORY_REFUTE)
        result = await self.store.remember(actor, command, self.policy)
        # The memory transaction commits before external work is enqueued.  A
        # retried publish replays the same memory and re-attempts this
        # deterministic enqueue, repairing a crash in between the two steps.
        if self.work is not None and self.embeddings is not None and result.memory is not None:
            model = self.embeddings.model_name
            dedupe_key = f"embed:{result.memory.memory_id}:{model}"
            idempotency_digest = hashlib.sha256(dedupe_key.encode("utf-8")).hexdigest()
            await self.work.enqueue(
                actor,
                EnqueueWorkCommand(
                    idempotency_key=f"embed:{idempotency_digest}",
                    kind=WorkKind.EMBED_MEMORY,
                    subject_id=result.memory.memory_id,
                    payload={
                        # Embedding providers consume text.  Structured memory
                        # remains canonical in MemoryStore and crosses this
                        # boundary only through its stable JSON projection.
                        "content": memory_content_text(result.memory.content),
                        "model": model,
                    },
                    dedupe_key=dedupe_key,
                    available_at=result.memory.recorded_from,
                ),
            )
        if self.write_observer is not None and result.memory is not None:
            await self.write_observer.observe(actor, result.memory)
        return result

    async def recall(
        self,
        actor: ActorContext,
        query: RecallQuery,
        *,
        purpose: RetrievalPurpose = RetrievalPurpose.INTERACTIVE_RECALL,
        seed_memory_ids: tuple[MemoryId, ...] = (),
        token_budget: int | None = None,
    ) -> RecallBundle:
        recalled = await self._recall(
            actor,
            query,
            purpose=purpose,
            seed_memory_ids=seed_memory_ids,
            token_budget=token_budget,
            include_execution=False,
        )
        assert isinstance(recalled, RecallBundle)
        return recalled

    async def recall_for_activation(
        self,
        actor: ActorContext,
        query: RecallQuery,
        *,
        purpose: RetrievalPurpose,
        seed_memory_ids: tuple[MemoryId, ...],
        token_budget: int | None,
    ) -> RecallBundle | RetrievalExecution:
        """Recall while retaining the trace used by selective activation."""

        return await self._recall(
            actor,
            query,
            purpose=purpose,
            seed_memory_ids=seed_memory_ids,
            token_budget=token_budget,
            include_execution=True,
        )

    async def read_expand(
        self,
        actor: ActorContext,
        request: ReadExpandMemoryRequest,
    ) -> ReadExpandMemoryResult:
        """Perform a lease-validated caller's bounded iterative read step."""

        require_capability(actor, Capability.MEMORY_RECALL)
        if self.retrieval is None:
            raise RuntimeError("read-expand requires the canonical retrieval service")
        return await self.retrieval.read_expand(actor, request)

    async def _recall(
        self,
        actor: ActorContext,
        query: RecallQuery,
        *,
        purpose: RetrievalPurpose,
        seed_memory_ids: tuple[MemoryId, ...],
        token_budget: int | None,
        include_execution: bool,
    ) -> RecallBundle | RetrievalExecution:
        require_capability(actor, Capability.MEMORY_RECALL)
        used_retrieval = self.retrieval is not None
        execution: RetrievalExecution | None = None
        if self.retrieval is None:
            bundle = await self.store.recall(actor, query)
            bundle = await self._merge_semantic_if_eligible(actor, query, bundle)
        else:
            query_vector: Sequence[float] | None = None
            dense_query: DenseQuery | None = None
            if (
                self._semantic_eligible(query)
                and self.embeddings is not None
                and self.embedding_index is not None
            ):
                # Provider calls stay outside the CockroachDB read transaction;
                # only ANN lookup and canonical hydration join the snapshot.
                try:
                    query_vector = await self.embeddings.embed_query(query.text)
                    dense_query = DenseQuery(
                        model=self.embeddings.model_name,
                        dimensions=self.embeddings.dimensions,
                        values=tuple(query_vector),
                    )
                except Exception as exc:
                    if self.retrieval.has_signal(RetrievalSignal.DENSE):
                        dense_query = DenseQuery(
                            model=self.embeddings.model_name,
                            dimensions=self.embeddings.dimensions,
                            unavailable_reason=exc.__class__.__name__,
                        )
            async with self.retrieval.snapshot():
                execution = await self.retrieval.execute(
                    actor,
                    query,
                    purpose=purpose,
                    seed_memory_ids=seed_memory_ids,
                    dense_query=dense_query,
                    token_budget=token_budget,
                )
                bundle = execution.bundle
                # Transitional compatibility for callers that compose a v1
                # RetrievalService without a dense CandidateBatch gateway.
                # Production runtimes install the v2 lane and therefore fuse
                # dense through the same trace and weighted RRF as all signals.
                if query_vector is not None and not self.retrieval.has_signal(
                    RetrievalSignal.DENSE
                ):
                    with suppress(Exception):
                        bundle = await self._merge_semantic(
                            actor,
                            query,
                            bundle,
                            query_vector=query_vector,
                        )
                        # The compatibility merge did not participate in the
                        # recorded trace, so do not expose that trace as if it
                        # described the final bundle.
                        execution = None
        if used_retrieval:
            recorder = getattr(self.store, "record_retrieval_reuse", None)
            if recorder is not None:
                await recorder(
                    actor,
                    tuple(hit.memory.memory_id for hit in bundle.hits),
                )
        if include_execution and execution is not None:
            return execution
        return bundle

    async def _merge_semantic_if_eligible(
        self,
        actor: ActorContext,
        query: RecallQuery,
        bundle: RecallBundle,
    ) -> RecallBundle:
        semantic_eligible = self._semantic_eligible(query)
        # The v6 ANN plane is current-only and has no bitemporal projection
        # identity. Historical semantic retrieval belongs to phase 3.
        if semantic_eligible and self.embeddings is not None and self.embedding_index is not None:
            # Dense recall is optional; provider/index outages preserve the
            # already-authoritative lexical bundle.
            with suppress(Exception):
                bundle = await self._merge_semantic(actor, query, bundle)
        return bundle

    @staticmethod
    def _semantic_eligible(query: RecallQuery) -> bool:
        return not (
            query.recorded_at is not None
            or query.world_at is not None
            or query.referenced_valid_from is not None
            or query.include_refuted
            or query.include_superseded
        )

    async def _merge_semantic(
        self,
        actor: ActorContext,
        query: RecallQuery,
        bundle: RecallBundle,
        *,
        query_vector: Sequence[float] | None = None,
    ) -> RecallBundle:
        """Blend ANN scores without bypassing authoritative recall filters."""

        assert self.embeddings is not None and self.embedding_index is not None
        if query_vector is None:
            query_vector = await self.embeddings.embed_query(query.text)
        candidate_limit = min(100, max(query.limit, query.limit * 5))
        matches = await self.embedding_index.search_embeddings(
            actor,
            query_vector,
            model=self.embeddings.model_name,
            limit=candidate_limit,
            min_score=query.min_score,
        )
        if query.memory_ids:
            matches = tuple(match for match in matches if match.memory_id in query.memory_ids)
        if not matches:
            return bundle

        semantic = {match.memory_id: match.score for match in matches}
        stable_order = {hit.memory.memory_id: rank for rank, hit in enumerate(bundle.hits)}
        next_rank = len(stable_order)
        for match in matches:
            if match.memory_id not in stable_order:
                stable_order[match.memory_id] = next_rank
                next_rank += 1
        merged: dict[str, RecallHit] = {}
        for hit in bundle.hits:
            memory_id = hit.memory.memory_id
            score = semantic.get(memory_id)
            merged[memory_id] = hit if score is None else self._boost(hit, score)

        missing = frozenset(memory_id for memory_id in semantic if memory_id not in merged)
        if missing and self.canonical_reader is not None:
            memories = await self.canonical_reader.hydrate_recallable(
                actor,
                query,
                tuple(sorted(missing)),
            )
            for memory in memories:
                merged[memory.memory_id] = RecallHit(
                    memory=memory,
                    score=semantic[memory.memory_id],
                    reasons=(SEMANTIC_MATCH_REASON,),
                    evidence=memory.evidence if query.include_evidence else (),
                )

        ranked = sorted(
            merged.values(),
            key=lambda hit: (
                -hit.score,
                stable_order[hit.memory.memory_id],
                -hit.memory.recorded_from.timestamp(),
                hit.memory.memory_id,
            ),
        )
        return RecallBundle(
            query=query,
            hits=tuple(ranked[: query.limit]),
            generated_at=bundle.generated_at,
            total_candidates=max(bundle.total_candidates, len(merged)),
            truncated=bundle.truncated or len(merged) > query.limit,
        )

    @staticmethod
    def _boost(hit: RecallHit, score: float) -> RecallHit:
        return hit.model_copy(
            update={
                "score": max(hit.score, score),
                "reasons": tuple(dict.fromkeys((*hit.reasons, SEMANTIC_MATCH_REASON))),
            }
        )

    async def lineage(self, actor: ActorContext, memory_id: MemoryId) -> MemoryLineage:
        require_capability(actor, Capability.MEMORY_RECALL)
        return await self.store.get_lineage(actor, memory_id)

    async def reject_source(
        self,
        actor: ActorContext,
        command: RejectSourceCommand,
    ) -> SourceRejectionResult:
        require_capability(actor, Capability.SOURCE_REVIEW)
        return await self.store.reject_source(actor, command)

    async def review(
        self,
        actor: ActorContext,
        command: ReviewMemoryCommand,
    ) -> ReviewMemoryResult:
        if self.review_store is None:
            raise RuntimeError("configured memory store does not support review operations")
        capability = (
            Capability.MEMORY_CONFIRM
            if command.decision is MemoryReviewDecision.CONFIRM
            else Capability.MEMORY_REFUTE
        )
        require_capability(actor, capability)
        return await self.review_store.review_memory(actor, command)
