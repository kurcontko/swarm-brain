from __future__ import annotations

from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.common import MemoryId
from swarmbrain.domain.evidence import RejectSourceCommand, SourceRejectionResult
from swarmbrain.domain.memory import (
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
from swarmbrain.domain.work import EnqueueWorkCommand, WorkKind
from swarmbrain.ports.embeddings import EmbeddingIndex, EmbeddingProvider
from swarmbrain.ports.memory_store import MemoryOperationPolicy, MemoryReviewStore, MemoryStore

from .capabilities import require_capability
from .work import DurableWorkService

SEMANTIC_MATCH_REASON = "semantic_match"


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
    ) -> None:
        self.store = store
        self.policy = policy
        self.review_store = review_store
        self.embeddings = embeddings
        self.embedding_index = embedding_index
        self.work = work

    async def publish(self, actor: ActorContext, command: RememberCommand) -> RememberResult:
        require_capability(actor, Capability.MEMORY_PUBLISH)
        if command.desired_state is MemoryState.CONFIRMED:
            require_capability(actor, Capability.MEMORY_CONFIRM)
        if command.desired_state is MemoryState.REFUTED:
            require_capability(actor, Capability.MEMORY_REFUTE)
        result = await self.store.remember(actor, command, self.policy)
        # Enqueued after commit and idempotent by dedupe key: a crash between
        # the two is repaired by the client's retried publish, which replays
        # the memory and re-attempts this enqueue.
        if self.work is not None and self.embeddings is not None and result.memory is not None:
            key = f"embed:{result.memory.memory_id}:{self.embeddings.model_name}"
            await self.work.enqueue(
                actor,
                EnqueueWorkCommand(
                    idempotency_key=key[:128],
                    kind=WorkKind.EMBED_MEMORY,
                    subject_id=result.memory.memory_id,
                    payload={
                        "content": result.memory.content,
                        "model": self.embeddings.model_name,
                    },
                    dedupe_key=key[:512],
                    # Anchored to the memory so a replayed publish produces a
                    # byte-identical command and hits the idempotency record.
                    available_at=result.memory.recorded_from,
                ),
            )
        return result

    async def recall(self, actor: ActorContext, query: RecallQuery) -> RecallBundle:
        require_capability(actor, Capability.MEMORY_RECALL)
        bundle = await self.store.recall(actor, query)
        if self.embeddings is None or self.embedding_index is None:
            return bundle
        return await self._merge_semantic(actor, query, bundle)

    async def _merge_semantic(
        self,
        actor: ActorContext,
        query: RecallQuery,
        bundle: RecallBundle,
    ) -> RecallBundle:
        """Blend ANN matches into lexical recall without bypassing store filters.

        The index returns ids and scores only; any id the lexical pass did not
        return is re-fetched through ``store.recall`` so visibility, state, and
        temporal rules still decide whether the actor may read it.
        """

        assert self.embeddings is not None and self.embedding_index is not None
        query_vector = await self.embeddings.embed_query(query.text)
        matches = await self.embedding_index.search_embeddings(
            actor,
            query_vector,
            model=self.embeddings.model_name,
            limit=query.limit,
        )
        if not matches:
            return bundle
        semantic = {match.memory_id: match.score for match in matches}

        merged: dict[str, RecallHit] = {}
        for hit in bundle.hits:
            memory_id = hit.memory.memory_id
            score = semantic.get(memory_id, 0.0)
            merged[memory_id] = hit if score <= hit.score else self._boost(hit, score)

        missing = frozenset(
            memory_id
            for memory_id, score in semantic.items()
            if memory_id not in merged and score >= query.min_score
        )
        if missing:
            supplement = await self.store.recall(
                actor,
                query.model_copy(
                    update={
                        "memory_ids": missing,
                        "min_score": 0.0,
                        "limit": min(100, max(1, len(missing))),
                    }
                ),
            )
            for hit in supplement.hits:
                merged[hit.memory.memory_id] = self._boost(
                    hit, max(hit.score, semantic[hit.memory.memory_id])
                )

        ranked = sorted(
            merged.values(),
            key=lambda hit: (hit.score, hit.memory.recorded_from, hit.memory.memory_id),
            reverse=True,
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
                "score": score,
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
