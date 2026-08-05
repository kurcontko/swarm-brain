"""Atomic in-memory reference implementation of the durable work contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4, uuid5

from swarmbrain.application.errors import IdempotencyConflict, ResourceNotFound
from swarmbrain.application.memory_policy import memory_content_text
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import utc_now
from swarmbrain.domain.evidence import EvidenceSource, SourceReviewState, SourceTrust
from swarmbrain.domain.extraction import (
    ExtractionInput,
    ExtractionProvenance,
    PreparedSourceIngest,
    SourceChunk,
    SourceIngestResult,
)
from swarmbrain.domain.memory import EmbeddingMatch, EmbeddingVector
from swarmbrain.domain.work import (
    AppliedWorkEffect,
    ApplyEmbeddingWorkCommand,
    ApplyExtractionWorkCommand,
    ApplyWorkResult,
    ClaimWorkCommand,
    CompleteWorkCommand,
    CompleteWorkResult,
    EnqueueWorkResult,
    FailWorkCommand,
    PreparedWorkEnqueue,
    SourceExtractionStatus,
    WorkCompletionConflict,
    WorkEffect,
    WorkEffectConflict,
    WorkFailureConflict,
    WorkItem,
    WorkKind,
    WorkLease,
    WorkLeaseBatch,
    WorkLeaseLost,
    WorkStatus,
    durable_work_id,
    source_extraction_state,
)


@dataclass(frozen=True, slots=True)
class _StoredSource:
    actor: ActorContext
    source: EvidenceSource
    raw_content: str


@dataclass(frozen=True, slots=True)
class _IngestReplay:
    fingerprint: str
    result: SourceIngestResult


@dataclass(frozen=True, slots=True)
class _EnqueueReplay:
    fingerprint: str
    result: EnqueueWorkResult


@dataclass(frozen=True, slots=True)
class _StoredEmbedding:
    tenant_id: str
    project_id: str
    repository_id: str
    vector: EmbeddingVector


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class InMemoryWorkStore:
    """Reference lease/fence/effect semantics for local tests and development."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._clock = clock
        self._lock = asyncio.Lock()
        self.sources: dict[str, _StoredSource] = {}
        self.chunks: dict[str, tuple[SourceChunk, ...]] = {}
        self.items: dict[str, WorkItem] = {}
        self.attempts: dict[tuple[str, int, str], ExtractionProvenance] = {}
        self.effects: dict[tuple[str, str], AppliedWorkEffect] = {}
        self.effect_resources: dict[str, dict[str, Any]] = {}
        self.events: dict[str, dict[str, Any]] = {}
        self.outbox_events: dict[str, dict[str, Any]] = {}
        self.memory_embeddings: dict[tuple[str, str], _StoredEmbedding] = {}
        self._ingest_replays: dict[tuple[str, str, str, str], _IngestReplay] = {}
        self._enqueue_replays: dict[tuple[str, str, str, str], _EnqueueReplay] = {}
        self._work_dedupe: dict[tuple[str, str, WorkKind, str], str] = {}

    async def enqueue_work(
        self,
        actor: ActorContext,
        prepared: PreparedWorkEnqueue,
    ) -> EnqueueWorkResult:
        command = prepared.command
        fingerprint = self._enqueue_fingerprint(prepared)
        replay_key = (
            actor.tenant_id,
            actor.run_id,
            actor.agent_id,
            command.idempotency_key,
        )
        dedupe_key = (actor.tenant_id, actor.run_id, command.kind, command.dedupe_key)
        async with self._lock:
            replay = self._enqueue_replays.get(replay_key)
            if replay is not None:
                if replay.fingerprint != fingerprint:
                    raise IdempotencyConflict(command.idempotency_key)
                return replay.result.model_copy(update={"replayed": True})

            current_id = self._work_dedupe.get(dedupe_key)
            if current_id is not None:
                current = self.items[current_id]
                if (
                    current.work_id != prepared.work_id
                    or current.project_id != actor.project_id
                    or current.repository_id != actor.repository_id
                    or current.swarm_id != actor.swarm_id
                    or current.subject_id != command.subject_id
                    or current.task_id != command.task_id
                    or current.payload != command.payload
                ):
                    raise IdempotencyConflict(command.idempotency_key)
                result = EnqueueWorkResult(item=current, replayed=True)
                self._enqueue_replays[replay_key] = _EnqueueReplay(fingerprint, result)
                return result

            now = self._clock()
            item = WorkItem(
                work_id=prepared.work_id,
                tenant_id=actor.tenant_id,
                project_id=actor.project_id,
                repository_id=actor.repository_id,
                swarm_id=actor.swarm_id,
                run_id=actor.run_id,
                task_id=command.task_id,
                requested_by_agent_id=actor.agent_id,
                kind=command.kind,
                subject_id=command.subject_id,
                payload=command.payload,
                dedupe_key=command.dedupe_key,
                priority=command.priority,
                max_attempts=command.max_attempts,
                available_at=command.available_at,
                created_at=now,
                updated_at=now,
            )
            result = EnqueueWorkResult(item=item)
            self.items[item.work_id] = item
            self._work_dedupe[dedupe_key] = item.work_id
            self._enqueue_replays[replay_key] = _EnqueueReplay(fingerprint, result)
            return result

    async def ingest_source(
        self,
        actor: ActorContext,
        prepared: PreparedSourceIngest,
    ) -> SourceIngestResult:
        fingerprint = self._fingerprint(prepared)
        replay_key = (
            actor.tenant_id,
            actor.run_id,
            actor.agent_id,
            prepared.command.idempotency_key,
        )
        async with self._lock:
            replay = self._ingest_replays.get(replay_key)
            if replay is not None:
                if replay.fingerprint != fingerprint:
                    raise IdempotencyConflict(prepared.command.idempotency_key)
                return replay.result.model_copy(update={"replayed": True})

            current = self.sources.get(prepared.source_id)
            if current is not None and (
                current.actor.tenant_id != actor.tenant_id
                or current.actor.repository_id != actor.repository_id
                or current.actor.run_id != actor.run_id
                or current.source.content_sha256 != prepared.command.content_sha256
            ):
                raise IdempotencyConflict(prepared.command.idempotency_key)
            if current is not None:
                result = SourceIngestResult(
                    source=current.source,
                    work_id=prepared.work_id,
                    chunk_count=len(self.chunks[prepared.source_id]),
                    replayed=True,
                )
                self._ingest_replays[replay_key] = _IngestReplay(fingerprint, result)
                return result

            now = self._clock()
            occurrence = prepared.command.occurrence_key or prepared.command.idempotency_key
            source = EvidenceSource(
                source_id=prepared.source_id,
                run_id=actor.run_id,
                task_id=prepared.command.task_id,
                kind=prepared.command.kind,
                uri=prepared.command.uri,
                content_sha256=prepared.command.content_sha256,
                occurrence_key=occurrence,
                trust=prepared.command.trust,
                review_state=SourceReviewState.PENDING,
                observed_at=prepared.command.observed_at or now,
                recorded_at=now,
                metadata=prepared.command.metadata,
            )
            cancellation_reason = self._cancellation_reason(prepared)
            item = WorkItem(
                work_id=prepared.work_id,
                tenant_id=actor.tenant_id,
                project_id=actor.project_id,
                repository_id=actor.repository_id,
                swarm_id=actor.swarm_id,
                run_id=actor.run_id,
                task_id=prepared.command.task_id,
                requested_by_agent_id=actor.agent_id,
                kind=WorkKind.EXTRACT_SOURCE,
                subject_id=prepared.source_id,
                payload={
                    "extractor_name": prepared.extractor_name,
                    "extractor_revision": prepared.extractor_revision,
                    "use_provider": prepared.command.use_provider,
                    "embedding_model": prepared.embedding_model,
                },
                dedupe_key=prepared.work_dedupe_key,
                priority=prepared.command.priority,
                status=(
                    WorkStatus.CANCELLED if cancellation_reason is not None else WorkStatus.PENDING
                ),
                outcome=cancellation_reason,
                available_at=now,
                created_at=now,
                updated_at=now,
            )
            result = SourceIngestResult(
                source=source,
                work_id=prepared.work_id,
                chunk_count=len(prepared.chunks),
            )
            self.sources[prepared.source_id] = _StoredSource(
                actor=actor,
                source=source,
                raw_content=prepared.command.content,
            )
            self.chunks[prepared.source_id] = prepared.chunks
            self.items[prepared.work_id] = item
            self._work_dedupe[
                (
                    actor.tenant_id,
                    actor.run_id,
                    WorkKind.EXTRACT_SOURCE,
                    prepared.work_dedupe_key,
                )
            ] = prepared.work_id
            self._ingest_replays[replay_key] = _IngestReplay(fingerprint, result)
            return result

    async def get_extraction_status(
        self,
        actor: ActorContext,
        source_id: str,
    ) -> SourceExtractionStatus:
        async with self._lock:
            stored = self.sources.get(source_id)
            if stored is None or (
                stored.actor.tenant_id,
                stored.actor.project_id,
                stored.actor.repository_id,
                stored.actor.swarm_id,
                stored.actor.run_id,
            ) != (
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ):
                raise ResourceNotFound("source", source_id)
            item = next(
                (
                    candidate
                    for candidate in self.items.values()
                    if candidate.kind is WorkKind.EXTRACT_SOURCE
                    and candidate.subject_id == source_id
                ),
                None,
            )
            if item is None:
                raise ResourceNotFound("source extraction", source_id)
            memory_ids = tuple(
                sorted(
                    effect.resource_id
                    for (work_id, _), effect in self.effects.items()
                    if work_id == item.work_id
                )
            )
            return SourceExtractionStatus(
                source_id=source_id,
                work_id=item.work_id,
                status=source_extraction_state(item.status),
                attempts=item.attempts,
                max_attempts=item.max_attempts,
                outcome=item.outcome,
                route=(item.result or {}).get("route"),
                candidate_count=(item.result or {}).get("candidate_count"),
                memory_ids=memory_ids,
                created_at=item.created_at,
                updated_at=item.updated_at,
                completed_at=item.completed_at,
            )

    async def claim_work(self, command: ClaimWorkCommand) -> WorkLeaseBatch:
        now = self._clock()
        async with self._lock:
            for work_id, item in tuple(self.items.items()):
                if (
                    item.status is WorkStatus.LEASED
                    and item.locked_until is not None
                    and item.locked_until <= now
                    and item.attempts >= item.max_attempts
                ):
                    self.items[work_id] = item.model_copy(
                        update={
                            "status": WorkStatus.FAILED,
                            "outcome": "failed",
                            "last_error": "lease_expired",
                            "locked_by": None,
                            "lease_token": None,
                            "locked_until": None,
                            "completed_at": now,
                            "updated_at": now,
                            "version": item.version + 1,
                        }
                    )
            candidates = [
                item
                for item in self.items.values()
                if item.kind in command.kinds
                and item.attempts < item.max_attempts
                and item.available_at <= now
                and (
                    item.kind is not WorkKind.EXTRACT_SOURCE
                    or command.extractor_name is None
                    or (
                        item.payload.get("extractor_name") == command.extractor_name
                        and item.payload.get("extractor_revision") == command.extractor_revision
                    )
                )
                and (
                    item.kind is not WorkKind.EMBED_MEMORY
                    or command.embedding_model is None
                    or item.payload.get("model") == command.embedding_model
                )
                and (
                    item.status in {WorkStatus.PENDING, WorkStatus.RETRY}
                    or (
                        item.status is WorkStatus.LEASED
                        and item.locked_until is not None
                        and item.locked_until <= now
                    )
                )
                and self._source_is_eligible(item)
            ]
            candidates.sort(
                key=lambda item: (
                    -item.priority,
                    item.available_at,
                    item.created_at,
                    item.work_id,
                )
            )
            leases: list[WorkLease] = []
            for item in candidates[: command.limit]:
                token = str(uuid4())
                locked_until = now + timedelta(seconds=command.lease_seconds)
                claimed = item.model_copy(
                    update={
                        "status": WorkStatus.LEASED,
                        "attempts": item.attempts + 1,
                        "locked_by": command.worker_id,
                        "lease_token": token,
                        "lease_version": item.lease_version + 1,
                        "locked_until": locked_until,
                        "updated_at": now,
                        "version": item.version + 1,
                    }
                )
                self.items[item.work_id] = claimed
                leases.append(
                    WorkLease(
                        item=claimed,
                        worker_id=command.worker_id,
                        lease_token=token,
                        lease_version=claimed.lease_version,
                        work_version=claimed.version,
                        attempt=claimed.attempts,
                        locked_until=locked_until,
                    )
                )
            return WorkLeaseBatch(leases=tuple(leases))

    async def load_extraction_input(self, lease: WorkLease) -> ExtractionInput:
        async with self._lock:
            item = self.items.get(lease.item.work_id)
            self._require_lease(
                item,
                worker_id=lease.worker_id,
                lease_token=lease.lease_token,
                lease_version=lease.lease_version,
                work_version=lease.work_version,
                attempt=lease.attempt,
            )
            assert item is not None
            stored = self.sources[item.subject_id]
            return ExtractionInput(
                work_id=item.work_id,
                attempt=item.attempts,
                source=stored.source,
                raw_content=stored.raw_content,
                chunks=self.chunks[item.subject_id],
            )

    async def apply_extraction(
        self,
        command: ApplyExtractionWorkCommand,
    ) -> ApplyWorkResult:
        async with self._lock:
            item = self.items.get(command.work_id)
            if item is not None and item.status is WorkStatus.COMPLETED:
                self._require_completion_fence(item, command)
                applied = self._existing_effects(command)
                return ApplyWorkResult(item=item, effects=applied, replayed=True)

            self._require_lease(
                item,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                lease_version=command.lease_version,
                work_version=command.expected_work_version,
                attempt=command.attempt,
            )
            assert item is not None
            self._validate_effect_spans(item, command.effects)

            new_effects: list[AppliedWorkEffect] = []
            new_resources: dict[str, dict[str, Any]] = {}
            new_events: dict[str, dict[str, Any]] = {}
            new_outbox_events: dict[str, dict[str, Any]] = {}
            for effect in command.effects:
                key = (command.work_id, effect.effect_key)
                current = self.effects.get(key)
                if current is not None:
                    if (
                        current.payload_sha256 != effect.payload_sha256
                        or current.kind is not effect.kind
                    ):
                        raise WorkEffectConflict(effect.effect_key)
                    new_effects.append(current)
                    continue
                resource_id = str(
                    uuid5(UUID(command.work_id), f"effect:{effect.effect_key}:memory")
                )
                applied = AppliedWorkEffect(
                    work_id=command.work_id,
                    effect_key=effect.effect_key,
                    kind=effect.kind,
                    payload_sha256=effect.payload_sha256,
                    resource_id=resource_id,
                )
                new_effects.append(applied)
                event_id = str(uuid5(UUID(command.work_id), f"effect:{effect.effect_key}:event"))
                outbox_id = str(uuid5(UUID(command.work_id), f"effect:{effect.effect_key}:outbox"))
                new_resources[resource_id] = {
                    "state": "tentative",
                    "visibility": "run",
                    "candidate": effect.candidate.model_dump(mode="json"),
                    "evidence": [span.model_dump(mode="json") for span in effect.candidate.spans],
                    "supersedes_memory_id": None,
                }
                new_events[event_id] = {
                    "event_type": "memory.added",
                    "aggregate_id": resource_id,
                }
                new_outbox_events[outbox_id] = {
                    "event_id": event_id,
                    "dedupe_key": f"event:{event_id}",
                }

            now = self._clock()
            new_embedding_items: dict[str, WorkItem] = {}
            embedding_model = item.payload.get("embedding_model")
            if isinstance(embedding_model, str):
                for effect, applied_effect in zip(
                    command.effects,
                    new_effects,
                    strict=True,
                ):
                    embedding_item = self._embedding_work_item(
                        item,
                        applied_effect.resource_id,
                        memory_content_text(effect.candidate.content),
                        embedding_model,
                        now,
                    )
                    dedupe_key = (
                        item.tenant_id,
                        item.run_id,
                        WorkKind.EMBED_MEMORY,
                        embedding_item.dedupe_key,
                    )
                    current_id = self._work_dedupe.get(dedupe_key)
                    if current_id is not None:
                        current = self.items[current_id]
                        if (
                            current.work_id != embedding_item.work_id
                            or current.subject_id != embedding_item.subject_id
                            or current.payload != embedding_item.payload
                        ):
                            raise WorkEffectConflict(f"embedding_work:{applied_effect.resource_id}")
                        continue
                    new_embedding_items[embedding_item.work_id] = embedding_item

            completed = item.model_copy(
                update={
                    "status": WorkStatus.COMPLETED,
                    "outcome": command.outcome,
                    "result": command.result,
                    "last_error": None,
                    "locked_until": None,
                    "completed_at": now,
                    "updated_at": now,
                    "version": item.version + 1,
                }
            )
            for provenance in command.provenance:
                attempt_key = (command.work_id, provenance.attempt, provenance.stage.value)
                current = self.attempts.get(attempt_key)
                if current is not None and current != provenance:
                    raise WorkEffectConflict(
                        f"attempt:{provenance.attempt}:{provenance.stage.value}"
                    )

            for applied in new_effects:
                self.effects[(command.work_id, applied.effect_key)] = applied
            self.effect_resources.update(new_resources)
            self.events.update(new_events)
            self.outbox_events.update(new_outbox_events)
            for embedding_item in new_embedding_items.values():
                self.items[embedding_item.work_id] = embedding_item
                self._work_dedupe[
                    (
                        embedding_item.tenant_id,
                        embedding_item.run_id,
                        WorkKind.EMBED_MEMORY,
                        embedding_item.dedupe_key,
                    )
                ] = embedding_item.work_id
            for provenance in command.provenance:
                self.attempts[(command.work_id, provenance.attempt, provenance.stage.value)] = (
                    provenance
                )
            self.items[command.work_id] = completed
            return ApplyWorkResult(item=completed, effects=tuple(new_effects))

    async def apply_embedding(
        self,
        command: ApplyEmbeddingWorkCommand,
    ) -> CompleteWorkResult:
        async with self._lock:
            item = self.items.get(command.work_id)
            if item is not None and item.status is WorkStatus.COMPLETED:
                self._require_completion_fence(item, command)
                if item.outcome != command.outcome or (item.result or {}) != command.result:
                    raise WorkCompletionConflict(command.work_id)
                stored = self.memory_embeddings.get(
                    (command.vector.memory_id, command.vector.model)
                )
                if stored is None or (
                    stored.vector.dimensions != command.vector.dimensions
                    or stored.vector.values != command.vector.values
                ):
                    raise WorkCompletionConflict("completed_embedding_vector")
                return CompleteWorkResult(item=item, replayed=True)

            self._require_lease(
                item,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                lease_version=command.lease_version,
                work_version=command.expected_work_version,
                attempt=command.attempt,
            )
            assert item is not None
            self._validate_embedding_apply(item, command.vector)
            now = self._clock()
            completed = item.model_copy(
                update={
                    "status": WorkStatus.COMPLETED,
                    "outcome": command.outcome,
                    "result": command.result,
                    "last_error": None,
                    "locked_until": None,
                    "completed_at": now,
                    "updated_at": now,
                    "version": item.version + 1,
                }
            )
            self.memory_embeddings[(command.vector.memory_id, command.vector.model)] = (
                _StoredEmbedding(
                    tenant_id=item.tenant_id,
                    project_id=item.project_id,
                    repository_id=item.repository_id,
                    vector=command.vector,
                )
            )
            self.items[item.work_id] = completed
            return CompleteWorkResult(item=completed)

    async def complete_work(self, command: CompleteWorkCommand) -> CompleteWorkResult:
        async with self._lock:
            item = self.items.get(command.work_id)
            if item is not None and item.status is WorkStatus.COMPLETED:
                self._require_completion_fence(item, command)
                if item.outcome != command.outcome or (item.result or {}) != command.result:
                    raise WorkCompletionConflict(command.work_id)
                return CompleteWorkResult(item=item, replayed=True)

            self._require_lease(
                item,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                lease_version=command.lease_version,
                work_version=command.expected_work_version,
                attempt=command.attempt,
            )
            assert item is not None
            if item.kind in {WorkKind.EXTRACT_SOURCE, WorkKind.EMBED_MEMORY}:
                raise WorkCompletionConflict(f"{item.kind.value}_requires_fenced_apply")
            now = self._clock()
            completed = item.model_copy(
                update={
                    "status": WorkStatus.COMPLETED,
                    "outcome": command.outcome,
                    "result": command.result,
                    "last_error": None,
                    "locked_until": None,
                    "completed_at": now,
                    "updated_at": now,
                    "version": item.version + 1,
                }
            )
            self.items[item.work_id] = completed
            return CompleteWorkResult(item=completed)

    async def upsert_embeddings(
        self,
        actor: ActorContext,
        vectors: Sequence[EmbeddingVector],
        *,
        idempotency_key: str,
    ) -> None:
        """Direct index boundary for tests; workers use fenced apply_embedding."""

        del idempotency_key
        async with self._lock:
            for vector in vectors:
                self.memory_embeddings[(vector.memory_id, vector.model)] = _StoredEmbedding(
                    tenant_id=actor.tenant_id,
                    project_id=actor.project_id,
                    repository_id=actor.repository_id,
                    vector=vector,
                )

    async def search_embeddings(
        self,
        actor: ActorContext,
        query_vector: Sequence[float],
        *,
        model: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> tuple[EmbeddingMatch, ...]:
        async with self._lock:
            matches: list[EmbeddingMatch] = []
            for (memory_id, vector_model), stored in self.memory_embeddings.items():
                if vector_model != model or (
                    stored.tenant_id,
                    stored.project_id,
                    stored.repository_id,
                ) != (
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                ):
                    continue
                if len(stored.vector.values) != len(query_vector):
                    continue
                score = max(
                    0.0,
                    min(
                        1.0,
                        _cosine_similarity(stored.vector.values, query_vector),
                    ),
                )
                if score >= min_score:
                    matches.append(EmbeddingMatch(memory_id=memory_id, score=score))
            matches.sort(key=lambda match: (match.score, match.memory_id), reverse=True)
            return tuple(matches[: max(1, min(100, limit))])

    async def fail_work(self, command: FailWorkCommand) -> WorkItem:
        async with self._lock:
            item = self.items.get(command.work_id)
            if item is not None and item.status in {WorkStatus.RETRY, WorkStatus.FAILED}:
                self._require_failure_replay(item, command)
                return item
            self._require_lease(
                item,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                lease_version=command.lease_version,
                work_version=command.expected_work_version,
                attempt=command.attempt,
            )
            assert item is not None
            now = self._clock()
            status = WorkStatus.RETRY if item.attempts < item.max_attempts else WorkStatus.FAILED
            failed = item.model_copy(
                update={
                    "status": status,
                    "available_at": command.retry_at,
                    "locked_until": None,
                    "outcome": "failed" if status is WorkStatus.FAILED else None,
                    "last_error": command.error,
                    "completed_at": now if status is WorkStatus.FAILED else None,
                    "updated_at": now,
                    "version": item.version + 1,
                }
            )
            for provenance in command.provenance:
                self.attempts[(command.work_id, provenance.attempt, provenance.stage.value)] = (
                    provenance
                )
            self.items[command.work_id] = failed
            return failed

    @staticmethod
    def _require_failure_replay(item: WorkItem, command: FailWorkCommand) -> None:
        expected_status = (
            WorkStatus.RETRY if item.attempts < item.max_attempts else WorkStatus.FAILED
        )
        if (
            item.status is not expected_status
            or item.locked_by != command.worker_id
            or item.lease_token != command.lease_token
            or item.lease_version != command.lease_version
            or item.attempts != command.attempt
            or item.version != command.expected_work_version + 1
        ):
            raise WorkLeaseLost(command.work_id)
        if item.last_error != command.error or item.available_at != command.retry_at:
            raise WorkFailureConflict(command.work_id)

    def _source_is_eligible(self, item: WorkItem) -> bool:
        if item.kind is not WorkKind.EXTRACT_SOURCE:
            return True
        source = self.sources.get(item.subject_id)
        return source is not None and (
            source.source.review_state is not SourceReviewState.REJECTED
            and source.source.trust is not SourceTrust.UNTRUSTED
        )

    def _embedding_work_item(
        self,
        parent: WorkItem,
        memory_id: str,
        content: str,
        embedding_model: str,
        now: datetime,
    ) -> WorkItem:
        dedupe_key = f"embed:{memory_id}:{embedding_model}"
        work_id = durable_work_id(
            parent.tenant_id,
            parent.run_id,
            WorkKind.EMBED_MEMORY,
            dedupe_key,
        )
        return WorkItem(
            work_id=work_id,
            tenant_id=parent.tenant_id,
            project_id=parent.project_id,
            repository_id=parent.repository_id,
            swarm_id=parent.swarm_id,
            run_id=parent.run_id,
            task_id=parent.task_id,
            requested_by_agent_id=parent.requested_by_agent_id,
            kind=WorkKind.EMBED_MEMORY,
            subject_id=memory_id,
            payload={
                "content": content,
                "model": embedding_model,
                "metadata": {},
            },
            dedupe_key=dedupe_key,
            priority=parent.priority,
            available_at=now,
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def _validate_embedding_apply(item: WorkItem, vector: EmbeddingVector) -> None:
        if item.kind is not WorkKind.EMBED_MEMORY:
            raise WorkCompletionConflict("apply_embedding_requires_embed_memory")
        if item.subject_id != vector.memory_id or item.payload.get("model") != vector.model:
            raise WorkCompletionConflict("embedding_vector_does_not_match_work")

    @staticmethod
    def _cancellation_reason(prepared: PreparedSourceIngest) -> str | None:
        if prepared.command.trust is SourceTrust.UNTRUSTED:
            return "source_untrusted"
        return None

    def _validate_effect_spans(
        self,
        item: WorkItem,
        effects: Sequence[WorkEffect],
    ) -> None:
        stored = self.sources.get(item.subject_id)
        chunks = {chunk.chunk_index: chunk for chunk in self.chunks.get(item.subject_id, ())}
        if stored is None:
            raise WorkEffectConflict("evidence_source_missing")
        for effect in effects:
            for span in effect.candidate.spans:
                chunk = chunks.get(span.chunk_index)
                if (
                    chunk is None
                    or span.char_start < chunk.char_start
                    or span.char_end > chunk.char_end
                    or stored.raw_content[span.char_start : span.char_end] != span.excerpt
                ):
                    raise WorkEffectConflict(f"evidence_span:{effect.effect_key}")

    @staticmethod
    def _fingerprint(prepared: PreparedSourceIngest) -> str:
        payload = json.dumps(
            prepared.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _enqueue_fingerprint(prepared: PreparedWorkEnqueue) -> str:
        payload = json.dumps(
            prepared.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _existing_effects(
        self,
        command: ApplyExtractionWorkCommand,
    ) -> tuple[AppliedWorkEffect, ...]:
        requested_keys = {effect.effect_key for effect in command.effects}
        durable_keys = {
            effect_key for work_id, effect_key in self.effects if work_id == command.work_id
        }
        if requested_keys != durable_keys:
            raise WorkEffectConflict("completed_effect_set")
        existing: list[AppliedWorkEffect] = []
        for effect in command.effects:
            current = self.effects.get((command.work_id, effect.effect_key))
            if current is None:
                raise WorkEffectConflict(effect.effect_key)
            if current.payload_sha256 != effect.payload_sha256 or current.kind is not effect.kind:
                raise WorkEffectConflict(effect.effect_key)
            existing.append(current)
        return tuple(existing)

    @staticmethod
    def _require_completion_fence(
        item: WorkItem,
        command: (ApplyExtractionWorkCommand | ApplyEmbeddingWorkCommand | CompleteWorkCommand),
    ) -> None:
        if (
            item.locked_by != command.worker_id
            or item.lease_token != command.lease_token
            or item.lease_version != command.lease_version
            or item.attempts != command.attempt
            or item.version != command.expected_work_version + 1
        ):
            raise WorkLeaseLost(command.work_id)

    def _require_lease(
        self,
        item: WorkItem | None,
        *,
        worker_id: str,
        lease_token: str,
        lease_version: int,
        work_version: int,
        attempt: int,
    ) -> None:
        now = self._clock()
        if (
            item is None
            or item.status is not WorkStatus.LEASED
            or item.locked_by != worker_id
            or item.lease_token != lease_token
            or item.lease_version != lease_version
            or item.version != work_version
            or item.attempts != attempt
            or item.locked_until is None
            or item.locked_until <= now
        ):
            raise WorkLeaseLost(item.work_id if item is not None else "unknown")


__all__ = ["InMemoryWorkStore"]
