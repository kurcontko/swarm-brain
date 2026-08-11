"""Observer/Reflector orchestration for evidence-gated consolidation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from swarmbrain.application.capabilities import require_capability
from swarmbrain.application.memory_policy import memory_content_text
from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.consolidation import (
    ConsolidationActionKind,
    ConsolidationActionResult,
    ConsolidationApplyResult,
    ConsolidationObservation,
    ConsolidationProposal,
    ConsolidationReflection,
    ConsolidationRoute,
    ConsolidationWorkPayload,
    ScheduleConsolidationCommand,
    consolidation_input_sha256,
    consolidation_plan_sha256,
    memory_snapshot_sha256,
)
from swarmbrain.domain.evidence import EvidenceRef
from swarmbrain.domain.memory import (
    Memory,
    MemoryOperation,
    MemoryState,
    RecallQuery,
    RememberCommand,
    Visibility,
)
from swarmbrain.domain.work import EnqueueWorkCommand, EnqueueWorkResult, WorkKind
from swarmbrain.ports.consolidation import ConsolidationProvider, DeterministicConsolidator
from swarmbrain.ports.memory_store import MemoryStore
from swarmbrain.ports.retrieval import CanonicalMemoryReader

from .memory_service import MemoryService
from .work import DurableWorkService

_MAX_PROPOSAL_BYTES = 32_768


def _encoded_size(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


class ConsolidationObserver:
    """Freeze bounded canonical memory windows into durable work items."""

    def __init__(
        self,
        store: MemoryStore,
        canonical_reader: CanonicalMemoryReader,
        work: DurableWorkService,
        *,
        enabled: bool = False,
        use_provider: bool = False,
        max_memories: int = 12,
        max_actions: int = 4,
        max_input_bytes: int = 65_536,
    ) -> None:
        if not 2 <= max_memories <= 32:
            raise ValueError("max_memories must be between 2 and 32")
        if not 1 <= max_actions <= 8:
            raise ValueError("max_actions must be between 1 and 8")
        if not 1024 <= max_input_bytes <= 1_000_000:
            raise ValueError("max_input_bytes must be between 1024 and 1000000")
        self.store = store
        self.canonical_reader = canonical_reader
        self.work = work
        self.enabled = enabled
        self.use_provider = use_provider
        self.max_memories = max_memories
        self.max_actions = max_actions
        self.max_input_bytes = max_input_bytes

    async def observe(
        self,
        actor: ActorContext,
        memory: Memory,
    ) -> EnqueueWorkResult | None:
        """Observe a confirmed write and queue a related bounded window."""

        if not self.enabled or not self._eligible(actor, memory):
            return None
        require_capability(actor, Capability.MEMORY_RECALL)
        bundle = await self.store.recall(
            actor,
            RecallQuery(
                text=memory_content_text(memory.content),
                task_id=memory.task_id,
                states=frozenset({MemoryState.CONFIRMED}),
                limit=self.max_memories,
                include_evidence=True,
            ),
        )
        ids = tuple(
            dict.fromkeys(
                (
                    memory.memory_id,
                    *(hit.memory.memory_id for hit in bundle.hits),
                )
            )
        )[: self.max_memories]
        if len(ids) < 2:
            return None
        digest = hashlib.sha256(":".join(ids).encode("utf-8")).hexdigest()
        return await self.schedule(
            actor,
            ScheduleConsolidationCommand(
                idempotency_key=f"observe:{memory.memory_id}:{memory.version}:{digest[:16]}",
                memory_ids=ids,
                task_id=memory.task_id,
            ),
        )

    async def schedule(
        self,
        actor: ActorContext,
        command: ScheduleConsolidationCommand,
    ) -> EnqueueWorkResult | None:
        require_capability(actor, Capability.MEMORY_RECALL)
        require_capability(actor, Capability.MEMORY_PUBLISH)
        requested_ids = command.memory_ids[: self.max_memories]
        memories = await self.canonical_reader.hydrate_recallable(
            actor,
            RecallQuery(
                text="evidence gated consolidation snapshot",
                task_id=command.task_id,
                states=frozenset({MemoryState.CONFIRMED}),
                limit=100,
                include_evidence=True,
            ),
            requested_ids,
        )
        by_id = {memory.memory_id: memory for memory in memories}
        ordered = tuple(by_id[memory_id] for memory_id in requested_ids if memory_id in by_id)
        if len(ordered) != len(requested_ids):
            return None

        selected: list[Memory] = []
        for memory in ordered:
            if not self._eligible(actor, memory):
                continue
            projected = [
                ConsolidationObservation.from_memory(f"m{index}", item).model_dump(mode="json")
                for index, item in enumerate((*selected, memory))
            ]
            if _encoded_size(projected) > self.max_input_bytes:
                continue
            selected.append(memory)
        if len(selected) < 2:
            return None

        observations = tuple(
            ConsolidationObservation.from_memory(f"m{index}", memory)
            for index, memory in enumerate(selected)
        )
        input_sha256 = consolidation_input_sha256(
            observations,
            use_provider=self.use_provider,
            max_actions=self.max_actions,
        )
        task_ids = {memory.task_id for memory in selected if memory.visibility is Visibility.TASK}
        task_id = next(iter(task_ids)) if len(task_ids) == 1 else command.task_id
        payload = ConsolidationWorkPayload(
            observations=observations,
            use_provider=self.use_provider,
            max_actions=self.max_actions,
            input_sha256=input_sha256,
            task_id=task_id,
        )
        return await self.work.enqueue(
            actor,
            EnqueueWorkCommand(
                idempotency_key=command.idempotency_key,
                kind=WorkKind.CONSOLIDATE_MEMORY,
                subject_id=observations[0].memory_id,
                payload=payload.model_dump(mode="json"),
                dedupe_key=f"snapshot:{input_sha256}",
                task_id=task_id,
                max_attempts=5,
                # The snapshot is immediately runnable on the same logical
                # clock that produced its memory versions.  Do not use the
                # command model's wall-clock default: deterministic runtimes
                # may intentionally operate at a different instant.
                available_at=max(memory.recorded_from for memory in selected),
            ),
        )

    @staticmethod
    def _eligible(actor: ActorContext, memory: Memory) -> bool:
        return (
            memory.run_id == actor.run_id
            and memory.state is MemoryState.CONFIRMED
            and memory.recorded_to is None
            and bool(memory.evidence)
        )


class ConsolidationService:
    """Reflect outside transactions; apply only through governed memory writes."""

    def __init__(
        self,
        memory: MemoryService,
        canonical_reader: CanonicalMemoryReader,
        deterministic: DeterministicConsolidator,
        *,
        provider: ConsolidationProvider | None = None,
    ) -> None:
        self.memory = memory
        self.canonical_reader = canonical_reader
        self.deterministic = deterministic
        self.provider = provider

    async def reflect(self, payload: ConsolidationWorkPayload) -> ConsolidationReflection:
        deterministic = self._canonical_proposals(
            payload,
            await self.deterministic.reflect(payload),
        )
        route = ConsolidationRoute.DETERMINISTIC
        proposals = deterministic
        provider_descriptor = None
        fallback_reason = None
        if payload.use_provider:
            if self.provider is None:
                route = ConsolidationRoute.FALLBACK
                fallback_reason = "provider_unconfigured"
            else:
                try:
                    proposals = self._canonical_proposals(
                        payload,
                        await self.provider.reflect(payload),
                    )
                    provider_descriptor = self.provider.descriptor
                    route = ConsolidationRoute.PROVIDER
                except Exception as exc:
                    route = ConsolidationRoute.FALLBACK
                    fallback_reason = f"provider_{type(exc).__name__}"[:255]
                    proposals = deterministic
        return ConsolidationReflection(
            route=route,
            proposals=proposals,
            input_sha256=payload.input_sha256,
            plan_sha256=consolidation_plan_sha256(proposals),
            provider=provider_descriptor,
            fallback_reason=fallback_reason,
        )

    async def apply(
        self,
        actor: ActorContext,
        payload: ConsolidationWorkPayload,
        reflection: ConsolidationReflection,
        *,
        work_id: str,
    ) -> ConsolidationApplyResult:
        require_capability(actor, Capability.MEMORY_PUBLISH)
        if reflection.input_sha256 != payload.input_sha256:
            raise ValueError("reflection does not belong to the observed snapshot")
        current = await self.canonical_reader.hydrate_recallable(
            actor,
            RecallQuery(
                text="evidence gated consolidation revalidation",
                task_id=payload.task_id,
                states=frozenset({MemoryState.CONFIRMED}),
                limit=100,
                include_evidence=True,
            ),
            tuple(item.memory_id for item in payload.observations),
        )
        current_by_id = {memory.memory_id: memory for memory in current}
        if any(
            (memory := current_by_id.get(observation.memory_id)) is None
            or memory.version != observation.memory_version
            or memory_snapshot_sha256(memory) != observation.memory_sha256
            for observation in payload.observations
        ):
            return ConsolidationApplyResult(
                status="stale_noop",
                input_sha256=payload.input_sha256,
                plan_sha256=reflection.plan_sha256,
                actions=(),
            )

        observations = {item.key: item for item in payload.observations}
        results: list[ConsolidationActionResult] = []
        for index, proposal in enumerate(reflection.proposals):
            if proposal.action is ConsolidationActionKind.NOOP:
                results.append(
                    ConsolidationActionResult(
                        action=proposal.action,
                        operation=MemoryOperation.NOOP,
                    )
                )
                continue
            supports = tuple(observations[key] for key in proposal.support_keys)
            scope = self._derived_scope(supports)
            if scope is None:
                results.append(
                    ConsolidationActionResult(
                        action=proposal.action,
                        operation=MemoryOperation.NOOP,
                    )
                )
                continue
            visibility, task_id = scope
            evidence = self._evidence_union(supports)
            target_id = (
                observations[proposal.target_key].memory_id
                if proposal.target_key is not None
                else None
            )
            proposal_payload = proposal.model_dump(mode="json")
            proposal_sha256 = hashlib.sha256(
                json.dumps(
                    proposal_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            result = await self.memory.publish(
                actor,
                RememberCommand(
                    idempotency_key=f"consolidate:{work_id}:{index}",
                    kind=proposal.kind,
                    content=proposal.content,
                    desired_state=MemoryState.TENTATIVE,
                    visibility=visibility,
                    task_id=task_id,
                    title=proposal.title,
                    tags=tuple(dict.fromkeys((*proposal.tags, "consolidated"))),
                    confidence=min(
                        proposal.confidence,
                        min(item.confidence for item in supports),
                    ),
                    evidence=evidence,
                    supersedes_memory_id=target_id,
                    derived_from_memory_ids=tuple(item.memory_id for item in supports),
                    metadata={
                        "consolidation": {
                            "work_id": work_id,
                            "input_sha256": payload.input_sha256,
                            "plan_sha256": reflection.plan_sha256,
                            "proposal_sha256": proposal_sha256,
                            "action": proposal.action.value,
                            "reason": proposal.reason,
                            "inputs": [
                                {
                                    "memory_id": item.memory_id,
                                    "version": item.memory_version,
                                    "evidence_ids": [
                                        evidence.evidence_id for evidence in item.evidence
                                    ],
                                }
                                for item in supports
                            ],
                            "provider": (
                                reflection.provider.model_dump(mode="json")
                                if reflection.provider is not None
                                else None
                            ),
                        }
                    },
                ),
            )
            results.append(
                ConsolidationActionResult(
                    action=proposal.action,
                    operation=result.operation,
                    memory_id=result.memory.memory_id if result.memory is not None else None,
                    replayed=result.replayed,
                )
            )
        status = "applied" if any(item.memory_id for item in results) else "noop"
        return ConsolidationApplyResult(
            status=status,
            input_sha256=payload.input_sha256,
            plan_sha256=reflection.plan_sha256,
            actions=tuple(results),
        )

    @staticmethod
    def _canonical_proposals(
        payload: ConsolidationWorkPayload,
        proposals: Sequence[ConsolidationProposal],
    ) -> tuple[ConsolidationProposal, ...]:
        known = {item.key for item in payload.observations}
        unique: dict[str, ConsolidationProposal] = {}
        for proposal in proposals:
            if not set(proposal.support_keys).issubset(known):
                raise ValueError("proposal refers to an unknown supporting observation")
            if proposal.target_key is not None and proposal.target_key not in known:
                raise ValueError("proposal refers to an unknown target observation")
            dumped = proposal.model_dump(mode="json")
            if _encoded_size(dumped) > _MAX_PROPOSAL_BYTES:
                raise ValueError("proposal exceeds the local byte limit")
            digest = hashlib.sha256(
                json.dumps(dumped, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            unique.setdefault(digest, proposal)
        priority = {
            ConsolidationActionKind.APPEND: 0,
            ConsolidationActionKind.LINK: 1,
            ConsolidationActionKind.NOOP: 2,
            ConsolidationActionKind.SUPERSEDE: 3,
        }
        ordered = sorted(
            unique.items(),
            key=lambda item: (priority[item[1].action], item[0]),
        )
        return tuple(proposal for _, proposal in ordered[: payload.max_actions])

    @staticmethod
    def _derived_scope(
        supports: Sequence[ConsolidationObservation],
    ) -> tuple[Visibility, str | None] | None:
        task_scoped = [item for item in supports if item.visibility is Visibility.TASK]
        if task_scoped:
            task_ids = {item.task_id for item in task_scoped}
            if len(task_ids) != 1 or None in task_ids:
                return None
            return Visibility.TASK, next(iter(task_ids))
        if any(item.visibility is Visibility.RUN for item in supports):
            return Visibility.RUN, None
        return Visibility.REPOSITORY, None

    @staticmethod
    def _evidence_union(
        supports: Sequence[ConsolidationObservation],
    ) -> tuple[EvidenceRef, ...]:
        by_id: dict[str, EvidenceRef] = {}
        for observation in supports:
            for reference in observation.evidence:
                current = by_id.get(reference.evidence_id)
                if current is not None and current != reference:
                    raise ValueError("one evidence ID resolved to conflicting immutable references")
                by_id[reference.evidence_id] = reference
        return tuple(by_id[key] for key in sorted(by_id))


__all__ = ["ConsolidationObserver", "ConsolidationService"]
