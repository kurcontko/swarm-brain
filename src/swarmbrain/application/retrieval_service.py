"""Purpose-aware multi-lane retrieval orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from inspect import isawaitable
from uuid import uuid4

from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import MemoryId, utc_now
from swarmbrain.domain.memory import RecallBundle, RecallHit, RecallQuery
from swarmbrain.domain.retrieval import (
    CandidateBatch,
    HydrationRejection,
    RetrievalPurpose,
    RetrievalTrace,
)
from swarmbrain.ports.retrieval import (
    CanonicalMemoryReader,
    RetrievalGateway,
    RetrievalTraceSink,
)
from swarmbrain.retrieval import RetrievalPlanner, parse_query_identifiers, weighted_rrf


@asynccontextmanager
async def _retrieval_snapshot(reader: CanonicalMemoryReader) -> AsyncIterator[None]:
    """Use an adapter snapshot when available without widening the base port."""

    factory = getattr(reader, "retrieval_snapshot", None)
    if factory is None:
        yield
        return
    context = factory()
    if isawaitable(context):
        context = await context
    async with context:
        yield


async def _retrieval_now(reader: CanonicalMemoryReader) -> datetime:
    clock = getattr(reader, "retrieval_now", None)
    if clock is None:
        return utc_now()
    value = clock()
    if isawaitable(value):
        value = await value
    if not isinstance(value, datetime):
        raise TypeError("retrieval_now() must return a datetime")
    return value


@dataclass(frozen=True, slots=True)
class RetrievalExecution:
    bundle: RecallBundle
    trace: RetrievalTrace


class RetrievalService:
    def __init__(
        self,
        gateways: tuple[RetrievalGateway, ...],
        canonical_reader: CanonicalMemoryReader,
        *,
        planner: RetrievalPlanner | None = None,
        trace_sink: RetrievalTraceSink | None = None,
    ) -> None:
        signals = [gateway.signal for gateway in gateways]
        if len(signals) != len(set(signals)):
            raise ValueError("only one retrieval gateway may own a signal lane")
        self.gateways = gateways
        self.canonical_reader = canonical_reader
        self.planner = planner or RetrievalPlanner()
        self.trace_sink = trace_sink

    def snapshot(self) -> AbstractAsyncContextManager[None]:
        """Fence hybrid orchestration to the canonical reader's snapshot."""

        return _retrieval_snapshot(self.canonical_reader)

    async def execute(
        self,
        actor: ActorContext,
        query: RecallQuery,
        *,
        purpose: RetrievalPurpose = RetrievalPurpose.INTERACTIVE_RECALL,
        seed_memory_ids: tuple[MemoryId, ...] = (),
    ) -> RetrievalExecution:
        async with _retrieval_snapshot(self.canonical_reader):
            started_at = await _retrieval_now(self.canonical_reader)
            plan = self.planner.plan(
                actor,
                query,
                purpose=purpose,
                available_signals=(gateway.signal for gateway in self.gateways),
                seed_memory_ids=seed_memory_ids,
            )
            selected = tuple(
                gateway for gateway in self.gateways if gateway.signal in plan.signal_lanes
            )
            raw_batches = await asyncio.gather(
                *(gateway.retrieve(actor, plan, query) for gateway in selected),
                return_exceptions=True,
            )
            batches: list[CandidateBatch] = []
            for gateway, result in zip(selected, raw_batches, strict=True):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                if isinstance(result, Exception):
                    batches.append(
                        CandidateBatch(
                            lane=gateway.signal,
                            examined_count=0,
                            latency_ms=0.0,
                            degraded=True,
                            degradation_reason=result.__class__.__name__,
                        )
                    )
                elif isinstance(result, BaseException):
                    raise result
                else:
                    batches.append(result)

            fused = weighted_rrf(tuple(batches), plan)
            candidate_ids = tuple(candidate.canonical_id for candidate in fused)
            hydrated = await self.canonical_reader.hydrate_recallable(
                actor,
                query,
                candidate_ids,
            )
            completed_at = await _retrieval_now(self.canonical_reader)
        hydrated_by_id = {memory.memory_id: memory for memory in hydrated}
        candidate_domain = self._candidate_domains(tuple(batches))

        eligible: list[RecallHit] = []
        for candidate in fused:
            memory = hydrated_by_id.get(candidate.canonical_id)
            if memory is None or candidate.normalized_score < query.min_score:
                continue
            reasons = list(candidate.reasons)
            reasons.append(f"purpose:{purpose.value}")
            section = self._section(purpose, candidate_domain.get(candidate.canonical_id))
            if section is not None:
                reasons.append(f"section:{section}")
            eligible.append(
                RecallHit(
                    memory=memory,
                    score=candidate.normalized_score,
                    reasons=tuple(dict.fromkeys(reasons)),
                    evidence=memory.evidence if query.include_evidence else (),
                )
            )

        hits = tuple(eligible[: query.limit])
        hydrated_ids = tuple(memory.memory_id for memory in hydrated)
        hydrated_set = frozenset(hydrated_ids)
        trace = RetrievalTrace(
            trace_id=str(uuid4()),
            plan=plan,
            parsed_identifiers=tuple(
                dict.fromkeys((*parse_query_identifiers(query.text), *plan.seed_memory_ids))
            ),
            batches=tuple(batches),
            fused_candidates=fused,
            hydrated_ids=hydrated_ids,
            hydration_rejections=tuple(
                HydrationRejection(canonical_id=candidate_id)
                for candidate_id in candidate_ids
                if candidate_id not in hydrated_set
            ),
            final_ids=tuple(hit.memory.memory_id for hit in hits),
            degraded_lanes=frozenset(batch.lane for batch in batches if batch.degraded),
            abstained=not hits,
            abstention_reason="no_relevant_recallable_candidates" if not hits else None,
            started_at=started_at,
            completed_at=completed_at,
        )
        if self.trace_sink is not None:
            with suppress(Exception):
                await self.trace_sink.record(trace)
        return RetrievalExecution(
            bundle=RecallBundle(
                query=query,
                hits=hits,
                generated_at=completed_at,
                total_candidates=len(fused),
                truncated=(
                    len(eligible) > query.limit or any(batch.truncated for batch in batches)
                ),
            ),
            trace=trace,
        )

    @staticmethod
    def _candidate_domains(batches: tuple[CandidateBatch, ...]) -> dict[str, str]:
        result: dict[str, str] = {}
        for batch in batches:
            for candidate in batch.candidates:
                result.setdefault(candidate.canonical_id, candidate.domain_lane)
        return result

    @staticmethod
    def _section(purpose: RetrievalPurpose, domain_lane: str | None) -> str | None:
        if purpose is not RetrievalPurpose.TASK_BOOTSTRAP:
            return None
        return {
            "handoff": "handoff",
            "playbook": "playbook",
            "execution_history": "prior_attempts",
            "knowledge": "knowledge",
        }.get(domain_lane or "")


__all__ = ["RetrievalExecution", "RetrievalService"]
