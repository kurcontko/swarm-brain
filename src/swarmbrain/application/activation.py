"""Deterministic, content-safe selective memory activation."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from swarmbrain.application.errors import SwarmBrainError
from swarmbrain.domain.activation import (
    ActivationDecision,
    ActivationReason,
    ActivationTrigger,
    MemoryActivationRequest,
    MemoryActivationResult,
    MemoryActivationTelemetry,
)
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.memory import MemoryState, RecallBundle, RecallQuery, Visibility
from swarmbrain.domain.retrieval import RetrievalPurpose, RetrievalTrace
from swarmbrain.retrieval import estimate_tokens, pack_to_budget, render_recall_hit


class MemoryRecaller(Protocol):
    """The narrow portion of :class:`MemoryService` activation consumes."""

    async def recall(
        self,
        actor: ActorContext,
        query: RecallQuery,
        *,
        purpose: RetrievalPurpose,
        seed_memory_ids: tuple[str, ...],
        token_budget: int | None,
    ) -> RecallBundle: ...


_DEEP_TRIGGERS = frozenset(
    {
        ActivationTrigger.TOOL_ERROR,
        ActivationTrigger.REPEATED_FAILURE,
        ActivationTrigger.CHECKPOINT_RESUME,
    }
)
_MAX_ACTIVATION_QUERY_CHARS = 8_192


def _default_purpose(trigger: ActivationTrigger) -> RetrievalPurpose:
    if trigger is ActivationTrigger.CHECKPOINT_RESUME:
        return RetrievalPurpose.HANDOFF_RECOVERY
    if trigger in {ActivationTrigger.TOOL_ERROR, ActivationTrigger.REPEATED_FAILURE}:
        return RetrievalPurpose.PLANNING
    if trigger is ActivationTrigger.EXPLICIT:
        return RetrievalPurpose.INTERACTIVE_RECALL
    return RetrievalPurpose.TASK_BOOTSTRAP


class MemoryActivationService:
    """Turn a server-observed trigger into a bounded memory intervention.

    Query text exists only for this call.  It is placed in ``RecallQuery`` for
    retrieval, but never copied into the request contract or telemetry result.
    Whether the recaller returns a bundle or a richer retrieval execution, the
    service re-renders and re-packs the selected hits canonically. This makes
    the token estimate describe the exact string exposed at HTTP/MCP rather
    than trusting an alternate representation supplied by an adapter.
    """

    def __init__(
        self,
        memory_service: MemoryRecaller,
    ) -> None:
        self.memory_service = memory_service

    async def activate(
        self,
        actor: ActorContext,
        request: MemoryActivationRequest,
        *,
        query_text: str,
    ) -> MemoryActivationResult:
        purpose = request.purpose or _default_purpose(request.trigger)
        bounded_query_text = query_text[:_MAX_ACTIVATION_QUERY_CHARS].strip()
        if not bounded_query_text:
            return self._empty_result(
                actor,
                request,
                purpose=purpose,
                decision=ActivationDecision.SKIP,
                reason=ActivationReason.EMPTY_QUERY,
            )

        deep = request.trigger in _DEEP_TRIGGERS
        query = RecallQuery(
            text=bounded_query_text,
            task_id=request.task_id,
            visibilities=frozenset(Visibility),
            states=frozenset({MemoryState.CONFIRMED}),
            min_score=request.min_score,
            limit=min(100, request.limit * (2 if deep else 1)),
        )
        try:
            recall_for_activation = getattr(self.memory_service, "recall_for_activation", None)
            recall = recall_for_activation or self.memory_service.recall
            recalled: Any = await recall(
                actor,
                query,
                purpose=purpose,
                seed_memory_ids=request.seed_memory_ids,
                token_budget=request.token_budget,
            )
            bundle, trace = self._unpack_recall(recalled)
        except asyncio.CancelledError:
            raise
        except SwarmBrainError:
            # Authentication, authorization, scope, and lease failures are not
            # availability signals and must retain their normal fail-closed path.
            raise
        except Exception:
            # Provider/database outages may defer optional context, but neither
            # exception messages nor the ephemeral query cross into telemetry.
            return self._empty_result(
                actor,
                request,
                purpose=purpose,
                decision=ActivationDecision.DEFER,
                reason=ActivationReason.RECALL_UNAVAILABLE,
            )

        packing = trace.packing if trace is not None else None
        trace_dropped = packing.dropped_ids if packing is not None else ()
        bundle, rendered_context, local_dropped, estimated_tokens = self._pack_bundle(
            bundle,
            request.token_budget,
            limit=request.limit,
        )
        dropped_memory_ids = tuple(dict.fromkeys((*trace_dropped, *local_dropped)))

        if not bundle.hits:
            return self._empty_result(
                actor,
                request,
                purpose=purpose,
                decision=ActivationDecision.SKIP,
                reason=self._safe_abstention_reason(trace),
                bundle=bundle,
                trace=trace,
                dropped_memory_ids=dropped_memory_ids,
            )

        decision = ActivationDecision.DEEP_RECALL if deep else ActivationDecision.RECALL
        memory_ids = tuple(hit.memory.memory_id for hit in bundle.hits)
        return MemoryActivationResult(
            telemetry=MemoryActivationTelemetry(
                activation_id=request.activation_id,
                run_id=actor.run_id,
                agent_id=actor.agent_id,
                task_id=request.task_id,
                lease_id=request.lease_id,
                trigger=request.trigger,
                decision=decision,
                purpose=purpose,
                reason=ActivationReason.CONTEXT_ACTIVATED,
                trace_id=trace.trace_id if trace is not None else None,
                memory_ids=memory_ids,
                memory_versions={hit.memory.memory_id: hit.memory.version for hit in bundle.hits},
                dropped_memory_ids=dropped_memory_ids,
                token_budget=request.token_budget,
                estimated_tokens=estimated_tokens,
                min_score=request.min_score,
                candidate_count=bundle.total_candidates,
                truncated=bundle.truncated,
            ),
            bundle=bundle,
            rendered_context=rendered_context,
        )

    def _unpack_recall(
        self,
        recalled: object,
    ) -> tuple[RecallBundle, RetrievalTrace | None]:
        """Accept today's bundle and a future execution without API churn."""

        if isinstance(recalled, RecallBundle):
            bundle = recalled
            trace = None
        else:
            bundle = getattr(recalled, "bundle", None)
            if not isinstance(bundle, RecallBundle):
                raise TypeError("memory recall returned neither a bundle nor an execution")
            candidate_trace = getattr(recalled, "trace", None)
            trace = candidate_trace if isinstance(candidate_trace, RetrievalTrace) else None
        return bundle, trace

    def _pack_bundle(
        self,
        bundle: RecallBundle,
        token_budget: int,
        *,
        limit: int,
    ) -> tuple[RecallBundle, str, tuple[str, ...], int]:
        """Render and enforce the public activation cap at the delivery boundary."""

        rendered = tuple(
            ("" if index == 0 else "\n\n") + render_recall_hit(hit)
            for index, hit in enumerate(bundle.hits)
        )
        packed = pack_to_budget(
            tuple(estimate_tokens(value) for value in rendered),
            token_budget,
            policy="greedy",
        )
        kept_indices = packed.kept_indices[:limit]
        dropped_indices = tuple(
            dict.fromkeys((*packed.dropped_indices, *packed.kept_indices[limit:]))
        )
        kept_hits = tuple(bundle.hits[index] for index in kept_indices)
        dropped_ids = tuple(bundle.hits[index].memory.memory_id for index in dropped_indices)
        context = "".join(rendered[index] for index in kept_indices).lstrip()
        bounded = bundle.model_copy(
            update={
                "hits": kept_hits,
                "truncated": bundle.truncated or bool(dropped_indices),
            }
        )
        return bounded, context, dropped_ids, estimate_tokens(context)

    @staticmethod
    def _safe_abstention_reason(trace: RetrievalTrace | None) -> ActivationReason:
        """Map internal diagnostics onto the closed telemetry vocabulary."""

        if trace is None or trace.abstention_reason is None:
            return ActivationReason.NO_RELEVANT_MEMORY
        try:
            return ActivationReason(trace.abstention_reason)
        except ValueError:
            return ActivationReason.NO_RELEVANT_MEMORY

    @staticmethod
    def _empty_result(
        actor: ActorContext,
        request: MemoryActivationRequest,
        *,
        purpose: RetrievalPurpose,
        decision: ActivationDecision,
        reason: ActivationReason,
        bundle: RecallBundle | None = None,
        trace: RetrievalTrace | None = None,
        dropped_memory_ids: tuple[str, ...] = (),
    ) -> MemoryActivationResult:
        packing = trace.packing if trace is not None else None
        return MemoryActivationResult(
            telemetry=MemoryActivationTelemetry(
                activation_id=request.activation_id,
                run_id=actor.run_id,
                agent_id=actor.agent_id,
                task_id=request.task_id,
                lease_id=request.lease_id,
                trigger=request.trigger,
                decision=decision,
                purpose=purpose,
                reason=reason,
                trace_id=trace.trace_id if trace is not None else None,
                dropped_memory_ids=(
                    packing.dropped_ids if packing is not None else dropped_memory_ids
                ),
                token_budget=request.token_budget,
                min_score=request.min_score,
                candidate_count=bundle.total_candidates if bundle is not None else 0,
                truncated=bundle.truncated if bundle is not None else False,
            ),
            bundle=bundle,
        )


__all__ = ["MemoryActivationService", "MemoryRecaller"]
