"""Purpose-aware multi-lane retrieval orchestration."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from inspect import isawaitable
from time import perf_counter_ns
from typing import cast
from uuid import uuid4

from swarmbrain.application.errors import SwarmBrainError
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import MemoryId, utc_now
from swarmbrain.domain.exploration import ReadExpandMemoryRequest, ReadExpandMemoryResult
from swarmbrain.domain.memory import (
    Memory,
    MemoryState,
    RecallBundle,
    RecallHit,
    RecallQuery,
    Visibility,
)
from swarmbrain.domain.reranking import (
    LearnedRerankPolicy,
    LearnedRerankRequest,
    LearnedRerankTrace,
)
from swarmbrain.domain.retrieval import (
    CandidateBatch,
    CandidateRelevance,
    DenseQuery,
    FusedCandidate,
    HydrationRejection,
    PackingPolicy,
    PackingTrace,
    RetrievalPurpose,
    RetrievalSignal,
    RetrievalTrace,
)
from swarmbrain.ports.reranking import LearnedRerankerProvider
from swarmbrain.ports.retrieval import (
    CanonicalMemoryReader,
    GraphExpansionGateway,
    RetrievalGateway,
    RetrievalTraceSink,
)
from swarmbrain.retrieval import (
    MAX_FACILITY_CANDIDATES,
    RELEVANCE_VERSION,
    RetrievalPlanner,
    build_packing_features,
    candidate_relevance,
    estimate_tokens,
    pack_to_budget,
    parse_query_identifiers,
    relevance_query,
    relevance_reranked,
    render_recall_hit,
    search_text,
    weighted_rrf,
)
from swarmbrain.retrieval.learned_reranking import (
    LearnedRerankValidationError,
    build_swarm_memory_rerank_request,
    learned_score_reranked,
    validate_learned_rerank_result,
)

_PROVIDER_REQUEST_ID_WINDOW = 65_536


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
    rendered_context: str = ""


class RetrievalService:
    def __init__(
        self,
        gateways: tuple[RetrievalGateway | GraphExpansionGateway, ...],
        canonical_reader: CanonicalMemoryReader,
        *,
        planner: RetrievalPlanner | None = None,
        trace_sink: RetrievalTraceSink | None = None,
        packing_policy: PackingPolicy | str | None = None,
        learned_reranker: LearnedRerankerProvider | None = None,
        learned_rerank_policy: LearnedRerankPolicy | None = None,
    ) -> None:
        signals = [gateway.signal for gateway in gateways]
        if len(signals) != len(set(signals)):
            raise ValueError("only one retrieval gateway may own a signal lane")
        for gateway in gateways:
            if gateway.signal is RetrievalSignal.GRAPH and not isinstance(
                gateway, GraphExpansionGateway
            ):
                raise ValueError("the graph retrieval lane must implement expand()")
        if (learned_reranker is None) != (learned_rerank_policy is None):
            raise ValueError("learned reranker and policy must be configured together")
        if (
            learned_reranker is not None
            and learned_rerank_policy is not None
            and learned_reranker.identity != learned_rerank_policy.identity
        ):
            raise ValueError("learned reranker identity must equal the policy identity")
        self.gateways = gateways
        self.canonical_reader = canonical_reader
        self.planner = planner or RetrievalPlanner()
        self.trace_sink = trace_sink
        self.packing_policy = None if packing_policy is None else PackingPolicy(packing_policy)
        self.learned_reranker = learned_reranker
        self.learned_rerank_policy = learned_rerank_policy
        self._provider_request_ids: set[str] = set()
        self._provider_request_order: deque[str] = deque()

    def has_signal(self, signal: RetrievalSignal) -> bool:
        return any(gateway.signal is signal for gateway in self.gateways)

    def snapshot(self) -> AbstractAsyncContextManager[None]:
        """Fence hybrid orchestration to the canonical reader's snapshot."""

        return _retrieval_snapshot(self.canonical_reader)

    async def read_expand(
        self,
        actor: ActorContext,
        request: ReadExpandMemoryRequest,
    ) -> ReadExpandMemoryResult:
        """Hydrate exact seeds, follow bounded links, and pack one read context.

        Candidate IDs are never trusted as content. Both the requested seeds
        and every graph neighbor are canonically hydrated under one retrieval
        snapshot, which reapplies scope, current lifecycle, valid time, and
        evidence trust before any rendered text crosses the application seam.
        """

        query = RecallQuery(
            text=request.query_text,
            task_id=request.task_id,
            states=frozenset({MemoryState.CONFIRMED}),
            visibilities=frozenset(Visibility),
            include_evidence=request.include_evidence,
            referenced_valid_from=request.referenced_valid_from,
            referenced_valid_to=request.referenced_valid_to,
            limit=100,
        )
        graph_gateway = next(
            (
                cast(GraphExpansionGateway, gateway)
                for gateway in self.gateways
                if gateway.signal is RetrievalSignal.GRAPH
            ),
            None,
        )
        expanded_batch: CandidateBatch | None = None
        graph_degraded = False
        async with _retrieval_snapshot(self.canonical_reader):
            expanded_ids: tuple[MemoryId, ...] = ()
            if request.max_depth > 0 and graph_gateway is not None:
                expansion_limit = min(
                    92,
                    len(request.memory_ids)
                    * sum(request.max_fanout**hop for hop in range(1, request.max_depth + 1)),
                )
                plan = self.planner.plan(
                    actor,
                    query,
                    purpose=RetrievalPurpose.PLANNING,
                    available_signals=(RetrievalSignal.GRAPH,),
                    seed_memory_ids=request.memory_ids,
                ).model_copy(
                    update={
                        "signal_lanes": frozenset({RetrievalSignal.GRAPH}),
                        "lane_budgets": {RetrievalSignal.GRAPH.value: max(1, expansion_limit)},
                        "lane_weights": {RetrievalSignal.GRAPH.value: 1.0},
                        "max_graph_hops": request.max_depth,
                        "graph_seed_limit": len(request.memory_ids),
                        "graph_max_fanout": request.max_fanout,
                        "graph_edge_budget": min(
                            10_000,
                            max(
                                1,
                                len(request.memory_ids)
                                * request.max_fanout
                                * request.max_depth
                                * 8,
                            ),
                        ),
                        "token_budget": request.token_budget,
                    }
                )
                try:
                    expanded_batch = await graph_gateway.expand(actor, plan, query, ())
                    expanded_ids = tuple(
                        candidate.canonical_id for candidate in expanded_batch.candidates
                    )
                except asyncio.CancelledError:
                    raise
                except SwarmBrainError:
                    raise
                except Exception:
                    # Exact reads remain useful when graph projection access is
                    # degraded. Only the closed truncation bit crosses the API.
                    graph_degraded = True
            elif request.max_depth > 0:
                graph_degraded = True

            ordered_ids = tuple(dict.fromkeys((*request.memory_ids, *expanded_ids)))[:100]
            memories = await self.canonical_reader.hydrate_recallable(
                actor,
                query,
                ordered_ids,
            )

        by_id = {memory.memory_id: memory for memory in memories}
        candidate_by_id = {
            candidate.canonical_id: candidate
            for candidate in (expanded_batch.candidates if expanded_batch is not None else ())
        }
        hits: list[RecallHit] = []
        provenance: dict[MemoryId, tuple[str, ...]] = {}
        for memory_id in ordered_ids:
            memory = by_id.get(memory_id)
            if memory is None:
                continue
            candidate = candidate_by_id.get(memory_id)
            reasons = (
                ("explicit_read",)
                if candidate is None
                else tuple(dict.fromkeys((f"signal:{candidate.signal.value}", *candidate.reasons)))
            )
            score = 1.0 if candidate is None else max(0.0, min(1.0, candidate.raw_score or 0.0))
            hits.append(
                RecallHit(
                    memory=memory,
                    score=score,
                    reasons=reasons,
                    evidence=memory.evidence if request.include_evidence else (),
                )
            )
            provenance[memory_id] = reasons

        rendered = tuple(
            ("" if index == 0 else "\n\n") + render_recall_hit(hit)
            for index, hit in enumerate(hits)
        )
        packed = pack_to_budget(
            tuple(estimate_tokens(value) for value in rendered),
            request.token_budget,
            policy="greedy",
        )
        kept_hits = tuple(hits[index] for index in packed.kept_indices)
        selected_ids = tuple(hit.memory.memory_id for hit in kept_hits)
        dropped_ids = tuple(hits[index].memory.memory_id for index in packed.dropped_indices)
        context = "".join(rendered[index] for index in packed.kept_indices).lstrip()
        return ReadExpandMemoryResult(
            task_id=request.task_id,
            lease_id=request.lease_id,
            context=context,
            memory_ids=selected_ids,
            memory_versions={hit.memory.memory_id: hit.memory.version for hit in kept_hits},
            provenance={memory_id: provenance[memory_id] for memory_id in selected_ids},
            dropped_memory_ids=dropped_ids,
            token_budget=request.token_budget,
            estimated_tokens=estimate_tokens(context),
            max_depth=request.max_depth,
            truncated=(
                graph_degraded
                or (expanded_batch.truncated if expanded_batch is not None else False)
                or bool(packed.dropped_indices)
            ),
        )

    async def execute(
        self,
        actor: ActorContext,
        query: RecallQuery,
        *,
        purpose: RetrievalPurpose = RetrievalPurpose.INTERACTIVE_RECALL,
        seed_memory_ids: tuple[MemoryId, ...] = (),
        dense_query: DenseQuery | None = None,
        token_budget: int | None = None,
    ) -> RetrievalExecution:
        async with _retrieval_snapshot(self.canonical_reader):
            started_at = await _retrieval_now(self.canonical_reader)
            available_signals = tuple(
                gateway.signal
                for gateway in self.gateways
                if gateway.signal is not RetrievalSignal.DENSE or dense_query is not None
            )
            plan = self.planner.plan(
                actor,
                query,
                purpose=purpose,
                available_signals=available_signals,
                seed_memory_ids=seed_memory_ids,
            )
            if token_budget is not None:
                plan = plan.model_copy(update={"token_budget": token_budget})
            if self.packing_policy is not None:
                plan = plan.model_copy(update={"packing_policy": self.packing_policy})
            selected_primary = tuple(
                cast(RetrievalGateway, gateway)
                for gateway in self.gateways
                if gateway.signal in plan.signal_lanes
                and gateway.signal is not RetrievalSignal.GRAPH
            )
            raw_batches = await asyncio.gather(
                *(
                    gateway.retrieve(actor, plan, query, dense_query)
                    if gateway.signal is RetrievalSignal.DENSE
                    else gateway.retrieve(actor, plan, query)
                    for gateway in selected_primary
                ),
                return_exceptions=True,
            )
            batches: list[CandidateBatch] = []
            for gateway, result in zip(selected_primary, raw_batches, strict=True):
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

            direct_fused = weighted_rrf(tuple(batches), plan)
            graph_gateway = next(
                (
                    cast(GraphExpansionGateway, gateway)
                    for gateway in self.gateways
                    if gateway.signal is RetrievalSignal.GRAPH
                    and gateway.signal in plan.signal_lanes
                ),
                None,
            )
            if graph_gateway is not None:
                try:
                    graph_batch = await graph_gateway.expand(
                        actor,
                        plan,
                        query,
                        direct_fused,
                        dense_query,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    graph_batch = CandidateBatch(
                        lane=RetrievalSignal.GRAPH,
                        examined_count=0,
                        latency_ms=0.0,
                        degraded=True,
                        degradation_reason=exc.__class__.__name__,
                    )
                batches.append(graph_batch)

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

        # ``min_score`` is a floor on calibrated relevance, not on the public
        # score.  The public score is a ranking statement — weighted RRF, then
        # the relevance blend applied below — so thresholding it cannot express
        # "nothing here is actually about the query": it says where a candidate
        # placed, not whether the field was any good.  Relevance is the
        # rank-independent per-hit quantity defined in
        # ``swarmbrain.retrieval.relevance``, and it is what the floor reads.
        #
        # ``min_score = 0.0`` (the default) admits every relevance, so the floor
        # itself never changes a ranking.
        #
        # Cost: with reranking disabled, candidates are evaluated lazily in
        # fused order and the walk stops one past the public limit — at most
        # ``limit + 1`` relevance computations.  With reranking enabled the
        # window is evaluated up front instead, so the bound becomes
        # ``plan.rerank_window`` (four times the limit, floored at 32).  Both
        # paths reuse one cache and neither issues an extra query, because the
        # whole fused list was hydrated above.
        terms = relevance_query(query.text)
        scored: dict[str, CandidateRelevance] = {}

        def _relevance(candidate: FusedCandidate, memory: Memory) -> CandidateRelevance:
            cached = scored.get(candidate.canonical_id)
            if cached is None:
                cached = candidate_relevance(terms, memory, candidate.contributions)
                scored[candidate.canonical_id] = cached
            return cached

        # Reranking needs relevance for the whole window up front rather than
        # lazily, because the point is to promote candidates the fused order
        # would never have walked to.  Every candidate in the window was
        # already hydrated above, so this costs relevance arithmetic and no
        # extra queries; the plan bounds the window and the walk below still
        # stops one past the public limit.
        ranked = fused
        if plan.rerank:
            window: dict[str, float] = {}
            for candidate in fused[: plan.rerank_window]:
                memory = hydrated_by_id.get(candidate.canonical_id)
                if memory is None:
                    continue
                window[candidate.canonical_id] = _relevance(candidate, memory).relevance
            ranked = relevance_reranked(
                fused,
                window,
                alpha=plan.rerank_alpha,
                window=plan.rerank_window,
            )

        learned_rerank: LearnedRerankTrace | None = None
        if self.learned_reranker is not None and self.learned_rerank_policy is not None:
            ranked, learned_rerank = await self._learned_rerank(
                ranked,
                hydrated_by_id,
                query,
            )

        relevance_scores: list[CandidateRelevance] = []
        eligible: list[RecallHit] = []
        overflowed = False
        candidate_limit = query.limit
        if plan.token_budget is not None:
            candidate_limit = min(512, query.limit * 4)
            if plan.packing_policy is PackingPolicy.FACILITY_LOCATION:
                # Facility similarity is pairwise.  The selector owns the same
                # hard cap, and bounding hydration here avoids feature work on
                # candidates the selector is forbidden to examine.
                candidate_limit = min(MAX_FACILITY_CANDIDATES, candidate_limit)
        for candidate in ranked:
            memory = hydrated_by_id.get(candidate.canonical_id)
            if memory is None:
                continue
            relevance = _relevance(candidate, memory)
            relevance_scores.append(relevance)
            if relevance.relevance < query.min_score:
                continue
            if len(eligible) >= candidate_limit:
                overflowed = True
                break
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

        candidate_hits = tuple(eligible)
        rendered_context = ""
        packing: PackingTrace | None = None
        if plan.token_budget is None:
            hits = candidate_hits[: query.limit]
            if len(candidate_hits) > query.limit:
                overflowed = True
        else:
            rendered = tuple(
                ("" if index == 0 else "\n\n") + render_recall_hit(hit)
                for index, hit in enumerate(candidate_hits)
            )
            sizes = tuple(estimate_tokens(value) for value in rendered)
            packing_features = None
            if plan.packing_policy is PackingPolicy.FACILITY_LOCATION:
                packing_features = tuple(
                    build_packing_features(
                        search_text(
                            title=hit.memory.title,
                            content=hit.memory.content,
                            tags=hit.memory.tags,
                            metadata=hit.memory.metadata,
                        ),
                        query_terms=terms.tokens,
                        relevance=hit.score,
                        diversity_labels=(
                            f"kind:{hit.memory.kind}",
                            f"visibility:{hit.memory.visibility.value}",
                            f"author:{hit.memory.author_agent_id}",
                            *(f"tag:{tag}" for tag in hit.memory.tags),
                        ),
                    )
                    for hit in candidate_hits
                )
            packed = pack_to_budget(
                sizes,
                plan.token_budget,
                policy=plan.packing_policy,
                features=packing_features,
                max_items=(
                    query.limit if plan.packing_policy is PackingPolicy.FACILITY_LOCATION else None
                ),
            )
            kept_indices = packed.kept_indices[: query.limit]
            limit_dropped = packed.kept_indices[query.limit :]
            dropped_indices = tuple(dict.fromkeys((*packed.dropped_indices, *limit_dropped)))
            hits = tuple(candidate_hits[index] for index in kept_indices)
            rendered_context = "".join(rendered[index] for index in kept_indices).lstrip()
            used_tokens = estimate_tokens(rendered_context)
            packing = PackingTrace(
                policy=plan.packing_policy,
                token_budget=plan.token_budget,
                used_tokens=used_tokens,
                candidate_token_counts={
                    hit.memory.memory_id: size
                    for hit, size in zip(candidate_hits, sizes, strict=True)
                },
                kept_ids=tuple(hit.memory.memory_id for hit in hits),
                dropped_ids=tuple(
                    candidate_hits[index].memory.memory_id for index in dropped_indices
                ),
            )
            if dropped_indices or len(packed.kept_indices) > query.limit:
                overflowed = True
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
            relevance_version=RELEVANCE_VERSION,
            candidate_relevance=tuple(relevance_scores),
            hydrated_ids=hydrated_ids,
            hydration_rejections=tuple(
                HydrationRejection(canonical_id=candidate_id)
                for candidate_id in candidate_ids
                if candidate_id not in hydrated_set
            ),
            packing=packing,
            learned_rerank=learned_rerank,
            final_ids=tuple(hit.memory.memory_id for hit in hits),
            degraded_lanes=frozenset(batch.lane for batch in batches if batch.degraded),
            abstained=not hits,
            abstention_reason=self._abstention_reason(
                query,
                hits,
                relevance_scores,
                token_budget_exhausted=(bool(candidate_hits) and not hits and packing is not None),
            ),
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
                truncated=(overflowed or any(batch.truncated for batch in batches)),
            ),
            trace=trace,
            rendered_context=rendered_context,
        )

    async def _learned_rerank(
        self,
        baseline: tuple[FusedCandidate, ...],
        hydrated_by_id: dict[str, Memory],
        query: RecallQuery,
    ) -> tuple[tuple[FusedCandidate, ...], LearnedRerankTrace]:
        """Apply one optional score-only stage or preserve ``baseline`` exactly."""

        provider = self.learned_reranker
        policy = self.learned_rerank_policy
        assert provider is not None and policy is not None
        selected = tuple(
            (candidate, memory)
            for candidate in baseline[: policy.window]
            if (memory := hydrated_by_id.get(candidate.canonical_id)) is not None
        )
        if not selected:
            return baseline, LearnedRerankTrace(
                policy=policy,
                identity=policy.identity,
                attempted=False,
                applied=False,
                degraded=False,
                latency_ms=0.0,
            )

        request: LearnedRerankRequest | None = None
        started_ns = perf_counter_ns()
        try:
            # Re-read the property immediately before invocation.  A mutable
            # adapter cannot pass constructor validation and then silently
            # serve a different deployment identity.
            if provider.identity != policy.identity:
                raise LearnedRerankValidationError("provider identity drifted")
            request = build_swarm_memory_rerank_request(
                policy,
                query=query.text,
                candidates=selected,
            )
            result = await asyncio.wait_for(
                provider.rerank(request),
                timeout=float(policy.timeout_seconds),
            )
            validate_learned_rerank_result(
                request,
                result,
                expected_identity=policy.identity,
            )
            provider_request_id = result.receipt.provider_request_id
            if provider_request_id == request.request_id:
                raise LearnedRerankValidationError(
                    "provider request ID must differ from client request ID"
                )
            if provider_request_id in self._provider_request_ids:
                raise LearnedRerankValidationError("provider request ID was reused")
            score_by_id = {item.candidate_id: float(item.score) for item in result.scores}
            reranked = learned_score_reranked(
                baseline,
                score_by_id,
                alpha=float(policy.alpha),
                window=policy.window,
            )
            input_ids = tuple(candidate.candidate_id for candidate in request.candidates)
            input_set = frozenset(input_ids)
            output_ids = tuple(
                candidate.canonical_id
                for candidate in reranked[: policy.window]
                if candidate.canonical_id in input_set
            )
            self._remember_provider_request_id(provider_request_id)
            latency_ms = (perf_counter_ns() - started_ns) / 1_000_000
            return reranked, LearnedRerankTrace(
                policy=policy,
                identity=policy.identity,
                attempted=True,
                applied=True,
                degraded=False,
                serializer_revision=request.serializer_revision,
                request_id=request.request_id,
                provider_request_id=provider_request_id,
                request_sha256=request.request_sha256,
                query_sha256=request.query_sha256,
                candidate_pool_sha256=request.candidate_pool_sha256,
                candidate_document_sha256={
                    candidate.candidate_id: candidate.document_sha256
                    for candidate in request.candidates
                },
                candidate_temporal_sha256={
                    candidate.candidate_id: candidate.temporal_sha256
                    for candidate in request.candidates
                },
                input_ids=input_ids,
                output_ids=output_ids,
                scores=result.scores,
                usage=result.receipt.usage,
                response_sha256=result.receipt.response_sha256,
                latency_ms=latency_ms,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            latency_ms = (perf_counter_ns() - started_ns) / 1_000_000
            return baseline, self._degraded_learned_rerank_trace(
                policy,
                request,
                latency_ms=latency_ms,
                reason=self._learned_rerank_failure_reason(exc),
            )

    @staticmethod
    def _degraded_learned_rerank_trace(
        policy: LearnedRerankPolicy,
        request: LearnedRerankRequest | None,
        *,
        latency_ms: float,
        reason: str,
    ) -> LearnedRerankTrace:
        candidates = () if request is None else request.candidates
        return LearnedRerankTrace(
            policy=policy,
            identity=policy.identity,
            attempted=True,
            applied=False,
            degraded=True,
            serializer_revision=None if request is None else request.serializer_revision,
            request_id=None if request is None else request.request_id,
            request_sha256=None if request is None else request.request_sha256,
            query_sha256=None if request is None else request.query_sha256,
            candidate_pool_sha256=None if request is None else request.candidate_pool_sha256,
            candidate_document_sha256={
                candidate.candidate_id: candidate.document_sha256 for candidate in candidates
            },
            candidate_temporal_sha256={
                candidate.candidate_id: candidate.temporal_sha256 for candidate in candidates
            },
            input_ids=tuple(candidate.candidate_id for candidate in candidates),
            latency_ms=latency_ms,
            degradation_reason=reason,
        )

    @staticmethod
    def _learned_rerank_failure_reason(exc: Exception) -> str:
        if isinstance(exc, TimeoutError):
            return "provider_timeout"
        if isinstance(exc, LearnedRerankValidationError):
            return "provider_contract_violation"
        return f"provider_{type(exc).__name__}"[:255]

    def _remember_provider_request_id(self, value: str) -> None:
        self._provider_request_ids.add(value)
        self._provider_request_order.append(value)
        if len(self._provider_request_order) <= _PROVIDER_REQUEST_ID_WINDOW:
            return
        expired = self._provider_request_order.popleft()
        self._provider_request_ids.discard(expired)

    @staticmethod
    def _abstention_reason(
        query: RecallQuery,
        hits: tuple[RecallHit, ...],
        relevance_scores: Sequence[CandidateRelevance],
        *,
        token_budget_exhausted: bool = False,
    ) -> str | None:
        """Name why an empty bundle is empty, separating the two causes.

        ``below_relevance_floor`` means recallable candidates existed and every
        one of them failed the caller's calibrated floor — the abstention this
        feature adds.  ``no_relevant_recallable_candidates`` keeps its previous
        meaning: nothing survived scope, fusion, or hydration at all.
        """

        if hits:
            return None
        if token_budget_exhausted:
            return "token_budget_exhausted"
        if query.min_score > 0.0 and relevance_scores:
            return "below_relevance_floor"
        return "no_relevant_recallable_candidates"

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
