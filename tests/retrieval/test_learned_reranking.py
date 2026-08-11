from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from conftest import make_actor, new_id
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.domain.memory import Memory, MemoryState, RecallQuery, Visibility
from swarmbrain.domain.reranking import (
    LearnedRerankCandidate,
    LearnedRerankerComponent,
    LearnedRerankerIdentity,
    LearnedRerankPolicy,
    LearnedRerankScore,
    LearnedRerankTrace,
    LearnedRerankUsage,
    learned_reranker_model_bundle_payload,
    learned_reranker_tokenizer_bundle_payload,
    rerank_sha256_json,
)
from swarmbrain.domain.retrieval import Candidate, CandidateBatch, RetrievalSignal
from swarmbrain.retrieval.learned_reranking import (
    LearnedRerankValidationError,
    build_learned_rerank_request,
    build_learned_rerank_result,
    request_usage_dimensions,
)


def _identity(*, suffix: str = "a") -> LearnedRerankerIdentity:
    component = LearnedRerankerComponent(
        role="cross_encoder",
        model="Qwen/Qwen3-Reranker-8B",
        revision=suffix * 40,
        model_artifact_sha256="b" * 64,
        tokenizer_revision=suffix * 40,
        tokenizer_artifact_sha256="c" * 64,
        weight=1.0,
    )
    components = (component,)
    return LearnedRerankerIdentity(
        provider="fixture",
        model="Qwen/Qwen3-Reranker-8B",
        revision=suffix * 40,
        components=components,
        model_artifact_sha256=rerank_sha256_json(learned_reranker_model_bundle_payload(components)),
        tokenizer_artifact_sha256=rerank_sha256_json(
            learned_reranker_tokenizer_bundle_payload(components)
        ),
        deployment_manifest_sha256="d" * 64,
        adapter_artifact_sha256="e" * 64,
        runtime_environment_sha256="f" * 64,
        protocol_revision="fixture-jsonl-v1",
    )


def _policy(
    identity: LearnedRerankerIdentity | None = None,
    **updates: object,
) -> LearnedRerankPolicy:
    return LearnedRerankPolicy(identity=identity or _identity(), **updates)


def _usage(request: object, *, input_tokens: int = 17) -> LearnedRerankUsage:
    dimensions = request_usage_dimensions(request)  # type: ignore[arg-type]
    return LearnedRerankUsage(
        **dimensions,
        input_tokens=input_tokens,
        output_tokens=0,
        total_tokens=input_tokens,
        tokenized_input_sha256=rerank_sha256_json(
            {"request_sha256": request.request_sha256, "token_count": input_tokens}  # type: ignore[attr-defined]
        ),
    )


class _Provider:
    def __init__(
        self,
        identity: LearnedRerankerIdentity,
        scores: dict[str, float],
        *,
        response_identity: LearnedRerankerIdentity | None = None,
        provider_request_id: str | None = None,
    ) -> None:
        self.identity = identity
        self.scores = scores
        self.response_identity = response_identity
        self.provider_request_id = provider_request_id
        self.requests = []

    async def rerank(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return build_learned_rerank_result(
            request,
            scores=tuple(self.scores[item.candidate_id] for item in request.candidates),
            usage=_usage(request),
            provider_request_id=self.provider_request_id or str(uuid4()),
            identity=self.response_identity,
        )


class _FailingProvider:
    def __init__(self, identity: LearnedRerankerIdentity) -> None:
        self.identity = identity
        self.requests = []

    async def rerank(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        raise RuntimeError("secret provider detail")


class _SleepingProvider:
    def __init__(self, identity: LearnedRerankerIdentity) -> None:
        self.identity = identity
        self.started = asyncio.Event()

    async def rerank(self, _request):  # type: ignore[no-untyped-def]
        self.started.set()
        await asyncio.sleep(60)
        raise AssertionError("unreachable")


class _DriftingProvider(_Provider):
    def __init__(self, identity: LearnedRerankerIdentity, scores: dict[str, float]) -> None:
        self._identity = identity
        super().__init__(identity, scores)

    @property
    def identity(self) -> LearnedRerankerIdentity:
        return self._identity

    @identity.setter
    def identity(self, value: LearnedRerankerIdentity) -> None:
        self._identity = value


class _Gateway:
    signal = RetrievalSignal.LEXICAL

    def __init__(self, memory_ids: Sequence[str]) -> None:
        self.memory_ids = tuple(memory_ids)

    async def retrieve(self, *_args: object) -> CandidateBatch:
        return CandidateBatch(
            lane=self.signal,
            candidates=tuple(
                Candidate(
                    resource_type="memory",
                    resource_id=memory_id,
                    resource_version=1,
                    canonical_id=memory_id,
                    domain_lane="knowledge",
                    signal=self.signal,
                    rank=rank,
                    raw_score=1.0 / rank,
                )
                for rank, memory_id in enumerate(self.memory_ids, start=1)
            ),
            examined_count=len(self.memory_ids),
            latency_ms=0.1,
        )


class _Reader:
    def __init__(self, memories: Sequence[Memory]) -> None:
        self.memories = {memory.memory_id: memory for memory in memories}

    async def hydrate_recallable(
        self,
        _actor: object,
        _query: object,
        candidate_ids: Sequence[str],
    ) -> tuple[Memory, ...]:
        return tuple(self.memories[value] for value in candidate_ids if value in self.memories)


def _memory(actor: object, memory_id: str, rank: int) -> Memory:
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)
    return Memory(
        memory_id=memory_id,
        tenant_id=actor.tenant_id,  # type: ignore[attr-defined]
        project_id=actor.project_id,  # type: ignore[attr-defined]
        repository_id=actor.repository_id,  # type: ignore[attr-defined]
        swarm_id=actor.swarm_id,  # type: ignore[attr-defined]
        run_id=actor.run_id,  # type: ignore[attr-defined]
        author_agent_id=actor.agent_id,  # type: ignore[attr-defined]
        kind="observation",
        state=MemoryState.CONFIRMED,
        visibility=Visibility.REPOSITORY,
        title=f"candidate {rank}",
        content=f"target memory payload {rank}",
        valid_from=now - timedelta(days=rank),
        recorded_from=now - timedelta(hours=rank),
    )


def _service_inputs(
    scope_ids: dict[str, str],
    *,
    count: int = 3,
    missing_positions: frozenset[int] = frozenset(),
) -> tuple[object, tuple[str, ...], tuple[Memory, ...], _Gateway, _Reader]:
    actor = make_actor(scope_ids)
    memory_ids = tuple(new_id() for _ in range(count))
    memories = tuple(
        _memory(actor, memory_id, position)
        for position, memory_id in enumerate(memory_ids, start=1)
        if position not in missing_positions
    )
    return actor, memory_ids, memories, _Gateway(memory_ids), _Reader(memories)


def test_composite_identity_binds_order_artifacts_tokenizers_and_weights() -> None:
    first = LearnedRerankerComponent(
        role="cross_encoder",
        model="Qwen/Qwen3-Reranker-8B",
        revision="a" * 40,
        model_artifact_sha256="b" * 64,
        tokenizer_revision="a" * 40,
        tokenizer_artifact_sha256="c" * 64,
        weight=0.7,
    )
    second = LearnedRerankerComponent(
        role="colbert",
        model="answerdotai/answerai-colbert-small-v1",
        revision="d" * 40,
        model_artifact_sha256="e" * 64,
        tokenizer_revision="d" * 40,
        tokenizer_artifact_sha256="f" * 64,
        weight=0.3,
    )
    components = (first, second)
    identity = LearnedRerankerIdentity(
        provider="local-jsonl",
        model="smartsearch-ce-colbert-0.7-0.3",
        revision="1" * 64,
        components=components,
        model_artifact_sha256=rerank_sha256_json(learned_reranker_model_bundle_payload(components)),
        tokenizer_artifact_sha256=rerank_sha256_json(
            learned_reranker_tokenizer_bundle_payload(components)
        ),
        deployment_manifest_sha256="2" * 64,
        adapter_artifact_sha256="3" * 64,
        runtime_environment_sha256="4" * 64,
        protocol_revision="fixture-v1",
    )

    assert tuple(component.weight for component in identity.components) == (0.7, 0.3)
    with pytest.raises(ValidationError, match="weights must sum"):
        LearnedRerankerIdentity(
            **identity.model_dump(exclude={"components", "model_artifact_sha256"}),
            components=(first, second.model_copy(update={"weight": 0.2})),
            model_artifact_sha256="4" * 64,
        )


def test_request_contract_rejects_content_digest_drift_and_total_byte_overflow() -> None:
    policy = _policy(max_request_bytes=1_024)
    with pytest.raises(ValidationError, match="document_sha256"):
        LearnedRerankCandidate(
            candidate_id="candidate",
            document="canonical",
            document_sha256="0" * 64,
            temporal_context="{}",
            temporal_sha256=rerank_sha256_json({}),
        )
    with pytest.raises(LearnedRerankValidationError, match="request exceeds byte bound"):
        build_learned_rerank_request(
            policy,
            serializer_revision="fixture-v1",
            query="question",
            candidates=(("candidate", "x" * 500, "{}"),),
        )
    with pytest.raises(ValidationError, match="must be numeric"):
        LearnedRerankScore(candidate_id="candidate", score="0.5")  # type: ignore[arg-type]


def test_service_requires_an_identity_bound_provider_policy_pair(
    scope_ids: dict[str, str],
) -> None:
    _actor, _ids, _memories, gateway, reader = _service_inputs(scope_ids)
    identity = _identity()
    provider = _Provider(identity, {})

    with pytest.raises(ValueError, match="configured together"):
        RetrievalService((gateway,), reader, learned_reranker=provider)
    with pytest.raises(ValueError, match="identity"):
        RetrievalService(
            (gateway,),
            reader,
            learned_reranker=provider,
            learned_rerank_policy=_policy(_identity(suffix="f")),
        )


@pytest.mark.asyncio
async def test_default_service_never_invokes_or_traces_a_learned_stage(
    scope_ids: dict[str, str],
) -> None:
    actor, memory_ids, _memories, gateway, reader = _service_inputs(scope_ids)
    execution = await RetrievalService((gateway,), reader).execute(
        actor,  # type: ignore[arg-type]
        RecallQuery(text="target", limit=3),
    )

    assert execution.trace.learned_rerank is None
    assert execution.trace.final_ids == memory_ids


@pytest.mark.asyncio
async def test_successful_learned_stage_is_score_only_receipted_and_content_free(
    scope_ids: dict[str, str],
) -> None:
    actor, memory_ids, memories, gateway, reader = _service_inputs(scope_ids)
    identity = _identity()
    provider = _Provider(
        identity,
        {memory_ids[0]: 0.1, memory_ids[1]: 0.9, memory_ids[2]: 0.2},
    )
    service = RetrievalService(
        (gateway,),
        reader,
        learned_reranker=provider,
        learned_rerank_policy=_policy(identity, window=3),
    )

    execution = await service.execute(
        actor,  # type: ignore[arg-type]
        RecallQuery(text="target", limit=3),
    )

    assert execution.trace.final_ids == (memory_ids[1], memory_ids[2], memory_ids[0])
    trace = execution.trace.learned_rerank
    assert trace is not None
    assert (trace.attempted, trace.applied, trace.degraded) == (True, True, False)
    assert trace.input_ids == memory_ids
    assert trace.output_ids == (memory_ids[1], memory_ids[2], memory_ids[0])
    assert trace.usage is not None and trace.usage.candidate_count == 3
    assert trace.identity.components[0].model == "Qwen/Qwen3-Reranker-8B"
    assert trace.request_id != trace.provider_request_id
    assert trace.query_sha256 == provider.requests[0].query_sha256
    assert set(trace.candidate_document_sha256) == set(memory_ids)
    assert set(trace.candidate_temporal_sha256) == set(memory_ids)
    serialized_trace = trace.model_dump_json()
    assert "target memory payload" not in serialized_trace
    assert all(memory.content not in serialized_trace for memory in memories)  # type: ignore[operator]


@pytest.mark.asyncio
async def test_hydration_gap_and_tail_keep_monotone_public_scores(
    scope_ids: dict[str, str],
) -> None:
    actor, memory_ids, _memories, gateway, reader = _service_inputs(
        scope_ids,
        count=5,
        missing_positions=frozenset({2}),
    )
    identity = _identity()
    provider = _Provider(
        identity,
        {
            memory_ids[0]: 0.1,
            memory_ids[2]: 0.9,
        },
    )
    execution = await RetrievalService(
        (gateway,),
        reader,
        learned_reranker=provider,
        learned_rerank_policy=_policy(identity, window=3),
    ).execute(
        actor,  # type: ignore[arg-type]
        RecallQuery(text="target", limit=5),
    )

    assert execution.trace.final_ids == (
        memory_ids[2],
        memory_ids[0],
        memory_ids[3],
        memory_ids[4],
    )
    trace = execution.trace.learned_rerank
    assert trace is not None
    assert trace.input_ids == (memory_ids[0], memory_ids[2])
    assert trace.output_ids == (memory_ids[2], memory_ids[0])
    public_scores = tuple(hit.score for hit in execution.bundle.hits)
    assert public_scores == tuple(sorted(public_scores, reverse=True))


@pytest.mark.asyncio
async def test_provider_failure_preserves_exact_baseline_and_only_traces_request_evidence(
    scope_ids: dict[str, str],
) -> None:
    actor, _memory_ids, _memories, gateway, reader = _service_inputs(scope_ids)
    identity = _identity()
    query = RecallQuery(text="target", limit=3)
    baseline = await RetrievalService((gateway,), reader).execute(actor, query)  # type: ignore[arg-type]
    provider = _FailingProvider(identity)

    degraded = await RetrievalService(
        (gateway,),
        reader,
        learned_reranker=provider,
        learned_rerank_policy=_policy(identity, window=3),
    ).execute(actor, query)  # type: ignore[arg-type]

    assert degraded.trace.final_ids == baseline.trace.final_ids
    assert tuple(hit.score for hit in degraded.bundle.hits) == tuple(
        hit.score for hit in baseline.bundle.hits
    )
    trace = degraded.trace.learned_rerank
    assert trace is not None
    assert (trace.attempted, trace.applied, trace.degraded) == (True, False, True)
    assert trace.degradation_reason == "provider_RuntimeError"
    assert trace.request_sha256 is not None
    assert trace.input_ids
    assert trace.output_ids == ()
    assert trace.scores == ()
    assert trace.usage is None
    assert trace.provider_request_id is None
    assert "secret provider detail" not in trace.model_dump_json()


@pytest.mark.asyncio
async def test_identity_drift_fails_before_provider_invocation_and_timeout_is_explicit(
    scope_ids: dict[str, str],
) -> None:
    actor, memory_ids, _memories, gateway, reader = _service_inputs(scope_ids)
    identity = _identity()
    drifting = _DriftingProvider(identity, {value: 1.0 for value in memory_ids})
    service = RetrievalService(
        (gateway,),
        reader,
        learned_reranker=drifting,
        learned_rerank_policy=_policy(identity),
    )
    drifting.identity = _identity(suffix="f")

    drifted = await service.execute(actor, RecallQuery(text="target", limit=3))  # type: ignore[arg-type]

    assert drifting.requests == []
    assert drifted.trace.learned_rerank is not None
    assert drifted.trace.learned_rerank.degradation_reason == "provider_contract_violation"

    sleeping = _SleepingProvider(identity)
    timed = await RetrievalService(
        (gateway,),
        reader,
        learned_reranker=sleeping,
        learned_rerank_policy=_policy(identity, timeout_seconds=0.05),
    ).execute(actor, RecallQuery(text="target", limit=3))  # type: ignore[arg-type]
    assert timed.trace.learned_rerank is not None
    assert timed.trace.learned_rerank.degradation_reason == "provider_timeout"


@pytest.mark.asyncio
async def test_caller_cancellation_propagates_through_learned_provider(
    scope_ids: dict[str, str],
) -> None:
    actor, _memory_ids, _memories, gateway, reader = _service_inputs(scope_ids)
    identity = _identity()
    provider = _SleepingProvider(identity)
    service = RetrievalService(
        (gateway,),
        reader,
        learned_reranker=provider,
        learned_rerank_policy=_policy(identity),
    )
    task = asyncio.create_task(
        service.execute(actor, RecallQuery(text="target", limit=3))  # type: ignore[arg-type]
    )
    await asyncio.wait_for(provider.started.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


def test_trace_contract_rejects_policy_identity_drift() -> None:
    with pytest.raises(ValidationError, match="policy identity"):
        LearnedRerankTrace(
            policy=_policy(_identity()),
            identity=_identity(suffix="f"),
            attempted=False,
            applied=False,
            degraded=False,
            latency_ms=0.0,
        )
