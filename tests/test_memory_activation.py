from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest

from conftest import make_actor, new_id
from swarmbrain.application.activation import MemoryActivationService
from swarmbrain.application.coordination import CoordinationService
from swarmbrain.application.errors import Forbidden
from swarmbrain.domain.activation import (
    ActivationDecision,
    ActivationReason,
    ActivationTrigger,
    MemoryActivationRequest,
    MemoryActivationTelemetry,
    memory_activation_id,
)
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.memory import (
    Memory,
    MemoryKind,
    MemoryState,
    RecallBundle,
    RecallHit,
    RecallQuery,
)
from swarmbrain.domain.retrieval import PackingTrace, RetrievalPurpose, RetrievalTrace
from swarmbrain.retrieval import estimate_tokens, render_recall_hit


def _memory(actor: ActorContext, memory_id: str, content: str) -> Memory:
    return Memory(
        memory_id=memory_id,
        tenant_id=actor.tenant_id,
        project_id=actor.project_id,
        repository_id=actor.repository_id,
        swarm_id=actor.swarm_id,
        run_id=actor.run_id,
        author_agent_id=new_id(),
        kind=MemoryKind.PROCEDURE,
        content=content,
        valid_from=datetime(2026, 8, 9, tzinfo=UTC),
    )


class _RecallStub:
    def __init__(self, response: object = None, failure: Exception | None = None) -> None:
        self.response = response
        self.failure = failure
        self.calls: list[tuple[object, RecallQuery, dict[str, object]]] = []

    async def recall(self, actor: object, query: RecallQuery, **kwargs: object) -> object:
        self.calls.append((actor, query, kwargs))
        if self.failure is not None:
            raise self.failure
        if self.response is None:
            return RecallBundle(query=query)
        return self.response


def _request(**updates: object) -> MemoryActivationRequest:
    values: dict[str, object] = {
        "task_id": new_id(),
        "lease_id": new_id(),
        "trigger": ActivationTrigger.TASK_CLAIM,
    }
    values.update(updates)
    return MemoryActivationRequest(**values)


def test_activation_identity_is_stable_and_names_task_lease_and_trigger() -> None:
    task_id, lease_id = new_id(), new_id()
    first = memory_activation_id(task_id, lease_id, ActivationTrigger.TASK_CLAIM)

    assert UUID(first)
    assert first == memory_activation_id(task_id, lease_id, ActivationTrigger.TASK_CLAIM)
    assert first != memory_activation_id(task_id, lease_id, ActivationTrigger.TOOL_ERROR)
    assert first != memory_activation_id(task_id, new_id(), ActivationTrigger.TASK_CLAIM)
    assert first != memory_activation_id(new_id(), lease_id, ActivationTrigger.TASK_CLAIM)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"initial_memory_limit": 101}, "initial_memory_limit"),
        ({"initial_memory_token_budget": 0}, "initial_memory_token_budget"),
        ({"initial_memory_min_score": 1.1}, "initial_memory_min_score"),
    ),
)
def test_coordination_rejects_invalid_activation_policy_before_claim(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CoordinationService(SimpleNamespace(), **kwargs)  # type: ignore[arg-type]


def test_activation_telemetry_canonicalizes_ids_and_rejects_overlap() -> None:
    request = _request()
    selected, dropped = new_id(), new_id()
    telemetry = MemoryActivationTelemetry(
        activation_id=request.activation_id,
        run_id=new_id(),
        agent_id=new_id(),
        task_id=request.task_id,
        lease_id=request.lease_id,
        trigger=request.trigger,
        decision=ActivationDecision.RECALL,
        purpose=RetrievalPurpose.TASK_BOOTSTRAP,
        reason=ActivationReason.CONTEXT_ACTIVATED,
        memory_ids=(selected, selected),
        memory_versions={selected: 1},
        dropped_memory_ids=(dropped, dropped),
        token_budget=100,
        estimated_tokens=10,
        min_score=0.4,
    )

    assert telemetry.memory_ids == (selected,)
    assert telemetry.memory_versions == {selected: 1}
    assert telemetry.dropped_memory_ids == (dropped,)
    with pytest.raises(ValueError, match="must not overlap"):
        telemetry.model_copy(update={"dropped_memory_ids": (selected,)}).model_validate(
            telemetry.model_copy(update={"dropped_memory_ids": (selected,)}).model_dump()
        )
    with pytest.raises(ValueError, match="activation_id must match"):
        MemoryActivationTelemetry(**{**telemetry.model_dump(), "activation_id": new_id()})
    with pytest.raises(ValueError, match="memory_versions"):
        MemoryActivationTelemetry(**{**telemetry.model_dump(), "memory_versions": {}})

    too_many = tuple(new_id() for _ in range(101))
    with pytest.raises(ValueError, match="at most 100"):
        MemoryActivationTelemetry(
            **{
                **telemetry.model_dump(),
                "memory_ids": too_many,
                "memory_versions": {memory_id: 1 for memory_id in too_many},
            }
        )


@pytest.mark.asyncio
async def test_recall_activation_passes_server_policy_and_serializes_only_safe_telemetry(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    secret_query = "customer secret query that must not enter telemetry"
    secret_content = "customer secret memory content"
    memory = _memory(actor, new_id(), secret_content)
    request = _request(
        seed_memory_ids=(memory.memory_id, memory.memory_id),
        token_budget=777,
        min_score=0.55,
        limit=7,
    )
    bundle = RecallBundle(
        query=RecallQuery(text=secret_query, task_id=request.task_id),
        hits=(RecallHit(memory=memory, score=0.9),),
        total_candidates=4,
    )
    recaller = _RecallStub(bundle)

    result = await MemoryActivationService(recaller).activate(  # type: ignore[arg-type]
        actor,
        request,
        query_text=secret_query,
    )

    assert result.decision is ActivationDecision.RECALL
    assert result.bundle == bundle
    assert memory.memory_id in result.rendered_context
    assert secret_content in result.rendered_context
    assert request.seed_memory_ids == (memory.memory_id,)
    _, query, kwargs = recaller.calls[0]
    assert query.task_id == request.task_id
    assert query.text == secret_query
    assert query.states == frozenset({MemoryState.CONFIRMED})
    assert query.min_score == pytest.approx(0.55)
    assert query.limit == 7
    assert kwargs == {
        "purpose": RetrievalPurpose.TASK_BOOTSTRAP,
        "seed_memory_ids": (memory.memory_id,),
        "token_budget": 777,
    }

    persisted = result.model_dump_json()
    assert secret_query not in persisted
    assert secret_content not in persisted
    assert "rendered_context" not in persisted
    assert "bundle" not in persisted
    assert result.telemetry_payload() == result.telemetry.model_dump(mode="json")
    assert result.telemetry.memory_versions == {memory.memory_id: memory.version}
    assert "query" not in request.model_dump_json()


@pytest.mark.asyncio
async def test_deep_trigger_overfetches_but_keeps_the_requested_output_cap(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    request = _request(trigger=ActivationTrigger.CHECKPOINT_RESUME, limit=2)
    memories = tuple(
        _memory(actor, new_id(), f"resume with fenced checkpoint {index}") for index in range(4)
    )
    bundle = RecallBundle(
        query=RecallQuery(text="resume", task_id=request.task_id),
        hits=tuple(RecallHit(memory=memory, score=1.0) for memory in memories),
    )
    recaller = _RecallStub(bundle)

    result = await MemoryActivationService(recaller).activate(  # type: ignore[arg-type]
        actor,
        request,
        query_text="resume",
    )

    assert result.decision is ActivationDecision.DEEP_RECALL
    _, query, kwargs = recaller.calls[0]
    assert query.limit == 4
    assert kwargs["purpose"] is RetrievalPurpose.HANDOFF_RECOVERY
    assert result.bundle is not None
    assert len(result.bundle.hits) == request.limit
    assert result.telemetry.memory_ids == tuple(memory.memory_id for memory in memories[:2])


@pytest.mark.asyncio
async def test_bundle_fallback_still_greedily_enforces_the_activation_budget(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    oversized = _memory(actor, new_id(), "x" * 8_000)
    compact = _memory(actor, new_id(), "reuse the verified command")
    large_hit = RecallHit(memory=oversized, score=1.0)
    compact_hit = RecallHit(memory=compact, score=0.9)
    compact_cost = estimate_tokens("\n\n" + render_recall_hit(compact_hit))
    request = _request(token_budget=compact_cost)
    bundle = RecallBundle(
        query=RecallQuery(text="verified command", task_id=request.task_id),
        hits=(large_hit, compact_hit),
        total_candidates=2,
    )

    result = await MemoryActivationService(  # type: ignore[arg-type]
        _RecallStub(bundle)
    ).activate(actor, request, query_text="verified command")

    assert result.decision is ActivationDecision.RECALL
    assert result.bundle is not None
    assert [hit.memory.memory_id for hit in result.bundle.hits] == [compact.memory_id]
    assert result.telemetry.dropped_memory_ids == (oversized.memory_id,)
    assert result.telemetry.estimated_tokens <= compact_cost
    assert result.telemetry.truncated is True


@pytest.mark.asyncio
async def test_empty_or_irrelevant_context_skips_and_outage_defers(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    request = _request()
    empty = _RecallStub()

    blank = await MemoryActivationService(empty).activate(  # type: ignore[arg-type]
        actor, request, query_text="   "
    )
    skipped = await MemoryActivationService(empty).activate(  # type: ignore[arg-type]
        actor, request, query_text="nothing relevant"
    )
    deferred = await MemoryActivationService(
        _RecallStub(failure=RuntimeError("secret database diagnostic"))  # type: ignore[arg-type]
    ).activate(actor, request, query_text="private query")

    assert blank.decision is ActivationDecision.SKIP
    assert len(empty.calls) == 1
    assert skipped.decision is ActivationDecision.SKIP
    assert skipped.bundle is not None and skipped.bundle.hits == ()
    assert deferred.decision is ActivationDecision.DEFER
    assert deferred.bundle is None
    assert "secret database diagnostic" not in deferred.model_dump_json()
    assert "private query" not in deferred.model_dump_json()


@pytest.mark.asyncio
async def test_canonical_budgeted_context_is_authoritative_when_execution_is_available(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    request = _request(token_budget=100)
    kept, dropped = new_id(), new_id()
    memory = _memory(actor, kept, "the fallback renderer must not win")
    bundle = RecallBundle(
        query=RecallQuery(text="resume", task_id=request.task_id),
        hits=(RecallHit(memory=memory, score=1.0),),
        total_candidates=2,
        truncated=True,
    )
    trace = RetrievalTrace(
        trace_id=new_id(),
        plan={
            "purpose": "task_bootstrap",
            "domain_lanes": ["knowledge"],
            "signal_lanes": [],
            "hard_scope": {
                **scope_ids,
                "task_id": request.task_id,
                "visibilities": ["run"],
            },
            "lane_budgets": {},
            "lane_weights": {},
            "token_budget": 100,
        },
        packing=PackingTrace(
            token_budget=100,
            used_tokens=9,
            kept_ids=(kept,),
            dropped_ids=(dropped,),
        ),
        final_ids=(kept,),
        started_at=datetime(2026, 8, 9, tzinfo=UTC),
        completed_at=datetime(2026, 8, 9, tzinfo=UTC),
    )
    execution = SimpleNamespace(
        bundle=bundle,
        trace=trace,
        rendered_context="authoritative execution rendering",
    )

    result = await MemoryActivationService(  # type: ignore[arg-type]
        _RecallStub(execution)
    ).activate(actor, request, query_text="resume")

    expected_context = render_recall_hit(bundle.hits[0])
    assert result.rendered_context == expected_context
    assert result.telemetry.trace_id == trace.trace_id
    assert result.telemetry.estimated_tokens == estimate_tokens(expected_context)
    assert result.telemetry.dropped_memory_ids == (dropped,)


@pytest.mark.asyncio
async def test_activation_bounds_query_before_it_reaches_a_provider(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    recaller = _RecallStub()

    await MemoryActivationService(recaller).activate(  # type: ignore[arg-type]
        actor,
        _request(),
        query_text="q" * 10_000,
    )

    assert len(recaller.calls[0][1].text) == 8_192


@pytest.mark.asyncio
async def test_authorization_failures_are_not_downgraded_to_defer(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    service = MemoryActivationService(  # type: ignore[arg-type]
        _RecallStub(failure=Forbidden("memory:recall"))
    )

    with pytest.raises(Forbidden):
        await service.activate(actor, _request(), query_text="must remain fail closed")
