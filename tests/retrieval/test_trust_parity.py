from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from conftest import make_actor, new_id
from swarmbrain.adapters.memory import InMemoryKernel, in_memory_retrieval_gateways
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.application.memory_service import MemoryService
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    EvidenceKind,
    RegisterEvidenceSourceCommand,
    SourceTrust,
)
from swarmbrain.domain.memory import RecallQuery, RememberCommand, Visibility


@pytest.mark.asyncio
async def test_mixed_trust_memory_survives_with_only_accepted_citation(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    kernel = InMemoryKernel(clock=lambda: now)
    service = MemoryService(
        kernel,
        ConservativeMemoryPolicy(),
        retrieval=RetrievalService(in_memory_retrieval_gateways(kernel), kernel),
        canonical_reader=kernel,
    )

    async def evidence(trust: SourceTrust, suffix: str) -> object:
        payload = f"source-{suffix}".encode()
        source = await kernel.register_source(
            actor,
            RegisterEvidenceSourceCommand(
                idempotency_key=f"source-{suffix}",
                occurrence_key=f"source-{suffix}",
                kind=EvidenceKind.DOCUMENT,
                content_sha256=hashlib.sha256(payload).hexdigest(),
                observed_at=now,
                trust=trust,
            ),
        )
        item = await kernel.add_evidence(
            actor,
            AddEvidenceCommand(
                idempotency_key=f"evidence-{suffix}",
                source_id=source.source_id,
                kind=EvidenceKind.DOCUMENT,
                excerpt=f"citation {suffix}",
            ),
        )
        return item.as_ref()

    trusted = await evidence(SourceTrust.TRUSTED, "trusted")
    poisoned = await evidence(SourceTrust.UNTRUSTED, "poisoned")
    published = await service.publish(
        actor,
        RememberCommand(
            idempotency_key="mixed-trust-memory",
            kind="invariant",
            visibility=Visibility.REPOSITORY,
            content="mixed trust serialization sentinel",
            evidence=(trusted, poisoned),  # type: ignore[arg-type]
        ),
    )
    assert published.memory is not None

    recalled = await service.recall(actor, RecallQuery(text="serialization sentinel"))

    assert [hit.memory.memory_id for hit in recalled.hits] == [published.memory.memory_id]
    assert recalled.hits[0].memory.evidence == (trusted,)
    assert recalled.hits[0].evidence == (trusted,)

    next_run = actor.model_copy(
        update={"swarm_id": new_id(), "run_id": new_id(), "agent_id": new_id()}
    )
    cross_run = await service.recall(
        next_run,
        RecallQuery(text="mixed trust serialization sentinel"),
    )
    assert [hit.memory.memory_id for hit in cross_run.hits] == [published.memory.memory_id]
    assert cross_run.hits[0].memory.evidence == (trusted,)

    poisoned_only = await service.publish(
        actor,
        RememberCommand(
            idempotency_key="poisoned-only-memory",
            kind="observation",
            visibility=Visibility.REPOSITORY,
            content="audit-only poisoned citation marker",
            evidence=(poisoned,),  # type: ignore[arg-type]
        ),
    )
    assert poisoned_only.memory is not None

    audited = await service.recall(
        actor,
        RecallQuery(
            text="audit-only poisoned citation marker",
            include_refuted=True,
            include_evidence=True,
        ),
    )

    target = next(
        hit for hit in audited.hits if hit.memory.memory_id == poisoned_only.memory.memory_id
    )
    assert target.memory.evidence == ()
    assert target.evidence == ()
    metrics = await kernel.get_run_metrics(actor, actor.run_id)
    assert metrics.memories_reused == len(recalled.hits) + len(audited.hits)
