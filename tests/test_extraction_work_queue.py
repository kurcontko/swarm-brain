from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from conftest import make_actor
from swarmbrain.adapters.extraction import (
    CodingRuleExtractor,
    InMemoryWorkStore,
    LazyExtractionProvider,
)
from swarmbrain.application.errors import IdempotencyConflict
from swarmbrain.application.extraction import ExtractionService
from swarmbrain.domain.evidence import EvidenceKind, SourceTrust
from swarmbrain.domain.extraction import (
    ExtractionCandidate,
    ExtractionOutcome,
    ExtractionProvenance,
    ExtractionStage,
    IngestRawSourceCommand,
    ProviderDescriptor,
    SourceSpan,
)
from swarmbrain.domain.memory import MemoryKind
from swarmbrain.domain.work import (
    ApplyExtractionWorkCommand,
    ClaimWorkCommand,
    WorkEffect,
    WorkEffectConflict,
    WorkKind,
    WorkLeaseLost,
    WorkStatus,
)
from swarmbrain.workers import ExtractionWorker


class MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _command(
    *,
    content: str = "def load_config():\n    pass\n# TODO: validate the config\n",
    idempotency_key: str = "raw-source-1",
    use_provider: bool = False,
    trust: SourceTrust = SourceTrust.UNKNOWN,
) -> IngestRawSourceCommand:
    return IngestRawSourceCommand(
        idempotency_key=idempotency_key,
        kind=EvidenceKind.SOURCE_CODE,
        content=content,
        observed_at=datetime(2026, 8, 2, 11, 0, tzinfo=UTC),
        uri="src/config.py",
        occurrence_key="src/config.py@abc123",
        trust=trust,
        use_provider=use_provider,
    )


def _payload_hash(candidate: ExtractionCandidate) -> str:
    payload = json.dumps(
        candidate.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _provenance(attempt: int, input_sha256: str) -> ExtractionProvenance:
    timestamp = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)
    return ExtractionProvenance(
        stage=ExtractionStage.DETERMINISTIC,
        attempt=attempt,
        outcome=ExtractionOutcome.SUCCEEDED,
        input_sha256=input_sha256,
        output_sha256=hashlib.sha256(b"[]").hexdigest(),
        started_at=timestamp,
        finished_at=timestamp,
    )


@pytest.mark.asyncio
async def test_raw_ingest_preserves_exact_content_chunks_and_work_atomically(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    actor = make_actor(scope_ids)
    command = _command()

    first = await service.ingest(actor, command)
    replay = await service.ingest(actor, command)

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.work_id == first.work_id
    assert "".join(chunk.content for chunk in first.chunks) == command.content
    assert store.sources[first.source.source_id].raw_content == command.content
    assert store.items[first.work_id].subject_id == first.source.source_id

    changed = command.model_copy(update={"content": command.content + "# changed\n"})
    with pytest.raises(IdempotencyConflict):
        await service.ingest(actor, changed)


@pytest.mark.asyncio
async def test_deterministic_coding_route_produces_only_exact_source_spans(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    actor = make_actor(scope_ids)
    ingested = await service.ingest(actor, _command())
    lease = (await store.claim_work(ClaimWorkCommand(worker_id="worker-a", limit=1))).leases[0]
    request = await store.load_extraction_input(lease)

    result = await service.extract(request)

    assert result.route.value == "coding"
    assert {candidate.kind for candidate in result.candidates} == {
        MemoryKind.OBSERVATION,
        MemoryKind.WARNING,
    }
    assert ingested.source.source_id == request.source.source_id
    for candidate in result.candidates:
        for span in candidate.spans:
            assert request.raw_content[span.char_start : span.char_end] == span.excerpt


@pytest.mark.parametrize(
    "protected",
    [
        {"memory_id": "11111111-1111-1111-1111-111111111111"},
        {"tenant_id": "11111111-1111-1111-1111-111111111111"},
        {"state": "confirmed"},
        {"visibility": "repository"},
        {"evidence": []},
        {"supersedes_memory_id": "11111111-1111-1111-1111-111111111111"},
    ],
)
def test_candidate_contract_rejects_protected_storage_fields(
    protected: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "kind": "observation",
        "content": "validated fact",
        "spans": [
            {
                "chunk_index": 0,
                "char_start": 0,
                "char_end": 3,
                "excerpt": "def",
            }
        ],
        **protected,
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExtractionCandidate.model_validate(payload)


class InvalidProvider:
    descriptor = ProviderDescriptor(provider="test", model="malicious")

    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, request: Any) -> list[dict[str, object]]:
        self.calls += 1
        return [
            {
                "kind": "observation",
                "content": "provider tried to confirm this",
                "state": "confirmed",
                "spans": [
                    {
                        "chunk_index": 0,
                        "char_start": 0,
                        "char_end": 3,
                        "excerpt": request.raw_content[:3],
                    }
                ],
            }
        ]


@pytest.mark.asyncio
async def test_invalid_provider_falls_back_and_raw_source_survives(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    provider = InvalidProvider()
    service = ExtractionService(store, CodingRuleExtractor(), provider=provider)
    actor = make_actor(scope_ids)
    ingested = await service.ingest(actor, _command(use_provider=True))
    worker = ExtractionWorker(store, store, service)

    applied = await worker.run_once("worker-provider")

    assert provider.calls == 1
    assert store.sources[ingested.source.source_id].raw_content == _command().content
    assert len(applied) == 1
    assert applied[0].item.outcome == "fallback"
    assert applied[0].effects
    provider_attempt = store.attempts[(ingested.work_id, 1, "provider")]
    validation_attempt = store.attempts[(ingested.work_id, 1, "validation")]
    assert provider_attempt.output_sha256 is not None
    assert provider_attempt.outcome is ExtractionOutcome.SUCCEEDED
    assert validation_attempt.input_sha256 == provider_attempt.output_sha256
    assert validation_attempt.outcome is ExtractionOutcome.REJECTED
    for effect in applied[0].effects:
        persisted = store.effect_resources[effect.resource_id]
        assert persisted["state"] == "tentative"
        assert persisted["evidence"]
        assert persisted["supersedes_memory_id"] is None


@pytest.mark.asyncio
async def test_lazy_unavailable_provider_is_not_loaded_until_selected_and_falls_back(
    scope_ids: dict[str, str],
) -> None:
    factory_calls = 0

    def unavailable() -> Any:
        nonlocal factory_calls
        factory_calls += 1
        raise RuntimeError("credential is absent")

    lazy = LazyExtractionProvider(
        ProviderDescriptor(provider="optional", model="remote"),
        unavailable,
    )
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor(), provider=lazy)
    actor = make_actor(scope_ids)
    await service.ingest(actor, _command(use_provider=False))

    await ExtractionWorker(store, store, service).run_once("worker-local")
    assert factory_calls == 0

    await service.ingest(
        actor,
        _command(
            idempotency_key="raw-source-2",
            content="class Cache:\n    pass\n",
            use_provider=True,
        ).model_copy(update={"occurrence_key": "src/cache.py@abc123"}),
    )
    applied = await ExtractionWorker(store, store, service).run_once("worker-remote")

    assert factory_calls == 1
    assert applied[0].item.outcome == "fallback"


@pytest.mark.asyncio
async def test_two_workers_cannot_claim_the_same_live_lease(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    await service.ingest(make_actor(scope_ids), _command())

    first, second = await asyncio.gather(
        store.claim_work(ClaimWorkCommand(worker_id="worker-a")),
        store.claim_work(ClaimWorkCommand(worker_id="worker-b")),
    )

    leases = (*first.leases, *second.leases)
    assert len(leases) == 1
    assert leases[0].item.status is WorkStatus.LEASED


@pytest.mark.asyncio
async def test_expired_worker_fence_cannot_apply_after_reclaim(
    scope_ids: dict[str, str],
) -> None:
    clock = MutableClock()
    store = InMemoryWorkStore(clock=clock)
    service = ExtractionService(store, CodingRuleExtractor())
    actor = make_actor(scope_ids)
    ingested = await service.ingest(actor, _command())
    stale = (
        await store.claim_work(ClaimWorkCommand(worker_id="worker-stale", lease_seconds=5))
    ).leases[0]
    clock.now += timedelta(seconds=6)
    current = (
        await store.claim_work(ClaimWorkCommand(worker_id="worker-current", lease_seconds=5))
    ).leases[0]

    with pytest.raises(WorkLeaseLost):
        await store.apply_extraction(
            ApplyExtractionWorkCommand(
                work_id=stale.item.work_id,
                worker_id=stale.worker_id,
                lease_token=stale.lease_token,
                lease_version=stale.lease_version,
                expected_work_version=stale.work_version,
                attempt=stale.attempt,
                provenance=(_provenance(stale.attempt, ingested.source.content_sha256),),
            )
        )
    assert current.lease_version == stale.lease_version + 1


@pytest.mark.asyncio
async def test_repeated_delivery_replays_one_durable_effect(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    actor = make_actor(scope_ids)
    await service.ingest(actor, _command())
    lease = (await store.claim_work(ClaimWorkCommand(worker_id="worker-once"))).leases[0]
    request = await store.load_extraction_input(lease)
    extraction = await service.extract(request)
    candidate = extraction.candidates[0]
    digest = _payload_hash(candidate)
    command = ApplyExtractionWorkCommand(
        work_id=lease.item.work_id,
        worker_id=lease.worker_id,
        lease_token=lease.lease_token,
        lease_version=lease.lease_version,
        expected_work_version=lease.work_version,
        attempt=lease.attempt,
        effects=(
            WorkEffect(
                effect_key=f"memory:{digest}",
                payload_sha256=digest,
                candidate=candidate,
            ),
        ),
        provenance=extraction.provenance,
        result={"candidate_count": 1},
    )

    first = await store.apply_extraction(command)
    replay = await store.apply_extraction(command)

    assert first.replayed is False
    assert replay.replayed is True
    assert first.effects == replay.effects
    assert len(store.effects) == 1
    assert len(store.effect_resources) == 1
    assert len(store.events) == 1
    assert len(store.outbox_events) == 1


@pytest.mark.asyncio
async def test_apply_rejects_a_span_that_does_not_quote_the_raw_source(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    actor = make_actor(scope_ids)
    ingested = await service.ingest(actor, _command())
    lease = (await store.claim_work(ClaimWorkCommand(worker_id="worker-evidence"))).leases[0]
    candidate = ExtractionCandidate(
        kind=MemoryKind.OBSERVATION,
        content="candidate with a forged quotation",
        spans=(
            SourceSpan(
                chunk_index=0,
                char_start=0,
                char_end=3,
                excerpt="xyz",
            ),
        ),
    )
    digest = _payload_hash(candidate)

    with pytest.raises(WorkEffectConflict, match="evidence_span"):
        await store.apply_extraction(
            ApplyExtractionWorkCommand(
                work_id=lease.item.work_id,
                worker_id=lease.worker_id,
                lease_token=lease.lease_token,
                lease_version=lease.lease_version,
                expected_work_version=lease.work_version,
                attempt=lease.attempt,
                effects=(
                    WorkEffect(
                        effect_key=f"memory:{digest}",
                        payload_sha256=digest,
                        candidate=candidate,
                    ),
                ),
                provenance=(_provenance(lease.attempt, ingested.source.content_sha256),),
            )
        )

    assert store.effects == {}
    assert store.effect_resources == {}


class CountingProvider:
    descriptor = ProviderDescriptor(provider="test", model="counting")

    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, request: Any) -> list[dict[str, object]]:
        self.calls += 1
        excerpt = request.raw_content[:3]
        return [
            {
                "kind": "observation",
                "content": "provider candidate",
                "spans": [
                    {
                        "chunk_index": 0,
                        "char_start": 0,
                        "char_end": len(excerpt),
                        "excerpt": excerpt,
                    }
                ],
            }
        ]


class InternalRetryQueue:
    """Simulate a queue adapter retrying a 40001 inside apply_extraction."""

    def __init__(self, delegate: InMemoryWorkStore) -> None:
        self.delegate = delegate
        self.transaction_attempts = 0

    async def claim_work(self, command: Any) -> Any:
        return await self.delegate.claim_work(command)

    async def apply_extraction(self, command: Any) -> Any:
        self.transaction_attempts += 1
        try:
            raise SerializationFailure()
        except SerializationFailure as exc:
            assert getattr(exc, "sqlstate", None) == "40001"
        self.transaction_attempts += 1
        return await self.delegate.apply_extraction(command)

    async def fail_work(self, command: Any) -> Any:
        return await self.delegate.fail_work(command)


class SerializationFailure(RuntimeError):
    sqlstate = "40001"


@pytest.mark.asyncio
async def test_provider_is_called_once_when_apply_transaction_retries_40001(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    provider = CountingProvider()
    service = ExtractionService(store, CodingRuleExtractor(), provider=provider)
    actor = make_actor(scope_ids)
    await service.ingest(actor, _command(use_provider=True))
    queue = InternalRetryQueue(store)
    worker = ExtractionWorker(queue, store, service)

    applied = await worker.run_once("worker-retry")

    assert len(applied) == 1
    assert provider.calls == 1
    assert queue.transaction_attempts == 2
    assert len(store.effects) == len(applied[0].effects)


@pytest.mark.asyncio
async def test_untrusted_source_is_preserved_but_never_claimed(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    ingested = await service.ingest(
        make_actor(scope_ids),
        _command(trust=SourceTrust.UNTRUSTED),
    )

    batch = await store.claim_work(
        ClaimWorkCommand(
            worker_id="worker-safe",
            kinds=frozenset({WorkKind.EXTRACT_SOURCE}),
        )
    )

    assert store.sources[ingested.source.source_id].raw_content == _command().content
    assert store.items[ingested.work_id].status is WorkStatus.CANCELLED
    assert batch.leases == ()


@pytest.mark.asyncio
async def test_unsupported_source_is_preserved_but_never_claimed(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    command = _command(idempotency_key="raw-url-1").model_copy(
        update={
            "kind": EvidenceKind.URL,
            "occurrence_key": "https://example.invalid/source",
        }
    )
    ingested = await service.ingest(make_actor(scope_ids), command)

    batch = await store.claim_work(
        ClaimWorkCommand(
            worker_id="worker-supported-only",
            kinds=frozenset({WorkKind.EXTRACT_SOURCE}),
        )
    )

    assert store.sources[ingested.source.source_id].raw_content == command.content
    assert store.items[ingested.work_id].status is WorkStatus.CANCELLED
    assert store.items[ingested.work_id].outcome == "source_kind_unsupported"
    assert batch.leases == ()
