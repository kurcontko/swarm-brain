"""Lease-driven extraction worker with provider calls outside database transactions."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from datetime import UTC, datetime, timedelta

from swarmbrain.application.extraction import ExtractionService
from swarmbrain.domain.extraction import ExtractionCandidate, ExtractionStage
from swarmbrain.domain.work import (
    ApplyExtractionWorkCommand,
    ApplyWorkResult,
    ClaimWorkCommand,
    FailWorkCommand,
    WorkEffect,
    WorkKind,
    WorkLease,
    WorkLeaseLost,
)
from swarmbrain.ports.extraction import SourceIngestStore
from swarmbrain.ports.work_queue import WorkQueueStore


def _candidate_hash(candidate: ExtractionCandidate) -> str:
    payload = json.dumps(
        candidate.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_origin(
    payload_hash: str,
    deterministic_hashes: set[str],
    provider_hashes: set[str],
) -> ExtractionStage:
    if payload_hash in deterministic_hashes:
        return ExtractionStage.DETERMINISTIC
    if payload_hash in provider_hashes:
        return ExtractionStage.PROVIDER
    raise ValueError("combined extraction candidate has no recorded origin")


class ExtractionWorker:
    """Compute outside the transaction, then apply all effects through one fence."""

    def __init__(
        self,
        queue: WorkQueueStore,
        sources: SourceIngestStore,
        extraction: ExtractionService,
        *,
        retry_delay_seconds: int = 30,
    ) -> None:
        if retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds must be positive")
        self.queue = queue
        self.sources = sources
        self.extraction = extraction
        self.retry_delay_seconds = retry_delay_seconds

    async def run_once(
        self,
        worker_id: str,
        *,
        limit: int = 1,
        lease_seconds: int = 60,
    ) -> tuple[ApplyWorkResult, ...]:
        batch = await self.queue.claim_work(
            ClaimWorkCommand(
                worker_id=worker_id,
                kinds=frozenset({WorkKind.EXTRACT_SOURCE}),
                limit=limit,
                lease_seconds=lease_seconds,
                extractor_name=self.extraction.deterministic.name,
                extractor_revision=self.extraction.deterministic.revision,
            )
        )
        applied: list[ApplyWorkResult] = []
        for lease in batch.leases:
            result = await self._process(lease)
            if result is not None:
                applied.append(result)
        return tuple(applied)

    async def _process(self, lease: WorkLease) -> ApplyWorkResult | None:
        try:
            request = await self.sources.load_extraction_input(lease)
            if (
                lease.item.payload.get("extractor_name") != self.extraction.deterministic.name
                or lease.item.payload.get("extractor_revision")
                != self.extraction.deterministic.revision
            ):
                raise ValueError("leased work requires a different extractor profile")
            # This may invoke a remote provider. It deliberately precedes the
            # short, retryable apply transaction below.
            extraction = await self.extraction.extract(
                request,
                use_provider=bool(lease.item.payload.get("use_provider", False)),
            )
            deterministic_hashes = {
                _candidate_hash(candidate) for candidate in extraction.deterministic_candidates
            }
            provider_hashes = {
                _candidate_hash(candidate) for candidate in extraction.provider_candidates
            }
            effects = tuple(
                WorkEffect(
                    effect_key=f"memory:{payload_hash}",
                    payload_sha256=payload_hash,
                    candidate=candidate,
                    candidate_origin=_candidate_origin(
                        payload_hash,
                        deterministic_hashes,
                        provider_hashes,
                    ),
                )
                for candidate in extraction.candidates
                for payload_hash in (_candidate_hash(candidate),)
            )
            return await self.queue.apply_extraction(
                ApplyExtractionWorkCommand(
                    work_id=lease.item.work_id,
                    worker_id=lease.worker_id,
                    lease_token=lease.lease_token,
                    lease_version=lease.lease_version,
                    expected_work_version=lease.work_version,
                    attempt=lease.attempt,
                    effects=effects,
                    provenance=extraction.provenance,
                    outcome=extraction.status.value,
                    result={
                        "route": extraction.route.value,
                        "status": extraction.status.value,
                        "candidate_count": len(extraction.candidates),
                        "fallback_reason": extraction.fallback_reason,
                    },
                )
            )
        except WorkLeaseLost:
            return None
        except Exception as exc:
            error = f"worker_{type(exc).__name__}"
            with suppress(WorkLeaseLost):
                await self.queue.fail_work(
                    FailWorkCommand(
                        work_id=lease.item.work_id,
                        worker_id=lease.worker_id,
                        lease_token=lease.lease_token,
                        lease_version=lease.lease_version,
                        expected_work_version=lease.work_version,
                        attempt=lease.attempt,
                        error=error,
                        retry_at=datetime.now(UTC) + timedelta(seconds=self.retry_delay_seconds),
                    )
                )
            return None


__all__ = ["ExtractionWorker"]
