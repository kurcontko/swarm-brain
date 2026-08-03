"""Raw ingest and hybrid extraction orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.common import utc_now
from swarmbrain.domain.evidence import EvidenceKind, SourceReviewState, SourceTrust
from swarmbrain.domain.extraction import (
    GENERAL_SOURCE_KINDS,
    ExtractionCandidate,
    ExtractionInput,
    ExtractionOutcome,
    ExtractionProvenance,
    ExtractionResult,
    ExtractionRoute,
    ExtractionRunStatus,
    ExtractionStage,
    IngestRawSourceCommand,
    PreparedSourceIngest,
    SourceChunk,
    SourceIngestResult,
)
from swarmbrain.ports.extraction import (
    DeterministicExtractor,
    ExtractionProvider,
    ProviderCandidate,
    SourceIngestStore,
)

from .capabilities import require_capability


def _sha256(value: bytes | str) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _candidate_hash(candidate: ExtractionCandidate) -> str:
    return _sha256(_json_bytes(candidate.model_dump(mode="json")))


class ExtractionService:
    """Preserve raw input first, then run fallible extraction outside storage TXs."""

    def __init__(
        self,
        ingest_store: SourceIngestStore,
        deterministic: DeterministicExtractor,
        *,
        provider: ExtractionProvider | None = None,
    ) -> None:
        self.ingest_store = ingest_store
        self.deterministic = deterministic
        self.provider = provider

    async def ingest(
        self,
        actor: ActorContext,
        command: IngestRawSourceCommand,
    ) -> SourceIngestResult:
        require_capability(actor, Capability.SOURCE_INGEST)
        prepared = self.prepare_ingest(actor, command)
        return await self.ingest_store.ingest_source(actor, prepared)

    def prepare_ingest(
        self,
        actor: ActorContext,
        command: IngestRawSourceCommand,
    ) -> PreparedSourceIngest:
        occurrence = command.occurrence_key or command.idempotency_key
        source_id = uuid5(
            NAMESPACE_URL,
            ":".join(
                (
                    "swarmbrain-source",
                    actor.tenant_id,
                    actor.repository_id,
                    actor.run_id,
                    occurrence,
                )
            ),
        )
        work_id = uuid5(
            NAMESPACE_URL,
            f"swarmbrain-extract:{source_id}:{self.deterministic.name}:{self.deterministic.revision}",
        )
        chunks = self._chunks(source_id, command.content, command.chunk_chars)
        return PreparedSourceIngest(
            source_id=str(source_id),
            work_id=str(work_id),
            command=command,
            chunks=chunks,
            extractor_name=self.deterministic.name,
            extractor_revision=self.deterministic.revision,
            work_dedupe_key=(
                f"extract_source:{source_id}:{self.deterministic.name}:"
                f"{self.deterministic.revision}"
            ),
        )

    async def extract(
        self,
        request: ExtractionInput,
        *,
        use_provider: bool = False,
    ) -> ExtractionResult:
        """Generate candidates only; callers persist them in a later fenced TX."""

        route, skip_reason = self._route(request)
        input_sha256 = request.source.content_sha256
        if route is ExtractionRoute.SKIP:
            now = utc_now()
            provenance = ExtractionProvenance(
                stage=ExtractionStage.DETERMINISTIC,
                attempt=request.attempt,
                outcome=ExtractionOutcome.SKIPPED,
                input_sha256=input_sha256,
                error=skip_reason,
                started_at=now,
                finished_at=now,
            )
            return ExtractionResult(
                route=route,
                status=ExtractionRunStatus.SKIPPED,
                provenance=(provenance,),
                fallback_reason=skip_reason,
            )

        deterministic_started = utc_now()
        deterministic = (
            tuple(await self.deterministic.extract(request))
            if route is ExtractionRoute.CODING
            else ()
        )
        deterministic = tuple(self._validated_candidates(deterministic, request))
        deterministic_finished = utc_now()
        provenance: list[ExtractionProvenance] = [
            ExtractionProvenance(
                stage=ExtractionStage.DETERMINISTIC,
                attempt=request.attempt,
                outcome=ExtractionOutcome.SUCCEEDED,
                input_sha256=input_sha256,
                output_sha256=_sha256(
                    _json_bytes([item.model_dump(mode="json") for item in deterministic])
                ),
                provider="local",
                model=self.deterministic.name,
                revision=self.deterministic.revision,
                candidate_count=len(deterministic),
                started_at=deterministic_started,
                finished_at=deterministic_finished,
            )
        ]

        if not use_provider:
            return ExtractionResult(
                route=route,
                status=ExtractionRunStatus.COMPLETED,
                candidates=deterministic,
                deterministic_candidates=deterministic,
                provenance=tuple(provenance),
            )

        if self.provider is None:
            return self._fallback(
                route,
                request,
                deterministic,
                provenance,
                "provider_not_configured",
            )

        provider_started = utc_now()
        try:
            descriptor = self.provider.descriptor
            raw_candidates = tuple(await self.provider.extract(request))
            raw_hash = _sha256(_json_bytes(self._serializable_provider_output(raw_candidates)))
        except Exception as exc:
            # Persist only a bounded class name. Provider messages may contain secrets or payloads.
            return self._fallback(
                route,
                request,
                deterministic,
                provenance,
                f"provider_{type(exc).__name__}",
                started_at=provider_started,
            )

        provider_finished = utc_now()
        provenance.append(
            ExtractionProvenance(
                stage=ExtractionStage.PROVIDER,
                attempt=request.attempt,
                outcome=ExtractionOutcome.SUCCEEDED,
                input_sha256=input_sha256,
                output_sha256=raw_hash,
                provider=descriptor.provider,
                model=descriptor.model,
                revision=descriptor.revision,
                prompt_id=descriptor.prompt_id,
                prompt_sha256=descriptor.prompt_sha256,
                candidate_count=len(raw_candidates),
                started_at=provider_started,
                finished_at=provider_finished,
            )
        )

        validation_started = utc_now()
        try:
            provider_candidates = tuple(self._validated_candidates(raw_candidates, request))
        except Exception as exc:
            return self._fallback(
                route,
                request,
                deterministic,
                provenance,
                f"validation_{type(exc).__name__}",
                stage=ExtractionStage.VALIDATION,
                outcome=ExtractionOutcome.REJECTED,
                input_sha256=raw_hash,
                candidate_count=len(raw_candidates),
                started_at=validation_started,
            )

        validation_finished = utc_now()
        provenance.append(
            ExtractionProvenance(
                stage=ExtractionStage.VALIDATION,
                attempt=request.attempt,
                outcome=ExtractionOutcome.SUCCEEDED,
                input_sha256=raw_hash,
                output_sha256=_sha256(
                    _json_bytes([item.model_dump(mode="json") for item in provider_candidates])
                ),
                provider=descriptor.provider,
                model=descriptor.model,
                revision=descriptor.revision,
                prompt_id=descriptor.prompt_id,
                prompt_sha256=descriptor.prompt_sha256,
                candidate_count=len(provider_candidates),
                started_at=validation_started,
                finished_at=validation_finished,
            )
        )
        combined = self._dedupe((*deterministic, *provider_candidates))
        return ExtractionResult(
            route=route,
            status=ExtractionRunStatus.COMPLETED,
            candidates=combined,
            deterministic_candidates=deterministic,
            provider_candidates=provider_candidates,
            provenance=tuple(provenance),
        )

    @staticmethod
    def _chunks(source_id: UUID, content: str, chunk_chars: int) -> tuple[SourceChunk, ...]:
        chunks: list[SourceChunk] = []
        for index, start in enumerate(range(0, len(content), chunk_chars)):
            chunk_content = content[start : start + chunk_chars]
            chunk_id = uuid5(source_id, f"chunk:{index}:{_sha256(chunk_content)}")
            chunks.append(
                SourceChunk(
                    chunk_id=str(chunk_id),
                    source_id=str(source_id),
                    chunk_index=index,
                    content=chunk_content,
                    content_sha256=_sha256(chunk_content),
                    token_count=len(chunk_content.split()),
                    char_start=start,
                    char_end=start + len(chunk_content),
                )
            )
        return tuple(chunks)

    @staticmethod
    def _route(request: ExtractionInput) -> tuple[ExtractionRoute, str | None]:
        source = request.source
        if source.review_state is SourceReviewState.REJECTED:
            return ExtractionRoute.SKIP, "source_rejected"
        if source.trust is SourceTrust.UNTRUSTED:
            return ExtractionRoute.SKIP, "source_untrusted"
        if source.kind == EvidenceKind.SOURCE_CODE:
            return ExtractionRoute.CODING, None
        # Every trusted textual source is preservable and eligible for the
        # general extractor.  Registered built-ins only select specialized
        # routing; an application-defined source label is not an error.
        return ExtractionRoute.GENERAL, None

    @staticmethod
    def _serializable_provider_output(values: Sequence[ProviderCandidate]) -> list[object]:
        return [
            value.model_dump(mode="json") if isinstance(value, ExtractionCandidate) else dict(value)
            for value in values
        ]

    def _validated_candidates(
        self,
        values: Sequence[ProviderCandidate],
        request: ExtractionInput,
    ) -> list[ExtractionCandidate]:
        candidates: list[ExtractionCandidate] = []
        chunks = {chunk.chunk_index: chunk for chunk in request.chunks}
        for value in values:
            payload: Mapping[str, object] | ExtractionCandidate = value
            candidate = (
                payload
                if isinstance(payload, ExtractionCandidate)
                else ExtractionCandidate.model_validate(payload)
            )
            for span in candidate.spans:
                chunk = chunks.get(span.chunk_index)
                if chunk is None:
                    raise ValueError("candidate span refers to an unknown chunk")
                if span.char_start < chunk.char_start or span.char_end > chunk.char_end:
                    raise ValueError("candidate span falls outside its chunk")
                if request.raw_content[span.char_start : span.char_end] != span.excerpt:
                    raise ValueError("candidate span does not quote the raw source")
            candidates.append(candidate)
        return candidates

    @staticmethod
    def _dedupe(values: Sequence[ExtractionCandidate]) -> tuple[ExtractionCandidate, ...]:
        unique: dict[str, ExtractionCandidate] = {}
        for candidate in values:
            unique.setdefault(_candidate_hash(candidate), candidate)
        return tuple(unique.values())

    def _fallback(
        self,
        route: ExtractionRoute,
        request: ExtractionInput,
        deterministic: tuple[ExtractionCandidate, ...],
        provenance: list[ExtractionProvenance],
        reason: str,
        *,
        stage: ExtractionStage = ExtractionStage.PROVIDER,
        outcome: ExtractionOutcome = ExtractionOutcome.FALLBACK,
        input_sha256: str | None = None,
        output_sha256: str | None = None,
        candidate_count: int = 0,
        started_at: datetime | None = None,
    ) -> ExtractionResult:
        try:
            descriptor = self.provider.descriptor if self.provider is not None else None
        except Exception:
            descriptor = None
        timestamp = utc_now()
        provenance.append(
            ExtractionProvenance(
                stage=stage,
                attempt=request.attempt,
                outcome=outcome,
                input_sha256=input_sha256 or request.source.content_sha256,
                output_sha256=output_sha256,
                provider=descriptor.provider if descriptor else None,
                model=descriptor.model if descriptor else None,
                revision=descriptor.revision if descriptor else None,
                prompt_id=descriptor.prompt_id if descriptor else None,
                prompt_sha256=descriptor.prompt_sha256 if descriptor else None,
                candidate_count=candidate_count,
                error=reason,
                started_at=started_at or timestamp,
                finished_at=timestamp,
            )
        )
        return ExtractionResult(
            route=route,
            status=ExtractionRunStatus.FALLBACK,
            candidates=deterministic,
            deterministic_candidates=deterministic,
            provenance=tuple(provenance),
            fallback_reason=reason,
        )


__all__ = ["ExtractionService", "GENERAL_SOURCE_KINDS"]
