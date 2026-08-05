"""Deterministic extraction for explicit structured-memory source envelopes."""

from __future__ import annotations

import json

from pydantic import Field

from swarmbrain.domain.common import ContractModel
from swarmbrain.domain.evidence import EvidenceKind
from swarmbrain.domain.extraction import ExtractionCandidate, ExtractionInput

from .coding import CodingRuleExtractor

STRUCTURED_MEMORY_SOURCE_KIND = "application/vnd.swarmbrain.memory+json"


class StructuredMemoryEnvelope(ContractModel):
    """Caller-authored proposals; storage identity and lifecycle remain absent."""

    memories: tuple[ExtractionCandidate, ...] = Field(min_length=1, max_length=128)


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


class DefaultRuleExtractor:
    """Route source code to exact-span rules and explicit JSON envelopes to Pydantic."""

    name = "default-rules"
    revision = "1"

    def __init__(self, *, max_candidates: int = 64) -> None:
        self.coding = CodingRuleExtractor(max_candidates=max_candidates)

    async def extract(self, request: ExtractionInput) -> tuple[ExtractionCandidate, ...]:
        if request.source.kind == EvidenceKind.SOURCE_CODE:
            return await self.coding.extract(request)
        if str(request.source.kind) != STRUCTURED_MEMORY_SOURCE_KIND:
            return ()
        payload = json.loads(
            request.raw_content,
            parse_constant=_reject_nonstandard_constant,
        )
        return StructuredMemoryEnvelope.model_validate(payload).memories


__all__ = [
    "DefaultRuleExtractor",
    "STRUCTURED_MEMORY_SOURCE_KIND",
    "StructuredMemoryEnvelope",
]
