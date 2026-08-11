"""Fail-closed deterministic consolidation fallback."""

from __future__ import annotations

from swarmbrain.domain.consolidation import (
    ConsolidationActionKind,
    ConsolidationProposal,
    ConsolidationWorkPayload,
)


class SafeDeterministicConsolidator:
    """Abstain rather than invent a summary without a semantic reflector."""

    name = "swarmbrain-safe-consolidator"
    revision = "v1"

    async def reflect(
        self,
        request: ConsolidationWorkPayload,
    ) -> tuple[ConsolidationProposal, ...]:
        del request
        return (
            ConsolidationProposal(
                action=ConsolidationActionKind.NOOP,
                reason="no configured semantic reflector; preserve the evidence-backed originals",
            ),
        )


__all__ = ["SafeDeterministicConsolidator"]
