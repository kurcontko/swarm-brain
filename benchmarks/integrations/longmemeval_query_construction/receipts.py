"""Offline raw-response replay into source-bound E7 constructor receipts."""

from __future__ import annotations

from .contracts import (
    CONSTRUCTOR_INPUT_FIELDS,
    PROVIDER_RESPONSE_PARSER,
    ConstructionAccounting,
    ConstructorIdentity,
    QueryConstructionError,
    QueryWindow,
    WindowConstructionReceipt,
    replay_constructor_provider_response,
    sha256_bytes,
    sha256_json,
)
from .windowing import QueryWindowBatch, constructor_request_bytes


def replay_window_construction_receipt(
    batch: QueryWindowBatch,
    window: QueryWindow,
    *,
    raw_response: bytes,
    request_id: str,
    identity: ConstructorIdentity,
    latency_ms: float,
    cost_usd: float,
) -> WindowConstructionReceipt:
    """Build a receipt solely from a frozen request and exact provider bytes.

    The provider request ID, response model, decisions, and token usage are
    parsed from ``raw_response``.  Only the local request ID, observed latency,
    and externally priced cost remain caller supplied.
    """

    if not isinstance(batch, QueryWindowBatch):
        raise QueryConstructionError("receipt replay requires a QueryWindowBatch")
    if not isinstance(window, QueryWindow):
        raise QueryConstructionError("receipt replay requires a QueryWindow")
    if not isinstance(identity, ConstructorIdentity):
        raise QueryConstructionError("receipt replay requires a constructor identity")
    request = constructor_request_bytes(batch, window)
    replayed = replay_constructor_provider_response(
        raw_response,
        turn_ids=tuple(message.turn.turn_id for message in window.messages),
    )
    if replayed.response_model != identity.model:
        raise QueryConstructionError(
            "provider response model differs from the pinned constructor model"
        )
    decisions = replayed.decisions
    return WindowConstructionReceipt(
        question_id=batch.pool.question_id,
        window_sha256=window.window_sha256,
        request_sha256=sha256_bytes(request),
        request_utf8_bytes=len(request),
        response_parser=PROVIDER_RESPONSE_PARSER,
        raw_response=raw_response,
        response_sha256=sha256_json([decision.response_payload() for decision in decisions]),
        request_id=request_id,
        provider_request_id=replayed.provider_request_id,
        input_fields=CONSTRUCTOR_INPUT_FIELDS,
        identity=identity,
        accounting=ConstructionAccounting(
            input_tokens=replayed.input_tokens,
            output_tokens=replayed.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        ),
        decisions=decisions,
    )


__all__ = ["replay_window_construction_receipt"]
