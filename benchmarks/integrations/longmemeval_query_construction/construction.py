"""Validate external E7 decisions and compile source-bound reader material."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .contracts import (
    ARTIFACT_TYPE,
    CONSTRUCTOR_INPUT_FIELDS,
    EMPTY_MEMORY_CONTEXT,
    EXTRACTIVE_SEPARATOR,
    PAPER_REPOSITORY,
    PAPER_REPOSITORY_COMMIT,
    PAPER_URL,
    PAPER_WINDOW_PROTOCOL,
    PROTOCOL_VERSION,
    READER_CONTEXT_SERIALIZER,
    RECEIPT_AUTHENTICATION,
    SCHEMA_VERSION,
    ConstructedMemoryItem,
    ConstructionCell,
    ConstructionDecision,
    DecisionOperation,
    QueryConstructionError,
    QueryConstructionResult,
    RetentionStyle,
    SourceSpan,
    WindowConstructionReceipt,
    WindowMessage,
    canonical_json_bytes,
    render_reader_context,
    sha256_json,
    sha256_utf8,
)
from .windowing import (
    CONSTRUCTOR_SYSTEM_PROMPT_SHA256,
    CONSTRUCTOR_USER_PROMPT_SHA256,
    QueryWindowBatch,
    constructor_request_bytes,
    implementation_fingerprint,
)


def _span_material(message: WindowMessage, span: SourceSpan) -> str:
    raw = message.turn.original_content.encode("utf-8")
    if span.start_byte >= len(raw) or span.end_byte > len(raw):
        raise QueryConstructionError("support span is outside its immutable source message")
    try:
        return raw[span.start_byte : span.end_byte].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QueryConstructionError("support span splits a UTF-8 code point") from exc


def _validate_support(
    decision: ConstructionDecision,
    message: WindowMessage,
    *,
    grounded_only: bool,
) -> tuple[str, ...]:
    if decision.turn_id != message.turn.turn_id:
        raise QueryConstructionError("constructor decision is not aligned to its window message")
    if decision.operation is DecisionOperation.DROP:
        return ()
    if grounded_only and decision.style is RetentionStyle.ABSTRACTIVE:
        raise QueryConstructionError("E7-C forbids caller-attested abstractive construction")

    previous_end = -1
    material: list[str] = []
    for span in decision.support_spans:
        if span.start_byte < previous_end:
            raise QueryConstructionError("support spans must be ordered and non-overlapping")
        material.append(_span_material(message, span))
        previous_end = span.end_byte

    if decision.style is RetentionStyle.VERBATIM:
        if len(material) != 1 or decision.compressed_content != material[0]:
            raise QueryConstructionError(
                "verbatim KEEP must exactly equal its single cited UTF-8 source span"
            )
    elif decision.style is RetentionStyle.EXTRACTIVE:
        if decision.compressed_content != EXTRACTIVE_SEPARATOR.join(material):
            raise QueryConstructionError("extractive KEEP must exactly join its cited source spans")
    elif decision.style is RetentionStyle.ABSTRACTIVE:
        # The cited spans make provenance inspectable, but an offline equality
        # check cannot establish semantic faithfulness of a paraphrase.
        pass
    else:  # pragma: no cover - dataclass validation excludes DROP on KEEP
        raise QueryConstructionError("KEEP uses an unsupported retention style")
    return tuple(material)


def _validate_receipts(
    batch: QueryWindowBatch,
    receipts: tuple[WindowConstructionReceipt, ...],
    *,
    cell: ConstructionCell,
) -> tuple[dict[int, WindowConstructionReceipt], dict[str, Any]]:
    if not isinstance(receipts, tuple):
        raise QueryConstructionError("constructor receipts must be frozen in a tuple")
    if len(receipts) != len(batch.windows):
        raise QueryConstructionError("constructor receipts must cover every window exactly once")
    if cell not in {
        ConstructionCell.QUERY_CONSTRUCTION,
        ConstructionCell.GROUNDED_QUERY_CONSTRUCTION,
    }:
        raise QueryConstructionError("external receipts are valid only for E7-B or E7-C")

    by_window: dict[int, WindowConstructionReceipt] = {}
    request_ids: set[str] = set()
    provider_ids: set[str] = set()
    identity_sha256: str | None = None
    input_tokens = 0
    output_tokens = 0
    provider_response_bytes = 0
    latency_sum = 0.0
    latency_max = 0.0
    cost = 0.0
    abstractive_keeps = 0
    extractive_keeps = 0
    verbatim_keeps = 0

    window_by_sha = {window.window_sha256: window for window in batch.windows}
    if len(window_by_sha) != len(batch.windows):
        raise QueryConstructionError("window batch repeats a content-free window identity")

    for receipt in receipts:
        if not isinstance(receipt, WindowConstructionReceipt):
            raise QueryConstructionError("constructor receipts contain an invalid value")
        window = window_by_sha.get(receipt.window_sha256)
        if window is None:
            raise QueryConstructionError("constructor receipt refers to an unknown window")
        if window.window_index in by_window:
            raise QueryConstructionError("constructor receipts repeat a window")
        if receipt.question_id != batch.pool.question_id:
            raise QueryConstructionError("constructor receipt is cross-question")
        request = constructor_request_bytes(batch, window)
        if receipt.request_sha256 != sha256_utf8(request.decode("utf-8")):
            raise QueryConstructionError("constructor request digest differs from frozen material")
        if receipt.request_utf8_bytes != len(request):
            raise QueryConstructionError("constructor request byte count is inconsistent")
        if receipt.input_fields != CONSTRUCTOR_INPUT_FIELDS:
            raise QueryConstructionError("constructor input allowlist drifted")
        if receipt.identity.system_prompt_sha256 != CONSTRUCTOR_SYSTEM_PROMPT_SHA256:
            raise QueryConstructionError("constructor system prompt identity drifted")
        if receipt.identity.user_prompt_sha256 != CONSTRUCTOR_USER_PROMPT_SHA256:
            raise QueryConstructionError("constructor user prompt identity drifted")
        current_identity = receipt.identity.identity_sha256
        if identity_sha256 is None:
            identity_sha256 = current_identity
        elif current_identity != identity_sha256:
            raise QueryConstructionError("constructor identity drifted across windows")
        if receipt.request_id in request_ids:
            raise QueryConstructionError("constructor request ID was reused")
        if receipt.provider_request_id in provider_ids:
            raise QueryConstructionError("constructor provider request ID was reused")
        request_ids.add(receipt.request_id)
        provider_ids.add(receipt.provider_request_id)
        if len(receipt.decisions) != len(window.messages):
            raise QueryConstructionError(
                "constructor must return exactly one decision per window message"
            )
        for message, decision in zip(window.messages, receipt.decisions, strict=True):
            _validate_support(
                decision,
                message,
                grounded_only=(cell is ConstructionCell.GROUNDED_QUERY_CONSTRUCTION),
            )
            if decision.operation is DecisionOperation.KEEP:
                if decision.style is RetentionStyle.ABSTRACTIVE:
                    abstractive_keeps += 1
                elif decision.style is RetentionStyle.EXTRACTIVE:
                    extractive_keeps += 1
                elif decision.style is RetentionStyle.VERBATIM:
                    verbatim_keeps += 1
        accounting = receipt.accounting
        input_tokens += accounting.input_tokens
        output_tokens += accounting.output_tokens
        provider_response_bytes += len(receipt.raw_response)
        latency_sum += accounting.latency_ms
        latency_max = max(latency_max, accounting.latency_ms)
        cost += accounting.cost_usd
        by_window[window.window_index] = receipt

    if set(by_window) != set(range(len(batch.windows))):
        raise QueryConstructionError("constructor receipt coverage is incomplete")
    return by_window, {
        "constructor_identity_sha256": identity_sha256,
        "constructor_windows": len(receipts),
        "constructor_input_tokens": input_tokens,
        "constructor_output_tokens": output_tokens,
        "constructor_provider_response_bytes": provider_response_bytes,
        "constructor_usage_replayed_windows": len(receipts),
        "constructor_latency_sum_ms": latency_sum,
        "constructor_window_max_latency_ms": latency_max,
        "constructor_batch_wall_clock_ms": None,
        "constructor_cost_usd": cost,
        "verbatim_keep_decisions": verbatim_keeps,
        "extractive_keep_decisions": extractive_keeps,
        "abstractive_keep_decisions": abstractive_keeps,
    }


def _raw_result(batch: QueryWindowBatch) -> QueryConstructionResult:
    index_by_id: dict[Any, int] = {}
    first_window_by_id: dict[Any, int] = {}
    appearances: dict[Any, int] = defaultdict(int)
    for window in batch.windows:
        for message in window.messages:
            index_by_id[message.turn.turn_id] = message.global_message_index
            first_window_by_id.setdefault(message.turn.turn_id, window.window_index)
            appearances[message.turn.turn_id] += 1
    ranks = {turn.turn_id: rank for rank, turn in enumerate(batch.pool.turns, start=1)}
    items: list[ConstructedMemoryItem] = []
    for turn in batch.pool.turns:
        raw = turn.original_content.encode("utf-8")
        if not raw:
            raise QueryConstructionError("E7-A cannot serialize an empty retrieved source turn")
        span = SourceSpan(turn_id=turn.turn_id, start_byte=0, end_byte=len(raw))
        items.append(
            ConstructedMemoryItem(
                turn=turn,
                global_message_index=index_by_id[turn.turn_id],
                retrieval_rank=ranks[turn.turn_id],
                compressed_content=turn.original_content,
                style=RetentionStyle.VERBATIM,
                support_spans=(span,),
                selected_window_index=first_window_by_id[turn.turn_id],
                constructor_receipt_sha256=sha256_json(
                    {
                        "cell": ConstructionCell.RAW_TOP50.value,
                        "turn_id": list(turn.turn_id.as_tuple()),
                        "source_turn": turn.source_turn.as_dict(),
                    }
                ),
                keep_votes=1,
                appearances=max(1, appearances[turn.turn_id]),
            )
        )
    ordered = tuple(sorted(items, key=lambda item: item.global_message_index))
    return _result(
        batch=batch,
        cell=ConstructionCell.RAW_TOP50,
        items=ordered,
        receipts=(),
        accounting={
            "constructor_identity_sha256": None,
            "constructor_windows": 0,
            "constructor_input_tokens": 0,
            "constructor_output_tokens": 0,
            "constructor_provider_response_bytes": 0,
            "constructor_usage_replayed_windows": 0,
            "constructor_latency_sum_ms": 0.0,
            "constructor_window_max_latency_ms": 0.0,
            "constructor_batch_wall_clock_ms": 0.0,
            "constructor_cost_usd": 0.0,
            "verbatim_keep_decisions": len(ordered),
            "extractive_keep_decisions": 0,
            "abstractive_keep_decisions": 0,
        },
    )


def _constructed_items(
    batch: QueryWindowBatch,
    by_window: dict[int, WindowConstructionReceipt],
) -> tuple[ConstructedMemoryItem, ...]:
    appearances: dict[Any, int] = defaultdict(int)
    keep_candidates: dict[
        Any,
        list[tuple[ConstructionDecision, WindowMessage, WindowConstructionReceipt, int]],
    ] = defaultdict(list)
    for window in batch.windows:
        receipt = by_window[window.window_index]
        for message, decision in zip(window.messages, receipt.decisions, strict=True):
            turn_id = message.turn.turn_id
            appearances[turn_id] += 1
            if decision.operation is DecisionOperation.KEEP:
                keep_candidates[turn_id].append((decision, message, receipt, window.window_index))

    items: list[ConstructedMemoryItem] = []
    for turn_id, candidates in keep_candidates.items():
        # LazyMem commit af41099 retains the longest compression for duplicate
        # message identities.  Freeze earliest window as the deterministic tie.
        selected = min(
            candidates,
            key=lambda item: (-len(item[0].compressed_content), item[3]),
        )
        decision, message, receipt, window_index = selected
        items.append(
            ConstructedMemoryItem(
                turn=message.turn,
                global_message_index=message.global_message_index,
                retrieval_rank=message.retrieval_rank,
                compressed_content=decision.compressed_content,
                style=decision.style,
                support_spans=decision.support_spans,
                selected_window_index=window_index,
                constructor_receipt_sha256=receipt.receipt_sha256,
                keep_votes=len(candidates),
                appearances=appearances[turn_id],
            )
        )
    return tuple(sorted(items, key=lambda item: item.global_message_index))


def _result(
    *,
    batch: QueryWindowBatch,
    cell: ConstructionCell,
    items: tuple[ConstructedMemoryItem, ...],
    receipts: tuple[WindowConstructionReceipt, ...],
    accounting: dict[str, Any],
) -> QueryConstructionResult:
    if batch.content_free_trace().get("implementation") != implementation_fingerprint():
        raise QueryConstructionError("E7 implementation drifted after window construction")
    reader_context = render_reader_context(items)
    item_bindings = [item.content_free_binding() for item in items]
    receipt_bindings = [receipt.content_free_binding() for receipt in receipts]
    unique_source_turns = {item.turn.turn_id for item in items}
    styles = {item.style for item in items}
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "cell": cell.value,
        "paper_transfer": {
            "source": PAPER_URL,
            "repository": PAPER_REPOSITORY,
            "repository_commit": PAPER_REPOSITORY_COMMIT,
            "window_protocol": PAPER_WINDOW_PROTOCOL,
            "paper_reproduction_claimed": False,
            "paper_duplicate_policy": "longest-compression-then-earliest-window",
        },
        "window_batch": {
            "trace_sha256": batch.trace_sha256,
            "source": batch.pool.content_free_binding(),
            "windows": len(batch.windows),
        },
        "constructor": {
            "input_fields": list(CONSTRUCTOR_INPUT_FIELDS),
            "receipt_authentication": RECEIPT_AUTHENTICATION,
            "receipts": receipt_bindings,
            "receipts_sha256": sha256_json(receipt_bindings),
        },
        "output": {
            "serializer": READER_CONTEXT_SERIALIZER,
            "items": item_bindings,
            "items_sha256": sha256_json(item_bindings),
            "item_count": len(items),
            "unique_source_turns": len(unique_source_turns),
            "reader_context_utf8_bytes": len(reader_context.encode("utf-8")),
            "reader_context_sha256": sha256_utf8(reader_context),
        },
        "accounting": {
            **accounting,
            "compressed_content_utf8_bytes": sum(
                len(item.compressed_content.encode("utf-8")) for item in items
            ),
            "source_support_spans": sum(len(item.support_spans) for item in items),
        },
        "claims": {
            "source_turns_immutable": True,
            "every_keep_has_source_spans": True,
            "all_output_byte_grounded": RetentionStyle.ABSTRACTIVE not in styles,
            "abstractive_faithfulness_proven": False,
            "external_constructor_identity_verified": False,
            "external_receipts_authenticated": False,
            "raw_provider_response_replay_required": cell is not ConstructionCell.RAW_TOP50,
            "raw_provider_response_replay_complete": True,
            "provider_usage_replay_complete": True,
            "latency_and_cost_verified_from_provider": False,
            "parallel_execution_or_batch_wall_clock_proven": cell is ConstructionCell.RAW_TOP50,
            "exact_reader_prompt_packed": False,
            "reader_or_judge_executed": False,
            "qa_improvement_proven": False,
            "serving_eligibility_proven": False,
        },
    }
    return QueryConstructionResult(
        cell=cell,
        items=items,
        reader_context=reader_context,
        receipts=receipts,
        _trace_canonical_json=canonical_json_bytes(payload).decode("utf-8"),
    )


def compile_query_construction(
    batch: QueryWindowBatch,
    *,
    cell: ConstructionCell,
    receipts: tuple[WindowConstructionReceipt, ...] = (),
) -> QueryConstructionResult:
    """Compile E7-A raw or E7-B/C externally constructed reader material."""

    if not isinstance(batch, QueryWindowBatch):
        raise QueryConstructionError("construction requires a QueryWindowBatch")
    if not isinstance(cell, ConstructionCell):
        raise QueryConstructionError("construction cell is not registered")
    if cell is ConstructionCell.RAW_TOP50:
        if receipts:
            raise QueryConstructionError("E7-A raw control cannot consume constructor receipts")
        return _raw_result(batch)
    by_window, accounting = _validate_receipts(batch, receipts, cell=cell)
    items = _constructed_items(batch, by_window)
    return _result(
        batch=batch,
        cell=cell,
        items=items,
        receipts=receipts,
        accounting=accounting,
    )


__all__ = [
    "EMPTY_MEMORY_CONTEXT",
    "READER_CONTEXT_SERIALIZER",
    "compile_query_construction",
]
