#!/usr/bin/env python3
"""Pinned real-model E7-A/E7-C query-time construction experiment.

The source is the immutable E1-B top-50 from the completed E1 protocol-v3
run.  DeepSeek V4 Flash makes one source-safe decision per message in each
frozen LazyMem-shaped window.  Only byte-verifiable E7-C output reaches the
reader; the abstractive E7-B cell remains ineligible.  E7-A and E7-C are packed
as whole constructed items under the same exact 8,192-token DeepSeek prompt
budget.  Nothing here changes production serving.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import sys
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from benchmarks.integrations.longmemeval_official_preflight import (
    ExactTokenizerPin,
    RunPreflightManifest,
    freeze_official_preflight,
)
from benchmarks.integrations.longmemeval_query_construction import (
    CONSTRUCTOR_SYSTEM_PROMPT_SHA256,
    CONSTRUCTOR_USER_PROMPT_SHA256,
    EMPTY_MEMORY_CONTEXT,
    ConstructionCell,
    ConstructorIdentity,
    QueryConstructionResult,
    QueryWindowBatch,
    build_query_windows,
    build_retrieved_turn_pool,
    compile_query_construction,
    constructor_request_bytes,
    render_reader_context,
    replay_window_construction_receipt,
)
from benchmarks.integrations.longmemeval_query_construction import (
    implementation_fingerprint as e7_implementation_fingerprint,
)
from scripts._longmemeval_common import OFFICIAL_ANSWER_TEMPLATE
from scripts.run_longmemeval_e1_external import (
    ARTIFACT_SCHEMA_VERSION,
    CROSS_ENCODER_BATCH_SIZE,
    DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION,
    DEEPSEEK_MODEL,
    DEEPSEEK_OUTPUT_USD_PER_MILLION,
    DEEPSEEK_REVISION,
    DEFAULT_CE_ROOT,
    DEFAULT_DATASET,
    DEFAULT_DEEPSEEK_ROOT,
    DEFAULT_QWEN_ROOT,
    DEFAULT_SAMPLE,
    DEFAULT_SEED,
    QWEN_BATCH_SIZE,
    TOKEN_BUDGET,
    ChatClient,
    ChatResult,
    DeepSeekExactTokenizer,
    ExperimentContext,
    ExternalE1Error,
    SelectedQuestion,
    _judge_text_for_arm,
    _load_receipts,
    _mean,
    _paired_bootstrap_interval,
    _receipt_bytes,
    _retrieval_metrics,
    _safe_question_id,
    _snapshot_artifact,
    _turn_id_payload,
    _wilson_interval,
    atomic_write,
    build_context,
    chat_receipt_record,
    is_abstention_question,
    judge_label,
    load_json,
    phase_path,
    replay_cross_encoder_row,
    replay_dense_row,
    seal_artifact,
    sha256_bytes,
    sha256_json,
    validate_chat_receipt_record,
    validate_sealed_artifact,
    verify_provider_prompt_tokens,
    write_json,
)
from scripts.run_longmemeval_qa import chat_result_from_raw_response

E7_RUN_PROTOCOL_VERSION = "swarmbrain-longmemeval-e7-real-model-development-v1"
E7_PACK_PROTOCOL_VERSION = "swarmbrain-e7-constructed-item-greedy-pack-v1"
E7_CELLS = ("E7-A", "E7-C")
DEFAULT_E1_OUTPUT = Path("/private/tmp/swarmbrain-longmemeval-e1-pilot-v3")
DEFAULT_E7_OUTPUT = Path("/private/tmp/swarmbrain-longmemeval-e7-pilot-v1")
CONSTRUCTOR_REVISION = "official-api-alias-observed-2026-08-09"
CONSTRUCTOR_DEPLOYMENT = "https://api.deepseek.com/v1/chat/completions"
CONSTRUCTOR_MAX_TOKENS = 4096
READER_MAX_TOKENS = 4096
JUDGE_MAX_TOKENS = 64
CHAT_MAX_ATTEMPTS = 4

IMPLEMENTATION_FILES = (
    "scripts/run_longmemeval_e1_external.py",
    "scripts/run_longmemeval_e7_external.py",
)


def _e7_upper_bound_cost(result: ChatResult) -> float:
    """Price retained usage plus worst-case usage for every unseen retry."""

    request = result.request
    successful_attempt = (
        result.prompt_tokens * DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION
        + result.completion_tokens * DEEPSEEK_OUTPUT_USD_PER_MILLION
    ) / 1_000_000.0
    unseen_attempt = (
        result.prompt_tokens * DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION
        + request.max_tokens * DEEPSEEK_OUTPUT_USD_PER_MILLION
    ) / 1_000_000.0
    return successful_attempt + (result.attempts - 1) * unseen_attempt


class ExternalE7Error(ExternalE1Error):
    """Saved E7 evidence cannot support deterministic replay."""


@dataclass(frozen=True, slots=True)
class E7Context:
    e1: ExperimentContext
    source_bytes: bytes
    preflight: RunPreflightManifest
    output_dir: Path
    manifest: dict[str, Any]


def _e1_namespace(args: argparse.Namespace) -> Namespace:
    return Namespace(
        dataset=args.dataset,
        output_dir=args.e1_output_dir,
        sample=args.sample,
        seed=args.seed,
        qwen_root=args.qwen_root,
        cross_encoder_root=args.cross_encoder_root,
        deepseek_root=args.deepseek_root,
        device=args.device,
        qwen_batch_size=args.qwen_batch_size,
        cross_encoder_batch_size=CROSS_ENCODER_BATCH_SIZE,
    )


def _runner_implementation_fingerprint() -> dict[str, Any]:
    files = {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in IMPLEMENTATION_FILES
    }
    return {
        "runner_files": files,
        "runner_tree_sha256": sha256_json(files),
        "e7_contract": e7_implementation_fingerprint(),
    }


def _tokenizer_executable_sha256(e1: ExperimentContext) -> str:
    encoding = e1.deepseek_root / "encoding" / "encoding_dsv4.py"
    files = {
        "encoding_dsv4.py": hashlib.sha256(encoding.read_bytes()).hexdigest(),
        **{
            relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
            for relative in IMPLEMENTATION_FILES
        },
    }
    return sha256_json(files)


def _constructor_model_artifact_sha256() -> str:
    return sha256_json(
        {
            "model_alias": DEEPSEEK_MODEL,
            "deployment": CONSTRUCTOR_DEPLOYMENT,
            "served_weights_immutable": False,
            "classification": "caller-attested-api-alias-not-weight-snapshot",
        }
    )


def _constructor_identity(context: E7Context) -> ConstructorIdentity:
    return ConstructorIdentity(
        producer="scripts.run_longmemeval_e7_external.DeepSeekConstructor",
        model=DEEPSEEK_MODEL,
        revision=CONSTRUCTOR_REVISION,
        deployment=CONSTRUCTOR_DEPLOYMENT,
        model_artifact_sha256=_constructor_model_artifact_sha256(),
        system_prompt_sha256=CONSTRUCTOR_SYSTEM_PROMPT_SHA256,
        user_prompt_sha256=CONSTRUCTOR_USER_PROMPT_SHA256,
        tokenizer_model=DEEPSEEK_MODEL,
        tokenizer_revision=DEEPSEEK_REVISION,
        tokenizer_artifact_sha256=_snapshot_artifact(
            context.e1,
            "deepseek_tokenizer",
        ),
    )


def build_e7_context(args: argparse.Namespace) -> E7Context:
    e1 = build_context(_e1_namespace(args))
    source_bytes = Path(args.dataset).resolve().read_bytes()
    tokenizer_pin = ExactTokenizerPin(
        model=DEEPSEEK_MODEL,
        revision=DEEPSEEK_REVISION,
        artifact_sha256=_snapshot_artifact(e1, "deepseek_tokenizer"),
        executable_sha256=_tokenizer_executable_sha256(e1),
    )
    preflight = freeze_official_preflight(source_bytes, tokenizer=tokenizer_pin)
    sources: list[dict[str, Any]] = []
    for question in e1.selected:
        dense = load_json(phase_path(e1, "dense", question), sealed=True)
        e1a = replay_dense_row(e1, question, dense)
        cross_encoder = load_json(
            phase_path(e1, "cross_encoder", question),
            sealed=True,
        )
        _, _, e1b = replay_cross_encoder_row(e1, question, cross_encoder)
        sources.append(
            {
                "question_id": question.question_id,
                "dense_artifact_sha256": dense["artifact_sha256"],
                "cross_encoder_artifact_sha256": cross_encoder["artifact_sha256"],
                "e1a_trace_sha256": e1a.trace_sha256,
                "e1b_trace_sha256": e1b.trace_sha256,
            }
        )
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e7-real-model-run-manifest",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E7_RUN_PROTOCOL_VERSION,
        "classification": "development-experiment-not-official-longmemeval-score",
        "production_configuration": False,
        "source_e1_manifest_sha256": e1.manifest["artifact_sha256"],
        "source_e1_artifacts": sources,
        "source_e1_artifacts_sha256": sha256_json(sources),
        "official_preflight_manifest_sha256": preflight.manifest_sha256,
        "dataset": e1.manifest["dataset"],
        "turn_projection": e1.manifest["turn_projection"],
        "sample": e1.manifest["sample"],
        "cells": list(E7_CELLS),
        "construction": {
            "source": "frozen-E1-B-top-50",
            "cell": "E7-C-byte-grounded",
            "E7-B_executed": False,
            "model": DEEPSEEK_MODEL,
            "revision": CONSTRUCTOR_REVISION,
            "deployment": CONSTRUCTOR_DEPLOYMENT,
            "thinking": "disabled",
            "temperature": 0.0,
            "max_tokens": CONSTRUCTOR_MAX_TOKENS,
            "maximum_attempts": CHAT_MAX_ATTEMPTS,
            "model_artifact_sha256": _constructor_model_artifact_sha256(),
            "identity_authentication": "caller-attested-unverified",
        },
        "packing": {
            "protocol": E7_PACK_PROTOCOL_VERSION,
            "complete_reader_prompt_token_budget": TOKEN_BUDGET,
            "order": "chronological-constructed-items",
            "policy": "greedy-whole-item-skip-over-budget-and-continue",
            "reader_context_serializer": (
                "[parent_session_date] role: compressed_content joined by newline"
            ),
        },
        "reader_and_development_judge": {
            **e1.manifest["reader_and_development_judge"],
            "deployment": CONSTRUCTOR_DEPLOYMENT,
            "reader_max_tokens": READER_MAX_TOKENS,
            "development_judge_max_tokens": JUDGE_MAX_TOKENS,
            "maximum_attempts": CHAT_MAX_ATTEMPTS,
            "required_response_model": DEEPSEEK_MODEL,
            "provider_request_id_required": True,
        },
        "cost_accounting": {
            "cache_miss_input_usd_per_million": (DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION),
            "output_usd_per_million": DEEPSEEK_OUTPUT_USD_PER_MILLION,
            "retained_successful_attempt": "provider-reported-usage",
            "each_unseen_prior_attempt": ("provider-prompt-tokens-plus-request-max-tokens"),
            "maximum_attempts": CHAT_MAX_ATTEMPTS,
            "billed_cost_claimed": False,
        },
        "model_snapshots": {
            "deepseek_tokenizer": e1.manifest["model_snapshots"]["deepseek_tokenizer"]
        },
        "implementation": _runner_implementation_fingerprint(),
        "claims": {
            "gold_fields_used_for_construction": False,
            "E7-B_abstractive_faithfulness_proven": False,
            "official_longmemeval_score": False,
            "production_policy_changed": False,
        },
    }
    manifest = seal_artifact(payload)
    output_dir = Path(args.output_dir).resolve()
    path = output_dir / "manifest.json"
    if path.exists():
        if load_json(path, sealed=True) != manifest:
            raise ExternalE7Error("E7 output directory belongs to a different manifest")
    else:
        write_json(path, manifest)
    return E7Context(
        e1=e1,
        source_bytes=source_bytes,
        preflight=preflight,
        output_dir=output_dir,
        manifest=manifest,
    )


def e7_phase_path(context: E7Context, phase: str, question: SelectedQuestion) -> Path:
    name = f"{question.position:03d}-{_safe_question_id(question.question_id)}.json"
    return context.output_dir / phase / name


def e7_constructor_receipt_path(
    context: E7Context,
    question: SelectedQuestion,
) -> Path:
    name = f"{question.position:03d}-{_safe_question_id(question.question_id)}.jsonl"
    return context.output_dir / "constructor-receipts" / name


def e7_qa_receipt_path(context: E7Context, question: SelectedQuestion) -> Path:
    name = f"{question.position:03d}-{_safe_question_id(question.question_id)}.jsonl"
    return context.output_dir / "qa-receipts" / name


def _selected_prefix(context: E7Context, limit: int | None) -> tuple[SelectedQuestion, ...]:
    if limit is None:
        return context.e1.selected
    if isinstance(limit, bool) or limit < 1:
        raise ExternalE7Error("execution limit must be a positive integer")
    return context.e1.selected[:limit]


def _e1_material(context: E7Context, question: SelectedQuestion):
    dense = load_json(phase_path(context.e1, "dense", question), sealed=True)
    e1a = replay_dense_row(context.e1, question, dense)
    cross_encoder = load_json(
        phase_path(context.e1, "cross_encoder", question),
        sealed=True,
    )
    _, observation, e1b = replay_cross_encoder_row(
        context.e1,
        question,
        cross_encoder,
    )
    return dense, e1a, cross_encoder, observation, e1b


def _build_batch(
    context: E7Context,
    question: SelectedQuestion,
) -> tuple[QueryWindowBatch, dict[str, Any]]:
    dense, e1a, cross_encoder, observation, e1b = _e1_material(context, question)
    pool = build_retrieved_turn_pool(
        context.e1.corpus,
        e1b,
        source=e1a,
        cross_encoder=observation,
        query=question.question,
    )
    batch = build_query_windows(
        context.e1.corpus,
        pool,
        context.preflight,
        source_bytes=context.source_bytes,
        selection=e1b,
        e1_source=e1a,
        cross_encoder=observation,
        query=question.question,
        current_date=question.current_date,
    )
    sources = {
        "dense_artifact_sha256": dense["artifact_sha256"],
        "cross_encoder_artifact_sha256": cross_encoder["artifact_sha256"],
        "e1a_trace_sha256": e1a.trace_sha256,
        "e1b_trace_sha256": e1b.trace_sha256,
    }
    return batch, sources


def _compute_windows_row(
    context: E7Context,
    question: SelectedQuestion,
) -> tuple[dict[str, Any], QueryWindowBatch]:
    started = perf_counter()
    batch, sources = _build_batch(context, question)
    requests = [
        {
            "window_index": window.window_index,
            "window_sha256": window.window_sha256,
            "request_sha256": sha256_bytes(constructor_request_bytes(batch, window)),
            "request_utf8_bytes": len(constructor_request_bytes(batch, window)),
            "messages": len(window.messages),
        }
        for window in batch.windows
    ]
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e7-window-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E7_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "sources": sources,
        "preflight_manifest_sha256": context.preflight.manifest_sha256,
        "pool": batch.pool.content_free_binding(),
        "window_trace": batch.content_free_trace(),
        "constructor_requests": requests,
        "constructor_requests_sha256": sha256_json(requests),
        "wall_ms": (perf_counter() - started) * 1000.0,
        "claims": {
            "gold_fields_consumed": False,
            "source_bytes_refrozen": True,
            "E1-B_replayed": True,
            "constructor_executed": False,
            "reader_or_judge_executed": False,
            "production_policy_changed": False,
        },
    }
    return seal_artifact(payload), batch


def replay_windows_row(
    context: E7Context,
    question: SelectedQuestion,
    row: Any,
) -> QueryWindowBatch:
    row = validate_sealed_artifact(row)
    expected, batch = _compute_windows_row(context, question)
    if (
        isinstance(row.get("wall_ms"), bool)
        or not isinstance(
            row.get("wall_ms"),
            (int, float),
        )
        or not math.isfinite(float(row["wall_ms"]))
        or row["wall_ms"] < 0
    ):
        raise ExternalE7Error("saved E7 window wall time is invalid")
    expected = seal_artifact(
        {
            **{
                key: value
                for key, value in expected.items()
                if key not in {"artifact_sha256", "wall_ms"}
            },
            "wall_ms": row["wall_ms"],
        }
    )
    if row != expected:
        raise ExternalE7Error("saved E7 window artifact differs from exact replay")
    return batch


def run_windows_phase(context: E7Context, *, limit: int | None = None) -> None:
    selected = _selected_prefix(context, limit)
    for completed, question in enumerate(selected, start=1):
        path = e7_phase_path(context, "windows", question)
        if path.exists():
            row = load_json(path, sealed=True)
            batch = replay_windows_row(context, question, row)
            action = "verified"
        else:
            row, batch = _compute_windows_row(context, question)
            write_json(path, row)
            action = "built"
        print(
            f"  windows: {completed}/{len(selected)} {action} {question.question_id} "
            f"windows={len(batch.windows)} appearances="
            f"{sum(len(window.messages) for window in batch.windows)}",
            file=sys.stderr,
            flush=True,
        )


CONSTRUCTOR_CHAT_RECEIPT_TYPE = "swarmbrain-e7-constructor-chat-receipt"
CONSTRUCTOR_CHAT_RECEIPT_VERSION = "swarmbrain-e7-constructor-chat-receipt-v1"


def _constructor_receipt_id(question_id: str, window_index: int) -> str:
    return f"{question_id}:e7-c:window-{window_index:04d}"


def _constructor_chat_record(
    question: SelectedQuestion,
    window_index: int,
    result: ChatResult,
) -> dict[str, Any]:
    return {
        "artifact_type": CONSTRUCTOR_CHAT_RECEIPT_TYPE,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": CONSTRUCTOR_CHAT_RECEIPT_VERSION,
        "question_id": question.question_id,
        "question_position": question.position,
        "window_index": window_index,
        "local_request_id": _constructor_receipt_id(
            question.question_id,
            window_index,
        ),
        "prompt": {
            "sha256": result.prompt_sha256,
            "utf8_bytes": result.prompt_utf8_bytes,
            "raw_base64": base64.b64encode(result.prompt_bytes).decode("ascii"),
        },
        "provider_request": {
            "raw_bytes": len(result.raw_request),
            "raw_sha256": result.raw_request_sha256,
            "raw_base64": base64.b64encode(result.raw_request).decode("ascii"),
        },
        "provider_response": {
            "raw_bytes": len(result.raw_response),
            "raw_sha256": result.raw_response_sha256,
            "raw_base64": base64.b64encode(result.raw_response).decode("ascii"),
        },
        "transport": {
            "endpoint_url": result.endpoint_url,
            "attempts": result.attempts,
            "latency_ms": result.latency_ms,
        },
    }


def _base64_bytes(value: Any, *, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ExternalE7Error(f"{label} must be non-empty base64 text")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ExternalE7Error(f"{label} is invalid base64") from exc


def _validate_constructor_chat_record(
    question: SelectedQuestion,
    window_index: int,
    record: Any,
) -> ChatResult:
    if not isinstance(record, dict) or set(record) != {
        "artifact_type",
        "schema_version",
        "protocol_version",
        "question_id",
        "question_position",
        "window_index",
        "local_request_id",
        "prompt",
        "provider_request",
        "provider_response",
        "transport",
    }:
        raise ExternalE7Error("constructor chat receipt fields differ from exact schema")
    expected = {
        "artifact_type": CONSTRUCTOR_CHAT_RECEIPT_TYPE,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": CONSTRUCTOR_CHAT_RECEIPT_VERSION,
        "question_id": question.question_id,
        "question_position": question.position,
        "window_index": window_index,
        "local_request_id": _constructor_receipt_id(
            question.question_id,
            window_index,
        ),
    }
    for key, value in expected.items():
        if type(record.get(key)) is not type(value) or record.get(key) != value:
            raise ExternalE7Error(f"constructor chat receipt {key} drifted")
    prompt = record.get("prompt")
    request = record.get("provider_request")
    response = record.get("provider_response")
    transport = record.get("transport")
    if not all(isinstance(value, dict) for value in (prompt, request, response, transport)):
        raise ExternalE7Error("constructor chat receipt nested evidence is invalid")
    if set(prompt) != {"sha256", "utf8_bytes", "raw_base64"}:
        raise ExternalE7Error("constructor prompt receipt fields differ from exact schema")
    if set(request) != {"raw_bytes", "raw_sha256", "raw_base64"}:
        raise ExternalE7Error("constructor request receipt fields differ from exact schema")
    if set(response) != {"raw_bytes", "raw_sha256", "raw_base64"}:
        raise ExternalE7Error("constructor response receipt fields differ from exact schema")
    if set(transport) != {"endpoint_url", "attempts", "latency_ms"}:
        raise ExternalE7Error("constructor transport receipt fields differ from exact schema")
    prompt_bytes = _base64_bytes(prompt.get("raw_base64"), label="constructor prompt")
    raw_request = _base64_bytes(
        request.get("raw_base64"),
        label="constructor provider request",
    )
    raw_response = _base64_bytes(
        response.get("raw_base64"),
        label="constructor provider response",
    )
    for value, raw, bytes_key, digest_key, label in (
        (prompt, prompt_bytes, "utf8_bytes", "sha256", "prompt"),
        (request, raw_request, "raw_bytes", "raw_sha256", "provider request"),
        (response, raw_response, "raw_bytes", "raw_sha256", "provider response"),
    ):
        byte_count = value.get(bytes_key)
        digest = value.get(digest_key)
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or not isinstance(digest, str)
            or byte_count != len(raw)
            or digest != sha256_bytes(raw)
        ):
            raise ExternalE7Error(f"constructor {label} bytes or digest drifted")
    try:
        prompt_text = prompt_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExternalE7Error("constructor prompt is not UTF-8") from exc
    endpoint = transport.get("endpoint_url")
    attempts = transport.get("attempts")
    latency = transport.get("latency_ms")
    if not isinstance(endpoint, str):
        raise ExternalE7Error("constructor endpoint is invalid")
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or not 1 <= attempts <= CHAT_MAX_ATTEMPTS
    ):
        raise ExternalE7Error("constructor attempts are invalid")
    if type(latency) is not float or not math.isfinite(latency) or latency < 0:
        raise ExternalE7Error("constructor latency is invalid")
    result = chat_result_from_raw_response(
        raw_response,
        prompt=prompt_text,
        attempts=attempts,
        latency_ms=float(latency),
        raw_request=raw_request,
        endpoint_url=endpoint,
    )
    if record != _constructor_chat_record(question, window_index, result):
        raise ExternalE7Error("constructor receipt differs from exact normalized replay")
    return result


def _reject_constructor_constant(value: str) -> None:
    raise ExternalE7Error(f"non-finite constructor JSON constant {value!r} is forbidden")


def _reject_constructor_duplicate_fields(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExternalE7Error(f"duplicate constructor JSON field {key!r} is forbidden")
        result[key] = value
    return result


def _load_constructor_chat_records(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise ExternalE7Error(f"cannot read constructor receipt artifact {path}") from exc
    if not lines:
        raise ExternalE7Error("constructor receipt artifact cannot be empty")
    records: list[dict[str, Any]] = []
    for raw in lines:
        try:
            value = json.loads(
                raw,
                parse_constant=_reject_constructor_constant,
                object_pairs_hook=_reject_constructor_duplicate_fields,
            )
            canonical = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (
            ExternalE7Error,
            TypeError,
            ValueError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ExternalE7Error("constructor receipt JSONL is malformed") from exc
        if not isinstance(value, dict) or canonical != raw:
            raise ExternalE7Error("constructor receipt JSONL is not canonical")
        records.append(value)
    return tuple(records)


def _constructor_chat_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for record in records
    )


def _window_receipt(
    context: E7Context,
    question: SelectedQuestion,
    batch: QueryWindowBatch,
    window_index: int,
    result: ChatResult,
    *,
    tokenizer: DeepSeekExactTokenizer,
):
    window = batch.windows[window_index]
    prompt_bytes = constructor_request_bytes(batch, window)
    prompt = prompt_bytes.decode("utf-8")
    if result.prompt_bytes != prompt_bytes:
        raise ExternalE7Error("constructor chat prompt differs from frozen E7 request")
    request = result.request
    if (
        request.model != DEEPSEEK_MODEL
        or request.temperature != 0.0
        or request.max_tokens != CONSTRUCTOR_MAX_TOKENS
        or request.thinking_mode != "disabled"
    ):
        raise ExternalE7Error("constructor chat request configuration drifted")
    if result.endpoint_url != CONSTRUCTOR_DEPLOYMENT:
        raise ExternalE7Error("constructor endpoint drifted")
    expected_tokens = tokenizer.exact_count(prompt)
    verify_provider_prompt_tokens(
        result,
        expected=expected_tokens,
        label=f"{question.question_id}/window-{window_index}/constructor",
    )
    receipt = replay_window_construction_receipt(
        batch,
        window,
        raw_response=result.raw_response,
        request_id=_constructor_receipt_id(question.question_id, window_index),
        identity=_constructor_identity(context),
        latency_ms=result.latency_ms,
        cost_usd=_e7_upper_bound_cost(result),
    )
    return receipt, expected_tokens


def _construction_artifacts(
    context: E7Context,
    question: SelectedQuestion,
    *,
    batch: QueryWindowBatch,
    window_row: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    tokenizer: DeepSeekExactTokenizer,
    wall_ms: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, QueryConstructionResult]]:
    if len(records) != len(batch.windows):
        raise ExternalE7Error("constructor receipts do not cover every frozen window")
    results_by_window: list[ChatResult] = []
    window_receipts = []
    token_reconciliation = []
    for window_index, record in enumerate(records):
        result = _validate_constructor_chat_record(question, window_index, record)
        receipt, exact_tokens = _window_receipt(
            context,
            question,
            batch,
            window_index,
            result,
            tokenizer=tokenizer,
        )
        results_by_window.append(result)
        window_receipts.append(receipt)
        token_reconciliation.append(
            {
                "window_index": window_index,
                "window_sha256": batch.windows[window_index].window_sha256,
                "exact_local_prompt_tokens": exact_tokens,
                "api_prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "attempts": result.attempts,
                "provider_request_id_sha256": sha256_bytes(str(result.request_id).encode("utf-8")),
                "raw_request_sha256": result.raw_request_sha256,
                "raw_response_sha256": result.raw_response_sha256,
                "normalized_receipt_sha256": receipt.receipt_sha256,
                "estimated_cost_upper_bound_usd": _e7_upper_bound_cost(result),
            }
        )
    raw = compile_query_construction(batch, cell=ConstructionCell.RAW_TOP50)
    grounded = compile_query_construction(
        batch,
        cell=ConstructionCell.GROUNDED_QUERY_CONSTRUCTION,
        receipts=tuple(window_receipts),
    )
    cells = {"E7-A": raw, "E7-C": grounded}
    contexts_payload = {
        "artifact_type": "swarmbrain-longmemeval-e7-reader-contexts-sensitive",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E7_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "classification": "contains-public-benchmark-memory-text",
        "cells": {
            cell: {
                "reader_context": result.reader_context,
                "reader_context_sha256": sha256_bytes(result.reader_context.encode("utf-8")),
                "reader_context_utf8_bytes": len(result.reader_context.encode("utf-8")),
                "construction_trace_sha256": result.trace_sha256,
            }
            for cell, result in cells.items()
        },
    }
    contexts = seal_artifact(contexts_payload)
    raw_receipts = _constructor_chat_bytes(records)
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e7-construction-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E7_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_window_artifact_sha256": window_row["artifact_sha256"],
        "window_trace_sha256": batch.trace_sha256,
        "constructor_identity": _constructor_identity(context).content_free_binding(),
        "constructor_chat_receipts_sha256": sha256_bytes(raw_receipts),
        "constructor_chat_receipts_bytes": len(raw_receipts),
        "constructor_chat_receipt_count": len(records),
        "token_reconciliation": token_reconciliation,
        "token_reconciliation_sha256": sha256_json(token_reconciliation),
        "cells": {cell: result.content_free_trace() for cell, result in cells.items()},
        "contexts_artifact_sha256": contexts["artifact_sha256"],
        "estimated_constructor_cost_upper_bound_usd": sum(
            _e7_upper_bound_cost(result) for result in results_by_window
        ),
        "wall_ms": wall_ms,
        "claims": {
            "gold_fields_consumed": False,
            "all_E7-C_output_byte_grounded": True,
            "E7-B_executed": False,
            "api_and_local_prompt_tokens_reconciled": True,
            "constructor_identity_authenticated": False,
            "reader_or_judge_executed": False,
            "production_policy_changed": False,
        },
    }
    return seal_artifact(payload), contexts, cells


def replay_construction_row(
    context: E7Context,
    question: SelectedQuestion,
    *,
    construction_row: Any,
    contexts_row: Any,
    records: Sequence[dict[str, Any]],
    tokenizer: DeepSeekExactTokenizer,
) -> dict[str, QueryConstructionResult]:
    construction_row = validate_sealed_artifact(construction_row)
    contexts_row = validate_sealed_artifact(contexts_row)
    window_path = e7_phase_path(context, "windows", question)
    window_row = load_json(window_path, sealed=True)
    batch = replay_windows_row(context, question, window_row)
    wall_ms = construction_row.get("wall_ms")
    if (
        isinstance(wall_ms, bool)
        or not isinstance(wall_ms, (int, float))
        or not math.isfinite(float(wall_ms))
        or wall_ms < 0
    ):
        raise ExternalE7Error("E7 construction wall time is invalid")
    expected_construction, expected_contexts, cells = _construction_artifacts(
        context,
        question,
        batch=batch,
        window_row=window_row,
        records=records,
        tokenizer=tokenizer,
        wall_ms=float(wall_ms),
    )
    if construction_row != expected_construction:
        raise ExternalE7Error("saved E7 construction artifact differs from exact replay")
    if contexts_row != expected_contexts:
        raise ExternalE7Error("saved E7 reader contexts differ from exact replay")
    return cells


async def _run_constructor_calls(
    context: E7Context,
    question: SelectedQuestion,
    *,
    batch: QueryWindowBatch,
    tokenizer: DeepSeekExactTokenizer,
    base_url: str,
    api_key: str,
    records: list[dict[str, Any]],
    receipt_path: Path,
) -> None:
    client = ChatClient(
        base_url=base_url,
        model=DEEPSEEK_MODEL,
        api_key=api_key,
        temperature=0.0,
        max_tokens=CONSTRUCTOR_MAX_TOKENS,
        required_response_model=DEEPSEEK_MODEL,
        require_request_id=True,
        thinking_mode="disabled",
        attempts=CHAT_MAX_ATTEMPTS,
    )
    try:
        for window_index in range(len(records), len(batch.windows)):
            prompt = constructor_request_bytes(batch, batch.windows[window_index]).decode("utf-8")
            result = await client.complete(prompt)
            record = _constructor_chat_record(question, window_index, result)
            records.append(record)
            atomic_write(receipt_path, _constructor_chat_bytes(records))
            # Validate only after preserving the exact provider bytes. A schema
            # or grounding failure remains evidence instead of being discarded.
            receipt, exact_tokens = _window_receipt(
                context,
                question,
                batch,
                window_index,
                result,
                tokenizer=tokenizer,
            )
            styles = sorted(
                {
                    decision.style.value
                    for decision in receipt.decisions
                    if decision.operation.value == "KEEP"
                }
            )
            print(
                f"    constructor: {window_index + 1}/{len(batch.windows)} "
                f"tokens={exact_tokens}+{result.completion_tokens} "
                f"keeps={sum(d.operation.value == 'KEEP' for d in receipt.decisions)} "
                f"styles={styles or ['none']} cost<=${_e7_upper_bound_cost(result):.5f}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        await client.aclose()


def run_construct_phase(
    context: E7Context,
    *,
    base_url: str,
    api_key_env: str,
    limit: int | None = None,
) -> None:
    if base_url.strip().rstrip("/") not in {
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
    }:
        raise ExternalE7Error("E7 constructor is frozen to the official DeepSeek endpoint")
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise ExternalE7Error(f"environment variable {api_key_env!r} is missing")
    tokenizer = DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    selected = _selected_prefix(context, limit)
    for completed, question in enumerate(selected, start=1):
        construction_path = e7_phase_path(context, "construction", question)
        contexts_path = e7_phase_path(context, "contexts", question)
        receipt_path = e7_constructor_receipt_path(context, question)
        existing = (
            construction_path.exists(),
            contexts_path.exists(),
            receipt_path.exists(),
        )
        if existing[0] != existing[1]:
            raise ExternalE7Error("E7 construction has only one normalized artifact")
        if existing[0] and not existing[2]:
            raise ExternalE7Error("E7 construction is missing raw constructor receipts")
        if existing[0]:
            records = _load_constructor_chat_records(receipt_path)
            cells = replay_construction_row(
                context,
                question,
                construction_row=load_json(construction_path, sealed=True),
                contexts_row=load_json(contexts_path, sealed=True),
                records=records,
                tokenizer=tokenizer,
            )
            print(
                f"  construct: {completed}/{len(selected)} verified {question.question_id} "
                f"windows={len(records)} kept={len(cells['E7-C'].items)}",
                file=sys.stderr,
                flush=True,
            )
            continue
        window_path = e7_phase_path(context, "windows", question)
        if not window_path.exists():
            raise ExternalE7Error("E7 construction requires a window artifact first")
        window_row = load_json(window_path, sealed=True)
        batch = replay_windows_row(context, question, window_row)
        records = (
            list(_load_constructor_chat_records(receipt_path)) if receipt_path.exists() else []
        )
        if len(records) > len(batch.windows):
            raise ExternalE7Error("constructor receipt sidecar exceeds frozen window count")
        for window_index, record in enumerate(records):
            result = _validate_constructor_chat_record(question, window_index, record)
            _window_receipt(
                context,
                question,
                batch,
                window_index,
                result,
                tokenizer=tokenizer,
            )
        started = perf_counter()
        asyncio.run(
            _run_constructor_calls(
                context,
                question,
                batch=batch,
                tokenizer=tokenizer,
                base_url=base_url,
                api_key=api_key,
                records=records,
                receipt_path=receipt_path,
            )
        )
        construction, contexts, cells = _construction_artifacts(
            context,
            question,
            batch=batch,
            window_row=window_row,
            records=records,
            tokenizer=tokenizer,
            wall_ms=(perf_counter() - started) * 1000.0,
        )
        # The content-bearing context is written first; a construction artifact
        # can never point at a sidecar that was not durably preserved.
        write_json(contexts_path, contexts)
        write_json(construction_path, construction)
        print(
            f"  construct: {completed}/{len(selected)} ran {question.question_id} "
            f"windows={len(records)} kept={len(cells['E7-C'].items)} "
            f"cost<=${construction['estimated_constructor_cost_upper_bound_usd']:.5f}",
            file=sys.stderr,
            flush=True,
        )


def _reader_prompt(question: SelectedQuestion, items: Sequence[Any]) -> str:
    frozen = tuple(items)
    history = render_reader_context(frozen) if frozen else EMPTY_MEMORY_CONTEXT
    return OFFICIAL_ANSWER_TEMPLATE.format(
        history,
        question.current_date,
        question.question,
    )


def _pack_cell(
    question: SelectedQuestion,
    *,
    cell: str,
    result: QueryConstructionResult,
    tokenizer: DeepSeekExactTokenizer,
) -> tuple[dict[str, Any], str]:
    tokenizer.reset_receipts()
    query_sha256 = sha256_bytes(question.question.encode("utf-8"))
    observations: list[dict[str, Any]] = []

    def observe(prompt: str, *, purpose: str, item: Any | None) -> int:
        receipt = tokenizer.count_prompt(prompt, query_sha256=query_sha256)
        observations.append(
            {
                "sequence": len(observations) + 1,
                "purpose": purpose,
                "turn_id": None if item is None else _turn_id_payload(item.turn.turn_id),
                "receipt": receipt.content_free_binding(),
            }
        )
        return receipt.token_count

    selected: list[Any] = []
    initial_prompt = _reader_prompt(question, selected)
    initial_tokens = observe(
        initial_prompt,
        purpose="initial-empty-context",
        item=None,
    )
    if initial_tokens > TOKEN_BUDGET:
        raise ExternalE7Error("E7 empty reader prompt already exceeds the token budget")
    decisions: list[dict[str, Any]] = []
    for position, item in enumerate(result.items, start=1):
        proposal = [*selected, item]
        proposal_prompt = _reader_prompt(question, proposal)
        proposal_tokens = observe(
            proposal_prompt,
            purpose="candidate-proposal",
            item=item,
        )
        accepted = proposal_tokens <= TOKEN_BUDGET
        oversized_alone = False
        singleton_tokens = None
        if accepted:
            selected.append(item)
        else:
            singleton_tokens = observe(
                _reader_prompt(question, [item]),
                purpose="candidate-alone",
                item=item,
            )
            oversized_alone = singleton_tokens > TOKEN_BUDGET
        decisions.append(
            {
                "candidate_position": position,
                "turn_id": _turn_id_payload(item.turn.turn_id),
                "compressed_content_sha256": sha256_bytes(item.compressed_content.encode("utf-8")),
                "proposal_tokens": proposal_tokens,
                "accepted": accepted,
                "singleton_tokens_if_rejected": singleton_tokens,
                "oversized_alone": oversized_alone,
            }
        )
    prompt = _reader_prompt(question, selected)
    final_tokens = observe(prompt, purpose="final-independent-recount", item=None)
    if final_tokens > TOKEN_BUDGET:
        raise ExternalE7Error("E7 final reader prompt exceeds the exact token budget")
    candidate_order = [item.content_free_binding() for item in result.items]
    kept = [item.content_free_binding() for item in selected]
    trace = {
        "artifact_type": "swarmbrain-longmemeval-e7-constructed-item-pack",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E7_PACK_PROTOCOL_VERSION,
        "cell": cell,
        "question_id": question.question_id,
        "question_sha256": query_sha256,
        "current_date_sha256": sha256_bytes(question.current_date.encode("utf-8")),
        "source_construction_trace_sha256": result.trace_sha256,
        "tokenizer": tokenizer.identity.as_dict(),
        "budget": TOKEN_BUDGET,
        "policy": "chronological-greedy-whole-item-skip-over-budget-and-continue",
        "candidate_order": candidate_order,
        "candidate_order_sha256": sha256_json(candidate_order),
        "decisions": decisions,
        "decisions_sha256": sha256_json(decisions),
        "kept_items": kept,
        "kept_items_sha256": sha256_json(kept),
        "kept_turn_ids": [_turn_id_payload(item.turn.turn_id) for item in selected],
        "dropped_turn_ids": [
            decision["turn_id"] for decision in decisions if not decision["accepted"]
        ],
        "exact_count_observations": observations,
        "exact_count_observations_sha256": sha256_json(observations),
        "final_prompt": {
            "sha256": sha256_bytes(prompt.encode("utf-8")),
            "utf8_bytes": len(prompt.encode("utf-8")),
            "tokens": final_tokens,
            "within_budget": True,
        },
        "claims": {
            "complete_reader_prompt_counted": True,
            "whole_constructed_items_only": True,
            "gold_fields_consumed": False,
            "reader_or_judge_executed": False,
            "production_policy_changed": False,
        },
    }
    return {**trace, "trace_sha256": sha256_json(trace)}, prompt


def _compute_packs(
    context: E7Context,
    question: SelectedQuestion,
    *,
    tokenizer: DeepSeekExactTokenizer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    construction_path = e7_phase_path(context, "construction", question)
    contexts_path = e7_phase_path(context, "contexts", question)
    receipt_path = e7_constructor_receipt_path(context, question)
    if not all(path.exists() for path in (construction_path, contexts_path, receipt_path)):
        raise ExternalE7Error("E7 packing requires complete construction evidence")
    construction = load_json(construction_path, sealed=True)
    contexts = load_json(contexts_path, sealed=True)
    records = _load_constructor_chat_records(receipt_path)
    cells = replay_construction_row(
        context,
        question,
        construction_row=construction,
        contexts_row=contexts,
        records=records,
        tokenizer=tokenizer,
    )
    packed: dict[str, Any] = {}
    prompts: dict[str, Any] = {}
    for cell in E7_CELLS:
        trace, prompt = _pack_cell(
            question,
            cell=cell,
            result=cells[cell],
            tokenizer=tokenizer,
        )
        packed[cell] = trace
        prompts[cell] = {
            "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "prompt_utf8_bytes": len(prompt.encode("utf-8")),
            "prompt_tokens": trace["final_prompt"]["tokens"],
            "packing_trace_sha256": trace["trace_sha256"],
        }
    prompt_payload = {
        "artifact_type": "swarmbrain-longmemeval-e7-packed-prompts-sensitive",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E7_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "classification": "contains-public-benchmark-question-and-memory-text",
        "cells": prompts,
    }
    prompt_artifact = seal_artifact(prompt_payload)
    pack_payload = {
        "artifact_type": "swarmbrain-longmemeval-e7-pack-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E7_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_construction_artifact_sha256": construction["artifact_sha256"],
        "source_contexts_artifact_sha256": contexts["artifact_sha256"],
        "prompt_artifact_sha256": prompt_artifact["artifact_sha256"],
        "tokenizer": tokenizer.identity.as_dict(),
        "cells": packed,
        "claims": {
            "complete_reader_prompt_counted": True,
            "whole_constructed_items_only": True,
            "gold_fields_consumed": False,
            "reader_or_judge_executed": False,
            "production_policy_changed": False,
        },
    }
    return seal_artifact(pack_payload), prompt_artifact


def replay_pack_row(
    context: E7Context,
    question: SelectedQuestion,
    *,
    tokenizer: DeepSeekExactTokenizer,
    pack_row: Any,
    prompt_row: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pack_row = validate_sealed_artifact(pack_row)
    prompt_row = validate_sealed_artifact(prompt_row)
    expected_pack, expected_prompt = _compute_packs(
        context,
        question,
        tokenizer=tokenizer,
    )
    if pack_row != expected_pack:
        raise ExternalE7Error("saved E7 pack artifact differs from exact replay")
    if prompt_row != expected_prompt:
        raise ExternalE7Error("saved E7 plaintext prompt differs from exact replay")
    return pack_row, prompt_row


def run_pack_phase(context: E7Context, *, limit: int | None = None) -> None:
    tokenizer = DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    selected = _selected_prefix(context, limit)
    for completed, question in enumerate(selected, start=1):
        pack_path = e7_phase_path(context, "pack", question)
        prompt_path = e7_phase_path(context, "prompts", question)
        if pack_path.exists() != prompt_path.exists():
            raise ExternalE7Error("E7 pack has only one side of its bound artifact pair")
        started = perf_counter()
        if pack_path.exists():
            replay_pack_row(
                context,
                question,
                tokenizer=tokenizer,
                pack_row=load_json(pack_path, sealed=True),
                prompt_row=load_json(prompt_path, sealed=True),
            )
            action = "verified"
        else:
            pack, prompts = _compute_packs(context, question, tokenizer=tokenizer)
            write_json(prompt_path, prompts)
            write_json(pack_path, pack)
            action = "packed"
        row = load_json(pack_path, sealed=True)
        print(
            f"  pack: {completed}/{len(selected)} {action} {question.question_id} "
            f"A={row['cells']['E7-A']['final_prompt']['tokens']}/"
            f"{len(row['cells']['E7-A']['kept_turn_ids'])} "
            f"C={row['cells']['E7-C']['final_prompt']['tokens']}/"
            f"{len(row['cells']['E7-C']['kept_turn_ids'])} "
            f"wall={(perf_counter() - started):.1f}s",
            file=sys.stderr,
            flush=True,
        )


def _qa_arm_order(question: SelectedQuestion) -> tuple[str, str]:
    return ("E7-A", "E7-C") if question.position % 2 == 0 else ("E7-C", "E7-A")


def _qa_receipt_id(question_id: str, cell: str) -> str:
    return f"{question_id}:{cell.casefold()}"


def _validate_frozen_qa_result(
    result: ChatResult,
    *,
    expected_prompt: str,
    max_tokens: int,
    label: str,
) -> None:
    expected_prompt_bytes = expected_prompt.encode("utf-8")
    if result.prompt_bytes != expected_prompt_bytes:
        raise ExternalE7Error(f"{label} prompt bytes differ from the frozen prompt")
    request = result.request
    if (
        request.prompt != expected_prompt
        or request.model != DEEPSEEK_MODEL
        or request.temperature != 0.0
        or request.max_tokens != max_tokens
        or request.thinking_mode != "disabled"
    ):
        raise ExternalE7Error(f"{label} request configuration drifted")
    if result.endpoint_url != CONSTRUCTOR_DEPLOYMENT:
        raise ExternalE7Error(f"{label} endpoint drifted")
    if result.response_model != DEEPSEEK_MODEL:
        raise ExternalE7Error(f"{label} response model drifted")
    if not isinstance(result.request_id, str) or not result.request_id:
        raise ExternalE7Error(f"{label} provider request ID is missing")
    if not 1 <= result.attempts <= CHAT_MAX_ATTEMPTS:
        raise ExternalE7Error(f"{label} attempt count exceeds the frozen retry policy")
    if not result.content:
        raise ExternalE7Error(f"{label} response content is empty")


def _normalized_qa_arm(
    question: SelectedQuestion,
    cell: str,
    *,
    prompt_row: Mapping[str, Any],
    tokenizer: DeepSeekExactTokenizer,
    reader: ChatResult,
    judge: ChatResult,
    reader_receipt_sha256: str,
    judge_receipt_sha256: str,
) -> dict[str, Any]:
    prompt = str(prompt_row["cells"][cell]["prompt"])
    _validate_frozen_qa_result(
        reader,
        expected_prompt=prompt,
        max_tokens=READER_MAX_TOKENS,
        label=f"{question.question_id}/{cell}/reader",
    )
    exact_reader_tokens = tokenizer.exact_count(prompt)
    if exact_reader_tokens != prompt_row["cells"][cell]["prompt_tokens"]:
        raise ExternalE7Error("E7 packed prompt token count differs from replay")
    verify_provider_prompt_tokens(
        reader,
        expected=exact_reader_tokens,
        label=f"{question.question_id}/{cell}/reader",
    )
    judge_input = _judge_text_for_arm(question, reader.content)
    _validate_frozen_qa_result(
        judge,
        expected_prompt=judge_input,
        max_tokens=JUDGE_MAX_TOKENS,
        label=f"{question.question_id}/{cell}/judge",
    )
    exact_judge_tokens = tokenizer.exact_count(judge_input)
    verify_provider_prompt_tokens(
        judge,
        expected=exact_judge_tokens,
        label=f"{question.question_id}/{cell}/judge",
    )
    return {
        "hypothesis": reader.content,
        "development_judge_text": judge.content,
        "development_label": judge_label(judge.content),
        "reader": {
            "exact_local_prompt_tokens": exact_reader_tokens,
            "api_prompt_tokens": reader.prompt_tokens,
            "completion_tokens": reader.completion_tokens,
            "total_tokens": reader.total_tokens,
            "finish_reason": reader.finish_reason,
            "latency_ms": reader.latency_ms,
            "attempts": reader.attempts,
            "provider_request_id": reader.request_id,
            "raw_request_sha256": reader.raw_request_sha256,
            "raw_response_sha256": reader.raw_response_sha256,
            "receipt_sha256": reader_receipt_sha256,
            "estimated_cost_upper_bound_usd": _e7_upper_bound_cost(reader),
        },
        "development_judge": {
            "exact_local_prompt_tokens": exact_judge_tokens,
            "api_prompt_tokens": judge.prompt_tokens,
            "completion_tokens": judge.completion_tokens,
            "total_tokens": judge.total_tokens,
            "finish_reason": judge.finish_reason,
            "latency_ms": judge.latency_ms,
            "attempts": judge.attempts,
            "provider_request_id": judge.request_id,
            "raw_request_sha256": judge.raw_request_sha256,
            "raw_response_sha256": judge.raw_response_sha256,
            "receipt_sha256": judge_receipt_sha256,
            "estimated_cost_upper_bound_usd": _e7_upper_bound_cost(judge),
        },
        "estimated_cost_upper_bound_usd": (
            _e7_upper_bound_cost(reader) + _e7_upper_bound_cost(judge)
        ),
    }


def _qa_cost_basis() -> dict[str, Any]:
    return {
        "input": "all prompt tokens pessimistically priced as cache misses",
        "cache_miss_input_usd_per_million": (DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION),
        "output_usd_per_million": DEEPSEEK_OUTPUT_USD_PER_MILLION,
        "retained_successful_attempt": "provider-reported-usage",
        "each_unseen_prior_attempt": ("provider-prompt-tokens-plus-request-max-tokens"),
        "maximum_attempts": CHAT_MAX_ATTEMPTS,
    }


def _qa_claims() -> dict[str, Any]:
    return {
        "reader_model": DEEPSEEK_MODEL,
        "development_judge_model": DEEPSEEK_MODEL,
        "thinking_disabled": True,
        "api_and_local_prompt_tokens_reconciled": True,
        "E7_B_reader_executed": False,
        "official_gpt4o_judge_executed": False,
        "official_longmemeval_score": False,
    }


def replay_qa_row(
    context: E7Context,
    question: SelectedQuestion,
    *,
    tokenizer: DeepSeekExactTokenizer,
    qa_row: Any,
    receipt_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    qa_row = validate_sealed_artifact(qa_row)
    pack_path = e7_phase_path(context, "pack", question)
    prompt_path = e7_phase_path(context, "prompts", question)
    pack_row, prompt_row = replay_pack_row(
        context,
        question,
        tokenizer=tokenizer,
        pack_row=load_json(pack_path, sealed=True),
        prompt_row=load_json(prompt_path, sealed=True),
    )
    if len(receipt_records) != 4:
        raise ExternalE7Error("paired E7 QA requires two reader and two judge receipts")
    if any(not isinstance(receipt, dict) for receipt in receipt_records):
        raise ExternalE7Error("E7 QA receipt rows must be objects")
    arm_order = _qa_arm_order(question)
    expected_receipt_order = tuple(
        (receipt_id, role)
        for cell in arm_order
        for receipt_id in (_qa_receipt_id(question.question_id, cell),)
        for role in ("reader", "development_judge")
    )
    actual_receipt_order = tuple(
        (str(receipt.get("question_id")), str(receipt.get("call_role")))
        for receipt in receipt_records
    )
    if actual_receipt_order != expected_receipt_order:
        raise ExternalE7Error("E7 QA receipt order or arm/role coverage drifted")
    replayed: dict[tuple[str, str], ChatResult] = {}
    receipt_sha256: dict[tuple[str, str], str] = {}
    for receipt in receipt_records:
        result = validate_chat_receipt_record(receipt)
        key = (str(receipt["question_id"]), str(receipt["call_role"]))
        if key in replayed:
            raise ExternalE7Error("E7 QA receipts repeat an arm/role pair")
        replayed[key] = result
        receipt_sha256[key] = sha256_json(receipt)
    expected_arms: dict[str, Any] = {}
    total_cost = 0.0
    for cell in E7_CELLS:
        receipt_id = _qa_receipt_id(question.question_id, cell)
        reader_key = (receipt_id, "reader")
        judge_key = (receipt_id, "development_judge")
        if reader_key not in replayed or judge_key not in replayed:
            raise ExternalE7Error("E7 QA receipts do not cover every arm and role")
        expected_arm = _normalized_qa_arm(
            question,
            cell,
            prompt_row=prompt_row,
            tokenizer=tokenizer,
            reader=replayed[reader_key],
            judge=replayed[judge_key],
            reader_receipt_sha256=receipt_sha256[reader_key],
            judge_receipt_sha256=receipt_sha256[judge_key],
        )
        expected_arms[cell] = expected_arm
        total_cost += float(expected_arm["estimated_cost_upper_bound_usd"])
    wall_ms = qa_row.get("wall_ms")
    if type(wall_ms) is not float or not math.isfinite(wall_ms) or wall_ms < 0:
        raise ExternalE7Error("E7 QA wall time is invalid")
    raw_receipts = _receipt_bytes(receipt_records)
    expected = seal_artifact(
        {
            "artifact_type": "swarmbrain-longmemeval-e7-deepseek-qa-question",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "protocol_version": E7_RUN_PROTOCOL_VERSION,
            "run_manifest_sha256": context.manifest["artifact_sha256"],
            "question_id": question.question_id,
            "question_position": question.position,
            "source_pack_artifact_sha256": pack_row["artifact_sha256"],
            "source_prompt_artifact_sha256": prompt_row["artifact_sha256"],
            "receipt_artifact_sha256": sha256_bytes(raw_receipts),
            "receipt_artifact_bytes": len(raw_receipts),
            "arm_order": list(arm_order),
            "arms": expected_arms,
            "estimated_cost_upper_bound_usd": total_cost,
            "wall_ms": wall_ms,
            "cost_basis": _qa_cost_basis(),
            "claims": _qa_claims(),
        }
    )
    if qa_row != expected:
        raise ExternalE7Error("E7 QA artifact differs from exact raw-receipt replay")
    return qa_row


async def _run_qa_calls(
    context: E7Context,
    *,
    tokenizer: DeepSeekExactTokenizer,
    base_url: str,
    api_key: str,
    limit: int | None,
) -> None:
    reader_client = ChatClient(
        base_url=base_url,
        model=DEEPSEEK_MODEL,
        api_key=api_key,
        temperature=0.0,
        max_tokens=READER_MAX_TOKENS,
        required_response_model=DEEPSEEK_MODEL,
        require_request_id=True,
        thinking_mode="disabled",
        attempts=CHAT_MAX_ATTEMPTS,
    )
    judge_client = ChatClient(
        base_url=base_url,
        model=DEEPSEEK_MODEL,
        api_key=api_key,
        temperature=0.0,
        max_tokens=JUDGE_MAX_TOKENS,
        required_response_model=DEEPSEEK_MODEL,
        require_request_id=True,
        thinking_mode="disabled",
        attempts=CHAT_MAX_ATTEMPTS,
    )
    selected = _selected_prefix(context, limit)
    try:
        for completed, question in enumerate(selected, start=1):
            qa_path = e7_phase_path(context, "qa", question)
            receipts_path = e7_qa_receipt_path(context, question)
            if qa_path.exists() != receipts_path.exists():
                raise ExternalE7Error("E7 QA has only one side of its evidence pair")
            if qa_path.exists():
                replay_qa_row(
                    context,
                    question,
                    tokenizer=tokenizer,
                    qa_row=load_json(qa_path, sealed=True),
                    receipt_records=_load_receipts(receipts_path),
                )
                print(
                    f"  qa: {completed}/{len(selected)} verified {question.question_id}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            pack_path = e7_phase_path(context, "pack", question)
            prompt_path = e7_phase_path(context, "prompts", question)
            if not pack_path.exists() or not prompt_path.exists():
                raise ExternalE7Error("E7 QA requires packed prompts first")
            pack_row, prompt_row = replay_pack_row(
                context,
                question,
                tokenizer=tokenizer,
                pack_row=load_json(pack_path, sealed=True),
                prompt_row=load_json(prompt_path, sealed=True),
            )
            started = perf_counter()
            receipts: list[dict[str, Any]] = []
            arms: dict[str, Any] = {}
            for cell in _qa_arm_order(question):
                prompt = str(prompt_row["cells"][cell]["prompt"])
                exact_reader_tokens = tokenizer.exact_count(prompt)
                reader = await reader_client.complete(prompt)
                verify_provider_prompt_tokens(
                    reader,
                    expected=exact_reader_tokens,
                    label=f"{question.question_id}/{cell}/reader",
                )
                judge_input = _judge_text_for_arm(question, reader.content)
                exact_judge_tokens = tokenizer.exact_count(judge_input)
                judge = await judge_client.complete(judge_input)
                verify_provider_prompt_tokens(
                    judge,
                    expected=exact_judge_tokens,
                    label=f"{question.question_id}/{cell}/judge",
                )
                receipt_id = _qa_receipt_id(question.question_id, cell)
                reader_receipt = chat_receipt_record(receipt_id, "reader", reader)
                judge_receipt = chat_receipt_record(
                    receipt_id,
                    "development_judge",
                    judge,
                )
                validate_chat_receipt_record(reader_receipt)
                validate_chat_receipt_record(judge_receipt)
                receipts.extend((reader_receipt, judge_receipt))
                arms[cell] = _normalized_qa_arm(
                    question,
                    cell,
                    prompt_row=prompt_row,
                    tokenizer=tokenizer,
                    reader=reader,
                    judge=judge,
                    reader_receipt_sha256=sha256_json(reader_receipt),
                    judge_receipt_sha256=sha256_json(judge_receipt),
                )
            raw_receipts = _receipt_bytes(receipts)
            payload = {
                "artifact_type": "swarmbrain-longmemeval-e7-deepseek-qa-question",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "protocol_version": E7_RUN_PROTOCOL_VERSION,
                "run_manifest_sha256": context.manifest["artifact_sha256"],
                "question_id": question.question_id,
                "question_position": question.position,
                "source_pack_artifact_sha256": pack_row["artifact_sha256"],
                "source_prompt_artifact_sha256": prompt_row["artifact_sha256"],
                "receipt_artifact_sha256": sha256_bytes(raw_receipts),
                "receipt_artifact_bytes": len(raw_receipts),
                "arm_order": list(_qa_arm_order(question)),
                "arms": arms,
                "estimated_cost_upper_bound_usd": sum(
                    float(arm["estimated_cost_upper_bound_usd"]) for arm in arms.values()
                ),
                "wall_ms": (perf_counter() - started) * 1000.0,
                "cost_basis": _qa_cost_basis(),
                "claims": _qa_claims(),
            }
            sealed = seal_artifact(payload)
            atomic_write(receipts_path, raw_receipts)
            replay_qa_row(
                context,
                question,
                tokenizer=tokenizer,
                qa_row=sealed,
                receipt_records=tuple(receipts),
            )
            write_json(qa_path, sealed)
            print(
                f"  qa: {completed}/{len(selected)} ran {question.question_id} "
                f"A={arms['E7-A']['development_label']} "
                f"C={arms['E7-C']['development_label']} "
                f"cost<=${payload['estimated_cost_upper_bound_usd']:.5f}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        await reader_client.aclose()
        await judge_client.aclose()


def run_qa_phase(
    context: E7Context,
    *,
    base_url: str,
    api_key_env: str,
    limit: int | None = None,
) -> None:
    if base_url.strip().rstrip("/") not in {
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
    }:
        raise ExternalE7Error("development QA is frozen to the official DeepSeek endpoint")
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise ExternalE7Error(f"environment variable {api_key_env!r} is missing")
    tokenizer = DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    asyncio.run(
        _run_qa_calls(
            context,
            tokenizer=tokenizer,
            base_url=base_url,
            api_key=api_key,
            limit=limit,
        )
    )


def _style_counts(result: QueryConstructionResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in result.items:
        counts[item.style.value] = counts.get(item.style.value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def build_report(context: E7Context) -> dict[str, Any]:
    tokenizer = DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    cases: list[dict[str, Any]] = []
    source_artifacts: dict[str, list[str]] = {
        "windows": [],
        "construction": [],
        "contexts": [],
        "constructor_receipts": [],
        "pack": [],
        "prompts": [],
        "qa": [],
        "qa_receipts": [],
    }
    for question in context.e1.selected:
        windows = load_json(
            e7_phase_path(context, "windows", question),
            sealed=True,
        )
        replay_windows_row(context, question, windows)
        construction = load_json(
            e7_phase_path(context, "construction", question),
            sealed=True,
        )
        contexts = load_json(
            e7_phase_path(context, "contexts", question),
            sealed=True,
        )
        constructor_records = _load_constructor_chat_records(
            e7_constructor_receipt_path(context, question)
        )
        cells = replay_construction_row(
            context,
            question,
            construction_row=construction,
            contexts_row=contexts,
            records=constructor_records,
            tokenizer=tokenizer,
        )
        pack = load_json(e7_phase_path(context, "pack", question), sealed=True)
        prompts = load_json(
            e7_phase_path(context, "prompts", question),
            sealed=True,
        )
        replay_pack_row(
            context,
            question,
            tokenizer=tokenizer,
            pack_row=pack,
            prompt_row=prompts,
        )
        qa = load_json(e7_phase_path(context, "qa", question), sealed=True)
        qa_receipts = _load_receipts(e7_qa_receipt_path(context, question))
        replay_qa_row(
            context,
            question,
            tokenizer=tokenizer,
            qa_row=qa,
            receipt_records=qa_receipts,
        )
        constructor_receipt_bytes = _constructor_chat_bytes(constructor_records)
        qa_receipt_bytes = _receipt_bytes(qa_receipts)
        source_artifacts["windows"].append(windows["artifact_sha256"])
        source_artifacts["construction"].append(construction["artifact_sha256"])
        source_artifacts["contexts"].append(contexts["artifact_sha256"])
        source_artifacts["constructor_receipts"].append(sha256_bytes(constructor_receipt_bytes))
        source_artifacts["pack"].append(pack["artifact_sha256"])
        source_artifacts["prompts"].append(prompts["artifact_sha256"])
        source_artifacts["qa"].append(qa["artifact_sha256"])
        source_artifacts["qa_receipts"].append(sha256_bytes(qa_receipt_bytes))
        constructor_cost = float(construction["estimated_constructor_cost_upper_bound_usd"])
        arm_metrics: dict[str, Any] = {}
        for cell in E7_CELLS:
            result = cells[cell]
            pack_trace = pack["cells"][cell]
            ranked_ids = [_turn_id_payload(item.turn.turn_id) for item in result.items]
            arm_metrics[cell] = {
                "development_label": qa["arms"][cell]["development_label"],
                "prompt_tokens": pack_trace["final_prompt"]["tokens"],
                "candidate_items": len(pack_trace["candidate_order"]),
                "kept_items": len(pack_trace["kept_turn_ids"]),
                "retention_styles": _style_counts(result),
                "candidate_content_utf8_bytes": sum(
                    len(item.compressed_content.encode("utf-8")) for item in result.items
                ),
                "kept_content_utf8_bytes": sum(
                    int(item["compressed_content_utf8"]["bytes"])
                    for item in pack_trace["kept_items"]
                ),
                "reader_latency_ms": qa["arms"][cell]["reader"]["latency_ms"],
                "judge_latency_ms": qa["arms"][cell]["development_judge"]["latency_ms"],
                "estimated_constructor_cost_upper_bound_usd": (
                    0.0 if cell == "E7-A" else constructor_cost
                ),
                "estimated_reader_cost_upper_bound_usd": qa["arms"][cell]["reader"][
                    "estimated_cost_upper_bound_usd"
                ],
                "estimated_judge_cost_upper_bound_usd": qa["arms"][cell]["development_judge"][
                    "estimated_cost_upper_bound_usd"
                ],
                "estimated_qa_cost_upper_bound_usd": qa["arms"][cell][
                    "estimated_cost_upper_bound_usd"
                ],
                "estimated_online_cost_upper_bound_usd": (
                    (0.0 if cell == "E7-A" else constructor_cost)
                    + qa["arms"][cell]["reader"]["estimated_cost_upper_bound_usd"]
                ),
                "retrieval": _retrieval_metrics(
                    question,
                    ranked_ids=ranked_ids,
                    kept_ids=pack_trace["kept_turn_ids"],
                ),
            }
        token_reconciliation = construction["token_reconciliation"]
        cases.append(
            {
                "question_id": question.question_id,
                "question_position": question.position,
                "question_type": str(question.record["question_type"]),
                "abstention": is_abstention_question(question.question_id),
                "arm_order": qa["arm_order"],
                "construction": {
                    "windows": construction["constructor_chat_receipt_count"],
                    "source_window_appearances": sum(
                        int(request["messages"]) for request in windows["constructor_requests"]
                    ),
                    "E7_C_items": len(cells["E7-C"].items),
                    "E7_C_retention_styles": _style_counts(cells["E7-C"]),
                    "input_tokens": sum(
                        int(row["exact_local_prompt_tokens"]) for row in token_reconciliation
                    ),
                    "output_tokens": sum(
                        int(row["completion_tokens"]) for row in token_reconciliation
                    ),
                    "latency_ms_sum": sum(
                        float(
                            _validate_constructor_chat_record(
                                question,
                                index,
                                record,
                            ).latency_ms
                        )
                        for index, record in enumerate(constructor_records)
                    ),
                    "question_wall_ms": construction["wall_ms"],
                    "estimated_cost_upper_bound_usd": construction[
                        "estimated_constructor_cost_upper_bound_usd"
                    ],
                },
                "arms": arm_metrics,
                "paired_delta": int(arm_metrics["E7-C"]["development_label"])
                - int(arm_metrics["E7-A"]["development_label"]),
            }
        )
    count = len(cases)
    summaries: dict[str, Any] = {}
    for cell in E7_CELLS:
        successes = sum(bool(case["arms"][cell]["development_label"]) for case in cases)
        gold_cases = [
            case for case in cases if case["arms"][cell]["retrieval"]["gold_session_positions"] > 0
        ]
        summaries[cell] = {
            "development_accuracy": successes / count,
            "development_accuracy_wilson_95": list(_wilson_interval(successes, count)),
            "mean_prompt_tokens": _mean(case["arms"][cell]["prompt_tokens"] for case in cases),
            "mean_candidate_items": _mean(case["arms"][cell]["candidate_items"] for case in cases),
            "mean_kept_items": _mean(case["arms"][cell]["kept_items"] for case in cases),
            "mean_candidate_content_utf8_bytes": _mean(
                case["arms"][cell]["candidate_content_utf8_bytes"] for case in cases
            ),
            "mean_kept_content_utf8_bytes": _mean(
                case["arms"][cell]["kept_content_utf8_bytes"] for case in cases
            ),
            "mean_reader_latency_ms": _mean(
                case["arms"][cell]["reader_latency_ms"] for case in cases
            ),
            "total_estimated_constructor_cost_upper_bound_usd": sum(
                case["arms"][cell]["estimated_constructor_cost_upper_bound_usd"] for case in cases
            ),
            "total_estimated_reader_cost_upper_bound_usd": sum(
                case["arms"][cell]["estimated_reader_cost_upper_bound_usd"] for case in cases
            ),
            "total_estimated_judge_cost_upper_bound_usd": sum(
                case["arms"][cell]["estimated_judge_cost_upper_bound_usd"] for case in cases
            ),
            "total_estimated_online_cost_upper_bound_usd": sum(
                case["arms"][cell]["estimated_online_cost_upper_bound_usd"] for case in cases
            ),
            "mean_estimated_qa_cost_upper_bound_usd": _mean(
                case["arms"][cell]["estimated_qa_cost_upper_bound_usd"] for case in cases
            ),
            "mean_estimated_online_cost_upper_bound_usd": _mean(
                case["arms"][cell]["estimated_online_cost_upper_bound_usd"] for case in cases
            ),
            "gold_eligible_questions": len(gold_cases),
            "any_gold_session_in_prompt": _mean(
                float(case["arms"][cell]["retrieval"]["any_gold_session_in_prompt"])
                for case in gold_cases
            ),
            "all_gold_sessions_in_prompt": _mean(
                float(case["arms"][cell]["retrieval"]["all_gold_sessions_in_prompt"])
                for case in gold_cases
            ),
            "mean_answer_session_recall": _mean(
                case["arms"][cell]["retrieval"]["answer_session_recall"] for case in gold_cases
            ),
            "mean_candidate_order_mrr": _mean(
                case["arms"][cell]["retrieval"]["candidate_mrr"] for case in gold_cases
            ),
        }
    construction_summary = {
        "mean_windows": _mean(case["construction"]["windows"] for case in cases),
        "mean_source_window_appearances": _mean(
            case["construction"]["source_window_appearances"] for case in cases
        ),
        "mean_E7_C_items": _mean(case["construction"]["E7_C_items"] for case in cases),
        "total_input_tokens": sum(case["construction"]["input_tokens"] for case in cases),
        "total_output_tokens": sum(case["construction"]["output_tokens"] for case in cases),
        "total_estimated_cost_upper_bound_usd": sum(
            case["construction"]["estimated_cost_upper_bound_usd"] for case in cases
        ),
    }
    differences = [int(case["paired_delta"]) for case in cases]
    bootstrap = _paired_bootstrap_interval(differences)
    improved = sum(value > 0 for value in differences)
    regressed = sum(value < 0 for value in differences)
    tied = count - improved - regressed
    qa_cost = sum(
        case["arms"][cell]["estimated_qa_cost_upper_bound_usd"]
        for case in cases
        for cell in E7_CELLS
    )
    gold_denominator_valid = (
        summaries["E7-A"]["gold_eligible_questions"] == summaries["E7-C"]["gold_eligible_questions"]
        and summaries["E7-A"]["gold_eligible_questions"] > 0
    )
    point_signal = (
        gold_denominator_valid
        and summaries["E7-C"]["development_accuracy"] > summaries["E7-A"]["development_accuracy"]
        and summaries["E7-C"]["any_gold_session_in_prompt"]
        >= summaries["E7-A"]["any_gold_session_in_prompt"]
        and summaries["E7-C"]["all_gold_sessions_in_prompt"]
        >= summaries["E7-A"]["all_gold_sessions_in_prompt"]
        and summaries["E7-C"]["mean_answer_session_recall"]
        >= summaries["E7-A"]["mean_answer_session_recall"]
        and summaries["E7-C"]["total_estimated_online_cost_upper_bound_usd"]
        < summaries["E7-A"]["total_estimated_online_cost_upper_bound_usd"]
    )
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e7-development-pilot-report",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E7_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "source_e1_manifest_sha256": context.e1.manifest["artifact_sha256"],
        "classification": "exploratory-development-sample-not-official-longmemeval-score",
        "question_count": count,
        "reader": DEEPSEEK_MODEL,
        "judge": {
            "model": DEEPSEEK_MODEL,
            "role": "development-judge",
            "official_gpt4o_executed": False,
        },
        "constructor": {
            "model": DEEPSEEK_MODEL,
            "identity_authentication": "caller-attested-unverified",
            "E7_B_executed": False,
            **construction_summary,
        },
        "reader_arms": summaries,
        "paired": {
            "contrast": "E7-C-minus-E7-A",
            "mean_accuracy_delta": _mean(differences),
            "bootstrap_95": list(bootstrap),
            "bootstrap_samples": 50_000,
            "bootstrap_seed": 20260809,
            "improved": improved,
            "regressed": regressed,
            "tied": tied,
        },
        "cost_basis": {
            "shared_E1_B_retrieval_and_reranking_cost_excluded_from_delta": True,
            "online_cost": "constructor-plus-reader",
            "judge_cost_excluded_from_online_decision": True,
            "input": "all prompt tokens pessimistically priced as cache misses",
            "retry_attempts_priced": True,
            "billed_cost_or_endpoint_authentication_claimed": False,
        },
        "total_estimated_cost_upper_bound_usd": (
            construction_summary["total_estimated_cost_upper_bound_usd"] + qa_cost
        ),
        "source_artifact_sets": {
            key: {"count": len(values), "ordered_sha256": sha256_json(values)}
            for key, values in source_artifacts.items()
        },
        "cases": cases,
        "decision": {
            "gold_denominator_nonzero_and_shared": gold_denominator_valid,
            "e7c_point_signal_on_development_sample": point_signal,
            "positive_paired_lower_confidence_bound": bootstrap[0] > 0.0,
            "e7c_confirmatory_signal": point_signal and bootstrap[0] > 0.0,
            "eligible_for_composition_or_production_promotion": False,
            "required_next_if_promising": (
                "scale the frozen E7-C paired cell before composition or production changes"
            ),
        },
        "claims": {
            "gold_used_only_for_posthoc_report_metrics_and_development_judging": True,
            "all_E7_C_output_byte_grounded": True,
            "E7_B_abstractive_cell_executed": False,
            "paper_reproduction_claimed": False,
            "official_longmemeval_score": False,
            "sealed_heldout_confirmation": False,
            "causal_production_improvement_proven": False,
            "production_policy_changed": False,
        },
    }
    report = seal_artifact(payload)
    write_json(context.output_dir / "report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("windows", "construct", "pack", "qa", "report", "all"),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--e1-output-dir", type=Path, default=DEFAULT_E1_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_E7_OUTPUT)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--qwen-root", type=Path, default=DEFAULT_QWEN_ROOT)
    parser.add_argument("--cross-encoder-root", type=Path, default=DEFAULT_CE_ROOT)
    parser.add_argument("--deepseek-root", type=Path, default=DEFAULT_DEEPSEEK_ROOT)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--qwen-batch-size", type=int, default=QWEN_BATCH_SIZE)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="execute only the selected prefix; procedural smoke control, not in manifest",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.qwen_batch_size < 1:
        raise SystemExit("batch size must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("execution limit must be positive")
    if args.limit is not None and args.phase in {"report", "all"}:
        raise SystemExit("report/all requires the complete frozen sample; omit --limit")
    context = build_e7_context(args)
    phases = (
        ("windows", "construct", "pack", "qa", "report") if args.phase == "all" else (args.phase,)
    )
    for phase in phases:
        if phase == "windows":
            run_windows_phase(context, limit=args.limit)
        elif phase == "construct":
            run_construct_phase(
                context,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                limit=args.limit,
            )
        elif phase == "pack":
            run_pack_phase(context, limit=args.limit)
        elif phase == "qa":
            run_qa_phase(
                context,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                limit=args.limit,
            )
        else:
            report = build_report(context)
            print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
