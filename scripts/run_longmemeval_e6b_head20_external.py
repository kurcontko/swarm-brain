#!/usr/bin/env python3
"""Preregistered LongMemEval E6b R0-versus-head-matched-R1 experiment.

This runner evaluates one deliberately narrow representation hypothesis:

``R0``
    Rank the immutable raw F0 turns with the already-observed pinned Qwen
    cosine scores from E1.
``R1@20``
    Add exactly one source-only merged summary/fact/keyword navigation key per
    raw turn, rank that family with the same pinned Qwen encoder, fuse the two
    family heads with the frozen E6 RRF contract, retain exactly the first 20
    fused canonical values, and hydrate only those raw turns into the reader
    prompt. This matches R0's candidate head before packing.

The run is resumable and fail-closed.  Exact provider request/response bytes,
usage, retries, latency, prices, local tokenizer reconciliation, local model
snapshots, prompts, packs, and post-hoc gold context metrics are retained or
bound by digest. Reader and development-judge calls use DeepSeek V4 Flash and
run only after the context gate. The final paired decision is frozen before
outcomes are observed. GPT-4o is intentionally deferred until a later stable,
confirmed run; no E6b output is an official LongMemEval score.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from argparse import Namespace
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
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
from benchmarks.integrations.longmemeval_representation import (
    KEY_FAMILY_DEPTH,
    CanonicalValue,
    KeyFamily,
    RankedFamilyObservation,
    RankedKeyScore,
    RepresentationCell,
    RepresentationCorpus,
    RepresentationResult,
    ScorerIdentity,
    compile_question_canonical_values,
    evaluate_representation_cell,
    raw_key_id,
)
from benchmarks.integrations.longmemeval_representation.diagnostic import (
    CASE_ARTIFACT_TYPE,
    CASE_SCHEMA_VERSION,
    compile_r0_r1_diagnostic,
    seal_case_input,
)
from benchmarks.integrations.longmemeval_representation.diagnostic import (
    PROTOCOL_VERSION as DIAGNOSTIC_PROTOCOL_VERSION,
)
from benchmarks.integrations.longmemeval_representation.evidence import (
    DEEPSEEK_DEPLOYMENT_ID,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS,
    DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS,
    PROMPT_IDENTITY_SHA256,
    DeepSeekR1ExtractionEvidence,
    DeepSeekR1PricingIdentity,
    DeepSeekR1ProviderAttempt,
    RepresentationEvidenceError,
    build_deepseek_r1_attempt_record,
    build_deepseek_r1_evidence_record,
    deepseek_r1_attempt_jsonl_bytes,
    deepseek_r1_evidence_jsonl_bytes,
    deepseek_r1_extractor_identity,
    deepseek_r1_request_bytes,
    replay_deepseek_r1_attempt_jsonl,
    replay_deepseek_r1_evidence_jsonl,
    replay_deepseek_r1_evidence_record,
)
from benchmarks.integrations.longmemeval_representation.head_matched import (
    HEAD_MATCHED_VALUE_COUNT,
    head_match_representation_result,
)
from benchmarks.integrations.longmemeval_representation.packing_bridge import (
    RepresentationPromptPackingResult,
    pack_representation_result,
)
from benchmarks.integrations.longmemeval_selection_report.contracts import (
    BOOTSTRAP_CONFIDENCE as SELECTION_BOOTSTRAP_CONFIDENCE,
)
from benchmarks.integrations.longmemeval_selection_report.contracts import (
    BOOTSTRAP_RESAMPLES as SELECTION_BOOTSTRAP_RESAMPLES,
)
from benchmarks.integrations.longmemeval_selection_report.contracts import (
    BOOTSTRAP_SEED as SELECTION_BOOTSTRAP_SEED,
)
from benchmarks.integrations.longmemeval_selection_report.contracts import MAX_TYPE_REGRESSION
from benchmarks.integrations.longmemeval_selection_report.metrics import (
    ArmOutcome,
    PairedQACase,
    paired_qa_summary,
    qa_by_question_type,
)
from benchmarks.integrations.longmemeval_turns import (
    TurnProjection,
    TurnProjectionCorpus,
    compile_official_longmemeval_s,
)
from run_longmemeval_qa import (
    ChatProtocolError,
    chat_request_bytes,
    chat_result_from_raw_response,
    judge_label,
    replay_chat_request,
)
from scripts._longmemeval_common import QWEN_QUERY_INSTRUCTION
from scripts.run_longmemeval_e1_external import (
    ARTIFACT_SCHEMA_VERSION,
    DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION,
    DEEPSEEK_FILES,
    DEEPSEEK_MODEL,
    DEEPSEEK_OUTPUT_USD_PER_MILLION,
    DEEPSEEK_REVISION,
    DEFAULT_CE_ROOT,
    DEFAULT_DATASET,
    DEFAULT_DEEPSEEK_ROOT,
    DEFAULT_QWEN_ROOT,
    QWEN_ATTENTION_CELL_BUDGET,
    QWEN_BATCH_SIZE,
    QWEN_FILES,
    QWEN_MAX_LENGTH,
    QWEN_MAX_PADDING_RATIO,
    QWEN_MODEL,
    QWEN_REVISION,
    TOKEN_BUDGET,
    ChatResult,
    DeepSeekExactTokenizer,
    ExperimentContext,
    ExternalE1Error,
    QwenEmbedder,
    SelectedQuestion,
    _judge_text_for_arm,
    _load_receipts,
    _receipt_bytes,
    _safe_question_id,
    _snapshot_artifact,
    _turn_id_from_payload,
    _turn_id_payload,
    atomic_write,
    chat_receipt_record,
    load_json,
    phase_path,
    qwen_query_text,
    replay_dense_row,
    seal_artifact,
    selected_positions,
    sha256_bytes,
    sha256_json,
    validate_chat_receipt_record,
    verify_provider_prompt_tokens,
    verify_snapshot,
    write_json,
)

E6_RUN_PROTOCOL_VERSION = "swarmbrain-longmemeval-e6b-head20-development-v1"
E1_PROTOCOL_VERSION = "swarmbrain-longmemeval-e1-real-model-development-v3"
DEFAULT_E1_OUTPUT = Path("/private/tmp/swarmbrain-longmemeval-e1-e6b-n160-v1")
DEFAULT_E6_OUTPUT = Path("/private/tmp/swarmbrain-longmemeval-e6b-head20-n160-v1")

E6B_SAMPLE = 160
E6B_SEED = 20282059
E6B_SELECTION_SEARCH_START = 20260810
E6B_PYTHON_VERSION = "3.12.13"
E6B_DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
E6B_SOURCE_E6_OUTPUT = Path("/private/tmp/swarmbrain-longmemeval-e6-pilot-v2")
E6B_SOURCE_E6_REPORT_SHA256 = "c56dd9ff90c881f52416c53be52e654018adb29cbcc8f847fe92eaf34933bb49"
E6B_SOURCE_E6_MANIFEST_SHA256 = "ab6a7a5d943ac46c0b423c7bb71c35fedcdc45f44a54ea9bb453b20f6a99ba1b"
E6B_EXCLUSION_ROWS_SHA256 = "468b2ce21fa3885b4a1e17cb15e19299b7ecb8f184431f25c649b182edb87ff2"
E6B_POSITIONS_SHA256 = "13188de9750669fc53e8704dbefa60d4f30b4dff9992edd0d1c9053b03ae7b95"
E6B_SELECTOR_BINDING_SHA256 = "dfdd07fe82288be055b0e6359cad5cc0192a3663b17202c9b9997e628f6353d4"
E6B_POSITION_QUESTION_ROWS_SHA256 = (
    "41a7d73a546b768606fd50b655c53e2d2b9ed570a16e15f3719619183342dd03"
)
E6B_QUESTION_IDS_SHA256 = "0047fd256b16dcdc424957508f227149c095932ec38fafbb3f8037f674b16b34"
E6B_RUNNER_QUESTION_BINDING_SHA256 = (
    "e8f39c354b4a1773fbf194d2547227f3730a86c8e05e351c099ece36def1c84b"
)
E6B_FULL_RUN_ROWS_SHA256 = "0da325831620039dd2cd96a20aa56f0b80bc3c4a5fae7b2d6f6e44eca872786a"
E6B_FIRST40_RUN_ROWS_SHA256 = "a7fa2bfa0c61c25773f106cf3abc550c07b1efeec1797013adb3b4d95c73521d"
E6B_E1_SAMPLE_BINDING_SHA256 = "f9189d2ec0cdaf94e59be0ec7e9545fe6f906788dccf5a890c130376c0947243"
E6B_ORIGINAL_PILOT_POSITIONS = frozenset({17, 29, 160, 169, 185, 221, 228, 394, 422, 478})
E6B_TARGET_TYPE_COUNTS = {
    "knowledge-update": 25,
    "multi-session": 42,
    "single-session-assistant": 18,
    "single-session-preference": 10,
    "single-session-user": 22,
    "temporal-reasoning": 43,
}
E6B_ACTUAL_TYPE_COUNTS = {
    "knowledge-update": 25,
    "multi-session": 42,
    "single-session-assistant": 19,
    "single-session-preference": 10,
    "single-session-user": 22,
    "temporal-reasoning": 42,
}
E6B_ABS_COUNT = 10
E6B_SOURCE_TURN_COUNT = 79_130
E6B_QWEN_DEVICE = "mps"
E6B_QWEN_BATCH_SIZE = 8
E6B_TORCH_VERSION = "2.7.1"
E6B_TRANSFORMERS_VERSION = "4.55.4"
E6B_QWEN_DTYPE = "float16"
E6B_BOOTSTRAP_SEED = SELECTION_BOOTSTRAP_SEED
E6B_BOOTSTRAP_SAMPLES = SELECTION_BOOTSTRAP_RESAMPLES

E6_CELLS = (RepresentationCell.RAW.value, RepresentationCell.RAW_MERGED_SFK.value)
EXTRACTOR_MODEL_REVISION = "official-api-alias-observed-2026-08-09"
PRICING_VERSION = "deepseek-public-pricing-observed-2026-08-09"
EXTRACTION_CONCURRENCY = 24
READER_MAX_TOKENS = 4096
JUDGE_MAX_TOKENS = 64
QA_HTTP_ATTEMPTS = 4
MAX_EXTERNAL_COST_MICROUSD = 5_600_000
QWEN_COSINE_TOLERANCE = 0.00001

QWEN_SCORER_PROTOCOL = "qwen3-embedding-last-token-cosine-v1"
QWEN_IDENTITY_PRODUCER = "scripts.run_longmemeval_e1_external.QwenEmbedder"

IMPLEMENTATION_FILES = (
    "benchmarks/integrations/longmemeval_e1/__init__.py",
    "benchmarks/integrations/longmemeval_e1/contracts.py",
    "benchmarks/integrations/longmemeval_e1/selection.py",
    "benchmarks/integrations/longmemeval_official_preflight/__init__.py",
    "benchmarks/integrations/longmemeval_official_preflight/contracts.py",
    "benchmarks/integrations/longmemeval_official_preflight/preflight.py",
    "benchmarks/integrations/longmemeval_representation/__init__.py",
    "benchmarks/integrations/longmemeval_representation/contracts.py",
    "benchmarks/integrations/longmemeval_representation/diagnostic.py",
    "benchmarks/integrations/longmemeval_representation/evidence.py",
    "benchmarks/integrations/longmemeval_representation/experiment.py",
    "benchmarks/integrations/longmemeval_representation/packing_bridge.py",
    "benchmarks/integrations/longmemeval_turn_prompt/__init__.py",
    "benchmarks/integrations/longmemeval_turn_prompt/contracts.py",
    "benchmarks/integrations/longmemeval_turn_prompt/packer.py",
    "benchmarks/integrations/longmemeval_turn_retrieval/__init__.py",
    "benchmarks/integrations/longmemeval_turn_retrieval/contracts.py",
    "benchmarks/integrations/longmemeval_turn_retrieval/fusion.py",
    "benchmarks/integrations/longmemeval_turns/__init__.py",
    "benchmarks/integrations/longmemeval_turns/compiler.py",
    "pyproject.toml",
    "scripts/_longmemeval_common.py",
    "scripts/_longmemeval_tokenizer.py",
    "scripts/run_longmemeval_e1_external.py",
    "scripts/run_longmemeval_e6b_head20_external.py",
    "benchmarks/integrations/longmemeval_representation/head_matched.py",
    "benchmarks/integrations/longmemeval_selection_report/contracts.py",
    "benchmarks/integrations/longmemeval_selection_report/metrics.py",
    "docs/research/longmemeval-e6b-head20-protocol-2026-08-09.md",
    "scripts/run_longmemeval_qa.py",
    "src/swarmbrain/adapters/memory/retrieval.py",
    "src/swarmbrain/retrieval/projection.py",
    "uv.lock",
)

IMPLEMENTATION_TREE_ROOTS = (
    "benchmarks/integrations",
    "src/swarmbrain/retrieval",
)


class ExternalE6Error(ExternalE1Error):
    """Saved E6 evidence is incomplete, inconsistent, or outside the protocol."""


class ExternalCostCapExceeded(ExternalE6Error):
    """The preregistered DeepSeek ledger cannot admit another required call."""


_EXTRACTION_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_EXTRACTION_BACKOFF_SECONDS = (1.0, 3.0, 8.0, 20.0)


class _ExtractionChatClient:
    """Return the first valid HTTP 2xx envelope, including empty content.

    The shared QA client intentionally retries empty model content.  E6 treats
    empty content as an application-schema attempt that must retain its exact
    provider bytes, so extraction needs this narrower transport boundary.
    Transport/status retries remain pessimistically priced and represented by
    ``ChatResult.attempts``; the first 2xx response is never silently dropped.
    """

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.endpoint_url = DEEPSEEK_DEPLOYMENT_ID
        self.max_tokens = DEEPSEEK_MAX_TOKENS
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
        return self._client

    async def complete(
        self,
        prompt: str,
        *,
        raw_request: bytes,
        on_first_2xx: Callable[[bytes, int, float, str], None] | None = None,
    ) -> ChatResult:
        client = self._ensure_client()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }
        started = perf_counter()
        last = "no attempt was made"
        for attempt in range(DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS):
            try:
                response = await client.post(
                    self.endpoint_url,
                    content=raw_request,
                    headers=headers,
                )
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if status in _EXTRACTION_RETRYABLE_STATUS:
                    last = f"HTTP {status}"
                elif not 200 <= status < 300:
                    raise ChatProtocolError(
                        f"DeepSeek rejected the E6 extraction request with HTTP {status}"
                    )
                else:
                    raw_response = bytes(response.content)
                    latency_ms = (perf_counter() - started) * 1000.0
                    encoding = str(response.headers.get("content-encoding", ""))
                    if on_first_2xx is not None:
                        on_first_2xx(raw_response, attempt + 1, latency_ms, encoding)
                    if encoding.strip().casefold() not in {"", "identity"}:
                        raise ChatProtocolError(
                            "DeepSeek returned an unsupported extraction content encoding"
                        )
                    result = chat_result_from_raw_response(
                        raw_response,
                        prompt=prompt,
                        attempts=attempt + 1,
                        latency_ms=latency_ms,
                        raw_request=raw_request,
                        endpoint_url=self.endpoint_url,
                    )
                    if result.response_model != DEEPSEEK_MODEL:
                        raise ChatProtocolError("DeepSeek extraction response model drifted")
                    if not result.request_id:
                        raise ChatProtocolError(
                            "DeepSeek extraction response has no provider request ID"
                        )
                    return result
            if attempt + 1 < DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS:
                await asyncio.sleep(
                    _EXTRACTION_BACKOFF_SECONDS[min(attempt, len(_EXTRACTION_BACKOFF_SECONDS) - 1)]
                )
        raise ExternalE6Error(
            "DeepSeek extraction transport failed after "
            f"{DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS} attempts: {last}"
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


class _QAFirst2xxChatClient:
    """Retain and return the first QA 2xx before semantic validation."""

    def __init__(
        self,
        *,
        api_key: str,
        max_tokens: int,
        timeout_seconds: float = 300.0,
        client: Any | None = None,
    ) -> None:
        self.endpoint_url = DEEPSEEK_DEPLOYMENT_ID
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
        return self._client

    async def complete(
        self,
        prompt: str,
        *,
        raw_request: bytes,
        on_first_2xx: Callable[[bytes, int, float, str], None],
    ) -> ChatResult:
        request = replay_chat_request(raw_request)
        if (
            request.prompt != prompt
            or request.model != DEEPSEEK_MODEL
            or request.temperature != 0.0
            or request.max_tokens != self.max_tokens
            or request.thinking_mode != "disabled"
        ):
            raise ExternalE6Error("QA first-2xx client request differs from frozen configuration")
        client = self._ensure_client()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }
        started = perf_counter()
        last = "no attempt was made"
        for attempt in range(QA_HTTP_ATTEMPTS):
            try:
                response = await client.post(
                    self.endpoint_url,
                    content=raw_request,
                    headers=headers,
                )
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if status in _EXTRACTION_RETRYABLE_STATUS:
                    last = f"HTTP {status}"
                elif not 200 <= status < 300:
                    raise ChatProtocolError(
                        f"DeepSeek rejected the E6 QA request with HTTP {status}"
                    )
                else:
                    raw_response = bytes(response.content)
                    latency_ms = (perf_counter() - started) * 1000.0
                    encoding = str(response.headers.get("content-encoding", ""))
                    on_first_2xx(raw_response, attempt + 1, latency_ms, encoding)
                    if encoding.strip().casefold() not in {"", "identity"}:
                        raise ChatProtocolError(
                            "DeepSeek returned an unsupported QA content encoding"
                        )
                    return chat_result_from_raw_response(
                        raw_response,
                        prompt=prompt,
                        attempts=attempt + 1,
                        latency_ms=latency_ms,
                        raw_request=raw_request,
                        endpoint_url=self.endpoint_url,
                    )
            if attempt + 1 < QA_HTTP_ATTEMPTS:
                await asyncio.sleep(
                    _EXTRACTION_BACKOFF_SECONDS[min(attempt, len(_EXTRACTION_BACKOFF_SECONDS) - 1)]
                )
        raise ExternalE6Error(
            f"DeepSeek QA transport failed after {QA_HTTP_ATTEMPTS} attempts: {last}"
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


@dataclass(frozen=True, slots=True)
class E6Context:
    e1: ExperimentContext
    source_bytes: bytes
    preflight: RunPreflightManifest
    output_dir: Path
    manifest: dict[str, Any]
    extractor: Any
    pricing: DeepSeekR1PricingIdentity
    values_by_question_id: Mapping[str, tuple[CanonicalValue, ...]]


@contextmanager
def _output_process_lock(output_dir: Path) -> Iterator[None]:
    raw_output = Path(output_dir)
    if raw_output.is_symlink():
        raise ExternalE6Error("E6 output directory cannot be a symlink")
    resolved = raw_output.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    lock_path = resolved / ".e6b-run.lock"
    if lock_path.is_symlink():
        raise ExternalE6Error("E6 process lock cannot be a symlink")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    handle = os.fdopen(descriptor, "a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ExternalE6Error("another E6 process already owns this output directory") from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_journal_parent(
    context: E6Context,
    paths: _CallJournalPaths,
) -> None:
    output = context.output_dir
    journal_root = output / "external-call-journal"
    namespace_root = paths.reservation.parent
    for directory in (output, journal_root, namespace_root):
        if directory.is_symlink():
            raise ExternalE6Error("external call journal directory cannot be a symlink")
        if directory.exists() and not directory.is_dir():
            raise ExternalE6Error("external call journal directory path is not a directory")
        if not directory.exists():
            directory.mkdir()
        _fsync_parent(directory)
        _fsync_directory(directory)


def _durable_write_json(path: Path, value: Any) -> None:
    write_json(path, value)
    _fsync_parent(path)


def _durable_atomic_write(path: Path, payload: bytes) -> None:
    atomic_write(path, payload)
    _fsync_parent(path)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _strict_records(path: Path) -> tuple[dict[str, Any], ...]:
    value = load_json(path)
    if not isinstance(value, list) or not value or any(not isinstance(row, dict) for row in value):
        raise ExternalE6Error("LongMemEval source must be a non-empty array of objects")
    return tuple(value)


def _selected_questions(
    corpus: TurnProjectionCorpus,
    records: tuple[dict[str, Any], ...],
    *,
    sample: int,
    seed: int,
) -> tuple[SelectedQuestion, ...]:
    positions = selected_positions(len(records), sample=sample, seed=seed)
    turns_by_question: dict[str, list[TurnProjection]] = {}
    for turn in corpus.turns:
        turns_by_question.setdefault(turn.turn_id.question_id, []).append(turn)
    return tuple(
        SelectedQuestion(
            position=position,
            record=records[position],
            turns=tuple(turns_by_question[str(records[position]["question_id"])]),
        )
        for position in positions
    )


def _e6b_selection_summary(
    records: Sequence[Mapping[str, Any]],
    positions: Sequence[int],
) -> dict[str, Any]:
    type_counts = Counter(str(records[position]["question_type"]) for position in positions)
    abs_count = sum("_abs" in str(records[position]["question_id"]) for position in positions)
    return {
        "type_counts": dict(sorted(type_counts.items())),
        "abs_count": abs_count,
    }


def _e6b_selection_qualifies(
    records: Sequence[Mapping[str, Any]],
    positions: Sequence[int],
) -> bool:
    if E6B_ORIGINAL_PILOT_POSITIONS.intersection(positions):
        return False
    summary = _e6b_selection_summary(records, positions)
    counts = summary["type_counts"]
    return bool(
        summary["abs_count"] == E6B_ABS_COUNT
        and set(counts) == set(E6B_TARGET_TYPE_COUNTS)
        and max(abs(int(counts[name]) - target) for name, target in E6B_TARGET_TYPE_COUNTS.items())
        <= 1
    )


def _validate_e6b_selection(
    records: Sequence[Mapping[str, Any]],
    selected: Sequence[SelectedQuestion],
) -> dict[str, Any]:
    if sys.version.split()[0] != E6B_PYTHON_VERSION:
        raise ExternalE6Error(
            f"E6b selection requires CPython {E6B_PYTHON_VERSION}, got {sys.version.split()[0]}"
        )
    positions = tuple(question.position for question in selected)
    if len(records) != 500 or len(positions) != E6B_SAMPLE:
        raise ExternalE6Error("E6b requires the frozen 500-question source and n=160 cohort")
    if positions != selected_positions(len(records), sample=E6B_SAMPLE, seed=E6B_SEED):
        raise ExternalE6Error("E6b cohort differs from the frozen Python-random selection")
    summary = _e6b_selection_summary(records, positions)
    if summary["type_counts"] != E6B_ACTUAL_TYPE_COUNTS or not _e6b_selection_qualifies(
        records, positions
    ):
        raise ExternalE6Error("E6b cohort no longer satisfies its metadata-only balance rule")
    first_qualifying_seed = None
    for candidate_seed in range(E6B_SELECTION_SEARCH_START, E6B_SEED + 1):
        candidate = selected_positions(len(records), sample=E6B_SAMPLE, seed=candidate_seed)
        if _e6b_selection_qualifies(records, candidate):
            first_qualifying_seed = candidate_seed
            break
    if first_qualifying_seed != E6B_SEED:
        raise ExternalE6Error("E6b seed is not the first seed satisfying the frozen selection rule")
    source_turn_count = sum(len(question.turns) for question in selected)
    if source_turn_count != E6B_SOURCE_TURN_COUNT:
        raise ExternalE6Error("E6b cohort source-turn count differs from preregistration")
    binding = [
        {
            "position": question.position,
            "question_id": question.question_id,
            "question_type": str(question.record["question_type"]),
            "abs": "_abs" in question.question_id,
        }
        for question in selected
    ]
    exclusion_rows = [
        {
            "abstention": "_abs" in str(records[position]["question_id"]),
            "position": position,
            "question_id": str(records[position]["question_id"]),
            "question_type": str(records[position]["question_type"]),
        }
        for position in sorted(E6B_ORIGINAL_PILOT_POSITIONS)
    ]
    position_question_rows = [
        {"position": question.position, "question_id": question.question_id}
        for question in selected
    ]
    full_run_rows = [
        {
            "abstention": "_abs" in question.question_id,
            "position": question.position,
            "question_id": question.question_id,
            "question_type": str(question.record["question_type"]),
            "run_position": run_position,
        }
        for run_position, question in enumerate(selected)
    ]
    selector_binding = {
        "protocol": "python-3.12-random-sample-sorted-v1",
        "total": len(records),
        "sample": E6B_SAMPLE,
        "seed": E6B_SEED,
        "positions": list(positions),
    }
    digests = {
        "exclusion_rows_sha256": sha256_json(exclusion_rows),
        "positions_sha256": sha256_json(list(positions)),
        "selector_binding_sha256": sha256_json(selector_binding),
        "position_question_rows_sha256": sha256_json(position_question_rows),
        "question_ids_sha256": sha256_json([question.question_id for question in selected]),
        "runner_question_binding_sha256": sha256_json(binding),
        "full_run_rows_sha256": sha256_json(full_run_rows),
        "first40_run_rows_sha256": sha256_json(full_run_rows[:40]),
    }
    expected_digests = {
        "exclusion_rows_sha256": E6B_EXCLUSION_ROWS_SHA256,
        "positions_sha256": E6B_POSITIONS_SHA256,
        "selector_binding_sha256": E6B_SELECTOR_BINDING_SHA256,
        "position_question_rows_sha256": E6B_POSITION_QUESTION_ROWS_SHA256,
        "question_ids_sha256": E6B_QUESTION_IDS_SHA256,
        "runner_question_binding_sha256": E6B_RUNNER_QUESTION_BINDING_SHA256,
        "full_run_rows_sha256": E6B_FULL_RUN_ROWS_SHA256,
        "first40_run_rows_sha256": E6B_FIRST40_RUN_ROWS_SHA256,
    }
    if digests != expected_digests:
        raise ExternalE6Error("E6b selected cohort differs from frozen protocol digests")
    return {
        "method": "first-python-random-seed-satisfying-metadata-only-rule",
        "search_start_inclusive": E6B_SELECTION_SEARCH_START,
        "selected_seed": E6B_SEED,
        "sample_size": E6B_SAMPLE,
        "excluded_original_pilot_positions": sorted(E6B_ORIGINAL_PILOT_POSITIONS),
        "disjoint_from_original_pilot": True,
        "remaining_corpus_largest_remainder_type_target": E6B_TARGET_TYPE_COUNTS,
        "maximum_absolute_type_count_deviation": 1,
        "actual_type_counts": summary["type_counts"],
        "required_abs_count": E6B_ABS_COUNT,
        "actual_abs_count": summary["abs_count"],
        "canonical_source_turn_count": source_turn_count,
        "outcome_fields_used": False,
        "digests": digests,
    }


def _load_e1_context(args: argparse.Namespace) -> ExperimentContext:
    dataset = Path(args.dataset).resolve()
    corpus = compile_official_longmemeval_s(dataset)
    if corpus.source_artifact.as_dict().get("sha256") != E6B_DATASET_SHA256:
        raise ExternalE6Error("E6b dataset SHA-256 differs from preregistration")
    records = _strict_records(dataset)
    if len(records) != len(corpus.questions):
        raise ExternalE6Error("dataset records and F0 question bindings disagree")
    selected = _selected_questions(corpus, records, sample=args.sample, seed=args.seed)
    output_dir = Path(args.e1_output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = load_json(manifest_path, sealed=True)
    if manifest.get("protocol_version") != E1_PROTOCOL_VERSION:
        raise ExternalE6Error("E6 requires the completed frozen E1 protocol-v3 source")
    if manifest.get("dataset") != corpus.source_artifact.as_dict():
        raise ExternalE6Error("E1 manifest dataset differs from the E6 source")
    if manifest.get("turn_projection") != corpus.fingerprint():
        raise ExternalE6Error("E1 manifest projection differs from the E6 F0 projection")
    sample = manifest.get("sample")
    if not isinstance(sample, dict):
        raise ExternalE6Error("E1 manifest has no frozen sample")
    expected_sample = [(question.position, question.question_id) for question in selected]
    observed_questions = sample.get("questions")
    if not isinstance(observed_questions, list):
        raise ExternalE6Error("E1 manifest sample cases are malformed")
    observed_sample = [
        (row.get("position"), row.get("question_id"))
        for row in observed_questions
        if isinstance(row, dict)
    ]
    if (
        sample.get("seed") != args.seed
        or sample.get("requested") != args.sample
        or sample.get("count") != len(selected)
        or sample.get("questions_sha256") != E6B_E1_SAMPLE_BINDING_SHA256
        or observed_sample != expected_sample
    ):
        raise ExternalE6Error("E1 and E6 sample bindings differ")
    context = ExperimentContext(
        corpus=corpus,
        records=records,
        selected=selected,
        manifest=manifest,
        output_dir=output_dir,
        qwen_root=Path(args.qwen_root).resolve(),
        cross_encoder_root=Path(args.cross_encoder_root).resolve(),
        deepseek_root=Path(args.deepseek_root).resolve(),
    )
    for question in selected:
        dense_path = phase_path(context, "dense", question)
        if not dense_path.is_file():
            raise ExternalE6Error("E6 requires every selected E1 dense artifact")
        dense = load_json(dense_path, sealed=True)
        replay_dense_row(context, question, dense)
        if dense.get("dense", {}).get("runtime") != {
            "batch_size": E6B_QWEN_BATCH_SIZE,
            "device": E6B_QWEN_DEVICE,
            "dtype": E6B_QWEN_DTYPE,
            "python": E6B_PYTHON_VERSION,
            "torch": E6B_TORCH_VERSION,
            "transformers": E6B_TRANSFORMERS_VERSION,
        }:
            raise ExternalE6Error("E6b raw dense artifact runtime differs from preregistration")
    return context


def _implementation_fingerprint() -> dict[str, Any]:
    files: dict[str, str] = {}
    relatives = set(IMPLEMENTATION_FILES)
    for root_name in IMPLEMENTATION_TREE_ROOTS:
        root = REPO_ROOT / root_name
        if root.is_symlink() or not root.is_dir():
            raise ExternalE6Error(f"E6b implementation tree is missing or unsafe: {root_name}")
        relatives.update(
            str(path.relative_to(REPO_ROOT))
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    for relative in sorted(relatives):
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ExternalE6Error(f"E6 implementation file is missing or unsafe: {relative}")
        files[relative] = _sha256_file(path)
    return {"files": files, "tree_sha256": sha256_json(files)}


def _source_e6_motivation_binding() -> dict[str, Any]:
    paths = {
        "run_manifest": (
            E6B_SOURCE_E6_OUTPUT / "manifest.json",
            E6B_SOURCE_E6_MANIFEST_SHA256,
        ),
        "report": (
            E6B_SOURCE_E6_OUTPUT / "report.json",
            E6B_SOURCE_E6_REPORT_SHA256,
        ),
    }
    binding: dict[str, Any] = {}
    for name, (path, expected) in paths.items():
        artifact = load_json(path, sealed=True)
        if artifact.get("artifact_sha256") != expected:
            raise ExternalE6Error(f"source E6 v2 {name} differs from preregistered motivation")
        binding[name] = {
            "path": str(path.resolve()),
            "artifact_sha256": expected,
            "file_sha256": _sha256_file(path),
        }
    return binding


def _tokenizer_executable_sha256(e1: ExperimentContext) -> str:
    paths = {
        "encoding_dsv4.py": e1.deepseek_root / "encoding" / "encoding_dsv4.py",
        "turn_prompt/contracts.py": (
            REPO_ROOT / "benchmarks/integrations/longmemeval_turn_prompt/contracts.py"
        ),
        "turn_prompt/packer.py": (
            REPO_ROOT / "benchmarks/integrations/longmemeval_turn_prompt/packer.py"
        ),
        "representation/packing_bridge.py": (
            REPO_ROOT / "benchmarks/integrations/longmemeval_representation/packing_bridge.py"
        ),
        "run_longmemeval_e1_external.py": (REPO_ROOT / "scripts/run_longmemeval_e1_external.py"),
    }
    return sha256_json({name: _sha256_file(path) for name, path in sorted(paths.items())})


def _extractor_model_artifact_sha256() -> str:
    return sha256_json(
        {
            "model_alias": DEEPSEEK_MODEL,
            "deployment": DEEPSEEK_DEPLOYMENT_ID,
            "served_weights_immutable": False,
            "classification": "caller-attested-api-alias-not-weight-snapshot",
        }
    )


def _extractor_identity_artifact_sha256() -> str:
    return sha256_json(
        {
            "model_alias": DEEPSEEK_MODEL,
            "model_revision": EXTRACTOR_MODEL_REVISION,
            "deployment": DEEPSEEK_DEPLOYMENT_ID,
            "prompt_identity_sha256": PROMPT_IDENTITY_SHA256,
            "temperature": 0.0,
            "max_tokens": DEEPSEEK_MAX_TOKENS,
            "thinking": "disabled",
            "response_model_required": DEEPSEEK_MODEL,
            "provider_request_id_required": True,
            "identity_authentication": "caller-attested-unverified",
        }
    )


def _pricing_identity() -> DeepSeekR1PricingIdentity:
    binding = {
        "version": PRICING_VERSION,
        "cache_miss_input_usd_per_million_tokens": (DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION),
        "output_usd_per_million_tokens": DEEPSEEK_OUTPUT_USD_PER_MILLION,
        "source": "https://api-docs.deepseek.com/quick_start/pricing",
        "verified_utc_date": "2026-08-09",
    }
    return DeepSeekR1PricingIdentity(
        version=PRICING_VERSION,
        artifact_sha256=sha256_json(binding),
        cache_miss_input_microusd_per_million_tokens=int(
            DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION * 1_000_000
        ),
        output_microusd_per_million_tokens=int(DEEPSEEK_OUTPUT_USD_PER_MILLION * 1_000_000),
    )


def _e6_namespace(args: argparse.Namespace) -> Namespace:
    return Namespace(
        dataset=args.dataset,
        e1_output_dir=args.e1_output_dir,
        sample=args.sample,
        seed=args.seed,
        qwen_root=args.qwen_root,
        cross_encoder_root=args.cross_encoder_root,
        deepseek_root=args.deepseek_root,
    )


def build_e6_context(args: argparse.Namespace) -> E6Context:
    e1 = _load_e1_context(_e6_namespace(args))
    selection_preregistration = _validate_e6b_selection(e1.records, e1.selected)
    source_e6_motivation = _source_e6_motivation_binding()
    frozen_runtime = {
        "python": sys.version.split()[0],
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "dtype": "float16" if args.device in {"mps", "cuda"} else "float32",
    }
    expected_runtime = {
        "python": E6B_PYTHON_VERSION,
        "torch": E6B_TORCH_VERSION,
        "transformers": E6B_TRANSFORMERS_VERSION,
        "dtype": E6B_QWEN_DTYPE,
    }
    if (
        args.device != E6B_QWEN_DEVICE
        or args.qwen_batch_size != E6B_QWEN_BATCH_SIZE
        or frozen_runtime != expected_runtime
    ):
        raise ExternalE6Error("E6b Qwen runtime differs from the preregistered E6-v2 runtime")
    values_by_question_id = MappingProxyType(
        {
            question.question_id: compile_question_canonical_values(
                e1.corpus,
                question_id=question.question_id,
            )
            for question in e1.selected
        }
    )
    source_bytes = Path(args.dataset).resolve().read_bytes()
    snapshots = {
        "qwen_embedding": verify_snapshot(
            e1.qwen_root,
            name=QWEN_MODEL,
            revision=QWEN_REVISION,
            expected=QWEN_FILES,
        ),
        "deepseek_tokenizer": verify_snapshot(
            e1.deepseek_root,
            name="deepseek-ai/DeepSeek-V4-Flash",
            revision=DEEPSEEK_REVISION,
            expected=DEEPSEEK_FILES,
        ),
    }
    for name, snapshot in snapshots.items():
        if snapshot != e1.manifest["model_snapshots"][name]:
            raise ExternalE6Error(f"local {name} snapshot differs from frozen E1 evidence")
    tokenizer_pin = ExactTokenizerPin(
        model=DEEPSEEK_MODEL,
        revision=DEEPSEEK_REVISION,
        artifact_sha256=snapshots["deepseek_tokenizer"]["artifact_sha256"],
        executable_sha256=_tokenizer_executable_sha256(e1),
    )
    preflight = freeze_official_preflight(source_bytes, tokenizer=tokenizer_pin)
    extractor = deepseek_r1_extractor_identity(
        model_revision=EXTRACTOR_MODEL_REVISION,
        model_artifact_sha256=_extractor_model_artifact_sha256(),
        identity_artifact_sha256=_extractor_identity_artifact_sha256(),
    )
    pricing = _pricing_identity()
    dense_sources = []
    for question in e1.selected:
        dense = load_json(phase_path(e1, "dense", question), sealed=True)
        dense_sources.append(
            {
                "question_id": question.question_id,
                "question_position": question.position,
                "dense_artifact_sha256": dense["artifact_sha256"],
                "dense_observations_sha256": dense["dense"]["identity"][
                    "observation_artifact_sha256"
                ],
            }
        )
    implementation = _implementation_fingerprint()
    output_dir = Path(args.output_dir).resolve()
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e6b-head20-run-manifest",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E6_RUN_PROTOCOL_VERSION,
        "classification": "development-diagnostic-not-official-longmemeval-score",
        "production_configuration": False,
        "output_namespace": str(output_dir),
        "source_e6_v2_motivation_evidence": source_e6_motivation,
        "source_e1_manifest_sha256": e1.manifest["artifact_sha256"],
        "source_e1_dense_artifacts": dense_sources,
        "source_e1_dense_artifacts_sha256": sha256_json(dense_sources),
        "official_preflight_manifest_sha256": preflight.manifest_sha256,
        "dataset": e1.manifest["dataset"],
        "turn_projection": e1.manifest["turn_projection"],
        "sample": e1.manifest["sample"],
        "sample_selection_preregistration": selection_preregistration,
        "cells": list(E6_CELLS),
        "representation": {
            "protocol": "E6b/SB-HMR-head20-v1",
            "R0": "canonical-raw-turn-keys",
            "R1": "canonical-raw-plus-one-merged-sfk-key-per-source-turn-then-head20",
            "key_family_depth": KEY_FAMILY_DEPTH,
            "fusion": "equal-family-RRF-k60",
            "post_fusion_head_matching": {
                "applies_to": RepresentationCell.RAW_MERGED_SFK.value,
                "retained_canonical_value_count": HEAD_MATCHED_VALUE_COUNT,
                "position": "after-full-family-ranking-and-fusion-before-packing",
                "R0_head_count": HEAD_MATCHED_VALUE_COUNT,
                "outcome_or_gold_fields_used": False,
            },
            "hydration": "canonical-raw-turns-only",
        },
        "extraction": {
            "source_only": True,
            "input_fields": ["source_value"],
            "question_gold_answer_and_judge_fields_forbidden": True,
            "extractor": extractor.content_free_binding(),
            "prompt_identity_sha256": PROMPT_IDENTITY_SHA256,
            "deployment": DEEPSEEK_DEPLOYMENT_ID,
            "temperature": 0.0,
            "max_tokens": DEEPSEEK_MAX_TOKENS,
            "thinking": "disabled",
            "maximum_http_attempts_per_application_attempt": (DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS),
            "maximum_application_schema_attempts": (DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS),
            "concurrency": args.extraction_concurrency,
            "raw_invalid_attempts_retained": True,
        },
        "ranking": {
            "model": QWEN_MODEL,
            "revision": QWEN_REVISION,
            "query_instruction_sha256": sha256_bytes(QWEN_QUERY_INSTRUCTION.encode("utf-8")),
            "raw_scores": "replayed-frozen-E1-exhaustive-dense-observations",
            "merged_scores": "fresh-exhaustive-cosine-over-all-derived-keys",
            "device": args.device,
            "maximum_batch_size": args.qwen_batch_size,
            "maximum_input_tokens": QWEN_MAX_LENGTH,
            "maximum_padding_ratio": QWEN_MAX_PADDING_RATIO,
            "attention_cell_budget": QWEN_ATTENTION_CELL_BUDGET,
            "cosine_validation_tolerance": QWEN_COSINE_TOLERANCE,
            "gate_accounting_local_latency": (
                "zero-excluded-because-caller-clock-is-unauthenticated"
            ),
            "runtime": frozen_runtime,
        },
        "packing": {
            "bridge_protocol": ("swarmbrain-longmemeval-e6-representation-packing-bridge-v1"),
            "complete_reader_prompt_token_budget": TOKEN_BUDGET,
            "exact_tokenizer": tokenizer_pin.content_free_binding(),
            "prepared_run_receipt_emitted": False,
            "official_prepared_run_admission": False,
            "tokenizer_receipt_namespace": (
                "independent-per-arm-per-question-development-artifact"
            ),
            "global_tokenizer_receipt_uniqueness_claimed": False,
            "preflight_admission": {
                "manifest_freeze_executed": True,
                "prepared_run_validation_executed": False,
                "prepared_run_receipt_sha256": None,
                "official_prepared_run_claimed": False,
                "manifest_case_count": len(preflight.cases),
                "selected_development_case_count": len(e1.selected),
                "arms_per_case": len(E6_CELLS),
                "reason": (
                    "development sample and paired arms do not satisfy full "
                    "one-prompt-per-case official admission"
                ),
            },
        },
        "decision": {
            "compiler_protocol": DIAGNOSTIC_PROTOCOL_VERSION,
            "context_first": True,
            "zero_margin_gold_noninferiority": True,
            "fresh_cohort_only_for_final_inference": True,
            "paired_stratified_bootstrap": {
                "samples": E6B_BOOTSTRAP_SAMPLES,
                "seed": E6B_BOOTSTRAP_SEED,
                "strata": "question_type",
                "percentile_confidence_interval": 0.95,
                "two_sided": True,
                "required_accuracy_delta_lower_bound": "strictly-greater-than-zero",
            },
            "pareto_axes": [
                "candidate-and-prompt-any/all/recall/MRR:higher",
                "prompt-tokens:lower",
                "construction-plus-query-latency:lower",
                "construction-plus-query-cost:lower",
            ],
            "local_qwen_latency_used_for_gate": False,
            "reader_and_development_judge_only_if_context_gate_advances": True,
            "official_gpt4o_executed": False,
            "official_gpt4o_deferred_until_stable_confirmed_winner": True,
            "optional_stopping_permitted": False,
            "limited_prefix_runs_are_operational-only": True,
        },
        "reader_and_development_judge": {
            "provider": "DeepSeek API",
            "model": DEEPSEEK_MODEL,
            "deployment": DEEPSEEK_DEPLOYMENT_ID,
            "thinking": "disabled",
            "temperature": 0.0,
            "reader_max_tokens": READER_MAX_TOKENS,
            "judge_max_tokens": JUDGE_MAX_TOKENS,
            "maximum_http_attempts": QA_HTTP_ATTEMPTS,
            "http_retry_condition": "transport-or-registered-status-before-first-2xx",
            "first_2xx_is_final_for_route": True,
            "empty_or_semantically_invalid_2xx_reissued": False,
            "first_2xx_raw_body_wal_before_validation": True,
            "required_response_model": DEEPSEEK_MODEL,
            "provider_request_id_required": True,
            "arm_order": "counterbalanced-by-zero-based-selected-run-position-parity",
            "even_run_positions": list(E6_CELLS),
            "odd_run_positions": list(reversed(E6_CELLS)),
        },
        "cost_accounting": {
            "pricing": pricing.content_free_binding(),
            "pricing_source_url": "https://api-docs.deepseek.com/quick_start/pricing",
            "pricing_verified_utc_date": "2026-08-09",
            "hard_external_cost_limit_microusd": MAX_EXTERNAL_COST_MICROUSD,
            "hard_external_cost_limit_usd": MAX_EXTERNAL_COST_MICROUSD / 1_000_000,
            "planning_estimate_usd": 4.858992,
            "indicative_exact-turn-scaled_estimate_usd": 4.594,
            "expected_upper_range_usd": 5.19,
            "planning_values_are_not_caps": True,
            "authorization": "operator-authorized-DeepSeek-development-spend-2026-08-09",
            "every_application_attempt_counted": True,
            "successful_attempt_usage": "provider-reported",
            "unseen_prior_http_attempt": "prompt-tokens-plus-request-max-output",
            "billed_cost_claimed": False,
        },
        "model_snapshots": snapshots,
        "implementation": implementation,
        "local_replay_cache": {
            "semantic_role": "immutable-local-replay-cache-only",
            "projection_digest": "computed-once-after-complete-corpus-validation",
            "selected_canonical_values": "precomputed-before-phase-timers",
            "external_request_bytes_routes_or_order_changed": False,
            "gate_accounting_changed": False,
        },
        "claims": {
            "gold_fields_used_for_extraction_or_ranking": False,
            "sample_is_held_out": False,
            "official_longmemeval_score": False,
            "official_gpt4o_judge_executed": False,
            "external_api_alias_is_an_immutable_weight_snapshot": False,
            "eligible_for_composition": False,
            "production_policy_changed": False,
        },
    }
    manifest = seal_artifact(payload)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if load_json(manifest_path, sealed=True) != manifest:
            raise ExternalE6Error("E6 output directory belongs to a different frozen manifest")
    else:
        write_json(manifest_path, manifest)
    return E6Context(
        e1=e1,
        source_bytes=source_bytes,
        preflight=preflight,
        output_dir=output_dir,
        manifest=manifest,
        extractor=extractor,
        pricing=pricing,
        values_by_question_id=values_by_question_id,
    )


def e6_phase_path(context: E6Context, phase: str, question: SelectedQuestion) -> Path:
    name = f"{question.position:03d}-{_safe_question_id(question.question_id)}.json"
    return context.output_dir / phase / name


def e6_jsonl_path(context: E6Context, phase: str, question: SelectedQuestion) -> Path:
    name = f"{question.position:03d}-{_safe_question_id(question.question_id)}.jsonl"
    return context.output_dir / phase / name


def _selected_prefix(context: E6Context, limit: int | None) -> tuple[SelectedQuestion, ...]:
    if limit is None:
        return context.e1.selected
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ExternalE6Error("execution limit must be a positive integer")
    return context.e1.selected[:limit]


def _source_values(
    context: E6Context,
    question: SelectedQuestion,
) -> tuple[CanonicalValue, ...]:
    values = context.values_by_question_id.get(question.question_id)
    if values is None:
        raise ExternalE6Error("selected question has no cached canonical values")
    return values


def _value_record_path(
    context: E6Context,
    question: SelectedQuestion,
    source_position: int,
) -> Path:
    return (
        context.output_dir
        / "extraction-values"
        / f"{question.position:03d}-{_safe_question_id(question.question_id)}"
        / f"{source_position:05d}.json"
    )


def _value_attempt_path(
    context: E6Context,
    question: SelectedQuestion,
    source_position: int,
) -> Path:
    return _value_record_path(context, question, source_position).with_suffix(".attempts.jsonl")


@dataclass(frozen=True, slots=True)
class _CallJournalPaths:
    reservation: Path
    response: Path
    settlement: Path


def _call_journal_paths(
    context: E6Context,
    *,
    namespace: str,
    route: str,
) -> _CallJournalPaths:
    if namespace not in {"extraction", "qa"}:
        raise ExternalE6Error("external call journal namespace is not registered")
    digest = sha256_bytes(route.encode("utf-8"))
    root = context.output_dir / "external-call-journal" / namespace
    stem = root / digest
    return _CallJournalPaths(
        reservation=stem.with_suffix(".reservation.json"),
        response=stem.with_suffix(".response.json"),
        settlement=stem.with_suffix(".settlement.json"),
    )


def _reservation_artifact(
    context: E6Context,
    *,
    namespace: str,
    route: str,
    raw_request_sha256: str,
    exact_prompt_tokens: int,
    request_max_tokens: int,
    reserved_microusd: int,
) -> dict[str, Any]:
    if (
        not isinstance(route, str)
        or not route
        or route != route.strip()
        or isinstance(exact_prompt_tokens, bool)
        or not isinstance(exact_prompt_tokens, int)
        or exact_prompt_tokens < 1
        or isinstance(request_max_tokens, bool)
        or not isinstance(request_max_tokens, int)
        or request_max_tokens < 1
        or isinstance(reserved_microusd, bool)
        or not isinstance(reserved_microusd, int)
        or reserved_microusd < 1
    ):
        raise ExternalE6Error("external call reservation inputs are malformed")
    expected_reserved = context.pricing.upper_bound_microusd(
        input_tokens=exact_prompt_tokens,
        output_tokens=request_max_tokens,
        retry_count=DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS - 1,
        request_max_tokens=request_max_tokens,
    )
    if reserved_microusd != expected_reserved:
        raise ExternalE6Error("external call reservation differs from frozen worst-case pricing")
    if (
        not isinstance(raw_request_sha256, str)
        or len(raw_request_sha256) != 64
        or any(character not in "0123456789abcdef" for character in raw_request_sha256)
    ):
        raise ExternalE6Error("external call reservation request digest is malformed")
    return seal_artifact(
        {
            "artifact_type": "swarmbrain-longmemeval-e6-external-call-reservation",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "protocol_version": E6_RUN_PROTOCOL_VERSION,
            "run_manifest_sha256": context.manifest["artifact_sha256"],
            "namespace": namespace,
            "route": route,
            "raw_request_sha256": raw_request_sha256,
            "exact_local_prompt_tokens": exact_prompt_tokens,
            "request_max_tokens": request_max_tokens,
            "maximum_http_attempts": DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS,
            "reserved_microusd": reserved_microusd,
            "state": "durable-pre-call-worst-case-reservation",
        }
    )


def _write_reservation(
    context: E6Context,
    paths: _CallJournalPaths,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    _ensure_durable_journal_parent(context, paths)
    if paths.reservation.is_symlink():
        raise ExternalE6Error("external call reservation cannot be a symlink")
    expected = dict(artifact)
    if paths.reservation.exists():
        existing = load_json(paths.reservation, sealed=True)
        if existing != expected:
            raise ExternalE6Error("external call reservation journal drifted")
        return existing
    _durable_write_json(paths.reservation, expected)
    return expected


def _raw_journal_block(
    raw: bytes,
    *,
    encoding: str,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if not isinstance(raw, bytes) or (not raw and not allow_empty):
        raise ExternalE6Error("external call journal raw byte boundary is invalid")
    return {
        "encoding": encoding,
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "raw_base64": base64.b64encode(raw).decode("ascii"),
    }


def _raw_call_response_artifact(
    context: E6Context,
    *,
    reservation: Mapping[str, Any],
    raw_request: bytes,
    raw_response: bytes,
    endpoint_url: str,
    attempts: int,
    latency_ms: float,
    content_encoding: str,
) -> dict[str, Any]:
    normalized_encoding = str(content_encoding).strip().casefold()
    if normalized_encoding in {"", "identity"}:
        normalized_encoding = "identity"
    return seal_artifact(
        {
            "artifact_type": "swarmbrain-longmemeval-e6-external-call-response-wal",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "protocol_version": E6_RUN_PROTOCOL_VERSION,
            "run_manifest_sha256": context.manifest["artifact_sha256"],
            "reservation_artifact_sha256": reservation["artifact_sha256"],
            "namespace": reservation["namespace"],
            "route": reservation["route"],
            "provider_request": _raw_journal_block(
                raw_request,
                encoding="base64-exact-http-request-body",
            ),
            "provider_response": _raw_journal_block(
                raw_response,
                encoding="base64-exact-decoded-http-body",
                allow_empty=True,
            ),
            "transport": {
                "endpoint_url": endpoint_url,
                "attempts": attempts,
                "latency_ms": latency_ms,
                "content_encoding": normalized_encoding,
                "latency_source": "caller-observed-monotonic-clock",
            },
            "state": "durable-first-2xx-provider-response",
        }
    )


def _call_response_artifact(
    context: E6Context,
    *,
    reservation: Mapping[str, Any],
    result: ChatResult,
) -> dict[str, Any]:
    return _raw_call_response_artifact(
        context,
        reservation=reservation,
        raw_request=result.raw_request,
        raw_response=result.raw_response,
        endpoint_url=result.endpoint_url,
        attempts=result.attempts,
        latency_ms=result.latency_ms,
        content_encoding="identity",
    )


def _write_call_response(
    paths: _CallJournalPaths,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    if paths.response.is_symlink():
        raise ExternalE6Error("external call response WAL cannot be a symlink")
    expected = dict(artifact)
    if paths.response.exists():
        existing = load_json(paths.response, sealed=True)
        if existing != expected:
            raise ExternalE6Error("external call response WAL drifted")
        return existing
    _durable_write_json(paths.response, expected)
    return expected


def _decode_journal_block(
    value: Any,
    *,
    label: str,
    encoding: str,
    allow_empty: bool = False,
) -> bytes:
    if not isinstance(value, dict) or set(value) != {
        "encoding",
        "raw_bytes",
        "raw_sha256",
        "raw_base64",
    }:
        raise ExternalE6Error(f"{label} fields differ from the frozen schema")
    if value.get("encoding") != encoding:
        raise ExternalE6Error(f"{label} encoding drifted")
    encoded = value.get("raw_base64")
    if not isinstance(encoded, str) or (not encoded and not allow_empty):
        raise ExternalE6Error(f"{label} has no raw base64 bytes")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ExternalE6Error(f"{label} contains invalid base64") from exc
    if value.get("raw_bytes") != len(raw) or value.get("raw_sha256") != sha256_bytes(raw):
        raise ExternalE6Error(f"{label} raw byte binding drifted")
    return raw


def _replay_call_response(
    context: E6Context,
    paths: _CallJournalPaths,
    *,
    reservation: Mapping[str, Any],
) -> tuple[dict[str, Any], ChatResult]:
    response = load_json(paths.response, sealed=True)
    if set(response) != {
        "artifact_type",
        "schema_version",
        "protocol_version",
        "run_manifest_sha256",
        "reservation_artifact_sha256",
        "namespace",
        "route",
        "provider_request",
        "provider_response",
        "transport",
        "state",
        "artifact_sha256",
    }:
        raise ExternalE6Error("external call response WAL fields differ from the schema")
    if (
        response.get("artifact_type") != "swarmbrain-longmemeval-e6-external-call-response-wal"
        or response.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or response.get("protocol_version") != E6_RUN_PROTOCOL_VERSION
        or response.get("run_manifest_sha256") != context.manifest["artifact_sha256"]
        or response.get("reservation_artifact_sha256") != reservation.get("artifact_sha256")
        or response.get("namespace") != reservation.get("namespace")
        or response.get("route") != reservation.get("route")
        or response.get("state") != "durable-first-2xx-provider-response"
    ):
        raise ExternalE6Error("external call response WAL binding drifted")
    raw_request = _decode_journal_block(
        response.get("provider_request"),
        label="external call provider request",
        encoding="base64-exact-http-request-body",
    )
    raw_response = _decode_journal_block(
        response.get("provider_response"),
        label="external call provider response",
        encoding="base64-exact-decoded-http-body",
        allow_empty=True,
    )
    if sha256_bytes(raw_request) != reservation.get("raw_request_sha256"):
        raise ExternalE6Error("external call response request differs from its reservation")
    request = replay_chat_request(raw_request)
    if request.max_tokens != reservation.get("request_max_tokens"):
        raise ExternalE6Error("external call response max_tokens differs from its reservation")
    transport = response.get("transport")
    if not isinstance(transport, dict) or set(transport) != {
        "endpoint_url",
        "attempts",
        "latency_ms",
        "content_encoding",
        "latency_source",
    }:
        raise ExternalE6Error("external call response transport evidence is malformed")
    if transport.get("latency_source") != "caller-observed-monotonic-clock":
        raise ExternalE6Error("external call latency source drifted")
    if str(transport.get("content_encoding", "")).strip().casefold() not in {
        "",
        "identity",
    }:
        raise ExternalE6Error("external call response content encoding is unsupported")
    result = chat_result_from_raw_response(
        raw_response,
        prompt=request.prompt,
        attempts=transport.get("attempts"),
        latency_ms=transport.get("latency_ms"),
        raw_request=raw_request,
        endpoint_url=transport.get("endpoint_url"),
    )
    if not 1 <= result.attempts <= reservation.get("maximum_http_attempts"):
        raise ExternalE6Error("external call response HTTP attempt count is outside its bound")
    return response, result


def _settlement_artifact(
    context: E6Context,
    *,
    reservation: Mapping[str, Any],
    evidence_sha256: str,
    actual_microusd: int,
) -> dict[str, Any]:
    reserved = reservation.get("reserved_microusd")
    if (
        isinstance(reserved, bool)
        or not isinstance(reserved, int)
        or isinstance(actual_microusd, bool)
        or not isinstance(actual_microusd, int)
        or not 0 <= actual_microusd <= reserved
    ):
        raise ExternalE6Error("external call settlement exceeds its reservation")
    return seal_artifact(
        {
            "artifact_type": "swarmbrain-longmemeval-e6-external-call-settlement",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "protocol_version": E6_RUN_PROTOCOL_VERSION,
            "run_manifest_sha256": context.manifest["artifact_sha256"],
            "reservation_artifact_sha256": reservation["artifact_sha256"],
            "namespace": reservation["namespace"],
            "route": reservation["route"],
            "evidence_sha256": evidence_sha256,
            "actual_microusd": actual_microusd,
            "state": "settled-from-durable-provider-evidence",
        }
    )


def _write_settlement(
    paths: _CallJournalPaths,
    settlement: Mapping[str, Any],
) -> dict[str, Any]:
    if paths.settlement.is_symlink():
        raise ExternalE6Error("external call settlement cannot be a symlink")
    expected = dict(settlement)
    if paths.settlement.exists():
        existing = load_json(paths.settlement, sealed=True)
        if existing != expected:
            raise ExternalE6Error("external call settlement journal drifted")
        return existing
    _durable_write_json(paths.settlement, expected)
    return expected


def _external_journal_cost(context: E6Context) -> tuple[int, int]:
    root = context.output_dir / "external-call-journal"
    if root.is_symlink():
        raise ExternalE6Error("external call journal root is unsafe")
    if not root.exists():
        return (0, 0)
    if not root.is_dir():
        raise ExternalE6Error("external call journal root is unsafe")
    for namespace_path in root.iterdir():
        if namespace_path.is_symlink() or not namespace_path.is_dir():
            raise ExternalE6Error("external call journal namespace path is unsafe")
        if namespace_path.name not in {"extraction", "qa"}:
            raise ExternalE6Error("external call journal contains an unknown namespace")
    reservation_paths = sorted(root.glob("*/*.reservation.json"))
    reservation_stems = {
        path.with_name(path.name.replace(".reservation.json", "")) for path in reservation_paths
    }
    for suffix in (".response.json", ".settlement.json"):
        for path in sorted(root.glob(f"*/*{suffix}")):
            if path.is_symlink():
                raise ExternalE6Error("external call journal artifact cannot be a symlink")
            stem = path.with_name(path.name.removesuffix(suffix))
            if stem not in reservation_stems:
                raise ExternalE6Error("external call journal contains an orphan artifact")
    total = 0
    unresolved = 0
    for reservation_path in reservation_paths:
        if reservation_path.is_symlink():
            raise ExternalE6Error("external call reservation path cannot be a symlink")
        reservation = load_json(reservation_path, sealed=True)
        if set(reservation) != {
            "artifact_type",
            "schema_version",
            "protocol_version",
            "run_manifest_sha256",
            "namespace",
            "route",
            "raw_request_sha256",
            "exact_local_prompt_tokens",
            "request_max_tokens",
            "maximum_http_attempts",
            "reserved_microusd",
            "state",
            "artifact_sha256",
        }:
            raise ExternalE6Error("external call reservation fields differ from the schema")
        if (
            reservation.get("artifact_type")
            != "swarmbrain-longmemeval-e6-external-call-reservation"
            or reservation.get("schema_version") != ARTIFACT_SCHEMA_VERSION
            or reservation.get("protocol_version") != E6_RUN_PROTOCOL_VERSION
            or reservation.get("run_manifest_sha256") != context.manifest["artifact_sha256"]
            or reservation.get("namespace") != reservation_path.parent.name
            or reservation.get("state") != "durable-pre-call-worst-case-reservation"
        ):
            raise ExternalE6Error("external call reservation journal binding drifted")
        expected_paths = _call_journal_paths(
            context,
            namespace=str(reservation["namespace"]),
            route=str(reservation["route"]),
        )
        if expected_paths.reservation != reservation_path:
            raise ExternalE6Error("external call reservation filename differs from its route")
        expected_reserved = context.pricing.upper_bound_microusd(
            input_tokens=reservation.get("exact_local_prompt_tokens"),
            output_tokens=reservation.get("request_max_tokens"),
            retry_count=DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS - 1,
            request_max_tokens=reservation.get("request_max_tokens"),
        )
        if (
            reservation.get("maximum_http_attempts") != DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS
            or reservation.get("reserved_microusd") != expected_reserved
        ):
            raise ExternalE6Error("external call reservation amount or retry bound drifted")
        paths = _CallJournalPaths(
            reservation=reservation_path,
            response=reservation_path.with_name(
                reservation_path.name.replace(".reservation.json", ".response.json")
            ),
            settlement=reservation_path.with_name(
                reservation_path.name.replace(".reservation.json", ".settlement.json")
            ),
        )
        reserved = reservation.get("reserved_microusd")
        if isinstance(reserved, bool) or not isinstance(reserved, int) or reserved < 0:
            raise ExternalE6Error("external call reservation cost is malformed")
        if paths.settlement.exists():
            if not paths.response.exists():
                raise ExternalE6Error("settled external call has no retained response WAL")
            response, result = _replay_call_response(
                context,
                paths,
                reservation=reservation,
            )
            settlement = load_json(paths.settlement, sealed=True)
            if set(settlement) != {
                "artifact_type",
                "schema_version",
                "protocol_version",
                "run_manifest_sha256",
                "reservation_artifact_sha256",
                "namespace",
                "route",
                "evidence_sha256",
                "actual_microusd",
                "state",
                "artifact_sha256",
            }:
                raise ExternalE6Error("external call settlement fields differ from the schema")
            if (
                settlement.get("artifact_type")
                != "swarmbrain-longmemeval-e6-external-call-settlement"
                or settlement.get("schema_version") != ARTIFACT_SCHEMA_VERSION
                or settlement.get("protocol_version") != E6_RUN_PROTOCOL_VERSION
                or settlement.get("run_manifest_sha256") != context.manifest["artifact_sha256"]
                or settlement.get("reservation_artifact_sha256") != reservation["artifact_sha256"]
                or settlement.get("namespace") != reservation.get("namespace")
                or settlement.get("route") != reservation.get("route")
                or settlement.get("evidence_sha256") != response["artifact_sha256"]
                or settlement.get("state") != "settled-from-durable-provider-evidence"
            ):
                raise ExternalE6Error("external call settlement binding drifted")
            actual = settlement.get("actual_microusd")
            if (
                isinstance(actual, bool)
                or not isinstance(actual, int)
                or not 0 <= actual <= reserved
            ):
                raise ExternalE6Error("external call settled cost is malformed")
            if actual != _chat_cost_microusd(result, pricing=context.pricing):
                raise ExternalE6Error("external call settled cost differs from raw replay")
            total += actual
        else:
            if paths.response.exists():
                try:
                    _, result = _replay_call_response(
                        context,
                        paths,
                        reservation=reservation,
                    )
                except ChatProtocolError:
                    # The exact first-2xx bytes remain durable, but a malformed
                    # provider envelope cannot become domain evidence.  Count
                    # the full reservation and leave the route terminally
                    # unresolved; never reissue it or admit a normal report.
                    result = None
                if (
                    result is not None
                    and _chat_cost_microusd(result, pricing=context.pricing) > reserved
                ):
                    raise ExternalE6Error("pending external response exceeds its reservation")
            total += reserved
            unresolved += 1
    if total > MAX_EXTERNAL_COST_MICROUSD:
        raise ExternalCostCapExceeded("durable external call journal exceeds the $5.60 hard cap")
    return total, unresolved


def _load_attempt_ledger(
    path: Path,
    *,
    source: Any,
    extractor: Any,
) -> tuple[list[dict[str, Any]], list[DeepSeekR1ProviderAttempt]]:
    if not path.exists():
        return [], []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExternalE6Error(f"cannot read E6 attempt artifact {path}") from exc
    try:
        attempts = replay_deepseek_r1_attempt_jsonl(
            raw,
            source=source,
            extractor=extractor,
        )
    except RepresentationEvidenceError as exc:
        raise ExternalE6Error(f"E6 attempt artifact {path} failed exact replay: {exc}") from exc
    records = [json.loads(line) for line in raw.split(b"\n")[:-1]]
    return records, list(attempts)


def _decode_attempt_block(value: Any, *, label: str) -> bytes:
    if not isinstance(value, dict):
        raise ExternalE6Error(f"{label} block is malformed")
    encoded = value.get("raw_base64")
    if not isinstance(encoded, str):
        raise ExternalE6Error(f"{label} block has no base64 bytes")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ExternalE6Error(f"{label} block is invalid base64") from exc
    if value.get("raw_bytes") != len(raw) or value.get("raw_sha256") != sha256_bytes(raw):
        raise ExternalE6Error(f"{label} byte binding drifted")
    return raw


def _attempt_cost_microusd(
    record: Mapping[str, Any],
    *,
    pricing: DeepSeekR1PricingIdentity,
) -> int:
    response = record.get("response")
    if not isinstance(response, dict):
        raise ExternalE6Error("E6 extraction attempt has no response accounting")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ExternalE6Error("E6 extraction attempt has no provider usage")
    return pricing.upper_bound_microusd(
        input_tokens=int(usage["prompt_tokens"]),
        output_tokens=int(usage["completion_tokens"]),
        retry_count=int(record["http_attempts"]) - 1,
        request_max_tokens=DEEPSEEK_MAX_TOKENS,
    )


def _evidence_for_value(
    context: E6Context,
    question: SelectedQuestion,
    source_position: int,
    source: Any,
) -> DeepSeekR1ExtractionEvidence | None:
    path = _value_record_path(context, question, source_position)
    if not path.exists():
        return None
    record = load_json(path)
    evidence = replay_deepseek_r1_evidence_record(
        record,
        source=source,
        extractor=context.extractor,
        pricing=context.pricing,
    )
    attempt_rows, _ = _load_attempt_ledger(
        _value_attempt_path(context, question, source_position),
        source=source,
        extractor=context.extractor,
    )
    if attempt_rows != record.get("application_attempts"):
        raise ExternalE6Error("final E6 evidence differs from its durable attempt ledger")
    return evidence


@dataclass(slots=True)
class _SpendLedger:
    spent_microusd: int
    maximum_microusd: int
    reserved_microusd: int = 0
    lock: asyncio.Lock | None = None

    def __post_init__(self) -> None:
        if self.spent_microusd < 0 or self.maximum_microusd < 1:
            raise ExternalE6Error("external spend ledger bounds are invalid")
        self.lock = asyncio.Lock()

    async def reserve(self, amount: int) -> None:
        assert self.lock is not None
        async with self.lock:
            if self.spent_microusd + self.reserved_microusd + amount > self.maximum_microusd:
                raise ExternalCostCapExceeded(
                    "E6b conservative external spend reservation would exceed the $5.60 cap"
                )
            self.reserved_microusd += amount

    async def settle(self, *, reserved: int, actual: int) -> None:
        assert self.lock is not None
        async with self.lock:
            if reserved > self.reserved_microusd or actual < 0:
                raise ExternalE6Error("external spend ledger settlement is inconsistent")
            self.reserved_microusd -= reserved
            self.spent_microusd += actual
            if self.spent_microusd > self.maximum_microusd:
                raise ExternalCostCapExceeded(
                    "E6b conservative external spend exceeded its hard cap"
                )

    async def release(self, amount: int) -> None:
        assert self.lock is not None
        async with self.lock:
            if amount > self.reserved_microusd:
                raise ExternalE6Error("external spend ledger release is inconsistent")
            self.reserved_microusd -= amount

    async def charge_failed_reservation(self, amount: int) -> None:
        assert self.lock is not None
        async with self.lock:
            if amount > self.reserved_microusd:
                raise ExternalE6Error("failed-call spend reservation is inconsistent")
            self.reserved_microusd -= amount
            self.spent_microusd += amount


def _maximum_call_reservation(
    *,
    exact_prompt_tokens: int,
    pricing: DeepSeekR1PricingIdentity,
) -> int:
    # Reserve a maximum-length successful response and all three unseen HTTP
    # retries.  Settlement uses retained provider usage if the call succeeds.
    return pricing.upper_bound_microusd(
        input_tokens=exact_prompt_tokens,
        output_tokens=DEEPSEEK_MAX_TOKENS,
        retry_count=DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS - 1,
        request_max_tokens=DEEPSEEK_MAX_TOKENS,
    )


def _extraction_journal_route(
    question: SelectedQuestion,
    source_position: int,
    application_attempt: int,
) -> str:
    return (
        f"extract/{question.position:03d}/{_safe_question_id(question.question_id)}/"
        f"source/{source_position:05d}/application/{application_attempt}"
    )


def _expected_extraction_reservation(
    context: E6Context,
    question: SelectedQuestion,
    source_position: int,
    application_attempt: int,
    *,
    expected_request: bytes,
    exact_prompt_tokens: int,
    reserved_microusd: int,
) -> tuple[_CallJournalPaths, dict[str, Any]]:
    route = _extraction_journal_route(question, source_position, application_attempt)
    paths = _call_journal_paths(context, namespace="extraction", route=route)
    artifact = _reservation_artifact(
        context,
        namespace="extraction",
        route=route,
        raw_request_sha256=sha256_bytes(expected_request),
        exact_prompt_tokens=exact_prompt_tokens,
        request_max_tokens=DEEPSEEK_MAX_TOKENS,
        reserved_microusd=reserved_microusd,
    )
    return paths, artifact


def _validate_extraction_result(
    context: E6Context,
    question: SelectedQuestion,
    source_position: int,
    application_attempt: int,
    *,
    source: Any,
    result: ChatResult,
    expected_request: bytes,
    exact_prompt_tokens: int,
) -> tuple[DeepSeekR1ProviderAttempt, dict[str, Any], int]:
    if result.raw_request != expected_request:
        raise ExternalE6Error("DeepSeek call request bytes differ from source-only E6 bytes")
    if (
        result.endpoint_url != DEEPSEEK_DEPLOYMENT_ID
        or result.response_model != DEEPSEEK_MODEL
        or not result.request_id
        or not 1 <= result.attempts <= DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS
    ):
        raise ExternalE6Error("DeepSeek extraction provider response configuration drifted")
    verify_provider_prompt_tokens(
        result,
        expected=exact_prompt_tokens,
        label=f"{question.question_id}/source-{source_position}/application-{application_attempt}",
    )
    attempt = DeepSeekR1ProviderAttempt(
        raw_request=result.raw_request,
        raw_response=result.raw_response,
        http_attempts=result.attempts,
        latency_microseconds=int(math.ceil(result.latency_ms * 1000.0)),
    )
    attempt_record = build_deepseek_r1_attempt_record(
        source=source,
        extractor=context.extractor,
        attempt=attempt,
        application_attempt=application_attempt,
    )
    actual_cost = _attempt_cost_microusd(attempt_record, pricing=context.pricing)
    return attempt, attempt_record, actual_cost


def _persist_extraction_response(
    context: E6Context,
    question: SelectedQuestion,
    source_position: int,
    application_attempt: int,
    *,
    source: Any,
    result: ChatResult,
    expected_request: bytes,
    exact_prompt_tokens: int,
    paths: _CallJournalPaths,
    reservation_artifact: Mapping[str, Any],
    response_artifact: Mapping[str, Any],
    attempt_records: list[dict[str, Any]],
    attempts: list[DeepSeekR1ProviderAttempt],
) -> int:
    attempt, attempt_record, actual_cost = _validate_extraction_result(
        context,
        question,
        source_position,
        application_attempt,
        source=source,
        result=result,
        expected_request=expected_request,
        exact_prompt_tokens=exact_prompt_tokens,
    )
    if application_attempt <= len(attempt_records):
        if attempt_records[application_attempt - 1] != attempt_record:
            raise ExternalE6Error("extraction response WAL differs from its attempt ledger")
        if attempts[application_attempt - 1] != attempt:
            raise ExternalE6Error("extraction response replay differs from its attempt object")
    elif application_attempt == len(attempt_records) + 1:
        attempt_records.append(attempt_record)
        attempts.append(attempt)
        _durable_atomic_write(
            _value_attempt_path(context, question, source_position),
            deepseek_r1_attempt_jsonl_bytes(attempt_records),
        )
    else:
        raise ExternalE6Error("extraction response WAL skips an application attempt")
    settlement = _settlement_artifact(
        context,
        reservation=reservation_artifact,
        evidence_sha256=response_artifact["artifact_sha256"],
        actual_microusd=actual_cost,
    )
    _write_settlement(paths, settlement)
    return actual_cost


def _reconcile_extraction_value(
    context: E6Context,
    question: SelectedQuestion,
    source_position: int,
    source: Any,
    *,
    tokenizer: DeepSeekExactTokenizer,
) -> None:
    attempt_path = _value_attempt_path(context, question, source_position)
    attempt_records, attempts = _load_attempt_ledger(
        attempt_path,
        source=source,
        extractor=context.extractor,
    )
    expected_request = deepseek_r1_request_bytes(source, context.extractor)
    request = json.loads(expected_request)
    prompt = str(request["messages"][0]["content"])
    exact_prompt_tokens = tokenizer.exact_count(prompt)
    reserved = _maximum_call_reservation(
        exact_prompt_tokens=exact_prompt_tokens,
        pricing=context.pricing,
    )
    accepted_seen = False
    for application_attempt in range(1, DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS + 1):
        paths, expected_reservation = _expected_extraction_reservation(
            context,
            question,
            source_position,
            application_attempt,
            expected_request=expected_request,
            exact_prompt_tokens=exact_prompt_tokens,
            reserved_microusd=reserved,
        )
        has_any = paths.reservation.exists() or paths.response.exists() or paths.settlement.exists()
        if accepted_seen and has_any:
            raise ExternalE6Error("extraction journal continued after a schema-valid response")
        if not paths.reservation.exists():
            if paths.response.exists() or paths.settlement.exists():
                raise ExternalE6Error(
                    "extraction journal contains an orphan response or settlement"
                )
            if application_attempt <= len(attempt_records):
                raise ExternalE6Error("extraction attempt ledger has no durable call reservation")
            continue
        reservation_artifact = load_json(paths.reservation, sealed=True)
        if reservation_artifact != expected_reservation:
            raise ExternalE6Error("extraction call reservation differs from frozen routing")
        if not paths.response.exists():
            if paths.settlement.exists():
                raise ExternalE6Error("extraction settlement has no provider response WAL")
            raise ExternalE6Error(
                "unresolved extraction reservation has no response; refusing to reissue"
            )
        response_artifact, result = _replay_call_response(
            context,
            paths,
            reservation=reservation_artifact,
        )
        _persist_extraction_response(
            context,
            question,
            source_position,
            application_attempt,
            source=source,
            result=result,
            expected_request=expected_request,
            exact_prompt_tokens=exact_prompt_tokens,
            paths=paths,
            reservation_artifact=reservation_artifact,
            response_artifact=response_artifact,
            attempt_records=attempt_records,
            attempts=attempts,
        )
        accepted_seen = bool(
            attempt_records[application_attempt - 1]["output_validation"]["accepted"]
        )
    if len(attempt_records) != len(attempts):
        raise ExternalE6Error("extraction attempt ledger replay count drifted")
    if any(bool(row["output_validation"]["accepted"]) for row in attempt_records):
        final = build_deepseek_r1_evidence_record(
            source=source,
            extractor=context.extractor,
            pricing=context.pricing,
            application_attempts=attempts,
        )
        final_path = _value_record_path(context, question, source_position)
        if final_path.exists():
            if load_json(final_path) != final:
                raise ExternalE6Error("saved extraction evidence differs from journal recovery")
        else:
            write_json(final_path, final)
    elif _value_record_path(context, question, source_position).exists():
        raise ExternalE6Error("final extraction evidence exists without an accepted attempt")


def _reconcile_extraction_journals(
    context: E6Context,
    selected: Sequence[SelectedQuestion],
    *,
    tokenizer: DeepSeekExactTokenizer,
) -> None:
    for question in selected:
        for source_position, source in enumerate(_source_values(context, question)):
            _reconcile_extraction_value(
                context,
                question,
                source_position,
                source,
                tokenizer=tokenizer,
            )


async def _extract_one_value(
    context: E6Context,
    question: SelectedQuestion,
    source_position: int,
    source: Any,
    *,
    client: _ExtractionChatClient,
    tokenizer: DeepSeekExactTokenizer,
    ledger: _SpendLedger,
    semaphore: asyncio.Semaphore,
) -> tuple[dict[str, Any], int]:
    final_path = _value_record_path(context, question, source_position)
    attempt_path = _value_attempt_path(context, question, source_position)
    existing_evidence = _evidence_for_value(context, question, source_position, source)
    if existing_evidence is not None:
        return load_json(final_path), existing_evidence.construction_receipt.cost_microusd

    attempt_records, attempts = _load_attempt_ledger(
        attempt_path,
        source=source,
        extractor=context.extractor,
    )
    if any(bool(record["output_validation"]["accepted"]) for record in attempt_records):
        if not bool(attempt_records[-1]["output_validation"]["accepted"]):
            raise ExternalE6Error("E6 attempt ledger continued after an accepted output")
        final = build_deepseek_r1_evidence_record(
            source=source,
            extractor=context.extractor,
            pricing=context.pricing,
            application_attempts=attempts,
        )
        write_json(final_path, final)
        evidence = replay_deepseek_r1_evidence_record(
            final,
            source=source,
            extractor=context.extractor,
            pricing=context.pricing,
        )
        return final, evidence.construction_receipt.cost_microusd

    expected_request = deepseek_r1_request_bytes(source, context.extractor)
    request = json.loads(expected_request)
    prompt = str(request["messages"][0]["content"])
    exact_prompt_tokens = tokenizer.exact_count(prompt)
    reservation = _maximum_call_reservation(
        exact_prompt_tokens=exact_prompt_tokens,
        pricing=context.pricing,
    )
    for application_attempt in range(
        len(attempts) + 1,
        DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS + 1,
    ):
        paths, reservation_artifact = _expected_extraction_reservation(
            context,
            question,
            source_position,
            application_attempt,
            expected_request=expected_request,
            exact_prompt_tokens=exact_prompt_tokens,
            reserved_microusd=reservation,
        )
        async with semaphore:
            if paths.reservation.exists() or paths.response.exists() or paths.settlement.exists():
                raise ExternalE6Error(
                    "live extraction route already has durable journal state; reconciliation failed"
                )
            await ledger.reserve(reservation)
            try:
                _write_reservation(context, paths, reservation_artifact)
            except Exception:
                await ledger.release(reservation)
                raise
            try:

                def checkpoint_first_2xx(
                    raw_response: bytes,
                    http_attempts: int,
                    latency_ms: float,
                    content_encoding: str,
                    journal_paths: _CallJournalPaths = paths,
                    journal_reservation: Mapping[str, Any] = reservation_artifact,
                ) -> None:
                    _write_call_response(
                        journal_paths,
                        _raw_call_response_artifact(
                            context,
                            reservation=journal_reservation,
                            raw_request=expected_request,
                            raw_response=raw_response,
                            endpoint_url=DEEPSEEK_DEPLOYMENT_ID,
                            attempts=http_attempts,
                            latency_ms=latency_ms,
                            content_encoding=content_encoding,
                        ),
                    )

                result = await client.complete(
                    prompt,
                    raw_request=expected_request,
                    on_first_2xx=checkpoint_first_2xx,
                )
            except Exception:
                await ledger.charge_failed_reservation(reservation)
                raise
            try:
                response_artifact = _call_response_artifact(
                    context,
                    reservation=reservation_artifact,
                    result=result,
                )
                _write_call_response(paths, response_artifact)
                actual_cost = _persist_extraction_response(
                    context,
                    question,
                    source_position,
                    application_attempt,
                    source=source,
                    result=result,
                    expected_request=expected_request,
                    exact_prompt_tokens=exact_prompt_tokens,
                    paths=paths,
                    reservation_artifact=reservation_artifact,
                    response_artifact=response_artifact,
                    attempt_records=attempt_records,
                    attempts=attempts,
                )
            except Exception:
                await ledger.charge_failed_reservation(reservation)
                raise
            await ledger.settle(reserved=reservation, actual=actual_cost)
        attempt_record = attempt_records[-1]
        if bool(attempt_record["output_validation"]["accepted"]):
            final = build_deepseek_r1_evidence_record(
                source=source,
                extractor=context.extractor,
                pricing=context.pricing,
                application_attempts=attempts,
            )
            write_json(final_path, final)
            evidence = replay_deepseek_r1_evidence_record(
                final,
                source=source,
                extractor=context.extractor,
                pricing=context.pricing,
            )
            return final, evidence.construction_receipt.cost_microusd
    raise ExternalE6Error(
        f"source {question.question_id}/{source_position} exhausted all schema attempts"
    )


def _extraction_summary(
    context: E6Context,
    question: SelectedQuestion,
    *,
    raw_jsonl: bytes,
    evidences: Sequence[DeepSeekR1ExtractionEvidence],
    wall_microseconds: int,
) -> dict[str, Any]:
    receipts = [item.construction_receipt for item in evidences]
    bindings = [item.content_free_binding() for item in evidences]
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e6b-head20-r1-extraction-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E6_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_value_count": len(evidences),
        "source_values_sha256": sha256_json(
            [item.source.content_free_binding() for item in evidences]
        ),
        "evidence_jsonl": {
            "bytes": len(raw_jsonl),
            "sha256": sha256_bytes(raw_jsonl),
            "records": len(evidences),
        },
        "evidence_bindings": bindings,
        "evidence_bindings_sha256": sha256_json(bindings),
        "accounting": {
            "application_attempts": sum(item.selected_application_attempt for item in evidences),
            "http_attempts": sum(
                sum(attempt.http_attempts for attempt in item.application_attempts)
                for item in evidences
            ),
            "input_tokens": sum(receipt.input_tokens for receipt in receipts),
            "output_tokens": sum(receipt.output_tokens for receipt in receipts),
            "latency_microseconds": sum(receipt.latency_microseconds for receipt in receipts),
            "cost_microusd": sum(receipt.cost_microusd for receipt in receipts),
            "wall_microseconds": wall_microseconds,
        },
        "claims": {
            "source_only": True,
            "question_gold_answer_or_judge_consumed": False,
            "all_invalid_application_attempts_retained": True,
            "api_and_local_prompt_tokens_reconciled": True,
            "extractor_identity_authenticated": False,
            "reader_or_judge_executed": False,
            "production_policy_changed": False,
        },
    }
    return seal_artifact(payload)


def replay_extraction_question(
    context: E6Context,
    question: SelectedQuestion,
) -> tuple[dict[str, Any], tuple[DeepSeekR1ExtractionEvidence, ...]]:
    sources = _source_values(context, question)
    jsonl_path = e6_jsonl_path(context, "extraction", question)
    summary_path = e6_phase_path(context, "extraction-summary", question)
    if not jsonl_path.is_file() or not summary_path.is_file():
        raise ExternalE6Error("complete E6 extraction requires JSONL and summary artifacts")
    raw = jsonl_path.read_bytes()
    evidences = replay_deepseek_r1_evidence_jsonl(
        raw,
        sources=sources,
        extractor=context.extractor,
        pricing=context.pricing,
    )
    if tuple(item.source for item in evidences) != sources:
        raise ExternalE6Error(
            "E6 evidence JSONL must cover every source exactly once in canonical order"
        )
    summary = load_json(summary_path, sealed=True)
    wall = summary.get("accounting", {}).get("wall_microseconds")
    if isinstance(wall, bool) or not isinstance(wall, int) or wall < 0:
        raise ExternalE6Error("E6 extraction wall time is invalid")
    expected = _extraction_summary(
        context,
        question,
        raw_jsonl=raw,
        evidences=evidences,
        wall_microseconds=wall,
    )
    if summary != expected:
        raise ExternalE6Error("E6 extraction summary differs from exact evidence replay")
    for position, (source, evidence) in enumerate(zip(sources, evidences, strict=True)):
        if _evidence_for_value(context, question, position, source) != evidence:
            raise ExternalE6Error("consolidated E6 evidence differs from per-value evidence")
    return summary, evidences


def _repair_or_build_extraction_question(
    context: E6Context,
    question: SelectedQuestion,
    *,
    default_wall_microseconds: int,
) -> tuple[dict[str, Any], tuple[DeepSeekR1ExtractionEvidence, ...]]:
    sources = _source_values(context, question)
    records: list[dict[str, Any]] = []
    evidences: list[DeepSeekR1ExtractionEvidence] = []
    for position, source in enumerate(sources):
        evidence = _evidence_for_value(context, question, position, source)
        if evidence is None:
            raise ExternalE6Error("cannot repair consolidated extraction before every value exists")
        records.append(load_json(_value_record_path(context, question, position)))
        evidences.append(evidence)
    raw = deepseek_r1_evidence_jsonl_bytes(records)
    replayed = replay_deepseek_r1_evidence_jsonl(
        raw,
        sources=sources,
        extractor=context.extractor,
        pricing=context.pricing,
    )
    if tuple(evidences) != replayed:
        raise ExternalE6Error("per-value extraction evidence differs from consolidated replay")
    jsonl_path = e6_jsonl_path(context, "extraction", question)
    summary_path = e6_phase_path(context, "extraction-summary", question)
    if jsonl_path.exists() and jsonl_path.read_bytes() != raw:
        raise ExternalE6Error("saved consolidated extraction JSONL differs from per-value evidence")
    wall = default_wall_microseconds
    if summary_path.exists():
        saved = load_json(summary_path, sealed=True)
        wall = saved.get("accounting", {}).get("wall_microseconds")
        if isinstance(wall, bool) or not isinstance(wall, int) or wall < 0:
            raise ExternalE6Error("saved consolidated extraction wall time is invalid")
    summary = _extraction_summary(
        context,
        question,
        raw_jsonl=raw,
        evidences=replayed,
        wall_microseconds=wall,
    )
    if summary_path.exists() and load_json(summary_path, sealed=True) != summary:
        raise ExternalE6Error("saved consolidated extraction summary cannot be repaired exactly")
    if not jsonl_path.exists():
        atomic_write(jsonl_path, raw)
    if not summary_path.exists():
        write_json(summary_path, summary)
    return replay_extraction_question(context, question)


async def _run_extraction_async(
    context: E6Context,
    *,
    selected: Sequence[SelectedQuestion],
    tokenizer: DeepSeekExactTokenizer,
    base_url: str,
    api_key: str,
) -> None:
    if base_url.strip().rstrip("/") not in {
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
    }:
        raise ExternalE6Error("E6 extraction is frozen to the official DeepSeek endpoint")
    client = _ExtractionChatClient(api_key=api_key)
    _reconcile_extraction_journals(context, selected, tokenizer=tokenizer)
    journal_cost, unresolved = _external_journal_cost(context)
    if unresolved:
        raise ExternalE6Error("unresolved external call reservations block extraction resume")
    ledger = _SpendLedger(
        spent_microusd=journal_cost,
        maximum_microusd=MAX_EXTERNAL_COST_MICROUSD,
    )
    semaphore = asyncio.Semaphore(context.manifest["extraction"]["concurrency"])
    try:
        for question_index, question in enumerate(selected, start=1):
            summary_path = e6_phase_path(context, "extraction-summary", question)
            jsonl_path = e6_jsonl_path(context, "extraction", question)
            if summary_path.exists() or jsonl_path.exists():
                summary, evidences = _repair_or_build_extraction_question(
                    context,
                    question,
                    default_wall_microseconds=0,
                )
                print(
                    f"  extract: {question_index}/{len(selected)} verified "
                    f"{question.question_id} values={len(evidences)} "
                    f"cost=${summary['accounting']['cost_microusd'] / 1_000_000:.6f}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            sources = _source_values(context, question)
            started = perf_counter()
            tasks = [
                _extract_one_value(
                    context,
                    question,
                    position,
                    source,
                    client=client,
                    tokenizer=tokenizer,
                    ledger=ledger,
                    semaphore=semaphore,
                )
                for position, source in enumerate(sources)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failures = [result for result in results if isinstance(result, BaseException)]
            if failures:
                cost_cap = next(
                    (
                        failure
                        for failure in failures
                        if isinstance(failure, ExternalCostCapExceeded)
                    ),
                    None,
                )
                if cost_cap is not None:
                    raise cost_cap
                first = failures[0]
                raise ExternalE6Error(
                    f"E6 extraction failed for {len(failures)} source values: {first}"
                ) from first
            summary, evidences = _repair_or_build_extraction_question(
                context,
                question,
                default_wall_microseconds=int(math.ceil((perf_counter() - started) * 1_000_000)),
            )
            invalid = sum(item.selected_application_attempt - 1 for item in evidences)
            print(
                f"  extract: {question_index}/{len(selected)} ran {question.question_id} "
                f"values={len(evidences)} schema_retries={invalid} "
                f"cost=${summary['accounting']['cost_microusd'] / 1_000_000:.6f} "
                f"ledger=${ledger.spent_microusd / 1_000_000:.6f}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        await client.aclose()


def run_extraction_phase(
    context: E6Context,
    *,
    base_url: str,
    api_key_env: str,
    limit: int | None = None,
) -> None:
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise ExternalE6Error(f"environment variable {api_key_env!r} is missing")
    tokenizer = DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    asyncio.run(
        _run_extraction_async(
            context,
            selected=_selected_prefix(context, limit),
            tokenizer=tokenizer,
            base_url=base_url,
            api_key=api_key,
        )
    )


def _qwen_scorer_identity(context: E6Context) -> ScorerIdentity:
    snapshot = context.manifest["model_snapshots"]["qwen_embedding"]
    identity_material = {
        "producer": QWEN_IDENTITY_PRODUCER,
        "protocol": QWEN_SCORER_PROTOCOL,
        "model": QWEN_MODEL,
        "revision": QWEN_REVISION,
        "snapshot_artifact_sha256": snapshot["artifact_sha256"],
        "pooling": "official-last-valid-token-then-l2-normalize-float32",
        "maximum_length": 8192,
        "padding_side": "right-MPS-stability-transfer-choice",
        "query_rendering": "qwen_query_text",
    }
    return ScorerIdentity(
        producer=QWEN_IDENTITY_PRODUCER,
        protocol=QWEN_SCORER_PROTOCOL,
        model_id=QWEN_MODEL,
        model_revision=QWEN_REVISION,
        model_artifact_sha256=str(snapshot["artifact_sha256"]),
        identity_artifact_sha256=sha256_json(identity_material),
    )


def _representation_corpora(
    context: E6Context,
    question: SelectedQuestion,
) -> tuple[
    RepresentationCorpus,
    RepresentationCorpus,
    tuple[DeepSeekR1ExtractionEvidence, ...],
    dict[str, Any],
]:
    extraction, evidences = replay_extraction_question(context, question)
    values = _source_values(context, question)
    if tuple(item.source for item in evidences) != values:
        raise ExternalE6Error("E6 extraction evidence order differs from canonical values")
    r0 = RepresentationCorpus(
        projection_corpus=context.e1.corpus,
        values=values,
        derived_keys=(),
        construction_receipts=(),
    )
    r1 = RepresentationCorpus(
        projection_corpus=context.e1.corpus,
        values=values,
        derived_keys=tuple(item.derived_key for item in evidences),
        construction_receipts=tuple(item.construction_receipt for item in evidences),
    )
    return r0, r1, evidences, extraction


def _raw_score_rows(
    context: E6Context,
    question: SelectedQuestion,
    corpus: RepresentationCorpus,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = phase_path(context.e1, "dense", question)
    artifact = load_json(path, sealed=True)
    replay_dense_row(context.e1, question, artifact)
    observations = artifact["dense"]["observations"]
    values_by_turn = {value.turn.turn_id: value for value in corpus.values}
    rows: list[dict[str, Any]] = []
    for observation in observations:
        turn_id = _turn_id_from_payload(observation["turn_id"])
        value = values_by_turn.get(turn_id)
        if value is None:
            raise ExternalE6Error("E1 raw score refers to an unknown E6 canonical value")
        score = observation.get("raw_cosine")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ExternalE6Error("E1 raw score is not numeric")
        score = float(score)
        if not math.isfinite(score):
            raise ExternalE6Error("E1 raw score is not finite")
        rows.append(
            {
                "key_id": raw_key_id(value),
                "turn_id": _turn_id_payload(turn_id),
                "key_text_sha256": value.raw_value_sha256,
                "input_tokens_after_truncation": int(observation["input_tokens_after_truncation"]),
                "truncated_right_at_8192": bool(observation["truncated_right_at_8192"]),
                "raw_cosine": score,
            }
        )
    if len(rows) != len(corpus.values) or len({row["key_id"] for row in rows}) != len(rows):
        raise ExternalE6Error("E1 raw score rows are not an exhaustive E6 raw index")
    return rows, artifact


def _validate_merged_score_rows(
    rows: Any,
    *,
    corpus: RepresentationCorpus,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(corpus.derived_keys):
        raise ExternalE6Error("merged Qwen scores do not cover every derived key")
    ordered_keys = list(corpus.derived_keys)
    keys = {key.key_id: key for key in ordered_keys}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "key_id",
            "key_text_sha256",
            "input_tokens_after_truncation",
            "truncated_right_at_8192",
            "raw_cosine",
        }:
            raise ExternalE6Error("merged Qwen score row fields differ from the frozen schema")
        key_id = row.get("key_id")
        key = keys.get(key_id)
        if key is None or key_id in seen:
            raise ExternalE6Error("merged Qwen score repeats or invents a derived key")
        if key_id != ordered_keys[position].key_id:
            raise ExternalE6Error("merged Qwen scores are not in canonical derived-key order")
        if row.get("key_text_sha256") != key.key_text_sha256:
            raise ExternalE6Error("merged Qwen score key text digest drifted")
        tokens = row.get("input_tokens_after_truncation")
        truncated = row.get("truncated_right_at_8192")
        score = row.get("raw_cosine")
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or not 1 <= tokens <= QWEN_MAX_LENGTH
            or not isinstance(truncated, bool)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ExternalE6Error("merged Qwen score accounting is invalid")
        if truncated and tokens != QWEN_MAX_LENGTH:
            raise ExternalE6Error("merged Qwen truncation flag is inconsistent with token count")
        if not -1.0 - QWEN_COSINE_TOLERANCE <= float(score) <= 1.0 + QWEN_COSINE_TOLERANCE:
            raise ExternalE6Error("merged Qwen cosine lies outside its numeric tolerance")
        normalized.append(
            {
                "key_id": key_id,
                "key_text_sha256": key.key_text_sha256,
                "input_tokens_after_truncation": tokens,
                "truncated_right_at_8192": truncated,
                "raw_cosine": float(score),
            }
        )
        seen.add(str(key_id))
    if seen != set(keys):
        raise ExternalE6Error("merged Qwen score coverage is incomplete")
    return normalized


def _family_observation(
    corpus: RepresentationCorpus,
    *,
    family: KeyFamily,
    question: SelectedQuestion,
    scorer: ScorerIdentity,
    rows: Sequence[Mapping[str, Any]],
) -> RankedFamilyObservation:
    ranked = sorted(
        (
            RankedKeyScore(key_id=str(row["key_id"]), raw_score=float(row["raw_cosine"]))
            for row in rows
        ),
        key=lambda item: (-item.raw_score, item.key_id),
    )
    return RankedFamilyObservation.create(
        family=family,
        corpus=corpus,
        query_sha256=sha256_bytes(question.question.encode("utf-8")),
        scorer=scorer,
        observation_artifact_sha256=sha256_json(list(rows)),
        ranked_keys=tuple(ranked[:KEY_FAMILY_DEPTH]),
    )


def _evaluate_r0_r1(
    context: E6Context,
    question: SelectedQuestion,
    *,
    r0_corpus: RepresentationCorpus,
    r1_corpus: RepresentationCorpus,
    raw_rows: Sequence[Mapping[str, Any]],
    merged_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, RepresentationResult], dict[str, Any]]:
    scorer = _qwen_scorer_identity(context)
    r0_raw = _family_observation(
        r0_corpus,
        family=KeyFamily.RAW,
        question=question,
        scorer=scorer,
        rows=raw_rows,
    )
    r1_raw = _family_observation(
        r1_corpus,
        family=KeyFamily.RAW,
        question=question,
        scorer=scorer,
        rows=raw_rows,
    )
    r1_merged = _family_observation(
        r1_corpus,
        family=KeyFamily.MERGED_SFK,
        question=question,
        scorer=scorer,
        rows=merged_rows,
    )
    r0 = evaluate_representation_cell(
        r0_corpus,
        cell=RepresentationCell.RAW,
        observations=(r0_raw,),
    )
    if len(r0.hydrated_values) != HEAD_MATCHED_VALUE_COUNT:
        raise ExternalE6Error("E6b requires R0 to expose exactly the frozen 20-value head")
    r1_full = evaluate_representation_cell(
        r1_corpus,
        cell=RepresentationCell.RAW_MERGED_SFK,
        observations=(r1_raw, r1_merged),
    )
    r1_head = head_match_representation_result(r1_full)
    head_matching_control = {
        "method": "deterministic-existing-ranking-prefix-v1",
        "applies_after": "full-equal-family-RRF-k60-ranking",
        "applies_before": "canonical-raw-turn-packing",
        "source_full_trace_sha256": r1_full.trace_sha256,
        "source_full_hydrated_value_count": len(r1_full.hydrated_values),
        "source_full_hydrated_value_ids_sha256": sha256_json(
            [value.value_id for value in r1_full.hydrated_values]
        ),
        "retained_value_count": len(r1_head.hydrated_values),
        "retained_value_ids_sha256": sha256_json(
            [value.value_id for value in r1_head.hydrated_values]
        ),
        "derived_head_trace_sha256": r1_head.trace_sha256,
        "source_result_mutated": False,
        "reranking_executed": False,
        "gold_or_outcome_fields_used": False,
    }
    return (
        {
            RepresentationCell.RAW.value: r0,
            RepresentationCell.RAW_MERGED_SFK.value: r1_head,
        },
        head_matching_control,
    )


def _stage(
    name: str,
    *,
    calls: int,
    input_tokens: int,
    latency_microseconds: int,
    retry_count: int = 0,
) -> dict[str, Any]:
    return {
        "name": name,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": 0,
        "latency_microseconds": latency_microseconds,
        "cost_microusd": 0,
        "retry_count": retry_count,
        "cache_hits": 0,
    }


def _raw_query_stage(dense: Mapping[str, Any]) -> dict[str, Any]:
    lane = dense["dense"]
    accounting = lane["accounting"]
    return _stage(
        "raw-qwen-index-query-ranking-replayed",
        calls=int(accounting["model_batches"]),
        input_tokens=(
            int(accounting["document_input_tokens_after_truncation"])
            + int(lane["query_input_tokens_after_truncation"])
        ),
        latency_microseconds=0,
        retry_count=int(accounting["singleton_retries_after_nonfinite_batch"])
        + int(bool(lane["query_singleton_retry_after_nonfinite_batch"])),
    )


def _expected_qwen_batch_plan(
    token_counts: Sequence[int],
    *,
    maximum_batch_size: int,
) -> list[list[int]]:
    if maximum_batch_size < 1 or not token_counts:
        raise ExternalE6Error("Qwen batch-plan inputs are invalid")
    ordered = sorted(enumerate(token_counts), key=lambda item: (item[1], item[0]))
    if any(
        isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or not 1 <= tokens <= QWEN_MAX_LENGTH
        for _, tokens in ordered
    ):
        raise ExternalE6Error("Qwen batch-plan token count is outside the frozen bound")
    batches: list[list[int]] = []
    current: list[tuple[int, int]] = []
    for item in ordered:
        proposed = [*current, item]
        maximum = item[1]
        minimum = proposed[0][1]
        exceeds = len(proposed) > maximum_batch_size or (
            len(proposed) > 1
            and (
                maximum / minimum > QWEN_MAX_PADDING_RATIO
                or len(proposed) * maximum * maximum > QWEN_ATTENTION_CELL_BUDGET
            )
        )
        if exceeds:
            if not current:
                raise ExternalE6Error("Qwen batch plan could not admit a singleton")
            batches.append([position for position, _ in current])
            current = [item]
        else:
            current = proposed
    if current:
        batches.append([position for position, _ in current])
    return batches


def _retry_positions(value: Any, *, maximum: int, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ExternalE6Error(f"{label} must be a list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ExternalE6Error(f"{label} must contain integers")
    if value != sorted(set(value)) or any(not 0 <= item < maximum for item in value):
        raise ExternalE6Error(f"{label} must be sorted, unique, and in range")
    return list(value)


def _merged_execution_material(
    context: E6Context,
    *,
    dense: Mapping[str, Any],
    merged_rows: Sequence[Mapping[str, Any]],
    receipt: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    if not isinstance(receipt, dict) or set(receipt) != {
        "source",
        "query_elapsed_microseconds",
        "document_elapsed_microseconds",
        "query_singleton_retry_positions",
        "document_singleton_retry_positions",
        "wall_microseconds",
    }:
        raise ExternalE6Error("merged Qwen local execution receipt fields drifted")
    if receipt.get("source") != "local-observed-unauthenticated":
        raise ExternalE6Error("merged Qwen local execution receipt source drifted")
    timings: dict[str, int] = {}
    for field in (
        "query_elapsed_microseconds",
        "document_elapsed_microseconds",
        "wall_microseconds",
    ):
        value = receipt.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExternalE6Error("merged Qwen local execution timing is malformed")
        timings[field] = value
    if (
        timings["query_elapsed_microseconds"] + timings["document_elapsed_microseconds"]
        > timings["wall_microseconds"]
    ):
        raise ExternalE6Error("merged Qwen component time exceeds question wall time")
    query_retries = _retry_positions(
        receipt.get("query_singleton_retry_positions"),
        maximum=1,
        label="query singleton retry positions",
    )
    document_retries = _retry_positions(
        receipt.get("document_singleton_retry_positions"),
        maximum=len(merged_rows),
        label="document singleton retry positions",
    )
    maximum_batch_size = int(context.manifest["ranking"]["maximum_batch_size"])
    query_tokens = int(dense["dense"]["query_input_tokens_after_truncation"])
    document_tokens = [int(row["input_tokens_after_truncation"]) for row in merged_rows]
    query_plan = _expected_qwen_batch_plan(
        [query_tokens],
        maximum_batch_size=maximum_batch_size,
    )
    document_plan = _expected_qwen_batch_plan(
        document_tokens,
        maximum_batch_size=maximum_batch_size,
    )
    normalized_receipt = {
        "source": "local-observed-unauthenticated",
        "query_elapsed_microseconds": timings["query_elapsed_microseconds"],
        "document_elapsed_microseconds": timings["document_elapsed_microseconds"],
        "query_singleton_retry_positions": query_retries,
        "document_singleton_retry_positions": document_retries,
        "wall_microseconds": timings["wall_microseconds"],
    }
    stage = _stage(
        "merged-qwen-index-query-ranking",
        calls=len(query_plan) + len(document_plan) + len(query_retries) + len(document_retries),
        input_tokens=query_tokens + sum(document_tokens),
        latency_microseconds=0,
        retry_count=len(query_retries) + len(document_retries),
    )
    frozen_runtime = context.manifest["ranking"]["runtime"]
    runtime = {
        "python": frozen_runtime["python"],
        "torch": frozen_runtime["torch"],
        "transformers": frozen_runtime["transformers"],
        "device": context.manifest["ranking"]["device"],
        "dtype": frozen_runtime["dtype"],
        "maximum_batch_size": maximum_batch_size,
        "query_batch_plan": query_plan,
        "document_batch_plan": document_plan,
        "query_singleton_retry_positions": query_retries,
        "document_singleton_retry_positions": document_retries,
    }
    return normalized_receipt, stage, runtime, timings["wall_microseconds"]


def _rank_payload(
    context: E6Context,
    question: SelectedQuestion,
    *,
    extraction: Mapping[str, Any],
    dense: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    merged_rows: Sequence[Mapping[str, Any]],
    results: Mapping[str, RepresentationResult],
    head_matching_control: Mapping[str, Any],
    local_execution_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    raw_stage = _raw_query_stage(dense)
    receipt, merged_stage, runtime, wall_microseconds = _merged_execution_material(
        context,
        dense=dense,
        merged_rows=merged_rows,
        receipt=local_execution_receipt,
    )
    r0_stages = [raw_stage]
    r1_stages = sorted([raw_stage, merged_stage], key=lambda row: str(row["name"]))
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e6b-head20-ranking-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E6_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_extraction_artifact_sha256": extraction["artifact_sha256"],
        "source_e1_dense_artifact_sha256": dense["artifact_sha256"],
        "query_sha256": sha256_bytes(question.question.encode("utf-8")),
        "scorer": _qwen_scorer_identity(context).content_free_binding(),
        "raw": {
            "source": "replayed-frozen-E1-exhaustive-observations",
            "observations": list(raw_rows),
            "observations_sha256": sha256_json(list(raw_rows)),
        },
        "merged_sfk": {
            "source": "fresh-pinned-Qwen-exhaustive-observations",
            "observations": list(merged_rows),
            "observations_sha256": sha256_json(list(merged_rows)),
        },
        "query_accounting": {
            RepresentationCell.RAW.value: {
                "complete": True,
                "source": "externally-attested-unverified",
                "stages": r0_stages,
                "stages_sha256": sha256_json(r0_stages),
            },
            RepresentationCell.RAW_MERGED_SFK.value: {
                "complete": True,
                "source": "externally-attested-unverified",
                "stages": r1_stages,
                "stages_sha256": sha256_json(r1_stages),
            },
        },
        "representation_results": {
            cell: result.content_free_artifact() for cell, result in results.items()
        },
        "head_matching_control": dict(head_matching_control),
        "raw_local_execution_observation": {
            "source": "replayed-E1-local-observed-unauthenticated",
            "model_elapsed_microseconds": int(
                math.ceil(float(dense["dense"]["accounting"]["model_elapsed_ms"]) * 1000.0)
            ),
            "used_for_gate_accounting": False,
        },
        "local_execution_receipt": receipt,
        "runtime": runtime,
        "wall_microseconds": wall_microseconds,
        "claims": {
            "full_raw_and_merged_family_indexes_scored": True,
            "merged_query_calls_tokens_and_retries_reconstructed": True,
            "local_execution_timings_authenticated": False,
            "local_execution_timings_used_for_gate": False,
            "gold_fields_consumed": False,
            "reader_or_judge_executed": False,
            "external_scorer_identity_authenticated": False,
            "production_policy_changed": False,
        },
    }
    return seal_artifact(payload)


def replay_rank_question(
    context: E6Context,
    question: SelectedQuestion,
) -> tuple[dict[str, Any], dict[str, RepresentationResult]]:
    path = e6_phase_path(context, "ranking", question)
    row = load_json(path, sealed=True)
    r0, r1, _, extraction = _representation_corpora(context, question)
    raw_rows, dense = _raw_score_rows(context, question, r0)
    merged_rows = _validate_merged_score_rows(
        row.get("merged_sfk", {}).get("observations"),
        corpus=r1,
    )
    results, head_matching_control = _evaluate_r0_r1(
        context,
        question,
        r0_corpus=r0,
        r1_corpus=r1,
        raw_rows=raw_rows,
        merged_rows=merged_rows,
    )
    expected = _rank_payload(
        context,
        question,
        extraction=extraction,
        dense=dense,
        raw_rows=raw_rows,
        merged_rows=merged_rows,
        results=results,
        head_matching_control=head_matching_control,
        local_execution_receipt=row.get("local_execution_receipt"),
    )
    if row != expected:
        raise ExternalE6Error("saved E6 ranking artifact differs from exact offline replay")
    return row, results


def run_rank_phase(
    context: E6Context,
    *,
    device: str,
    batch_size: int,
    limit: int | None = None,
) -> None:
    selected = _selected_prefix(context, limit)
    pending: list[SelectedQuestion] = []
    for question in selected:
        path = e6_phase_path(context, "ranking", question)
        if path.exists():
            row, results = replay_rank_question(context, question)
            print(
                f"  rank: verified {question.question_id} "
                f"R0={len(results['R0'].hydrated_values)} "
                f"R1={len(results['R1'].hydrated_values)}",
                file=sys.stderr,
                flush=True,
            )
        else:
            pending.append(question)
    if not pending:
        return
    embedder = QwenEmbedder(context.e1.qwen_root, device=device, batch_size=batch_size)
    try:
        frozen_runtime = context.manifest["ranking"]["runtime"]
        if (
            sys.version.split()[0] != frozen_runtime["python"]
            or embedder.torch.__version__ != frozen_runtime["torch"]
            or embedder.transformers_version != frozen_runtime["transformers"]
            or embedder.dtype_name != frozen_runtime["dtype"]
            or device != context.manifest["ranking"]["device"]
            or batch_size != context.manifest["ranking"]["maximum_batch_size"]
        ):
            raise ExternalE6Error("live Qwen runtime differs from the frozen manifest")
        for completed, question in enumerate(pending, start=1):
            started = perf_counter()
            r0, r1, _, extraction = _representation_corpora(context, question)
            raw_rows, dense = _raw_score_rows(context, question, r0)
            query_batch = embedder.embed([qwen_query_text(question.question)])
            document_batch = embedder.embed([key.key_text for key in r1.derived_keys])
            similarities = document_batch.vectors @ query_batch.vectors[0]
            merged_rows = [
                {
                    "key_id": key.key_id,
                    "key_text_sha256": key.key_text_sha256,
                    "input_tokens_after_truncation": document_batch.token_counts[position],
                    "truncated_right_at_8192": document_batch.truncated[position],
                    "raw_cosine": float(similarities[position].item()),
                }
                for position, key in enumerate(r1.derived_keys)
            ]
            merged_rows = _validate_merged_score_rows(merged_rows, corpus=r1)
            results, head_matching_control = _evaluate_r0_r1(
                context,
                question,
                r0_corpus=r0,
                r1_corpus=r1,
                raw_rows=raw_rows,
                merged_rows=merged_rows,
            )
            query_elapsed = int(math.ceil(query_batch.elapsed_ms * 1000.0))
            document_elapsed = int(math.ceil(document_batch.elapsed_ms * 1000.0))
            measured_wall = int(math.ceil((perf_counter() - started) * 1_000_000))
            local_execution_receipt = {
                "source": "local-observed-unauthenticated",
                "query_elapsed_microseconds": query_elapsed,
                "document_elapsed_microseconds": document_elapsed,
                "query_singleton_retry_positions": sorted(query_batch.singleton_retry_positions),
                "document_singleton_retry_positions": sorted(
                    document_batch.singleton_retry_positions
                ),
                "wall_microseconds": max(measured_wall, query_elapsed + document_elapsed),
            }
            row = _rank_payload(
                context,
                question,
                extraction=extraction,
                dense=dense,
                raw_rows=raw_rows,
                merged_rows=merged_rows,
                results=results,
                head_matching_control=head_matching_control,
                local_execution_receipt=local_execution_receipt,
            )
            live_merged_stage = next(
                stage
                for stage in row["query_accounting"][RepresentationCell.RAW_MERGED_SFK.value][
                    "stages"
                ]
                if stage["name"] == "merged-qwen-index-query-ranking"
            )
            if (
                row["runtime"]["query_batch_plan"]
                != [list(batch) for batch in query_batch.batch_plan]
                or row["runtime"]["document_batch_plan"]
                != [list(batch) for batch in document_batch.batch_plan]
                or live_merged_stage["calls"]
                != query_batch.model_batches + document_batch.model_batches
            ):
                raise ExternalE6Error("live Qwen execution differs from reconstructed accounting")
            write_json(e6_phase_path(context, "ranking", question), row)
            replay_rank_question(context, question)
            print(
                f"  rank: {completed}/{len(pending)} {question.question_id} "
                f"keys={len(merged_rows)} R0={len(results['R0'].hydrated_values)} "
                f"R1={len(results['R1'].hydrated_values)}",
                file=sys.stderr,
                flush=True,
            )
            del query_batch, document_batch, similarities
    finally:
        embedder.close()


def _final_tokenizer_receipt_sha256(packed: RepresentationPromptPackingResult) -> str:
    trace = packed.packed.trace
    final = trace.get("final_prompt")
    observations = trace.get("exact_count_observations")
    if not isinstance(final, dict) or not isinstance(observations, list):
        raise ExternalE6Error("E6 pack lacks exact tokenizer observations")
    sequence = final.get("final_observation_sequence")
    observation = next(
        (
            item
            for item in observations
            if isinstance(item, dict) and item.get("sequence") == sequence
        ),
        None,
    )
    if not isinstance(observation, dict) or not isinstance(observation.get("receipt"), dict):
        raise ExternalE6Error("E6 pack lacks its final tokenizer receipt")
    return sha256_json(observation["receipt"])


def _tokenizer_receipt_namespace(
    context: E6Context,
    question: SelectedQuestion,
    cell: str,
) -> dict[str, Any]:
    material = {
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_position": question.position,
        "question_id": question.question_id,
        "cell": cell,
    }
    return {**material, "namespace_sha256": sha256_json(material)}


def _compute_pack_artifacts(
    context: E6Context,
    question: SelectedQuestion,
    *,
    rank_row: Mapping[str, Any],
    results: Mapping[str, RepresentationResult],
    tokenizer: DeepSeekExactTokenizer,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, RepresentationPromptPackingResult]]:
    packed: dict[str, RepresentationPromptPackingResult] = {}
    prompts: dict[str, Any] = {}
    for cell in E6_CELLS:
        tokenizer.reset_receipts()
        bridge = pack_representation_result(
            results[cell],
            manifest=context.preflight,
            question_id=question.question_id,
            question=question.question,
            current_date=question.current_date,
            tokenizer=tokenizer,
        )
        packed[cell] = bridge
        receipt_sha256 = _final_tokenizer_receipt_sha256(bridge)
        receipt_namespace = _tokenizer_receipt_namespace(context, question, cell)
        prompts[cell] = {
            "prompt": bridge.prompt,
            "prompt_sha256": sha256_bytes(bridge.prompt.encode("utf-8")),
            "prompt_utf8_bytes": len(bridge.prompt.encode("utf-8")),
            "prompt_tokens": bridge.packed.trace["final_prompt"]["tokens"],
            "bridge_trace_sha256": bridge.trace_sha256,
            "packing_trace_sha256": bridge.packed.trace_sha256,
            "final_tokenizer_receipt_sha256": receipt_sha256,
            "tokenizer_receipt_namespace": receipt_namespace,
            "namespaced_final_tokenizer_receipt_sha256": sha256_json(
                {
                    "namespace_sha256": receipt_namespace["namespace_sha256"],
                    "receipt_sha256": receipt_sha256,
                }
            ),
        }
    prompt_payload = {
        "artifact_type": "swarmbrain-longmemeval-e6b-head20-prompts-sensitive",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E6_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "classification": "contains-public-benchmark-question-and-turn-text",
        "arms": prompts,
    }
    prompt_artifact = seal_artifact(prompt_payload)
    pack_payload = {
        "artifact_type": "swarmbrain-longmemeval-e6b-head20-pack-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E6_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_ranking_artifact_sha256": rank_row["artifact_sha256"],
        "source_representation_trace_sha256s": {
            cell: results[cell].trace_sha256 for cell in E6_CELLS
        },
        "prompt_artifact_sha256": prompt_artifact["artifact_sha256"],
        "preflight_manifest_sha256": context.preflight.manifest_sha256,
        "tokenizer": tokenizer.identity.as_dict(),
        "runtime": {
            "python": sys.version.split()[0],
            "transformers": tokenizer.transformers_version,
        },
        "preflight_admission": context.manifest["packing"]["preflight_admission"],
        "tokenizer_receipt_policy": {
            "namespace": "independent-per-arm-per-question-development-artifact",
            "request_ids_are_namespace_local": True,
            "global_request_id_uniqueness_claimed": False,
            "provider_ids_are_local_synthetic_ids": True,
            "prepared_run_receipt_emitted": False,
        },
        "arms": {cell: packed[cell].content_free_artifact() for cell in E6_CELLS},
        "claims": {
            "complete_reader_prompt_counted": True,
            "whole_canonical_turns_only": True,
            "derived_keys_delivered_to_reader": False,
            "gold_fields_consumed": False,
            "reader_or_judge_executed": False,
            "official_prepared_run_admission": False,
            "production_policy_changed": False,
        },
    }
    return seal_artifact(pack_payload), prompt_artifact, packed


def replay_pack_question(
    context: E6Context,
    question: SelectedQuestion,
    *,
    tokenizer: DeepSeekExactTokenizer,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, RepresentationPromptPackingResult],
    dict[str, RepresentationResult],
]:
    rank_row, results = replay_rank_question(context, question)
    pack_path = e6_phase_path(context, "pack", question)
    prompt_path = e6_phase_path(context, "prompts", question)
    if pack_path.exists() != prompt_path.exists():
        raise ExternalE6Error("E6 pack has only one side of its prompt/trace pair")
    if not pack_path.exists():
        raise ExternalE6Error("E6 pack artifacts are missing")
    pack_row = load_json(pack_path, sealed=True)
    prompt_row = load_json(prompt_path, sealed=True)
    expected_pack, expected_prompt, packed = _compute_pack_artifacts(
        context,
        question,
        rank_row=rank_row,
        results=results,
        tokenizer=tokenizer,
    )
    if pack_row != expected_pack or prompt_row != expected_prompt:
        raise ExternalE6Error("saved E6 pack differs from exact deterministic replay")
    return pack_row, prompt_row, packed, results


def run_pack_phase(
    context: E6Context,
    *,
    limit: int | None = None,
) -> None:
    tokenizer = DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    selected = _selected_prefix(context, limit)
    for completed, question in enumerate(selected, start=1):
        pack_path = e6_phase_path(context, "pack", question)
        prompt_path = e6_phase_path(context, "prompts", question)
        if pack_path.exists() or prompt_path.exists():
            pack, _, packed, _ = replay_pack_question(
                context,
                question,
                tokenizer=tokenizer,
            )
            action = "verified"
        else:
            rank, results = replay_rank_question(context, question)
            pack, prompts, packed = _compute_pack_artifacts(
                context,
                question,
                rank_row=rank,
                results=results,
                tokenizer=tokenizer,
            )
            write_json(prompt_path, prompts)
            write_json(pack_path, pack)
            replay_pack_question(context, question, tokenizer=tokenizer)
            action = "built"
        print(
            f"  pack: {completed}/{len(selected)} {action} {question.question_id} "
            + " ".join(
                f"{cell}={packed[cell].packed.trace['final_prompt']['tokens']}t/"
                f"{len(packed[cell].packed.trace['kept_ids'])}v"
                for cell in E6_CELLS
            ),
            file=sys.stderr,
            flush=True,
        )


def build_operational_audit(
    context: E6Context,
    *,
    limit: int,
) -> dict[str, Any]:
    if limit != 40:
        raise ExternalE6Error("E6b operational audit is frozen to run positions 0--39")
    if _qa_durable_state_exists(context):
        raise ExternalE6Error("operational tranche audit forbids any QA state")
    forbidden_paths = [
        context.output_dir / "report.json",
        *context.output_dir.glob("diagnostic*.json"),
    ]
    case_root = context.output_dir / "cases"
    if case_root.exists() and any(case_root.iterdir()):
        raise ExternalE6Error("operational tranche audit forbids post-hoc case artifacts")
    if any(path.exists() for path in forbidden_paths):
        raise ExternalE6Error("operational tranche audit found forbidden aggregate statistics")
    tokenizer = DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    extraction_artifacts: list[str] = []
    ranking_artifacts: list[str] = []
    pack_artifacts: list[str] = []
    prompt_artifacts: list[str] = []
    source_values = 0
    application_attempts = 0
    extraction_cost = 0
    expected_extraction_journals: dict[str, dict[str, Any]] = {}
    provider_request_ids: set[str] = set()
    for question in _selected_prefix(context, limit):
        extraction, evidences = replay_extraction_question(context, question)
        rank, results = replay_rank_question(context, question)
        pack, prompts, packed, replayed_results = replay_pack_question(
            context,
            question,
            tokenizer=tokenizer,
        )
        if results != replayed_results:
            raise ExternalE6Error("operational replay returned inconsistent representation results")
        control = rank.get("head_matching_control")
        if (
            not isinstance(control, dict)
            or control.get("retained_value_count") != HEAD_MATCHED_VALUE_COUNT
            or any(
                len(results[cell].hydrated_values) != HEAD_MATCHED_VALUE_COUNT for cell in E6_CELLS
            )
        ):
            raise ExternalE6Error("operational replay found a non-head-matched representation")
        if any(
            len(packed[cell].packed.trace["kept_ids"]) > HEAD_MATCHED_VALUE_COUNT
            for cell in E6_CELLS
        ):
            raise ExternalE6Error("operational replay packed more than the frozen candidate head")
        source_values += len(evidences)
        application_attempts += sum(len(evidence.application_attempts) for evidence in evidences)
        extraction_cost += int(extraction["accounting"]["cost_microusd"])
        for source_position, evidence in enumerate(evidences):
            final_record = load_json(_value_record_path(context, question, source_position))
            attempt_records = final_record["application_attempts"]
            for application_attempt, (attempt, attempt_record) in enumerate(
                zip(evidence.application_attempts, attempt_records, strict=True),
                start=1,
            ):
                route = _extraction_journal_route(
                    question,
                    source_position,
                    application_attempt,
                )
                if route in expected_extraction_journals:
                    raise ExternalE6Error("operational extraction route is duplicated")
                provider_request_id = attempt_record["response"]["provider_request_id"]
                if provider_request_id in provider_request_ids:
                    raise ExternalE6Error(
                        "operational provider request ID crossed extraction routes"
                    )
                provider_request_ids.add(provider_request_id)
                request = replay_chat_request(attempt.raw_request)
                exact_prompt_tokens = tokenizer.exact_count(request.prompt)
                reserved_microusd = context.pricing.upper_bound_microusd(
                    input_tokens=exact_prompt_tokens,
                    output_tokens=request.max_tokens,
                    retry_count=DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS - 1,
                    request_max_tokens=request.max_tokens,
                )
                expected_extraction_journals[route] = {
                    "raw_request_sha256": sha256_bytes(attempt.raw_request),
                    "raw_response_sha256": sha256_bytes(attempt.raw_response),
                    "attempts": attempt.http_attempts,
                    "latency_microseconds": attempt.latency_microseconds,
                    "exact_local_prompt_tokens": exact_prompt_tokens,
                    "request_max_tokens": request.max_tokens,
                    "reserved_microusd": reserved_microusd,
                    "actual_microusd": _attempt_cost_microusd(
                        attempt_record,
                        pricing=context.pricing,
                    ),
                }
        extraction_artifacts.append(str(extraction["artifact_sha256"]))
        ranking_artifacts.append(str(rank["artifact_sha256"]))
        pack_artifacts.append(str(pack["artifact_sha256"]))
        prompt_artifacts.append(str(prompts["artifact_sha256"]))
    _validate_expected_journal_bindings(
        context,
        namespace="extraction",
        expected=expected_extraction_journals,
    )
    journal_cost, unresolved = _external_journal_cost(context)
    journal_root = context.output_dir / "external-call-journal" / "extraction"
    journal_counts = {
        suffix: (len(list(journal_root.glob(f"*.{suffix}.json"))) if journal_root.exists() else 0)
        for suffix in ("reservation", "response", "settlement")
    }
    if (
        unresolved
        or journal_cost != extraction_cost
        or any(count != application_attempts for count in journal_counts.values())
    ):
        raise ExternalE6Error("operational tranche external-call journal is incomplete")
    artifact_sets = {
        "extraction": extraction_artifacts,
        "ranking": ranking_artifacts,
        "pack": pack_artifacts,
        "prompts": prompt_artifacts,
        "reservations": _journal_artifact_sha256s(context, "reservation"),
        "response_wals": _journal_artifact_sha256s(context, "response"),
        "settlements": _journal_artifact_sha256s(context, "settlement"),
    }
    audit = seal_artifact(
        {
            "artifact_type": "swarmbrain-longmemeval-e6b-head20-operational-audit",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "protocol_version": E6_RUN_PROTOCOL_VERSION,
            "run_manifest_sha256": context.manifest["artifact_sha256"],
            "run_positions": [0, 39],
            "question_count": limit,
            "source_values": source_values,
            "application_attempts": application_attempts,
            "provider_request_ids": len(provider_request_ids),
            "provider_request_ids_globally_unique": True,
            "external_cost_microusd": extraction_cost,
            "hard_cost_limit_microusd": MAX_EXTERNAL_COST_MICROUSD,
            "unresolved_reservations": unresolved,
            "journal_counts": journal_counts,
            "artifact_sets": {
                name: {"count": len(values), "ordered_sha256": sha256_json(values)}
                for name, values in artifact_sets.items()
            },
            "artifact_set_members": artifact_sets,
            "inspection_boundary": {
                "mechanical_evidence_only": True,
                "gold_context_metrics_compiled": False,
                "QA_metrics_compiled": False,
                "promotion_or_futility_verdict_compiled": False,
                "outcome_dependent_stop_or_extension_permitted": False,
            },
            "claims": {
                "quality_inference": False,
                "official_longmemeval_score": False,
                "official_gpt4o_executed": False,
                "production_policy_changed": False,
            },
        }
    )
    write_json(context.output_dir / "operational-audit-40.json", audit)
    return load_operational_audit(context)


def load_operational_audit(context: E6Context) -> dict[str, Any]:
    path = context.output_dir / "operational-audit-40.json"
    if not path.is_file():
        raise ExternalE6Error("E6b full execution requires the sealed first-40 operational audit")
    audit = load_json(path, sealed=True)
    if (
        audit.get("artifact_type") != "swarmbrain-longmemeval-e6b-head20-operational-audit"
        or audit.get("protocol_version") != E6_RUN_PROTOCOL_VERSION
        or audit.get("run_manifest_sha256") != context.manifest["artifact_sha256"]
        or audit.get("run_positions") != [0, 39]
        or audit.get("question_count") != 40
        or audit.get("unresolved_reservations") != 0
        or int(audit.get("external_cost_microusd", -1)) > MAX_EXTERNAL_COST_MICROUSD
    ):
        raise ExternalE6Error("E6b operational audit binding is invalid")
    inspection = audit.get("inspection_boundary")
    if not isinstance(inspection, dict) or inspection != {
        "mechanical_evidence_only": True,
        "gold_context_metrics_compiled": False,
        "QA_metrics_compiled": False,
        "promotion_or_futility_verdict_compiled": False,
        "outcome_dependent_stop_or_extension_permitted": False,
    }:
        raise ExternalE6Error("E6b operational audit crossed its inspection boundary")
    members = audit.get("artifact_set_members")
    summaries = audit.get("artifact_sets")
    if not isinstance(members, dict) or not isinstance(summaries, dict):
        raise ExternalE6Error("E6b operational audit artifact sets are missing")
    required_sets = {
        "extraction",
        "ranking",
        "pack",
        "prompts",
        "reservations",
        "response_wals",
        "settlements",
    }
    if set(members) != required_sets or set(summaries) != required_sets:
        raise ExternalE6Error("E6b operational audit artifact-set names drifted")
    for name in sorted(required_sets):
        values = members[name]
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ExternalE6Error("E6b operational audit artifact members are malformed")
        if summaries[name] != {
            "count": len(values),
            "ordered_sha256": sha256_json(values),
        }:
            raise ExternalE6Error("E6b operational audit artifact summary drifted")
    first40 = _selected_prefix(context, 40)
    phase_members = {
        "extraction": [
            str(
                load_json(e6_phase_path(context, "extraction-summary", question), sealed=True)[
                    "artifact_sha256"
                ]
            )
            for question in first40
        ],
        "ranking": [
            str(
                load_json(e6_phase_path(context, "ranking", question), sealed=True)[
                    "artifact_sha256"
                ]
            )
            for question in first40
        ],
        "pack": [
            str(load_json(e6_phase_path(context, "pack", question), sealed=True)["artifact_sha256"])
            for question in first40
        ],
        "prompts": [
            str(
                load_json(e6_phase_path(context, "prompts", question), sealed=True)[
                    "artifact_sha256"
                ]
            )
            for question in first40
        ],
    }
    if any(phase_members[name] != members[name] for name in phase_members):
        raise ExternalE6Error("E6b first-40 phase artifacts differ from operational audit")
    current_journals = {
        "reservations": set(_journal_artifact_sha256s(context, "reservation")),
        "response_wals": set(_journal_artifact_sha256s(context, "response")),
        "settlements": set(_journal_artifact_sha256s(context, "settlement")),
    }
    if any(not set(members[name]).issubset(current_journals[name]) for name in current_journals):
        raise ExternalE6Error("E6b first-40 journal evidence is no longer retained")
    return audit


def _session_digest(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def _candidate_session_sha256s(
    question: SelectedQuestion,
    result: RepresentationResult,
) -> list[str]:
    session_ids = [str(value) for value in question.record["haystack_session_ids"]]
    return [
        _session_digest(session_ids[value.turn.turn_id.session_position])
        for value in result.hydrated_values
    ]


def _prompt_value_ids(
    result: RepresentationResult,
    packed: RepresentationPromptPackingResult,
) -> list[str]:
    values_by_turn = {value.turn.turn_id: value for value in result.hydrated_values}
    output: list[str] = []
    for payload in packed.packed.trace["kept_ids"]:
        turn_id = _turn_id_from_payload(payload)
        value = values_by_turn.get(turn_id)
        if value is None:
            raise ExternalE6Error("E6 packed prompt contains a non-hydrated turn")
        output.append(value.value_id)
    return output


def _qa_case_binding(
    context: E6Context,
    question: SelectedQuestion,
    *,
    prompt_row: Mapping[str, Any],
    tokenizer: DeepSeekExactTokenizer,
) -> dict[str, dict[str, Any] | None]:
    qa_path = e6_phase_path(context, "qa", question)
    receipt_path = e6_jsonl_path(context, "qa-receipts", question)
    if qa_path.exists() and not receipt_path.exists():
        raise ExternalE6Error("E6 QA result has no raw receipt sidecar")
    if not qa_path.exists():
        if receipt_path.exists():
            _validate_qa_receipt_prefix(
                context,
                question,
                prompt_row=prompt_row,
                receipts=_load_receipts(receipt_path),
                tokenizer=tokenizer,
            )
        return {cell: None for cell in E6_CELLS}
    qa = load_json(qa_path, sealed=True)
    receipts = _load_receipts(receipt_path)
    replay_qa_question(context, question, qa_row=qa, receipts=receipts)
    return {
        cell: {
            "correct": bool(qa["arms"][cell]["development_label"]),
            "reader_receipt_sha256": qa["arms"][cell]["reader"]["receipt_sha256"],
            "judge_receipt_sha256": qa["arms"][cell]["development_judge"]["receipt_sha256"],
        }
        for cell in E6_CELLS
    }


def _case_payload(
    context: E6Context,
    question: SelectedQuestion,
    *,
    case_index: int,
    rank: Mapping[str, Any],
    prompt_row: Mapping[str, Any],
    packed: Mapping[str, RepresentationPromptPackingResult],
    results: Mapping[str, RepresentationResult],
    tokenizer: DeepSeekExactTokenizer,
) -> dict[str, Any]:
    gold = (
        []
        if "_abs" in question.question_id
        else sorted(
            {_session_digest(str(value)) for value in question.record.get("answer_session_ids", ())}
        )
    )
    qa = _qa_case_binding(
        context,
        question,
        prompt_row=prompt_row,
        tokenizer=tokenizer,
    )
    arms: dict[str, Any] = {}
    for cell in E6_CELLS:
        prompt = prompt_row["arms"][cell]
        arms[cell] = {
            "cell": cell,
            "representation": results[cell].content_free_artifact(),
            "context": {
                "candidate_session_sha256s": _candidate_session_sha256s(question, results[cell]),
                "prompt_value_ids": _prompt_value_ids(results[cell], packed[cell]),
                "prompt_tokens": int(prompt["prompt_tokens"]),
                "prompt_sha256": str(prompt["prompt_sha256"]),
                "tokenizer_artifact_sha256": _snapshot_artifact(context.e1, "deepseek_tokenizer"),
                "tokenizer_receipt_sha256": str(
                    prompt["namespaced_final_tokenizer_receipt_sha256"]
                ),
            },
            "query_accounting": rank["query_accounting"][cell],
            "qa": qa[cell],
        }
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "artifact_type": CASE_ARTIFACT_TYPE,
        "protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
        "case_index": case_index,
        "question_id": question.question_id,
        "question_type": str(question.record["question_type"]),
        "gold_session_sha256s": gold,
        "arms": arms,
    }


def build_case_phase(
    context: E6Context,
    *,
    limit: int | None = None,
) -> tuple[Path, ...]:
    tokenizer = DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    paths: list[Path] = []
    selected = _selected_prefix(context, limit)
    for case_index, question in enumerate(selected):
        rank, _ = replay_rank_question(context, question)
        _, prompts, packed, results = replay_pack_question(
            context,
            question,
            tokenizer=tokenizer,
        )
        case = seal_case_input(
            _case_payload(
                context,
                question,
                case_index=case_index,
                rank=rank,
                prompt_row=prompts,
                packed=packed,
                results=results,
                tokenizer=tokenizer,
            )
        )
        path = e6_phase_path(context, "cases", question)
        write_json(path, case)
        paths.append(path)
        print(
            f"  case: {case_index + 1}/{len(selected)} sealed {question.question_id}",
            file=sys.stderr,
            flush=True,
        )
    return tuple(paths)


def build_diagnostic_report(
    context: E6Context,
    *,
    limit: int | None = None,
    allow_complete_qa: bool = False,
) -> dict[str, Any]:
    if limit is not None:
        raise ExternalE6Error("E6b forbids aggregate diagnostics on an operational prefix")
    qa_complete = sum(
        e6_phase_path(context, "qa", question).is_file() for question in context.e1.selected
    )
    if _qa_durable_state_exists(context):
        if qa_complete != len(context.e1.selected):
            raise ExternalE6Error("E6b forbids aggregate diagnostics over partial QA evidence")
        if not allow_complete_qa:
            raise ExternalE6Error("E6b QA statistics may be compiled only by the final report")
    paths = build_case_phase(context, limit=limit)
    generic = compile_r0_r1_diagnostic(paths)
    generic_sha256 = generic["artifact_sha256"]
    report_payload = {
        key: value
        for key, value in generic.items()
        if key
        not in {
            "artifact_sha256",
            "artifact_type",
            "protocol_version",
            "early_stop",
            "decision",
            "claims",
        }
    }
    report_payload.update(
        {
            "artifact_type": "swarmbrain-longmemeval-e6b-head20-diagnostic",
            "protocol_version": E6_RUN_PROTOCOL_VERSION,
            "classification": "non-promotional-E6b-development-diagnostic",
            "source_generic_diagnostic": {
                "artifact_sha256": generic_sha256,
                "compiler_protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
                "generic_early_stop_and_decision_are_non_authoritative": True,
            },
            "decision_authority": (
                "E6b G1 and G2 in the sealed top-level report; no generic diagnostic verdict"
            ),
            "claims": {
                "official_longmemeval_score": False,
                "paper_reproduction": False,
                "serving_change_authorized": False,
                "production_policy_changed": False,
                "generic_E6_decision_applied": False,
            },
        }
    )
    report = seal_artifact(report_payload)
    suffix = "" if limit is None else f"-{limit}"
    write_json(context.output_dir / f"diagnostic{suffix}.json", report)
    return report


def _qa_arm_order(
    context: E6Context,
    question: SelectedQuestion,
) -> tuple[str, str]:
    run_position = next(
        (
            position
            for position, candidate in enumerate(context.e1.selected)
            if candidate.position == question.position
            and candidate.question_id == question.question_id
        ),
        None,
    )
    if run_position is None:
        raise ExternalE6Error("E6b QA question is outside the frozen run order")
    return E6_CELLS if run_position % 2 == 0 else tuple(reversed(E6_CELLS))


def _qa_receipt_id(question_id: str, cell: str) -> str:
    return f"{question_id}:e6:{cell.casefold()}"


def _qa_expected_routes(
    context: E6Context,
    question: SelectedQuestion,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (cell, _qa_receipt_id(question.question_id, cell), role)
        for cell in _qa_arm_order(context, question)
        for role in ("reader", "development_judge")
    )


def _chat_cost_microusd(
    result: ChatResult,
    *,
    pricing: DeepSeekR1PricingIdentity,
) -> int:
    return pricing.upper_bound_microusd(
        input_tokens=result.prompt_tokens,
        output_tokens=result.completion_tokens,
        retry_count=result.attempts - 1,
        request_max_tokens=result.request.max_tokens,
    )


def _validate_qa_chat(
    result: ChatResult,
    *,
    expected_prompt: str,
    expected_max_tokens: int,
    tokenizer: DeepSeekExactTokenizer,
    label: str,
) -> int:
    request = result.request
    if not result.content:
        raise ExternalE6Error(f"{label} response content is empty")
    if result.prompt_bytes != expected_prompt.encode("utf-8"):
        raise ExternalE6Error(f"{label} prompt bytes differ from the packed evidence")
    if (
        request.model != DEEPSEEK_MODEL
        or request.temperature != 0.0
        or request.max_tokens != expected_max_tokens
        or request.thinking_mode != "disabled"
        or result.endpoint_url != DEEPSEEK_DEPLOYMENT_ID
        or result.response_model != DEEPSEEK_MODEL
        or not result.request_id
        or not 1 <= result.attempts <= QA_HTTP_ATTEMPTS
    ):
        raise ExternalE6Error(f"{label} provider request/response configuration drifted")
    exact = tokenizer.exact_count(expected_prompt)
    verify_provider_prompt_tokens(result, expected=exact, label=label)
    return exact


def _validate_qa_receipt_prefix(
    context: E6Context,
    question: SelectedQuestion,
    *,
    prompt_row: Mapping[str, Any],
    receipts: Sequence[dict[str, Any]],
    tokenizer: DeepSeekExactTokenizer,
) -> dict[tuple[str, str], ChatResult]:
    expected = _qa_expected_routes(context, question)
    if len(receipts) > len(expected):
        raise ExternalE6Error("E6 QA receipt prefix exceeds the frozen route count")
    actual = tuple(
        (str(record.get("question_id")), str(record.get("call_role"))) for record in receipts
    )
    expected_pairs = tuple((receipt_id, role) for _, receipt_id, role in expected)
    if actual != expected_pairs[: len(actual)]:
        raise ExternalE6Error("partial E6 QA receipts violate the frozen route order")
    results: dict[tuple[str, str], ChatResult] = {}
    for route_index, record in enumerate(receipts):
        cell, receipt_id, role = expected[route_index]
        result = validate_chat_receipt_record(record)
        if role == "reader":
            prompt = str(prompt_row["arms"][cell]["prompt"])
            maximum = READER_MAX_TOKENS
        else:
            reader_key = (receipt_id, "reader")
            if reader_key not in results:
                raise ExternalE6Error("E6 QA judge receipt has no preceding reader receipt")
            prompt = _judge_text_for_arm(question, results[reader_key].content)
            maximum = JUDGE_MAX_TOKENS
        _validate_qa_chat(
            result,
            expected_prompt=prompt,
            expected_max_tokens=maximum,
            tokenizer=tokenizer,
            label=f"{question.question_id}/{cell}/{role}",
        )
        key = (receipt_id, role)
        if key in results:
            raise ExternalE6Error("E6 QA receipt route is duplicated")
        results[key] = result
    return results


def _qa_artifact(
    context: E6Context,
    question: SelectedQuestion,
    *,
    pack_row: Mapping[str, Any],
    prompt_row: Mapping[str, Any],
    receipts: Sequence[dict[str, Any]],
    tokenizer: DeepSeekExactTokenizer,
) -> dict[str, Any]:
    if len(receipts) != len(_qa_expected_routes(context, question)):
        raise ExternalE6Error("E6 QA receipt coverage differs from the frozen route count")
    results = _validate_qa_receipt_prefix(
        context,
        question,
        prompt_row=prompt_row,
        receipts=receipts,
        tokenizer=tokenizer,
    )
    receipt_sha: dict[tuple[str, str], str] = {}
    for record in receipts:
        key = (str(record["question_id"]), str(record["call_role"]))
        receipt_sha[key] = sha256_json(record)
    arms: dict[str, Any] = {}
    total_cost = 0
    for cell in E6_CELLS:
        receipt_id = _qa_receipt_id(question.question_id, cell)
        reader_key = (receipt_id, "reader")
        judge_key = (receipt_id, "development_judge")
        if reader_key not in results or judge_key not in results:
            raise ExternalE6Error("E6 QA receipts do not cover every arm and role")
        reader = results[reader_key]
        judge = results[judge_key]
        prompt = str(prompt_row["arms"][cell]["prompt"])
        reader_tokens = _validate_qa_chat(
            reader,
            expected_prompt=prompt,
            expected_max_tokens=READER_MAX_TOKENS,
            tokenizer=tokenizer,
            label=f"{question.question_id}/{cell}/reader",
        )
        judge_prompt = _judge_text_for_arm(question, reader.content)
        judge_tokens = _validate_qa_chat(
            judge,
            expected_prompt=judge_prompt,
            expected_max_tokens=JUDGE_MAX_TOKENS,
            tokenizer=tokenizer,
            label=f"{question.question_id}/{cell}/development-judge",
        )
        reader_cost = _chat_cost_microusd(reader, pricing=context.pricing)
        judge_cost = _chat_cost_microusd(judge, pricing=context.pricing)
        total_cost += reader_cost + judge_cost
        arms[cell] = {
            "hypothesis": reader.content,
            "development_judge_text": judge.content,
            "development_label": judge_label(judge.content),
            "reader": {
                "exact_local_prompt_tokens": reader_tokens,
                "api_prompt_tokens": reader.prompt_tokens,
                "completion_tokens": reader.completion_tokens,
                "total_tokens": reader.total_tokens,
                "finish_reason": reader.finish_reason,
                "latency_microseconds": int(math.ceil(reader.latency_ms * 1000.0)),
                "attempts": reader.attempts,
                "provider_request_id_sha256": sha256_bytes(str(reader.request_id).encode("utf-8")),
                "raw_request_sha256": reader.raw_request_sha256,
                "raw_response_sha256": reader.raw_response_sha256,
                "receipt_sha256": receipt_sha[reader_key],
                "cost_microusd": reader_cost,
            },
            "development_judge": {
                "exact_local_prompt_tokens": judge_tokens,
                "api_prompt_tokens": judge.prompt_tokens,
                "completion_tokens": judge.completion_tokens,
                "total_tokens": judge.total_tokens,
                "finish_reason": judge.finish_reason,
                "latency_microseconds": int(math.ceil(judge.latency_ms * 1000.0)),
                "attempts": judge.attempts,
                "provider_request_id_sha256": sha256_bytes(str(judge.request_id).encode("utf-8")),
                "raw_request_sha256": judge.raw_request_sha256,
                "raw_response_sha256": judge.raw_response_sha256,
                "receipt_sha256": receipt_sha[judge_key],
                "cost_microusd": judge_cost,
            },
            "cost_microusd": reader_cost + judge_cost,
        }
    raw_receipts = _receipt_bytes(receipts)
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e6b-head20-deepseek-qa-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E6_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_pack_artifact_sha256": pack_row["artifact_sha256"],
        "source_prompt_artifact_sha256": prompt_row["artifact_sha256"],
        "receipt_artifact_sha256": sha256_bytes(raw_receipts),
        "receipt_artifact_bytes": len(raw_receipts),
        "arm_order": list(_qa_arm_order(context, question)),
        "arms": arms,
        "cost_microusd": total_cost,
        "claims": {
            "reader_model": DEEPSEEK_MODEL,
            "development_judge_model": DEEPSEEK_MODEL,
            "thinking_disabled": True,
            "api_and_local_prompt_tokens_reconciled": True,
            "official_gpt4o_judge_executed": False,
            "official_longmemeval_score": False,
            "eligible_for_promotion": False,
            "production_policy_changed": False,
        },
    }
    return seal_artifact(payload)


def replay_qa_question(
    context: E6Context,
    question: SelectedQuestion,
    *,
    qa_row: Any,
    receipts: Sequence[dict[str, Any]],
    tokenizer: DeepSeekExactTokenizer | None = None,
) -> dict[str, Any]:
    own_tokenizer = tokenizer is None
    if tokenizer is None:
        tokenizer = DeepSeekExactTokenizer(
            context.e1.deepseek_root,
            artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
        )
    del own_tokenizer  # retained to make the lazy boundary explicit
    pack = load_json(e6_phase_path(context, "pack", question), sealed=True)
    prompts = load_json(e6_phase_path(context, "prompts", question), sealed=True)
    expected = _qa_artifact(
        context,
        question,
        pack_row=pack,
        prompt_row=prompts,
        receipts=receipts,
        tokenizer=tokenizer,
    )
    if qa_row != expected:
        raise ExternalE6Error("saved E6 QA artifact differs from exact raw receipt replay")
    return expected


def _qa_journal_route(
    context: E6Context,
    question: SelectedQuestion,
    *,
    cell: str,
    role: str,
    route_index: int,
) -> str:
    return "qa:" + sha256_json(
        {
            "run_manifest_sha256": context.manifest["artifact_sha256"],
            "question_position": question.position,
            "question_id": question.question_id,
            "cell": cell,
            "receipt_id": _qa_receipt_id(question.question_id, cell),
            "call_role": role,
            "route_index": route_index,
        }
    )


def _expected_qa_reservation(
    context: E6Context,
    question: SelectedQuestion,
    *,
    cell: str,
    role: str,
    route_index: int,
    prompt: str,
    max_tokens: int,
    tokenizer: DeepSeekExactTokenizer,
) -> tuple[_CallJournalPaths, dict[str, Any], bytes, int, int]:
    raw_request = chat_request_bytes(
        prompt=prompt,
        model=DEEPSEEK_MODEL,
        temperature=0.0,
        max_tokens=max_tokens,
        thinking_mode="disabled",
    )
    exact = tokenizer.exact_count(prompt)
    reserved = context.pricing.upper_bound_microusd(
        input_tokens=exact,
        output_tokens=max_tokens,
        retry_count=QA_HTTP_ATTEMPTS - 1,
        request_max_tokens=max_tokens,
    )
    route = _qa_journal_route(
        context,
        question,
        cell=cell,
        role=role,
        route_index=route_index,
    )
    paths = _call_journal_paths(context, namespace="qa", route=route)
    artifact = _reservation_artifact(
        context,
        namespace="qa",
        route=route,
        raw_request_sha256=sha256_bytes(raw_request),
        exact_prompt_tokens=exact,
        request_max_tokens=max_tokens,
        reserved_microusd=reserved,
    )
    return paths, artifact, raw_request, exact, reserved


def _settle_qa_response(
    context: E6Context,
    paths: _CallJournalPaths,
    *,
    reservation_artifact: Mapping[str, Any],
    response_artifact: Mapping[str, Any],
    result: ChatResult,
) -> int:
    actual = _chat_cost_microusd(result, pricing=context.pricing)
    _write_settlement(
        paths,
        _settlement_artifact(
            context,
            reservation=reservation_artifact,
            evidence_sha256=response_artifact["artifact_sha256"],
            actual_microusd=actual,
        ),
    )
    return actual


def _reconcile_qa_question_journals(
    context: E6Context,
    question: SelectedQuestion,
    *,
    tokenizer: DeepSeekExactTokenizer,
) -> tuple[dict[str, Any], ...]:
    prompt_row = load_json(e6_phase_path(context, "prompts", question), sealed=True)
    receipt_path = e6_jsonl_path(context, "qa-receipts", question)
    receipts = list(_load_receipts(receipt_path)) if receipt_path.exists() else []
    results = _validate_qa_receipt_prefix(
        context,
        question,
        prompt_row=prompt_row,
        receipts=receipts,
        tokenizer=tokenizer,
    )
    routes = _qa_expected_routes(context, question)
    for route_index, (cell, receipt_id, role) in enumerate(routes):
        if role == "reader":
            prompt = str(prompt_row["arms"][cell]["prompt"])
            maximum = READER_MAX_TOKENS
        else:
            reader = results.get((receipt_id, "reader"))
            prompt = "" if reader is None else _judge_text_for_arm(question, reader.content)
            maximum = JUDGE_MAX_TOKENS
        if not prompt:
            paths = _call_journal_paths(
                context,
                namespace="qa",
                route=_qa_journal_route(
                    context,
                    question,
                    cell=cell,
                    role=role,
                    route_index=route_index,
                ),
            )
            if paths.reservation.exists() or paths.response.exists() or paths.settlement.exists():
                raise ExternalE6Error("QA judge journal state exists before its reader")
            if route_index < len(receipts):
                raise ExternalE6Error("QA receipt prefix contains a judge without its reader")
            continue
        paths, expected_reservation, raw_request, _, _ = _expected_qa_reservation(
            context,
            question,
            cell=cell,
            role=role,
            route_index=route_index,
            prompt=prompt,
            max_tokens=maximum,
            tokenizer=tokenizer,
        )
        if not paths.reservation.exists():
            if paths.response.exists() or paths.settlement.exists():
                raise ExternalE6Error("QA journal contains an orphan response or settlement")
            if route_index < len(receipts):
                raise ExternalE6Error("QA receipt has no durable call reservation")
            continue
        reservation_artifact = load_json(paths.reservation, sealed=True)
        if reservation_artifact != expected_reservation:
            raise ExternalE6Error("QA call reservation differs from frozen routing")
        if not paths.response.exists():
            if paths.settlement.exists():
                raise ExternalE6Error("QA settlement has no provider response WAL")
            raise ExternalE6Error("unresolved QA reservation has no response; refusing to reissue")
        response_artifact, result = _replay_call_response(
            context,
            paths,
            reservation=reservation_artifact,
        )
        if result.raw_request != raw_request:
            raise ExternalE6Error("QA response WAL request differs from frozen prompt bytes")
        _settle_qa_response(
            context,
            paths,
            reservation_artifact=reservation_artifact,
            response_artifact=response_artifact,
            result=result,
        )
        _validate_qa_chat(
            result,
            expected_prompt=prompt,
            expected_max_tokens=maximum,
            tokenizer=tokenizer,
            label=f"{question.question_id}/{cell}/{role}",
        )
        record = chat_receipt_record(receipt_id, role, result)
        if route_index < len(receipts):
            if receipts[route_index] != record:
                raise ExternalE6Error("QA response WAL differs from its receipt sidecar")
        elif route_index == len(receipts):
            receipts.append(record)
            _durable_atomic_write(receipt_path, _receipt_bytes(receipts))
        else:
            raise ExternalE6Error("QA response WAL skips a frozen route")
        results[(receipt_id, role)] = result
    _validate_qa_receipt_prefix(
        context,
        question,
        prompt_row=prompt_row,
        receipts=receipts,
        tokenizer=tokenizer,
    )
    return tuple(receipts)


def _reconcile_qa_journals(
    context: E6Context,
    selected: Sequence[SelectedQuestion],
    *,
    tokenizer: DeepSeekExactTokenizer,
) -> None:
    for question in selected:
        _reconcile_qa_question_journals(context, question, tokenizer=tokenizer)


def _qa_durable_state_exists(context: E6Context) -> bool:
    roots = (
        context.output_dir / "qa",
        context.output_dir / "qa-receipts",
        context.output_dir / "external-call-journal" / "qa",
    )
    return any(root.exists() and any(root.iterdir()) for root in roots)


def _journal_namespace_cost(context: E6Context, namespace: str) -> int:
    _external_journal_cost(context)
    root = context.output_dir / "external-call-journal" / namespace
    if not root.exists():
        return 0
    total = 0
    for reservation_path in sorted(root.glob("*.reservation.json")):
        reservation = load_json(reservation_path, sealed=True)
        settlement_path = reservation_path.with_name(
            reservation_path.name.replace(".reservation.json", ".settlement.json")
        )
        if settlement_path.exists():
            total += int(load_json(settlement_path, sealed=True)["actual_microusd"])
        else:
            total += int(reservation["reserved_microusd"])
    return total


def _journal_artifact_sha256s(context: E6Context, suffix: str) -> list[str]:
    root = context.output_dir / "external-call-journal"
    if not root.exists():
        return []
    return [
        str(load_json(path, sealed=True)["artifact_sha256"])
        for path in sorted(root.glob(f"*/*.{suffix}.json"))
    ]


def _validate_expected_journal_bindings(
    context: E6Context,
    *,
    namespace: str,
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    root = context.output_dir / "external-call-journal" / namespace
    reservation_paths = sorted(root.glob("*.reservation.json")) if root.exists() else []
    actual: dict[str, tuple[_CallJournalPaths, dict[str, Any]]] = {}
    for reservation_path in reservation_paths:
        reservation = load_json(reservation_path, sealed=True)
        route = str(reservation.get("route"))
        if route in actual:
            raise ExternalE6Error("external call journal route is duplicated")
        paths = _call_journal_paths(context, namespace=namespace, route=route)
        if paths.reservation != reservation_path:
            raise ExternalE6Error("external call journal route filename drifted")
        actual[route] = (paths, reservation)
    if set(actual) != set(expected):
        raise ExternalE6Error(f"{namespace} journal routes do not match domain evidence one-to-one")
    for route, binding in expected.items():
        paths, reservation = actual[route]
        if not paths.response.exists() or not paths.settlement.exists():
            raise ExternalE6Error("expected external call route is not fully settled")
        _, result = _replay_call_response(context, paths, reservation=reservation)
        observed = {
            "raw_request_sha256": result.raw_request_sha256,
            "raw_response_sha256": result.raw_response_sha256,
            "attempts": result.attempts,
            "latency_microseconds": int(math.ceil(result.latency_ms * 1000.0)),
            "exact_local_prompt_tokens": reservation["exact_local_prompt_tokens"],
            "request_max_tokens": reservation["request_max_tokens"],
            "reserved_microusd": reservation["reserved_microusd"],
            "actual_microusd": int(load_json(paths.settlement, sealed=True)["actual_microusd"]),
        }
        expected_core = {
            key: binding[key]
            for key in (
                "raw_request_sha256",
                "raw_response_sha256",
                "attempts",
                "latency_microseconds",
                "exact_local_prompt_tokens",
                "request_max_tokens",
                "reserved_microusd",
                "actual_microusd",
            )
        }
        if observed != expected_core:
            raise ExternalE6Error("external call journal differs from domain raw evidence")
        receipt_sha256 = binding.get("receipt_sha256")
        if receipt_sha256 is not None:
            receipt = chat_receipt_record(
                str(binding["receipt_id"]),
                str(binding["role"]),
                result,
            )
            if sha256_json(receipt) != receipt_sha256:
                raise ExternalE6Error("QA journal response differs from its exact receipt row")


async def _complete_budgeted_chat(
    context: E6Context,
    question: SelectedQuestion,
    client: _QAFirst2xxChatClient,
    prompt: str,
    *,
    cell: str,
    role: str,
    route_index: int,
    receipt_id: str,
    tokenizer: DeepSeekExactTokenizer,
    ledger: _SpendLedger,
) -> tuple[ChatResult, dict[str, Any]]:
    paths, reservation_artifact, raw_request, _, reservation = _expected_qa_reservation(
        context,
        question,
        cell=cell,
        role=role,
        route_index=route_index,
        prompt=prompt,
        max_tokens=client.max_tokens,
        tokenizer=tokenizer,
    )
    if paths.reservation.exists() or paths.response.exists() or paths.settlement.exists():
        raise ExternalE6Error("live QA route already has journal state; reconciliation failed")
    await ledger.reserve(reservation)
    try:
        _write_reservation(context, paths, reservation_artifact)
    except Exception:
        await ledger.release(reservation)
        raise
    try:

        def checkpoint_first_2xx(
            raw_response: bytes,
            http_attempts: int,
            latency_ms: float,
            content_encoding: str,
        ) -> None:
            _write_call_response(
                paths,
                _raw_call_response_artifact(
                    context,
                    reservation=reservation_artifact,
                    raw_request=raw_request,
                    raw_response=raw_response,
                    endpoint_url=DEEPSEEK_DEPLOYMENT_ID,
                    attempts=http_attempts,
                    latency_ms=latency_ms,
                    content_encoding=content_encoding,
                ),
            )

        result = await client.complete(
            prompt,
            raw_request=raw_request,
            on_first_2xx=checkpoint_first_2xx,
        )
    except Exception:
        await ledger.charge_failed_reservation(reservation)
        raise
    try:
        response_artifact = _call_response_artifact(
            context,
            reservation=reservation_artifact,
            result=result,
        )
        _write_call_response(paths, response_artifact)
    except Exception:
        await ledger.charge_failed_reservation(reservation)
        raise
    if result.raw_request != raw_request:
        await ledger.charge_failed_reservation(reservation)
        raise ExternalE6Error("QA provider request bytes differ from frozen request bytes")
    actual = _settle_qa_response(
        context,
        paths,
        reservation_artifact=reservation_artifact,
        response_artifact=response_artifact,
        result=result,
    )
    await ledger.settle(
        reserved=reservation,
        actual=actual,
    )
    _validate_qa_chat(
        result,
        expected_prompt=prompt,
        expected_max_tokens=client.max_tokens,
        tokenizer=tokenizer,
        label=f"{question.question_id}/{cell}/{role}",
    )
    record = chat_receipt_record(receipt_id, role, result)
    validate_chat_receipt_record(record)
    return result, record


async def _run_qa_async(
    context: E6Context,
    *,
    selected: Sequence[SelectedQuestion],
    tokenizer: DeepSeekExactTokenizer,
    base_url: str,
    api_key: str,
) -> None:
    reader_client = _QAFirst2xxChatClient(
        api_key=api_key,
        max_tokens=READER_MAX_TOKENS,
    )
    judge_client = _QAFirst2xxChatClient(
        api_key=api_key,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    _reconcile_qa_journals(context, selected, tokenizer=tokenizer)
    journal_cost, unresolved = _external_journal_cost(context)
    if unresolved:
        raise ExternalE6Error("unresolved external call reservations block QA resume")
    ledger = _SpendLedger(
        spent_microusd=journal_cost,
        maximum_microusd=MAX_EXTERNAL_COST_MICROUSD,
    )
    try:
        for completed, question in enumerate(selected, start=1):
            qa_path = e6_phase_path(context, "qa", question)
            receipt_path = e6_jsonl_path(context, "qa-receipts", question)
            if qa_path.exists():
                if not receipt_path.exists():
                    raise ExternalE6Error("E6 QA artifact has no raw receipt sidecar")
                qa = load_json(qa_path, sealed=True)
                receipts = _load_receipts(receipt_path)
                replay_qa_question(
                    context,
                    question,
                    qa_row=qa,
                    receipts=receipts,
                    tokenizer=tokenizer,
                )
                print(
                    f"  qa: {completed}/{len(selected)} verified {question.question_id}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            pack = load_json(e6_phase_path(context, "pack", question), sealed=True)
            prompts = load_json(e6_phase_path(context, "prompts", question), sealed=True)
            receipts = list(_load_receipts(receipt_path)) if receipt_path.exists() else []
            _validate_qa_receipt_prefix(
                context,
                question,
                prompt_row=prompts,
                receipts=receipts,
                tokenizer=tokenizer,
            )
            expected_routes = _qa_expected_routes(context, question)
            while len(receipts) < len(expected_routes):
                route_index = len(receipts)
                cell, receipt_id, role = expected_routes[route_index]
                if role == "reader":
                    prompt = str(prompts["arms"][cell]["prompt"])
                    result, record = await _complete_budgeted_chat(
                        context,
                        question,
                        reader_client,
                        prompt,
                        cell=cell,
                        role=role,
                        route_index=route_index,
                        receipt_id=receipt_id,
                        tokenizer=tokenizer,
                        ledger=ledger,
                    )
                else:
                    reader_record = receipts[route_index - 1]
                    reader = validate_chat_receipt_record(reader_record)
                    prompt = _judge_text_for_arm(question, reader.content)
                    result, record = await _complete_budgeted_chat(
                        context,
                        question,
                        judge_client,
                        prompt,
                        cell=cell,
                        role=role,
                        route_index=route_index,
                        receipt_id=receipt_id,
                        tokenizer=tokenizer,
                        ledger=ledger,
                    )
                receipts.append(record)
                _durable_atomic_write(receipt_path, _receipt_bytes(receipts))
            qa = _qa_artifact(
                context,
                question,
                pack_row=pack,
                prompt_row=prompts,
                receipts=receipts,
                tokenizer=tokenizer,
            )
            write_json(qa_path, qa)
            replay_qa_question(
                context,
                question,
                qa_row=qa,
                receipts=receipts,
                tokenizer=tokenizer,
            )
            print(
                f"  qa: {completed}/{len(selected)} sealed {question.question_id} "
                f"cost=${qa['cost_microusd'] / 1_000_000:.6f} "
                f"ledger=${ledger.spent_microusd / 1_000_000:.6f}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        await reader_client.aclose()
        await judge_client.aclose()


def _qa_completion_artifact(context: E6Context) -> dict[str, Any]:
    return seal_artifact(
        {
            "artifact_type": "swarmbrain-longmemeval-e6b-head20-qa-completion",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "protocol_version": E6_RUN_PROTOCOL_VERSION,
            "run_manifest_sha256": context.manifest["artifact_sha256"],
            "question_count": E6B_SAMPLE,
            "reader_calls": E6B_SAMPLE * 2,
            "development_judge_calls": E6B_SAMPLE * 2,
            "aggregate_QA_statistics_compiled": False,
            "individual_labels_disclosed_by_progress_log": False,
            "official_gpt4o_calls": 0,
        }
    )


def run_qa_phase(
    context: E6Context,
    *,
    base_url: str,
    api_key_env: str,
    limit: int | None = None,
) -> dict[str, Any]:
    if limit is not None:
        raise ExternalE6Error("optional QA is forbidden on a limited reliability smoke")
    completed_qa = sum(
        e6_phase_path(context, "qa", question).is_file() for question in context.e1.selected
    )
    if _qa_durable_state_exists(context):
        diagnostic = load_json(context.output_dir / "diagnostic.json", sealed=True)
        if completed_qa != E6B_SAMPLE and diagnostic.get("qa", {}).get("available") is not False:
            raise ExternalE6Error("partial QA resume requires the frozen context-only diagnostic")
    else:
        diagnostic = build_diagnostic_report(context)
    context_gate = _e6b_context_gate_evidence(diagnostic)
    if context_gate["passed"] is not True:
        if _qa_durable_state_exists(context):
            raise ExternalE6Error("context early-stop is incompatible with retained QA state")
        print(
            "  qa: E6b context gate rejected R1H20; no reader or judge calls executed",
            file=sys.stderr,
            flush=True,
        )
        return diagnostic
    if base_url.strip().rstrip("/") not in {
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
    }:
        raise ExternalE6Error("E6 QA is frozen to the official DeepSeek endpoint")
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise ExternalE6Error(f"environment variable {api_key_env!r} is missing")
    tokenizer = DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    asyncio.run(
        _run_qa_async(
            context,
            selected=context.e1.selected,
            tokenizer=tokenizer,
            base_url=base_url,
            api_key=api_key,
        )
    )
    qa_completion = _qa_completion_artifact(context)
    write_json(context.output_dir / "qa-completion.json", qa_completion)
    return qa_completion


def _e6b_context_gate_evidence(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    quality = diagnostic.get("context_quality")
    efficiency = diagnostic.get("efficiency")
    cases = diagnostic.get("cases")
    if not isinstance(quality, dict) or quality.get("available") is not True:
        raise ExternalE6Error("E6b context gate requires complete gold-context evidence")
    if not isinstance(efficiency, dict) or not isinstance(cases, list):
        raise ExternalE6Error("E6b context gate evidence is malformed")
    if len(cases) != E6B_SAMPLE:
        raise ExternalE6Error("E6b context gate requires all 160 preregistered cases")
    gold_eligible_cases = int(quality.get("gold_eligible_cases", -1))
    arms = quality.get("arms")
    efficiency_arms = efficiency.get("arms")
    if not isinstance(arms, dict) or not isinstance(efficiency_arms, dict):
        raise ExternalE6Error("E6b context summaries have no paired arms")
    r0 = arms[RepresentationCell.RAW.value]
    r1 = arms[RepresentationCell.RAW_MERGED_SFK.value]
    noninferiority_axes = (
        "candidate_any_gold_in_context",
        "candidate_all_gold_in_context",
        "candidate_answer_session_recall",
        "prompt_any_gold_in_context",
        "prompt_all_gold_in_context",
        "prompt_answer_session_recall",
    )
    strict_mrr_axes = (
        "candidate_answer_session_mrr",
        "prompt_answer_session_mrr",
    )
    noninferiority = {
        axis: {
            "R0": float(r0[axis]),
            "R1H20": float(r1[axis]),
            "delta": float(r1[axis]) - float(r0[axis]),
            "passed": float(r1[axis]) >= float(r0[axis]),
        }
        for axis in noninferiority_axes
    }
    strict_mrr = {
        axis: {
            "R0": float(r0[axis]),
            "R1H20": float(r1[axis]),
            "delta": float(r1[axis]) - float(r0[axis]),
            "passed": float(r1[axis]) > float(r0[axis]),
        }
        for axis in strict_mrr_axes
    }
    r0_tokens = efficiency_arms[RepresentationCell.RAW.value]["prompt_tokens"]
    r1_tokens = efficiency_arms[RepresentationCell.RAW_MERGED_SFK.value]["prompt_tokens"]
    prompt_efficiency = {
        "total": {
            "R0": int(r0_tokens["total"]),
            "R1H20": int(r1_tokens["total"]),
            "passed": int(r1_tokens["total"]) <= int(r0_tokens["total"]),
        },
        "p95": {
            "R0": float(r0_tokens["p95"]),
            "R1H20": float(r1_tokens["p95"]),
            "passed": float(r1_tokens["p95"]) <= float(r0_tokens["p95"]),
        },
    }
    candidate_counts: dict[str, list[int]] = {cell: [] for cell in E6_CELLS}
    prompt_counts: dict[str, list[int]] = {cell: [] for cell in E6_CELLS}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("arms"), dict):
            raise ExternalE6Error("E6b diagnostic case is malformed")
        for cell in E6_CELLS:
            context = case["arms"][cell]["context"]
            candidate_counts[cell].append(int(context["candidate_value_count"]))
            prompt_counts[cell].append(int(context["prompt_value_count"]))
    exact_heads = all(
        count == HEAD_MATCHED_VALUE_COUNT
        for counts in candidate_counts.values()
        for count in counts
    )
    whole_turn_drops = {
        cell: {
            "cases_with_drops": sum(
                candidate > prompt
                for candidate, prompt in zip(
                    candidate_counts[cell], prompt_counts[cell], strict=True
                )
            ),
            "dropped_values": sum(
                candidate - prompt
                for candidate, prompt in zip(
                    candidate_counts[cell], prompt_counts[cell], strict=True
                )
            ),
            "minimum_prompt_value_count": min(prompt_counts[cell]),
        }
        for cell in E6_CELLS
    }
    if any(value["dropped_values"] < 0 for value in whole_turn_drops.values()):
        raise ExternalE6Error("E6b prompt contains more values than its candidate head")
    passed = bool(
        gold_eligible_cases == E6B_SAMPLE - E6B_ABS_COUNT
        and exact_heads
        and all(row["passed"] for row in noninferiority.values())
        and all(row["passed"] for row in strict_mrr.values())
        and all(row["passed"] for row in prompt_efficiency.values())
    )
    return {
        "gate": "G1-context-quality-and-token-parity",
        "passed": passed,
        "gold_eligible_non_abstention_cases": gold_eligible_cases,
        "required_gold_eligible_non_abstention_cases": E6B_SAMPLE - E6B_ABS_COUNT,
        "candidate_and_prompt_noninferiority": noninferiority,
        "strict_candidate_and_prompt_mrr_improvement": strict_mrr,
        "prompt_token_noninferiority": prompt_efficiency,
        "candidate_head_count": HEAD_MATCHED_VALUE_COUNT,
        "candidate_heads_exact_for_every_case_and_arm": exact_heads,
        "whole_turn_budget_drops": whole_turn_drops,
    }


def _e6b_arm_outcome(case: Mapping[str, Any], cell: str) -> ArmOutcome:
    arm = case["arms"][cell]
    correct = arm.get("qa_correct")
    if not isinstance(correct, bool):
        raise ExternalE6Error("E6b final QA gate requires a Boolean label for every arm")
    context = arm["context"]
    prompt_context = context["prompt"]
    any_gold = prompt_context["any_gold_in_context"]
    all_gold = prompt_context["all_gold_in_context"]
    answer_mrr = prompt_context["answer_session_mrr"]
    operational = arm["accounting"]["construction_plus_query"]
    operational_latency_ms = float(operational["latency_microseconds"]) / 1000.0
    operational_cost_usd = float(operational["cost_microusd"]) / 1_000_000.0
    return ArmOutcome(
        correct=correct,
        any_gold_in_context=None if any_gold is None else bool(any_gold),
        all_gold_in_context=None if all_gold is None else bool(all_gold),
        answer_session_mrr=None if answer_mrr is None else float(answer_mrr),
        prompt_tokens=int(context["prompt_tokens"]),
        operational_latency_ms=operational_latency_ms,
        end_to_end_latency_ms=operational_latency_ms,
        construction_plus_query_cost_usd=operational_cost_usd,
        reader_cost_usd=0.0,
        judge_cost_usd=0.0,
        total_cost_usd=operational_cost_usd,
        accounting={},
    )


def _e6b_qa_gate_evidence(
    diagnostic: Mapping[str, Any],
    *,
    integrity_gate_passed: bool,
    context_gate_passed: bool,
) -> dict[str, Any]:
    qa = diagnostic.get("qa")
    if not isinstance(qa, dict) or qa.get("available") is not True:
        return {
            "gate": "G2-paired-deepseek-development-qa",
            "available": False,
            "passed": False,
            "reason": "paired-QA-not-executed",
        }
    case_rows = diagnostic.get("cases")
    if not isinstance(case_rows, list) or len(case_rows) != E6B_SAMPLE:
        raise ExternalE6Error("E6b QA gate requires exactly 160 diagnostic cases")
    cases = tuple(
        PairedQACase(
            question_id=str(case["question_id"]),
            question_type=str(case["question_type"]),
            baseline=_e6b_arm_outcome(case, RepresentationCell.RAW.value),
            candidate=_e6b_arm_outcome(case, RepresentationCell.RAW_MERGED_SFK.value),
        )
        for case in case_rows
    )
    paired = paired_qa_summary(cases)
    by_type = qa_by_question_type(cases)
    if set(by_type) != set(E6B_TARGET_TYPE_COUNTS):
        raise ExternalE6Error("E6b QA strata differ from the six preregistered question types")
    type_nonregression = {
        name: {
            **row,
            "margin": -MAX_TYPE_REGRESSION,
            "passed": float(row["paired_delta"]) >= -MAX_TYPE_REGRESSION,
        }
        for name, row in by_type.items()
    }
    abs_cases = [case for case in cases if "_abs" in case.question_id]
    if len(abs_cases) != E6B_ABS_COUNT:
        raise ExternalE6Error("E6b abstention subgroup differs from the frozen ten cases")
    abs_r0_correct = sum(case.baseline.correct for case in abs_cases)
    abs_r1_correct = sum(case.candidate.correct for case in abs_cases)
    abs_delta = (abs_r1_correct - abs_r0_correct) / len(abs_cases)
    abs_nonregression = {
        "questions": len(abs_cases),
        "R0_correct": abs_r0_correct,
        "R1H20_correct": abs_r1_correct,
        "R0_accuracy": abs_r0_correct / len(abs_cases),
        "R1H20_accuracy": abs_r1_correct / len(abs_cases),
        "paired_delta": abs_delta,
        "improved_questions": sum(
            case.candidate.correct and not case.baseline.correct for case in abs_cases
        ),
        "regressed_questions": sum(
            case.baseline.correct and not case.candidate.correct for case in abs_cases
        ),
        "tied_questions": sum(
            case.baseline.correct == case.candidate.correct for case in abs_cases
        ),
        "margin": 0.0,
        "passed": abs_delta >= 0.0,
    }
    efficiency = diagnostic["efficiency"]["arms"]
    r0_efficiency = efficiency[RepresentationCell.RAW.value]
    r1_efficiency = efficiency[RepresentationCell.RAW_MERGED_SFK.value]
    pareto_values = {
        "development_accuracy": (
            float(paired["baseline"]["accuracy"]),
            float(paired["candidate"]["accuracy"]),
            "higher",
        ),
        "p95_prompt_tokens": (
            float(r0_efficiency["prompt_tokens"]["p95"]),
            float(r1_efficiency["prompt_tokens"]["p95"]),
            "lower",
        ),
        "p95_operational_latency_microseconds": (
            float(r0_efficiency["operational_latency_microseconds"]["p95"]),
            float(r1_efficiency["operational_latency_microseconds"]["p95"]),
            "lower",
        ),
        "total_construction_plus_query_cost_microusd": (
            float(r0_efficiency["construction_plus_query_cost_microusd"]["total"]),
            float(r1_efficiency["construction_plus_query_cost_microusd"]["total"]),
            "lower",
        ),
    }
    r0_no_worse = {
        name: left >= right if direction == "higher" else left <= right
        for name, (left, right, direction) in pareto_values.items()
    }
    r0_strict = {
        name: left > right if direction == "higher" else left < right
        for name, (left, right, direction) in pareto_values.items()
    }
    r0_dominates = all(r0_no_worse.values()) and any(r0_strict.values())
    overall_delta = float(paired["paired_delta"]["delta"])
    ci_low = float(paired["paired_delta"]["ci_low"])
    passed = bool(
        integrity_gate_passed
        and context_gate_passed
        and qa.get("complete_case_coverage") is True
        and int(qa.get("paired_cases", 0)) == E6B_SAMPLE
        and overall_delta > 0.0
        and ci_low > 0.0
        and all(row["passed"] for row in type_nonregression.values())
        and abs_nonregression["passed"]
        and not r0_dominates
    )
    return {
        "gate": "G2-paired-deepseek-development-qa",
        "available": True,
        "passed": passed,
        "G0_integrity_gate_passed": integrity_gate_passed,
        "G1_context_gate_passed": context_gate_passed,
        "overall": paired,
        "overall_delta_positive": overall_delta > 0.0,
        "bootstrap_lower_bound_strictly_positive": ci_low > 0.0,
        "bootstrap_contract_matches_existing_promotion_contract": bool(
            E6B_BOOTSTRAP_SAMPLES == SELECTION_BOOTSTRAP_RESAMPLES
            and E6B_BOOTSTRAP_SEED == SELECTION_BOOTSTRAP_SEED
            and SELECTION_BOOTSTRAP_CONFIDENCE == 0.95
        ),
        "by_question_type": type_nonregression,
        "maximum_type_regression": MAX_TYPE_REGRESSION,
        "all_question_types_noninferior_at_margin": all(
            row["passed"] for row in type_nonregression.values()
        ),
        "abstention_subgroup": abs_nonregression,
        "R0_pareto_dominates_R1H20": {
            "value": r0_dominates,
            "values": {
                name: {"R0": left, "R1H20": right, "direction": direction}
                for name, (left, right, direction) in pareto_values.items()
            },
            "R0_no_worse": r0_no_worse,
            "R0_strictly_better": r0_strict,
        },
    }


def build_report(context: E6Context) -> dict[str, Any]:
    operational_audit = load_operational_audit(context)
    tokenizer = DeepSeekExactTokenizer(
        context.e1.deepseek_root,
        artifact_sha256=_snapshot_artifact(context.e1, "deepseek_tokenizer"),
    )
    journal_total, unresolved_journals = _external_journal_cost(context)
    if unresolved_journals:
        raise ExternalE6Error("final E6 report forbids unresolved external call reservations")
    extraction_artifacts: list[str] = []
    ranking_artifacts: list[str] = []
    pack_artifacts: list[str] = []
    prompt_artifacts: list[str] = []
    case_artifacts: list[str] = []
    extraction_receipts: list[str] = []
    qa_artifacts: list[str] = []
    qa_receipts: list[str] = []
    extraction_cost = 0
    qa_cost = 0
    schema_retries = 0
    http_attempts = 0
    expected_extraction_journals: dict[str, dict[str, Any]] = {}
    expected_qa_journals: dict[str, dict[str, Any]] = {}
    provider_request_routes: dict[str, str] = {}

    def register_provider_request_id(provider_request_id: Any, *, route: str) -> None:
        if not isinstance(provider_request_id, str) or not provider_request_id.strip():
            raise ExternalE6Error("external response has no provider request ID")
        previous = provider_request_routes.get(provider_request_id)
        if previous is not None:
            raise ExternalE6Error(
                f"provider request ID crossed frozen external-call routes: {previous} and {route}"
            )
        provider_request_routes[provider_request_id] = route

    for question in context.e1.selected:
        extraction, evidences = replay_extraction_question(context, question)
        rank, _ = replay_rank_question(context, question)
        pack = load_json(e6_phase_path(context, "pack", question), sealed=True)
        prompts = load_json(e6_phase_path(context, "prompts", question), sealed=True)
        extraction_artifacts.append(extraction["artifact_sha256"])
        ranking_artifacts.append(rank["artifact_sha256"])
        pack_artifacts.append(pack["artifact_sha256"])
        prompt_artifacts.append(prompts["artifact_sha256"])
        raw_extraction = e6_jsonl_path(context, "extraction", question).read_bytes()
        extraction_receipts.append(sha256_bytes(raw_extraction))
        extraction_cost += int(extraction["accounting"]["cost_microusd"])
        schema_retries += sum(item.selected_application_attempt - 1 for item in evidences)
        http_attempts += int(extraction["accounting"]["http_attempts"])
        for source_position, evidence in enumerate(evidences):
            final_record = load_json(_value_record_path(context, question, source_position))
            attempt_records = final_record["application_attempts"]
            for application_attempt, (attempt, attempt_record) in enumerate(
                zip(evidence.application_attempts, attempt_records, strict=True),
                start=1,
            ):
                route = _extraction_journal_route(
                    question,
                    source_position,
                    application_attempt,
                )
                if route in expected_extraction_journals:
                    raise ExternalE6Error("expected extraction journal route is duplicated")
                register_provider_request_id(
                    attempt_record["response"]["provider_request_id"],
                    route=route,
                )
                request = replay_chat_request(attempt.raw_request)
                exact_prompt_tokens = tokenizer.exact_count(request.prompt)
                reserved_microusd = context.pricing.upper_bound_microusd(
                    input_tokens=exact_prompt_tokens,
                    output_tokens=request.max_tokens,
                    retry_count=DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS - 1,
                    request_max_tokens=request.max_tokens,
                )
                expected_extraction_journals[route] = {
                    "raw_request_sha256": sha256_bytes(attempt.raw_request),
                    "raw_response_sha256": sha256_bytes(attempt.raw_response),
                    "attempts": attempt.http_attempts,
                    "latency_microseconds": attempt.latency_microseconds,
                    "exact_local_prompt_tokens": exact_prompt_tokens,
                    "request_max_tokens": request.max_tokens,
                    "reserved_microusd": reserved_microusd,
                    "actual_microusd": _attempt_cost_microusd(
                        attempt_record,
                        pricing=context.pricing,
                    ),
                }
        qa_path = e6_phase_path(context, "qa", question)
        receipt_path = e6_jsonl_path(context, "qa-receipts", question)
        if qa_path.exists() != receipt_path.exists():
            raise ExternalE6Error("E6 report found partial QA evidence")
        if qa_path.exists():
            qa = load_json(qa_path, sealed=True)
            receipts = _load_receipts(receipt_path)
            replay_qa_question(
                context,
                question,
                qa_row=qa,
                receipts=receipts,
                tokenizer=tokenizer,
            )
            qa_artifacts.append(qa["artifact_sha256"])
            qa_receipts.append(sha256_bytes(_receipt_bytes(receipts)))
            qa_cost += int(qa["cost_microusd"])
            for route_index, ((cell, receipt_id, role), receipt) in enumerate(
                zip(_qa_expected_routes(context, question), receipts, strict=True)
            ):
                result = validate_chat_receipt_record(receipt)
                route = _qa_journal_route(
                    context,
                    question,
                    cell=cell,
                    role=role,
                    route_index=route_index,
                )
                if route in expected_qa_journals:
                    raise ExternalE6Error("expected QA journal route is duplicated")
                register_provider_request_id(result.request_id, route=route)
                request = result.request
                exact_prompt_tokens = tokenizer.exact_count(request.prompt)
                reserved_microusd = context.pricing.upper_bound_microusd(
                    input_tokens=exact_prompt_tokens,
                    output_tokens=request.max_tokens,
                    retry_count=DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS - 1,
                    request_max_tokens=request.max_tokens,
                )
                expected_qa_journals[route] = {
                    "raw_request_sha256": result.raw_request_sha256,
                    "raw_response_sha256": result.raw_response_sha256,
                    "attempts": result.attempts,
                    "latency_microseconds": int(math.ceil(result.latency_ms * 1000.0)),
                    "exact_local_prompt_tokens": exact_prompt_tokens,
                    "request_max_tokens": request.max_tokens,
                    "reserved_microusd": reserved_microusd,
                    "actual_microusd": _chat_cost_microusd(
                        result,
                        pricing=context.pricing,
                    ),
                    "receipt_id": receipt_id,
                    "role": role,
                    "receipt_sha256": sha256_json(receipt),
                }
    qa_complete = len(qa_artifacts) == len(context.e1.selected)
    qa_absent = not qa_artifacts
    qa_completion_path = context.output_dir / "qa-completion.json"
    qa_completion_artifacts: list[str] = []
    if qa_complete:
        qa_completion = load_json(qa_completion_path, sealed=True)
        if qa_completion != _qa_completion_artifact(context):
            raise ExternalE6Error("E6b QA completion artifact differs from frozen replay")
        qa_completion_artifacts.append(str(qa_completion["artifact_sha256"]))
    elif qa_completion_path.exists():
        raise ExternalE6Error("E6b QA completion artifact exists without complete paired QA")
    if not qa_complete and not qa_absent:
        raise ExternalE6Error("E6 report forbids partial paired QA coverage")
    _validate_expected_journal_bindings(
        context,
        namespace="extraction",
        expected=expected_extraction_journals,
    )
    _validate_expected_journal_bindings(
        context,
        namespace="qa",
        expected=expected_qa_journals,
    )
    extraction_journal_cost = _journal_namespace_cost(context, "extraction")
    qa_journal_cost = _journal_namespace_cost(context, "qa")
    if extraction_journal_cost != extraction_cost or qa_journal_cost != qa_cost:
        raise ExternalE6Error("external call journal cost differs from replayed domain receipts")
    if journal_total != extraction_cost + qa_cost:
        raise ExternalE6Error("global external call journal cost does not reconcile")
    if len(provider_request_routes) != len(expected_extraction_journals) + len(
        expected_qa_journals
    ):
        raise ExternalE6Error("provider request ID coverage differs from external routes")
    diagnostic_path = context.output_dir / "diagnostic.json"
    diagnostic = build_diagnostic_report(context, allow_complete_qa=True)
    if load_json(diagnostic_path, sealed=True) != diagnostic:
        raise ExternalE6Error("saved E6 diagnostic differs from reopened case compilation")
    case_artifacts = [
        str(load_json(e6_phase_path(context, "cases", question), sealed=True)["artifact_sha256"])
        for question in context.e1.selected
    ]
    context_gate = _e6b_context_gate_evidence(diagnostic)
    early_stop = not bool(context_gate["passed"])
    if early_stop and not qa_absent:
        raise ExternalE6Error("context early-stop forbids completed QA evidence")
    if early_stop and _qa_durable_state_exists(context):
        raise ExternalE6Error("context early-stop forbids any durable QA call state")
    if not early_stop and not qa_complete:
        raise ExternalE6Error("an advanced context gate requires complete paired QA evidence")
    source_sets = {
        "extraction": extraction_artifacts,
        "extraction_receipts": extraction_receipts,
        "ranking": ranking_artifacts,
        "pack": pack_artifacts,
        "prompts": prompt_artifacts,
        "cases": case_artifacts,
        "qa": qa_artifacts,
        "qa_receipts": qa_receipts,
        "qa_completion": qa_completion_artifacts,
        "external_call_reservations": _journal_artifact_sha256s(context, "reservation"),
        "external_call_response_wals": _journal_artifact_sha256s(context, "response"),
        "external_call_settlements": _journal_artifact_sha256s(context, "settlement"),
        "operational_audit": [str(operational_audit["artifact_sha256"])],
    }
    total_external_cost = extraction_cost + qa_cost
    g0_passed = bool(
        len(extraction_artifacts) == E6B_SAMPLE
        and len(ranking_artifacts) == E6B_SAMPLE
        and len(pack_artifacts) == E6B_SAMPLE
        and len(case_artifacts) == E6B_SAMPLE
        and not unresolved_journals
        and journal_total == total_external_cost
        and total_external_cost <= MAX_EXTERNAL_COST_MICROUSD
        and operational_audit.get("artifact_sha256")
        and len(provider_request_routes)
        == len(expected_extraction_journals) + len(expected_qa_journals)
    )
    qa_gate = _e6b_qa_gate_evidence(
        diagnostic,
        integrity_gate_passed=g0_passed,
        context_gate_passed=bool(context_gate["passed"]),
    )
    if not g0_passed:
        verdict = "incomplete-or-invalid-e6b-run"
    elif not context_gate["passed"]:
        verdict = "reject-R1H20-at-context-gate"
    elif not qa_gate["passed"]:
        verdict = "reject-R1H20-at-development-qa-gate"
    else:
        verdict = "retain-R1H20-for-separate-heldout-confirmation-only"
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e6b-head20-development-report",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": E6_RUN_PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "classification": "exploratory-development-diagnostic-not-official-score",
        "question_count": len(context.e1.selected),
        "cells": list(E6_CELLS),
        "diagnostic_artifact_sha256": diagnostic["artifact_sha256"],
        "diagnostic": diagnostic,
        "preflight_admission": context.manifest["packing"]["preflight_admission"],
        "execution": {
            "source_values": sum(len(question.turns) for question in context.e1.selected),
            "extraction_schema_retries": schema_retries,
            "extraction_http_attempts": http_attempts,
            "paired_qa_complete": qa_complete,
            "reader_calls": len(qa_artifacts) * 2,
            "development_judge_calls": len(qa_artifacts) * 2,
            "official_gpt4o_calls": 0,
            "official_gpt4o_reservations": 0,
            "provider_request_ids": len(provider_request_routes),
            "provider_request_ids_globally_unique": True,
            "provider_request_route_binding_sha256": sha256_json(
                [
                    {
                        "provider_request_id_sha256": sha256_bytes(
                            provider_request_id.encode("utf-8")
                        ),
                        "route": route,
                    }
                    for provider_request_id, route in sorted(provider_request_routes.items())
                ]
            ),
            "selection_outcomes_used_to_change_sample_or_protocol": False,
        },
        "external_cost": {
            "extraction_microusd": extraction_cost,
            "reader_and_development_judge_microusd": qa_cost,
            "total_microusd": total_external_cost,
            "total_usd_upper_bound": total_external_cost / 1_000_000,
            "hard_limit_microusd": MAX_EXTERNAL_COST_MICROUSD,
            "within_hard_limit": total_external_cost <= MAX_EXTERNAL_COST_MICROUSD,
            "billed_cost_claimed": False,
            "durable_journal_reconciled": True,
            "unresolved_reservations": 0,
        },
        "source_artifact_sets": {
            name: {
                "count": len(values),
                "ordered_sha256": sha256_json(values),
            }
            for name, values in source_sets.items()
        },
        "gates": {
            "G0_evidence_integrity_and_budget": {
                "passed": g0_passed,
                "complete_question_count": len(extraction_artifacts),
                "required_question_count": E6B_SAMPLE,
                "durable_journal_reconciled": journal_total == total_external_cost,
                "unresolved_reservations": unresolved_journals,
                "within_hard_cost_limit": total_external_cost <= MAX_EXTERNAL_COST_MICROUSD,
                "operational_audit_sha256": operational_audit["artifact_sha256"],
                "provider_request_ids_globally_unique": True,
                "official_gpt4o_reservations": 0,
                "official_gpt4o_calls": 0,
            },
            "G1_context_quality_and_token_parity": context_gate,
            "G2_paired_deepseek_development_qa": qa_gate,
        },
        "decision": {
            "verdict": verdict,
            "context_gate_advanced_to_paired_qa": not early_stop,
            "paired_deepseek_qa_executed": qa_complete,
            "eligible_for_composition": False,
            "eligible_for_serving_promotion": False,
            "eligible_for_official_gpt4o_confirmation": False,
            "required_next_if_retained": (
                "run a separately preregistered heldout task-or-corpus confirmation"
            ),
            "official_gpt4o_remains_deferred": True,
        },
        "claims": {
            "gold_used_only_for_posthoc_context_metrics_and_development_judging": True,
            "official_longmemeval_score": False,
            "official_gpt4o_judge_executed": False,
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
        choices=("extract", "rank", "pack", "audit", "diagnose", "qa", "report", "all"),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--e1-output-dir", type=Path, default=DEFAULT_E1_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_E6_OUTPUT)
    parser.add_argument("--sample", type=int, default=E6B_SAMPLE)
    parser.add_argument("--seed", type=int, default=E6B_SEED)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--qwen-root", type=Path, default=DEFAULT_QWEN_ROOT)
    parser.add_argument("--cross-encoder-root", type=Path, default=DEFAULT_CE_ROOT)
    parser.add_argument("--deepseek-root", type=Path, default=DEFAULT_DEEPSEEK_ROOT)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--qwen-batch-size", type=int, default=QWEN_BATCH_SIZE)
    parser.add_argument(
        "--extraction-concurrency",
        type=int,
        default=EXTRACTION_CONCURRENCY,
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return parser


def _execute_run(context: E6Context, args: argparse.Namespace) -> int:
    if args.phase == "all":
        phases = (
            ("extract", "rank", "pack", "audit")
            if args.limit is not None
            else ("extract", "rank", "pack", "diagnose", "qa", "report")
        )
    else:
        phases = (args.phase,)
    for phase in phases:
        if phase == "extract":
            run_extraction_phase(
                context,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                limit=args.limit,
            )
        elif phase == "rank":
            run_rank_phase(
                context,
                device=args.device,
                batch_size=args.qwen_batch_size,
                limit=args.limit,
            )
        elif phase == "pack":
            run_pack_phase(context, limit=args.limit)
        elif phase == "audit":
            if args.limit is None:
                raise ExternalE6Error("E6b operational audit requires --limit 40")
            report = build_operational_audit(context, limit=args.limit)
            print(json.dumps(report, indent=2, sort_keys=True))
        elif phase == "diagnose":
            completed_qa = sum(
                e6_phase_path(context, "qa", question).is_file() for question in context.e1.selected
            )
            if args.phase == "all" and _qa_durable_state_exists(context):
                report = load_json(context.output_dir / "diagnostic.json", sealed=True)
                if (
                    completed_qa != E6B_SAMPLE
                    and report.get("qa", {}).get("available") is not False
                ):
                    raise ExternalE6Error(
                        "partial QA resume lost its frozen context-only diagnostic"
                    )
            else:
                report = build_diagnostic_report(context, limit=args.limit)
            print(json.dumps(report, indent=2, sort_keys=True))
        elif phase == "qa":
            report = run_qa_phase(
                context,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                limit=args.limit,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            report = build_report(context)
            print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _write_incomplete_cost_status(
    context: E6Context,
    error: ExternalCostCapExceeded,
) -> dict[str, Any]:
    try:
        journal_cost, unresolved = _external_journal_cost(context)
    except ExternalCostCapExceeded:
        journal_cost = MAX_EXTERNAL_COST_MICROUSD + 1
        unresolved = 0
    status = seal_artifact(
        {
            "artifact_type": "swarmbrain-longmemeval-e6b-head20-terminal-status",
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "protocol_version": E6_RUN_PROTOCOL_VERSION,
            "run_manifest_sha256": context.manifest["artifact_sha256"],
            "status": "incomplete-cost-cap",
            "reason_sha256": sha256_bytes(str(error).encode("utf-8")),
            "journal_cost_or_reservation_microusd": journal_cost,
            "hard_limit_microusd": MAX_EXTERNAL_COST_MICROUSD,
            "unresolved_reservations": unresolved,
            "quality_inference_eligible": False,
            "eligible_for_composition": False,
            "eligible_for_serving_promotion": False,
            "eligible_for_official_gpt4o_confirmation": False,
            "official_gpt4o_calls": 0,
        }
    )
    path = context.output_dir / "status.json"
    if path.exists():
        if load_json(path, sealed=True) != status:
            raise ExternalE6Error("existing E6b terminal status differs from cost-cap replay")
    else:
        _durable_write_json(path, status)
    return status


def _run_locked(args: argparse.Namespace) -> int:
    context = build_e6_context(args)
    status_path = context.output_dir / "status.json"
    if status_path.exists():
        status = load_json(status_path, sealed=True)
        if status.get("status") != "incomplete-cost-cap":
            raise ExternalE6Error("E6b output contains an unknown terminal status")
        print(json.dumps(status, indent=2, sort_keys=True))
        return 2
    if args.limit is None:
        load_operational_audit(context)
    try:
        return _execute_run(context, args)
    except ExternalCostCapExceeded as exc:
        status = _write_incomplete_cost_status(context, exc)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.sample != E6B_SAMPLE or args.seed != E6B_SEED:
        raise SystemExit("E6b is frozen to the preregistered fresh 160-question cohort")
    if args.qwen_batch_size < 1 or args.extraction_concurrency < 1:
        raise SystemExit("batch size and extraction concurrency must be positive")
    if args.device != E6B_QWEN_DEVICE or args.qwen_batch_size != E6B_QWEN_BATCH_SIZE:
        raise SystemExit("E6b Qwen execution is frozen to MPS with batch size 8")
    if args.extraction_concurrency != EXTRACTION_CONCURRENCY:
        raise SystemExit(f"E6b extraction concurrency is frozen at {EXTRACTION_CONCURRENCY}")
    if Path(args.dataset).resolve() != Path(DEFAULT_DATASET).resolve():
        raise SystemExit("E6b dataset path is frozen by the preregistration")
    if Path(args.e1_output_dir).resolve() != DEFAULT_E1_OUTPUT.resolve():
        raise SystemExit("E6b E1 staging namespace is frozen by the preregistration")
    if Path(args.output_dir).resolve() != DEFAULT_E6_OUTPUT.resolve():
        raise SystemExit("E6b output namespace is frozen by the preregistration")
    if args.limit is not None and args.limit != 40:
        raise SystemExit("E6b permits only the frozen 40-question operational tranche")
    if args.limit is not None and args.phase in {"diagnose", "qa", "report"}:
        raise SystemExit("aggregate diagnostics and QA are forbidden on the operational tranche")
    if args.limit is None and args.phase == "audit":
        raise SystemExit("the operational audit requires --limit 40")
    with _output_process_lock(Path(args.output_dir)):
        return _run_locked(args)


if __name__ == "__main__":
    raise SystemExit(main())
