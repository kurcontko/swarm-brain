#!/usr/bin/env python3
"""Pinned real-model LongMemEval E1-A/E1-B development experiment.

This runner is intentionally separate from the production retrieval path.  It
executes the already-frozen evaluation contracts in five resumable phases:

``dense``
    Production-shaped lexical scoring plus Qwen3-Embedding over every turn in
    each selected question-local corpus, followed by deterministic E1-A RRF.
``cross-encoder``
    Mixedbread raw-logit scoring over exactly the fixed E1-A pool, followed by
    deterministic E1-B selection.
``pack``
    Exact DeepSeek-V4 chat-token packing of both E1-A and E1-B into the frozen
    8,192-token official reader prompt.
``qa``
    Paired DeepSeek-V4-Flash reader and development-judge calls with raw request
    and response receipts.  API prompt usage must equal the local exact count.
``report``
    A content-free, explicitly exploratory paired development report.

Model imports are lazy so repository tests do not require Torch or
Transformers.  The actual model phases are expected to run in the dedicated
evaluation environment documented by each emitted artifact.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from _longmemeval_common import (
    QWEN_QUERY_INSTRUCTION,
    select_questions,
)
from benchmarks.integrations.longmemeval_e1 import (
    ExternalScorerIdentity,
    PoolScoreObservation,
    ScoreChannel,
    TurnScoreObservation,
    bind_e1a_pool,
    select_e1b,
)
from benchmarks.integrations.longmemeval_turn_prompt import (
    TOKENIZER_PROTOCOL,
    ExactTokenCountReceipt,
    OrderedTurnBlocks,
    TokenizerIdentity,
    pack_turn_prompt,
)
from benchmarks.integrations.longmemeval_turn_retrieval import (
    DENSE_LANE_DEPTH,
    E1A_PROTOCOL,
    LEXICAL_LANE_DEPTH,
    ExternalLaneIdentity,
    ImmutableArtifactIdentity,
    RankedLaneObservation,
    RankedTurnObservation,
    RetrievalLane,
    fuse_question_turns,
)
from benchmarks.integrations.longmemeval_turns import (
    LongMemEvalTurnId,
    TurnProjection,
    TurnProjectionCorpus,
    compile_official_longmemeval_s,
)
from run_longmemeval_qa import (
    ChatClient,
    ChatResult,
    chat_receipt_record,
    is_abstention_question,
    judge_label,
    judge_prompt,
    validate_chat_receipt_record,
)

from swarmbrain.retrieval.projection import MAX_QUERY_CHARS, MAX_QUERY_TOKENS, normalize_term

ARTIFACT_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "swarmbrain-longmemeval-e1-real-model-development-v3"
DEFAULT_SAMPLE = 10
DEFAULT_SEED = 20260807
TOKEN_BUDGET = 8192
QWEN_MAX_LENGTH = 8192
CROSS_ENCODER_MAX_LENGTH = 512
QWEN_BATCH_SIZE = 8
CROSS_ENCODER_BATCH_SIZE = 8
QWEN_ATTENTION_CELL_BUDGET = 33_554_432
QWEN_MAX_PADDING_RATIO = 2.0
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"
QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
CROSS_ENCODER_MODEL = "mixedbread-ai/mxbai-rerank-large-v1"
CROSS_ENCODER_REVISION = "98f655841d5caf0b16eaff79c2b4ca109d920d17"
DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION = 0.14
DEEPSEEK_OUTPUT_USD_PER_MILLION = 0.28

DEFAULT_DATASET = Path("/private/tmp/longmemeval_s_cleaned.json")
DEFAULT_OUTPUT = Path("/private/tmp/swarmbrain-longmemeval-e1-pilot")
DEFAULT_QWEN_ROOT = Path("/private/tmp/swarmbrain-models/qwen3-embedding-0.6b")
DEFAULT_CE_ROOT = Path("/private/tmp/swarmbrain-models/mxbai-rerank-large-v1")
DEFAULT_DEEPSEEK_ROOT = Path("/private/tmp/swarmbrain-models/deepseek-v4-flash-tokenizer")

_TOKENS = re.compile(r"\w+")
_SAFE_QUESTION_ID = re.compile(r"[^A-Za-z0-9._-]")

QWEN_FILES: Mapping[str, tuple[int, str]] = {
    "README.md": (17237, "c34d9b7e5a267ad3fdd13227a253686bc90844ff4744a2a6a86c7c905e3d06f3"),
    "config.json": (727, "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9"),
    "config_sentence_transformers.json": (
        215,
        "10667c72ddb772627bf1780cb7f86af8e2ae0032b8c243c731172064105c6961",
    ),
    "generation_config.json": (
        117,
        "28396d421a2108acce96383f6a7de78008f7f1b17f807958f3c14c51dbfb65fb",
    ),
    "merges.txt": (
        1671853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    "model.safetensors": (
        1191586416,
        "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    ),
    "modules.json": (
        349,
        "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    ),
    "tokenizer.json": (
        11423705,
        "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    ),
    "tokenizer_config.json": (
        9706,
        "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    ),
    "vocab.json": (
        2776833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
}

CROSS_ENCODER_FILES: Mapping[str, tuple[int, str]] = {
    "LICENSE": (10762, "4b0dfefcb74f1e50a8df72a9f2bf0088753f8568bc479387292469b4948705d4"),
    "README.md": (49535, "08fdbe962874534777f7cbb616b3185221804aab07ffffa6feec459b6e13852f"),
    "added_tokens.json": (
        23,
        "dc046d04c9b0ada7ae6f1dc89c465801799acdf0c9a6aab8c15a1b2d5ca4e91f",
    ),
    "config.json": (970, "03da84d30a3abff6bd817cb504b63972a95cc6c2b8dcf7e195f5874fb4d18316"),
    "model.safetensors": (
        870174306,
        "03c87424793dbe55bdd8740ac2b72e33113cac0ff2b2c04c379844a4306ca990",
    ),
    "special_tokens_map.json": (
        970,
        "b2f1b2f15f29a6b6d9d6ea4eca1675d2c231a71477f151d48f79cc83a625ba21",
    ),
    "spm.model": (
        2464616,
        "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
    ),
    "tokenizer.json": (
        8649139,
        "305674b4d785287feecfb5f73f24aa75e9b57c87c579cfe24fbd207987d4b4c4",
    ),
    "tokenizer_config.json": (
        1447,
        "aafc9f36a056307bf0cbfcbd42fe00d9df89083d23db6114466c8bfaedb09ce5",
    ),
}

DEEPSEEK_FILES: Mapping[str, tuple[int, str]] = {
    "LICENSE": (1084, "f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040"),
    "encoding/README.md": (
        8118,
        "605363e9e43ee91beba88ea96c7806ce6ecdb2924e481459c9d16e1526470c10",
    ),
    "encoding/encoding_dsv4.py": (
        27908,
        "bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0",
    ),
    "inference/README.md": (
        951,
        "b9fe8027f91f2160b46582b61eaaf7b9873b3da36ee9f98bc76a900e7c1fed94",
    ),
    "inference/generate.py": (
        6296,
        "d4d443c0be8499b20ae5eaff0a623df02f47a8309be6feeba4eb4e0eeb5342c3",
    ),
    "tokenizer.json": (
        6367146,
        "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
    ),
    "tokenizer_config.json": (
        801,
        "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
    ),
}

IMPLEMENTATION_FILES = (
    "benchmarks/integrations/longmemeval_e1/contracts.py",
    "benchmarks/integrations/longmemeval_e1/selection.py",
    "benchmarks/integrations/longmemeval_turn_prompt/contracts.py",
    "benchmarks/integrations/longmemeval_turn_prompt/packer.py",
    "benchmarks/integrations/longmemeval_turn_retrieval/contracts.py",
    "benchmarks/integrations/longmemeval_turn_retrieval/fusion.py",
    "benchmarks/integrations/longmemeval_turns/compiler.py",
    "pyproject.toml",
    "scripts/_longmemeval_common.py",
    "scripts/run_longmemeval_e1_external.py",
    "scripts/run_longmemeval_qa.py",
    "src/swarmbrain/adapters/memory/in_memory.py",
    "src/swarmbrain/adapters/memory/retrieval.py",
    "src/swarmbrain/retrieval/projection.py",
    "uv.lock",
)


class ExternalE1Error(ValueError):
    """An input or saved artifact cannot support exact experiment replay."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ExternalE1Error("value is not finite canonical UTF-8 JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def seal_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "artifact_sha256" in payload:
        raise ExternalE1Error("unsealed payload cannot already carry artifact_sha256")
    copied = dict(payload)
    return {**copied, "artifact_sha256": sha256_json(copied)}


def validate_sealed_artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExternalE1Error("sealed artifact must be a JSON object")
    digest = value.get("artifact_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ExternalE1Error("sealed artifact has no SHA-256 binding")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if sha256_json(payload) != digest:
        raise ExternalE1Error("sealed artifact digest does not match its payload")
    return value


def _reject_constant(value: str) -> None:
    raise ExternalE1Error(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExternalE1Error(f"duplicate JSON field {key!r} is forbidden")
        result[key] = value
    return result


def load_json(path: Path, *, sealed: bool = False) -> Any:
    try:
        value = json.loads(
            path.read_bytes(),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_fields,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalE1Error(f"cannot read strict JSON artifact {path}") from exc
    return validate_sealed_artifact(value) if sealed else value


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_json_bytes(value) + b"\n")


def verify_snapshot(
    root: Path,
    *,
    name: str,
    revision: str,
    expected: Mapping[str, tuple[int, str]],
) -> dict[str, Any]:
    resolved_root = root.resolve()
    if root.is_symlink() or not resolved_root.is_dir():
        raise ExternalE1Error(f"model snapshot is missing or unsafe: {root}")
    files: dict[str, dict[str, int | str]] = {}
    for relative, (expected_bytes, expected_sha256) in sorted(expected.items()):
        path = resolved_root / relative
        if path.is_symlink() or not path.is_file():
            raise ExternalE1Error(f"pinned snapshot file is missing or unsafe: {path}")
        observed_bytes = path.stat().st_size
        observed_sha256 = _sha256_file(path)
        if (observed_bytes, observed_sha256) != (expected_bytes, expected_sha256):
            raise ExternalE1Error(f"pinned snapshot file drifted: {name}/{relative}")
        files[relative] = {"bytes": observed_bytes, "sha256": observed_sha256}
    tree_payload = {
        "model": name,
        "revision": revision,
        "files": files,
    }
    return {**tree_payload, "artifact_sha256": sha256_json(tree_payload)}


def implementation_fingerprint() -> dict[str, Any]:
    files: dict[str, str] = {}
    for relative in IMPLEMENTATION_FILES:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ExternalE1Error(f"implementation file is missing or unsafe: {relative}")
        files[relative] = _sha256_file(path)
    return {"files": files, "tree_sha256": sha256_json(files)}


def selected_positions(total: int, *, sample: int | None, seed: int) -> tuple[int, ...]:
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ExternalE1Error("dataset question count must be positive")
    if sample is None or sample >= total:
        return tuple(range(total))
    if isinstance(sample, bool) or not isinstance(sample, int) or sample < 1:
        raise ExternalE1Error("sample must be a positive integer")
    chosen = random.Random(seed).sample(range(total), sample)
    return tuple(sorted(chosen))


@dataclass(frozen=True, slots=True)
class SelectedQuestion:
    position: int
    record: dict[str, Any]
    turns: tuple[TurnProjection, ...]

    @property
    def question_id(self) -> str:
        return str(self.record["question_id"])

    @property
    def question(self) -> str:
        return str(self.record["question"])

    @property
    def current_date(self) -> str:
        return str(self.record["question_date"])


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    corpus: TurnProjectionCorpus
    records: tuple[dict[str, Any], ...]
    selected: tuple[SelectedQuestion, ...]
    manifest: dict[str, Any]
    output_dir: Path
    qwen_root: Path
    cross_encoder_root: Path
    deepseek_root: Path


def _safe_question_id(value: str) -> str:
    rendered = _SAFE_QUESTION_ID.sub("_", value)
    return rendered[:80] or "question"


def phase_path(context: ExperimentContext, phase: str, question: SelectedQuestion) -> Path:
    name = f"{question.position:03d}-{_safe_question_id(question.question_id)}.json"
    return context.output_dir / phase / name


def _read_dataset_records(path: Path) -> tuple[dict[str, Any], ...]:
    value = load_json(path)
    if not isinstance(value, list) or not value or any(not isinstance(row, dict) for row in value):
        raise ExternalE1Error("LongMemEval source must be a non-empty array of objects")
    return tuple(value)


def build_context(args: argparse.Namespace) -> ExperimentContext:
    dataset = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    qwen_root = Path(args.qwen_root).resolve()
    cross_encoder_root = Path(args.cross_encoder_root).resolve()
    deepseek_root = Path(args.deepseek_root).resolve()
    corpus = compile_official_longmemeval_s(dataset)
    records = _read_dataset_records(dataset)
    if len(records) != len(corpus.questions):
        raise ExternalE1Error("dataset records and F0 question bindings disagree")
    positions = selected_positions(len(records), sample=args.sample, seed=args.seed)
    selected_records = select_questions(records, sample=args.sample, seed=args.seed)
    if [records[position] for position in positions] != selected_records:
        raise ExternalE1Error("shared LongMemEval sample selector drifted")
    turns_by_question: dict[str, list[TurnProjection]] = {}
    for turn in corpus.turns:
        turns_by_question.setdefault(turn.turn_id.question_id, []).append(turn)
    selected = tuple(
        SelectedQuestion(
            position=position,
            record=records[position],
            turns=tuple(turns_by_question[str(records[position]["question_id"])]),
        )
        for position in positions
    )
    sample_binding = [
        {
            "position": question.position,
            "question_id": question.question_id,
            "question_type": str(question.record["question_type"]),
            "source_record": corpus.questions[question.position].source_record.as_dict(),
            "turns": len(question.turns),
        }
        for question in selected
    ]
    snapshots = {
        "qwen_embedding": verify_snapshot(
            qwen_root,
            name=QWEN_MODEL,
            revision=QWEN_REVISION,
            expected=QWEN_FILES,
        ),
        "mixedbread_cross_encoder": verify_snapshot(
            cross_encoder_root,
            name=CROSS_ENCODER_MODEL,
            revision=CROSS_ENCODER_REVISION,
            expected=CROSS_ENCODER_FILES,
        ),
        "deepseek_tokenizer": verify_snapshot(
            deepseek_root,
            name="deepseek-ai/DeepSeek-V4-Flash",
            revision=DEEPSEEK_REVISION,
            expected=DEEPSEEK_FILES,
        ),
    }
    manifest_payload = {
        "artifact_type": "swarmbrain-longmemeval-e1-real-model-run-manifest",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "classification": "development-experiment-not-official-longmemeval-score",
        "production_configuration": False,
        "dataset": corpus.source_artifact.as_dict(),
        "turn_projection": corpus.fingerprint(),
        "sample": {
            "seed": args.seed,
            "requested": args.sample,
            "count": len(selected),
            "questions": sample_binding,
            "questions_sha256": sha256_json(sample_binding),
        },
        "cells": ["E1-A", "E1-B"],
        "retrieval": {
            "lexical_depth": LEXICAL_LANE_DEPTH,
            "dense_depth": DENSE_LANE_DEPTH,
            "e1a_protocol": E1A_PROTOCOL.as_dict(),
            "qwen_query_instruction_sha256": sha256_bytes(QWEN_QUERY_INSTRUCTION.encode("utf-8")),
            "qwen_max_length": QWEN_MAX_LENGTH,
            "qwen_max_length_classification": "frozen-development-transfer-choice",
            "qwen_padding_side": "right",
            "qwen_padding_classification": (
                "MPS-stability-transfer-choice-using-official-last-valid-token-branch"
            ),
            "qwen_batching": {
                "method": "length-sorted-padding-aware-dynamic-batching",
                "maximum_batch_size": getattr(args, "qwen_batch_size", QWEN_BATCH_SIZE),
                "maximum_padding_ratio": QWEN_MAX_PADDING_RATIO,
                "attention_cell_budget": QWEN_ATTENTION_CELL_BUDGET,
                "attention_cell_formula": "batch-size*maximum-sequence-length-squared",
                "oversized_singleton_allowed": True,
            },
        },
        "cross_encoder": {
            "max_length": CROSS_ENCODER_MAX_LENGTH,
            "score_surface": "raw-sequence-classification-logit",
        },
        "local_execution": {
            "device": getattr(args, "device", "mps"),
            "qwen_maximum_batch_size": getattr(
                args,
                "qwen_batch_size",
                QWEN_BATCH_SIZE,
            ),
            "cross_encoder_batch_size": getattr(
                args,
                "cross_encoder_batch_size",
                CROSS_ENCODER_BATCH_SIZE,
            ),
        },
        "prompt": {
            "layout": "linear-e1",
            "complete_reader_prompt_token_budget": TOKEN_BUDGET,
            "tokenizer_chat_mode": "chat-thinking-disabled",
        },
        "reader_and_development_judge": {
            "provider": "DeepSeek API",
            "model": DEEPSEEK_MODEL,
            "thinking": "disabled",
            "temperature": 0.0,
            "official_gpt4o_judge_executed": False,
        },
        "model_snapshots": snapshots,
        "implementation": implementation_fingerprint(),
        "claims": {
            "gold_fields_used_for_retrieval_or_selection": False,
            "sample_is_held_out": False,
            "sample_is_publishable_full_benchmark": False,
            "official_longmemeval_score": False,
            "production_policy_changed": False,
        },
    }
    manifest = seal_artifact(manifest_payload)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = load_json(manifest_path, sealed=True)
        if existing != manifest:
            raise ExternalE1Error("output directory is bound to a different experiment manifest")
    else:
        write_json(manifest_path, manifest)
    return ExperimentContext(
        corpus=corpus,
        records=records,
        selected=selected,
        manifest=manifest,
        output_dir=output_dir,
        qwen_root=qwen_root,
        cross_encoder_root=cross_encoder_root,
        deepseek_root=deepseek_root,
    )


def lexical_scores(
    query: str,
    turns: Sequence[TurnProjection],
) -> tuple[tuple[TurnProjection, float], ...]:
    """Mirror the production in-memory lexical score over immutable F0 documents."""

    normalized_query = normalize_term(query[:MAX_QUERY_CHARS])
    query_tokens = set(_TOKENS.findall(normalized_query)[:MAX_QUERY_TOKENS])
    if not query_tokens:
        return ()
    scored: list[tuple[TurnProjection, float]] = []
    for turn in turns:
        normalized_document = normalize_term(turn.serialized_text)
        document_tokens = set(_TOKENS.findall(normalized_document))
        overlap = len(query_tokens & document_tokens) / len(query_tokens)
        substring = normalized_query in normalized_document
        score = overlap + (0.2 if substring else 0.0)
        if score > 0.0:
            scored.append((turn, score))
    scored.sort(
        key=lambda item: (
            -item[1],
            -datetime.fromisoformat(
                item[0].parent_session_date_utc.replace("Z", "+00:00")
            ).timestamp(),
            item[0].turn_id.canonical_id,
        )
    )
    return tuple(scored)


def qwen_query_text(query: str) -> str:
    return f"Instruct: {QWEN_QUERY_INSTRUCTION}\nQuery: {query}"


def _turn_id_payload(turn_id: LongMemEvalTurnId) -> list[str | int]:
    return list(turn_id.as_tuple())


def _turn_id_from_payload(value: Any) -> LongMemEvalTurnId:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not isinstance(value[0], str)
        or isinstance(value[1], bool)
        or not isinstance(value[1], int)
        or isinstance(value[2], bool)
        or not isinstance(value[2], int)
    ):
        raise ExternalE1Error("turn ID payload is invalid")
    return LongMemEvalTurnId(value[0], value[1], value[2])


def _snapshot_artifact(context: ExperimentContext, key: str) -> str:
    snapshots = context.manifest["model_snapshots"]
    return str(snapshots[key]["artifact_sha256"])


def _lexical_implementation_identity(context: ExperimentContext) -> ImmutableArtifactIdentity:
    files = context.manifest["implementation"]["files"]
    relevant = {
        name: files[name]
        for name in (
            "scripts/run_longmemeval_e1_external.py",
            "src/swarmbrain/adapters/memory/retrieval.py",
            "src/swarmbrain/retrieval/projection.py",
        )
    }
    return ImmutableArtifactIdentity(
        name="swarmbrain-in-memory-lexical-turn-transfer",
        revision=PROTOCOL_VERSION,
        artifact_sha256=sha256_json(relevant),
    )


def _projection_identity(context: ExperimentContext) -> ImmutableArtifactIdentity:
    return ImmutableArtifactIdentity(
        name="LongMemEval-S immutable F0 turn projection",
        revision=context.corpus.serializer_version,
        artifact_sha256=context.corpus.projection_sha256,
    )


def padding_aware_batches(
    encoded: Sequence[tuple[int, list[int], bool]],
    *,
    maximum_batch_size: int,
) -> tuple[tuple[tuple[int, list[int], bool], ...], ...]:
    """Group length-sorted inputs without multiplying long-context attention work."""

    if maximum_batch_size < 1:
        raise ExternalE1Error("maximum batch size must be positive")
    batches: list[tuple[tuple[int, list[int], bool], ...]] = []
    current: list[tuple[int, list[int], bool]] = []
    prior_length = 0
    for item in encoded:
        length = len(item[1])
        if length < 1:
            raise ExternalE1Error("encoded Qwen input cannot be empty")
        if length < prior_length:
            raise ExternalE1Error("padding-aware Qwen inputs must be sorted by length")
        prior_length = length
        proposed = [*current, item]
        maximum = length
        minimum = len(proposed[0][1])
        exceeds = len(proposed) > maximum_batch_size or (
            len(proposed) > 1
            and (
                maximum / minimum > QWEN_MAX_PADDING_RATIO
                or len(proposed) * maximum * maximum > QWEN_ATTENTION_CELL_BUDGET
            )
        )
        if exceeds:
            if not current:
                raise ExternalE1Error("padding-aware batching could not admit a singleton")
            batches.append(tuple(current))
            current = [item]
        else:
            current = proposed
    if current:
        batches.append(tuple(current))
    return tuple(batches)


@dataclass(frozen=True, slots=True)
class EmbeddedBatch:
    vectors: Any
    token_counts: tuple[int, ...]
    truncated: tuple[bool, ...]
    singleton_retry_positions: tuple[int, ...]
    batch_plan: tuple[tuple[int, ...], ...]
    model_batches: int
    padded_tokens: int
    padded_attention_cells: int
    elapsed_ms: float


class QwenEmbedder:
    """Pinned local Qwen encoder with official last-token pooling."""

    def __init__(self, root: Path, *, device: str, batch_size: int) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ExternalE1Error(
                "dense phase requires the pinned Torch/Transformers evaluation environment"
            ) from exc
        if device == "mps" and not torch.backends.mps.is_available():
            raise ExternalE1Error("requested MPS device is unavailable")
        self.torch = torch
        self.transformers_version = transformers.__version__
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        # Qwen's published pooling helper explicitly supports both branches.
        # On Torch 2.7.1/MPS, left-padded rows produce NaNs at scale; right
        # padding leaves every real causal-token state unchanged and selects
        # the last valid position through the helper's non-left branch.
        self.tokenizer.padding_side = "right"
        dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
        self.dtype_name = str(dtype).replace("torch.", "")
        self.model = AutoModel.from_pretrained(
            root,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()

    def embed(self, texts: Sequence[str]) -> EmbeddedBatch:
        if not texts:
            raise ExternalE1Error("embedding input cannot be empty")
        encoded: list[tuple[int, list[int], bool]] = []
        for position, text in enumerate(texts):
            input_ids = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
            full_count = len(input_ids)
            encoded.append((position, input_ids[:QWEN_MAX_LENGTH], full_count > QWEN_MAX_LENGTH))
        encoded.sort(key=lambda item: (len(item[1]), item[0]))
        vectors_by_position: dict[int, Any] = {}
        token_counts = [0] * len(texts)
        truncated = [False] * len(texts)
        singleton_retry_positions: set[int] = set()
        batches = padding_aware_batches(encoded, maximum_batch_size=self.batch_size)
        model_batches = len(batches)
        padded_tokens = 0
        padded_attention_cells = 0
        started = perf_counter()
        for batch in batches:
            padded_length = max(len(item[1]) for item in batch)
            padded_tokens += len(batch) * padded_length
            padded_attention_cells += len(batch) * padded_length * padded_length
            padded = self.tokenizer.pad(
                {"input_ids": [item[1] for item in batch]},
                padding=True,
                return_tensors="pt",
            )
            padded = {key: value.to(self.device) for key, value in padded.items()}
            with self.torch.inference_mode():
                hidden = self.model(**padded).last_hidden_state
                attention = padded["attention_mask"]
                if bool((attention[:, -1].sum() == attention.shape[0]).item()):
                    pooled = hidden[:, -1]
                else:
                    positions = attention.sum(dim=1) - 1
                    pooled = hidden[
                        self.torch.arange(hidden.shape[0], device=hidden.device), positions
                    ]
                pooled = self.torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                finite = self.torch.isfinite(pooled).all(dim=1)
            retries: dict[int, Any] = {}
            if not bool(finite.all().item()):
                for batch_position, (original_position, ids, _) in enumerate(batch):
                    if bool(finite[batch_position].item()):
                        continue
                    single = self.tokenizer.pad(
                        {"input_ids": [ids]},
                        padding=True,
                        return_tensors="pt",
                    )
                    single = {key: value.to(self.device) for key, value in single.items()}
                    with self.torch.inference_mode():
                        single_hidden = self.model(**single).last_hidden_state
                        single_attention = single["attention_mask"]
                        if bool(single_attention[0, -1].item()):
                            single_pooled = single_hidden[:, -1]
                        else:
                            single_position = single_attention.sum(dim=1) - 1
                            single_pooled = single_hidden[
                                self.torch.arange(1, device=single_hidden.device),
                                single_position,
                            ]
                        single_pooled = self.torch.nn.functional.normalize(
                            single_pooled.float(), p=2, dim=1
                        )
                    if not bool(self.torch.isfinite(single_pooled).all().item()):
                        raise ExternalE1Error(
                            "Qwen produced a non-finite singleton embedding "
                            f"at input_position={original_position}, tokens={len(ids)}"
                        )
                    retries[batch_position] = single_pooled[0].cpu()
                    singleton_retry_positions.add(original_position)
                    model_batches += 1
                    padded_tokens += len(ids)
                    padded_attention_cells += len(ids) * len(ids)
                    del single, single_hidden, single_attention, single_pooled
            for batch_position, (original_position, ids, was_truncated) in enumerate(batch):
                vectors_by_position[original_position] = retries.get(
                    batch_position,
                    pooled[batch_position].cpu(),
                )
                token_counts[original_position] = len(ids)
                truncated[original_position] = was_truncated
            del padded, hidden, attention, pooled, finite
        vectors = self.torch.stack([vectors_by_position[index] for index in range(len(texts))])
        return EmbeddedBatch(
            vectors=vectors,
            token_counts=tuple(token_counts),
            truncated=tuple(truncated),
            singleton_retry_positions=tuple(sorted(singleton_retry_positions)),
            batch_plan=tuple(tuple(item[0] for item in batch) for batch in batches),
            model_batches=model_batches,
            padded_tokens=padded_tokens,
            padded_attention_cells=padded_attention_cells,
            elapsed_ms=(perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device == "mps":
            self.torch.mps.empty_cache()


def _ranked_lane(
    *,
    lane: RetrievalLane,
    depth: int,
    query_sha256: str,
    corpus_sha256: str,
    identity: ExternalLaneIdentity,
    ranked: Sequence[tuple[LongMemEvalTurnId, float]],
    examined_count: int,
) -> RankedLaneObservation:
    return RankedLaneObservation(
        lane=lane,
        requested_depth=depth,
        query_sha256=query_sha256,
        turn_corpus_projection_sha256=corpus_sha256,
        identity=identity,
        candidates=tuple(
            RankedTurnObservation(turn_id=turn_id, raw_score=score)
            for turn_id, score in ranked[:depth]
        ),
        examined_count=examined_count,
    )


def _lane_identity_payload(identity: ExternalLaneIdentity) -> dict[str, Any]:
    return {
        "producer": identity.producer,
        "scorer": identity.scorer.as_dict(),
        "projection": identity.projection.as_dict(),
        "observation_artifact_sha256": identity.observation_artifact_sha256,
    }


def _lane_identity_from_payload(value: Any) -> ExternalLaneIdentity:
    if not isinstance(value, dict):
        raise ExternalE1Error("lane identity payload must be an object")
    scorer = value.get("scorer")
    projection = value.get("projection")
    if not isinstance(scorer, dict) or not isinstance(projection, dict):
        raise ExternalE1Error("lane identity is missing scorer or projection")
    return ExternalLaneIdentity(
        producer=str(value["producer"]),
        scorer=ImmutableArtifactIdentity(
            name=str(scorer["name"]),
            revision=str(scorer["revision"]),
            artifact_sha256=str(scorer["artifact_sha256"]),
        ),
        projection=ImmutableArtifactIdentity(
            name=str(projection["name"]),
            revision=str(projection["revision"]),
            artifact_sha256=str(projection["artifact_sha256"]),
        ),
        observation_artifact_sha256=str(value["observation_artifact_sha256"]),
    )


def _dense_query_binding(question: SelectedQuestion) -> dict[str, Any]:
    raw = question.question.encode("utf-8")
    rendered = qwen_query_text(question.question).encode("utf-8")
    return {
        "raw_query_sha256": sha256_bytes(raw),
        "raw_query_utf8_bytes": len(raw),
        "instruction": {
            "sha256": sha256_bytes(QWEN_QUERY_INSTRUCTION.encode("utf-8")),
            "utf8_bytes": len(QWEN_QUERY_INSTRUCTION.encode("utf-8")),
            "literal_stored": False,
        },
        "rendered_query_sha256": sha256_bytes(rendered),
        "rendered_query_utf8_bytes": len(rendered),
        "format": "Instruct: {instruction}\\nQuery: {query}",
    }


def _dense_observation_rows(
    question: SelectedQuestion,
    *,
    cosine_scores: Sequence[float],
    token_counts: Sequence[int],
    truncated: Sequence[bool],
    singleton_retries: Sequence[bool] | None = None,
) -> list[dict[str, Any]]:
    retries = tuple(singleton_retries or (False,) * len(question.turns))
    if not (
        len(question.turns)
        == len(cosine_scores)
        == len(token_counts)
        == len(truncated)
        == len(retries)
    ):
        raise ExternalE1Error("dense model output does not cover the question corpus exactly")
    rows: list[dict[str, Any]] = []
    for turn, raw_score, input_tokens, was_truncated, retried_singleton in zip(
        question.turns,
        cosine_scores,
        token_counts,
        truncated,
        retries,
        strict=True,
    ):
        raw = float(raw_score)
        if not math.isfinite(raw):
            raise ExternalE1Error("Qwen produced a non-finite cosine score")
        lane_score = max(0.0, min(1.0, raw))
        rows.append(
            {
                "turn_id": _turn_id_payload(turn.turn_id),
                "candidate_payload_sha256": turn.serialized_document_utf8.sha256,
                "candidate_payload_utf8_bytes": turn.serialized_document_utf8.bytes,
                "input_tokens_after_truncation": int(input_tokens),
                "truncated_right_at_8192": bool(was_truncated),
                "singleton_retry_after_nonfinite_batch": bool(retried_singleton),
                "raw_cosine": raw,
                "lane_score": lane_score,
            }
        )
    return rows


def _rank_dense_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["lane_score"]),
            _turn_id_from_payload(row["turn_id"]).canonical_id,
        ),
        reverse=True,
    )


def _lexical_observation_rows(
    question: SelectedQuestion,
) -> list[dict[str, Any]]:
    return [
        {
            "turn_id": _turn_id_payload(turn.turn_id),
            "candidate_payload_sha256": turn.serialized_document_utf8.sha256,
            "raw_score": score,
        }
        for turn, score in lexical_scores(question.question, question.turns)
    ]


def _ranked_payload(rows: Sequence[Mapping[str, Any]], *, score_key: str) -> list[dict[str, Any]]:
    return [
        {
            "turn_id": list(row["turn_id"]),
            "rank": rank,
            "raw_score": float(row[score_key]),
        }
        for rank, row in enumerate(rows, start=1)
    ]


def _build_e1a(
    context: ExperimentContext,
    question: SelectedQuestion,
    *,
    lexical_rows: Sequence[Mapping[str, Any]],
    dense_rows: Sequence[Mapping[str, Any]],
    lexical_identity: ExternalLaneIdentity,
    dense_identity: ExternalLaneIdentity,
):
    query_sha256 = sha256_bytes(question.question.encode("utf-8"))
    dense_ranked = _rank_dense_rows(dense_rows)
    lexical = _ranked_lane(
        lane=RetrievalLane.LEXICAL,
        depth=LEXICAL_LANE_DEPTH,
        query_sha256=query_sha256,
        corpus_sha256=context.corpus.projection_sha256,
        identity=lexical_identity,
        ranked=[
            (_turn_id_from_payload(row["turn_id"]), float(row["raw_score"])) for row in lexical_rows
        ],
        examined_count=len(question.turns),
    )
    dense = _ranked_lane(
        lane=RetrievalLane.DENSE,
        depth=DENSE_LANE_DEPTH,
        query_sha256=query_sha256,
        corpus_sha256=context.corpus.projection_sha256,
        identity=dense_identity,
        ranked=[
            (_turn_id_from_payload(row["turn_id"]), float(row["lane_score"]))
            for row in dense_ranked
        ],
        examined_count=len(question.turns),
    )
    return fuse_question_turns(
        context.corpus,
        question_id=question.question_id,
        query_text=question.question,
        lexical=lexical,
        dense=dense,
    )


def replay_dense_row(
    context: ExperimentContext,
    question: SelectedQuestion,
    row: Any,
):
    row = validate_sealed_artifact(row)
    expected = {
        "artifact_type": "swarmbrain-longmemeval-e1-dense-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
    }
    for key, value in expected.items():
        if type(row.get(key)) is not type(value) or row.get(key) != value:
            raise ExternalE1Error(f"dense row {key} does not match the frozen run")
    if row.get("query") != _dense_query_binding(question):
        raise ExternalE1Error("dense row query binding drifted")
    lexical = row.get("lexical")
    dense = row.get("dense")
    if not isinstance(lexical, dict) or not isinstance(dense, dict):
        raise ExternalE1Error("dense row is missing lane evidence")
    lexical_rows = lexical.get("observations")
    dense_rows = dense.get("observations")
    if not isinstance(lexical_rows, list) or not isinstance(dense_rows, list):
        raise ExternalE1Error("dense row lane observations must be arrays")
    if len(dense_rows) != len(question.turns):
        raise ExternalE1Error("dense observations do not cover every question turn")
    expected_lexical = _lexical_observation_rows(question)
    if lexical_rows != expected_lexical:
        raise ExternalE1Error("saved lexical scores differ from deterministic replay")
    lexical_identity = _lane_identity_from_payload(lexical.get("identity"))
    dense_identity = _lane_identity_from_payload(dense.get("identity"))
    if lexical_identity.observation_artifact_sha256 != sha256_json(lexical_rows):
        raise ExternalE1Error("lexical observation digest does not match")
    if dense_identity.observation_artifact_sha256 != sha256_json(dense_rows):
        raise ExternalE1Error("dense observation digest does not match")
    if lexical_identity.scorer != _lexical_implementation_identity(context):
        raise ExternalE1Error("lexical scorer identity drifted")
    expected_dense_scorer = ImmutableArtifactIdentity(
        name=QWEN_MODEL,
        revision=QWEN_REVISION,
        artifact_sha256=_snapshot_artifact(context, "qwen_embedding"),
    )
    if dense_identity.scorer != expected_dense_scorer:
        raise ExternalE1Error("dense scorer identity drifted")
    if lexical_identity.projection != _projection_identity(
        context
    ) or dense_identity.projection != _projection_identity(context):
        raise ExternalE1Error("turn projection identity drifted")
    turns_by_id = {turn.turn_id: turn for turn in question.turns}
    seen: set[LongMemEvalTurnId] = set()
    for observation in dense_rows:
        if not isinstance(observation, dict):
            raise ExternalE1Error("dense observation must be an object")
        turn_id = _turn_id_from_payload(observation.get("turn_id"))
        turn = turns_by_id.get(turn_id)
        if turn is None or turn_id in seen:
            raise ExternalE1Error("dense observation has an unknown or duplicate turn")
        seen.add(turn_id)
        if observation.get("candidate_payload_sha256") != turn.serialized_document_utf8.sha256:
            raise ExternalE1Error("dense observation candidate digest drifted")
        raw = observation.get("raw_cosine")
        lane = observation.get("lane_score")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ExternalE1Error("dense raw cosine is invalid")
        if lane != max(0.0, min(1.0, float(raw))):
            raise ExternalE1Error("dense lane score is not the production cosine clamp")
    e1a = _build_e1a(
        context,
        question,
        lexical_rows=lexical_rows,
        dense_rows=dense_rows,
        lexical_identity=lexical_identity,
        dense_identity=dense_identity,
    )
    if row.get("e1a_trace") != e1a.content_free_trace():
        raise ExternalE1Error("saved E1-A trace differs from deterministic replay")
    lexical_ranked = _ranked_payload(lexical_rows[:LEXICAL_LANE_DEPTH], score_key="raw_score")
    dense_ranked = _ranked_payload(
        _rank_dense_rows(dense_rows)[:DENSE_LANE_DEPTH], score_key="lane_score"
    )
    if lexical.get("returned") != lexical_ranked or dense.get("returned") != dense_ranked:
        raise ExternalE1Error("saved lane head differs from its full observations")
    return e1a


def run_dense_phase(context: ExperimentContext, *, device: str, batch_size: int) -> None:
    pending: list[SelectedQuestion] = []
    for question in context.selected:
        path = phase_path(context, "dense", question)
        if path.exists():
            replay_dense_row(context, question, load_json(path, sealed=True))
            print(f"  dense: verified {question.question_id}", file=sys.stderr, flush=True)
        else:
            pending.append(question)
    if not pending:
        print("  dense: all question artifacts already verified", file=sys.stderr, flush=True)
        return

    embedder = QwenEmbedder(context.qwen_root, device=device, batch_size=batch_size)
    try:
        for completed, question in enumerate(pending, start=1):
            started = perf_counter()
            query_batch = embedder.embed([qwen_query_text(question.question)])
            document_batch = embedder.embed([turn.serialized_text for turn in question.turns])
            similarities = document_batch.vectors @ query_batch.vectors[0]
            dense_rows = _dense_observation_rows(
                question,
                cosine_scores=similarities.tolist(),
                token_counts=document_batch.token_counts,
                truncated=document_batch.truncated,
                singleton_retries=tuple(
                    position in set(document_batch.singleton_retry_positions)
                    for position in range(len(question.turns))
                ),
            )
            lexical_rows = _lexical_observation_rows(question)
            projection_identity = _projection_identity(context)
            lexical_identity = ExternalLaneIdentity(
                producer="scripts.run_longmemeval_e1_external.lexical_scores",
                scorer=_lexical_implementation_identity(context),
                projection=projection_identity,
                observation_artifact_sha256=sha256_json(lexical_rows),
            )
            dense_identity = ExternalLaneIdentity(
                producer="scripts.run_longmemeval_e1_external.QwenEmbedder",
                scorer=ImmutableArtifactIdentity(
                    name=QWEN_MODEL,
                    revision=QWEN_REVISION,
                    artifact_sha256=_snapshot_artifact(context, "qwen_embedding"),
                ),
                projection=projection_identity,
                observation_artifact_sha256=sha256_json(dense_rows),
            )
            e1a = _build_e1a(
                context,
                question,
                lexical_rows=lexical_rows,
                dense_rows=dense_rows,
                lexical_identity=lexical_identity,
                dense_identity=dense_identity,
            )
            dense_ranked = _rank_dense_rows(dense_rows)
            model_input_binding = {
                "query": _dense_query_binding(question),
                "documents": [
                    {
                        "turn_id": row["turn_id"],
                        "candidate_payload_sha256": row["candidate_payload_sha256"],
                        "input_tokens_after_truncation": row["input_tokens_after_truncation"],
                        "truncated_right_at_8192": row["truncated_right_at_8192"],
                        "singleton_retry_after_nonfinite_batch": row[
                            "singleton_retry_after_nonfinite_batch"
                        ],
                    }
                    for row in dense_rows
                ],
                "query_batch_plan": [list(batch) for batch in query_batch.batch_plan],
                "document_batch_plan": [list(batch) for batch in document_batch.batch_plan],
            }
            payload = {
                "artifact_type": "swarmbrain-longmemeval-e1-dense-question",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "run_manifest_sha256": context.manifest["artifact_sha256"],
                "question_id": question.question_id,
                "question_position": question.position,
                "query": _dense_query_binding(question),
                "lexical": {
                    "scoring": "production-normalize-token-overlap-plus-exact-substring",
                    "tie_break": [
                        "raw-score-descending",
                        "parent-recorded-from-descending",
                        "canonical-turn-id-ascending",
                    ],
                    "examined_count": len(question.turns),
                    "identity": _lane_identity_payload(lexical_identity),
                    "observations": lexical_rows,
                    "returned": _ranked_payload(
                        lexical_rows[:LEXICAL_LANE_DEPTH], score_key="raw_score"
                    ),
                },
                "dense": {
                    "model": QWEN_MODEL,
                    "revision": QWEN_REVISION,
                    "pooling": "official-last-valid-token-then-l2-normalize-float32",
                    "padding_side": "right-MPS-stability-transfer-choice",
                    "document_truncation": "right-at-8192-tokens",
                    "score_projection": "cosine-clamped-to-[0,1]-like-production-index",
                    "tie_break": ["lane-score-descending", "canonical-turn-id-descending"],
                    "examined_count": len(question.turns),
                    "identity": _lane_identity_payload(dense_identity),
                    "model_input_sha256": sha256_json(model_input_binding),
                    "query_input_tokens_after_truncation": query_batch.token_counts[0],
                    "query_truncated_right_at_8192": query_batch.truncated[0],
                    "query_singleton_retry_after_nonfinite_batch": bool(
                        query_batch.singleton_retry_positions
                    ),
                    "query_batch_plan": [list(batch) for batch in query_batch.batch_plan],
                    "document_batch_plan": [list(batch) for batch in document_batch.batch_plan],
                    "observations": dense_rows,
                    "returned": _ranked_payload(
                        dense_ranked[:DENSE_LANE_DEPTH], score_key="lane_score"
                    ),
                    "accounting": {
                        "documents": len(dense_rows),
                        "document_input_tokens_after_truncation": sum(document_batch.token_counts),
                        "padded_model_input_tokens": (
                            query_batch.padded_tokens + document_batch.padded_tokens
                        ),
                        "padded_attention_cells": (
                            query_batch.padded_attention_cells
                            + document_batch.padded_attention_cells
                        ),
                        "documents_truncated": sum(document_batch.truncated),
                        "singleton_retries_after_nonfinite_batch": len(
                            document_batch.singleton_retry_positions
                        ),
                        "model_batches": query_batch.model_batches + document_batch.model_batches,
                        "model_elapsed_ms": (query_batch.elapsed_ms + document_batch.elapsed_ms),
                    },
                    "runtime": {
                        "python": sys.version.split()[0],
                        "torch": embedder.torch.__version__,
                        "transformers": embedder.transformers_version,
                        "device": device,
                        "dtype": embedder.dtype_name,
                        "batch_size": batch_size,
                    },
                },
                "e1a_trace": e1a.content_free_trace(),
                "wall_ms": (perf_counter() - started) * 1000.0,
                "claims": {
                    "full_question_local_dense_scan": True,
                    "gold_fields_consumed": False,
                    "reader_or_judge_executed": False,
                    "production_policy_changed": False,
                },
            }
            sealed = seal_artifact(payload)
            replay_dense_row(context, question, sealed)
            write_json(phase_path(context, "dense", question), sealed)
            print(
                f"  dense: {completed}/{len(pending)} {question.question_id} "
                f"turns={len(question.turns)} wall={payload['wall_ms'] / 1000:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            del query_batch, document_batch, similarities
    finally:
        embedder.close()


@dataclass(frozen=True, slots=True)
class CrossEncoderBatch:
    logits: tuple[float, ...]
    token_counts: tuple[int, ...]
    truncated: tuple[bool, ...]
    elapsed_ms: float


class MixedbreadCrossEncoder:
    """Pinned local Mixedbread sequence-classification logit scorer."""

    def __init__(self, root: Path, *, device: str, batch_size: int) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ExternalE1Error(
                "cross-encoder phase requires the pinned Torch/Transformers environment"
            ) from exc
        if device == "mps" and not torch.backends.mps.is_available():
            raise ExternalE1Error("requested MPS device is unavailable")
        self.torch = torch
        self.transformers_version = transformers.__version__
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
        self.dtype_name = str(dtype).replace("torch.", "")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            root,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()

    def score(self, query: str, documents: Sequence[str]) -> CrossEncoderBatch:
        if not documents:
            raise ExternalE1Error("CrossEncoder input cannot be empty")
        encoded: list[tuple[int, dict[str, list[int]], bool]] = []
        for position, document in enumerate(documents):
            full = self.tokenizer(
                query,
                document,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=True,
            )
            feature = self.tokenizer(
                query,
                document,
                add_special_tokens=True,
                truncation="longest_first",
                max_length=CROSS_ENCODER_MAX_LENGTH,
                return_attention_mask=True,
            )
            encoded.append(
                (
                    position,
                    feature,
                    len(full["input_ids"]) > CROSS_ENCODER_MAX_LENGTH,
                )
            )
        encoded.sort(key=lambda item: (len(item[1]["input_ids"]), item[0]))
        logits_by_position: dict[int, float] = {}
        token_counts = [0] * len(documents)
        truncated = [False] * len(documents)
        started = perf_counter()
        for offset in range(0, len(encoded), self.batch_size):
            batch = encoded[offset : offset + self.batch_size]
            padded = self.tokenizer.pad(
                [item[1] for item in batch],
                padding=True,
                return_tensors="pt",
            )
            padded = {key: value.to(self.device) for key, value in padded.items()}
            with self.torch.inference_mode():
                logits = self.model(**padded).logits.float().reshape(-1)
            if logits.shape[0] != len(batch):
                raise ExternalE1Error("CrossEncoder returned an unexpected logit shape")
            for batch_position, (original_position, feature, was_truncated) in enumerate(batch):
                score = float(logits[batch_position].cpu().item())
                if not math.isfinite(score):
                    raise ExternalE1Error("CrossEncoder returned a non-finite logit")
                logits_by_position[original_position] = score
                token_counts[original_position] = len(feature["input_ids"])
                truncated[original_position] = was_truncated
            del padded, logits
        return CrossEncoderBatch(
            logits=tuple(logits_by_position[index] for index in range(len(documents))),
            token_counts=tuple(token_counts),
            truncated=tuple(truncated),
            elapsed_ms=(perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device == "mps":
            self.torch.mps.empty_cache()


def _scorer_identity_payload(identity: ExternalScorerIdentity) -> dict[str, str]:
    return {
        "producer": identity.producer,
        "scorer": identity.scorer,
        "model": identity.model,
        "revision": identity.revision,
        "artifact_sha256": identity.artifact_sha256,
        "observation_artifact_sha256": identity.observation_artifact_sha256,
    }


def _scorer_identity_from_payload(value: Any) -> ExternalScorerIdentity:
    if not isinstance(value, dict):
        raise ExternalE1Error("CrossEncoder identity payload must be an object")
    return ExternalScorerIdentity(
        producer=str(value["producer"]),
        scorer=str(value["scorer"]),
        model=str(value["model"]),
        revision=str(value["revision"]),
        artifact_sha256=str(value["artifact_sha256"]),
        observation_artifact_sha256=str(value["observation_artifact_sha256"]),
    )


def _cross_encoder_observation(
    context: ExperimentContext,
    e1a: Any,
    *,
    identity: ExternalScorerIdentity,
    scores: Sequence[Mapping[str, Any]],
) -> PoolScoreObservation:
    binding = bind_e1a_pool(e1a)
    return PoolScoreObservation(
        channel=ScoreChannel.CROSS_ENCODER_LOGIT,
        question_id=binding.question_id,
        query_sha256=binding.query_sha256,
        turn_corpus_projection_sha256=binding.turn_corpus_projection_sha256,
        e1a_trace_sha256=binding.e1a_trace_sha256,
        e1a_pool_sha256=binding.e1a_pool_sha256,
        pool_count=binding.pool_count,
        identity=identity,
        scores=tuple(
            TurnScoreObservation(
                turn_id=_turn_id_from_payload(row["turn_id"]),
                raw_score=float(row["raw_logit"]),
            )
            for row in scores
        ),
    )


def replay_cross_encoder_row(
    context: ExperimentContext,
    question: SelectedQuestion,
    row: Any,
):
    dense_path = phase_path(context, "dense", question)
    if not dense_path.exists():
        raise ExternalE1Error("cross-encoder replay requires the dense phase artifact")
    e1a = replay_dense_row(context, question, load_json(dense_path, sealed=True))
    row = validate_sealed_artifact(row)
    expected = {
        "artifact_type": "swarmbrain-longmemeval-e1-cross-encoder-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_dense_artifact_sha256": load_json(dense_path, sealed=True)["artifact_sha256"],
        "source_e1a_trace_sha256": e1a.trace_sha256,
    }
    for key, value in expected.items():
        if type(row.get(key)) is not type(value) or row.get(key) != value:
            raise ExternalE1Error(f"cross-encoder row {key} differs from frozen input")
    scores = row.get("scores")
    if not isinstance(scores, list) or len(scores) != len(e1a.candidates):
        raise ExternalE1Error("CrossEncoder scores must cover the fixed E1-A pool")
    expected_ids = [candidate.turn_id for candidate in e1a.candidates]
    observed_ids = [_turn_id_from_payload(item.get("turn_id")) for item in scores]
    if observed_ids != expected_ids:
        raise ExternalE1Error("CrossEncoder score order differs from the E1-A pool")
    for item, candidate in zip(scores, e1a.candidates, strict=True):
        if item.get("candidate_payload_sha256") != candidate.turn.serialized_document_utf8.sha256:
            raise ExternalE1Error("CrossEncoder candidate payload digest drifted")
        score = item.get("raw_logit")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ExternalE1Error("CrossEncoder logit must be numeric")
        if not math.isfinite(float(score)):
            raise ExternalE1Error("CrossEncoder logit must be finite")
    identity = _scorer_identity_from_payload(row.get("identity"))
    expected_identity = ExternalScorerIdentity(
        producer="scripts.run_longmemeval_e1_external.MixedbreadCrossEncoder",
        scorer="AutoModelForSequenceClassification raw logit",
        model=CROSS_ENCODER_MODEL,
        revision=CROSS_ENCODER_REVISION,
        artifact_sha256=_snapshot_artifact(context, "mixedbread_cross_encoder"),
        observation_artifact_sha256=sha256_json(scores),
    )
    if identity != expected_identity:
        raise ExternalE1Error("CrossEncoder scorer or observation identity drifted")
    observation = _cross_encoder_observation(
        context,
        e1a,
        identity=identity,
        scores=scores,
    )
    e1b = select_e1b(e1a, cross_encoder=observation)
    if row.get("e1b_trace") != e1b.content_free_trace():
        raise ExternalE1Error("saved E1-B trace differs from deterministic replay")
    return e1a, observation, e1b


def run_cross_encoder_phase(
    context: ExperimentContext,
    *,
    device: str,
    batch_size: int,
) -> None:
    pending: list[SelectedQuestion] = []
    for question in context.selected:
        path = phase_path(context, "cross_encoder", question)
        if path.exists():
            replay_cross_encoder_row(context, question, load_json(path, sealed=True))
            print(f"  cross-encoder: verified {question.question_id}", file=sys.stderr, flush=True)
        else:
            pending.append(question)
    if not pending:
        print(
            "  cross-encoder: all question artifacts already verified",
            file=sys.stderr,
            flush=True,
        )
        return

    scorer = MixedbreadCrossEncoder(
        context.cross_encoder_root,
        device=device,
        batch_size=batch_size,
    )
    try:
        for completed, question in enumerate(pending, start=1):
            started = perf_counter()
            dense_path = phase_path(context, "dense", question)
            if not dense_path.exists():
                raise ExternalE1Error("cross-encoder phase requires dense artifacts first")
            dense_row = load_json(dense_path, sealed=True)
            e1a = replay_dense_row(context, question, dense_row)
            documents = [candidate.turn.serialized_text for candidate in e1a.candidates]
            scored = scorer.score(question.question, documents)
            scores = [
                {
                    "turn_id": _turn_id_payload(candidate.turn_id),
                    "candidate_payload_sha256": candidate.turn.serialized_document_utf8.sha256,
                    "candidate_payload_utf8_bytes": candidate.turn.serialized_document_utf8.bytes,
                    "pair_tokens_after_truncation": scored.token_counts[position],
                    "truncated_longest_first_at_512": scored.truncated[position],
                    "raw_logit": scored.logits[position],
                }
                for position, candidate in enumerate(e1a.candidates)
            ]
            identity = ExternalScorerIdentity(
                producer="scripts.run_longmemeval_e1_external.MixedbreadCrossEncoder",
                scorer="AutoModelForSequenceClassification raw logit",
                model=CROSS_ENCODER_MODEL,
                revision=CROSS_ENCODER_REVISION,
                artifact_sha256=_snapshot_artifact(context, "mixedbread_cross_encoder"),
                observation_artifact_sha256=sha256_json(scores),
            )
            observation = _cross_encoder_observation(
                context,
                e1a,
                identity=identity,
                scores=scores,
            )
            e1b = select_e1b(e1a, cross_encoder=observation)
            payload = {
                "artifact_type": "swarmbrain-longmemeval-e1-cross-encoder-question",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "run_manifest_sha256": context.manifest["artifact_sha256"],
                "question_id": question.question_id,
                "question_position": question.position,
                "source_dense_artifact_sha256": dense_row["artifact_sha256"],
                "source_e1a_trace_sha256": e1a.trace_sha256,
                "identity": _scorer_identity_payload(identity),
                "model_input_sha256": sha256_json(
                    {
                        "query_sha256": sha256_bytes(question.question.encode("utf-8")),
                        "candidate_payload_sha256": [
                            candidate.turn.serialized_document_utf8.sha256
                            for candidate in e1a.candidates
                        ],
                        "pair_tokens_after_truncation": list(scored.token_counts),
                        "truncated": list(scored.truncated),
                    }
                ),
                "scores": scores,
                "e1b_trace": e1b.content_free_trace(),
                "accounting": {
                    "fixed_e1a_pool_candidates": len(scores),
                    "pair_input_tokens_after_truncation": sum(scored.token_counts),
                    "pairs_truncated": sum(scored.truncated),
                    "model_batches": math.ceil(len(scores) / batch_size),
                    "model_elapsed_ms": scored.elapsed_ms,
                },
                "runtime": {
                    "python": sys.version.split()[0],
                    "torch": scorer.torch.__version__,
                    "transformers": scorer.transformers_version,
                    "device": device,
                    "dtype": scorer.dtype_name,
                    "batch_size": batch_size,
                },
                "wall_ms": (perf_counter() - started) * 1000.0,
                "claims": {
                    "every_fixed_e1a_pool_candidate_scored_once": True,
                    "raw_logits_preserved": True,
                    "gold_fields_consumed": False,
                    "reader_or_judge_executed": False,
                    "production_policy_changed": False,
                },
            }
            sealed = seal_artifact(payload)
            replay_cross_encoder_row(context, question, sealed)
            write_json(phase_path(context, "cross_encoder", question), sealed)
            print(
                f"  cross-encoder: {completed}/{len(pending)} {question.question_id} "
                f"pairs={len(scores)} wall={payload['wall_ms'] / 1000:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    finally:
        scorer.close()


def render_deepseek_chat_prompt(prompt: str) -> str:
    """The pinned DeepSeek-V4 one-user-message, thinking-disabled surface."""

    return "<｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜></think>"


class DeepSeekExactTokenizer:
    """Exact local tokenizer boundary for the API's one-message chat request."""

    def __init__(self, root: Path, *, artifact_sha256: str) -> None:
        try:
            import transformers
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ExternalE1Error(
                "pack/qa phases require the pinned Transformers evaluation environment"
            ) from exc
        encoding_path = root / "encoding" / "encoding_dsv4.py"
        if _sha256_file(encoding_path) != DEEPSEEK_FILES["encoding/encoding_dsv4.py"][1]:
            raise ExternalE1Error("pinned DeepSeek encoding source drifted")
        spec = importlib.util.spec_from_file_location(
            "swarmbrain_pinned_encoding_dsv4", encoding_path
        )
        if spec is None or spec.loader is None:
            raise ExternalE1Error("cannot load pinned DeepSeek encoding source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._encode_messages = module.encode_messages
        self._tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.transformers_version = transformers.__version__
        self._request_id = 0
        self._identity = TokenizerIdentity(
            protocol=TOKENIZER_PROTOCOL,
            model=DEEPSEEK_MODEL,
            revision=DEEPSEEK_REVISION,
            artifact_sha256=artifact_sha256,
        )
        sentinel = "swarmbrain exact tokenizer sentinel"
        if self._render(sentinel) != render_deepseek_chat_prompt(sentinel):
            raise ExternalE1Error("local DeepSeek chat rendering differs from pinned reference")

    @property
    def identity(self) -> TokenizerIdentity:
        return self._identity

    def reset_receipts(self) -> None:
        self._request_id = 0

    def _render(self, prompt: str) -> str:
        rendered = self._encode_messages(
            [{"role": "user", "content": prompt}],
            thinking_mode="chat",
        )
        if not isinstance(rendered, str):
            raise ExternalE1Error("pinned DeepSeek encoder returned non-text")
        return rendered

    def exact_count(self, prompt: str) -> int:
        rendered = self._render(prompt)
        if rendered != render_deepseek_chat_prompt(prompt):
            raise ExternalE1Error("DeepSeek chat renderer drifted for an experiment prompt")
        input_ids = self._tokenizer.encode(rendered, add_special_tokens=False)
        if not input_ids:
            raise ExternalE1Error("DeepSeek tokenizer returned an empty encoding")
        return len(input_ids)

    def count_prompt(
        self,
        prompt: str,
        *,
        query_sha256: str,
    ) -> ExactTokenCountReceipt:
        self._request_id += 1
        prompt_bytes = prompt.encode("utf-8")
        prompt_sha256 = sha256_bytes(prompt_bytes)
        return ExactTokenCountReceipt(
            request_id=self._request_id,
            provider_request_id=(f"deepseek-v4-local-{self._request_id:06d}-{prompt_sha256[:20]}"),
            tokenizer_identity_sha256=self.identity.identity_sha256,
            query_sha256=query_sha256,
            prompt_sha256=prompt_sha256,
            prompt_utf8_bytes=len(prompt_bytes),
            token_count=self.exact_count(prompt),
        )


def _compute_packs(
    context: ExperimentContext,
    question: SelectedQuestion,
    *,
    tokenizer: DeepSeekExactTokenizer,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dense_row = load_json(phase_path(context, "dense", question), sealed=True)
    e1a = replay_dense_row(context, question, dense_row)
    ce_row = load_json(phase_path(context, "cross_encoder", question), sealed=True)
    _, _, e1b = replay_cross_encoder_row(context, question, ce_row)
    candidates_by_cell = {
        "E1-A": tuple(candidate.turn for candidate in e1a.candidates),
        "E1-B": e1b.turns,
    }
    packed: dict[str, Any] = {}
    prompt_arms: dict[str, Any] = {}
    for cell, turns in candidates_by_cell.items():
        tokenizer.reset_receipts()
        result = pack_turn_prompt(
            OrderedTurnBlocks.linear(turns),
            question_id=question.question_id,
            question=question.question,
            current_date=question.current_date,
            token_budget=TOKEN_BUDGET,
            tokenizer=tokenizer,
        )
        artifact = result.content_free_artifact()
        packed[cell] = artifact
        prompt_arms[cell] = {
            "prompt": result.prompt,
            "prompt_sha256": sha256_bytes(result.prompt.encode("utf-8")),
            "prompt_utf8_bytes": len(result.prompt.encode("utf-8")),
            "prompt_tokens": artifact["final_prompt"]["tokens"],
            "packing_trace_sha256": artifact["trace_sha256"],
        }
    prompt_payload = {
        "artifact_type": "swarmbrain-longmemeval-e1-packed-prompts-sensitive",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "classification": "contains-public-benchmark-question-and-turn-text",
        "arms": prompt_arms,
    }
    prompt_artifact = seal_artifact(prompt_payload)
    pack_payload = {
        "artifact_type": "swarmbrain-longmemeval-e1-pack-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_dense_artifact_sha256": dense_row["artifact_sha256"],
        "source_cross_encoder_artifact_sha256": ce_row["artifact_sha256"],
        "source_e1a_trace_sha256": e1a.trace_sha256,
        "source_e1b_trace_sha256": e1b.trace_sha256,
        "prompt_artifact_sha256": prompt_artifact["artifact_sha256"],
        "tokenizer": tokenizer.identity.as_dict(),
        "runtime": {
            "python": sys.version.split()[0],
            "transformers": tokenizer.transformers_version,
        },
        "arms": packed,
        "claims": {
            "complete_reader_prompt_counted": True,
            "whole_turns_only": True,
            "gold_fields_consumed": False,
            "reader_or_judge_executed": False,
            "production_policy_changed": False,
        },
    }
    return seal_artifact(pack_payload), prompt_artifact


def replay_pack_row(
    context: ExperimentContext,
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
        raise ExternalE1Error("saved packing artifact differs from exact replay")
    if prompt_row != expected_prompt:
        raise ExternalE1Error("saved plaintext prompt artifact differs from exact replay")
    return pack_row, prompt_row


def run_pack_phase(context: ExperimentContext) -> None:
    tokenizer = DeepSeekExactTokenizer(
        context.deepseek_root,
        artifact_sha256=_snapshot_artifact(context, "deepseek_tokenizer"),
    )
    for completed, question in enumerate(context.selected, start=1):
        pack_path = phase_path(context, "pack", question)
        prompt_path = phase_path(context, "prompts", question)
        if pack_path.exists() != prompt_path.exists():
            raise ExternalE1Error("packing phase has only one side of its bound artifact pair")
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
            pack_row, prompt_row = _compute_packs(
                context,
                question,
                tokenizer=tokenizer,
            )
            write_json(prompt_path, prompt_row)
            write_json(pack_path, pack_row)
            action = "packed"
        print(
            f"  pack: {completed}/{len(context.selected)} {action} {question.question_id} "
            f"wall={(perf_counter() - started):.1f}s",
            file=sys.stderr,
            flush=True,
        )


def verify_provider_prompt_tokens(
    result: ChatResult,
    *,
    expected: int,
    label: str,
) -> None:
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ExternalE1Error("expected prompt token count must be positive")
    if result.prompt_tokens != expected:
        raise ExternalE1Error(
            f"{label} API prompt_tokens={result.prompt_tokens} differs from exact local count={expected}"
        )


def _receipt_path(context: ExperimentContext, question: SelectedQuestion) -> Path:
    name = f"{question.position:03d}-{_safe_question_id(question.question_id)}.jsonl"
    return context.output_dir / "receipts" / name


def _receipt_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _load_receipts(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        raw_lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise ExternalE1Error(f"cannot read receipt artifact {path}") from exc
    if not raw_lines:
        raise ExternalE1Error("receipt artifact cannot be empty")
    records: list[dict[str, Any]] = []
    for raw in raw_lines:
        try:
            value = json.loads(
                raw,
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_fields,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ExternalE1Error("receipt JSONL contains an invalid row") from exc
        if not isinstance(value, dict):
            raise ExternalE1Error("receipt JSONL rows must be objects")
        validate_chat_receipt_record(value)
        if canonical_json_bytes(value) != raw:
            raise ExternalE1Error("receipt JSONL row is not canonical JSON")
        records.append(value)
    return tuple(records)


def _upper_bound_cost(result: ChatResult) -> float:
    one_attempt = (
        result.prompt_tokens * DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION
        + result.completion_tokens * DEEPSEEK_OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    return one_attempt * result.attempts


def _receipt_question_id(question_id: str, cell: str) -> str:
    return f"{question_id}:{cell.casefold()}"


def _qa_arm_order(question: SelectedQuestion) -> tuple[str, str]:
    return ("E1-A", "E1-B") if question.position % 2 == 0 else ("E1-B", "E1-A")


def _judge_text_for_arm(question: SelectedQuestion, hypothesis: str) -> str:
    return judge_prompt(
        str(question.record["question_type"]),
        question.question,
        str(question.record["answer"]),
        hypothesis,
        abstention=is_abstention_question(question.question_id),
    )


def replay_qa_row(
    context: ExperimentContext,
    question: SelectedQuestion,
    *,
    tokenizer: DeepSeekExactTokenizer,
    qa_row: Any,
    receipt_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    qa_row = validate_sealed_artifact(qa_row)
    pack_path = phase_path(context, "pack", question)
    prompt_path = phase_path(context, "prompts", question)
    pack_row, prompt_row = replay_pack_row(
        context,
        question,
        tokenizer=tokenizer,
        pack_row=load_json(pack_path, sealed=True),
        prompt_row=load_json(prompt_path, sealed=True),
    )
    raw_receipts = _receipt_bytes(receipt_records)
    expected = {
        "artifact_type": "swarmbrain-longmemeval-e1-deepseek-qa-question",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "question_id": question.question_id,
        "question_position": question.position,
        "source_pack_artifact_sha256": pack_row["artifact_sha256"],
        "source_prompt_artifact_sha256": prompt_row["artifact_sha256"],
        "receipt_artifact_sha256": sha256_bytes(raw_receipts),
        "receipt_artifact_bytes": len(raw_receipts),
        "arm_order": list(_qa_arm_order(question)),
    }
    for key, value in expected.items():
        if type(qa_row.get(key)) is not type(value) or qa_row.get(key) != value:
            raise ExternalE1Error(f"QA row {key} differs from its frozen evidence")
    arms = qa_row.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"E1-A", "E1-B"}:
        raise ExternalE1Error("QA row must contain exactly the two paired E1 arms")
    if len(receipt_records) != 4:
        raise ExternalE1Error("paired QA requires exactly two reader and two judge receipts")
    replayed: dict[tuple[str, str], ChatResult] = {}
    receipt_sha256: dict[tuple[str, str], str] = {}
    for receipt in receipt_records:
        result = validate_chat_receipt_record(receipt)
        receipt_id = str(receipt["question_id"])
        call_role = str(receipt["call_role"])
        key = (receipt_id, call_role)
        if key in replayed:
            raise ExternalE1Error("QA receipt artifact repeats an arm/role pair")
        replayed[key] = result
        receipt_sha256[key] = sha256_json(receipt)
    total_cost = 0.0
    for cell in ("E1-A", "E1-B"):
        arm = arms[cell]
        if not isinstance(arm, dict):
            raise ExternalE1Error("QA arm must be an object")
        receipt_id = _receipt_question_id(question.question_id, cell)
        reader_key = (receipt_id, "reader")
        judge_key = (receipt_id, "development_judge")
        if reader_key not in replayed or judge_key not in replayed:
            raise ExternalE1Error("QA receipts do not cover both calls for each arm")
        reader = replayed[reader_key]
        judge = replayed[judge_key]
        prompt = str(prompt_row["arms"][cell]["prompt"])
        exact_reader_tokens = tokenizer.exact_count(prompt)
        if exact_reader_tokens != prompt_row["arms"][cell]["prompt_tokens"]:
            raise ExternalE1Error("packed prompt token count differs from independent replay")
        verify_provider_prompt_tokens(
            reader,
            expected=exact_reader_tokens,
            label=f"{question.question_id}/{cell}/reader",
        )
        expected_judge_prompt = _judge_text_for_arm(question, reader.content)
        exact_judge_tokens = tokenizer.exact_count(expected_judge_prompt)
        verify_provider_prompt_tokens(
            judge,
            expected=exact_judge_tokens,
            label=f"{question.question_id}/{cell}/judge",
        )
        expected_arm = {
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
                "receipt_sha256": receipt_sha256[reader_key],
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
                "receipt_sha256": receipt_sha256[judge_key],
            },
            "estimated_cost_upper_bound_usd": _upper_bound_cost(reader) + _upper_bound_cost(judge),
        }
        if arm != expected_arm:
            raise ExternalE1Error("QA normalized arm differs from raw receipt replay")
        total_cost += expected_arm["estimated_cost_upper_bound_usd"]
    if qa_row.get("estimated_cost_upper_bound_usd") != total_cost:
        raise ExternalE1Error("QA cost ledger differs from receipt-derived usage")
    return qa_row


async def _run_qa_calls(
    context: ExperimentContext,
    *,
    tokenizer: DeepSeekExactTokenizer,
    base_url: str,
    api_key: str,
) -> None:
    reader_client = ChatClient(
        base_url=base_url,
        model=DEEPSEEK_MODEL,
        api_key=api_key,
        temperature=0.0,
        max_tokens=4096,
        required_response_model=DEEPSEEK_MODEL,
        require_request_id=True,
        thinking_mode="disabled",
    )
    judge_client = ChatClient(
        base_url=base_url,
        model=DEEPSEEK_MODEL,
        api_key=api_key,
        temperature=0.0,
        max_tokens=64,
        required_response_model=DEEPSEEK_MODEL,
        require_request_id=True,
        thinking_mode="disabled",
    )
    try:
        for completed, question in enumerate(context.selected, start=1):
            qa_path = phase_path(context, "qa", question)
            receipts_path = _receipt_path(context, question)
            if qa_path.exists() != receipts_path.exists():
                raise ExternalE1Error("QA phase has only one side of its receipt/result pair")
            if qa_path.exists():
                replay_qa_row(
                    context,
                    question,
                    tokenizer=tokenizer,
                    qa_row=load_json(qa_path, sealed=True),
                    receipt_records=_load_receipts(receipts_path),
                )
                print(
                    f"  qa: {completed}/{len(context.selected)} verified {question.question_id}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            pack_path = phase_path(context, "pack", question)
            prompt_path = phase_path(context, "prompts", question)
            if not pack_path.exists() or not prompt_path.exists():
                raise ExternalE1Error("QA phase requires packed prompts first")
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
                prompt = str(prompt_row["arms"][cell]["prompt"])
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
                receipt_id = _receipt_question_id(question.question_id, cell)
                reader_receipt = chat_receipt_record(receipt_id, "reader", reader)
                judge_receipt = chat_receipt_record(
                    receipt_id,
                    "development_judge",
                    judge,
                )
                validate_chat_receipt_record(reader_receipt)
                validate_chat_receipt_record(judge_receipt)
                receipts.extend((reader_receipt, judge_receipt))
                arms[cell] = {
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
                        "receipt_sha256": sha256_json(reader_receipt),
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
                        "receipt_sha256": sha256_json(judge_receipt),
                    },
                    "estimated_cost_upper_bound_usd": _upper_bound_cost(reader)
                    + _upper_bound_cost(judge),
                }
            raw_receipts = _receipt_bytes(receipts)
            payload = {
                "artifact_type": "swarmbrain-longmemeval-e1-deepseek-qa-question",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
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
                "cost_basis": {
                    "input": "all prompt tokens pessimistically priced as cache misses",
                    "cache_miss_input_usd_per_million": (DEEPSEEK_CACHE_MISS_INPUT_USD_PER_MILLION),
                    "output_usd_per_million": DEEPSEEK_OUTPUT_USD_PER_MILLION,
                    "attempt_multiplier": True,
                },
                "claims": {
                    "reader_model": DEEPSEEK_MODEL,
                    "development_judge_model": DEEPSEEK_MODEL,
                    "thinking_disabled": True,
                    "api_and_local_prompt_tokens_reconciled": True,
                    "official_gpt4o_judge_executed": False,
                    "official_longmemeval_score": False,
                },
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
                f"  qa: {completed}/{len(context.selected)} ran {question.question_id} "
                f"A={arms['E1-A']['development_label']} "
                f"B={arms['E1-B']['development_label']} "
                f"cost<=${payload['estimated_cost_upper_bound_usd']:.5f}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        await reader_client.aclose()
        await judge_client.aclose()


def run_qa_phase(context: ExperimentContext, *, base_url: str, api_key_env: str) -> None:
    if base_url.strip().rstrip("/") not in {
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
    }:
        raise ExternalE1Error("development QA is frozen to the official DeepSeek API endpoint")
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise ExternalE1Error(f"environment variable {api_key_env!r} is missing")
    tokenizer = DeepSeekExactTokenizer(
        context.deepseek_root,
        artifact_sha256=_snapshot_artifact(context, "deepseek_tokenizer"),
    )
    asyncio.run(
        _run_qa_calls(
            context,
            tokenizer=tokenizer,
            base_url=base_url,
            api_key=api_key,
        )
    )


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total < 1:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4 * total)) / total) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _paired_bootstrap_interval(
    differences: Sequence[int],
    *,
    samples: int = 50_000,
    seed: int = 20260809,
) -> tuple[float, float]:
    if not differences:
        return (0.0, 0.0)
    generator = random.Random(seed)
    count = len(differences)
    means = sorted(
        sum(differences[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return (lower, upper)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def _retrieval_metrics(
    question: SelectedQuestion,
    *,
    ranked_ids: Sequence[list[str | int]],
    kept_ids: Sequence[list[str | int]],
) -> dict[str, Any]:
    session_ids = [str(value) for value in question.record["haystack_session_ids"]]
    answer_ids = {str(value) for value in question.record.get("answer_session_ids", ())}
    gold_positions = {
        position for position, session_id in enumerate(session_ids) if session_id in answer_ids
    }
    ranked = [_turn_id_from_payload(value) for value in ranked_ids]
    kept = [_turn_id_from_payload(value) for value in kept_ids]
    first_rank = next(
        (
            rank
            for rank, turn_id in enumerate(ranked, start=1)
            if turn_id.session_position in gold_positions
        ),
        None,
    )
    hit_positions = {turn_id.session_position for turn_id in kept} & gold_positions
    return {
        "gold_session_positions": len(gold_positions),
        "packed_gold_session_positions": len(hit_positions),
        "any_gold_session_in_prompt": bool(hit_positions),
        "all_gold_sessions_in_prompt": bool(gold_positions) and hit_positions == gold_positions,
        "answer_session_recall": (
            len(hit_positions) / len(gold_positions) if gold_positions else 0.0
        ),
        "candidate_mrr": 1.0 / first_rank if first_rank is not None else 0.0,
    }


def build_report(context: ExperimentContext) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    source_artifacts: dict[str, list[str]] = {
        "dense": [],
        "cross_encoder": [],
        "pack": [],
        "prompts": [],
        "qa": [],
        "receipts": [],
    }
    for question in context.selected:
        dense = load_json(phase_path(context, "dense", question), sealed=True)
        e1a = replay_dense_row(context, question, dense)
        ce = load_json(phase_path(context, "cross_encoder", question), sealed=True)
        _, _, e1b = replay_cross_encoder_row(context, question, ce)
        pack = load_json(phase_path(context, "pack", question), sealed=True)
        prompts = load_json(phase_path(context, "prompts", question), sealed=True)
        qa = load_json(phase_path(context, "qa", question), sealed=True)
        receipts = _load_receipts(_receipt_path(context, question))
        tokenizer = DeepSeekExactTokenizer(
            context.deepseek_root,
            artifact_sha256=_snapshot_artifact(context, "deepseek_tokenizer"),
        )
        replay_qa_row(
            context,
            question,
            tokenizer=tokenizer,
            qa_row=qa,
            receipt_records=receipts,
        )
        source_artifacts["dense"].append(dense["artifact_sha256"])
        source_artifacts["cross_encoder"].append(ce["artifact_sha256"])
        source_artifacts["pack"].append(pack["artifact_sha256"])
        source_artifacts["prompts"].append(prompts["artifact_sha256"])
        source_artifacts["qa"].append(qa["artifact_sha256"])
        source_artifacts["receipts"].append(sha256_bytes(_receipt_bytes(receipts)))
        arm_metrics: dict[str, Any] = {}
        for cell, result in (("E1-A", e1a), ("E1-B", e1b)):
            ranked_ids = [_turn_id_payload(candidate.turn_id) for candidate in result.candidates]
            pack_trace = pack["arms"][cell]
            arm_metrics[cell] = {
                "development_label": qa["arms"][cell]["development_label"],
                "prompt_tokens": pack_trace["final_prompt"]["tokens"],
                "kept_turns": len(pack_trace["kept_ids"]),
                "reader_latency_ms": qa["arms"][cell]["reader"]["latency_ms"],
                "judge_latency_ms": qa["arms"][cell]["development_judge"]["latency_ms"],
                "estimated_cost_upper_bound_usd": qa["arms"][cell][
                    "estimated_cost_upper_bound_usd"
                ],
                "retrieval": _retrieval_metrics(
                    question,
                    ranked_ids=ranked_ids,
                    kept_ids=pack_trace["kept_ids"],
                ),
            }
        cases.append(
            {
                "question_id": question.question_id,
                "question_position": question.position,
                "question_type": str(question.record["question_type"]),
                "abstention": is_abstention_question(question.question_id),
                "arm_order": qa["arm_order"],
                "arms": arm_metrics,
                "paired_delta": int(arm_metrics["E1-B"]["development_label"])
                - int(arm_metrics["E1-A"]["development_label"]),
            }
        )
    count = len(cases)
    summaries: dict[str, Any] = {}
    for cell in ("E1-A", "E1-B"):
        labels = [bool(case["arms"][cell]["development_label"]) for case in cases]
        successes = sum(labels)
        summaries[cell] = {
            "development_accuracy": successes / count,
            "development_accuracy_wilson_95": list(_wilson_interval(successes, count)),
            "mean_prompt_tokens": _mean(case["arms"][cell]["prompt_tokens"] for case in cases),
            "mean_kept_turns": _mean(case["arms"][cell]["kept_turns"] for case in cases),
            "mean_reader_latency_ms": _mean(
                case["arms"][cell]["reader_latency_ms"] for case in cases
            ),
            "mean_estimated_cost_upper_bound_usd": _mean(
                case["arms"][cell]["estimated_cost_upper_bound_usd"] for case in cases
            ),
            "any_gold_session_in_prompt": _mean(
                float(case["arms"][cell]["retrieval"]["any_gold_session_in_prompt"])
                for case in cases
            ),
            "all_gold_sessions_in_prompt": _mean(
                float(case["arms"][cell]["retrieval"]["all_gold_sessions_in_prompt"])
                for case in cases
            ),
            "mean_answer_session_recall": _mean(
                case["arms"][cell]["retrieval"]["answer_session_recall"] for case in cases
            ),
            "mean_candidate_mrr": _mean(
                case["arms"][cell]["retrieval"]["candidate_mrr"] for case in cases
            ),
        }
    differences = [int(case["paired_delta"]) for case in cases]
    bootstrap = _paired_bootstrap_interval(differences)
    improved = sum(value > 0 for value in differences)
    regressed = sum(value < 0 for value in differences)
    tied = count - improved - regressed
    payload = {
        "artifact_type": "swarmbrain-longmemeval-e1-development-pilot-report",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "run_manifest_sha256": context.manifest["artifact_sha256"],
        "classification": "exploratory-development-sample-not-official-longmemeval-score",
        "question_count": count,
        "reader": DEEPSEEK_MODEL,
        "judge": {
            "model": DEEPSEEK_MODEL,
            "role": "development-judge",
            "official_gpt4o_executed": False,
        },
        "arms": summaries,
        "paired": {
            "mean_accuracy_delta_e1b_minus_e1a": _mean(differences),
            "bootstrap_95": list(bootstrap),
            "bootstrap_samples": 50_000,
            "bootstrap_seed": 20260809,
            "improved": improved,
            "regressed": regressed,
            "tied": tied,
        },
        "total_estimated_cost_upper_bound_usd": sum(
            case["arms"][cell]["estimated_cost_upper_bound_usd"]
            for case in cases
            for cell in ("E1-A", "E1-B")
        ),
        "source_artifact_sets": {
            key: {"count": len(values), "ordered_sha256": sha256_json(values)}
            for key, values in source_artifacts.items()
        },
        "cases": cases,
        "decision": {
            "e1b_promising_on_development_sample": (
                summaries["E1-B"]["development_accuracy"]
                > summaries["E1-A"]["development_accuracy"]
                and summaries["E1-B"]["mean_answer_session_recall"]
                >= summaries["E1-A"]["mean_answer_session_recall"]
            ),
            "positive_paired_lower_confidence_bound": bootstrap[0] > 0.0,
            "eligible_for_production_promotion": False,
            "required_next_if_promising": (
                "scale the frozen paired cell before any E1-C or production change"
            ),
        },
        "claims": {
            "gold_used_only_for_posthoc_report_metrics_and_development_judging": True,
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
        choices=("dense", "cross-encoder", "pack", "qa", "report", "all"),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--qwen-root", type=Path, default=DEFAULT_QWEN_ROOT)
    parser.add_argument("--cross-encoder-root", type=Path, default=DEFAULT_CE_ROOT)
    parser.add_argument("--deepseek-root", type=Path, default=DEFAULT_DEEPSEEK_ROOT)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--qwen-batch-size", type=int, default=QWEN_BATCH_SIZE)
    parser.add_argument(
        "--cross-encoder-batch-size",
        type=int,
        default=CROSS_ENCODER_BATCH_SIZE,
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.qwen_batch_size < 1 or args.cross_encoder_batch_size < 1:
        raise SystemExit("batch sizes must be positive")
    context = build_context(args)
    phases = (
        ("dense", "cross-encoder", "pack", "qa", "report") if args.phase == "all" else (args.phase,)
    )
    for phase in phases:
        if phase == "dense":
            run_dense_phase(
                context,
                device=args.device,
                batch_size=args.qwen_batch_size,
            )
        elif phase == "cross-encoder":
            run_cross_encoder_phase(
                context,
                device=args.device,
                batch_size=args.cross_encoder_batch_size,
            )
        elif phase == "pack":
            run_pack_phase(context)
        elif phase == "qa":
            run_qa_phase(
                context,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
            )
        else:
            report = build_report(context)
            print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
