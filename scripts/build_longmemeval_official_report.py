#!/usr/bin/env python3
"""Build a strict SOTA artifact from official LongMemEval judge outputs.

The upstream evaluator writes one JSON object per hypothesis and adds an
``autoeval_label`` produced by ``gpt-4o-2024-08-06``.  This command joins those
labels back to the immutable generation runs, rejects partial/mixed/tampered
inputs, and aggregates repeated full-500 runs into the schema consumed by the
SOTA readiness gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlsplit

from _longmemeval_common import LONGMEMEVAL_S_SHA256
from run_longmemeval_qa import (
    CHAT_RECEIPT_PROTOCOL_VERSION,
    CHAT_REQUEST_PARSER,
    CHAT_RESPONSE_PARSER,
    QA_ARTIFACT_SCHEMA_VERSION,
    QA_PROTOCOL_VERSION,
    QA_RUN_ARTIFACT_TYPE,
    RETRIEVAL_ARTIFACT_SCHEMA_VERSION,
    RETRIEVAL_PROTOCOL_VERSION,
    RETRIEVAL_RUN_ARTIFACT_TYPE,
    ChatProtocolError,
    ChatResult,
    build_reader_prompt,
    is_abstention_question,
    judge_label,
    judge_prompt,
    load_chat_receipt_artifact,
    replay_retrieval_case,
    retrieval_publishability_errors,
    validate_chat_receipt_record,
    validate_implementation_fingerprint,
    validate_retrieval_run_protocol,
)

OFFICIAL_JUDGE_MODEL = "gpt-4o-2024-08-06"
OFFICIAL_EVALUATOR = (
    "https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "sota"
    / "longmemeval-s-official-report.json"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPORT_ARTIFACT_TYPE = "swarmbrain-longmemeval-official-report"
OFFICIAL_REPORT_SCHEMA_VERSION = 4
OFFICIAL_REPORT_PROTOCOL_VERSION = "swarmbrain-longmemeval-official-report-v4"
_QA_FINGERPRINT_REQUIRED_FILES = frozenset(
    {
        "scripts/_longmemeval_common.py",
        "scripts/run_longmemeval_qa.py",
        "pyproject.toml",
        "uv.lock",
    }
)


class OfficialReportError(ValueError):
    """An input cannot support an official, comparable benchmark claim."""


@dataclass(frozen=True, slots=True)
class JudgedQuestion:
    question_id: str
    question_type: str
    label: bool


@dataclass(frozen=True, slots=True)
class OfficialRun:
    run_path: Path
    labels_path: Path
    run_id: str
    started_at: str
    reader_model: str
    prompt_style: str
    reader_protocol_json: str
    retrieval_protocol_json: str
    retrieval_limit: int
    run_sha256: str
    run_bytes: int
    labels_sha256: str
    labels_bytes: int
    judge_receipt_path: Path
    judge_receipt_sha256: str
    judge_receipt_bytes: int
    hypothesis_path: Path
    hypothesis_sha256: str
    hypothesis_bytes: int
    chat_receipt_path: Path
    chat_receipt_sha256: str
    chat_receipt_bytes: int
    retrieval_source_path: Path
    retrieval_source_sha256: str
    retrieval_source_bytes: int
    dataset_path: Path
    dataset_sha256: str
    dataset_bytes: int
    qa_implementation_tree_sha256: str
    retrieval_implementation_tree_sha256: str
    reader_request_ids: tuple[str, ...]
    official_judge_request_ids: tuple[str, ...]
    official_judge_prompt_tokens: int
    official_judge_completion_tokens: int
    questions: tuple[JudgedQuestion, ...]
    reader_failures: int

    @property
    def accuracy(self) -> float:
        return mean(question.label for question in self.questions)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        action="append",
        nargs=3,
        metavar=("GENERATION_RUN", "OFFICIAL_LABELS", "OFFICIAL_JUDGE_RECEIPTS"),
        required=True,
        help="repeat once per independent generation+official-judge run",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    return parser


def _reject_json_constant(value: str) -> None:
    raise OfficialReportError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OfficialReportError(f"duplicate JSON field {key!r} is forbidden")
        result[key] = value
    return result


def _strict_json_loads(text: str, *, label: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_fields,
        )
    except (json.JSONDecodeError, OfficialReportError) as exc:
        raise OfficialReportError(f"invalid strict JSON in {label}: {exc}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = _strict_json_loads(path.read_text(encoding="utf-8"), label=str(path))
    except (OSError, UnicodeError) as exc:
        raise OfficialReportError(f"cannot read JSON object {path}: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise OfficialReportError(f"{path} must contain one JSON object")
    return payload


def _load_json_value(path: Path) -> Any:
    try:
        return _strict_json_loads(path.read_text(encoding="utf-8"), label=str(path))
    except (OSError, UnicodeError) as exc:
        raise OfficialReportError(
            f"cannot read JSON artifact {path}: {type(exc).__name__}"
        ) from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise OfficialReportError(f"cannot read JSONL {path}: {type(exc).__name__}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = _strict_json_loads(line, label=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise OfficialReportError(f"JSONL record at {path}:{line_number} must be an object")
        records.append(value)
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    except OSError as exc:
        raise OfficialReportError(f"cannot hash {path}: {type(exc).__name__}") from exc
    return digest.hexdigest()


def _artifact_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise OfficialReportError(f"evidence artifact is missing or unsafe: {path}")
    return {
        "path": resolved,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise OfficialReportError(f"cannot canonically encode receipt: {exc}") from exc


def _bound_chat_receipt(
    records: Sequence[dict[str, Any]],
    reference: Any,
    *,
    question_id: str,
    call_role: str,
    referenced_indexes: set[int],
) -> ChatResult | None:
    if reference is None:
        return None
    if not isinstance(reference, dict) or set(reference) != {"index", "sha256"}:
        raise OfficialReportError(f"{call_role} receipt reference for {question_id} is malformed")
    index = reference.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records):
        raise OfficialReportError(f"{call_role} receipt index for {question_id} is invalid")
    if index in referenced_indexes:
        raise OfficialReportError(f"chat receipt index {index} is referenced more than once")
    record = records[index]
    digest = hashlib.sha256(_canonical_json_bytes(record)).hexdigest()
    if reference.get("sha256") != digest:
        raise OfficialReportError(f"{call_role} receipt digest for {question_id} is inconsistent")
    if record.get("question_id") != question_id or record.get("call_role") != call_role:
        raise OfficialReportError(f"{call_role} receipt route for {question_id} is inconsistent")
    try:
        result = validate_chat_receipt_record(record)
    except (ValueError, ChatProtocolError) as exc:
        raise OfficialReportError(
            f"{call_role} receipt for {question_id} failed replay: {exc}"
        ) from exc
    referenced_indexes.add(index)
    return result


def _resolve_bound_artifact(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OfficialReportError(f"{label} must be a path/bytes/SHA-256 object")
    raw_path = value.get("path")
    expected_bytes = value.get("bytes")
    expected_sha256 = value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise OfficialReportError(f"{label} path must be a non-empty string")
    supplied = Path(raw_path)
    if supplied.is_symlink():
        raise OfficialReportError(f"{label} path must not be a symbolic link")
    if supplied.is_absolute():
        resolved = supplied.resolve()
    else:
        resolved = (REPO_ROOT / supplied).resolve()
        try:
            canonical_relative = resolved.relative_to(REPO_ROOT).as_posix()
        except ValueError as exc:
            raise OfficialReportError(f"{label} relative path escapes the repository") from exc
        if canonical_relative != supplied.as_posix():
            raise OfficialReportError(f"{label} path is not canonical")
    actual = _artifact_identity(resolved)
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise OfficialReportError(f"{label} bytes must be an integer")
    if actual["bytes"] != expected_bytes:
        raise OfficialReportError(
            f"{label} byte length mismatch: expected {expected_bytes}, got {actual['bytes']}"
        )
    if not isinstance(expected_sha256, str) or actual["sha256"] != expected_sha256:
        raise OfficialReportError(f"{label} SHA-256 mismatch")
    return actual


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OfficialReportError(f"{label} must be a non-empty string")
    return value


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OfficialReportError(f"{label} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise OfficialReportError(f"{label} must be a finite number")
    return numeric


def _canonical_chat_base_url(value: Any, *, label: str) -> str:
    base_url = _nonempty_string(value, label=label).strip().rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/v1"}
    ):
        raise OfficialReportError(f"{label} is not a canonical chat API base URL")
    if base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")]
    return base_url


def _protocol_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _artifact_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _unique_by_id(records: Sequence[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise OfficialReportError(f"{label} contains a non-object record")
        question_id = _nonempty_string(record.get("question_id"), label=f"{label} question_id")
        if question_id in indexed:
            raise OfficialReportError(f"{label} contains duplicate question_id {question_id!r}")
        indexed[question_id] = record
    return indexed


def _validated_implementation(
    value: Any,
    *,
    label: str,
    required_files: frozenset[str],
) -> dict[str, Any]:
    try:
        return validate_implementation_fingerprint(
            value,
            label=label,
            required_files=required_files,
        )
    except ValueError as exc:
        raise OfficialReportError(str(exc)) from exc


def _validate_retrieval_evidence(
    retrieval: dict[str, Any], *, run_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if retrieval.get("mode") != "replayed_saved_run":
        raise OfficialReportError(f"{run_path} did not replay a saved retrieval artifact")
    if retrieval.get("source_publishable") is not True:
        raise OfficialReportError(f"{run_path} retrieval source is marked nonpublishable")
    if retrieval.get("source_publishability_errors") != []:
        raise OfficialReportError(f"{run_path} carries retrieval publishability errors")
    source_identity = _resolve_bound_artifact(
        retrieval.get("source_artifact"), label=f"{run_path} retrieval source artifact"
    )
    source_path = source_identity["path"]
    source = _load_json(source_path)
    try:
        validate_retrieval_run_protocol(source)
        publishability_errors = retrieval_publishability_errors(source)
    except ValueError as exc:
        raise OfficialReportError(f"invalid bound retrieval source {source_path}: {exc}") from exc
    if publishability_errors:
        raise OfficialReportError(
            f"bound retrieval source {source_path} is not publishable: "
            + "; ".join(publishability_errors)
        )

    source_artifact = retrieval["source_artifact"]
    snapshots = {
        "source_run": source_artifact["path"],
        "source_artifact_type": source["artifact_type"],
        "source_schema_version": source["schema_version"],
        "source_protocol_version": source["protocol_version"],
        "source_implementation": source["implementation"],
        "granularity": source["granularity"],
        "source_limit": source["recall_limit"],
        "source_saved_ranking_depth": source["saved_ranking_depth"],
        "dense_lane_enabled": source["dense_lane_enabled"],
        "embedding": source["embedding"],
        "source_embedding_call_accounting": source["embedding_call_accounting"],
        "temporal_query_routing": source["temporal_query_routing"],
    }
    for field, expected in snapshots.items():
        if retrieval.get(field) != expected:
            raise OfficialReportError(
                f"{run_path} retrieval {field} does not match its bound source artifact"
            )
    replay_calls = retrieval.get("replay_embedding_call_accounting")
    expected_replay_calls = {
        "document_inputs": 0,
        "document_batch_calls": 0,
        "query_calls": 0,
        "successful_http_calls": 0,
        "source": "artifact-replay-no-provider-calls",
    }
    if replay_calls != expected_replay_calls:
        raise OfficialReportError(f"{run_path} has invalid replay embedding call accounting")
    return source_identity, source


def load_official_run(
    run_path: Path,
    labels_path: Path,
    judge_receipts_path: Path,
) -> OfficialRun:
    run = _load_json(run_path)
    expected_envelope = {
        "artifact_type": QA_RUN_ARTIFACT_TYPE,
        "schema_version": QA_ARTIFACT_SCHEMA_VERSION,
        "protocol_version": QA_PROTOCOL_VERSION,
    }
    for field, expected in expected_envelope.items():
        if type(run.get(field)) is not type(expected) or run.get(field) != expected:
            raise OfficialReportError(
                f"{run_path} {field} must be {expected!r}, got {run.get(field)!r}"
            )
    qa_implementation = _validated_implementation(
        run.get("implementation"),
        label=f"{run_path} QA implementation",
        required_files=_QA_FINGERPRINT_REQUIRED_FILES,
    )
    if run.get("harness") != "scripts/run_longmemeval_qa.py":
        raise OfficialReportError(f"{run_path} does not name the canonical QA harness")
    if run.get("task") != "longmemeval-s end-to-end QA":
        raise OfficialReportError(f"{run_path} does not name the canonical QA task")
    run_id = _nonempty_string(run.get("run_id"), label="generation run_id")
    started_at = _nonempty_string(run.get("started_at"), label="generation started_at")
    dataset = run.get("dataset")
    if not isinstance(dataset, dict):
        raise OfficialReportError(f"{run_path} has no dataset metadata")
    expected_dataset = {
        "name": "LongMemEval-S",
        "evaluated_questions": 500,
        "total_questions": 500,
        "sha256": LONGMEMEVAL_S_SHA256,
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            raise OfficialReportError(
                f"{run_path} dataset {key} must be {expected!r}, got {dataset.get(key)!r}"
            )
    dataset_identity = _resolve_bound_artifact(
        dataset.get("artifact"), label=f"{run_path} LongMemEval-S dataset artifact"
    )
    if dataset_identity["sha256"] != LONGMEMEVAL_S_SHA256:
        raise OfficialReportError(f"{run_path} is not bound to the pinned LongMemEval-S bytes")
    dataset_raw = _load_json_value(dataset_identity["path"])
    if not isinstance(dataset_raw, list) or len(dataset_raw) != 500:
        raise OfficialReportError("bound LongMemEval-S dataset must contain exactly 500 records")
    dataset_records = _unique_by_id(dataset_raw, label="bound LongMemEval-S dataset")
    judge = run.get("judge")
    if not isinstance(judge, dict) or judge.get("official_judge_model") != OFFICIAL_JUDGE_MODEL:
        raise OfficialReportError(f"{run_path} does not name the official LongMemEval judge")
    reader = run.get("reader")
    if not isinstance(reader, dict):
        raise OfficialReportError(f"{run_path} has no reader metadata")
    reader_model = _nonempty_string(reader.get("model"), label="reader model")
    reader_revision = _nonempty_string(reader.get("revision"), label="reader revision")
    if reader.get("revision_source") != "operator-pinned deployment/checkpoint":
        raise OfficialReportError("reader revision source is not the canonical declared source")
    if reader.get("response_model_requirement") != reader_model:
        raise OfficialReportError("reader response model requirement must equal the request model")
    if reader.get("request_id_required") is not True:
        raise OfficialReportError("reader provider request IDs were not required")
    if reader.get("response_parser") != CHAT_RESPONSE_PARSER:
        raise OfficialReportError("reader raw response parser identity is invalid")
    if reader.get("request_parser") != CHAT_REQUEST_PARSER:
        raise OfficialReportError("reader raw request parser identity is invalid")
    if reader.get("raw_request_receipts_required") is not True:
        raise OfficialReportError("reader raw request receipts were not required")
    if reader.get("raw_prompt_receipts_required") is not True:
        raise OfficialReportError("reader raw prompt receipts were not required")
    if reader.get("raw_response_receipts_required") is not True:
        raise OfficialReportError("reader raw response receipts were not required")
    if reader.get("provider_usage_replay_required") is not True:
        raise OfficialReportError("reader provider usage replay was not required")
    if reader.get("response_evidence_publishable") is not True:
        raise OfficialReportError("reader response evidence is marked nonpublishable")
    prompt_style = _nonempty_string(reader.get("prompt_style"), label="reader prompt style")
    reader_base_url = _canonical_chat_base_url(reader.get("base_url"), label="reader base URL")
    reader_endpoint_url = f"{reader_base_url}/v1/chat/completions"
    reader_temperature = _finite_number(reader.get("temperature"), label="reader temperature")
    reader_max_tokens = reader.get("max_tokens")
    if isinstance(reader_max_tokens, bool) or not isinstance(reader_max_tokens, int):
        raise OfficialReportError("reader max_tokens must be an integer")
    if not 1 <= reader_max_tokens <= 1_000_000:
        raise OfficialReportError("reader max_tokens is out of range")
    reader_thinking_mode = reader.get("thinking_mode")
    if reader_thinking_mode not in {None, "enabled", "disabled"}:
        raise OfficialReportError("reader thinking_mode is invalid")
    expected_thinking_source = (
        "explicit-request-field" if reader_thinking_mode is not None else "provider-default-omitted"
    )
    if reader.get("thinking_mode_source") != expected_thinking_source:
        raise OfficialReportError("reader thinking_mode_source is inconsistent")
    prompt_template_source = _nonempty_string(
        reader.get("prompt_template_source"), label="reader prompt template source"
    )
    reader_protocol_json = _protocol_json(
        {
            "model": reader_model,
            "revision": reader_revision,
            "revision_source": "operator-pinned deployment/checkpoint",
            "response_model_requirement": reader_model,
            "request_id_required": True,
            "response_parser": CHAT_RESPONSE_PARSER,
            "request_parser": CHAT_REQUEST_PARSER,
            "raw_request_receipts_required": True,
            "raw_prompt_receipts_required": True,
            "raw_response_receipts_required": True,
            "provider_usage_replay_required": True,
            "temperature": reader_temperature,
            "max_tokens": reader_max_tokens,
            "thinking_mode": reader_thinking_mode,
            "thinking_mode_source": expected_thinking_source,
            "base_url": reader_base_url,
            "endpoint_url": reader_endpoint_url,
            "prompt_style": prompt_style,
            "prompt_template_source": prompt_template_source,
        }
    )

    retrieval = run.get("retrieval")
    if not isinstance(retrieval, dict):
        raise OfficialReportError(f"{run_path} has no retrieval metadata")
    retrieval_source_identity, retrieval_source = _validate_retrieval_evidence(
        retrieval, run_path=run_path
    )
    retrieval_limit = retrieval.get("limit")
    if isinstance(retrieval_limit, bool) or not isinstance(retrieval_limit, int):
        raise OfficialReportError("retrieval limit must be an integer")
    if not 1 <= retrieval_limit <= 50:
        raise OfficialReportError("retrieval limit must be between 1 and 50")
    if retrieval_limit > retrieval_source["recall_limit"]:
        raise OfficialReportError("retrieval limit exceeds the bound source recall_limit")
    retrieval_min_score = _finite_number(retrieval.get("min_score"), label="retrieval min_score")
    if retrieval_min_score != 0.0:
        raise OfficialReportError("saved-run official replay requires retrieval min_score 0.0")
    dense_lane_enabled = retrieval.get("dense_lane_enabled")
    if not isinstance(dense_lane_enabled, bool):
        raise OfficialReportError("retrieval dense_lane_enabled must be boolean")
    granularity = _nonempty_string(retrieval.get("granularity"), label="retrieval granularity")
    embedding = retrieval.get("embedding")
    if embedding is not None and not isinstance(embedding, dict):
        raise OfficialReportError("retrieval embedding metadata must be an object or null")
    retrieval_protocol_json = _protocol_json(
        {
            "mode": retrieval.get("mode"),
            "source_artifact_type": retrieval_source["artifact_type"],
            "source_schema_version": retrieval_source["schema_version"],
            "source_protocol_version": retrieval_source["protocol_version"],
            "source_sha256": retrieval_source_identity["sha256"],
            "source_bytes": retrieval_source_identity["bytes"],
            "source_implementation_tree_sha256": retrieval_source["implementation"]["tree_sha256"],
            "granularity": granularity,
            "source_limit": retrieval.get("source_limit"),
            "source_saved_ranking_depth": retrieval.get("source_saved_ranking_depth"),
            "limit": retrieval_limit,
            "min_score": retrieval_min_score,
            "dense_lane_enabled": dense_lane_enabled,
            "embedding": embedding,
            "source_embedding_call_accounting": retrieval.get("source_embedding_call_accounting"),
            "replay_embedding_call_accounting": retrieval.get("replay_embedding_call_accounting"),
            "temporal_query_routing": retrieval.get("temporal_query_routing"),
        }
    )
    questions_raw = run.get("questions")
    if not isinstance(questions_raw, list) or len(questions_raw) != 500:
        raise OfficialReportError(f"{run_path} must contain exactly 500 question records")
    questions = _unique_by_id(questions_raw, label="generation run")
    if set(dataset_records) != set(questions):
        raise OfficialReportError("bound dataset coverage differs from generation run")
    if run.get("chat_receipt_protocol") != CHAT_RECEIPT_PROTOCOL_VERSION:
        raise OfficialReportError("generation run chat receipt protocol identity is invalid")
    chat_receipt_identity = _resolve_bound_artifact(
        run.get("chat_receipt_artifact"),
        label=f"{run_path} chat receipt artifact",
    )
    try:
        _, chat_receipts = load_chat_receipt_artifact(chat_receipt_identity["path"])
    except (ValueError, ChatProtocolError) as exc:
        raise OfficialReportError(f"invalid bound chat receipt artifact: {exc}") from exc
    receipt_count = run.get("chat_receipt_count")
    if (
        isinstance(receipt_count, bool)
        or not isinstance(receipt_count, int)
        or receipt_count != len(chat_receipts)
    ):
        raise OfficialReportError("generation run chat receipt count is inconsistent")

    source_cases_raw = retrieval_source.get("cases")
    if not isinstance(source_cases_raw, list):
        raise OfficialReportError("bound retrieval source has no case records")
    source_cases: dict[str, dict[str, Any]] = {}
    for source_case in source_cases_raw:
        if not isinstance(source_case, dict):
            raise OfficialReportError("bound retrieval source contains a malformed case")
        case_id = _nonempty_string(
            source_case.get("case_id"), label="bound retrieval source case_id"
        )
        if case_id in source_cases:
            raise OfficialReportError(
                f"bound retrieval source contains duplicate case_id {case_id!r}"
            )
        source_cases[case_id] = source_case
    if set(source_cases) != set(questions):
        raise OfficialReportError("bound retrieval source coverage differs from generation run")
    expected_reader_prompts: dict[str, bytes] = {}
    for question_id, question in questions.items():
        source_case = source_cases[question_id]
        dataset_record = dataset_records[question_id]
        haystack_sessions = dataset_record.get("haystack_sessions")
        if not isinstance(haystack_sessions, list):
            raise OfficialReportError(
                f"bound dataset record {question_id} has malformed haystack sessions"
            )
        if source_case.get("haystack_sessions") != len(haystack_sessions):
            raise OfficialReportError(
                f"bound retrieval source haystack count differs from dataset for {question_id}"
            )
        rankings = source_case.get("rankings")
        final = rankings.get("final") if isinstance(rankings, dict) else None
        relevance = source_case.get("final_relevance")
        if not isinstance(final, list) or not isinstance(relevance, list):
            raise OfficialReportError(
                f"bound retrieval source case {question_id} lacks final bundle evidence"
            )
        expected_keys = [str(value) for value in final[:retrieval_limit]]
        expected_relevance = relevance[: len(expected_keys)]
        if len(expected_relevance) != len(expected_keys):
            raise OfficialReportError(
                f"bound retrieval source case {question_id} has incomplete relevance evidence"
            )
        if question.get("retrieved_session_keys") != expected_keys:
            raise OfficialReportError(f"generation retrieval bundle mismatch for {question_id}")
        if question.get("retrieved_relevance") != expected_relevance:
            raise OfficialReportError(f"generation retrieval relevance mismatch for {question_id}")
        if question.get("temporal_routing") != source_case.get("temporal_routing"):
            raise OfficialReportError(f"generation temporal trace mismatch for {question_id}")
        try:
            replayed = replay_retrieval_case(
                dataset_record,
                source_case,
                limit=retrieval_limit,
                min_score=0.0,
            )
            prompt = build_reader_prompt(
                dataset_record,
                [(session.date, session.turns) for session, _ in replayed.hits],
                style=prompt_style,
                requested=retrieval_limit,
                floored=False,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OfficialReportError(
                f"cannot reconstruct reader prompt for {question_id}: {exc}"
            ) from exc
        expected_reader_prompts[question_id] = prompt.encode("utf-8")

    hypothesis_identity = _resolve_bound_artifact(
        run.get("hypothesis_artifact"), label=f"{run_path} hypothesis artifact"
    )
    hypothesis_reference = run.get("hypotheses")
    if hypothesis_reference != run["hypothesis_artifact"]["path"]:
        raise OfficialReportError(f"{run_path} hypothesis path is not bound consistently")
    hypotheses_raw = _load_jsonl(hypothesis_identity["path"])
    if len(hypotheses_raw) != 500:
        raise OfficialReportError(
            f"{hypothesis_identity['path']} must contain exactly 500 generated hypotheses"
        )
    hypotheses = _unique_by_id(hypotheses_raw, label="generated hypotheses")
    if set(hypotheses) != set(questions):
        raise OfficialReportError("generated hypothesis coverage differs from generation run")
    for question_id, hypothesis_record in hypotheses.items():
        if set(hypothesis_record) != {"question_id", "hypothesis"}:
            raise OfficialReportError(
                f"generated hypothesis record {question_id!r} has unexpected fields"
            )
        hypothesis = hypothesis_record.get("hypothesis")
        if not isinstance(hypothesis, str) or hypothesis != questions[question_id].get(
            "hypothesis"
        ):
            raise OfficialReportError(f"bound hypothesis artifact mismatch for {question_id}")

    labels_raw = _load_jsonl(labels_path)
    if len(labels_raw) != 500:
        raise OfficialReportError(f"{labels_path} must contain exactly 500 official labels")
    labels = _unique_by_id(labels_raw, label="official labels")
    if set(labels) != set(questions):
        missing = sorted(set(questions) - set(labels))[:3]
        extra = sorted(set(labels) - set(questions))[:3]
        raise OfficialReportError(
            f"official label coverage differs from generation run; missing={missing}, extra={extra}"
        )
    try:
        _, official_judge_receipts = load_chat_receipt_artifact(judge_receipts_path)
    except (ValueError, ChatProtocolError) as exc:
        raise OfficialReportError(f"invalid official judge receipt artifact: {exc}") from exc
    if len(official_judge_receipts) != 500:
        raise OfficialReportError("official judge receipt artifact must contain exactly 500 rows")
    official_judge_results: dict[str, ChatResult] = {}
    for record in official_judge_receipts:
        question_id = _nonempty_string(
            record.get("question_id"),
            label="official judge receipt question_id",
        )
        if record.get("call_role") != "official_judge":
            raise OfficialReportError("official judge receipt artifact contains a nonofficial role")
        if question_id in official_judge_results:
            raise OfficialReportError(f"official judge receipts repeat {question_id!r}")
        try:
            result = validate_chat_receipt_record(record)
        except (ValueError, ChatProtocolError) as exc:
            raise OfficialReportError(
                f"official judge receipt for {question_id} failed replay: {exc}"
            ) from exc
        official_judge_results[question_id] = result
    if set(official_judge_results) != set(questions):
        raise OfficialReportError("official judge receipt coverage differs from generation run")

    judged: list[JudgedQuestion] = []
    reader_request_ids: list[str] = []
    official_judge_request_ids: list[str] = []
    official_judge_prompt_tokens = 0
    official_judge_completion_tokens = 0
    reader_failures = 0
    referenced_receipt_indexes: set[int] = set()
    for question_id, question in questions.items():
        dataset_record = dataset_records[question_id]
        question_type = _nonempty_string(
            question.get("question_type"), label=f"question type for {question_id}"
        )
        if question_type != dataset_record.get("question_type"):
            raise OfficialReportError(f"generation question type mismatch for {question_id}")
        hypothesis = question.get("hypothesis")
        if not isinstance(hypothesis, str):
            raise OfficialReportError(f"generation hypothesis for {question_id} must be a string")
        reader_result = _bound_chat_receipt(
            chat_receipts,
            question.get("reader_receipt"),
            question_id=question_id,
            call_role="reader",
            referenced_indexes=referenced_receipt_indexes,
        )
        reader_error = question.get("reader_error")
        if reader_error is not None:
            if not isinstance(reader_error, str) or not reader_error:
                raise OfficialReportError(
                    f"reader error for {question_id} must be a string or null"
                )
            if reader_result is not None:
                raise OfficialReportError(
                    f"failed reader call for {question_id} cannot carry a success receipt"
                )
            reader_failures += 1
        else:
            if reader_result is None:
                raise OfficialReportError(f"reader response receipt is missing for {question_id}")
            if reader_result.response_model != reader_model:
                raise OfficialReportError(f"reader response model mismatch for {question_id}")
            if reader_result.finish_reason != "stop":
                raise OfficialReportError(f"reader response was truncated for {question_id}")
            reader_request = reader_result.request
            if (
                reader_request.model != reader_model
                or reader_request.temperature != reader_temperature
                or reader_request.max_tokens != reader_max_tokens
                or reader_request.thinking_mode != reader_thinking_mode
                or reader_result.endpoint_url != reader_endpoint_url
            ):
                raise OfficialReportError(
                    f"reader raw request controls differ from the frozen protocol for {question_id}"
                )
            if reader_result.prompt_bytes != expected_reader_prompts[question_id]:
                raise OfficialReportError(
                    f"reader prompt differs from bound dataset and retrieval for {question_id}"
                )
            expected_reader_fields = {
                "hypothesis": reader_result.content,
                "reader_prompt_tokens": reader_result.prompt_tokens,
                "reader_completion_tokens": reader_result.completion_tokens,
                "reader_total_tokens": reader_result.total_tokens,
                "reader_finish_reason": reader_result.finish_reason,
                "reader_attempts": reader_result.attempts,
                "reader_response_model": reader_result.response_model,
                "reader_request_id": reader_result.request_id,
                "reader_system_fingerprint": reader_result.system_fingerprint,
                "reader_prompt_sha256": reader_result.prompt_sha256,
                "reader_prompt_utf8_bytes": reader_result.prompt_utf8_bytes,
                "reader_raw_response_sha256": reader_result.raw_response_sha256,
                "reader_raw_request_sha256": reader_result.raw_request_sha256,
            }
            for field, expected_value in expected_reader_fields.items():
                if question.get(field) != expected_value:
                    raise OfficialReportError(
                        f"reader receipt field {field} mismatch for {question_id}"
                    )
            request_id = _nonempty_string(
                reader_result.request_id,
                label=f"reader provider request id for {question_id}",
            )
            reader_request_ids.append(request_id)
        development_judge = _bound_chat_receipt(
            chat_receipts,
            question.get("dev_judge_receipt"),
            question_id=question_id,
            call_role="development_judge",
            referenced_indexes=referenced_receipt_indexes,
        )
        if development_judge is not None:
            try:
                expected_judge_prompt = judge_prompt(
                    question_type,
                    str(dataset_record["question"]),
                    str(dataset_record["answer"]),
                    hypothesis,
                    abstention=is_abstention_question(question_id),
                ).encode("utf-8")
            except (KeyError, NotImplementedError) as exc:
                raise OfficialReportError(
                    f"cannot reconstruct judge prompt for {question_id}: {exc}"
                ) from exc
            if development_judge.prompt_bytes != expected_judge_prompt:
                raise OfficialReportError(
                    f"development judge prompt differs from bound dataset for {question_id}"
                )
            expected_judge_fields = {
                "dev_judge_response": development_judge.content,
                "dev_judge_prompt_tokens": development_judge.prompt_tokens,
                "dev_judge_completion_tokens": development_judge.completion_tokens,
                "dev_judge_total_tokens": development_judge.total_tokens,
                "dev_judge_response_model": development_judge.response_model,
                "dev_judge_request_id": development_judge.request_id,
                "dev_judge_system_fingerprint": development_judge.system_fingerprint,
                "dev_judge_prompt_sha256": development_judge.prompt_sha256,
                "dev_judge_prompt_utf8_bytes": development_judge.prompt_utf8_bytes,
                "dev_judge_raw_response_sha256": development_judge.raw_response_sha256,
                "dev_judge_raw_request_sha256": development_judge.raw_request_sha256,
            }
            for field, expected_value in expected_judge_fields.items():
                if question.get(field) != expected_value:
                    raise OfficialReportError(
                        f"development judge receipt field {field} mismatch for {question_id}"
                    )
        official = labels[question_id]
        if official.get("hypothesis") != hypothesis:
            raise OfficialReportError(f"official label hypothesis mismatch for {question_id}")
        autoeval = official.get("autoeval_label")
        if (
            not isinstance(autoeval, dict)
            or set(autoeval) != {"model", "label"}
            or autoeval.get("model") != OFFICIAL_JUDGE_MODEL
        ):
            raise OfficialReportError(f"official label for {question_id} uses a nonofficial judge")
        label = autoeval.get("label")
        if not isinstance(label, bool):
            raise OfficialReportError(f"official label for {question_id} is not boolean")
        judge_result = official_judge_results[question_id]
        if judge_result.response_model != OFFICIAL_JUDGE_MODEL:
            raise OfficialReportError(f"official judge response model mismatch for {question_id}")
        if judge_result.finish_reason != "stop":
            raise OfficialReportError(f"official judge response was truncated for {question_id}")
        official_request = judge_result.request
        if (
            official_request.model != OFFICIAL_JUDGE_MODEL
            or official_request.temperature != 0.0
            or official_request.max_tokens != 10
            or official_request.thinking_mode is not None
            or judge_result.endpoint_url != "https://api.openai.com/v1/chat/completions"
        ):
            raise OfficialReportError(
                f"official judge raw request controls differ from protocol for {question_id}"
            )
        try:
            expected_official_prompt = judge_prompt(
                question_type,
                str(dataset_record["question"]),
                str(dataset_record["answer"]),
                hypothesis,
                abstention=is_abstention_question(question_id),
            ).encode("utf-8")
        except (KeyError, NotImplementedError) as exc:
            raise OfficialReportError(
                f"cannot reconstruct official judge prompt for {question_id}: {exc}"
            ) from exc
        if judge_result.prompt_bytes != expected_official_prompt:
            raise OfficialReportError(
                f"official judge prompt differs from bound dataset for {question_id}"
            )
        judge_request_id = _nonempty_string(
            judge_result.request_id,
            label=f"official judge provider request id for {question_id}",
        )
        if judge_label(judge_result.content) is not label:
            raise OfficialReportError(
                f"official label for {question_id} differs from replayed GPT-4o response"
            )
        official_judge_request_ids.append(judge_request_id)
        official_judge_prompt_tokens += judge_result.prompt_tokens
        official_judge_completion_tokens += judge_result.completion_tokens
        judged.append(
            JudgedQuestion(question_id=question_id, question_type=question_type, label=label)
        )
    judged.sort(key=lambda value: value.question_id)
    if len(set(reader_request_ids)) != len(reader_request_ids):
        raise OfficialReportError("reader provider request IDs must be unique within a run")
    if len(set(official_judge_request_ids)) != len(official_judge_request_ids):
        raise OfficialReportError("official judge provider request IDs must be unique within a run")
    if referenced_receipt_indexes != set(range(len(chat_receipts))):
        raise OfficialReportError("chat receipt artifact contains unreferenced provider responses")
    run_identity = _artifact_identity(run_path)
    labels_identity = _artifact_identity(labels_path)
    judge_receipt_identity = _artifact_identity(judge_receipts_path)
    return OfficialRun(
        run_path=run_path,
        labels_path=labels_path,
        run_id=run_id,
        started_at=started_at,
        reader_model=reader_model,
        prompt_style=prompt_style,
        reader_protocol_json=reader_protocol_json,
        retrieval_protocol_json=retrieval_protocol_json,
        retrieval_limit=retrieval_limit,
        run_sha256=run_identity["sha256"],
        run_bytes=run_identity["bytes"],
        labels_sha256=labels_identity["sha256"],
        labels_bytes=labels_identity["bytes"],
        judge_receipt_path=judge_receipt_identity["path"],
        judge_receipt_sha256=judge_receipt_identity["sha256"],
        judge_receipt_bytes=judge_receipt_identity["bytes"],
        hypothesis_path=hypothesis_identity["path"],
        hypothesis_sha256=hypothesis_identity["sha256"],
        hypothesis_bytes=hypothesis_identity["bytes"],
        chat_receipt_path=chat_receipt_identity["path"],
        chat_receipt_sha256=chat_receipt_identity["sha256"],
        chat_receipt_bytes=chat_receipt_identity["bytes"],
        retrieval_source_path=retrieval_source_identity["path"],
        retrieval_source_sha256=retrieval_source_identity["sha256"],
        retrieval_source_bytes=retrieval_source_identity["bytes"],
        dataset_path=dataset_identity["path"],
        dataset_sha256=dataset_identity["sha256"],
        dataset_bytes=dataset_identity["bytes"],
        qa_implementation_tree_sha256=qa_implementation["tree_sha256"],
        retrieval_implementation_tree_sha256=retrieval_source["implementation"]["tree_sha256"],
        reader_request_ids=tuple(sorted(reader_request_ids)),
        official_judge_request_ids=tuple(sorted(official_judge_request_ids)),
        official_judge_prompt_tokens=official_judge_prompt_tokens,
        official_judge_completion_tokens=official_judge_completion_tokens,
        questions=tuple(judged),
        reader_failures=reader_failures,
    )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise OfficialReportError("cannot compute a percentile of no values")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _question_cluster_bootstrap(
    runs: Sequence[OfficialRun], *, samples: int, seed: int
) -> tuple[float, float]:
    if samples < 100:
        raise OfficialReportError("bootstrap_samples must be at least 100")
    question_ids = tuple(question.question_id for question in runs[0].questions)
    run_labels = [
        {question.question_id: float(question.label) for question in run.questions} for run in runs
    ]
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        drawn = [question_ids[rng.randrange(len(question_ids))] for _ in question_ids]
        values.append(mean(labels[question_id] for labels in run_labels for question_id in drawn))
    values.sort()
    return _percentile(values, 0.025), _percentile(values, 0.975)


def build_report(
    runs: Sequence[OfficialRun], *, bootstrap_samples: int = 10_000, bootstrap_seed: int = 20260809
) -> dict[str, Any]:
    if not runs:
        raise OfficialReportError("at least one official run is required")
    reference_ids = tuple(question.question_id for question in runs[0].questions)
    for run in runs[1:]:
        if tuple(question.question_id for question in run.questions) != reference_ids:
            raise OfficialReportError("official runs do not cover the same question IDs")
    readers = {run.reader_model for run in runs}
    prompt_styles = {run.prompt_style for run in runs}
    if len(readers) != 1:
        raise OfficialReportError("official runs must use one fixed reader model")
    if len(prompt_styles) != 1:
        raise OfficialReportError("official runs must use one fixed reader prompt style")
    if len({run.reader_protocol_json for run in runs}) != 1:
        raise OfficialReportError("official runs must use one fixed reader protocol")
    if len({run.retrieval_protocol_json for run in runs}) != 1:
        raise OfficialReportError("official runs must use one fixed retrieval protocol")
    if len({run.qa_implementation_tree_sha256 for run in runs}) != 1:
        raise OfficialReportError("official runs must use one fixed QA implementation")
    if len({run.retrieval_implementation_tree_sha256 for run in runs}) != 1:
        raise OfficialReportError("official runs must use one fixed retrieval implementation")
    if len({run.retrieval_source_sha256 for run in runs}) != 1:
        raise OfficialReportError("official runs must replay one exact retrieval source")
    if len({run.retrieval_source_bytes for run in runs}) != 1:
        raise OfficialReportError("official retrieval source byte lengths differ")
    if len({run.dataset_sha256 for run in runs}) != 1:
        raise OfficialReportError("official runs do not bind one exact LongMemEval-S dataset")
    if len({run.dataset_bytes for run in runs}) != 1:
        raise OfficialReportError("official LongMemEval-S dataset byte lengths differ")
    if len({run.run_id for run in runs}) != len(runs):
        raise OfficialReportError("official runs must have unique run IDs")
    if len({run.started_at for run in runs}) != len(runs):
        raise OfficialReportError("official runs must have unique start times")
    if len({run.run_path.resolve() for run in runs}) != len(runs):
        raise OfficialReportError("generation evidence paths must be unique")
    if len({run.labels_path.resolve() for run in runs}) != len(runs):
        raise OfficialReportError("official label evidence paths must be unique")
    if len({run.hypothesis_path.resolve() for run in runs}) != len(runs):
        raise OfficialReportError("generated hypothesis evidence paths must be unique")
    if len({run.chat_receipt_path.resolve() for run in runs}) != len(runs):
        raise OfficialReportError("chat receipt evidence paths must be unique")
    if len({run.judge_receipt_path.resolve() for run in runs}) != len(runs):
        raise OfficialReportError("official judge receipt evidence paths must be unique")
    seen_reader_request_ids: set[str] = set()
    seen_judge_request_ids: set[str] = set()
    for run in runs:
        overlap = seen_reader_request_ids.intersection(run.reader_request_ids)
        if overlap:
            raise OfficialReportError(
                "official runs reuse reader provider request IDs and are not independent"
            )
        seen_reader_request_ids.update(run.reader_request_ids)
        judge_overlap = seen_judge_request_ids.intersection(run.official_judge_request_ids)
        if judge_overlap:
            raise OfficialReportError(
                "official runs reuse judge provider request IDs and are not independent"
            )
        seen_judge_request_ids.update(run.official_judge_request_ids)

    by_type: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for question in run.questions:
            by_type[question.question_type].append(float(question.label))
    lower, upper = _question_cluster_bootstrap(
        runs,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    accuracies = [run.accuracy for run in runs]
    return {
        "artifact_type": OFFICIAL_REPORT_ARTIFACT_TYPE,
        "schema_version": OFFICIAL_REPORT_SCHEMA_VERSION,
        "protocol_version": OFFICIAL_REPORT_PROTOCOL_VERSION,
        "dataset": {
            "name": "LongMemEval-S",
            "evaluated_questions": 500,
            "total_questions": 500,
            "sha256": LONGMEMEVAL_S_SHA256,
            "artifact_bytes": runs[0].dataset_bytes,
            "prompt_source_bound": True,
        },
        "judge": {
            "official": True,
            "model": OFFICIAL_JUDGE_MODEL,
            "implementation": OFFICIAL_EVALUATOR,
            "raw_prompt_and_response_replay": True,
            "raw_request_and_response_replay": True,
            "prompt_reconstructed_from_bound_dataset": True,
            "receipt_protocol": CHAT_RECEIPT_PROTOCOL_VERSION,
            "temperature": 0.0,
            "max_tokens": 10,
        },
        "comparability": {
            "claim": "official-protocol LongMemEval-S QA result",
            "exabase_0_964_protocol_comparable": False,
            "reason": (
                "the external 0.964 result used a Mem0 fork, Gemini 3 Flash as both "
                "reader and judge, and top-50 retrieval; this artifact uses the official "
                "GPT-4o judge, a separately disclosed reader, and its bound retrieval limit"
            ),
        },
        "generation": {
            "reader_model": next(iter(readers)),
            "prompt_style": next(iter(prompt_styles)),
            "qa_artifact_type": QA_RUN_ARTIFACT_TYPE,
            "qa_schema_version": QA_ARTIFACT_SCHEMA_VERSION,
            "qa_protocol_version": QA_PROTOCOL_VERSION,
            "chat_receipt_protocol_version": CHAT_RECEIPT_PROTOCOL_VERSION,
            "raw_prompt_and_response_replay": True,
            "reader_prompt_reconstructed_from_bound_dataset_and_retrieval": True,
            "qa_implementation_tree_sha256": runs[0].qa_implementation_tree_sha256,
            "retrieval_artifact_type": RETRIEVAL_RUN_ARTIFACT_TYPE,
            "retrieval_schema_version": RETRIEVAL_ARTIFACT_SCHEMA_VERSION,
            "retrieval_protocol_version": RETRIEVAL_PROTOCOL_VERSION,
            "retrieval_implementation_tree_sha256": (runs[0].retrieval_implementation_tree_sha256),
            "retrieval_source_sha256": runs[0].retrieval_source_sha256,
            "retrieval_source_bytes": runs[0].retrieval_source_bytes,
            "reader": json.loads(runs[0].reader_protocol_json),
            "retrieval": json.loads(runs[0].retrieval_protocol_json),
            "protocol_sha256": hashlib.sha256(
                (runs[0].reader_protocol_json + runs[0].retrieval_protocol_json).encode()
            ).hexdigest(),
        },
        "runs": {
            "count": len(runs),
            "items": [
                {
                    "run_id": run.run_id,
                    "started_at": run.started_at,
                    "generation_run": _artifact_path(run.run_path),
                    "generation_run_bytes": run.run_bytes,
                    "generation_run_sha256": run.run_sha256,
                    "generated_hypotheses": _artifact_path(run.hypothesis_path),
                    "generated_hypotheses_bytes": run.hypothesis_bytes,
                    "generated_hypotheses_sha256": run.hypothesis_sha256,
                    "chat_receipts": _artifact_path(run.chat_receipt_path),
                    "chat_receipts_bytes": run.chat_receipt_bytes,
                    "chat_receipts_sha256": run.chat_receipt_sha256,
                    "retrieval_source": _artifact_path(run.retrieval_source_path),
                    "retrieval_source_bytes": run.retrieval_source_bytes,
                    "retrieval_source_sha256": run.retrieval_source_sha256,
                    "dataset": _artifact_path(run.dataset_path),
                    "dataset_bytes": run.dataset_bytes,
                    "dataset_sha256": run.dataset_sha256,
                    "official_labels": _artifact_path(run.labels_path),
                    "official_labels_bytes": run.labels_bytes,
                    "official_labels_sha256": run.labels_sha256,
                    "official_judge_receipts": _artifact_path(run.judge_receipt_path),
                    "official_judge_receipts_bytes": run.judge_receipt_bytes,
                    "official_judge_receipts_sha256": run.judge_receipt_sha256,
                    "official_judge_prompt_tokens": run.official_judge_prompt_tokens,
                    "official_judge_completion_tokens": run.official_judge_completion_tokens,
                    "official_judge_request_ids": len(run.official_judge_request_ids),
                    "official_judge_request_ids_sha256": hashlib.sha256(
                        "\n".join(run.official_judge_request_ids).encode("utf-8")
                    ).hexdigest(),
                    "correct": sum(question.label for question in run.questions),
                    "questions": len(run.questions),
                    "accuracy": run.accuracy,
                    "reader_failures": run.reader_failures,
                    "reader_request_ids": len(run.reader_request_ids),
                    "reader_request_ids_sha256": hashlib.sha256(
                        "\n".join(run.reader_request_ids).encode("utf-8")
                    ).hexdigest(),
                }
                for run in runs
            ],
        },
        "overall": {
            "accuracy_mean": mean(accuracies),
            "accuracy_min": min(accuracies),
            "accuracy_max": max(accuracies),
            "accuracy_ci95": {
                "lower": lower,
                "upper": upper,
                "method": "question-cluster-bootstrap-over-repeated-runs",
                "samples": bootstrap_samples,
                "seed": bootstrap_seed,
            },
        },
        "by_question_type": {
            question_type: {
                "accuracy": mean(labels),
                "judgments": len(labels),
            }
            for question_type, labels in sorted(by_type.items())
        },
        "failures": {
            "unjudged_questions": 0,
            "reader_failures": sum(run.reader_failures for run in runs),
        },
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        runs = tuple(
            load_official_run(
                Path(run_path),
                Path(labels_path),
                Path(judge_receipts_path),
            )
            for run_path, labels_path, judge_receipts_path in args.evidence
        )
        report = build_report(
            runs,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    except OfficialReportError as exc:
        print(f"cannot build official LongMemEval report: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Official accuracy mean: {report['overall']['accuracy_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
