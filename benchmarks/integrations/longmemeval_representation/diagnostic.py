"""Fail-closed R0-versus-R1 E6 development diagnostic compiler.

The compiler only reopens sealed, content-free case artifacts.  It does not
extract keys, score text, pack prompts, call a reader or judge, or authorize a
serving change.  QA evidence is optional and descriptive; it is never a
promotion surface in this development diagnostic.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    ARTIFACT_TYPE as REPRESENTATION_ARTIFACT_TYPE,
)
from .contracts import (
    MAX_HYDRATED_VALUES,
    RepresentationCell,
    RepresentationError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
)
from .contracts import (
    PROTOCOL_VERSION as REPRESENTATION_PROTOCOL_VERSION,
)
from .contracts import (
    SCHEMA_VERSION as REPRESENTATION_SCHEMA_VERSION,
)

CASE_ARTIFACT_TYPE = "swarmbrain-longmemeval-e6-r0-r1-development-case"
REPORT_ARTIFACT_TYPE = "swarmbrain-longmemeval-e6-r0-r1-development-diagnostic"
CASE_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "E6/R0-vs-R1-development-diagnostic-v1"
PROMPT_BUDGET_TOKENS = 8192
R0 = RepresentationCell.RAW.value
R1 = RepresentationCell.RAW_MERGED_SFK.value
ARMS = (R0, R1)

_CASE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "protocol_version",
        "case_index",
        "question_id",
        "question_type",
        "gold_session_sha256s",
        "arms",
        "artifact_sha256",
    }
)
_ARM_FIELDS = frozenset({"cell", "representation", "context", "query_accounting", "qa"})
_CONTEXT_FIELDS = frozenset(
    {
        "candidate_session_sha256s",
        "prompt_value_ids",
        "prompt_tokens",
        "prompt_sha256",
        "tokenizer_artifact_sha256",
        "tokenizer_receipt_sha256",
    }
)
_QUERY_ACCOUNTING_FIELDS = frozenset({"complete", "source", "stages", "stages_sha256"})
_STAGE_FIELDS = frozenset(
    {
        "name",
        "calls",
        "input_tokens",
        "output_tokens",
        "latency_microseconds",
        "cost_microusd",
        "retry_count",
        "cache_hits",
    }
)
_QA_FIELDS = frozenset({"correct", "reader_receipt_sha256", "judge_receipt_sha256"})
_REPRESENTATION_FIELDS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "protocol_version",
        "cell",
        "classification",
        "production_configuration",
        "paper_reproduction",
        "sb_hypothesis",
        "frozen_protocol",
        "promotion",
        "corpus",
        "observations",
        "observations_sha256",
        "ranking",
        "graph",
        "value_scores",
        "value_scores_sha256",
        "key_level_returned_count",
        "hydrated_value_pre_cap_count",
        "hydrated_value_cap",
        "hydrated_value_ids",
        "hydrated_raw_value_hashes",
        "hydrated_value_count",
        "hydration",
        "construction_and_index_accounting",
        "construction_input_contract",
        "claims",
        "trace_sha256",
    }
)
_CONSTRUCTION_FIELDS = frozenset(
    {
        "canonical_value_count",
        "canonical_value_utf8_bytes",
        "active_indexed_key_count",
        "active_indexed_key_utf8_bytes",
        "derived_key_count",
        "derived_key_utf8_bytes",
        "derived_objects_per_source",
        "derived_objects_per_source_sha256",
        "construction_receipt_count",
        "construction_receipts_sha256",
        "extractor_identities",
        "construction_artifact_sha256s",
        "construction_accounting",
        "duplicate_key_text",
        "orphan_keys",
        "update_rate",
        "index_token_count",
        "index_token_count_status",
    }
)
_CONSTRUCTION_ACCOUNTING_FIELDS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "latency_microseconds",
        "cost_microusd",
        "retry_count",
        "cache_hits",
    }
)

_QUALITY_AXES = (
    "candidate_any_gold_in_context",
    "candidate_all_gold_in_context",
    "candidate_answer_session_recall",
    "candidate_answer_session_mrr",
    "prompt_any_gold_in_context",
    "prompt_all_gold_in_context",
    "prompt_answer_session_recall",
    "prompt_answer_session_mrr",
)


class RepresentationDiagnosticError(RepresentationError):
    """A sealed E6 diagnostic input is malformed, incomplete, or incomparable."""


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise RepresentationDiagnosticError(f"duplicate JSON field {key!r}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise RepresentationDiagnosticError(f"non-finite JSON constant {value!r} is forbidden")


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RepresentationDiagnosticError) as exc:
        raise RepresentationDiagnosticError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RepresentationDiagnosticError(f"{label} must contain one JSON object")
    return value


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RepresentationDiagnosticError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], fields: frozenset[str], *, label: str) -> None:
    if set(value) != fields:
        raise RepresentationDiagnosticError(f"{label} fields differ from the frozen schema")


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RepresentationDiagnosticError(f"{label} must be non-empty trimmed text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise RepresentationDiagnosticError(f"{label} cannot contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise RepresentationDiagnosticError(f"{label} must be valid UTF-8") from exc
    return value


def _digest(value: Any, *, label: str) -> str:
    text = _text(value, label=label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RepresentationDiagnosticError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RepresentationDiagnosticError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise RepresentationDiagnosticError(f"{label} must be Boolean")
    return value


def _digest_list(
    value: Any,
    *,
    label: str,
    unique: bool,
    sorted_values: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise RepresentationDiagnosticError(f"{label} must be a list")
    output = [_digest(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    if unique and len(set(output)) != len(output):
        raise RepresentationDiagnosticError(f"{label} cannot contain duplicates")
    if sorted_values and output != sorted(output):
        raise RepresentationDiagnosticError(f"{label} must use canonical digest order")
    return output


def _identifier_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise RepresentationDiagnosticError(f"{label} must be a list")
    output = [_text(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    if len(set(output)) != len(output):
        raise RepresentationDiagnosticError(f"{label} cannot contain duplicates")
    return output


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "artifact_sha256" in payload:
        raise RepresentationDiagnosticError("unsealed payload cannot carry artifact_sha256")
    copied = dict(payload)
    try:
        digest = sha256_json(copied)
    except RepresentationError as exc:
        raise RepresentationDiagnosticError("artifact payload is not canonical JSON") from exc
    return {**copied, "artifact_sha256": digest}


def _validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    observed = _digest(value.get("artifact_sha256"), label=f"{label}.artifact_sha256")
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if observed != sha256_json(payload):
        raise RepresentationDiagnosticError(f"{label} artifact seal does not match its payload")


def _validate_observations(
    value: Any,
    *,
    cell: str,
    question_id: str,
) -> str:
    if not isinstance(value, list) or not value:
        raise RepresentationDiagnosticError(f"{cell} observations must be a non-empty list")
    expected_families = ["raw"] if cell == R0 else ["raw", "merged-sfk"]
    families: list[str] = []
    queries: set[str] = set()
    for index, raw in enumerate(value):
        observation = _mapping(raw, label=f"{cell}.observations[{index}]")
        family = _text(observation.get("family"), label=f"{cell}.observation.family")
        families.append(family)
        if observation.get("question_id") != question_id:
            raise RepresentationDiagnosticError(f"{cell} observation crosses question boundary")
        query = _digest(observation.get("query_sha256"), label=f"{cell}.query_sha256")
        queries.add(query)
        if observation.get("complete") is not True:
            raise RepresentationDiagnosticError(f"{cell} observation is incomplete")
        observed = _digest(
            observation.get("observation_sha256"),
            label=f"{cell}.observation_sha256",
        )
        payload = {key: item for key, item in observation.items() if key != "observation_sha256"}
        if observed != sha256_json(payload):
            raise RepresentationDiagnosticError(f"{cell} observation digest is inconsistent")
    if families != expected_families:
        raise RepresentationDiagnosticError(f"{cell} observation family order drifted")
    if len(queries) != 1:
        raise RepresentationDiagnosticError(f"{cell} observations do not share one query")
    return next(iter(queries))


def _validate_value_scores(
    trace: Mapping[str, Any],
    *,
    cell: str,
) -> tuple[list[str], list[str], dict[str, int]]:
    scores = trace.get("value_scores")
    if not isinstance(scores, list):
        raise RepresentationDiagnosticError(f"{cell}.value_scores must be a list")
    if trace.get("value_scores_sha256") != sha256_json(scores):
        raise RepresentationDiagnosticError(f"{cell}.value_scores digest is inconsistent")
    ids: list[str] = []
    hashes: list[str] = []
    bytes_by_id: dict[str, int] = {}
    for index, raw_score in enumerate(scores, start=1):
        score = _mapping(raw_score, label=f"{cell}.value_scores[{index - 1}]")
        if score.get("rank") != index:
            raise RepresentationDiagnosticError(f"{cell}.value_scores ranks are not contiguous")
        value = _mapping(score.get("value"), label=f"{cell}.value_scores[{index - 1}].value")
        value_id = _text(value.get("value_id"), label=f"{cell}.value_id")
        if value_id in bytes_by_id:
            raise RepresentationDiagnosticError(f"{cell}.value_scores repeats a value")
        raw_value = _mapping(value.get("raw_value"), label=f"{cell}.value.raw_value")
        raw_hash = _digest(raw_value.get("sha256"), label=f"{cell}.raw_value.sha256")
        raw_bytes = _integer(
            raw_value.get("utf8_bytes"),
            label=f"{cell}.raw_value.utf8_bytes",
            minimum=1,
        )
        ids.append(value_id)
        hashes.append(raw_hash)
        bytes_by_id[value_id] = raw_bytes
    hydrated_ids = _identifier_list(
        trace.get("hydrated_value_ids"), label=f"{cell}.hydrated_value_ids"
    )
    hydrated_hashes = _digest_list(
        trace.get("hydrated_raw_value_hashes"),
        label=f"{cell}.hydrated_raw_value_hashes",
        unique=False,
    )
    if ids != hydrated_ids or hashes != hydrated_hashes:
        raise RepresentationDiagnosticError(f"{cell} value scores and hydration disagree")
    if trace.get("hydrated_value_count") != len(ids):
        raise RepresentationDiagnosticError(f"{cell} hydrated value count is inconsistent")
    if trace.get("hydrated_value_cap") != MAX_HYDRATED_VALUES:
        raise RepresentationDiagnosticError(f"{cell} hydrated value cap drifted")
    pre_cap = _integer(
        trace.get("hydrated_value_pre_cap_count"),
        label=f"{cell}.hydrated_value_pre_cap_count",
    )
    if pre_cap < len(ids):
        raise RepresentationDiagnosticError(f"{cell} pre-cap count is smaller than hydration")
    return ids, hashes, bytes_by_id


def _validate_construction(
    value: Any,
    *,
    cell: str,
) -> tuple[dict[str, int], dict[str, int]]:
    accounting = _mapping(value, label=f"{cell}.construction_and_index_accounting")
    _exact_fields(accounting, _CONSTRUCTION_FIELDS, label=f"{cell}.construction accounting")
    if accounting.get("derived_objects_per_source_sha256") != sha256_json(
        accounting.get("derived_objects_per_source")
    ):
        raise RepresentationDiagnosticError(f"{cell} derived-object digest is inconsistent")
    construction = _mapping(
        accounting.get("construction_accounting"),
        label=f"{cell}.construction_accounting",
    )
    _exact_fields(
        construction,
        _CONSTRUCTION_ACCOUNTING_FIELDS,
        label=f"{cell}.construction_accounting",
    )
    normalized = {
        key: _integer(construction.get(key), label=f"{cell}.construction_accounting.{key}")
        for key in sorted(_CONSTRUCTION_ACCOUNTING_FIELDS)
    }
    receipt_count = _integer(
        accounting.get("construction_receipt_count"),
        label=f"{cell}.construction_receipt_count",
    )
    derived_count = _integer(
        accounting.get("derived_key_count"),
        label=f"{cell}.derived_key_count",
    )
    index = {
        "canonical_value_count": _integer(
            accounting.get("canonical_value_count"),
            label=f"{cell}.canonical_value_count",
            minimum=1,
        ),
        "canonical_value_utf8_bytes": _integer(
            accounting.get("canonical_value_utf8_bytes"),
            label=f"{cell}.canonical_value_utf8_bytes",
            minimum=1,
        ),
        "active_indexed_key_count": _integer(
            accounting.get("active_indexed_key_count"),
            label=f"{cell}.active_indexed_key_count",
            minimum=1,
        ),
        "active_indexed_key_utf8_bytes": _integer(
            accounting.get("active_indexed_key_utf8_bytes"),
            label=f"{cell}.active_indexed_key_utf8_bytes",
            minimum=1,
        ),
        "derived_key_count": derived_count,
        "derived_key_utf8_bytes": _integer(
            accounting.get("derived_key_utf8_bytes"),
            label=f"{cell}.derived_key_utf8_bytes",
        ),
    }
    if cell == R0:
        if receipt_count != 0 or derived_count != 0 or any(normalized.values()):
            raise RepresentationDiagnosticError("R0 must have zero derived construction accounting")
    elif receipt_count < 1 or derived_count < 1:
        raise RepresentationDiagnosticError("R1 must carry complete merged-key construction")
    return {"calls": receipt_count, **normalized}, index


def _validate_representation_artifact(
    value: Any,
    *,
    cell: str,
    question_id: str,
) -> dict[str, Any]:
    trace = _mapping(value, label=f"{cell}.representation")
    _exact_fields(trace, _REPRESENTATION_FIELDS, label=f"{cell}.representation")
    observed_trace = _digest(trace.get("trace_sha256"), label=f"{cell}.trace_sha256")
    trace_payload = {key: item for key, item in trace.items() if key != "trace_sha256"}
    if observed_trace != sha256_json(trace_payload):
        raise RepresentationDiagnosticError(f"{cell} representation trace seal is inconsistent")
    expected = {
        "artifact_type": REPRESENTATION_ARTIFACT_TYPE,
        "schema_version": REPRESENTATION_SCHEMA_VERSION,
        "protocol_version": REPRESENTATION_PROTOCOL_VERSION,
        "cell": cell,
        "classification": "benchmark-only-source-preserving-representation-control",
        "production_configuration": False,
        "paper_reproduction": False,
        "sb_hypothesis": "SB-HMR-v1",
        "graph": None,
    }
    for field, wanted in expected.items():
        if type(trace.get(field)) is not type(wanted) or trace.get(field) != wanted:
            raise RepresentationDiagnosticError(f"{cell}.representation.{field} drifted")
    claims = _mapping(trace.get("claims"), label=f"{cell}.claims")
    if claims.get("quality_improvement") is not False or claims.get("serving_change") is not False:
        raise RepresentationDiagnosticError(f"{cell} trace makes a forbidden quality claim")
    hydration = _mapping(trace.get("hydration"), label=f"{cell}.hydration")
    if hydration != {
        "reader_evidence": "canonical-raw-value",
        "derived_keys_delivered_to_reader": False,
        "source_values_byte_identical": True,
    }:
        raise RepresentationDiagnosticError(f"{cell} hydration is not source preserving")
    corpus = _mapping(trace.get("corpus"), label=f"{cell}.corpus")
    if corpus.get("question_id") != question_id:
        raise RepresentationDiagnosticError(f"{cell} corpus crosses the question boundary")
    corpus_binding = {
        "question_id": question_id,
        "source_artifact_sha256": _digest(
            corpus.get("source_artifact_sha256"), label=f"{cell}.source_artifact_sha256"
        ),
        "projection_sha256": _digest(
            corpus.get("projection_sha256"), label=f"{cell}.projection_sha256"
        ),
        "canonical_value_count": _integer(
            corpus.get("canonical_value_count"),
            label=f"{cell}.corpus.canonical_value_count",
            minimum=1,
        ),
        "canonical_value_order_sha256": _digest(
            corpus.get("canonical_value_order_sha256"),
            label=f"{cell}.canonical_value_order_sha256",
        ),
    }
    if corpus.get("complete_question_local_corpus_precedes_retrieval") is not True:
        raise RepresentationDiagnosticError(f"{cell} corpus is not complete before retrieval")
    observations = trace.get("observations")
    if trace.get("observations_sha256") != sha256_json(observations):
        raise RepresentationDiagnosticError(f"{cell} observation-list digest is inconsistent")
    query_sha256 = _validate_observations(
        observations,
        cell=cell,
        question_id=question_id,
    )
    candidate_ids, raw_hashes, bytes_by_id = _validate_value_scores(trace, cell=cell)
    construction, index = _validate_construction(
        trace.get("construction_and_index_accounting"),
        cell=cell,
    )
    if index["canonical_value_count"] != corpus_binding["canonical_value_count"]:
        raise RepresentationDiagnosticError(f"{cell} corpus and index value counts disagree")
    return {
        "trace_sha256": observed_trace,
        "corpus": corpus_binding,
        "query_sha256": query_sha256,
        "candidate_value_ids": candidate_ids,
        "candidate_raw_value_hashes": raw_hashes,
        "bytes_by_value_id": bytes_by_id,
        "construction": construction,
        "index": index,
    }


def _validate_query_accounting(
    value: Any, *, cell: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accounting = _mapping(value, label=f"{cell}.query_accounting")
    _exact_fields(accounting, _QUERY_ACCOUNTING_FIELDS, label=f"{cell}.query_accounting")
    if accounting.get("complete") is not True:
        raise RepresentationDiagnosticError(f"{cell} query accounting is incomplete")
    if accounting.get("source") != "externally-attested-unverified":
        raise RepresentationDiagnosticError(f"{cell} query accounting source drifted")
    stages_raw = accounting.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise RepresentationDiagnosticError(f"{cell} query accounting requires stages")
    stages: list[dict[str, Any]] = []
    names: list[str] = []
    for index, raw_stage in enumerate(stages_raw):
        stage = _mapping(raw_stage, label=f"{cell}.query.stages[{index}]")
        _exact_fields(stage, _STAGE_FIELDS, label=f"{cell}.query.stages[{index}]")
        name = _text(stage.get("name"), label=f"{cell}.query.stages[{index}].name")
        names.append(name)
        normalized = {"name": name}
        normalized.update(
            {
                field: _integer(
                    stage.get(field),
                    label=f"{cell}.query.stages[{index}].{field}",
                )
                for field in sorted(_STAGE_FIELDS - {"name"})
            }
        )
        if normalized["cache_hits"] > normalized["calls"]:
            raise RepresentationDiagnosticError(f"{cell} query cache hits exceed calls")
        stages.append(normalized)
    if names != sorted(names) or len(set(names)) != len(names):
        raise RepresentationDiagnosticError(f"{cell} query stages must be unique and sorted")
    if sum(stage["calls"] for stage in stages) < 1:
        raise RepresentationDiagnosticError(f"{cell} query accounting records no calls")
    if accounting.get("stages_sha256") != sha256_json(stages):
        raise RepresentationDiagnosticError(f"{cell} query-stage digest is inconsistent")
    totals = {
        field: sum(int(stage[field]) for stage in stages)
        for field in sorted(_STAGE_FIELDS - {"name"})
    }
    return stages, totals


def _context_metrics(
    ordered_sessions: Sequence[str], gold_sessions: Sequence[str]
) -> dict[str, Any]:
    gold = set(gold_sessions)
    hit_sessions = set(ordered_sessions) & gold
    first = next(
        (rank for rank, session in enumerate(ordered_sessions, start=1) if session in gold),
        None,
    )
    return {
        "any_gold_in_context": bool(hit_sessions),
        "all_gold_in_context": bool(gold) and hit_sessions == gold,
        "answer_session_recall": len(hit_sessions) / len(gold) if gold else None,
        "answer_session_mrr": 1.0 / first if first is not None else (0.0 if gold else None),
    }


def _validate_context(
    value: Any,
    *,
    cell: str,
    representation: Mapping[str, Any],
    gold_sessions: Sequence[str],
) -> dict[str, Any]:
    context = _mapping(value, label=f"{cell}.context")
    _exact_fields(context, _CONTEXT_FIELDS, label=f"{cell}.context")
    candidate_sessions = _digest_list(
        context.get("candidate_session_sha256s"),
        label=f"{cell}.candidate_session_sha256s",
        unique=False,
    )
    candidate_ids = list(representation["candidate_value_ids"])
    if len(candidate_sessions) != len(candidate_ids):
        raise RepresentationDiagnosticError(
            f"{cell} candidate session bindings do not cover the actual candidate order"
        )
    prompt_ids = _identifier_list(context.get("prompt_value_ids"), label=f"{cell}.prompt_value_ids")
    if not set(prompt_ids).issubset(candidate_ids):
        raise RepresentationDiagnosticError(f"{cell} prompt contains a non-candidate value")
    session_by_id = dict(zip(candidate_ids, candidate_sessions, strict=True))
    prompt_sessions = [session_by_id[value_id] for value_id in prompt_ids]
    prompt_tokens = _integer(
        context.get("prompt_tokens"),
        label=f"{cell}.prompt_tokens",
        minimum=1,
    )
    if prompt_tokens > PROMPT_BUDGET_TOKENS:
        raise RepresentationDiagnosticError(f"{cell} prompt exceeds the frozen token budget")
    prompt_sha256 = _digest(context.get("prompt_sha256"), label=f"{cell}.prompt_sha256")
    tokenizer_artifact_sha256 = _digest(
        context.get("tokenizer_artifact_sha256"),
        label=f"{cell}.tokenizer_artifact_sha256",
    )
    tokenizer_receipt_sha256 = _digest(
        context.get("tokenizer_receipt_sha256"),
        label=f"{cell}.tokenizer_receipt_sha256",
    )
    bytes_by_id = representation["bytes_by_value_id"]
    candidate_bytes = sum(int(bytes_by_id[value_id]) for value_id in candidate_ids)
    prompt_bytes = sum(int(bytes_by_id[value_id]) for value_id in prompt_ids)
    return {
        "candidate_value_count": len(candidate_ids),
        "candidate_context_utf8_bytes": candidate_bytes,
        "candidate": _context_metrics(candidate_sessions, gold_sessions),
        "prompt_value_count": len(prompt_ids),
        "prompt_context_utf8_bytes": prompt_bytes,
        "prompt_tokens": prompt_tokens,
        "prompt_sha256": prompt_sha256,
        "tokenizer_artifact_sha256": tokenizer_artifact_sha256,
        "tokenizer_receipt_sha256": tokenizer_receipt_sha256,
        "prompt": _context_metrics(prompt_sessions, gold_sessions),
    }


def _validate_qa(value: Any, *, cell: str) -> dict[str, Any] | None:
    if value is None:
        return None
    qa = _mapping(value, label=f"{cell}.qa")
    _exact_fields(qa, _QA_FIELDS, label=f"{cell}.qa")
    return {
        "correct": _boolean(qa.get("correct"), label=f"{cell}.qa.correct"),
        "reader_receipt_sha256": _digest(
            qa.get("reader_receipt_sha256"), label=f"{cell}.qa.reader_receipt_sha256"
        ),
        "judge_receipt_sha256": _digest(
            qa.get("judge_receipt_sha256"), label=f"{cell}.qa.judge_receipt_sha256"
        ),
    }


def _combined_accounting(
    construction: Mapping[str, int],
    query: Mapping[str, int],
) -> dict[str, int]:
    return {
        field: int(construction[field]) + int(query[field])
        for field in (
            "calls",
            "input_tokens",
            "output_tokens",
            "latency_microseconds",
            "cost_microusd",
            "retry_count",
            "cache_hits",
        )
    }


@dataclass(frozen=True, slots=True)
class _CaseEvidence:
    case_index: int
    question_id: str
    question_type: str
    gold_sessions: tuple[str, ...]
    arms: dict[str, dict[str, Any]]
    artifact_sha256: str
    file_bytes: int
    file_sha256: str


def _validate_case(value: Any, *, file_bytes: int = 0, file_sha256: str = "") -> _CaseEvidence:
    case = _mapping(value, label="case")
    _exact_fields(case, _CASE_FIELDS, label="case")
    _validate_seal(case, label="case")
    expected = {
        "schema_version": CASE_SCHEMA_VERSION,
        "artifact_type": CASE_ARTIFACT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
    }
    for field, wanted in expected.items():
        if type(case.get(field)) is not type(wanted) or case.get(field) != wanted:
            raise RepresentationDiagnosticError(f"case.{field} must be {wanted!r}")
    case_index = _integer(case.get("case_index"), label="case.case_index")
    question_id = _text(case.get("question_id"), label="case.question_id")
    question_type = _text(case.get("question_type"), label="case.question_type")
    gold_sessions = _digest_list(
        case.get("gold_session_sha256s"),
        label="case.gold_session_sha256s",
        unique=True,
        sorted_values=True,
    )
    raw_arms = _mapping(case.get("arms"), label="case.arms")
    if set(raw_arms) != set(ARMS):
        raise RepresentationDiagnosticError("case must contain exactly paired R0 and R1 arms")
    arms: dict[str, dict[str, Any]] = {}
    for cell in ARMS:
        arm = _mapping(raw_arms.get(cell), label=f"case.arms.{cell}")
        _exact_fields(arm, _ARM_FIELDS, label=f"case.arms.{cell}")
        if arm.get("cell") != cell:
            raise RepresentationDiagnosticError(f"case.arms.{cell}.cell drifted")
        representation = _validate_representation_artifact(
            arm.get("representation"),
            cell=cell,
            question_id=question_id,
        )
        stages, query = _validate_query_accounting(arm.get("query_accounting"), cell=cell)
        context = _validate_context(
            arm.get("context"),
            cell=cell,
            representation=representation,
            gold_sessions=gold_sessions,
        )
        qa = _validate_qa(arm.get("qa"), cell=cell)
        arms[cell] = {
            "representation": representation,
            "context": context,
            "query_stages": stages,
            "accounting": {
                "construction": representation["construction"],
                "query": query,
                "construction_plus_query": _combined_accounting(
                    representation["construction"], query
                ),
            },
            "qa": qa,
        }
    left = arms[R0]["representation"]
    right = arms[R1]["representation"]
    if left["corpus"] != right["corpus"]:
        raise RepresentationDiagnosticError("R0 and R1 do not share the canonical source corpus")
    if left["query_sha256"] != right["query_sha256"]:
        raise RepresentationDiagnosticError("R0 and R1 do not bind the same query")
    if (
        arms[R0]["context"]["tokenizer_artifact_sha256"]
        != arms[R1]["context"]["tokenizer_artifact_sha256"]
    ):
        raise RepresentationDiagnosticError("R0 and R1 do not share the tokenizer artifact")
    if (arms[R0]["qa"] is None) != (arms[R1]["qa"] is None):
        raise RepresentationDiagnosticError("optional QA must be present for both arms or neither")
    return _CaseEvidence(
        case_index=case_index,
        question_id=question_id,
        question_type=question_type,
        gold_sessions=tuple(gold_sessions),
        arms=arms,
        artifact_sha256=str(case["artifact_sha256"]),
        file_bytes=file_bytes,
        file_sha256=file_sha256,
    )


def seal_case_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal and validate one producer-side diagnostic case payload."""

    sealed = _seal(payload)
    _validate_case(sealed)
    return sealed


def validate_sealed_case_input(value: Any) -> dict[str, Any]:
    """Validate one already-parsed sealed case and return a canonical copy."""

    _validate_case(value)
    try:
        return json.loads(canonical_json_bytes(value))
    except (RepresentationError, json.JSONDecodeError) as exc:  # pragma: no cover - validated above
        raise RepresentationDiagnosticError("case cannot be canonically copied") from exc


def load_sealed_case_input(path: str | Path) -> dict[str, Any]:
    """Reopen one regular, non-symlink case artifact and validate its seal."""

    supplied = Path(path)
    if supplied.is_symlink():
        raise RepresentationDiagnosticError("case path cannot be a symbolic link")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RepresentationDiagnosticError("case artifact is missing") from exc
    if not resolved.is_file():
        raise RepresentationDiagnosticError("case artifact must be a regular file")
    raw = resolved.read_bytes()
    return validate_sealed_case_input(_strict_json_object(raw, label="case artifact"))


def _load_case_evidence(path: str | Path) -> _CaseEvidence:
    supplied = Path(path)
    if supplied.is_symlink():
        raise RepresentationDiagnosticError("case path cannot be a symbolic link")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RepresentationDiagnosticError("case artifact is missing") from exc
    if not resolved.is_file():
        raise RepresentationDiagnosticError("case artifact must be a regular file")
    raw = resolved.read_bytes()
    value = _strict_json_object(raw, label="case artifact")
    return _validate_case(value, file_bytes=len(raw), file_sha256=sha256_bytes(raw))


def _percentile(values: Sequence[int | float], quantile: float) -> float:
    if not values:
        raise RepresentationDiagnosticError("cannot compute a percentile over no values")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise RepresentationDiagnosticError("percentile values must be finite")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _numeric_summary(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        raise RepresentationDiagnosticError("cannot summarize no values")
    total: int | float
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        total = sum(int(value) for value in values)
    else:
        total = sum(float(value) for value in values)
    return {
        "count": len(values),
        "total": total,
        "mean": float(total) / len(values),
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _case_quality(arm: Mapping[str, Any]) -> dict[str, float]:
    context = arm["context"]
    candidate = context["candidate"]
    prompt = context["prompt"]
    return {
        "candidate_any_gold_in_context": float(candidate["any_gold_in_context"]),
        "candidate_all_gold_in_context": float(candidate["all_gold_in_context"]),
        "candidate_answer_session_recall": float(candidate["answer_session_recall"]),
        "candidate_answer_session_mrr": float(candidate["answer_session_mrr"]),
        "prompt_any_gold_in_context": float(prompt["any_gold_in_context"]),
        "prompt_all_gold_in_context": float(prompt["all_gold_in_context"]),
        "prompt_answer_session_recall": float(prompt["answer_session_recall"]),
        "prompt_answer_session_mrr": float(prompt["answer_session_mrr"]),
    }


def _quality_summary(cases: Sequence[_CaseEvidence]) -> dict[str, Any]:
    eligible = [case for case in cases if case.gold_sessions]
    if not eligible:
        return {
            "available": False,
            "gold_eligible_cases": 0,
            "reason": "no case carries a non-empty gold-session binding",
        }
    arms: dict[str, dict[str, float]] = {}
    for cell in ARMS:
        arms[cell] = {
            axis: sum(_case_quality(case.arms[cell])[axis] for case in eligible) / len(eligible)
            for axis in _QUALITY_AXES
        }
    return {
        "available": True,
        "gold_eligible_cases": len(eligible),
        "axes": {axis: "higher-is-better" for axis in _QUALITY_AXES},
        "arms": arms,
        "R1_minus_R0": {axis: arms[R1][axis] - arms[R0][axis] for axis in _QUALITY_AXES},
    }


def _sum_accounting(cases: Sequence[_CaseEvidence], *, cell: str, scope: str) -> dict[str, int]:
    fields = (
        "calls",
        "input_tokens",
        "output_tokens",
        "latency_microseconds",
        "cost_microusd",
        "retry_count",
        "cache_hits",
    )
    return {
        field: sum(int(case.arms[cell]["accounting"][scope][field]) for case in cases)
        for field in fields
    }


def _efficiency_summary(cases: Sequence[_CaseEvidence]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for cell in ARMS:
        prompt_tokens = [int(case.arms[cell]["context"]["prompt_tokens"]) for case in cases]
        candidate_bytes = [
            int(case.arms[cell]["context"]["candidate_context_utf8_bytes"]) for case in cases
        ]
        prompt_bytes = [
            int(case.arms[cell]["context"]["prompt_context_utf8_bytes"]) for case in cases
        ]
        combined = [case.arms[cell]["accounting"]["construction_plus_query"] for case in cases]
        index = [case.arms[cell]["representation"]["index"] for case in cases]
        arms[cell] = {
            "prompt_tokens": _numeric_summary(prompt_tokens),
            "candidate_context_utf8_bytes": _numeric_summary(candidate_bytes),
            "prompt_context_utf8_bytes": _numeric_summary(prompt_bytes),
            "operational_latency_microseconds": _numeric_summary(
                [int(row["latency_microseconds"]) for row in combined]
            ),
            "construction_plus_query_cost_microusd": _numeric_summary(
                [int(row["cost_microusd"]) for row in combined]
            ),
            "construction_accounting_totals": _sum_accounting(
                cases, cell=cell, scope="construction"
            ),
            "query_accounting_totals": _sum_accounting(cases, cell=cell, scope="query"),
            "construction_plus_query_accounting_totals": _sum_accounting(
                cases, cell=cell, scope="construction_plus_query"
            ),
            "index": {
                field: _numeric_summary([int(row[field]) for row in index])
                for field in (
                    "canonical_value_count",
                    "canonical_value_utf8_bytes",
                    "active_indexed_key_count",
                    "active_indexed_key_utf8_bytes",
                    "derived_key_count",
                    "derived_key_utf8_bytes",
                )
            },
        }
    return {
        "arms": arms,
        "R1_minus_R0": {
            "prompt_tokens": _numeric_summary(
                [
                    int(case.arms[R1]["context"]["prompt_tokens"])
                    - int(case.arms[R0]["context"]["prompt_tokens"])
                    for case in cases
                ]
            ),
            "operational_latency_microseconds": _numeric_summary(
                [
                    int(
                        case.arms[R1]["accounting"]["construction_plus_query"][
                            "latency_microseconds"
                        ]
                    )
                    - int(
                        case.arms[R0]["accounting"]["construction_plus_query"][
                            "latency_microseconds"
                        ]
                    )
                    for case in cases
                ]
            ),
            "construction_plus_query_cost_microusd": _numeric_summary(
                [
                    int(case.arms[R1]["accounting"]["construction_plus_query"]["cost_microusd"])
                    - int(case.arms[R0]["accounting"]["construction_plus_query"]["cost_microusd"])
                    for case in cases
                ]
            ),
        },
    }


def _qa_summary(cases: Sequence[_CaseEvidence]) -> dict[str, Any]:
    paired = [case for case in cases if case.arms[R0]["qa"] is not None]
    if not paired:
        return {
            "available": False,
            "paired_cases": 0,
            "complete_case_coverage": False,
            "promotion_use": "forbidden",
        }
    correct = {
        cell: sum(int(bool(case.arms[cell]["qa"]["correct"])) for case in paired) for cell in ARMS
    }
    return {
        "available": True,
        "paired_cases": len(paired),
        "complete_case_coverage": len(paired) == len(cases),
        "arms": {
            cell: {
                "correct": correct[cell],
                "accuracy": correct[cell] / len(paired),
            }
            for cell in ARMS
        },
        "paired_accuracy_delta_R1_minus_R0": (correct[R1] - correct[R0]) / len(paired),
        "promotion_use": "forbidden-development-diagnostic-only",
    }


def _r0_dominates(
    quality: Mapping[str, Any],
    efficiency: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    if quality.get("available") is not True:
        return False, {"quality_comparable": False, "strict_advantages": []}
    r0_quality = quality["arms"][R0]
    r1_quality = quality["arms"][R1]
    quality_comparisons = {axis: r0_quality[axis] >= r1_quality[axis] for axis in _QUALITY_AXES}
    r0_efficiency = efficiency["arms"][R0]
    r1_efficiency = efficiency["arms"][R1]
    efficiency_values = {
        "p95_prompt_tokens": (
            float(r0_efficiency["prompt_tokens"]["p95"]),
            float(r1_efficiency["prompt_tokens"]["p95"]),
        ),
        "p95_operational_latency_microseconds": (
            float(r0_efficiency["operational_latency_microseconds"]["p95"]),
            float(r1_efficiency["operational_latency_microseconds"]["p95"]),
        ),
        "total_construction_plus_query_cost_microusd": (
            float(r0_efficiency["construction_plus_query_cost_microusd"]["total"]),
            float(r1_efficiency["construction_plus_query_cost_microusd"]["total"]),
        ),
    }
    efficiency_comparisons = {
        axis: left <= right for axis, (left, right) in efficiency_values.items()
    }
    strict = [axis for axis in _QUALITY_AXES if r0_quality[axis] > r1_quality[axis]]
    strict.extend(axis for axis, (left, right) in efficiency_values.items() if left < right)
    dominates = (
        all(quality_comparisons.values()) and all(efficiency_comparisons.values()) and bool(strict)
    )
    return dominates, {
        "quality_comparable": True,
        "quality_R0_at_least_R1": quality_comparisons,
        "efficiency_R0_no_worse_than_R1": efficiency_comparisons,
        "strict_advantages": strict,
    }


def compile_r0_r1_diagnostic(case_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Reopen sealed paired cases and compile the non-promotional E6 diagnostic."""

    if isinstance(case_paths, (str, bytes, Path)) or not isinstance(case_paths, Sequence):
        raise RepresentationDiagnosticError("case_paths must be a sequence of artifact paths")
    if not case_paths:
        raise RepresentationDiagnosticError("at least one sealed case artifact is required")
    cases = sorted(
        (_load_case_evidence(path) for path in case_paths), key=lambda case: case.case_index
    )
    indexes = [case.case_index for case in cases]
    if indexes != list(range(len(cases))):
        raise RepresentationDiagnosticError("case indexes must be unique and contiguous from zero")
    question_ids = [case.question_id for case in cases]
    if len(set(question_ids)) != len(question_ids):
        raise RepresentationDiagnosticError("diagnostic repeats a question ID")
    sources = {
        case.arms[R0]["representation"]["corpus"]["source_artifact_sha256"] for case in cases
    }
    projections = {case.arms[R0]["representation"]["corpus"]["projection_sha256"] for case in cases}
    tokenizers = {
        case.arms[cell]["context"]["tokenizer_artifact_sha256"] for case in cases for cell in ARMS
    }
    if len(sources) != 1 or len(projections) != 1:
        raise RepresentationDiagnosticError("cases do not share one dataset/projection binding")
    if len(tokenizers) != 1:
        raise RepresentationDiagnosticError("cases do not share one tokenizer artifact")

    quality = _quality_summary(cases)
    efficiency = _efficiency_summary(cases)
    qa = _qa_summary(cases)
    gold_noninferior = bool(
        quality.get("available") is True
        and all(
            float(quality["arms"][R1][axis]) >= float(quality["arms"][R0][axis])
            for axis in _QUALITY_AXES
        )
    )
    r0_dominates, dominance_evidence = _r0_dominates(quality, efficiency)
    reasons: list[str] = []
    if not gold_noninferior:
        reasons.append("R1-fails-zero-margin-gold-context-noninferiority")
    if r0_dominates:
        reasons.append("R0-pareto-dominates-R1")
    early_stop = bool(reasons)

    case_rows: list[dict[str, Any]] = []
    source_cases: list[dict[str, Any]] = []
    for case in cases:
        source_cases.append(
            {
                "case_index": case.case_index,
                "question_id": case.question_id,
                "artifact_sha256": case.artifact_sha256,
                "file_bytes": case.file_bytes,
                "file_sha256": case.file_sha256,
            }
        )
        row_arms: dict[str, Any] = {}
        for cell in ARMS:
            arm = case.arms[cell]
            row_arms[cell] = {
                "representation_trace_sha256": arm["representation"]["trace_sha256"],
                "query_sha256": arm["representation"]["query_sha256"],
                "context": arm["context"],
                "accounting": arm["accounting"],
                "query_stages": arm["query_stages"],
                "qa_correct": None if arm["qa"] is None else arm["qa"]["correct"],
            }
        case_rows.append(
            {
                "case_index": case.case_index,
                "question_id": case.question_id,
                "question_type": case.question_type,
                "gold_eligible": bool(case.gold_sessions),
                "gold_session_count": len(case.gold_sessions),
                "arms": row_arms,
            }
        )

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "classification": "non-promotional-development-diagnostic",
        "representation_protocol_version": REPRESENTATION_PROTOCOL_VERSION,
        "arms": list(ARMS),
        "case_count": len(cases),
        "source": {
            "dataset_artifact_sha256": next(iter(sources)),
            "projection_sha256": next(iter(projections)),
            "tokenizer_artifact_sha256": next(iter(tokenizers)),
            "sealed_cases": source_cases,
            "sealed_cases_sha256": sha256_json(source_cases),
            "all_case_files_reopened": True,
            "all_case_object_seals_validated": True,
        },
        "context_quality": quality,
        "efficiency": efficiency,
        "qa": qa,
        "cases": case_rows,
        "early_stop": {
            "triggered": early_stop,
            "reasons": reasons,
            "gold_noninferiority": {
                "passed": gold_noninferior,
                "margin": 0.0,
                "requires_nonzero_shared_gold_denominator": True,
                "axes": list(_QUALITY_AXES),
            },
            "R0_pareto_dominates_R1": {
                "value": r0_dominates,
                "quality_axes": list(_QUALITY_AXES),
                "efficiency_axes": [
                    "p95_prompt_tokens:lower",
                    "p95_operational_latency_microseconds:lower",
                    "total_construction_plus_query_cost_microusd:lower",
                ],
                "evidence": dominance_evidence,
            },
            "continue_to_optional_qa": not early_stop,
        },
        "decision": {
            "verdict": (
                "early-stop-R1-development"
                if early_stop
                else "continue-development-diagnostic-only"
            ),
            "eligible_for_composition": False,
            "eligible_for_serving_promotion": False,
            "eligible_for_official_score": False,
            "qa_can_override_early_stop": False,
        },
        "claims": {
            "official_longmemeval_score": False,
            "paper_reproduction": False,
            "quality_improvement_proven": False,
            "causal_improvement_proven": False,
            "serving_change_authorized": False,
            "production_policy_changed": False,
            "external_execution_identity_authenticated": False,
        },
    }
    return _seal(payload)


def compile_representation_diagnostic(case_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Compatibility spelling for the R0-versus-R1 diagnostic compiler."""

    return compile_r0_r1_diagnostic(case_paths)


__all__ = [
    "ARMS",
    "CASE_ARTIFACT_TYPE",
    "CASE_SCHEMA_VERSION",
    "PROMPT_BUDGET_TOKENS",
    "PROTOCOL_VERSION",
    "R0",
    "R1",
    "REPORT_ARTIFACT_TYPE",
    "REPORT_SCHEMA_VERSION",
    "RepresentationDiagnosticError",
    "compile_r0_r1_diagnostic",
    "compile_representation_diagnostic",
    "load_sealed_case_input",
    "seal_case_input",
    "validate_sealed_case_input",
]
