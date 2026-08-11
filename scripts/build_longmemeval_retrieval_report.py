#!/usr/bin/env python3
"""Rebuild the canonical LongMemEval-S retrieval report entirely offline.

The compiler accepts one repository-local schema-v2 retrieval run, validates
its protocol, publishability envelope, current-tree fingerprint, case records,
and byte identity, reconstructs every exact reader prompt from the run's compact
public source material, then delegates every metric to ``run_retrieval_eval.py``.
It never loads the dataset and never calls a model, database, or endpoint.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_retrieval_eval as retrieval_eval
from _longmemeval_common import build_official_reader_prompt
from _longmemeval_tokenizer import TOKENIZER_PROTOCOL, tokenizer_response_identity_sha256
from run_longmemeval_qa import (
    retrieval_publishability_errors,
    validate_retrieval_run_protocol,
)

EXPECTED_QUESTION_COUNT = 500
EXPECTED_ABSTENTION_QUESTIONS = 30
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PROVIDER_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,256}")


class RetrievalReportError(ValueError):
    """The saved retrieval run cannot support a canonical offline report."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RetrievalReportError(f"duplicate JSON field {key!r} is forbidden")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise RetrievalReportError(f"non-finite JSON number {value!r} is forbidden")


def _repo_input(path: Path, *, repo_root: Path) -> tuple[Path, str]:
    if path.is_absolute() or ".." in path.parts:
        raise RetrievalReportError("--run must be a repository-local relative path")
    resolved_root = repo_root.resolve()
    current = resolved_root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise RetrievalReportError("--run cannot traverse symbolic links")
    try:
        resolved = (resolved_root / path).resolve(strict=True)
    except OSError as exc:
        raise RetrievalReportError(f"--run is missing: {path}") from exc
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise RetrievalReportError("--run must resolve to a regular file inside the repository")
    return resolved, resolved.relative_to(resolved_root).as_posix()


def _strict_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, RetrievalReportError) as exc:
        raise RetrievalReportError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RetrievalReportError(f"{label} must contain one JSON object")
    return payload


def _finite_number(value: Any, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrievalReportError(f"{label} must be a finite number >= {minimum}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum:
        raise RetrievalReportError(f"{label} must be a finite number >= {minimum}")
    return numeric


def _string_list(value: Any, *, label: str, maximum: int | None = None) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise RetrievalReportError(f"{label} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise RetrievalReportError(f"{label} contains duplicate identifiers")
    if maximum is not None and len(value) > maximum:
        raise RetrievalReportError(f"{label} exceeds its saved depth {maximum}")
    return value


def _validate_cases(payload: dict[str, Any]) -> None:
    cases = payload["cases"]
    recall_limit = payload["recall_limit"]
    saved_depth = payload["saved_ranking_depth"]
    seen: set[str] = set()
    abstention_count = 0
    for index, case in enumerate(cases):
        label = f"case[{index}]"
        if not isinstance(case, dict):
            raise RetrievalReportError(f"{label} must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise RetrievalReportError(f"{label} has a missing or duplicate case_id")
        seen.add(case_id)
        abstention = case.get("abstention_question")
        if not isinstance(abstention, bool) or abstention is not case_id.endswith("_abs"):
            raise RetrievalReportError(f"{label} has inconsistent abstention metadata")
        abstention_count += int(abstention)
        if not isinstance(case.get("category"), str) or not case["category"]:
            raise RetrievalReportError(f"{label} category must be a non-empty string")
        haystack_sessions = case.get("haystack_sessions")
        if isinstance(haystack_sessions, bool) or not isinstance(haystack_sessions, int):
            raise RetrievalReportError(f"{label} haystack_sessions must be a positive integer")
        if haystack_sessions < 1:
            raise RetrievalReportError(f"{label} haystack_sessions must be a positive integer")
        _string_list(case.get("relevant_ids"), label=f"{label}.relevant_ids")
        degraded = _string_list(case.get("degraded_lanes"), label=f"{label}.degraded_lanes")
        if degraded:
            raise RetrievalReportError(f"{label} contains degraded retrieval lanes")
        rankings = case.get("rankings")
        if not isinstance(rankings, dict) or "final" not in rankings:
            raise RetrievalReportError(f"{label}.rankings must contain the final lane")
        for lane, ranking in rankings.items():
            if not isinstance(lane, str) or not lane:
                raise RetrievalReportError(f"{label}.rankings has an invalid lane name")
            _string_list(ranking, label=f"{label}.rankings.{lane}", maximum=saved_depth)
        final = rankings["final"]
        if len(final) > recall_limit:
            raise RetrievalReportError(f"{label}.rankings.final exceeds recall_limit")
        relevance = case.get("final_relevance")
        if not isinstance(relevance, list) or len(relevance) != len(final):
            raise RetrievalReportError(f"{label}.final_relevance must align with final ranking")
        for value in relevance:
            if _finite_number(value, label=f"{label}.final_relevance") > 1.0:
                raise RetrievalReportError(f"{label}.final_relevance must be <= 1")
        tokens = case.get("final_tokens")
        if not isinstance(tokens, list) or len(tokens) != len(final):
            raise RetrievalReportError(f"{label}.final_tokens must align with final ranking")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in tokens
        ):
            raise RetrievalReportError(f"{label}.final_tokens must contain positive integers")
        _finite_number(case.get("wall_ms"), label=f"{label}.wall_ms")
        lanes = case.get("lane_latency_ms")
        if not isinstance(lanes, dict):
            raise RetrievalReportError(f"{label}.lane_latency_ms must be an object")
        for lane, value in lanes.items():
            if not isinstance(lane, str) or not lane:
                raise RetrievalReportError(f"{label}.lane_latency_ms has an invalid lane")
            _finite_number(value, label=f"{label}.lane_latency_ms.{lane}")
        if case.get("temporal_routing") is not None:
            raise RetrievalReportError(f"{label} unexpectedly contains a temporal routing trace")
    if len(cases) != EXPECTED_QUESTION_COUNT:
        raise RetrievalReportError("canonical retrieval evidence must contain all 500 questions")
    if abstention_count != EXPECTED_ABSTENTION_QUESTIONS:
        raise RetrievalReportError(
            "canonical retrieval evidence must contain exactly 30 abstentions"
        )


def _strict_positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RetrievalReportError(f"{label} must be a positive integer")
    return value


def _bound_file_identity(
    value: Any,
    *,
    label: str,
    repo_root: Path,
    executable: bool = False,
) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise RetrievalReportError(f"{label} must bind exactly path, bytes, and sha256")
    path = value.get("path")
    if not isinstance(path, str) or not path:
        raise RetrievalReportError(f"{label}.path must be non-empty")
    resolved, relative = _repo_input(Path(path), repo_root=repo_root)
    if relative != path:
        raise RetrievalReportError(f"{label}.path must be canonical repository-relative POSIX")
    raw = resolved.read_bytes()
    expected_bytes = value.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise RetrievalReportError(f"{label}.bytes must be an integer")
    if expected_bytes != len(raw) or value.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise RetrievalReportError(f"{label} byte identity does not match its repository file")
    if executable and not os.access(resolved, os.X_OK):
        raise RetrievalReportError(f"{label} is not executable")


def _observation(
    value: Any,
    *,
    label: str,
    expected_prompt: str,
    state_key: tuple[int, tuple[str, ...]],
    observations: dict[int, dict[str, Any]],
    observation_states: dict[int, tuple[int, tuple[str, ...]]],
    state_observations: dict[tuple[int, tuple[str, ...]], dict[str, Any]],
    observation_text_sizes: dict[int, tuple[int, int]],
    provider_ids: set[str],
    response_identity_sha256: str,
) -> dict[str, Any]:
    fields = {
        "request_id",
        "provider_request_id",
        "response_identity_sha256",
        "text_sha256",
        "token_count",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise RetrievalReportError(f"{label} must carry one exact tokenizer observation")
    request_id = _strict_positive_integer(value.get("request_id"), label=f"{label}.request_id")
    provider_id = value.get("provider_request_id")
    if not isinstance(provider_id, str) or _PROVIDER_REQUEST_ID_RE.fullmatch(provider_id) is None:
        raise RetrievalReportError(f"{label}.provider_request_id is invalid")
    digest = value.get("text_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise RetrievalReportError(f"{label}.text_sha256 is invalid")
    encoded_prompt = expected_prompt.encode("utf-8")
    if digest != hashlib.sha256(encoded_prompt).hexdigest():
        raise RetrievalReportError(
            f"{label}.text_sha256 does not match the reconstructed official reader prompt"
        )
    _strict_positive_integer(value.get("token_count"), label=f"{label}.token_count")
    if value.get("response_identity_sha256") != response_identity_sha256:
        raise RetrievalReportError(f"{label}.response_identity_sha256 is invalid")
    state_observation = state_observations.get(state_key)
    if state_observation is not None and state_observation != value:
        raise RetrievalReportError(
            f"{label} records inconsistent observations for one prompt state"
        )
    existing = observations.get(request_id)
    if existing is not None and existing != value:
        raise RetrievalReportError(f"{label} reuses a request ID with different evidence")
    if existing is not None and observation_states[request_id] != state_key:
        raise RetrievalReportError(f"{label} reuses a request ID for a different prompt state")
    if existing is None:
        if request_id != len(observations) + 1:
            raise RetrievalReportError(f"{label} request ID is out of execution order")
        if provider_id in provider_ids:
            raise RetrievalReportError(f"{label} reuses a provider request ID")
        observations[request_id] = value
        observation_states[request_id] = state_key
        observation_text_sizes[request_id] = (len(expected_prompt), len(encoded_prompt))
        provider_ids.add(provider_id)
    state_observations.setdefault(state_key, value)
    return value


def _exact_prompt_material(
    case: dict[str, Any],
    *,
    case_index: int,
) -> tuple[dict[str, str], dict[str, tuple[str, tuple[dict[str, str], ...]]]]:
    """Validate the compact source material used to reconstruct exact prompts."""

    label = f"case[{case_index}].exact_context_material"
    material = case.get("exact_context_material")
    fields = {"question", "question_date", "ranked_sessions"}
    if not isinstance(material, dict) or set(material) != fields:
        raise RetrievalReportError(f"{label} must carry exact prompt source material")
    question = material.get("question")
    question_date = material.get("question_date")
    if not isinstance(question, str) or not question:
        raise RetrievalReportError(f"{label}.question must be non-empty text")
    if not isinstance(question_date, str) or not question_date:
        raise RetrievalReportError(f"{label}.question_date must be non-empty text")
    ranked_sessions = material.get("ranked_sessions")
    if not isinstance(ranked_sessions, list):
        raise RetrievalReportError(f"{label}.ranked_sessions must be a list")

    expected_ids = case["rankings"]["final"][:10]
    if len(ranked_sessions) != len(expected_ids):
        raise RetrievalReportError(f"{label}.ranked_sessions must cover the final top 10")
    sessions: dict[str, tuple[str, tuple[dict[str, str], ...]]] = {}
    observed_ids: list[str] = []
    for session_index, raw_session in enumerate(ranked_sessions):
        session_label = f"{label}.ranked_sessions[{session_index}]"
        if not isinstance(raw_session, dict) or set(raw_session) != {
            "session_id",
            "date",
            "turns",
        }:
            raise RetrievalReportError(f"{session_label} has unexpected fields")
        session_id = raw_session.get("session_id")
        date = raw_session.get("date")
        raw_turns = raw_session.get("turns")
        if not isinstance(session_id, str) or not session_id or session_id in sessions:
            raise RetrievalReportError(f"{session_label}.session_id is missing or duplicated")
        if not isinstance(date, str) or not date:
            raise RetrievalReportError(f"{session_label}.date must be non-empty text")
        if not isinstance(raw_turns, list):
            raise RetrievalReportError(f"{session_label}.turns must be a list")
        turns: list[dict[str, str]] = []
        for turn_index, raw_turn in enumerate(raw_turns):
            turn_label = f"{session_label}.turns[{turn_index}]"
            if not isinstance(raw_turn, dict) or set(raw_turn) != {"role", "content"}:
                raise RetrievalReportError(f"{turn_label} must carry exactly role and content")
            role = raw_turn.get("role")
            content = raw_turn.get("content")
            if not isinstance(role, str) or not role:
                raise RetrievalReportError(f"{turn_label}.role must be non-empty text")
            if not isinstance(content, str):
                raise RetrievalReportError(f"{turn_label}.content must be text")
            if content != content.strip():
                raise RetrievalReportError(f"{turn_label}.content must use canonical stripping")
            turns.append({"role": role, "content": content})
        observed_ids.append(session_id)
        sessions[session_id] = (date, tuple(turns))
    if observed_ids != expected_ids:
        raise RetrievalReportError(
            f"{label}.ranked_sessions must match the final ranking's top 10 in order"
        )
    return {"question": question, "question_date": question_date}, sessions


def _reconstruct_prompt(
    record: dict[str, str],
    sessions: dict[str, tuple[str, tuple[dict[str, str], ...]]],
    selected_ids: list[str],
    *,
    label: str,
) -> str:
    try:
        selected = [sessions[session_id] for session_id in selected_ids]
    except KeyError as exc:
        raise RetrievalReportError(
            f"{label} references session material outside the top 10"
        ) from exc
    return build_official_reader_prompt(record, selected)


def _validate_exact_context_evidence(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    expected_fields = {
        "method",
        "provider",
        "mode",
        "counted_surface",
        "packing_observation_source",
        "exact_model_tokenizer",
        "tokenizer_model",
        "tokenizer_revision",
        "tokenizer_artifact",
        "tokenizer_executable",
        "protocol",
        "response_identity_sha256",
        "serializer",
        "observation_accounting",
    }
    if set(metadata) != expected_fields:
        raise RetrievalReportError("exact context token metadata has unexpected fields")
    expected_scalars = {
        "method": "exact_serialized_reader_prompt",
        "provider": "JsonlExactTokenizer",
        "mode": "publishable-exact",
        "counted_surface": "complete_official_reader_prompt",
        "packing_observation_source": "provider_observed_full_prompt_decisions",
        "exact_model_tokenizer": True,
        "protocol": TOKENIZER_PROTOCOL,
    }
    for field, expected in expected_scalars.items():
        if type(metadata.get(field)) is not type(expected) or metadata.get(field) != expected:
            raise RetrievalReportError(f"exact context token metadata {field} is invalid")
    for field in ("tokenizer_model", "tokenizer_revision"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise RetrievalReportError(f"exact context token metadata {field} must be non-empty")
    expected_response_identity = tokenizer_response_identity_sha256(
        model=metadata["tokenizer_model"],
        revision=metadata["tokenizer_revision"],
        artifact_sha256=metadata["tokenizer_artifact"].get("sha256", "")
        if isinstance(metadata.get("tokenizer_artifact"), dict)
        else "",
    )
    if metadata.get("response_identity_sha256") != expected_response_identity:
        raise RetrievalReportError("exact tokenizer response identity digest is invalid")
    if metadata.get("serializer") != retrieval_eval.exact_context_serializer_metadata():
        raise RetrievalReportError("exact context serializer does not match the current tree")
    _bound_file_identity(
        metadata.get("tokenizer_artifact"),
        label="tokenizer artifact",
        repo_root=repo_root,
    )
    _bound_file_identity(
        metadata.get("tokenizer_executable"),
        label="tokenizer executable",
        repo_root=repo_root,
        executable=True,
    )

    observations: dict[int, dict[str, Any]] = {}
    observation_states: dict[int, tuple[int, tuple[str, ...]]] = {}
    state_observations: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
    observation_text_sizes: dict[int, tuple[int, int]] = {}
    provider_ids: set[str] = set()
    expected_budget_keys = {
        "budget=none",
        "budget=32000",
        "budget=16000",
        "budget=8000",
        "budget=4000",
        "budget=2000",
    }
    for case_index, case in enumerate(payload["cases"]):
        record, prompt_sessions = _exact_prompt_material(case, case_index=case_index)
        packing = case.get("exact_context_packing")
        if not isinstance(packing, dict) or set(packing) != {"k=5", "k=10"}:
            raise RetrievalReportError(f"case[{case_index}] lacks complete exact packing evidence")
        final_ranking = case["rankings"]["final"]
        for k in (5, 10):
            candidates = final_ranking[:k]
            rows = packing[f"k={k}"]
            if not isinstance(rows, dict) or set(rows) != expected_budget_keys:
                raise RetrievalReportError(f"case[{case_index}] k={k} has incomplete budgets")
            for budget in retrieval_eval.ANSWER_IN_CONTEXT_BUDGETS:
                row_label = f"case[{case_index}].k={k}.{retrieval_eval._budget_label(budget)}"
                row = rows[retrieval_eval._budget_label(budget)]
                fields = {
                    "budget",
                    "policy",
                    "initial_observation",
                    "decisions",
                    "kept_ids",
                    "final_observation",
                }
                if not isinstance(row, dict) or set(row) != fields:
                    raise RetrievalReportError(f"{row_label} has unexpected fields")
                if row.get("budget") != budget or row.get("policy") != "exact_serialized_greedy":
                    raise RetrievalReportError(f"{row_label} has invalid budget or policy")
                if budget is None:
                    if row.get("initial_observation") is not None or row.get("decisions") != []:
                        raise RetrievalReportError(f"{row_label} unbounded trace must be direct")
                    if row.get("kept_ids") != candidates:
                        raise RetrievalReportError(f"{row_label} must keep every candidate")
                    _observation(
                        row.get("final_observation"),
                        label=f"{row_label}.final_observation",
                        expected_prompt=_reconstruct_prompt(
                            record,
                            prompt_sessions,
                            candidates,
                            label=f"{row_label}.final_observation",
                        ),
                        state_key=(case_index, tuple(candidates)),
                        observations=observations,
                        observation_states=observation_states,
                        state_observations=state_observations,
                        observation_text_sizes=observation_text_sizes,
                        provider_ids=provider_ids,
                        response_identity_sha256=expected_response_identity,
                    )
                    continue
                initial = _observation(
                    row.get("initial_observation"),
                    label=f"{row_label}.initial_observation",
                    expected_prompt=_reconstruct_prompt(
                        record,
                        prompt_sessions,
                        [],
                        label=f"{row_label}.initial_observation",
                    ),
                    state_key=(case_index, ()),
                    observations=observations,
                    observation_states=observation_states,
                    state_observations=state_observations,
                    observation_text_sizes=observation_text_sizes,
                    provider_ids=provider_ids,
                    response_identity_sha256=expected_response_identity,
                )
                decisions = row.get("decisions")
                if not isinstance(decisions, list) or len(decisions) != len(candidates):
                    raise RetrievalReportError(f"{row_label} must trace every candidate")
                selected: list[str] = []
                current = initial
                for decision_index, (candidate, decision) in enumerate(
                    zip(candidates, decisions, strict=True)
                ):
                    decision_label = f"{row_label}.decisions[{decision_index}]"
                    decision_fields = {
                        "candidate_id",
                        "selected_before_ids",
                        "proposed_ids",
                        "observation",
                        "accepted",
                    }
                    if not isinstance(decision, dict) or set(decision) != decision_fields:
                        raise RetrievalReportError(f"{decision_label} has unexpected fields")
                    if (
                        decision.get("candidate_id") != candidate
                        or decision.get("selected_before_ids") != selected
                        or decision.get("proposed_ids") != [*selected, candidate]
                    ):
                        raise RetrievalReportError(f"{decision_label} breaks greedy lineage")
                    observation = _observation(
                        decision.get("observation"),
                        label=f"{decision_label}.observation",
                        expected_prompt=_reconstruct_prompt(
                            record,
                            prompt_sessions,
                            [*selected, candidate],
                            label=f"{decision_label}.observation",
                        ),
                        state_key=(case_index, tuple([*selected, candidate])),
                        observations=observations,
                        observation_states=observation_states,
                        state_observations=state_observations,
                        observation_text_sizes=observation_text_sizes,
                        provider_ids=provider_ids,
                        response_identity_sha256=expected_response_identity,
                    )
                    accepted = observation["token_count"] <= budget
                    if decision.get("accepted") is not accepted:
                        raise RetrievalReportError(f"{decision_label} acceptance is inconsistent")
                    if accepted:
                        selected.append(candidate)
                        current = observation
                final_observation = _observation(
                    row.get("final_observation"),
                    label=f"{row_label}.final_observation",
                    expected_prompt=_reconstruct_prompt(
                        record,
                        prompt_sessions,
                        selected,
                        label=f"{row_label}.final_observation",
                    ),
                    state_key=(case_index, tuple(selected)),
                    observations=observations,
                    observation_states=observation_states,
                    state_observations=state_observations,
                    observation_text_sizes=observation_text_sizes,
                    provider_ids=provider_ids,
                    response_identity_sha256=expected_response_identity,
                )
                if row.get("kept_ids") != selected or final_observation != current:
                    raise RetrievalReportError(f"{row_label} final selection is inconsistent")

    accounting = metadata.get("observation_accounting")
    accounting_fields = {
        "source",
        "requests",
        "responses",
        "unique_provider_request_ids",
        "text_characters",
        "text_utf8_bytes",
        "exact_response_identity_verified",
    }
    if not isinstance(accounting, dict) or set(accounting) != accounting_fields:
        raise RetrievalReportError("exact tokenizer observation accounting is malformed")
    requests = len(observations)
    if set(observations) != set(range(1, requests + 1)):
        raise RetrievalReportError("exact tokenizer request IDs must be globally contiguous")
    expected_accounting = {
        "source": "provider-observed",
        "requests": requests,
        "responses": requests,
        "unique_provider_request_ids": len(provider_ids),
        "exact_response_identity_verified": True,
    }
    for field, expected in expected_accounting.items():
        if type(accounting.get(field)) is not type(expected) or accounting.get(field) != expected:
            raise RetrievalReportError(f"exact tokenizer accounting {field} is invalid")
    expected_text_accounting = {
        "text_characters": sum(value[0] for value in observation_text_sizes.values()),
        "text_utf8_bytes": sum(value[1] for value in observation_text_sizes.values()),
    }
    for field, expected in expected_text_accounting.items():
        if type(accounting.get(field)) is not int or accounting.get(field) != expected:
            raise RetrievalReportError(
                f"exact tokenizer accounting {field} does not match reconstructed prompts"
            )


def _validate_token_accounting(
    payload: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    metadata = payload.get("context_token_accounting")
    fallback = retrieval_eval.context_token_accounting()
    if metadata == fallback or metadata is None:
        if any(case.get("exact_context_packing") is not None for case in payload["cases"]):
            raise RetrievalReportError("development token estimates cannot carry exact traces")
        return
    if not isinstance(metadata, dict) or metadata.get("exact_model_tokenizer") is not True:
        raise RetrievalReportError(
            "--run token accounting is neither exact nor the current fallback"
        )
    _validate_exact_context_evidence(payload, metadata, repo_root=repo_root)


def compile_report(
    run: Path,
    output: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    run_path, _ = _repo_input(run, repo_root=repo_root)
    if output.is_symlink() or output.resolve() == run_path:
        raise RetrievalReportError("--output must not be a symlink or overwrite --run")
    raw = run_path.read_bytes()
    run_sha256 = hashlib.sha256(raw).hexdigest()
    payload = _strict_object(raw, label="--run")
    try:
        validate_retrieval_run_protocol(payload)
        publishability_errors = retrieval_publishability_errors(payload)
    except ValueError as exc:
        raise RetrievalReportError(str(exc)) from exc
    if publishability_errors:
        raise RetrievalReportError("--run is not publishable: " + "; ".join(publishability_errors))
    if payload["dataset"].get("sample_seed") is not None:
        raise RetrievalReportError("canonical full-500 evidence must not carry a sample seed")
    if payload["implementation"] != retrieval_eval.retrieval_implementation_fingerprint():
        raise RetrievalReportError("--run implementation fingerprint does not match current tree")
    _validate_cases(payload)
    _validate_token_accounting(payload, repo_root=repo_root)

    report = retrieval_eval.build_longmemeval_report(payload, run_path)
    identity = report.get("run_artifact")
    expected_identity = {
        "path": retrieval_eval._artifact_path(run_path),
        "bytes": len(raw),
        "sha256": run_sha256,
    }
    if identity != expected_identity or run_path.read_bytes() != raw:
        raise RetrievalReportError("saved run changed or its report identity did not reconcile")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = _parser().parse_args()
    try:
        compile_report(args.run, args.output)
    except (OSError, RetrievalReportError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
