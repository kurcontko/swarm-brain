#!/usr/bin/env python3
"""End-to-end LongMemEval-S QA: retrieve through swarm-brain, read, judge.

This is the QA stage that `scripts/run_retrieval_eval.py` deliberately stops
short of.  For each question it publishes the question's haystack into a fresh
runtime, recalls the top ``--limit`` sessions through the real
``RetrievalService`` path, renders them for a reader, asks an
OpenAI-compatible reader model to answer, and scores the answer with the
official LongMemEval judge prompts.

**The judge here is a development judge, not the official one.**  The official
protocol is `src/evaluation/evaluate_qa.py` from
https://github.com/xiaowu0162/LongMemEval run with GPT-4o
(`gpt-4o-2024-08-06`).  This harness reuses that file's per-question-type
prompts verbatim but calls whatever model ``--judge-model`` names, so every
accuracy it emits is keyed ``dev_judge_accuracy`` and is never to be reported
as a LongMemEval score.  The hypothesis file it writes is in the exact format
`evaluate_qa.py` consumes — one JSON object per line with ``question_id`` and
``hypothesis`` — so the official judge can be run over the saved file later
without regenerating anything.

Deviations from the official generation script, stated so they can be argued
with:

1. Retrieval is swarm-brain's, not the reference BM25/embedding retrievers, and
   context is rendered from the sessions our recall returned.  The reader
   prompt template, the session rendering and the chronological ordering are
   copied from the reference `src/generation/run_generation.py`
   (``flat-session`` retriever, ``nl`` history format, no CoT).
2. ``--prompt-style swarm`` (the default) appends one sentence to that template
   instructing the reader to answer from the evidence only and to say the
   information is not available when it is absent, and surfaces an empty or
   thin bundle to the reader.  ``--prompt-style official`` reproduces the
   reference template byte for byte instead.
3. A question whose reader call fails after all retries still gets a
   hypothesis line, with an empty hypothesis, so the file always covers every
   question asked.  The reference script drops such questions, which silently
   shrinks the denominator.  Failures are counted in the run and report JSON.

Outputs land next to the retrieval benchmarks in ``benchmarks/retrieval/``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from _longmemeval_common import (
    EMPTY_CONTEXT_NOTE as _SHARED_EMPTY_CONTEXT_NOTE,
)
from _longmemeval_common import (
    LONGMEMEVAL_S_SHA256,
    LONGMEMEVAL_S_URL,
    SessionMemory,
    configure_embeddings,
    default_longmemeval_path,
    embedding_metadata,
    ensure_longmemeval,
    mean,
    percentiles,
    render_reader_history,
    render_reader_session,
    retrieve_question,
    select_questions,
    temporal_parse_metadata,
)
from _longmemeval_common import (
    OFFICIAL_ANSWER_TEMPLATE as _SHARED_OFFICIAL_ANSWER_TEMPLATE,
)

from swarmbrain.retrieval import TEMPORAL_QUERY_PARSER_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "benchmarks" / "retrieval"

# Same default seed as the retrieval harness, so ``--lme-sample 50`` selects the
# same 50 questions here as there and the QA numbers sit on top of recall
# numbers measured over the identical subset.
DEFAULT_SAMPLE_SEED = 20260807

OFFICIAL_JUDGE_MODEL = "gpt-4o-2024-08-06"
OFFICIAL_REPO = "https://github.com/xiaowu0162/LongMemEval"

QA_ARTIFACT_SCHEMA_VERSION = 4
QA_PROTOCOL_VERSION = "swarmbrain-longmemeval-qa-v4"
QA_RUN_ARTIFACT_TYPE = "swarmbrain-longmemeval-qa-run"
QA_REPORT_ARTIFACT_TYPE = "swarmbrain-longmemeval-qa-report"
CHAT_RECEIPT_ARTIFACT_TYPE = "swarmbrain-longmemeval-chat-provider-receipt"
CHAT_RECEIPT_SCHEMA_VERSION = 2
CHAT_RECEIPT_PROTOCOL_VERSION = "swarmbrain-longmemeval-chat-provider-receipt-v2"
CHAT_REQUEST_PARSER = "openai-compatible-chat-completions-request-strict-v1"
CHAT_RESPONSE_PARSER = "openai-compatible-chat-completions-strict-v1"
MAX_CHAT_REQUEST_BYTES = 8_388_608
MAX_CHAT_RESPONSE_BYTES = 4_194_304
CHAT_CALL_ROLES = frozenset({"reader", "development_judge", "official_judge"})
RETRIEVAL_RUN_ARTIFACT_TYPE = "swarmbrain-retrieval-eval-run"
RETRIEVAL_ARTIFACT_SCHEMA_VERSION = 2
RETRIEVAL_PROTOCOL_VERSION = "swarmbrain-longmemeval-retrieval-v2"
QWEN_EMBEDDING_PROVIDER = "OpenAICompatibleEmbeddingProvider"
QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_EMBEDDING_DIMENSIONS = 1024
LONGMEMEVAL_S_QUESTION_COUNT = 500
LONGMEMEVAL_S_SESSION_COUNT = 23_867
QWEN_QUERY_INSTRUCTION_SHA256 = "a695bbf99f6e2c59bbedb4ca2b397a995afbe92114c2d965a84acfac4253727f"

_RETRIEVAL_FINGERPRINT_REQUIRED_FILES = frozenset(
    {
        "scripts/_longmemeval_common.py",
        "scripts/evaluate_retrieval_runs.py",
        "scripts/run_retrieval_eval.py",
        "pyproject.toml",
        "uv.lock",
    }
)
_HEX_DIGITS = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_HEX_DIGITS)


def _fingerprint_tree(files: Mapping[str, str]) -> str:
    canonical = json.dumps(dict(files), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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
        raise ValueError(f"value is not canonical finite UTF-8 JSON: {exc}") from exc


def qa_implementation_fingerprint() -> dict[str, Any]:
    """Bind QA evidence to the exact reader/retrieval integration code."""

    paths = [
        REPO_ROOT / "scripts" / "_longmemeval_common.py",
        REPO_ROOT / "scripts" / "run_longmemeval_qa.py",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "uv.lock",
    ]
    paths.extend((REPO_ROOT / "src" / "swarmbrain").rglob("*.py"))
    files = {
        path.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }
    return {"tree_sha256": _fingerprint_tree(files), "files": files}


def validate_implementation_fingerprint(
    value: Any,
    *,
    label: str,
    required_files: frozenset[str],
) -> dict[str, Any]:
    """Validate both the shape and canonical tree hash of an implementation map."""

    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    files = value.get("files")
    tree_sha256 = value.get("tree_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{label} files must be a non-empty object")
    if not all(
        isinstance(path, str) and path and _is_sha256(digest) for path, digest in files.items()
    ):
        raise ValueError(f"{label} files must map paths to lowercase SHA-256 digests")
    missing = sorted(required_files - set(files))
    if missing:
        raise ValueError(f"{label} is missing required files: {', '.join(missing)}")
    if not any(path.startswith("src/swarmbrain/") and path.endswith(".py") for path in files):
        raise ValueError(f"{label} carries no swarmbrain source files")
    if not _is_sha256(tree_sha256) or tree_sha256 != _fingerprint_tree(files):
        raise ValueError(f"{label} tree_sha256 does not match its file map")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r} is forbidden")
        result[key] = value
    return result


def _load_strict_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_fields,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read strict JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def artifact_identity(path: Path) -> dict[str, Any]:
    """Return a portable path plus byte length and digest for one regular file."""

    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"artifact is missing or unsafe: {path}")
    content = resolved.read_bytes()
    return {
        "path": _relative(resolved),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


# --------------------------------------------------------------------------- #
# reader prompt — reference templates, verbatim
# --------------------------------------------------------------------------- #

# `run_generation.py`, retriever_type='flat-session', merge_key_expansion='none',
# cot=false.  Reproduced exactly, including the triple newline.
OFFICIAL_ANSWER_TEMPLATE = _SHARED_OFFICIAL_ANSWER_TEMPLATE

# The house variant: the reference instruction plus one evidence-only sentence.
# The abstention wire from the SOTA plan lives in that sentence together
# with the empty/thin bundle notes below.
SWARM_ANSWER_TEMPLATE = (
    "I will give you several history chats between you and a user. Please answer "
    "the question based on the relevant chat history. Answer only from the "
    "evidence in the chat history; if the chat history does not contain the "
    "information needed to answer, say that the information is not available."
    "\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:"
)

ANSWER_TEMPLATES = {"official": OFFICIAL_ANSWER_TEMPLATE, "swarm": SWARM_ANSWER_TEMPLATE}

# Nothing cleared the relevance floor.  The reference harness asserts a
# non-empty history and therefore has no wording for this; it can only happen
# with a positive ``--min-score``, which is our abstention signal.
EMPTY_CONTEXT_NOTE = _SHARED_EMPTY_CONTEXT_NOTE

THIN_CONTEXT_NOTE = (
    "Note: only {kept} of the {requested} requested sessions passed the memory "
    "relevance threshold; everything else in memory was judged irrelevant to this "
    "question.\n"
)


def render_session(index: int, date: str, turns: Sequence[dict[str, Any]]) -> str:
    """One session block, in the reference ``flat-session`` + ``nl`` format."""

    return render_reader_session(index, date, turns)


def render_history(sessions: Sequence[tuple[str, Sequence[dict[str, Any]]]]) -> str:
    """Render retrieved sessions chronologically, as the reference harness does.

    LongMemEval haystack dates are ``YYYY/MM/DD (Day) HH:MM``, which sorts
    lexicographically in chronological order — this is the reference script's
    ``retrieved_chunks.sort(key=lambda x: x[0])``.  The sort is stable, so
    same-dated sessions keep retrieval rank order.  Ordering by date rather than
    by rank is what makes temporal questions and knowledge-update supersession
    readable: the newest statement is last.
    """

    return render_reader_history(sessions)


def build_reader_prompt(
    record: dict[str, Any],
    sessions: Sequence[tuple[str, Sequence[dict[str, Any]]]],
    *,
    style: str = "swarm",
    requested: int | None = None,
    floored: bool = False,
) -> str:
    """Full reader prompt: retrieved history, the question date, the question."""

    template = ANSWER_TEMPLATES[style]
    if not sessions:
        history = EMPTY_CONTEXT_NOTE
    else:
        history = render_history(sessions)
        thin = floored and requested is not None and len(sessions) < requested
        if thin and style == "swarm":
            history = THIN_CONTEXT_NOTE.format(kept=len(sessions), requested=requested) + history
    return template.format(history, str(record["question_date"]), str(record["question"]))


# --------------------------------------------------------------------------- #
# judge — evaluate_qa.py prompts, verbatim
# --------------------------------------------------------------------------- #

_JUDGE_DEFAULT = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_JUDGE_TEMPORAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_JUDGE_KNOWLEDGE_UPDATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_JUDGE_PREFERENCE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_JUDGE_ABSTENTION = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."


def judge_prompt(
    task: str, question: str, answer: str, response: str, *, abstention: bool = False
) -> str:
    """``get_anscheck_prompt`` from the official ``evaluate_qa.py``, unmodified."""

    if abstention:
        return _JUDGE_ABSTENTION.format(question, answer, response)
    if task in {"single-session-user", "single-session-assistant", "multi-session"}:
        return _JUDGE_DEFAULT.format(question, answer, response)
    if task == "temporal-reasoning":
        return _JUDGE_TEMPORAL.format(question, answer, response)
    if task == "knowledge-update":
        return _JUDGE_KNOWLEDGE_UPDATE.format(question, answer, response)
    if task == "single-session-preference":
        return _JUDGE_PREFERENCE.format(question, answer, response)
    raise NotImplementedError(f"no official judge prompt for question type {task!r}")


def is_abstention_question(question_id: str) -> bool:
    """The official rule: ``abstention='_abs' in entry['question_id']``."""

    return "_abs" in question_id


def judge_label(text: str) -> bool:
    """The official rule: ``label = 'yes' in eval_response.lower()``.

    Deliberately kept as loose as the reference, so a dev-judge label means the
    same thing a GPT-4o label would.  The raw judge text is saved per question
    in the run JSON, so a disagreement is auditable rather than lost.
    """

    return "yes" in text.strip().lower()


def hypothesis_line(question_id: str, hypothesis: str) -> str:
    """One line of the official hypothesis file, as ``evaluate_qa.py`` reads it."""

    return json.dumps({"question_id": question_id, "hypothesis": hypothesis})


# --------------------------------------------------------------------------- #
# OpenAI-compatible chat client
# --------------------------------------------------------------------------- #


class ChatUnavailable(RuntimeError):
    """The endpoint stayed unusable across every retry."""


class ChatProtocolError(RuntimeError):
    """The request itself is wrong; retrying cannot help."""


@dataclass(frozen=True, slots=True)
class ReplayedChatRequest:
    """Normalized fields deterministically recovered from one request body."""

    prompt: str
    model: str
    temperature: float
    max_tokens: int
    thinking_mode: str | None


@dataclass(frozen=True, slots=True)
class ReplayedChatResponse:
    """Normalized fields deterministically recovered from one response body."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    response_model: str | None
    request_id: str | None
    system_fingerprint: str | None


def _chat_text(value: Any, *, label: str, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ChatProtocolError(f"{label} must be non-empty text without surrounding whitespace")
    if len(value.encode("utf-8")) > 512:
        raise ChatProtocolError(f"{label} exceeds its byte cap")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ChatProtocolError(f"{label} contains a control character")
    return value


def _chat_usage_integer(usage: dict[str, Any], name: str) -> int:
    value = usage.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ChatProtocolError(f"chat response usage.{name} must be a non-negative integer")
    return value


def replay_chat_request(raw_request: bytes) -> ReplayedChatRequest:
    """Strictly replay one exact OpenAI-compatible JSON request body."""

    if not isinstance(raw_request, bytes) or not raw_request:
        raise ChatProtocolError("chat raw request must be non-empty bytes")
    if len(raw_request) > MAX_CHAT_REQUEST_BYTES:
        raise ChatProtocolError("chat raw request exceeds its byte cap")
    try:
        request_text = raw_request.decode("utf-8")
        body = json.loads(
            request_text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_fields,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise ChatProtocolError("chat request body is malformed JSON") from None
    if not isinstance(body, dict):
        raise ChatProtocolError("chat request body must be a JSON object")
    required = {"model", "messages", "temperature", "max_tokens"}
    optional = {"thinking"}
    fields = set(body)
    if fields != required and fields != required | optional:
        raise ChatProtocolError("chat request fields differ from the frozen schema")

    model = _chat_text(body.get("model"), label="chat request model", required=True)
    temperature_value = body.get("temperature")
    if (
        isinstance(temperature_value, bool)
        or not isinstance(temperature_value, (int, float))
        or not math.isfinite(float(temperature_value))
        or not 0.0 <= float(temperature_value) <= 2.0
    ):
        raise ChatProtocolError("chat request temperature must be finite and between 0 and 2")
    max_tokens = body.get("max_tokens")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 1 <= max_tokens <= 1_000_000
    ):
        raise ChatProtocolError("chat request max_tokens is out of range")
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ChatProtocolError("chat request must contain exactly one message")
    message = messages[0]
    if not isinstance(message, dict) or set(message) != {"role", "content"}:
        raise ChatProtocolError("chat request message differs from the frozen schema")
    if message.get("role") != "user":
        raise ChatProtocolError("chat request message role must be user")
    prompt = message.get("content")
    if not isinstance(prompt, str) or not prompt:
        raise ChatProtocolError("chat request prompt must be non-empty text")

    thinking_mode: str | None = None
    if "thinking" in body:
        thinking = body.get("thinking")
        if (
            not isinstance(thinking, dict)
            or set(thinking) != {"type"}
            or thinking.get("type") not in {"enabled", "disabled"}
        ):
            raise ChatProtocolError("chat request thinking mode is malformed")
        thinking_mode = thinking["type"]
    return ReplayedChatRequest(
        prompt=prompt,
        model=model or "",
        temperature=float(temperature_value),
        max_tokens=max_tokens,
        thinking_mode=thinking_mode,
    )


def chat_request_bytes(
    *,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking_mode: str | None = None,
) -> bytes:
    """Build the canonical credential-free HTTP body and validate it by replay."""

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if thinking_mode is not None:
        payload["thinking"] = {"type": thinking_mode}
    try:
        raw_request = _canonical_json_bytes(payload)
    except ValueError as exc:
        raise ChatProtocolError(f"chat request cannot be canonically encoded: {exc}") from exc
    replay_chat_request(raw_request)
    return raw_request


def replay_chat_response(raw_response: bytes) -> ReplayedChatResponse:
    """Strictly replay one decoded OpenAI-compatible JSON response body."""

    if not isinstance(raw_response, bytes) or not raw_response:
        raise ChatProtocolError("chat raw response must be non-empty bytes")
    if len(raw_response) > MAX_CHAT_RESPONSE_BYTES:
        raise ChatProtocolError("chat raw response exceeds its byte cap")
    try:
        response_text = raw_response.decode("utf-8")
        body = json.loads(
            response_text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_fields,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise ChatProtocolError("chat provider returned malformed JSON") from None
    if not isinstance(body, dict):
        raise ChatProtocolError("chat response must be a JSON object")

    response_model = _chat_text(
        body.get("model"),
        label="chat response model",
        required=False,
    )
    request_id = _chat_text(
        body.get("id"),
        label="chat provider request id",
        required=False,
    )
    system_fingerprint = _chat_text(
        body.get("system_fingerprint"),
        label="chat response system_fingerprint",
        required=False,
    )
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ChatProtocolError("chat response must carry exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ChatProtocolError("chat response choice is malformed")
    if isinstance(choice.get("index"), bool) or choice.get("index") != 0:
        raise ChatProtocolError("chat response choice index must be integer zero")
    finish_reason = _chat_text(
        choice.get("finish_reason"),
        label="chat finish_reason",
        required=True,
    )
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ChatProtocolError("chat response assistant message is malformed")
    if message.get("refusal") is not None or message.get("tool_calls") not in (None, []):
        raise ChatProtocolError("chat response contains a refusal or tool call")
    raw_content = message.get("content")
    if raw_content is None:
        content = ""
    elif isinstance(raw_content, str):
        content = raw_content.strip()
    else:
        raise ChatProtocolError("chat response content must be text or null")

    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise ChatProtocolError("chat response did not report token usage")
    prompt_tokens = _chat_usage_integer(usage, "prompt_tokens")
    completion_tokens = _chat_usage_integer(usage, "completion_tokens")
    total_tokens = _chat_usage_integer(usage, "total_tokens")
    if total_tokens != prompt_tokens + completion_tokens:
        raise ChatProtocolError("chat response token usage does not reconcile")
    return ReplayedChatResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        finish_reason=finish_reason or "",
        response_model=response_model,
        request_id=request_id,
        system_fingerprint=system_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    attempts: int
    latency_ms: float
    response_model: str | None
    request_id: str | None
    system_fingerprint: str | None
    prompt_sha256: str
    prompt_utf8_bytes: int
    request_parser: str
    response_parser: str
    endpoint_url: str
    prompt_bytes: bytes = field(repr=False)
    raw_request: bytes = field(repr=False)
    raw_response: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if self.request_parser != CHAT_REQUEST_PARSER:
            raise ChatProtocolError("chat request parser identity drifted")
        if self.response_parser != CHAT_RESPONSE_PARSER:
            raise ChatProtocolError("chat response parser identity drifted")
        if not _is_sha256(self.prompt_sha256):
            raise ChatProtocolError("chat prompt digest is not a lowercase SHA-256")
        if (
            isinstance(self.prompt_utf8_bytes, bool)
            or not isinstance(self.prompt_utf8_bytes, int)
            or self.prompt_utf8_bytes <= 0
        ):
            raise ChatProtocolError("chat prompt byte count must be positive")
        if not isinstance(self.prompt_bytes, bytes) or not self.prompt_bytes:
            raise ChatProtocolError("chat prompt receipt must retain non-empty bytes")
        if (
            len(self.prompt_bytes) != self.prompt_utf8_bytes
            or hashlib.sha256(self.prompt_bytes).hexdigest() != self.prompt_sha256
        ):
            raise ChatProtocolError("chat prompt bytes differ from their digest or byte count")
        try:
            self.prompt_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise ChatProtocolError("chat prompt receipt is not valid UTF-8") from None
        endpoint = urlsplit(self.endpoint_url)
        if (
            endpoint.scheme not in {"http", "https"}
            or not endpoint.netloc
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
            or endpoint.path != "/v1/chat/completions"
        ):
            raise ChatProtocolError("chat endpoint URL is not the canonical completion endpoint")
        if (
            isinstance(self.attempts, bool)
            or not isinstance(self.attempts, int)
            or self.attempts <= 0
        ):
            raise ChatProtocolError("chat attempts must be a positive integer")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(float(self.latency_ms))
            or self.latency_ms < 0
        ):
            raise ChatProtocolError("chat latency must be finite and non-negative")
        replayed_request = replay_chat_request(self.raw_request)
        if replayed_request.prompt.encode("utf-8") != self.prompt_bytes:
            raise ChatProtocolError("chat request prompt differs from retained prompt bytes")
        replayed = replay_chat_response(self.raw_response)
        if (
            self.content != replayed.content
            or self.prompt_tokens != replayed.prompt_tokens
            or self.completion_tokens != replayed.completion_tokens
            or self.total_tokens != replayed.total_tokens
            or self.finish_reason != replayed.finish_reason
            or self.response_model != replayed.response_model
            or self.request_id != replayed.request_id
            or self.system_fingerprint != replayed.system_fingerprint
        ):
            raise ChatProtocolError("chat result differs from replayed raw provider response")

    @property
    def raw_response_sha256(self) -> str:
        return hashlib.sha256(self.raw_response).hexdigest()

    @property
    def raw_request_sha256(self) -> str:
        return hashlib.sha256(self.raw_request).hexdigest()

    @property
    def request(self) -> ReplayedChatRequest:
        return replay_chat_request(self.raw_request)


def chat_result_from_raw_response(
    raw_response: bytes,
    *,
    prompt: str,
    attempts: int,
    latency_ms: float,
    raw_request: bytes | None = None,
    request_model: str | None = None,
    request_temperature: float = 0.0,
    request_max_tokens: int = 4096,
    thinking_mode: str | None = None,
    endpoint_url: str = "https://fixture.invalid/v1/chat/completions",
) -> ChatResult:
    """Build a normalized result only by replaying retained provider bytes."""

    if not isinstance(prompt, str) or not prompt:
        raise ChatProtocolError("chat prompt must be non-empty text")
    prompt_bytes = prompt.encode("utf-8")
    replayed = replay_chat_response(raw_response)
    if raw_request is None:
        raw_request = chat_request_bytes(
            prompt=prompt,
            model=request_model or replayed.response_model or "unverified-model",
            temperature=request_temperature,
            max_tokens=request_max_tokens,
            thinking_mode=thinking_mode,
        )
    replayed_request = replay_chat_request(raw_request)
    if replayed_request.prompt != prompt:
        raise ChatProtocolError("chat raw request prompt differs from the supplied prompt")
    return ChatResult(
        content=replayed.content,
        prompt_tokens=replayed.prompt_tokens,
        completion_tokens=replayed.completion_tokens,
        total_tokens=replayed.total_tokens,
        finish_reason=replayed.finish_reason,
        attempts=attempts,
        latency_ms=latency_ms,
        response_model=replayed.response_model,
        request_id=replayed.request_id,
        system_fingerprint=replayed.system_fingerprint,
        prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        prompt_utf8_bytes=len(prompt_bytes),
        request_parser=CHAT_REQUEST_PARSER,
        response_parser=CHAT_RESPONSE_PARSER,
        endpoint_url=endpoint_url,
        prompt_bytes=prompt_bytes,
        raw_request=raw_request,
        raw_response=raw_response,
    )


_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_BACKOFF_SECONDS = (1.0, 3.0, 8.0, 20.0)


class ChatClient:
    """Minimal ``/v1/chat/completions`` client with retry and backoff.

    Reasoning models (DeepSeek's ``deepseek-*`` family among them) emit
    ``reasoning_content`` before ``content``.  A small ``max_tokens`` is spent
    entirely on reasoning and the answer comes back empty with
    ``finish_reason='length'``, so ``max_tokens`` is generous by default and an
    empty ``content`` is treated as a retryable failure rather than an answer.
    ``reasoning_content`` is never read: it is not the model's answer, and
    feeding it to a judge would score the wrong text.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float = 300.0,
        attempts: int = 4,
        client: Any | None = None,
        required_response_model: str | None = None,
        require_request_id: bool = False,
        thinking_mode: str | None = None,
    ) -> None:
        stripped = base_url.strip().rstrip("/")
        parsed = urlsplit(stripped)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("chat base URL must be an http:// or https:// URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "chat base URL must not contain credentials, query parameters, or fragments"
            )
        if parsed.path not in {"", "/", "/v1"}:
            raise ValueError("chat base URL path must be empty or /v1")
        if stripped.endswith("/v1"):
            stripped = stripped[: -len("/v1")]
        self.base_url = stripped
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_mode = thinking_mode
        self.required_response_model = required_response_model
        self.require_request_id = require_request_id
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._attempts = attempts
        self._client = client
        chat_request_bytes(
            prompt="configuration validation",
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            thinking_mode=self.thinking_mode,
        )

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def complete(self, prompt: str) -> ChatResult:
        client = self._ensure_client()
        headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        raw_request = chat_request_bytes(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            thinking_mode=self.thinking_mode,
        )
        endpoint_url = f"{self.base_url}/v1/chat/completions"
        started = perf_counter()
        last: str = "no attempt was made"
        for attempt in range(self._attempts):
            try:
                response = await client.post(
                    endpoint_url,
                    content=raw_request,
                    headers=headers,
                )
            except Exception as exc:  # transport fault: retry
                last = f"{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if status in _RETRYABLE_STATUS:
                    last = f"HTTP {status}"
                elif not 200 <= status < 300:
                    raise ChatProtocolError(
                        f"{self.base_url} rejected the request with HTTP {status}"
                    )
                else:
                    encoding = str(getattr(response, "headers", {}).get("content-encoding", ""))
                    if encoding.strip().casefold() not in {"", "identity"}:
                        raise ChatProtocolError(
                            "chat provider returned an unsupported content encoding"
                        )
                    raw_response = bytes(response.content)
                    replayed = replay_chat_response(raw_response)
                    if (
                        self.required_response_model is not None
                        and replayed.response_model != self.required_response_model
                    ):
                        raise ChatProtocolError(
                            "chat response model mismatch: "
                            f"expected {self.required_response_model!r}, "
                            f"got {replayed.response_model!r}"
                        )
                    if self.require_request_id and not replayed.request_id:
                        raise ChatProtocolError("chat response carried no provider request id")
                    if replayed.content:
                        return chat_result_from_raw_response(
                            raw_response,
                            prompt=prompt,
                            attempts=attempt + 1,
                            latency_ms=(perf_counter() - started) * 1000.0,
                            raw_request=raw_request,
                            endpoint_url=endpoint_url,
                        )
                    # Empty content with finish_reason 'length' is the reasoning
                    # budget having eaten the whole allowance.
                    last = f"empty content (finish_reason={replayed.finish_reason!r})"
            if attempt + 1 < self._attempts:
                await asyncio.sleep(_BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)])
        raise ChatUnavailable(f"{self.model} failed after {self._attempts} attempts: {last}")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


# --------------------------------------------------------------------------- #
# one question, end to end
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class QuestionOutcome:
    record: dict[str, Any]
    retrieved_keys: tuple[str, ...]
    retrieved_relevance: tuple[float, ...]
    gold_keys: tuple[str, ...]
    hypothesis: str
    reader: ChatResult | None
    reader_error: str | None
    judge: ChatResult | None
    judge_error: str | None
    label: bool | None
    retrieval_ms: float
    total_ms: float
    temporal_routing: dict[str, Any] | None = None

    @property
    def judge_text(self) -> str | None:
        return self.judge.content if self.judge is not None else None

    @property
    def question_id(self) -> str:
        return str(self.record["question_id"])

    @property
    def question_type(self) -> str:
        return str(self.record["question_type"])

    @property
    def abstention(self) -> bool:
        return is_abstention_question(self.question_id)

    @property
    def gold_hits(self) -> int:
        return len(set(self.retrieved_keys) & set(self.gold_keys))


@dataclass(frozen=True, slots=True)
class ReplayedRetrieval:
    """A validated reader bundle reconstructed from a saved retrieval run."""

    hits: tuple[tuple[SessionMemory, float], ...]
    gold_keys: tuple[str, ...]
    wall_ms: float
    temporal_routing: dict[str, Any] | None


def replay_retrieval_case(
    record: dict[str, Any],
    case: Mapping[str, Any],
    *,
    limit: int,
    min_score: float,
) -> ReplayedRetrieval:
    """Rebuild exactly the saved final bundle without invoking retrieval again."""

    question_id = str(record["question_id"])
    if min_score != 0.0:
        raise ValueError(
            "saved-run replay requires --min-score 0.0 because the artifact does not "
            "carry calibrated relevance beyond its final truncated bundle"
        )
    if str(case.get("case_id")) != question_id:
        raise ValueError(f"retrieval case does not match question {question_id}")
    final = (case.get("rankings") or {}).get("final")
    relevance = case.get("final_relevance")
    if not isinstance(final, list) or not isinstance(relevance, list):
        raise ValueError(f"retrieval case {question_id} lacks final ranking/relevance")
    if len(relevance) < len(final):
        raise ValueError(f"retrieval case {question_id} has incomplete relevance evidence")

    session_ids = [str(value) for value in record["haystack_session_ids"]]
    dates = [str(value) for value in record["haystack_dates"]]
    turns = record["haystack_sessions"]
    if not (len(session_ids) == len(dates) == len(turns)):
        raise ValueError(f"LongMemEval question {question_id} has misaligned sessions")

    hits: list[tuple[SessionMemory, float]] = []
    seen: set[str] = set()
    for offset, raw_key in enumerate(final[:limit]):
        key = str(raw_key)
        if key in seen:
            raise ValueError(f"retrieval case {question_id} repeats session key {key}")
        seen.add(key)
        try:
            raw_position, saved_session_id = key.split(":", 1)
            position = int(raw_position)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"retrieval case {question_id} has invalid session key {key!r}"
            ) from exc
        if not 0 <= position < len(session_ids):
            raise ValueError(f"retrieval case {question_id} points outside its haystack")
        expected_key = f"{position:03d}:{session_ids[position]}"
        if key != expected_key or saved_session_id != session_ids[position]:
            raise ValueError(
                f"retrieval case {question_id} session key disagrees with the pinned dataset"
            )
        score = float(relevance[offset])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"retrieval case {question_id} has invalid relevance {score}")
        session = SessionMemory(
            position=position,
            session_id=session_ids[position],
            date=dates[position],
            turns=tuple(turns[position]),
            memory_id=f"replay:{question_id}:{key}",
        )
        hits.append((session, score))

    answers = {str(value) for value in record.get("answer_session_ids", ())}
    gold_keys = tuple(
        f"{position:03d}:{session_id}"
        for position, session_id in enumerate(session_ids)
        if session_id in answers
    )
    temporal = case.get("temporal_routing")
    if temporal is not None and not isinstance(temporal, dict):
        raise ValueError(f"retrieval case {question_id} has an invalid temporal trace")
    wall_ms = float(case.get("wall_ms") or 0.0)
    if not math.isfinite(wall_ms) or wall_ms < 0.0:
        raise ValueError(f"retrieval case {question_id} has invalid latency")
    return ReplayedRetrieval(
        hits=tuple(hits),
        gold_keys=gold_keys,
        wall_ms=wall_ms,
        temporal_routing=temporal,
    )


def _strict_integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def validate_retrieval_run_protocol(payload: Any) -> dict[str, Any]:
    """Validate the schema-v2 retrieval envelope shared by dev and official replay."""

    if not isinstance(payload, dict):
        raise ValueError("--retrieval-run must contain one JSON object")
    expected = {
        "artifact_type": RETRIEVAL_RUN_ARTIFACT_TYPE,
        "schema_version": RETRIEVAL_ARTIFACT_SCHEMA_VERSION,
        "protocol_version": RETRIEVAL_PROTOCOL_VERSION,
        "track": "longmemeval-s",
        "granularity": "one memory per haystack session",
    }
    for key, wanted in expected.items():
        if type(payload.get(key)) is not type(wanted) or payload.get(key) != wanted:
            raise ValueError(f"--retrieval-run {key} must be {wanted!r}, got {payload.get(key)!r}")
    validate_implementation_fingerprint(
        payload.get("implementation"),
        label="--retrieval-run implementation",
        required_files=_RETRIEVAL_FINGERPRINT_REQUIRED_FILES,
    )
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("--retrieval-run carries no dataset metadata")
    if dataset.get("name") != "LongMemEval-S":
        raise ValueError("--retrieval-run must name the LongMemEval-S dataset")
    if dataset.get("sha256") != LONGMEMEVAL_S_SHA256:
        raise ValueError("--retrieval-run was produced from a different dataset digest")
    total_questions = _strict_integer(
        dataset.get("total_questions"), label="retrieval dataset total_questions", minimum=1
    )
    evaluated_questions = _strict_integer(
        dataset.get("evaluated_questions"),
        label="retrieval dataset evaluated_questions",
        minimum=1,
    )
    if evaluated_questions > total_questions:
        raise ValueError("retrieval evaluated_questions exceeds total_questions")
    recall_limit = _strict_integer(
        payload.get("recall_limit"), label="retrieval recall_limit", minimum=1
    )
    saved_depth = _strict_integer(
        payload.get("saved_ranking_depth"),
        label="retrieval saved_ranking_depth",
        minimum=1,
    )
    if saved_depth < recall_limit:
        raise ValueError("retrieval saved_ranking_depth is below recall_limit")
    if not isinstance(payload.get("dense_lane_enabled"), bool):
        raise ValueError("retrieval dense_lane_enabled must be boolean")
    temporal = payload.get("temporal_query_routing")
    if not isinstance(temporal, dict) or not isinstance(temporal.get("enabled"), bool):
        raise ValueError("retrieval temporal_query_routing must carry a boolean enabled field")
    embedding = payload.get("embedding")
    if embedding is not None and not isinstance(embedding, dict):
        raise ValueError("retrieval embedding metadata must be an object or null")
    accounting = payload.get("embedding_call_accounting")
    if accounting is not None and not isinstance(accounting, dict):
        raise ValueError("retrieval embedding call accounting must be an object or null")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != evaluated_questions:
        raise ValueError("--retrieval-run case count does not match evaluated_questions")
    return payload


def retrieval_publishability_errors(payload: dict[str, Any]) -> list[str]:
    """Explain why a valid schema-v2 run cannot support an official QA replay."""

    validate_retrieval_run_protocol(payload)
    errors: list[str] = []
    dataset = payload["dataset"]
    cases = payload["cases"]
    if (
        dataset.get("total_questions") != LONGMEMEVAL_S_QUESTION_COUNT
        or dataset.get("evaluated_questions") != LONGMEMEVAL_S_QUESTION_COUNT
    ):
        errors.append("source retrieval must evaluate all 500 cleaned questions")
    if payload.get("saved_ranking_depth", 0) < 50:
        errors.append("source retrieval must save at least 50 ranking positions")
    if payload.get("dense_lane_enabled") is not True:
        errors.append("source retrieval must enable the semantic dense lane")
    temporal = payload.get("temporal_query_routing")
    if not isinstance(temporal, dict) or temporal.get("enabled") is not False:
        errors.append("source retrieval must use the canonical non-routed temporal protocol")

    embedding = payload.get("embedding")
    expected_embedding = {
        "provider": QWEN_EMBEDDING_PROVIDER,
        "model": QWEN_EMBEDDING_MODEL,
        "dimensions": QWEN_EMBEDDING_DIMENSIONS,
        "response_model_requirement": QWEN_EMBEDDING_MODEL,
        "query_instruction_sha256": QWEN_QUERY_INSTRUCTION_SHA256,
    }
    if not isinstance(embedding, dict):
        errors.append("source retrieval lacks semantic embedding metadata")
    else:
        for key, wanted in expected_embedding.items():
            if type(embedding.get(key)) is not type(wanted) or embedding.get(key) != wanted:
                errors.append(f"source embedding {key} must be {wanted!r}")

    case_ids: set[str] = set()
    document_inputs = 0
    degraded = 0
    malformed_cases = 0
    for case in cases:
        if not isinstance(case, dict):
            malformed_cases += 1
            continue
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            malformed_cases += 1
        else:
            case_ids.add(case_id)
        haystack_sessions = case.get("haystack_sessions")
        if (
            isinstance(haystack_sessions, bool)
            or not isinstance(haystack_sessions, int)
            or haystack_sessions < 1
        ):
            malformed_cases += 1
        else:
            document_inputs += haystack_sessions
        degraded_lanes = case.get("degraded_lanes")
        if not isinstance(degraded_lanes, list):
            malformed_cases += 1
        elif degraded_lanes:
            degraded += 1
    if malformed_cases:
        errors.append(f"source retrieval contains {malformed_cases} malformed case records")
    if degraded:
        errors.append(f"source retrieval contains {degraded} cases with degraded lanes")
    if document_inputs != LONGMEMEVAL_S_SESSION_COUNT:
        errors.append(
            "source retrieval document corpus must contain exactly "
            f"{LONGMEMEVAL_S_SESSION_COUNT} cleaned sessions"
        )

    accounting = payload.get("embedding_call_accounting")
    expected_calls = {
        "source": "provider-observed",
        "document_inputs": document_inputs,
        "document_batch_calls": len(cases),
        "query_calls": len(cases),
        "successful_http_calls": 2 * len(cases),
    }
    if not isinstance(accounting, dict):
        errors.append("source retrieval lacks embedding call accounting")
    else:
        for key, wanted in expected_calls.items():
            if type(accounting.get(key)) is not type(wanted) or accounting.get(key) != wanted:
                errors.append(f"source embedding call accounting {key} must be {wanted!r}")
        attempts = accounting.get("http_attempts")
        successes = expected_calls["successful_http_calls"]
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < successes:
            errors.append(
                "source embedding call accounting http_attempts must cover every successful call"
            )
    return errors


def load_retrieval_run(
    path: Path,
    records: Sequence[dict[str, Any]],
    *,
    limit: int,
    require_publishable: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load a dataset-bound schema-v2 run and validate selected coverage."""

    limit = _strict_integer(limit, label="requested replay limit", minimum=1)
    payload = validate_retrieval_run_protocol(_load_strict_json_object(path))
    publishability_errors = retrieval_publishability_errors(payload)
    if require_publishable and publishability_errors:
        raise ValueError("--retrieval-run is not publishable: " + "; ".join(publishability_errors))
    recall_limit = payload["recall_limit"]
    if recall_limit < limit:
        raise ValueError(
            f"--retrieval-run recall_limit {recall_limit} is below requested --limit {limit}"
        )
    raw_cases = payload["cases"]
    cases: dict[str, dict[str, Any]] = {}
    for case in raw_cases:
        if not isinstance(case, dict):
            raise ValueError("--retrieval-run contains a malformed case")
        case_id = str(case.get("case_id") or "")
        if not case_id or case_id in cases:
            raise ValueError(f"--retrieval-run has a missing or duplicate case id {case_id!r}")
        cases[case_id] = case
    missing = [
        str(record["question_id"]) for record in records if str(record["question_id"]) not in cases
    ]
    if missing:
        raise ValueError(f"--retrieval-run is missing {len(missing)} selected questions")
    for record in records:
        question_id = str(record["question_id"])
        case = cases[question_id]
        if case.get("haystack_sessions") != len(record["haystack_sessions"]):
            raise ValueError(
                f"retrieval case {question_id} haystack count disagrees with the pinned dataset"
            )
        replay_retrieval_case(record, case, limit=limit, min_score=0.0)
    return cases, payload


async def answer_question(
    record: dict[str, Any],
    *,
    reader: ChatClient,
    judge: ChatClient | None,
    limit: int,
    min_score: float,
    use_dense: bool,
    prompt_style: str,
    temporal_query_routing: bool = False,
    retrieval_case: Mapping[str, Any] | None = None,
) -> QuestionOutcome:
    started = perf_counter()
    if retrieval_case is None:
        retrieved = await retrieve_question(
            record,
            limit=limit,
            min_score=min_score,
            use_dense=use_dense,
            temporal_query_routing=temporal_query_routing,
        )
        hits: tuple[tuple[SessionMemory, float], ...] = retrieved.retrieved_sessions()
        gold_keys = retrieved.relevant_keys
        retrieval_ms = retrieved.wall_ms
        temporal_routing = temporal_parse_metadata(retrieved.temporal_parse)
    else:
        replayed = replay_retrieval_case(
            record,
            retrieval_case,
            limit=limit,
            min_score=min_score,
        )
        hits = replayed.hits
        gold_keys = replayed.gold_keys
        retrieval_ms = replayed.wall_ms
        temporal_routing = replayed.temporal_routing
    prompt = build_reader_prompt(
        record,
        [(session.date, session.turns) for session, _ in hits],
        style=prompt_style,
        requested=limit,
        floored=min_score > 0.0,
    )

    result: ChatResult | None = None
    reader_error: str | None = None
    hypothesis = ""
    try:
        result = await reader.complete(prompt)
        hypothesis = result.content
    except ChatUnavailable as exc:
        reader_error = str(exc)

    verdict: ChatResult | None = None
    judge_error: str | None = None
    label: bool | None = None
    if judge is not None:
        try:
            verdict = await judge.complete(
                judge_prompt(
                    str(record["question_type"]),
                    str(record["question"]),
                    str(record["answer"]),
                    hypothesis,
                    abstention=is_abstention_question(str(record["question_id"])),
                )
            )
            label = judge_label(verdict.content)
        except ChatUnavailable as exc:
            judge_error = str(exc)

    return QuestionOutcome(
        record=record,
        retrieved_keys=tuple(session.key for session, _ in hits),
        retrieved_relevance=tuple(round(score, 6) for _, score in hits),
        gold_keys=gold_keys,
        hypothesis=hypothesis,
        reader=result,
        reader_error=reader_error,
        judge=verdict,
        judge_error=judge_error,
        label=label,
        retrieval_ms=retrieval_ms,
        total_ms=(perf_counter() - started) * 1000.0,
        temporal_routing=temporal_routing,
    )


async def run_questions(
    records: Sequence[dict[str, Any]],
    *,
    reader: ChatClient,
    judge: ChatClient | None,
    limit: int,
    min_score: float,
    use_dense: bool,
    prompt_style: str,
    concurrency: int,
    temporal_query_routing: bool = False,
    retrieval_cases: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[QuestionOutcome]:
    """Run every question with a bounded number of readers in flight."""

    semaphore = asyncio.Semaphore(concurrency)
    done = 0
    total = len(records)

    async def one(record: dict[str, Any]) -> QuestionOutcome:
        nonlocal done
        async with semaphore:
            outcome = await answer_question(
                record,
                reader=reader,
                judge=judge,
                limit=limit,
                min_score=min_score,
                use_dense=use_dense,
                prompt_style=prompt_style,
                temporal_query_routing=temporal_query_routing,
                retrieval_case=(
                    retrieval_cases[str(record["question_id"])]
                    if retrieval_cases is not None
                    else None
                ),
            )
        done += 1
        if done % 10 == 0 or done == total:
            print(f"  qa: {done}/{total} questions", file=sys.stderr, flush=True)
        return outcome

    return list(await asyncio.gather(*(one(record) for record in records)))


# --------------------------------------------------------------------------- #
# raw provider-response receipt sidecar
# --------------------------------------------------------------------------- #


def chat_receipt_record(
    question_id: str,
    call_role: str,
    result: ChatResult,
) -> dict[str, Any]:
    """Serialize one content-bearing, exactly replayable provider receipt."""

    if not isinstance(question_id, str) or not question_id or question_id != question_id.strip():
        raise ValueError("chat receipt question_id must be non-empty canonical text")
    if call_role not in CHAT_CALL_ROLES:
        raise ValueError("chat receipt call_role is not registered")
    if not isinstance(result, ChatResult):
        raise ValueError("chat receipt requires a ChatResult")
    # Re-run validation at the serialization boundary rather than trusting that
    # an object was not constructed through an unsafe compatibility layer.
    replay_chat_request(result.raw_request)
    replay_chat_response(result.raw_response)
    return {
        "artifact_type": CHAT_RECEIPT_ARTIFACT_TYPE,
        "schema_version": CHAT_RECEIPT_SCHEMA_VERSION,
        "protocol_version": CHAT_RECEIPT_PROTOCOL_VERSION,
        "question_id": question_id,
        "call_role": call_role,
        "prompt": {
            "sha256": result.prompt_sha256,
            "utf8_bytes": result.prompt_utf8_bytes,
            "encoding": "base64-exact-utf8-prompt",
            "raw_base64": base64.b64encode(result.prompt_bytes).decode("ascii"),
        },
        "provider_request": {
            "parser": result.request_parser,
            "encoding": "base64-exact-http-request-body",
            "raw_bytes": len(result.raw_request),
            "raw_sha256": result.raw_request_sha256,
            "raw_base64": base64.b64encode(result.raw_request).decode("ascii"),
        },
        "provider_response": {
            "parser": result.response_parser,
            "encoding": "base64-exact-decoded-http-body",
            "raw_bytes": len(result.raw_response),
            "raw_sha256": result.raw_response_sha256,
            "raw_base64": base64.b64encode(result.raw_response).decode("ascii"),
        },
        "transport": {
            "endpoint_url": result.endpoint_url,
            "method": "POST",
            "content_type": "application/json",
            "accept_encoding": "identity",
            "attempts": result.attempts,
            "latency_ms": result.latency_ms,
            "latency_source": "caller-observed-monotonic-clock",
        },
    }


def chat_receipt_records(outcomes: Sequence[QuestionOutcome]) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for outcome in outcomes:
        if outcome.reader is not None:
            records.append(chat_receipt_record(outcome.question_id, "reader", outcome.reader))
        if outcome.judge is not None:
            records.append(
                chat_receipt_record(outcome.question_id, "development_judge", outcome.judge)
            )
    return tuple(records)


def chat_receipt_artifact_bytes(outcomes: Sequence[QuestionOutcome]) -> bytes:
    return b"".join(
        _canonical_json_bytes(record) + b"\n" for record in chat_receipt_records(outcomes)
    )


def write_chat_receipts(outcomes: Sequence[QuestionOutcome], path: Path) -> None:
    path.write_bytes(chat_receipt_artifact_bytes(outcomes))


def validate_chat_receipt_record(record: Any) -> ChatResult:
    """Decode and strictly replay one sidecar row without a caller-normalized response."""

    if not isinstance(record, dict) or set(record) != {
        "artifact_type",
        "schema_version",
        "protocol_version",
        "question_id",
        "call_role",
        "prompt",
        "provider_request",
        "provider_response",
        "transport",
    }:
        raise ValueError("chat receipt fields differ from the exact schema")
    expected = {
        "artifact_type": CHAT_RECEIPT_ARTIFACT_TYPE,
        "schema_version": CHAT_RECEIPT_SCHEMA_VERSION,
        "protocol_version": CHAT_RECEIPT_PROTOCOL_VERSION,
    }
    if any(
        type(record.get(key)) is not type(value) or record.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("chat receipt protocol identity drifted")
    question_id = record.get("question_id")
    if not isinstance(question_id, str) or not question_id or question_id != question_id.strip():
        raise ValueError("chat receipt question_id is invalid")
    if record.get("call_role") not in CHAT_CALL_ROLES:
        raise ValueError("chat receipt call_role is not registered")

    prompt = record.get("prompt")
    if not isinstance(prompt, dict) or set(prompt) != {
        "sha256",
        "utf8_bytes",
        "encoding",
        "raw_base64",
    }:
        raise ValueError("chat receipt prompt binding is malformed")
    if not _is_sha256(prompt.get("sha256")):
        raise ValueError("chat receipt prompt digest is invalid")
    prompt_bytes = prompt.get("utf8_bytes")
    if isinstance(prompt_bytes, bool) or not isinstance(prompt_bytes, int) or prompt_bytes <= 0:
        raise ValueError("chat receipt prompt byte count must be positive")
    if prompt.get("encoding") != "base64-exact-utf8-prompt":
        raise ValueError("chat receipt prompt encoding drifted")
    encoded_prompt = prompt.get("raw_base64")
    if not isinstance(encoded_prompt, str):
        raise ValueError("chat receipt raw prompt must be base64 text")
    try:
        raw_prompt = base64.b64decode(encoded_prompt, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("chat receipt raw prompt is invalid base64") from None
    if len(raw_prompt) != prompt_bytes:
        raise ValueError("chat receipt raw prompt byte count is inconsistent")
    if hashlib.sha256(raw_prompt).hexdigest() != prompt.get("sha256"):
        raise ValueError("chat receipt raw prompt digest is inconsistent")
    try:
        raw_prompt.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("chat receipt raw prompt is not valid UTF-8") from None

    request = record.get("provider_request")
    if not isinstance(request, dict) or set(request) != {
        "parser",
        "encoding",
        "raw_bytes",
        "raw_sha256",
        "raw_base64",
    }:
        raise ValueError("chat receipt provider-request binding is malformed")
    if request.get("parser") != CHAT_REQUEST_PARSER:
        raise ValueError("chat receipt request parser drifted")
    if request.get("encoding") != "base64-exact-http-request-body":
        raise ValueError("chat receipt request encoding drifted")
    encoded_request = request.get("raw_base64")
    if not isinstance(encoded_request, str):
        raise ValueError("chat receipt raw request must be base64 text")
    try:
        raw_request = base64.b64decode(encoded_request, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("chat receipt raw request is invalid base64") from None
    request_bytes = request.get("raw_bytes")
    if isinstance(request_bytes, bool) or not isinstance(request_bytes, int) or request_bytes <= 0:
        raise ValueError("chat receipt raw request byte count must be positive")
    if request_bytes != len(raw_request):
        raise ValueError("chat receipt raw request byte count is inconsistent")
    if request.get("raw_sha256") != hashlib.sha256(raw_request).hexdigest():
        raise ValueError("chat receipt raw request digest is inconsistent")
    replayed_request = replay_chat_request(raw_request)
    if replayed_request.prompt.encode("utf-8") != raw_prompt:
        raise ValueError("chat receipt raw request prompt differs from prompt binding")

    response = record.get("provider_response")
    if not isinstance(response, dict) or set(response) != {
        "parser",
        "encoding",
        "raw_bytes",
        "raw_sha256",
        "raw_base64",
    }:
        raise ValueError("chat receipt provider-response binding is malformed")
    if response.get("parser") != CHAT_RESPONSE_PARSER:
        raise ValueError("chat receipt response parser drifted")
    if response.get("encoding") != "base64-exact-decoded-http-body":
        raise ValueError("chat receipt response encoding drifted")
    encoded = response.get("raw_base64")
    if not isinstance(encoded, str):
        raise ValueError("chat receipt raw response must be base64 text")
    try:
        raw_response = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("chat receipt raw response is invalid base64") from None
    raw_bytes = response.get("raw_bytes")
    if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int) or raw_bytes <= 0:
        raise ValueError("chat receipt raw response byte count must be positive")
    if raw_bytes != len(raw_response):
        raise ValueError("chat receipt raw response byte count is inconsistent")
    if response.get("raw_sha256") != hashlib.sha256(raw_response).hexdigest():
        raise ValueError("chat receipt raw response digest is inconsistent")

    transport = record.get("transport")
    if not isinstance(transport, dict) or set(transport) != {
        "endpoint_url",
        "method",
        "content_type",
        "accept_encoding",
        "attempts",
        "latency_ms",
        "latency_source",
    }:
        raise ValueError("chat receipt transport binding is malformed")
    if (
        transport.get("method") != "POST"
        or transport.get("content_type") != "application/json"
        or transport.get("accept_encoding") != "identity"
    ):
        raise ValueError("chat receipt HTTP request transport drifted")
    if transport.get("latency_source") != "caller-observed-monotonic-clock":
        raise ValueError("chat receipt latency source drifted")
    replayed = replay_chat_response(raw_response)
    return ChatResult(
        content=replayed.content,
        prompt_tokens=replayed.prompt_tokens,
        completion_tokens=replayed.completion_tokens,
        total_tokens=replayed.total_tokens,
        finish_reason=replayed.finish_reason,
        attempts=transport.get("attempts"),
        latency_ms=transport.get("latency_ms"),
        response_model=replayed.response_model,
        request_id=replayed.request_id,
        system_fingerprint=replayed.system_fingerprint,
        prompt_sha256=prompt["sha256"],
        prompt_utf8_bytes=prompt_bytes,
        request_parser=CHAT_REQUEST_PARSER,
        response_parser=CHAT_RESPONSE_PARSER,
        endpoint_url=transport.get("endpoint_url"),
        prompt_bytes=raw_prompt,
        raw_request=raw_request,
        raw_response=raw_response,
    )


def load_chat_receipt_artifact(
    path: Path,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Load canonical JSONL receipts, reject duplicate routes, and replay every row."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read chat receipt artifact {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    routes: set[tuple[str, str]] = set()
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if not raw_line:
            raise ValueError(f"chat receipt artifact has an empty line at {line_number}")
        try:
            decoded = raw_line.decode("utf-8")
            record = json.loads(
                decoded,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_fields,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError):
            raise ValueError(f"chat receipt line {line_number} is malformed JSON") from None
        if _canonical_json_bytes(record) != raw_line:
            raise ValueError(f"chat receipt line {line_number} is not canonical JSON")
        validate_chat_receipt_record(record)
        route = (record["question_id"], record["call_role"])
        if route in routes:
            raise ValueError(f"chat receipt artifact repeats route {route!r}")
        routes.add(route)
        records.append(record)
    return artifact_identity(path), tuple(records)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _accuracy(outcomes: Sequence[QuestionOutcome]) -> dict[str, Any]:
    labels = [outcome.label for outcome in outcomes if outcome.label is not None]
    return {
        "questions": len(outcomes),
        "judged_questions": len(labels),
        "unjudged_questions": len(outcomes) - len(labels),
        "dev_judge_correct": sum(1 for value in labels if value),
        "dev_judge_accuracy": round(mean([1.0 if value else 0.0 for value in labels]), 4),
    }


def _retrieval_support(outcomes: Sequence[QuestionOutcome]) -> dict[str, Any]:
    """What the reader was actually handed, scored against the gold sessions.

    Recall here is over the bundle the reader saw, not the ranking: it is the
    ceiling on this run's accuracy, so a reader failure and a retrieval failure
    stay distinguishable in the same report.
    """

    with_gold = [outcome for outcome in outcomes if outcome.gold_keys]
    return {
        "mean_bundle_size": round(mean([float(len(o.retrieved_keys)) for o in outcomes]), 4),
        "empty_bundles": sum(1 for outcome in outcomes if not outcome.retrieved_keys),
        "any_gold_in_bundle": round(
            mean([1.0 if outcome.gold_hits else 0.0 for outcome in with_gold]), 4
        ),
        "all_gold_in_bundle": round(
            mean(
                [
                    1.0 if outcome.gold_hits == len(set(outcome.gold_keys)) else 0.0
                    for outcome in with_gold
                ]
            ),
            4,
        ),
        "recall_in_bundle": round(
            mean([outcome.gold_hits / len(set(outcome.gold_keys)) for outcome in with_gold]),
            4,
        ),
        "questions_with_gold": len(with_gold),
    }


def build_report(
    outcomes: Sequence[QuestionOutcome],
    *,
    metadata: dict[str, Any],
    hypothesis_path: Path,
    run_path: Path,
) -> dict[str, Any]:
    hypothesis_artifact = artifact_identity(hypothesis_path)
    run_artifact = artifact_identity(run_path)
    types = sorted({outcome.question_type for outcome in outcomes})
    prompt_tokens = [float(o.reader.prompt_tokens) for o in outcomes if o.reader is not None]
    completion_tokens = [
        float(o.reader.completion_tokens) for o in outcomes if o.reader is not None
    ]
    total_prompt = int(sum(prompt_tokens))
    total_completion = int(sum(completion_tokens))
    count = max(1, len(prompt_tokens))
    return {
        **metadata,
        "artifact_type": QA_REPORT_ARTIFACT_TYPE,
        "schema_version": QA_ARTIFACT_SCHEMA_VERSION,
        "protocol_version": QA_PROTOCOL_VERSION,
        "implementation": qa_implementation_fingerprint(),
        "hypotheses": hypothesis_artifact["path"],
        "hypothesis_artifact": hypothesis_artifact,
        "run": run_artifact["path"],
        "run_artifact": run_artifact,
        "overall": _accuracy(outcomes),
        "by_question_type": {
            question_type: _accuracy([o for o in outcomes if o.question_type == question_type])
            for question_type in types
        },
        "by_slice": {
            "abstention_questions": _accuracy([o for o in outcomes if o.abstention]),
            "answerable_questions": _accuracy([o for o in outcomes if not o.abstention]),
        },
        "retrieval_support": _retrieval_support(outcomes),
        "reader_tokens": {
            "prompt_total": total_prompt,
            "completion_total": total_completion,
            "prompt_mean": round(sum(prompt_tokens) / count, 1),
            "completion_mean": round(sum(completion_tokens) / count, 1),
            "prompt_percentiles": percentiles(prompt_tokens),
            "completion_percentiles": percentiles(completion_tokens),
        },
        "judge_tokens": {
            "prompt_total": int(sum(o.judge.prompt_tokens for o in outcomes if o.judge)),
            "completion_total": int(sum(o.judge.completion_tokens for o in outcomes if o.judge)),
        },
        "latency_ms": {
            "retrieval": percentiles([o.retrieval_ms for o in outcomes]),
            "reader": percentiles([o.reader.latency_ms for o in outcomes if o.reader is not None]),
            "question_total": percentiles([o.total_ms for o in outcomes]),
        },
        "failures": {
            "reader_failures": sum(1 for o in outcomes if o.reader_error),
            "judge_failures": sum(1 for o in outcomes if o.judge_error),
            "empty_hypotheses": sum(1 for o in outcomes if not o.hypothesis.strip()),
            "reader_retries": sum(
                (o.reader.attempts - 1) for o in outcomes if o.reader is not None
            ),
            "reader_truncated": sum(
                1 for o in outcomes if o.reader is not None and o.reader.finish_reason == "length"
            ),
        },
    }


def build_run(
    outcomes: Sequence[QuestionOutcome],
    *,
    metadata: dict[str, Any],
    hypothesis_path: Path,
    chat_receipt_path: Path,
) -> dict[str, Any]:
    hypothesis_artifact = artifact_identity(hypothesis_path)
    chat_receipt_artifact, chat_receipts = load_chat_receipt_artifact(chat_receipt_path)
    if chat_receipt_path.read_bytes() != chat_receipt_artifact_bytes(outcomes):
        raise ValueError("chat receipt artifact differs from the supplied QA outcomes")
    receipt_bindings = {
        (record["question_id"], record["call_role"]): {
            "index": index,
            "sha256": hashlib.sha256(_canonical_json_bytes(record)).hexdigest(),
        }
        for index, record in enumerate(chat_receipts)
    }
    return {
        **metadata,
        "artifact_type": QA_RUN_ARTIFACT_TYPE,
        "schema_version": QA_ARTIFACT_SCHEMA_VERSION,
        "protocol_version": QA_PROTOCOL_VERSION,
        "implementation": qa_implementation_fingerprint(),
        "hypotheses": hypothesis_artifact["path"],
        "hypothesis_artifact": hypothesis_artifact,
        "chat_receipt_artifact": chat_receipt_artifact,
        "chat_receipt_count": len(chat_receipts),
        "chat_receipt_protocol": CHAT_RECEIPT_PROTOCOL_VERSION,
        "questions": [
            {
                "question_id": outcome.question_id,
                "question_type": outcome.question_type,
                "abstention_question": outcome.abstention,
                "retrieved_session_keys": list(outcome.retrieved_keys),
                "retrieved_relevance": list(outcome.retrieved_relevance),
                "gold_session_keys": list(outcome.gold_keys),
                "gold_sessions_in_bundle": outcome.gold_hits,
                "hypothesis": outcome.hypothesis,
                "reader_prompt_tokens": outcome.reader.prompt_tokens if outcome.reader else None,
                "reader_completion_tokens": (
                    outcome.reader.completion_tokens if outcome.reader else None
                ),
                "reader_total_tokens": outcome.reader.total_tokens if outcome.reader else None,
                "reader_finish_reason": outcome.reader.finish_reason if outcome.reader else None,
                "reader_attempts": outcome.reader.attempts if outcome.reader else None,
                "reader_response_model": (
                    outcome.reader.response_model if outcome.reader else None
                ),
                "reader_request_id": outcome.reader.request_id if outcome.reader else None,
                "reader_system_fingerprint": (
                    outcome.reader.system_fingerprint if outcome.reader else None
                ),
                "reader_prompt_sha256": (outcome.reader.prompt_sha256 if outcome.reader else None),
                "reader_prompt_utf8_bytes": (
                    outcome.reader.prompt_utf8_bytes if outcome.reader else None
                ),
                "reader_raw_response_sha256": (
                    outcome.reader.raw_response_sha256 if outcome.reader else None
                ),
                "reader_raw_request_sha256": (
                    outcome.reader.raw_request_sha256 if outcome.reader else None
                ),
                "reader_receipt": receipt_bindings.get((outcome.question_id, "reader")),
                "reader_error": outcome.reader_error,
                "dev_judge_label": outcome.label,
                "dev_judge_response": outcome.judge_text,
                "dev_judge_prompt_tokens": outcome.judge.prompt_tokens if outcome.judge else None,
                "dev_judge_completion_tokens": (
                    outcome.judge.completion_tokens if outcome.judge else None
                ),
                "dev_judge_total_tokens": outcome.judge.total_tokens if outcome.judge else None,
                "dev_judge_response_model": (
                    outcome.judge.response_model if outcome.judge else None
                ),
                "dev_judge_request_id": outcome.judge.request_id if outcome.judge else None,
                "dev_judge_system_fingerprint": (
                    outcome.judge.system_fingerprint if outcome.judge else None
                ),
                "dev_judge_prompt_sha256": (outcome.judge.prompt_sha256 if outcome.judge else None),
                "dev_judge_prompt_utf8_bytes": (
                    outcome.judge.prompt_utf8_bytes if outcome.judge else None
                ),
                "dev_judge_raw_response_sha256": (
                    outcome.judge.raw_response_sha256 if outcome.judge else None
                ),
                "dev_judge_raw_request_sha256": (
                    outcome.judge.raw_request_sha256 if outcome.judge else None
                ),
                "dev_judge_receipt": receipt_bindings.get(
                    (outcome.question_id, "development_judge")
                ),
                "dev_judge_error": outcome.judge_error,
                "retrieval_ms": round(outcome.retrieval_ms, 3),
                "total_ms": round(outcome.total_ms, 3),
                "temporal_routing": outcome.temporal_routing,
            }
            for outcome in outcomes
        ],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _relative(path: Path) -> str:
    """Repo-relative when possible, so saved artifacts are location-independent."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "reader"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=10, help="sessions recalled per question")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help=(
            "calibrated relevance floor passed to RecallQuery; above 0.0 the bundle "
            "can come back thin or empty, which is surfaced to the reader"
        ),
    )
    parser.add_argument("--prompt-style", choices=("swarm", "official"), default="swarm")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--lme-path", type=Path, default=None)
    parser.add_argument("--lme-download", action="store_true")
    parser.add_argument("--lme-sample", type=int, default=50, help="0 or 500 runs every question")
    parser.add_argument("--lme-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--retrieval-run",
        type=Path,
        default=None,
        help=(
            "replay reader contexts from a publishable schema-v2 "
            "run_retrieval_eval.py artifact; skips all embedding and retrieval calls"
        ),
    )
    parser.add_argument(
        "--allow-nonpublishable-retrieval-run",
        action="store_true",
        help=(
            "development only: allow schema-v2 replay without the pinned Qwen semantic "
            "metadata/full-500 provider call evidence; official report compilation rejects it"
        ),
    )
    parser.add_argument(
        "--temporal-query-routing",
        action="store_true",
        help=(
            "experimental A/B: route only closed referenced-time parses against each "
            "question_date through the temporal retrieval lane"
        ),
    )
    parser.add_argument("--tag", default=None, help="extra filename tag, e.g. sample50")

    reader = parser.add_argument_group("reader")
    reader.add_argument("--reader-base-url", default=None, help="falls back to $OPENAI_BASE_URL")
    reader.add_argument("--reader-model", default=None, help="falls back to $MODEL")
    reader.add_argument(
        "--reader-api-key-env",
        default="OPENAI_API_KEY",
        help="environment variable holding the reader API key; never pass the key itself",
    )
    reader.add_argument("--reader-temperature", type=float, default=0.0)
    reader.add_argument("--reader-max-tokens", type=int, default=4096)
    reader.add_argument(
        "--reader-thinking-mode",
        choices=("provider-default", "enabled", "disabled"),
        default="provider-default",
        help=(
            "explicit OpenAI-compatible thinking toggle; provider-default omits the field. "
            "DeepSeek V4 defaults to enabled, where temperature is ignored"
        ),
    )
    reader.add_argument(
        "--reader-revision",
        default=None,
        help=(
            "immutable reader deployment/checkpoint revision; required before a full "
            "publishable replay starts"
        ),
    )
    reader.add_argument(
        "--allow-unverified-reader-response",
        action="store_true",
        help=(
            "development only: allow reader responses without an exact response model "
            "and provider request id; official report compilation rejects such runs"
        ),
    )
    reader.add_argument("--reader-tag", default=None, help="filename tag; defaults to the model")

    judge = parser.add_argument_group("development judge")
    judge.add_argument("--judge-base-url", default=None, help="defaults to the reader base URL")
    judge.add_argument("--judge-model", default=None, help="defaults to the reader model")
    judge.add_argument("--judge-api-key-env", default=None, help="defaults to the reader key env")
    judge.add_argument("--judge-max-tokens", type=int, default=1024)
    judge.add_argument(
        "--judge-thinking-mode",
        choices=("provider-default", "enabled", "disabled"),
        default="provider-default",
    )
    judge.add_argument(
        "--no-judge",
        action="store_true",
        help="generate hypotheses only, leaving all judging to the official script",
    )

    embeddings = parser.add_argument_group("dense lane")
    embeddings.add_argument("--embeddings", choices=("deterministic", "openai"), default="openai")
    embeddings.add_argument(
        "--embeddings-base-url",
        default=None,
        help="falls back to $SWARMBRAIN_EMBEDDINGS_BASE_URL",
    )
    embeddings.add_argument(
        "--embeddings-model", default=None, help="falls back to $SWARMBRAIN_EMBEDDINGS_MODEL"
    )
    embeddings.add_argument(
        "--embeddings-api-key-env",
        default="SWARMBRAIN_EMBEDDINGS_API_KEY",
        help="environment variable holding the embedding API key; never pass the key itself",
    )
    embeddings.add_argument(
        "--deterministic-embedder",
        action="store_true",
        help=(
            "escape hatch for when the embedding server is unreachable: run the dense "
            "lane on the hash embedder.  Retrieval quality collapses; never publish "
            "an accuracy measured this way"
        ),
    )
    embeddings.add_argument("--no-dense", action="store_true", help="lane ablation")
    return parser


def _resolve_reader(args: argparse.Namespace) -> ChatClient:
    base_url = args.reader_base_url or os.getenv("OPENAI_BASE_URL", "").strip()
    model = args.reader_model or os.getenv("MODEL", "").strip()
    if not base_url:
        raise SystemExit("set --reader-base-url or OPENAI_BASE_URL")
    if not model:
        raise SystemExit("set --reader-model or MODEL")
    return ChatClient(
        base_url=base_url,
        model=model,
        api_key=os.getenv(args.reader_api_key_env, "").strip() or None,
        temperature=args.reader_temperature,
        max_tokens=args.reader_max_tokens,
        required_response_model=(None if args.allow_unverified_reader_response else model),
        require_request_id=not args.allow_unverified_reader_response,
        thinking_mode=(
            None if args.reader_thinking_mode == "provider-default" else args.reader_thinking_mode
        ),
    )


def _resolve_judge(args: argparse.Namespace, reader: ChatClient) -> ChatClient | None:
    if args.no_judge:
        return None
    key_env = args.judge_api_key_env or args.reader_api_key_env
    return ChatClient(
        base_url=args.judge_base_url or reader.base_url,
        model=args.judge_model or reader.model,
        api_key=os.getenv(key_env, "").strip() or None,
        # The official judge runs at temperature 0 with max_tokens 10.  Ten
        # tokens is unusable against a reasoning model, which spends them all on
        # reasoning_content and returns empty content, so the allowance is the
        # only judge parameter that differs from the reference.
        temperature=0.0,
        max_tokens=args.judge_max_tokens,
        thinking_mode=(
            None if args.judge_thinking_mode == "provider-default" else args.judge_thinking_mode
        ),
    )


def _reader_requires_explicit_thinking_mode(reader: ChatClient) -> bool:
    hostname = (urlsplit(reader.base_url).hostname or "").casefold()
    return hostname == "api.deepseek.com" and reader.model in {
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    }


async def _main(args: argparse.Namespace) -> int:
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ensure_longmemeval(
        args.lme_path or default_longmemeval_path(), download=args.lme_download
    )
    everything: list[dict[str, Any]] = json.loads(dataset.read_text(encoding="utf-8"))
    sample = args.lme_sample if 0 < args.lme_sample < len(everything) else None
    records = select_questions(everything, sample=sample, seed=args.lme_seed)

    retrieval_cases: dict[str, dict[str, Any]] | None = None
    retrieval_payload: dict[str, Any] | None = None
    retrieval_source_artifact: dict[str, Any] | None = None
    retrieval_publishability: list[str] = []
    if args.allow_nonpublishable_retrieval_run and args.retrieval_run is None:
        raise SystemExit("--allow-nonpublishable-retrieval-run requires --retrieval-run")
    if args.retrieval_run is not None:
        if args.min_score != 0.0:
            raise SystemExit(
                "--retrieval-run requires --min-score 0.0; generate a dedicated live "
                "retrieval artifact to evaluate a positive relevance floor"
            )
        retrieval_cases, retrieval_payload = load_retrieval_run(
            args.retrieval_run,
            records,
            limit=args.limit,
            require_publishable=not args.allow_nonpublishable_retrieval_run,
        )
        retrieval_source_artifact = artifact_identity(args.retrieval_run)
        retrieval_publishability = retrieval_publishability_errors(retrieval_payload)
        source_temporal = bool(
            (retrieval_payload.get("temporal_query_routing") or {}).get("enabled")
        )
        if args.temporal_query_routing and not source_temporal:
            raise SystemExit(
                "--temporal-query-routing was requested but --retrieval-run did not use it"
            )
        use_dense = bool(retrieval_payload.get("dense_lane_enabled"))
    else:
        if args.deterministic_embedder:
            args.embeddings = "deterministic"
        configure_embeddings(
            args.embeddings,
            base_url=args.embeddings_base_url or os.getenv("SWARMBRAIN_EMBEDDINGS_BASE_URL"),
            model_id=args.embeddings_model or os.getenv("SWARMBRAIN_EMBEDDINGS_MODEL"),
            api_key=(
                os.getenv(args.embeddings_api_key_env, "").strip() or None
                if args.embeddings == "openai"
                else None
            ),
        )
        use_dense = not args.no_dense

    publishable_full_replay = (
        retrieval_payload is not None
        and not retrieval_publishability
        and len(records) == LONGMEMEVAL_S_QUESTION_COUNT
        and len(everything) == LONGMEMEVAL_S_QUESTION_COUNT
    )
    if (
        publishable_full_replay
        and not args.allow_unverified_reader_response
        and not (args.reader_revision or "").strip()
    ):
        raise SystemExit(
            "full publishable replay requires --reader-revision before reader calls begin"
        )

    reader = _resolve_reader(args)
    if (
        publishable_full_replay
        and _reader_requires_explicit_thinking_mode(reader)
        and reader.thinking_mode is None
    ):
        raise SystemExit(
            "full publishable DeepSeek V4 replay requires "
            "--reader-thinking-mode enabled or disabled; the provider default is mutable "
            "and thinking mode ignores temperature"
        )
    judge = _resolve_judge(args, reader)
    reader_tag = _slug(args.reader_tag or reader.model)
    tag = args.tag or (f"sample{len(records)}" if sample is not None else None)
    if args.temporal_query_routing:
        tag = f"{tag}-temporal" if tag else "temporal"
    if args.retrieval_run is not None:
        tag = f"{tag}-replay" if tag else "replay"
    stem = f"longmemeval-s-qa-{reader_tag}" + (f"-{_slug(tag)}" if tag else "")

    started_at = datetime.now(UTC)
    clock = perf_counter()
    try:
        outcomes = await run_questions(
            records,
            reader=reader,
            judge=judge,
            limit=args.limit,
            min_score=args.min_score,
            use_dense=use_dense,
            prompt_style=args.prompt_style,
            concurrency=args.concurrency,
            temporal_query_routing=args.temporal_query_routing,
            retrieval_cases=retrieval_cases,
        )
    finally:
        await reader.aclose()
        if judge is not None and judge is not reader:
            await judge.aclose()
    wall_seconds = perf_counter() - clock

    metadata: dict[str, Any] = {
        "run_id": str(uuid4()),
        "harness": "scripts/run_longmemeval_qa.py",
        "task": "longmemeval-s end-to-end QA",
        "started_at": started_at.isoformat(),
        "wall_seconds": round(wall_seconds, 3),
        "dataset": {
            "name": "LongMemEval-S",
            "source": LONGMEMEVAL_S_URL,
            "sha256": LONGMEMEVAL_S_SHA256,
            "artifact": artifact_identity(dataset),
            "total_questions": len(everything),
            "evaluated_questions": len(records),
            "sample_seed": args.lme_seed if sample is not None else None,
        },
        "retrieval": (
            {
                "mode": "replayed_saved_run",
                "source_run": _relative(args.retrieval_run),
                "source_artifact": retrieval_source_artifact,
                "source_artifact_type": retrieval_payload.get("artifact_type"),
                "source_schema_version": retrieval_payload.get("schema_version"),
                "source_protocol_version": retrieval_payload.get("protocol_version"),
                "source_implementation": retrieval_payload.get("implementation"),
                "source_publishable": not retrieval_publishability,
                "source_publishability_errors": retrieval_publishability,
                "granularity": retrieval_payload.get("granularity"),
                "source_limit": retrieval_payload.get("recall_limit"),
                "source_saved_ranking_depth": retrieval_payload.get("saved_ranking_depth"),
                "limit": args.limit,
                "min_score": args.min_score,
                "dense_lane_enabled": use_dense,
                "embedding": retrieval_payload.get("embedding"),
                "source_embedding_call_accounting": retrieval_payload.get(
                    "embedding_call_accounting"
                ),
                "replay_embedding_call_accounting": {
                    "document_inputs": 0,
                    "document_batch_calls": 0,
                    "query_calls": 0,
                    "successful_http_calls": 0,
                    "source": "artifact-replay-no-provider-calls",
                },
                "temporal_query_routing": retrieval_payload.get("temporal_query_routing"),
            }
            if retrieval_payload is not None and args.retrieval_run is not None
            else {
                "mode": "live_retrieval",
                "granularity": "one memory per haystack session",
                "limit": args.limit,
                "min_score": args.min_score,
                "dense_lane_enabled": use_dense,
                "embedding": embedding_metadata(use_dense),
                "embedding_call_accounting": (
                    {
                        "document_inputs": sum(
                            len(record["haystack_sessions"]) for record in records
                        ),
                        "document_batch_calls": len(records),
                        "query_calls": len(records),
                    }
                    if use_dense
                    else None
                ),
                "temporal_query_routing": {
                    "enabled": args.temporal_query_routing,
                    "parser": (
                        TEMPORAL_QUERY_PARSER_VERSION if args.temporal_query_routing else None
                    ),
                    "session_valid_from": "LongMemEval haystack_dates normalized to UTC",
                },
            }
        ),
        "reader": {
            "model": reader.model,
            "revision": (args.reader_revision or "").strip() or None,
            "revision_source": "operator-pinned deployment/checkpoint",
            "response_model_requirement": reader.required_response_model,
            "request_id_required": reader.require_request_id,
            "response_parser": CHAT_RESPONSE_PARSER,
            "request_parser": CHAT_REQUEST_PARSER,
            "raw_request_receipts_required": True,
            "raw_prompt_receipts_required": True,
            "raw_response_receipts_required": True,
            "provider_usage_replay_required": True,
            "response_evidence_publishable": bool(
                reader.required_response_model == reader.model
                and reader.require_request_id
                and (args.reader_revision or "").strip()
                and not (
                    _reader_requires_explicit_thinking_mode(reader) and reader.thinking_mode is None
                )
            ),
            "base_url": reader.base_url,
            "temperature": reader.temperature,
            "max_tokens": reader.max_tokens,
            "thinking_mode": reader.thinking_mode,
            "thinking_mode_source": (
                "explicit-request-field"
                if reader.thinking_mode is not None
                else "provider-default-omitted"
            ),
            "prompt_style": args.prompt_style,
            "prompt_template_source": (
                "LongMemEval src/generation/run_generation.py, flat-session + nl, no CoT"
                + ("" if args.prompt_style == "official" else ", plus one evidence-only sentence")
            ),
            "concurrency": args.concurrency,
        },
        "judge": {
            "role": "DEVELOPMENT JUDGE — NOT the official LongMemEval judge",
            "model": judge.model if judge else None,
            "base_url": judge.base_url if judge else None,
            "max_tokens": judge.max_tokens if judge else None,
            "thinking_mode": judge.thinking_mode if judge else None,
            "prompts": (
                "verbatim get_anscheck_prompt from "
                f"{OFFICIAL_REPO}/blob/main/src/evaluation/evaluate_qa.py"
            ),
            "official_judge_model": OFFICIAL_JUDGE_MODEL,
            "official_judge_command": (
                f"python src/evaluation/evaluate_qa.py gpt-4o <{stem}-hypotheses.jsonl> "
                "<longmemeval_s_cleaned.json>"
            ),
            "warning": (
                "every accuracy in this file is a dev-judge accuracy and must never be "
                "reported as a LongMemEval score"
            ),
        },
    }

    hypothesis_path = args.out_dir / f"{stem}-hypotheses.jsonl"
    chat_receipt_path = args.out_dir / f"{stem}-chat-receipts.jsonl"
    run_path = args.out_dir / f"{stem}-run.json"
    report_path = args.out_dir / f"{stem}-report.json"

    hypothesis_path.write_text(
        "".join(
            hypothesis_line(outcome.question_id, outcome.hypothesis) + "\n" for outcome in outcomes
        ),
        encoding="utf-8",
    )
    write_chat_receipts(outcomes, chat_receipt_path)
    run_path.write_text(
        json.dumps(
            build_run(
                outcomes,
                metadata=metadata,
                hypothesis_path=hypothesis_path,
                chat_receipt_path=chat_receipt_path,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_report(
        outcomes, metadata=metadata, hypothesis_path=hypothesis_path, run_path=run_path
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {hypothesis_path}, {chat_receipt_path}, {run_path} and {report_path}")
    print(
        "dev-judge accuracy: "
        f"{report['overall']['dev_judge_accuracy']} "
        f"over {report['overall']['judged_questions']} judged questions "
        f"({report['failures']['reader_failures']} reader failures, "
        f"{report['failures']['judge_failures']} judge failures)"
    )
    return 0


def main() -> int:
    return asyncio.run(_main(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHAT_RECEIPT_ARTIFACT_TYPE",
    "CHAT_RECEIPT_PROTOCOL_VERSION",
    "CHAT_RECEIPT_SCHEMA_VERSION",
    "CHAT_REQUEST_PARSER",
    "CHAT_RESPONSE_PARSER",
    "CHAT_CALL_ROLES",
    "ChatClient",
    "ChatProtocolError",
    "ChatResult",
    "ChatUnavailable",
    "QuestionOutcome",
    "ReplayedChatRequest",
    "ReplayedChatResponse",
    "QA_ARTIFACT_SCHEMA_VERSION",
    "QA_PROTOCOL_VERSION",
    "QA_REPORT_ARTIFACT_TYPE",
    "QA_RUN_ARTIFACT_TYPE",
    "RETRIEVAL_ARTIFACT_SCHEMA_VERSION",
    "RETRIEVAL_PROTOCOL_VERSION",
    "RETRIEVAL_RUN_ARTIFACT_TYPE",
    "answer_question",
    "artifact_identity",
    "build_reader_prompt",
    "build_report",
    "build_run",
    "chat_receipt_artifact_bytes",
    "chat_receipt_record",
    "chat_receipt_records",
    "chat_request_bytes",
    "chat_result_from_raw_response",
    "hypothesis_line",
    "is_abstention_question",
    "judge_label",
    "judge_prompt",
    "load_chat_receipt_artifact",
    "render_history",
    "render_session",
    "replay_chat_request",
    "retrieval_publishability_errors",
    "run_questions",
    "validate_implementation_fingerprint",
    "validate_chat_receipt_record",
    "validate_retrieval_run_protocol",
    "write_chat_receipts",
]
