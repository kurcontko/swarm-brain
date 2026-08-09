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
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

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
    retrieve_question,
    select_questions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "benchmarks" / "retrieval"

# Same default seed as the retrieval harness, so ``--lme-sample 50`` selects the
# same 50 questions here as there and the QA numbers sit on top of recall
# numbers measured over the identical subset.
DEFAULT_SAMPLE_SEED = 20260807

OFFICIAL_JUDGE_MODEL = "gpt-4o-2024-08-06"
OFFICIAL_REPO = "https://github.com/xiaowu0162/LongMemEval"


# --------------------------------------------------------------------------- #
# reader prompt — reference templates, verbatim
# --------------------------------------------------------------------------- #

# `run_generation.py`, retriever_type='flat-session', merge_key_expansion='none',
# cot=false.  Reproduced exactly, including the triple newline.
OFFICIAL_ANSWER_TEMPLATE = (
    "I will give you several history chats between you and a user. Please answer "
    "the question based on the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\n"
    "Current Date: {}\nQuestion: {}\nAnswer:"
)

# The house variant: the reference instruction plus one evidence-only sentence.
# The abstention wire from docs/sota-plan.md §3 lives in that sentence together
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
EMPTY_CONTEXT_NOTE = (
    "(No stored session passed the memory relevance threshold for this question. "
    "The memory returned nothing relevant.)"
)

THIN_CONTEXT_NOTE = (
    "Note: only {kept} of the {requested} requested sessions passed the memory "
    "relevance threshold; everything else in memory was judged irrelevant to this "
    "question.\n"
)


def render_session(index: int, date: str, turns: Sequence[dict[str, Any]]) -> str:
    """One session block, in the reference ``flat-session`` + ``nl`` format."""

    body = ""
    for turn in turns:
        role = turn.get("role", "user")
        content = str(turn.get("content", "")).strip()
        body += f"\n\n{role}: {content}"
    return f"\n### Session {index + 1}:\nSession Date: {date}\nSession Content:\n{body}\n"


def render_history(sessions: Sequence[tuple[str, Sequence[dict[str, Any]]]]) -> str:
    """Render retrieved sessions chronologically, as the reference harness does.

    LongMemEval haystack dates are ``YYYY/MM/DD (Day) HH:MM``, which sorts
    lexicographically in chronological order — this is the reference script's
    ``retrieved_chunks.sort(key=lambda x: x[0])``.  The sort is stable, so
    same-dated sessions keep retrieval rank order.  Ordering by date rather than
    by rank is what makes temporal questions and knowledge-update supersession
    readable: the newest statement is last.
    """

    ordered = sorted(sessions, key=lambda item: item[0])
    return "".join(
        render_session(index, date, turns) for index, (date, turns) in enumerate(ordered)
    )


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
            history = (
                THIN_CONTEXT_NOTE.format(kept=len(sessions), requested=requested) + history
            )
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
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    attempts: int
    latency_ms: float


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
    ) -> None:
        stripped = base_url.strip().rstrip("/")
        if not stripped.startswith(("http://", "https://")):
            raise ValueError("chat base URL must be an http:// or https:// URL")
        if stripped.endswith("/v1"):
            stripped = stripped[: -len("/v1")]
        self.base_url = stripped
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._attempts = attempts
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def complete(self, prompt: str) -> ChatResult:
        client = self._ensure_client()
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "n": 1,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        started = perf_counter()
        last: str = "no attempt was made"
        for attempt in range(self._attempts):
            try:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
            except Exception as exc:  # transport fault: retry
                last = f"{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if status in _RETRYABLE_STATUS:
                    last = f"HTTP {status}"
                elif 400 <= status < 500:
                    raise ChatProtocolError(
                        f"{self.base_url} rejected the request with HTTP {status}: "
                        f"{response.text[:400]}"
                    )
                else:
                    body = response.json()
                    choices = body.get("choices")
                    if not isinstance(choices, list) or not choices:
                        raise ChatProtocolError("chat response carried no choices")
                    message = choices[0].get("message") or {}
                    content = str(message.get("content") or "").strip()
                    finish = str(choices[0].get("finish_reason") or "")
                    usage = body.get("usage") or {}
                    if content:
                        return ChatResult(
                            content=content,
                            prompt_tokens=int(usage.get("prompt_tokens") or 0),
                            completion_tokens=int(usage.get("completion_tokens") or 0),
                            finish_reason=finish,
                            attempts=attempt + 1,
                            latency_ms=(perf_counter() - started) * 1000.0,
                        )
                    # Empty content with finish_reason 'length' is the reasoning
                    # budget having eaten the whole allowance.
                    last = f"empty content (finish_reason={finish!r})"
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


async def answer_question(
    record: dict[str, Any],
    *,
    reader: ChatClient,
    judge: ChatClient | None,
    limit: int,
    min_score: float,
    use_dense: bool,
    prompt_style: str,
) -> QuestionOutcome:
    started = perf_counter()
    retrieved = await retrieve_question(
        record, limit=limit, min_score=min_score, use_dense=use_dense
    )
    hits: tuple[tuple[SessionMemory, float], ...] = retrieved.retrieved_sessions()
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
        gold_keys=retrieved.relevant_keys,
        hypothesis=hypothesis,
        reader=result,
        reader_error=reader_error,
        judge=verdict,
        judge_error=judge_error,
        label=label,
        retrieval_ms=retrieved.wall_ms,
        total_ms=(perf_counter() - started) * 1000.0,
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
            )
        done += 1
        if done % 10 == 0 or done == total:
            print(f"  qa: {done}/{total} questions", file=sys.stderr, flush=True)
        return outcome

    return list(await asyncio.gather(*(one(record) for record in records)))


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
            mean(
                [outcome.gold_hits / len(set(outcome.gold_keys)) for outcome in with_gold]
            ),
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
        "hypotheses": _relative(hypothesis_path),
        "run": _relative(run_path),
        "overall": _accuracy(outcomes),
        "by_question_type": {
            question_type: _accuracy(
                [o for o in outcomes if o.question_type == question_type]
            )
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
            "reader": percentiles(
                [o.reader.latency_ms for o in outcomes if o.reader is not None]
            ),
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
) -> dict[str, Any]:
    return {
        **metadata,
        "hypotheses": _relative(hypothesis_path),
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
                "reader_finish_reason": outcome.reader.finish_reason if outcome.reader else None,
                "reader_attempts": outcome.reader.attempts if outcome.reader else None,
                "reader_error": outcome.reader_error,
                "dev_judge_label": outcome.label,
                "dev_judge_response": outcome.judge_text,
                "dev_judge_prompt_tokens": outcome.judge.prompt_tokens if outcome.judge else None,
                "dev_judge_completion_tokens": (
                    outcome.judge.completion_tokens if outcome.judge else None
                ),
                "dev_judge_error": outcome.judge_error,
                "retrieval_ms": round(outcome.retrieval_ms, 3),
                "total_ms": round(outcome.total_ms, 3),
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument(
        "--lme-sample", type=int, default=50, help="0 or 500 runs every question"
    )
    parser.add_argument("--lme-seed", type=int, default=DEFAULT_SAMPLE_SEED)
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
    reader.add_argument("--reader-tag", default=None, help="filename tag; defaults to the model")

    judge = parser.add_argument_group("development judge")
    judge.add_argument("--judge-base-url", default=None, help="defaults to the reader base URL")
    judge.add_argument("--judge-model", default=None, help="defaults to the reader model")
    judge.add_argument("--judge-api-key-env", default=None, help="defaults to the reader key env")
    judge.add_argument("--judge-max-tokens", type=int, default=1024)
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
    )


async def _main(args: argparse.Namespace) -> int:
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.deterministic_embedder:
        args.embeddings = "deterministic"
    configure_embeddings(
        args.embeddings,
        base_url=args.embeddings_base_url or os.getenv("SWARMBRAIN_EMBEDDINGS_BASE_URL"),
        model_id=args.embeddings_model or os.getenv("SWARMBRAIN_EMBEDDINGS_MODEL"),
    )
    use_dense = not args.no_dense

    dataset = ensure_longmemeval(
        args.lme_path or default_longmemeval_path(), download=args.lme_download
    )
    everything: list[dict[str, Any]] = json.loads(dataset.read_text(encoding="utf-8"))
    sample = args.lme_sample if 0 < args.lme_sample < len(everything) else None
    records = select_questions(everything, sample=sample, seed=args.lme_seed)

    reader = _resolve_reader(args)
    judge = _resolve_judge(args, reader)
    reader_tag = _slug(args.reader_tag or reader.model)
    tag = args.tag or (f"sample{len(records)}" if sample is not None else None)
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
        )
    finally:
        await reader.aclose()
        if judge is not None and judge is not reader:
            await judge.aclose()
    wall_seconds = perf_counter() - clock

    metadata: dict[str, Any] = {
        "harness": "scripts/run_longmemeval_qa.py",
        "task": "longmemeval-s end-to-end QA",
        "started_at": started_at.isoformat(),
        "wall_seconds": round(wall_seconds, 3),
        "dataset": {
            "name": "LongMemEval-S",
            "source": LONGMEMEVAL_S_URL,
            "sha256": LONGMEMEVAL_S_SHA256,
            "total_questions": len(everything),
            "evaluated_questions": len(records),
            "sample_seed": args.lme_seed if sample is not None else None,
        },
        "retrieval": {
            "granularity": "one memory per haystack session",
            "limit": args.limit,
            "min_score": args.min_score,
            "dense_lane_enabled": use_dense,
            "embedding": embedding_metadata(use_dense),
        },
        "reader": {
            "model": reader.model,
            "base_url": reader.base_url,
            "temperature": reader.temperature,
            "max_tokens": reader.max_tokens,
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
            "prompts": (
                "verbatim get_anscheck_prompt from "
                f"{OFFICIAL_REPO}/blob/main/src/evaluation/evaluate_qa.py"
            ),
            "official_judge_model": OFFICIAL_JUDGE_MODEL,
            "official_judge_command": (
                f"python src/evaluation/evaluate_qa.py gpt-4o <{stem}-hypotheses.jsonl> "
                "<longmemeval_s.json>"
            ),
            "warning": (
                "every accuracy in this file is a dev-judge accuracy and must never be "
                "reported as a LongMemEval score"
            ),
        },
    }

    hypothesis_path = args.out_dir / f"{stem}-hypotheses.jsonl"
    run_path = args.out_dir / f"{stem}-run.json"
    report_path = args.out_dir / f"{stem}-report.json"

    hypothesis_path.write_text(
        "".join(
            hypothesis_line(outcome.question_id, outcome.hypothesis) + "\n"
            for outcome in outcomes
        ),
        encoding="utf-8",
    )
    run_path.write_text(
        json.dumps(
            build_run(outcomes, metadata=metadata, hypothesis_path=hypothesis_path),
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

    print(f"wrote {hypothesis_path}, {run_path} and {report_path}")
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
    "ChatClient",
    "ChatProtocolError",
    "ChatResult",
    "ChatUnavailable",
    "QuestionOutcome",
    "answer_question",
    "build_reader_prompt",
    "build_report",
    "build_run",
    "hypothesis_line",
    "is_abstention_question",
    "judge_label",
    "judge_prompt",
    "render_history",
    "render_session",
    "run_questions",
]
