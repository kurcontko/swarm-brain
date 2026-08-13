"""Shared LongMemEval-S machinery for the retrieval and QA harnesses.

Both ``scripts/run_retrieval_eval.py`` (retrieval-only, Track 2) and
``scripts/run_longmemeval_qa.py`` (end-to-end QA) ingest the same pinned
dataset the same way and recall through the same real ``RetrievalService``
path.  That ingestion + recall is defined once, here, so a change to the
mapping can never make the two harnesses measure different systems.

Nothing in this module talks to a reader or a judge; it stops at the retrieved
bundle.  Mapping decisions (one memory per haystack session, a fresh runtime
per question, the haystack date in the title) are documented in
``docs/retrieval-benchmark.md`` under "Dataset and ingestion mapping".
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from swarmbrain.adapters.embeddings import DeterministicEmbeddingProvider
from swarmbrain.adapters.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from swarmbrain.adapters.extraction.in_memory import InMemoryWorkStore
from swarmbrain.adapters.memory import (
    InMemoryKernel,
    in_memory_hybrid_retrieval_gateways,
    in_memory_retrieval_gateways,
)
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy, memory_content_text
from swarmbrain.application.memory_service import MemoryService
from swarmbrain.application.retrieval_service import RetrievalExecution, RetrievalService
from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.memory import (
    EmbeddingVector,
    Memory,
    MemoryState,
    RecallQuery,
    RememberCommand,
    Visibility,
)
from swarmbrain.domain.reranking import LearnedRerankPolicy
from swarmbrain.domain.retrieval import DenseQuery, RetrievalPurpose, RetrievalSignal
from swarmbrain.ports.embeddings import EmbeddingProvider
from swarmbrain.ports.reranking import LearnedRerankerProvider
from swarmbrain.retrieval.temporal_query import TemporalQueryParse, parse_referenced_time

EMBEDDING_MODEL = "deterministic-eval-1024-v0"
EMBEDDING_DIMENSIONS = 1024
QWEN_QUERY_INSTRUCTION = "Given a coding-agent memory search query, retrieve relevant memories"
ALL_CAPABILITIES = frozenset(item.value for item in Capability)

# Pinned current official cleaned release.  LongMemEval replaced the original
# histories in September 2025 while preserving the 500 question IDs and answer
# keys.  The digest is the git-lfs object id of the file, which is its SHA-256,
# so a silent upstream replacement fails loudly here.
LONGMEMEVAL_S_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_s_cleaned.json"
)
LONGMEMEVAL_S_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"

# Official ``flat-session`` + ``nl`` reader serialization. Retrieval token
# evidence and QA generation import this single definition so "exact context"
# always means the bytes that the reader actually receives.
OFFICIAL_READER_SERIALIZER_VERSION = "longmemeval-flat-session-nl-official-prompt-v1"
OFFICIAL_ANSWER_TEMPLATE = (
    "I will give you several history chats between you and a user. Please answer "
    "the question based on the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\n"
    "Current Date: {}\nQuestion: {}\nAnswer:"
)
EMPTY_CONTEXT_NOTE = (
    "(No stored session passed the memory relevance threshold for this question. "
    "The memory returned nothing relevant.)"
)

# The clock every LongMemEval runtime starts from.  Fixed so ingestion order,
# and therefore recency tie-breaking, is identical on every run and in both
# harnesses.
LONGMEMEVAL_CLOCK_START = datetime(2026, 8, 7, 9, tzinfo=UTC)

# LongMemEval dates carry a local-looking wall clock but no timezone.  The
# benchmark treats them as one internally consistent calendar, so UTC is an
# explicit normalization convention here rather than an inference about the
# user's real location.  Weekday validation catches corrupt or silently
# reformatted dataset rows before they can affect temporal routing.
_LONGMEMEVAL_DATETIME = re.compile(
    r"(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2}) "
    r"\((?P<weekday>Mon|Tue|Wed|Thu|Fri|Sat|Sun)\) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2})"
)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Selected once per process from ``--embeddings``; the deterministic default
# keeps the original 2026-08-07 runs byte-reproducible.  A single shared
# instance serves every composition site so the openai provider reuses one
# HTTP client across the whole run.
_provider_singleton: EmbeddingProvider | None = None


def configure_embeddings(
    kind: str,
    *,
    base_url: str | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
) -> None:
    global _provider_singleton
    if kind == "deterministic":
        _provider_singleton = DeterministicEmbeddingProvider(
            dimensions=EMBEDDING_DIMENSIONS,
            model_name=EMBEDDING_MODEL,
        )
        return
    if base_url is None:
        raise SystemExit("--embeddings openai requires --embeddings-base-url")
    _provider_singleton = OpenAICompatibleEmbeddingProvider(
        base_url=base_url,
        model_id=model_id or "Qwen/Qwen3-Embedding-0.6B",
        dimensions=EMBEDDING_DIMENSIONS,
        api_key=api_key,
        required_response_model=model_id or "Qwen/Qwen3-Embedding-0.6B",
        query_instruction=QWEN_QUERY_INSTRUCTION,
        # Qwen3-Embedding-0.6B serves an 8192-token window; a handful of
        # LongMemEval haystack sessions exceed it, and vLLM 400s instead of
        # truncating unless the request opts in.
        truncate_prompt_tokens=8192,
    )


def make_provider() -> EmbeddingProvider:
    if _provider_singleton is None:
        configure_embeddings("deterministic")
    assert _provider_singleton is not None
    return _provider_singleton


def embedding_metadata(use_dense: bool) -> dict[str, Any] | None:
    if not use_dense:
        return None
    provider = make_provider()
    deterministic = isinstance(provider, DeterministicEmbeddingProvider)
    metadata = {
        "provider": type(provider).__name__,
        "model": provider.model_name,
        "dimensions": provider.dimensions,
        "document_batching": "one question-local corpus per embed_documents call",
        "note": (
            "hash bag-of-words; not a semantic model"
            if deterministic
            else "semantic model served over an OpenAI-compatible endpoint"
        ),
    }
    if isinstance(provider, OpenAICompatibleEmbeddingProvider):
        metadata["response_model_requirement"] = provider.required_response_model
        metadata["query_instruction_sha256"] = hashlib.sha256(
            QWEN_QUERY_INSTRUCTION.encode("utf-8")
        ).hexdigest()
    return metadata


def reset_embedding_call_accounting() -> None:
    provider = make_provider()
    if isinstance(provider, OpenAICompatibleEmbeddingProvider):
        provider.reset_call_accounting()


def observed_embedding_call_accounting() -> dict[str, int] | None:
    provider = make_provider()
    if not isinstance(provider, OpenAICompatibleEmbeddingProvider):
        return None
    return provider.call_accounting


# --------------------------------------------------------------------------- #
# scope and clock
# --------------------------------------------------------------------------- #


def scope_ids(seed: str | None) -> dict[str, str]:
    names = ("tenant_id", "project_id", "repository_id", "swarm_id", "run_id")
    if seed is None:
        return {name: str(uuid4()) for name in names}
    return {name: str(uuid5(NAMESPACE_URL, f"swarmbrain-eval/{seed}/{name}")) for name in names}


def make_actor(scope: dict[str, str], *, agent_seed: str | None = None) -> ActorContext:
    agent_id = (
        str(uuid4())
        if agent_seed is None
        else str(uuid5(NAMESPACE_URL, f"swarmbrain-eval/{agent_seed}/agent"))
    )
    return ActorContext(
        **scope,
        agent_id=agent_id,
        harness="swarmbrain-eval",
        provider="local",
        model="none",
        capabilities=ALL_CAPABILITIES,
    )


class SteppingClock:
    """Deterministic clock advanced explicitly between publishes."""

    def __init__(self, start: datetime) -> None:
        self.value = start

    def __call__(self) -> datetime:
        return self.value

    def step(self, seconds: int = 1) -> None:
        self.value = self.value + timedelta(seconds=seconds)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


def default_longmemeval_path() -> Path:
    """Cache the 278 MB release outside the repository; it must never be committed."""

    root = os.getenv("SWARMBRAIN_EVAL_DATA_DIR", "").strip()
    base = Path(root) if root else Path.home() / ".cache" / "swarmbrain-eval"
    return base / "longmemeval_s_cleaned.json"


def ensure_longmemeval(path: Path, *, download: bool) -> Path:
    """Return a verified local copy of the pinned LongMemEval-S release."""

    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"{path} is missing; rerun with --lme-download to fetch the pinned release"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(LONGMEMEVAL_S_URL) as response, path.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    if digest.hexdigest() != LONGMEMEVAL_S_SHA256:
        raise ValueError(f"{path} does not match the pinned LongMemEval-S digest")
    return path


def select_questions(
    records: Sequence[dict[str, Any]],
    *,
    sample: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Seeded subset in dataset order, so a sample is reproducible and stable.

    The same ``(sample, seed)`` picks the same questions here as in the
    retrieval harness, which is what makes a QA sample comparable with the
    recall numbers already published for that sample.
    """

    total = len(records)
    if sample is None or sample >= total:
        return list(records)
    chosen = random.Random(seed).sample(range(total), sample)
    chosen.sort()
    return [records[index] for index in chosen]


def session_text(turns: Sequence[dict[str, Any]]) -> str:
    return "\n".join(f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in turns)


def render_reader_session(index: int, date: str, turns: Sequence[dict[str, Any]]) -> str:
    """Serialize one session exactly as the official LongMemEval reader harness."""

    body = ""
    for turn in turns:
        role = turn.get("role", "user")
        content = str(turn.get("content", "")).strip()
        body += f"\n\n{role}: {content}"
    return f"\n### Session {index + 1}:\nSession Date: {date}\nSession Content:\n{body}\n"


def render_reader_history(
    sessions: Sequence[tuple[str, Sequence[dict[str, Any]]]],
) -> str:
    """Serialize retrieved sessions in the official chronological order."""

    ordered = sorted(sessions, key=lambda item: item[0])
    return "".join(
        render_reader_session(index, date, turns) for index, (date, turns) in enumerate(ordered)
    )


def build_official_reader_prompt(
    record: dict[str, Any],
    sessions: Sequence[tuple[str, Sequence[dict[str, Any]]]],
) -> str:
    """Return the exact official reader input for one selected session set."""

    history = render_reader_history(sessions) if sessions else EMPTY_CONTEXT_NOTE
    return OFFICIAL_ANSWER_TEMPLATE.format(
        history,
        str(record["question_date"]),
        str(record["question"]),
    )


def parse_longmemeval_datetime(value: str) -> datetime:
    """Parse the dataset's timestamp format into the benchmark UTC calendar."""

    match = _LONGMEMEVAL_DATETIME.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"unsupported LongMemEval datetime: {value!r}")
    parsed = datetime(
        int(match.group("year")),
        int(match.group("month")),
        int(match.group("day")),
        int(match.group("hour")),
        int(match.group("minute")),
        tzinfo=UTC,
    )
    expected_weekday = _WEEKDAYS[parsed.weekday()]
    if match.group("weekday") != expected_weekday:
        raise ValueError(f"LongMemEval datetime weekday mismatch: {value!r} is {expected_weekday}")
    return parsed


def parse_longmemeval_temporal_query(record: dict[str, Any]) -> TemporalQueryParse:
    """Parse referenced time using only the record's explicit question date."""

    anchor = parse_longmemeval_datetime(str(record["question_date"]))
    return parse_referenced_time(str(record["question"]), timezone=UTC, now=anchor)


def temporal_parse_metadata(parsed: TemporalQueryParse | None) -> dict[str, Any] | None:
    """Content-free, replayable parser trace for benchmark A/B artifacts."""

    if parsed is None:
        return None
    closed = parsed.closed_referenced_time
    interval = parsed.interval
    return {
        "status": parsed.status.value,
        "confidence": parsed.confidence.value,
        "reason": parsed.reason.value,
        "relative": parsed.relative,
        "routed": closed is not None,
        "valid_from": interval.valid_from.isoformat() if interval and interval.valid_from else None,
        "valid_to": interval.valid_to.isoformat() if interval and interval.valid_to else None,
    }


# --------------------------------------------------------------------------- #
# ingestion and recall, one question at a time
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SessionMemory:
    """One haystack session, as it was stored and as a reader would see it."""

    position: int
    session_id: str
    date: str
    turns: tuple[dict[str, Any], ...]
    memory_id: str

    @property
    def key(self) -> str:
        """Judgment key: session ids repeat inside 15 haystacks, positions do not."""

        return f"{self.position:03d}:{self.session_id}"


@dataclass(frozen=True, slots=True)
class QuestionRetrieval:
    record: dict[str, Any]
    sessions: tuple[SessionMemory, ...]
    execution: RetrievalExecution
    wall_ms: float
    temporal_parse: TemporalQueryParse | None = None

    @property
    def question_id(self) -> str:
        return str(self.record["question_id"])

    @property
    def by_memory_id(self) -> dict[str, SessionMemory]:
        return {session.memory_id: session for session in self.sessions}

    @property
    def key_by_memory_id(self) -> dict[str, str]:
        return {session.memory_id: session.key for session in self.sessions}

    @property
    def relevant_keys(self) -> tuple[str, ...]:
        answers = {str(value) for value in self.record.get("answer_session_ids", ())}
        return tuple(session.key for session in self.sessions if session.session_id in answers)

    def retrieved_sessions(self) -> tuple[tuple[SessionMemory, float], ...]:
        """Hit sessions with their calibrated relevance, in rank order."""

        relevance = {
            item.canonical_id: item.relevance for item in self.execution.trace.candidate_relevance
        }
        by_id = self.by_memory_id
        out: list[tuple[SessionMemory, float]] = []
        for hit in self.execution.bundle.hits:
            session = by_id.get(hit.memory.memory_id)
            if session is None:
                continue
            out.append((session, float(relevance.get(hit.memory.memory_id, 0.0))))
        return tuple(out)


async def execute_case(
    retrieval: RetrievalService,
    provider: EmbeddingProvider | None,
    actor: ActorContext,
    text: str,
    *,
    limit: int,
    min_score: float = 0.0,
    referenced_valid_from: datetime | None = None,
    referenced_valid_to: datetime | None = None,
) -> tuple[RetrievalExecution, float]:
    """Mirror ``MemoryService.recall``: embed outside the snapshot, then execute."""

    dense_query: DenseQuery | None = None
    if provider is not None and retrieval.has_signal(RetrievalSignal.DENSE):
        values = await provider.embed_query(text)
        dense_query = DenseQuery(
            model=provider.model_name,
            dimensions=provider.dimensions,
            values=tuple(values),
        )
    query = RecallQuery(
        text=text,
        limit=limit,
        min_score=min_score,
        referenced_valid_from=referenced_valid_from,
        referenced_valid_to=referenced_valid_to,
    )
    started = perf_counter()
    execution = await retrieval.execute(
        actor,
        query,
        purpose=RetrievalPurpose.INTERACTIVE_RECALL,
        dense_query=dense_query,
    )
    return execution, (perf_counter() - started) * 1000.0


async def retrieve_question(
    record: dict[str, Any],
    *,
    limit: int,
    min_score: float = 0.0,
    use_dense: bool = True,
    temporal_query_routing: bool = False,
    learned_reranker: LearnedRerankerProvider | None = None,
    learned_rerank_policy: LearnedRerankPolicy | None = None,
) -> QuestionRetrieval:
    """Publish one question's haystack into a fresh runtime and recall once.

    Each question owns its haystack, so each question gets its own kernel,
    scope and embedding index; the runtime is discarded when this returns.
    There is no leakage between questions and no shared corpus.
    """

    clock = SteppingClock(LONGMEMEVAL_CLOCK_START)
    kernel = InMemoryKernel(clock=clock)
    work_queue = InMemoryWorkStore()
    provider = make_provider() if use_dense else None
    retrieval = RetrievalService(
        (
            in_memory_hybrid_retrieval_gateways(kernel, work_queue)
            if use_dense
            else in_memory_retrieval_gateways(kernel)
        ),
        kernel,
        learned_reranker=learned_reranker,
        learned_rerank_policy=learned_rerank_policy,
    )
    service = MemoryService(
        kernel,
        ConservativeMemoryPolicy(),
        review_store=kernel,
        # The benchmark projects the complete question-local corpus in one
        # provider batch below. Publishing stays on the real memory path, while
        # avoiding one HTTP embedding request per session.
        retrieval=retrieval,
        canonical_reader=kernel,
    )
    actor = make_actor(scope_ids(f"lme/{record['question_id']}"), agent_seed="lme")

    session_ids = [str(value) for value in record["haystack_session_ids"]]
    dates = [str(value) for value in record.get("haystack_dates", [])]
    sessions: list[SessionMemory] = []
    stored_memories: list[Memory] = []
    for position, (session_id, turns) in enumerate(
        zip(session_ids, record["haystack_sessions"], strict=True)
    ):
        clock.step()
        text = session_text(turns).strip()
        if not text:
            text = "(empty session)"
        date = dates[position] if position < len(dates) else ""
        valid_from = parse_longmemeval_datetime(date) if date else None
        result = await service.publish(
            actor,
            RememberCommand(
                idempotency_key=f"lme:{record['question_id']}:{position}",
                kind="observation",
                desired_state=MemoryState.CONFIRMED,
                visibility=Visibility.REPOSITORY,
                title=f"Conversation session recorded {date}" if date else "Conversation session",
                content=text,
                tags=("longmemeval", "session"),
                valid_from=valid_from,
            ),
        )
        if result.memory is None:
            raise RuntimeError(f"session {position:03d}:{session_id} was not stored")
        stored_memories.append(result.memory)
        sessions.append(
            SessionMemory(
                position=position,
                session_id=session_id,
                date=date,
                turns=tuple(turns),
                memory_id=result.memory.memory_id,
            )
        )
    clock.step(3600)
    if provider is not None:
        # Head-truncate pathological outliers client-side (5 of 23,867 cleaned
        # LongMemEval-S sessions exceed 30k chars; the longest is 78k). 32k
        # chars is ~8k tokens of English, matching the model window, and stays
        # under the input size at which vLLM 0.26's pooling path has been
        # observed to hang even with truncate_prompt_tokens set.
        values = await provider.embed_documents(
            [memory_content_text(memory.content)[:32_000] for memory in stored_memories]
        )
        if len(values) != len(stored_memories):
            raise RuntimeError(
                f"embedding provider returned {len(values)} vectors for "
                f"{len(stored_memories)} sessions"
            )
        vectors = tuple(
            EmbeddingVector(
                memory_id=memory.memory_id,
                model=provider.model_name,
                dimensions=provider.dimensions,
                values=tuple(vector),
            )
            for memory, vector in zip(stored_memories, values, strict=True)
        )
        projection_key = hashlib.sha256(
            f"{record['question_id']}:{provider.model_name}".encode()
        ).hexdigest()
        await work_queue.upsert_embeddings(
            actor,
            vectors,
            idempotency_key=f"lme-batch:{projection_key}",
        )

    parsed = parse_longmemeval_temporal_query(record) if temporal_query_routing else None
    closed = parsed.closed_referenced_time if parsed is not None else None

    execution, wall_ms = await execute_case(
        retrieval,
        provider,
        actor,
        str(record["question"]),
        limit=limit,
        min_score=min_score,
        referenced_valid_from=(closed.referenced_valid_from if closed is not None else None),
        referenced_valid_to=(closed.referenced_valid_to if closed is not None else None),
    )
    return QuestionRetrieval(
        record=record,
        sessions=tuple(sessions),
        execution=execution,
        wall_ms=wall_ms,
        temporal_parse=parsed,
    )


# --------------------------------------------------------------------------- #
# small shared statistics
# --------------------------------------------------------------------------- #


def mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return round(ordered[index], 3)

    return {
        "p50": pick(0.5),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": round(ordered[-1], 3),
    }


__all__ = [
    "ALL_CAPABILITIES",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_MODEL",
    "EMPTY_CONTEXT_NOTE",
    "LONGMEMEVAL_CLOCK_START",
    "LONGMEMEVAL_S_SHA256",
    "LONGMEMEVAL_S_URL",
    "OFFICIAL_ANSWER_TEMPLATE",
    "OFFICIAL_READER_SERIALIZER_VERSION",
    "QuestionRetrieval",
    "SessionMemory",
    "SteppingClock",
    "configure_embeddings",
    "build_official_reader_prompt",
    "default_longmemeval_path",
    "embedding_metadata",
    "observed_embedding_call_accounting",
    "reset_embedding_call_accounting",
    "ensure_longmemeval",
    "execute_case",
    "make_actor",
    "make_provider",
    "mean",
    "percentiles",
    "parse_longmemeval_datetime",
    "parse_longmemeval_temporal_query",
    "retrieve_question",
    "render_reader_history",
    "render_reader_session",
    "scope_ids",
    "select_questions",
    "session_text",
    "temporal_parse_metadata",
]
