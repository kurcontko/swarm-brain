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
from swarmbrain.adapters.memory import InMemoryKernel, in_memory_hybrid_retrieval_gateways
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.application.memory_service import MemoryService
from swarmbrain.application.retrieval_service import RetrievalExecution, RetrievalService
from swarmbrain.application.work import DurableWorkService
from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.memory import MemoryState, RecallQuery, RememberCommand, Visibility
from swarmbrain.domain.retrieval import DenseQuery, RetrievalPurpose, RetrievalSignal
from swarmbrain.ports.embeddings import EmbeddingProvider
from swarmbrain.workers.durable import LeasedWorkWorker
from swarmbrain.workers.embedding import EmbedMemoryHandler

EMBEDDING_MODEL = "deterministic-eval-1024-v0"
EMBEDDING_DIMENSIONS = 1024
ALL_CAPABILITIES = frozenset(item.value for item in Capability)

# Pinned official release.  The digest is the git-lfs object id of the file,
# which is its SHA-256, so a silent upstream replacement fails loudly here.
LONGMEMEVAL_S_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s"
)
LONGMEMEVAL_S_SHA256 = "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894"

# The clock every LongMemEval runtime starts from.  Fixed so ingestion order,
# and therefore recency tie-breaking, is identical on every run and in both
# harnesses.
LONGMEMEVAL_CLOCK_START = datetime(2026, 8, 7, 9, tzinfo=UTC)

# Selected once per process from ``--embeddings``; the deterministic default
# keeps the original 2026-08-07 runs byte-reproducible.  A single shared
# instance serves every composition site so the openai provider reuses one
# HTTP client across the whole run.
_provider_singleton: EmbeddingProvider | None = None


def configure_embeddings(
    kind: str, *, base_url: str | None = None, model_id: str | None = None
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
    return {
        "provider": type(provider).__name__,
        "model": provider.model_name,
        "dimensions": provider.dimensions,
        "note": (
            "hash bag-of-words; not a semantic model"
            if deterministic
            else "semantic model served over an OpenAI-compatible endpoint"
        ),
    }


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
    return base / "longmemeval_s.json"


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
        return tuple(
            session.key for session in self.sessions if session.session_id in answers
        )

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
    query = RecallQuery(text=text, limit=limit, min_score=min_score)
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
) -> QuestionRetrieval:
    """Publish one question's haystack into a fresh runtime and recall once.

    Each question owns its haystack, so each question gets its own kernel,
    scope and embedding index; the runtime is discarded when this returns.
    There is no leakage between questions and no shared corpus.
    """

    clock = SteppingClock(LONGMEMEVAL_CLOCK_START)
    kernel = InMemoryKernel(clock=clock)
    work_queue = InMemoryWorkStore()
    provider = make_provider()
    retrieval = RetrievalService(
        in_memory_hybrid_retrieval_gateways(kernel, work_queue),
        kernel,
    )
    service = MemoryService(
        kernel,
        ConservativeMemoryPolicy(),
        review_store=kernel,
        embeddings=provider,
        embedding_index=work_queue,
        work=DurableWorkService(work_queue),
        retrieval=retrieval,
        canonical_reader=kernel,
    )
    actor = make_actor(scope_ids(f"lme/{record['question_id']}"), agent_seed="lme")

    session_ids = [str(value) for value in record["haystack_session_ids"]]
    dates = [str(value) for value in record.get("haystack_dates", [])]
    sessions: list[SessionMemory] = []
    for position, (session_id, turns) in enumerate(
        zip(session_ids, record["haystack_sessions"], strict=True)
    ):
        clock.step()
        text = session_text(turns).strip()
        if not text:
            text = "(empty session)"
        date = dates[position] if position < len(dates) else ""
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
            ),
        )
        if result.memory is None:
            raise RuntimeError(f"session {position:03d}:{session_id} was not stored")
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
    worker = LeasedWorkWorker(work_queue, [EmbedMemoryHandler(provider)])
    while await worker.run_once(f"lme-embed-{uuid4()}", limit=64):
        pass

    execution, wall_ms = await execute_case(
        retrieval,
        provider if use_dense else None,
        actor,
        str(record["question"]),
        limit=limit,
        min_score=min_score,
    )
    return QuestionRetrieval(
        record=record,
        sessions=tuple(sessions),
        execution=execution,
        wall_ms=wall_ms,
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
    "LONGMEMEVAL_CLOCK_START",
    "LONGMEMEVAL_S_SHA256",
    "LONGMEMEVAL_S_URL",
    "QuestionRetrieval",
    "SessionMemory",
    "SteppingClock",
    "configure_embeddings",
    "default_longmemeval_path",
    "embedding_metadata",
    "ensure_longmemeval",
    "execute_case",
    "make_actor",
    "make_provider",
    "mean",
    "percentiles",
    "retrieve_question",
    "scope_ids",
    "select_questions",
    "session_text",
]
