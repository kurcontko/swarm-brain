#!/usr/bin/env python3
"""Measure retrieval quality end to end on a versioned corpus.

Two tracks share one protocol and one saved-run format:

``swarm``
    The checked-in coding-swarm corpus in ``tests/fixtures/retrieval_eval_corpus``
    with hand-written relevance judgments, including explicit no-answer cases.
    Runs on the in-memory kernel and, when ``SWARMBRAIN_EVAL_DATABASE_URL`` is
    set, on CockroachDB, where it additionally measures ANN Recall@k against the
    exact-vector oracle.

``longmemeval``
    The official LongMemEval-S dataset, evaluated as a pure retrieval task using
    its labelled evidence sessions.  No reader and no LLM judge are involved, so
    these numbers are not comparable with published LongMemEval QA accuracy.

Everything the retriever sees goes through the real ``RetrievalService`` path.
Saved runs use the format documented in ``docs/retrieval-evaluation.md`` and are
scored by ``scripts/evaluate_retrieval_runs.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import NAMESPACE_URL, uuid4, uuid5

from swarmbrain.adapters.embeddings import DeterministicEmbeddingProvider
from swarmbrain.adapters.extraction.in_memory import InMemoryWorkStore
from swarmbrain.adapters.memory import InMemoryKernel, in_memory_hybrid_retrieval_gateways
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.application.memory_service import MemoryService
from swarmbrain.application.retrieval_service import RetrievalExecution, RetrievalService
from swarmbrain.application.work import DurableWorkService
from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.memory import MemoryState, RecallQuery, RememberCommand, Visibility
from swarmbrain.domain.retrieval import DenseQuery, RetrievalPurpose, RetrievalSignal
from swarmbrain.retrieval import RetrievalPlanner, weighted_rrf
from swarmbrain.retrieval.evaluation import ann_recall_at_k, evaluate_lanes
from swarmbrain.workers.durable import LeasedWorkWorker
from swarmbrain.workers.embedding import EmbedMemoryHandler

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "retrieval_eval_corpus"
DEFAULT_OUT_DIR = REPO_ROOT / "benchmarks" / "retrieval"
EMBEDDING_MODEL = "deterministic-eval-1024-v0"
EMBEDDING_DIMENSIONS = 1024
DEFAULT_K = (5, 10)
ALL_CAPABILITIES = frozenset(item.value for item in Capability)
LANE_ORDER = ("exact", "lexical", "fuzzy", "dense", "graph", "direct_fused", "fused", "final")
# Saved rankings are truncated well above the reported depths so the checked-in
# artifact stays reviewable; every metric in the report uses k <= 10.
SAVED_RANKING_DEPTH = 50

# Pinned official release.  The digest is the git-lfs object id of the file,
# which is its SHA-256, so a silent upstream replacement fails loudly here.
LONGMEMEVAL_S_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s"
)
LONGMEMEVAL_S_SHA256 = "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894"


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CorpusMemory:
    key: str
    kind: str
    state: MemoryState
    title: str
    content: str
    tags: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    related: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CorpusQuery:
    case_id: str
    category: str
    text: str
    relevant: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalCorpus:
    corpus_version: str
    judgments_revision: str
    clock: datetime
    memories: tuple[CorpusMemory, ...]
    queries: tuple[CorpusQuery, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(memory.key for memory in self.memories)


def load_corpus(directory: Path = CORPUS_DIR) -> EvalCorpus:
    """Load and validate the checked-in corpus and its relevance judgments."""

    corpus = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))
    judgments = json.loads((directory / "queries.json").read_text(encoding="utf-8"))
    if corpus["corpus_version"] != judgments["corpus_version"]:
        raise ValueError("judgments were written for a different corpus version")
    if corpus["judgments_revision"] != judgments["judgments_revision"]:
        raise ValueError("corpus and judgments disagree on the judgments revision")

    memories: list[CorpusMemory] = []
    seen: set[str] = set()
    for raw in corpus["memories"]:
        key = str(raw["key"])
        if key in seen:
            raise ValueError(f"duplicate corpus memory key: {key}")
        seen.add(key)
        related = tuple(str(value) for value in raw.get("related", ()))
        for target in related:
            if target not in seen:
                raise ValueError(f"{key} links forward or to an unknown memory: {target}")
        memories.append(
            CorpusMemory(
                key=key,
                kind=str(raw["kind"]),
                state=MemoryState(str(raw.get("state", "tentative"))),
                title=str(raw["title"]),
                content=str(raw["content"]),
                tags=tuple(str(tag) for tag in raw.get("tags", ())),
                metadata=dict(raw.get("metadata", {})),
                related=related,
            )
        )

    queries: list[CorpusQuery] = []
    case_ids: set[str] = set()
    for raw in judgments["queries"]:
        case_id = str(raw["case_id"])
        if case_id in case_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        case_ids.add(case_id)
        relevant = tuple(str(value) for value in raw.get("relevant", ()))
        unknown = [value for value in relevant if value not in seen]
        if unknown:
            raise ValueError(f"{case_id} judges unknown memories: {unknown}")
        queries.append(
            CorpusQuery(
                case_id=case_id,
                category=str(raw["category"]),
                text=str(raw["text"]),
                relevant=relevant,
            )
        )

    return EvalCorpus(
        corpus_version=str(corpus["corpus_version"]),
        judgments_revision=str(corpus["judgments_revision"]),
        clock=datetime.fromisoformat(str(corpus["clock"]).replace("Z", "+00:00")).astimezone(UTC),
        memories=tuple(memories),
        queries=tuple(queries),
    )


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
# lane capture
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CaseRun:
    case_id: str
    category: str
    rankings: dict[str, list[str]]
    wall_ms: float
    lane_latency_ms: dict[str, float]
    degraded_lanes: tuple[str, ...]


def _rankings(
    execution: RetrievalExecution,
    hit_ids: Sequence[str],
    to_key: Callable[[str], str | None],
) -> dict[str, list[str]]:
    def project(ids: Iterable[str]) -> list[str]:
        out: list[str] = []
        for value in ids:
            key = to_key(value)
            if key is not None and key not in out:
                out.append(key)
            if len(out) >= SAVED_RANKING_DEPTH:
                break
        return out

    trace = execution.trace
    rankings: dict[str, list[str]] = {}
    for batch in trace.batches:
        rankings[batch.lane.value] = (
            [] if batch.degraded else project(item.canonical_id for item in batch.candidates)
        )
    direct = weighted_rrf(
        tuple(batch for batch in trace.batches if batch.lane is not RetrievalSignal.GRAPH),
        trace.plan,
    )
    rankings["direct_fused"] = project(item.canonical_id for item in direct)
    rankings["fused"] = project(item.canonical_id for item in trace.fused_candidates)
    rankings["final"] = project(hit_ids)
    return {lane: rankings[lane] for lane in LANE_ORDER if lane in rankings}


async def _execute_case(
    retrieval: RetrievalService,
    provider: DeterministicEmbeddingProvider | None,
    actor: ActorContext,
    text: str,
    *,
    limit: int,
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
    query = RecallQuery(text=text, limit=limit)
    started = perf_counter()
    execution = await retrieval.execute(
        actor,
        query,
        purpose=RetrievalPurpose.INTERACTIVE_RECALL,
        dense_query=dense_query,
    )
    return execution, (perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# in-memory swarm track
# --------------------------------------------------------------------------- #


async def _publish_corpus_in_memory(
    corpus: EvalCorpus,
    *,
    scope_seed: str | None,
) -> tuple[
    MemoryService, RetrievalService, ActorContext, dict[str, str], DeterministicEmbeddingProvider
]:
    clock = SteppingClock(corpus.clock)
    kernel = InMemoryKernel(clock=clock)
    work_queue = InMemoryWorkStore()
    provider = DeterministicEmbeddingProvider(
        dimensions=EMBEDDING_DIMENSIONS,
        model_name=EMBEDDING_MODEL,
    )
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
    actor = make_actor(scope_ids(scope_seed), agent_seed=scope_seed)
    by_key = await _publish_all(service, actor, corpus, clock=clock)
    clock.step(3600)
    worker = LeasedWorkWorker(work_queue, [EmbedMemoryHandler(provider)])
    while await worker.run_once(f"eval-embed-{uuid4()}", limit=64):
        pass
    return service, retrieval, actor, by_key, provider


async def _publish_all(
    service: MemoryService,
    actor: ActorContext,
    corpus: EvalCorpus,
    *,
    clock: SteppingClock | None,
) -> dict[str, str]:
    by_key: dict[str, str] = {}
    for memory in corpus.memories:
        if clock is not None:
            clock.step()
        result = await service.publish(
            actor,
            RememberCommand(
                idempotency_key=f"eval:{corpus.corpus_version}:{memory.key}",
                kind=memory.kind,
                desired_state=memory.state,
                visibility=Visibility.REPOSITORY,
                title=memory.title,
                content=memory.content,
                tags=memory.tags,
                metadata=memory.metadata,
                related_memory_ids=tuple(by_key[key] for key in memory.related),
            ),
        )
        if result.memory is None:
            raise RuntimeError(f"corpus memory {memory.key} was not stored")
        by_key[memory.key] = result.memory.memory_id
    return by_key


# --------------------------------------------------------------------------- #
# CockroachDB swarm track
# --------------------------------------------------------------------------- #


def _eval_database_url() -> str | None:
    value = os.getenv("SWARMBRAIN_EVAL_DATABASE_URL", "").strip()
    return value or None


def _admin_url(database_url: str) -> str:
    parts = urlsplit(database_url)
    return urlunsplit((parts.scheme, parts.netloc, "/defaultdb", parts.query, parts.fragment))


def _database_name(database_url: str) -> str:
    return urlsplit(database_url).path.lstrip("/") or "defaultdb"


async def _ensure_eval_database(database_url: str) -> None:
    """Create the dedicated evaluation database and install the schema."""

    import psycopg

    name = _database_name(database_url)
    if name in {"swarmbrain_demo", "swarmbrain_test", "defaultdb", "postgres"}:
        raise RuntimeError(f"refusing to run the evaluation against {name!r}")
    async with await psycopg.AsyncConnection.connect(
        _admin_url(database_url), autocommit=True
    ) as connection:
        await connection.execute(f'CREATE DATABASE IF NOT EXISTS "{name}"')

    from swarmbrain.adapters.cockroach.schema import install_schema, verify_schema

    await install_schema(database_url)
    await verify_schema(database_url)


async def _publish_corpus_cockroach(
    corpus: EvalCorpus,
    database_url: str,
) -> tuple[
    Any,
    MemoryService,
    RetrievalService,
    ActorContext,
    dict[str, str],
    DeterministicEmbeddingProvider,
]:
    from swarmbrain.adapters.cockroach.database import CockroachDatabase
    from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
    from swarmbrain.adapters.cockroach.retrieval import cockroach_hybrid_retrieval_gateways
    from swarmbrain.adapters.cockroach.work_store import CockroachWorkStore

    database = CockroachDatabase(database_url, min_size=1, max_size=4)
    await database.start()
    memory_store = CockroachMemoryStore(database)
    work_store = CockroachWorkStore(database)
    provider = DeterministicEmbeddingProvider(
        dimensions=EMBEDDING_DIMENSIONS,
        model_name=EMBEDDING_MODEL,
    )
    retrieval = RetrievalService(
        cockroach_hybrid_retrieval_gateways(database),
        memory_store,
    )
    service = MemoryService(
        memory_store,
        ConservativeMemoryPolicy(),
        review_store=memory_store,
        embeddings=provider,
        embedding_index=memory_store,
        work=DurableWorkService(work_store),
        retrieval=retrieval,
        canonical_reader=memory_store,
    )
    actor = make_actor(scope_ids(None))

    async def insert_run(connection: Any) -> None:
        await connection.execute(
            """
            INSERT INTO runs (tenant_id, id, project_id, repository_id, swarm_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                actor.tenant_id,
                actor.run_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
            ),
        )

    await database.run(insert_run)
    by_key = await _publish_all(service, actor, corpus, clock=None)
    worker = LeasedWorkWorker(work_store, [EmbedMemoryHandler(provider)])
    while await worker.run_once(f"eval-embed-{uuid4()}", limit=32):
        pass
    return database, service, retrieval, actor, by_key, provider


async def _ann_recall(
    database: Any,
    memory_store: Any,
    actor: ActorContext,
    corpus: EvalCorpus,
    provider: DeterministicEmbeddingProvider,
    *,
    k_values: Sequence[int],
    limit: int,
) -> dict[str, Any]:
    """Compare the ANN ranking with the exact-vector oracle in one snapshot."""

    from swarmbrain.adapters.cockroach.retrieval import CockroachDenseRetrievalGateway

    gateway = CockroachDenseRetrievalGateway(database)
    planner = RetrievalPlanner()
    per_case: list[dict[str, Any]] = []
    ann_latency: list[float] = []
    exact_latency: list[float] = []
    for case in corpus.queries:
        query = RecallQuery(text=case.text, limit=limit)
        dense_query = DenseQuery(
            model=provider.model_name,
            dimensions=provider.dimensions,
            values=tuple(await provider.embed_query(case.text)),
        )
        plan = planner.plan(
            actor,
            query,
            purpose=RetrievalPurpose.INTERACTIVE_RECALL,
            available_signals=(RetrievalSignal.DENSE,),
        )
        async with memory_store.retrieval_snapshot():
            ann = await gateway.retrieve(actor, plan, query, dense_query)
            exact = await gateway.retrieve_exact(actor, plan, query, dense_query)
        ann_ids = tuple(candidate.canonical_id for candidate in ann.candidates)
        exact_ids = tuple(candidate.canonical_id for candidate in exact.candidates)
        ann_latency.append(ann.latency_ms)
        exact_latency.append(exact.latency_ms)
        per_case.append(
            {
                "case_id": case.case_id,
                "ann_returned": len(ann_ids),
                "exact_returned": len(exact_ids),
                **{
                    f"ann_recall_at_{k}": ann_recall_at_k(ann_ids, exact_ids, k=k) for k in k_values
                },
            }
        )
    summary = {
        f"mean_ann_recall_at_{k}": _mean([case[f"ann_recall_at_{k}"] for case in per_case])
        for k in k_values
    }
    for k in k_values:
        summary[f"min_ann_recall_at_{k}"] = min(
            (case[f"ann_recall_at_{k}"] for case in per_case), default=1.0
        )
    summary["cases"] = len(per_case)
    summary["ann_latency_ms"] = _percentiles(ann_latency)
    summary["exact_latency_ms"] = _percentiles(exact_latency)
    return {"summary": summary, "cases": per_case}


# --------------------------------------------------------------------------- #
# LongMemEval-S track
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


def _session_text(turns: Sequence[dict[str, Any]]) -> str:
    return "\n".join(f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in turns)


async def _run_longmemeval_question(
    record: dict[str, Any],
    *,
    limit: int,
    use_dense: bool = True,
) -> CaseRun:
    clock = SteppingClock(datetime(2026, 8, 7, 9, tzinfo=UTC))
    kernel = InMemoryKernel(clock=clock)
    work_queue = InMemoryWorkStore()
    provider = DeterministicEmbeddingProvider(
        dimensions=EMBEDDING_DIMENSIONS,
        model_name=EMBEDDING_MODEL,
    )
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
    memory_key_by_id: dict[str, str] = {}
    for position, (session_id, turns) in enumerate(
        zip(session_ids, record["haystack_sessions"], strict=True)
    ):
        clock.step()
        key = f"{position:03d}:{session_id}"
        text = _session_text(turns).strip()
        if not text:
            text = "(empty session)"
        result = await service.publish(
            actor,
            RememberCommand(
                idempotency_key=f"lme:{record['question_id']}:{position}",
                kind="observation",
                desired_state=MemoryState.CONFIRMED,
                visibility=Visibility.REPOSITORY,
                title=f"Conversation session recorded {dates[position]}"
                if position < len(dates)
                else "Conversation session",
                content=text,
                tags=("longmemeval", "session"),
            ),
        )
        if result.memory is None:
            raise RuntimeError(f"session {key} was not stored")
        memory_key_by_id[result.memory.memory_id] = key
    clock.step(3600)
    worker = LeasedWorkWorker(work_queue, [EmbedMemoryHandler(provider)])
    while await worker.run_once(f"lme-embed-{uuid4()}", limit=64):
        pass

    execution, wall_ms = await _execute_case(
        retrieval,
        provider if use_dense else None,
        actor,
        str(record["question"]),
        limit=limit,
    )
    rankings = _rankings(
        execution,
        [hit.memory.memory_id for hit in execution.bundle.hits],
        memory_key_by_id.get,
    )
    return CaseRun(
        case_id=str(record["question_id"]),
        category=str(record["question_type"]),
        rankings=rankings,
        wall_ms=wall_ms,
        lane_latency_ms={batch.lane.value: batch.latency_ms for batch in execution.trace.batches},
        degraded_lanes=tuple(sorted(lane.value for lane in execution.trace.degraded_lanes)),
    )


async def run_longmemeval(
    dataset: Path,
    *,
    sample: int | None,
    seed: int,
    limit: int,
    use_dense: bool = True,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = json.loads(dataset.read_text(encoding="utf-8"))
    total = len(records)
    if sample is not None and sample < total:
        chosen = random.Random(seed).sample(range(total), sample)
        chosen.sort()
        records = [records[index] for index in chosen]
    cases: list[dict[str, Any]] = []
    for position, record in enumerate(records, start=1):
        run = await _run_longmemeval_question(record, limit=limit, use_dense=use_dense)
        answer_sessions = {str(value) for value in record.get("answer_session_ids", ())}
        relevant = [
            f"{index:03d}:{session_id}"
            for index, session_id in enumerate(record["haystack_session_ids"])
            if str(session_id) in answer_sessions
        ]
        cases.append(
            {
                "case_id": run.case_id,
                "category": run.category,
                "abstention_question": run.case_id.endswith("_abs"),
                "relevant_ids": relevant,
                "haystack_sessions": len(record["haystack_session_ids"]),
                "wall_ms": round(run.wall_ms, 3),
                "lane_latency_ms": {
                    lane: round(value, 3) for lane, value in run.lane_latency_ms.items()
                },
                "degraded_lanes": list(run.degraded_lanes),
                "rankings": run.rankings,
            }
        )
        if position % 25 == 0:
            print(f"  longmemeval: {position}/{len(records)} questions", file=sys.stderr)
    return {
        "track": "longmemeval-s",
        "dataset": {
            "name": "LongMemEval-S",
            "source": LONGMEMEVAL_S_URL,
            "sha256": LONGMEMEVAL_S_SHA256,
            "total_questions": total,
            "evaluated_questions": len(cases),
            "sample_seed": seed if sample is not None and sample < total else None,
        },
        "granularity": "one memory per haystack session",
        "recall_limit": limit,
        "saved_ranking_depth": SAVED_RANKING_DEPTH,
        "dense_lane_enabled": use_dense,
        "embedding": (
            {
                "provider": "DeterministicEmbeddingProvider",
                "model": EMBEDDING_MODEL,
                "dimensions": EMBEDDING_DIMENSIONS,
                "note": "hash bag-of-words; not a semantic model",
            }
            if use_dense
            else None
        ),
        "cases": cases,
    }


# --------------------------------------------------------------------------- #
# swarm track driver
# --------------------------------------------------------------------------- #


async def run_swarm_track(
    corpus: EvalCorpus,
    *,
    backend: str,
    limit: int,
    scope_seed: str | None,
    database_url: str | None,
    k_values: Sequence[int],
    use_dense: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    database: Any = None
    ann: dict[str, Any] | None = None
    if backend == "cockroach":
        if database_url is None:
            raise RuntimeError("SWARMBRAIN_EVAL_DATABASE_URL is required for the cockroach backend")
        await _ensure_eval_database(database_url)
        (
            database,
            service,
            retrieval,
            actor,
            by_key,
            provider,
        ) = await _publish_corpus_cockroach(corpus, database_url)
    else:
        service, retrieval, actor, by_key, provider = await _publish_corpus_in_memory(
            corpus, scope_seed=scope_seed
        )

    key_by_id = {memory_id: key for key, memory_id in by_key.items()}
    cases: list[dict[str, Any]] = []
    try:
        for case in corpus.queries:
            execution, wall_ms = await _execute_case(
                retrieval,
                provider if use_dense else None,
                actor,
                case.text,
                limit=limit,
            )
            bundle = execution.bundle
            rankings = _rankings(
                execution,
                [hit.memory.memory_id for hit in bundle.hits],
                key_by_id.get,
            )
            cases.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "query": case.text,
                    "relevant_ids": list(case.relevant),
                    "final_scores": [round(hit.score, 6) for hit in bundle.hits],
                    "wall_ms": round(wall_ms, 3),
                    "lane_latency_ms": {
                        batch.lane.value: round(batch.latency_ms, 3)
                        for batch in execution.trace.batches
                    },
                    "degraded_lanes": sorted(lane.value for lane in execution.trace.degraded_lanes),
                    "abstained": execution.trace.abstained,
                    "rankings": rankings,
                }
            )
        if backend == "cockroach":
            ann = await _ann_recall(
                database,
                service.store,
                actor,
                corpus,
                provider,
                k_values=k_values,
                limit=limit,
            )
    finally:
        if database is not None:
            await database.close()

    payload = {
        "track": "swarm-native",
        "corpus_version": corpus.corpus_version,
        "judgments_revision": corpus.judgments_revision,
        "backend": backend,
        "memories": len(corpus.memories),
        "queries": len(corpus.queries),
        "recall_limit": limit,
        "saved_ranking_depth": SAVED_RANKING_DEPTH,
        "dense_lane_enabled": use_dense,
        "embedding": (
            {
                "provider": "DeterministicEmbeddingProvider",
                "model": EMBEDDING_MODEL,
                "dimensions": EMBEDDING_DIMENSIONS,
                "note": "hash bag-of-words; not a semantic model",
            }
            if use_dense
            else None
        ),
        "cases": cases,
    }
    return payload, ann


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _percentiles(values: Sequence[float]) -> dict[str, float]:
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


def evaluate_saved_run(run_path: Path, k: int) -> dict[str, Any]:
    """Score a saved run with the checked-in evaluator CLI, as the docs require."""

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_retrieval_runs.py"),
            str(run_path),
            "--k",
            str(k),
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    return json.loads(completed.stdout)


def no_answer_min_score_sweep(
    cases: Sequence[dict[str, Any]],
    *,
    k: int,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Show how the public ``min_score`` floor trades recall against abstention.

    The floor is a caller-supplied ``RecallQuery`` field, so this is a property
    of the deployed policy rather than of the retriever, and it is the only
    control that produces abstention on a topically absent query.
    """

    relevant = {
        str(case["case_id"]): tuple(str(value) for value in case.get("relevant_ids", ()))
        for case in cases
    }
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        filtered = {
            str(case["case_id"]): tuple(
                value
                for value, score in zip(
                    case["rankings"]["final"],
                    case.get("final_scores", ()),
                    strict=False,
                )
                if score >= threshold
            )
            for case in cases
        }
        metrics = evaluate_lanes(relevant, {"final": filtered}, k=k)["final"]
        rows.append(
            {
                "min_score": threshold,
                "recall_at_k": round(metrics.recall_at_k, 4),
                "mrr_at_k": round(metrics.mrr_at_k, 4),
                "ndcg_at_k": round(metrics.ndcg_at_k, 4),
                "no_answer_precision": round(metrics.no_answer_precision, 4),
                "no_answer_recall": round(metrics.no_answer_recall, 4),
                "mean_returned": round(
                    _mean([float(len(value)) for value in filtered.values()]), 3
                ),
            }
        )
    return rows


def slice_metrics(
    cases: Sequence[dict[str, Any]],
    *,
    k: int,
    selector: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    selected = [case for case in cases if selector(case)]
    if not selected:
        return {}
    relevant = {
        str(case["case_id"]): tuple(str(value) for value in case.get("relevant_ids", ()))
        for case in selected
    }
    lanes = tuple(dict.fromkeys(lane for case in selected for lane in (case.get("rankings") or {})))
    rankings = {
        lane: {
            str(case["case_id"]): tuple(
                str(value) for value in (case.get("rankings") or {}).get(lane, ())
            )
            for case in selected
        }
        for lane in lanes
    }
    metrics = evaluate_lanes(relevant, rankings, k=k)
    return {
        lane: {
            "cases": value.cases,
            "answerable_cases": value.answerable_cases,
            "recall_at_k": round(value.recall_at_k, 4),
            "mrr_at_k": round(value.mrr_at_k, 4),
            "ndcg_at_k": round(value.ndcg_at_k, 4),
            "no_answer_precision": round(value.no_answer_precision, 4),
            "no_answer_recall": round(value.no_answer_recall, 4),
        }
        for lane, value in metrics.items()
    }


def build_report(
    payload: dict[str, Any],
    run_path: Path,
    *,
    k_values: Sequence[int],
    category_key: str = "category",
) -> dict[str, Any]:
    cases = payload["cases"]
    report: dict[str, Any] = {
        "run": str(run_path.relative_to(REPO_ROOT)),
        "overall": {f"k={k}": evaluate_saved_run(run_path, k) for k in k_values},
        "by_category": {
            f"k={k}": {
                category: slice_metrics(
                    cases, k=k, selector=lambda case, value=category: case[category_key] == value
                )
                for category in sorted({str(case[category_key]) for case in cases})
            }
            for k in k_values
        },
        "latency_ms": {
            "wall": _percentiles([float(case["wall_ms"]) for case in cases]),
            "lanes": {
                lane: _percentiles(
                    [
                        float(case["lane_latency_ms"][lane])
                        for case in cases
                        if lane in case.get("lane_latency_ms", {})
                    ]
                )
                for lane in sorted(
                    {lane for case in cases for lane in case.get("lane_latency_ms", {})}
                )
            },
        },
    }
    degraded = sorted({lane for case in cases for lane in case.get("degraded_lanes", ())})
    report["degraded_lanes"] = degraded
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=("swarm", "longmemeval", "both"),
        default="swarm",
    )
    parser.add_argument("--backend", choices=("memory", "cockroach"), default="memory")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=10, help="public recall limit per query")
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    parser.add_argument("--scope-seed", default="swarm-native-v1")
    parser.add_argument(
        "--no-dense",
        action="store_true",
        help="lane ablation: run without the dense lane, as if no embedding provider existed",
    )
    parser.add_argument("--lme-path", type=Path, default=None)
    parser.add_argument("--lme-download", action="store_true")
    parser.add_argument("--lme-sample", type=int, default=150)
    parser.add_argument("--lme-seed", type=int, default=20260807)
    return parser


async def _main(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    k_values = tuple(sorted(set(args.k)))
    suffix = "-nodense" if args.no_dense else ""
    use_dense = not args.no_dense

    if args.track in {"swarm", "both"}:
        corpus = load_corpus(args.corpus)
        payload, ann = await run_swarm_track(
            corpus,
            backend=args.backend,
            limit=args.limit,
            scope_seed=args.scope_seed if args.backend == "memory" else None,
            database_url=_eval_database_url(),
            k_values=k_values,
            use_dense=use_dense,
        )
        run_path = args.out_dir / f"swarm-native-{args.backend}{suffix}-run.json"
        run_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = build_report(payload, run_path, k_values=k_values)
        report["corpus_version"] = corpus.corpus_version
        report["judgments_revision"] = corpus.judgments_revision
        report["backend"] = args.backend
        report["no_answer_min_score_sweep"] = {
            f"k={k}": no_answer_min_score_sweep(
                payload["cases"],
                k=k,
                thresholds=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
            )
            for k in k_values
        }
        if ann is not None:
            report["ann_vs_exact_oracle"] = ann
        report_path = args.out_dir / f"swarm-native-{args.backend}{suffix}-report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {run_path} and {report_path}")

    if args.track in {"longmemeval", "both"}:
        dataset = args.lme_path or default_longmemeval_path()
        dataset = ensure_longmemeval(dataset, download=args.lme_download)
        payload = await run_longmemeval(
            dataset,
            sample=args.lme_sample if args.lme_sample > 0 else None,
            seed=args.lme_seed,
            limit=args.limit,
            use_dense=use_dense,
        )
        run_path = args.out_dir / f"longmemeval-s-memory{suffix}-run.json"
        run_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = build_report(payload, run_path, k_values=k_values)
        report["dataset"] = payload["dataset"]
        report["by_abstention"] = {
            f"k={k}": {
                "abstention_questions": slice_metrics(
                    payload["cases"], k=k, selector=lambda case: bool(case["abstention_question"])
                ),
                "answerable_questions": slice_metrics(
                    payload["cases"],
                    k=k,
                    selector=lambda case: not case["abstention_question"],
                ),
            }
            for k in k_values
        }
        report_path = args.out_dir / f"longmemeval-s-memory{suffix}-report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {run_path} and {report_path}")

    return 0


def main() -> int:
    args = _parser().parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CorpusMemory",
    "CorpusQuery",
    "EvalCorpus",
    "build_report",
    "ensure_longmemeval",
    "evaluate_saved_run",
    "load_corpus",
    "run_longmemeval",
    "run_swarm_track",
    "slice_metrics",
]
