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
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

# The LongMemEval dataset, ingestion mapping and single-question recall path are
# defined once in the sibling module and shared verbatim with
# ``scripts/run_longmemeval_qa.py``, so the retrieval and QA harnesses can never
# drift into measuring different systems.
from _longmemeval_common import (
    LONGMEMEVAL_S_SHA256,
    LONGMEMEVAL_S_URL,
    OFFICIAL_ANSWER_TEMPLATE,
    OFFICIAL_READER_SERIALIZER_VERSION,
    QuestionRetrieval,
    SessionMemory,
    SteppingClock,
    build_official_reader_prompt,
    configure_embeddings,
    default_longmemeval_path,
    ensure_longmemeval,
    make_actor,
    observed_embedding_call_accounting,
    reset_embedding_call_accounting,
    retrieve_question,
    scope_ids,
    select_questions,
    temporal_parse_metadata,
)
from _longmemeval_common import (
    embedding_metadata as _embedding_metadata,
)
from _longmemeval_common import (
    execute_case as _execute_case,
)
from _longmemeval_common import (
    make_provider as _make_provider,
)
from _longmemeval_common import (
    mean as _mean,
)
from _longmemeval_common import (
    percentiles as _percentiles,
)
from _longmemeval_common import (
    session_text as _session_text,
)
from _longmemeval_tokenizer import (
    ExactTokenizer,
    ExactTokenizerError,
    JsonlExactTokenizer,
    TokenizerObservation,
)

from swarmbrain.adapters.embeddings import DeterministicEmbeddingProvider
from swarmbrain.adapters.extraction.in_memory import InMemoryWorkStore
from swarmbrain.adapters.memory import InMemoryKernel, in_memory_hybrid_retrieval_gateways
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy, memory_content_text
from swarmbrain.application.memory_service import MemoryService
from swarmbrain.application.retrieval_service import RetrievalExecution, RetrievalService
from swarmbrain.application.work import DurableWorkService
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.memory import MemoryState, RecallQuery, RememberCommand, Visibility
from swarmbrain.domain.retrieval import DenseQuery, RetrievalPurpose, RetrievalSignal
from swarmbrain.ports.embeddings import EmbeddingProvider
from swarmbrain.retrieval import (
    TEMPORAL_QUERY_PARSER_VERSION,
    RetrievalPlanner,
    estimate_tokens,
    weighted_rrf,
)
from swarmbrain.retrieval.evaluation import (
    RankingCase,
    ann_recall_at_k,
    evaluate_bundle,
    evaluate_lanes,
)
from swarmbrain.retrieval.packing import answer_in_context
from swarmbrain.workers.durable import LeasedWorkWorker
from swarmbrain.workers.embedding import EmbedMemoryHandler

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "retrieval_eval_corpus"
DEFAULT_OUT_DIR = REPO_ROOT / "benchmarks" / "retrieval"

DEFAULT_K = (5, 10)
LANE_ORDER = (
    "exact",
    "lexical",
    "fuzzy",
    "dense",
    "temporal",
    "graph",
    "direct_fused",
    "fused",
    "final",
)
# Saved rankings are truncated well above the reported depths so the checked-in
# artifact stays reviewable; every metric in the report uses k <= 10.
SAVED_RANKING_DEPTH = 50
RETRIEVAL_ARTIFACT_SCHEMA_VERSION = 2
RETRIEVAL_PROTOCOL_VERSION = "swarmbrain-longmemeval-retrieval-v2"
RUN_ARTIFACT_TYPE = "swarmbrain-retrieval-eval-run"
REPORT_ARTIFACT_TYPE = "swarmbrain-retrieval-eval-report"


def context_token_accounting() -> dict[str, Any]:
    """Describe the token proxy used by ``final_tokens`` without overstating it."""

    return {
        "method": "ceil_unicode_characters_div_4",
        "mode": "development-only",
        "counted_surface": "memory_title_and_content_per_hit",
        "packing_observation_source": "additive_item_estimate",
        "characters_per_token": 4,
        "minimum_nonempty_tokens": 1,
        "exact_model_tokenizer": False,
        "tokenizer_model": None,
        "tokenizer_revision": None,
    }


def exact_context_serializer_metadata() -> dict[str, Any]:
    return {
        "version": OFFICIAL_READER_SERIALIZER_VERSION,
        "prompt_style": "official",
        "counted_surface": "complete_reader_prompt",
        "prompt_template_sha256": hashlib.sha256(
            OFFICIAL_ANSWER_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "session_order": "chronological_stable",
        "empty_context_policy": "swarmbrain-explicit-note-reference-requires-nonempty",
    }


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
    # Calibrated relevance behind each returned hit, in hit order.  Recorded on
    # both tracks so bundle precision under a relevance floor — how much of what
    # the caller is handed is actually relevant — is measurable on the public
    # dataset and not only on the hand-judged corpus.
    final_relevance: tuple[float, ...] = ()
    # Estimated reader cost of each returned hit, in hit order, so
    # answer-in-context can be replayed at any token budget offline.
    final_tokens: tuple[int, ...] = ()
    temporal_routing: dict[str, Any] | None = None
    exact_context_packing: dict[str, Any] | None = None
    exact_context_material: dict[str, Any] | None = None


def _hit_tokens(execution: RetrievalExecution) -> tuple[int, ...]:
    """Estimated tokens for each hit, over the text a reader would be shown."""

    return tuple(
        estimate_tokens(f"{hit.memory.title}\n{memory_content_text(hit.memory.content)}")
        for hit in execution.bundle.hits
    )


def _budget_label(budget: int | None) -> str:
    return "budget=none" if budget is None else f"budget={budget}"


def _serialized_prompt(
    record: dict[str, Any],
    sessions: Sequence[SessionMemory],
) -> str:
    return build_official_reader_prompt(
        record,
        [(session.date, session.turns) for session in sessions],
    )


def exact_context_prompt_material(
    retrieved: QuestionRetrieval,
    *,
    depth: int = 10,
) -> dict[str, Any]:
    """Keep the minimal public material needed to replay exact prompt hashes.

    The expanded greedy prompts are deliberately not duplicated in the run.
    The final rank order plus these top-session role/content records is enough
    for the offline compiler to reconstruct every state observed at k=5/10.
    """

    ranked = [session for session, _ in retrieved.retrieved_sessions()][:depth]
    return {
        "question": str(retrieved.record["question"]),
        "question_date": str(retrieved.record["question_date"]),
        "ranked_sessions": [
            {
                "session_id": session.key,
                "date": session.date,
                "turns": [
                    {
                        "role": str(turn.get("role", "user")),
                        "content": str(turn.get("content", "")).strip(),
                    }
                    for turn in session.turns
                ],
            }
            for session in ranked
        ],
    }


def exact_context_packing_evidence(
    retrieved: QuestionRetrieval,
    tokenizer: ExactTokenizer,
    *,
    k_values: Sequence[int] = DEFAULT_K,
    budgets: Sequence[int | None] | None = None,
) -> dict[str, Any]:
    """Observe exact full-reader-prompt counts for every greedy packing decision."""

    effective_budgets = budgets if budgets is not None else ANSWER_IN_CONTEXT_BUDGETS
    ranked = [session for session, _ in retrieved.retrieved_sessions()]
    observed: dict[tuple[str, ...], TokenizerObservation] = {}

    def observe(sessions: Sequence[SessionMemory]) -> TokenizerObservation:
        key = tuple(session.key for session in sessions)
        observation = observed.get(key)
        if observation is None:
            observation = tokenizer.count(_serialized_prompt(retrieved.record, sessions))
            observed[key] = observation
        return observation

    evidence: dict[str, Any] = {}
    for k in k_values:
        candidates = ranked[:k]
        by_budget: dict[str, Any] = {}
        for budget in effective_budgets:
            if budget is None:
                observation = observe(candidates)
                by_budget[_budget_label(budget)] = {
                    "budget": None,
                    "policy": "exact_serialized_greedy",
                    "initial_observation": None,
                    "decisions": [],
                    "kept_ids": [session.key for session in candidates],
                    "final_observation": observation.evidence(),
                }
                continue
            selected: list[SessionMemory] = []
            selected_ids: list[str] = []
            initial = observe(selected)
            current = initial
            decisions: list[dict[str, Any]] = []
            for candidate in candidates:
                proposed = [*selected, candidate]
                observation = observe(proposed)
                accepted = observation.token_count <= budget
                decisions.append(
                    {
                        "candidate_id": candidate.key,
                        "selected_before_ids": list(selected_ids),
                        "proposed_ids": [*selected_ids, candidate.key],
                        "observation": observation.evidence(),
                        "accepted": accepted,
                    }
                )
                if accepted:
                    selected = proposed
                    selected_ids.append(candidate.key)
                    current = observation
            by_budget[_budget_label(budget)] = {
                "budget": budget,
                "policy": "exact_serialized_greedy",
                "initial_observation": initial.evidence(),
                "decisions": decisions,
                "kept_ids": selected_ids,
                "final_observation": current.evidence(),
            }
        evidence[f"k={k}"] = by_budget
    return evidence


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


# --------------------------------------------------------------------------- #
# in-memory swarm track
# --------------------------------------------------------------------------- #


async def _publish_corpus_in_memory(
    corpus: EvalCorpus,
    *,
    scope_seed: str | None,
) -> tuple[MemoryService, RetrievalService, ActorContext, dict[str, str], EmbeddingProvider]:
    clock = SteppingClock(corpus.clock)
    kernel = InMemoryKernel(clock=clock)
    work_queue = InMemoryWorkStore()
    provider = _make_provider()
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
    EmbeddingProvider,
]:
    from swarmbrain.adapters.cockroach.database import CockroachDatabase
    from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
    from swarmbrain.adapters.cockroach.retrieval import cockroach_hybrid_retrieval_gateways
    from swarmbrain.adapters.cockroach.work_store import CockroachWorkStore

    database = CockroachDatabase(database_url, min_size=1, max_size=4)
    await database.start()
    memory_store = CockroachMemoryStore(database)
    work_store = CockroachWorkStore(database)
    provider = _make_provider()
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


async def _run_longmemeval_question(
    record: dict[str, Any],
    *,
    limit: int,
    use_dense: bool = True,
    temporal_query_routing: bool = False,
    context_tokenizer: ExactTokenizer | None = None,
) -> CaseRun:
    retrieved = await retrieve_question(
        record,
        limit=limit,
        use_dense=use_dense,
        temporal_query_routing=temporal_query_routing,
    )
    execution = retrieved.execution
    rankings = _rankings(
        execution,
        [hit.memory.memory_id for hit in execution.bundle.hits],
        retrieved.key_by_memory_id.get,
    )
    relevance_by_id = {item.canonical_id: item for item in execution.trace.candidate_relevance}
    return CaseRun(
        case_id=retrieved.question_id,
        category=str(record["question_type"]),
        rankings=rankings,
        wall_ms=retrieved.wall_ms,
        lane_latency_ms={batch.lane.value: batch.latency_ms for batch in execution.trace.batches},
        degraded_lanes=tuple(sorted(lane.value for lane in execution.trace.degraded_lanes)),
        final_relevance=tuple(
            round(relevance_by_id[hit.memory.memory_id].relevance, 6)
            for hit in execution.bundle.hits
            if hit.memory.memory_id in relevance_by_id
        ),
        final_tokens=_hit_tokens(execution),
        temporal_routing=temporal_parse_metadata(retrieved.temporal_parse),
        exact_context_packing=(
            exact_context_packing_evidence(retrieved, context_tokenizer)
            if context_tokenizer is not None
            else None
        ),
        exact_context_material=(
            exact_context_prompt_material(retrieved) if context_tokenizer is not None else None
        ),
    )


async def run_longmemeval(
    dataset: Path,
    *,
    sample: int | None,
    seed: int,
    limit: int,
    use_dense: bool = True,
    temporal_query_routing: bool = False,
    context_tokenizer: ExactTokenizer | None = None,
) -> dict[str, Any]:
    if context_tokenizer is not None and not isinstance(context_tokenizer, JsonlExactTokenizer):
        raise ExactTokenizerError(
            "publishable LongMemEval runs require the pinned JsonlExactTokenizer boundary"
        )
    records: list[dict[str, Any]] = json.loads(dataset.read_text(encoding="utf-8"))
    total = len(records)
    records = select_questions(records, sample=sample, seed=seed)
    if use_dense:
        reset_embedding_call_accounting()
    cases: list[dict[str, Any]] = []
    for position, record in enumerate(records, start=1):
        run = await _run_longmemeval_question(
            record,
            limit=limit,
            use_dense=use_dense,
            temporal_query_routing=temporal_query_routing,
            context_tokenizer=context_tokenizer,
        )
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
                "final_relevance": list(run.final_relevance),
                "final_tokens": list(run.final_tokens),
                "rankings": run.rankings,
                "temporal_routing": run.temporal_routing,
                "exact_context_packing": run.exact_context_packing,
                **(
                    {"exact_context_material": run.exact_context_material}
                    if run.exact_context_material is not None
                    else {}
                ),
            }
        )
        if position % 25 == 0:
            print(f"  longmemeval: {position}/{len(records)} questions", file=sys.stderr)
    return {
        "artifact_type": RUN_ARTIFACT_TYPE,
        "schema_version": RETRIEVAL_ARTIFACT_SCHEMA_VERSION,
        "protocol_version": RETRIEVAL_PROTOCOL_VERSION,
        "implementation": retrieval_implementation_fingerprint(),
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
        "temporal_query_routing": {
            "enabled": temporal_query_routing,
            "parser": TEMPORAL_QUERY_PARSER_VERSION if temporal_query_routing else None,
            "session_valid_from": "LongMemEval haystack_dates normalized to UTC",
        },
        "embedding": _embedding_metadata(use_dense),
        "embedding_call_accounting": _longmemeval_embedding_call_accounting(
            records, use_dense=use_dense
        ),
        "item_token_accounting": context_token_accounting(),
        "context_token_accounting": (
            {
                **context_tokenizer.evidence,
                "mode": "publishable-exact",
                "counted_surface": "complete_official_reader_prompt",
                "packing_observation_source": "provider_observed_full_prompt_decisions",
                "serializer": exact_context_serializer_metadata(),
            }
            if context_tokenizer is not None
            else context_token_accounting()
        ),
        "cases": cases,
    }


def _longmemeval_embedding_call_accounting(
    records: Sequence[dict[str, Any]], *, use_dense: bool
) -> dict[str, Any] | None:
    if not use_dense:
        return None
    expected = {
        "document_inputs": sum(len(record["haystack_sessions"]) for record in records),
        "document_batch_calls": len(records),
        "query_calls": len(records),
    }
    observed = observed_embedding_call_accounting()
    if observed is None:
        return {**expected, "source": "protocol-derived-non-http-provider"}
    for name, value in expected.items():
        if observed.get(name) != value:
            raise RuntimeError(
                f"observed embedding {name}={observed.get(name)!r}, expected {value}"
            )
    expected_successes = expected["document_batch_calls"] + expected["query_calls"]
    if observed.get("successful_http_calls") != expected_successes:
        raise RuntimeError(
            "observed successful embedding HTTP calls do not reconcile with the protocol"
        )
    return {**observed, "source": "provider-observed"}


# --------------------------------------------------------------------------- #
# swarm track driver
# --------------------------------------------------------------------------- #


async def _relevance_sweep(
    retrieval: RetrievalService,
    provider: EmbeddingProvider | None,
    actor: ActorContext,
    corpus: EvalCorpus,
    key_by_id: dict[str, str],
    *,
    limit: int,
    k_values: Sequence[int],
    thresholds: Sequence[float],
) -> dict[str, list[dict[str, Any]]]:
    """Re-execute every query at each ``min_score`` floor and score the result.

    This is a real re-execution rather than a post-hoc filter of a saved
    top-``k``.  It has to be: the floor is applied before truncation, so raising
    it lets the server walk further down the fused list and backfill the
    remaining slots with candidates a saved top-10 never contained.  Filtering
    the saved ranking offline would report a pessimistic recall.
    """

    relevant = {case.case_id: case.relevant for case in corpus.queries}
    per_threshold: dict[float, dict[str, tuple[str, ...]]] = {}
    abstentions: dict[float, int] = {}
    for threshold in thresholds:
        returned: dict[str, tuple[str, ...]] = {}
        empty = 0
        for case in corpus.queries:
            execution, _ = await _execute_case(
                retrieval,
                provider,
                actor,
                case.text,
                limit=limit,
                min_score=threshold,
            )
            keys: list[str] = []
            for hit in execution.bundle.hits:
                key = key_by_id.get(hit.memory.memory_id)
                if key is not None and key not in keys:
                    keys.append(key)
            returned[case.case_id] = tuple(keys)
            empty += not execution.bundle.hits
        per_threshold[threshold] = returned
        abstentions[threshold] = empty

    rows: dict[str, list[dict[str, Any]]] = {}
    for k in k_values:
        rows[f"k={k}"] = []
        for threshold in thresholds:
            returned = per_threshold[threshold]
            metrics = evaluate_lanes(relevant, {"final": returned}, k=k)["final"]
            rows[f"k={k}"].append(
                {
                    "min_score": threshold,
                    "recall_at_k": round(metrics.recall_at_k, 4),
                    "mrr_at_k": round(metrics.mrr_at_k, 4),
                    "ndcg_at_k": round(metrics.ndcg_at_k, 4),
                    "no_answer_precision": round(metrics.no_answer_precision, 4),
                    "no_answer_recall": round(metrics.no_answer_recall, 4),
                    "empty_bundles": abstentions[threshold],
                    "mean_returned": round(
                        _mean([float(len(value)) for value in returned.values()]), 3
                    ),
                }
            )
    return rows


async def run_swarm_track(
    corpus: EvalCorpus,
    *,
    backend: str,
    limit: int,
    scope_seed: str | None,
    database_url: str | None,
    k_values: Sequence[int],
    use_dense: bool = True,
    relevance_thresholds: Sequence[float] = (),
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, list[dict[str, Any]]]]:
    database: Any = None
    ann: dict[str, Any] | None = None
    sweep: dict[str, list[dict[str, Any]]] = {}
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
            relevance_by_id = {
                item.canonical_id: item for item in execution.trace.candidate_relevance
            }
            cases.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "query": case.text,
                    "relevant_ids": list(case.relevant),
                    "final_scores": [round(hit.score, 6) for hit in bundle.hits],
                    # Rank-independent relevance behind the same hits, so the
                    # rank-anchored public score and the abstention signal can
                    # be compared per hit in the saved artifact.
                    "final_relevance": [
                        round(relevance_by_id[hit.memory.memory_id].relevance, 6)
                        for hit in bundle.hits
                        if hit.memory.memory_id in relevance_by_id
                    ],
                    "final_tokens": list(_hit_tokens(execution)),
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
        if relevance_thresholds:
            sweep = await _relevance_sweep(
                retrieval,
                provider if use_dense else None,
                actor,
                corpus,
                key_by_id,
                limit=limit,
                k_values=k_values,
                thresholds=relevance_thresholds,
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
        "artifact_type": RUN_ARTIFACT_TYPE,
        "schema_version": RETRIEVAL_ARTIFACT_SCHEMA_VERSION,
        "protocol_version": RETRIEVAL_PROTOCOL_VERSION,
        "implementation": retrieval_implementation_fingerprint(),
        "track": "swarm-native",
        "corpus_version": corpus.corpus_version,
        "judgments_revision": corpus.judgments_revision,
        "backend": backend,
        "memories": len(corpus.memories),
        "queries": len(corpus.queries),
        "recall_limit": limit,
        "saved_ranking_depth": SAVED_RANKING_DEPTH,
        "dense_lane_enabled": use_dense,
        "embedding": _embedding_metadata(use_dense),
        "cases": cases,
    }
    return payload, ann, sweep


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


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
            "precision_at_k": round(value.precision_at_k, 4),
            "mrr_at_k": round(value.mrr_at_k, 4),
            "ndcg_at_k": round(value.ndcg_at_k, 4),
            "no_answer_precision": round(value.no_answer_precision, 4),
            "no_answer_recall": round(value.no_answer_recall, 4),
        }
        for lane, value in metrics.items()
    }


# Floors reported for every run.  0.0 is the shipped default and anchors the
# "no floor" column; the rest bracket the calibrated range where the swarm
# corpus trades bundle size against answerable recall.
BUNDLE_FLOORS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7)


def bundle_metrics(
    cases: Sequence[dict[str, Any]],
    *,
    k: int,
    floor: float,
) -> dict[str, Any]:
    """Replay the relevance floor over a saved run's final bundle."""

    ranking_cases = tuple(
        RankingCase(
            case_id=str(case["case_id"]),
            relevant_ids=frozenset(str(value) for value in case.get("relevant_ids", ())),
            returned_ids=tuple(
                str(value) for value in (case.get("rankings") or {}).get("final", ())
            ),
        )
        for case in cases
    )
    relevance = {
        str(case["case_id"]): tuple(float(value) for value in case.get("final_relevance", ()))
        for case in cases
        if case.get("final_relevance")
    }
    result = evaluate_bundle(ranking_cases, relevance, k=k, floor=floor)
    return {
        "cases": result.cases,
        "answerable_cases": result.answerable_cases,
        "mean_bundle_size": round(result.mean_bundle_size, 4),
        "precision": round(result.precision, 4),
        "recall": round(result.recall, 4),
        "abstained_cases": result.abstained_cases,
        "no_answer_precision": round(result.no_answer_precision, 4),
        "no_answer_recall": round(result.no_answer_recall, 4),
    }


# Reader context budgets. Development runs apply them to additive chars/4 item
# estimates. Exact LongMemEval runs apply them to provider-observed counts of
# the complete serialized reader prompt. ``None`` anchors the unbounded row.
ANSWER_IN_CONTEXT_BUDGETS: tuple[int | None, ...] = (None, 32000, 16000, 8000, 4000, 2000)


def answer_in_context_metrics(
    cases: Sequence[dict[str, Any]],
    *,
    k: int,
    budget: int | None,
    policy: str = "greedy",
) -> dict[str, Any]:
    """Replay token-budgeted packing over a saved run's final bundle."""

    packed = answer_in_context(
        (
            (
                frozenset(str(value) for value in case.get("relevant_ids", ())),
                [str(value) for value in (case.get("rankings") or {}).get("final", ())][:k],
                [int(value) for value in case.get("final_tokens", ())][:k],
            )
            for case in cases
        ),
        budget=budget,
        policy=policy,
    )
    return {
        "budget": packed.budget,
        "policy": packed.policy,
        "cases": packed.cases,
        "answerable_cases": packed.answerable_cases,
        "any_gold_in_context": round(packed.any_gold, 4),
        "all_gold_in_context": round(packed.all_gold, 4),
        "mean_hits_kept": round(packed.mean_kept, 4),
        "mean_tokens": round(packed.mean_tokens, 1),
        "truncated_cases": packed.truncated_cases,
    }


def exact_answer_in_context_metrics(
    cases: Sequence[dict[str, Any]],
    *,
    k: int,
    budget: int | None,
) -> dict[str, Any]:
    """Aggregate provider-observed full-prompt packing without re-estimating tokens."""

    rows = [case["exact_context_packing"][f"k={k}"][_budget_label(budget)] for case in cases]
    answerable = 0
    any_gold = 0
    all_gold = 0
    kept_total = 0
    token_total = 0
    truncated = 0
    for case, row in zip(cases, rows, strict=True):
        relevant = {str(value) for value in case.get("relevant_ids", ())}
        kept = [str(value) for value in row["kept_ids"]]
        expected = [str(value) for value in case["rankings"]["final"]][:k]
        answerable += bool(relevant)
        any_gold += bool(relevant.intersection(kept))
        all_gold += bool(relevant) and relevant.issubset(kept)
        kept_total += len(kept)
        token_total += int(row["final_observation"]["token_count"])
        truncated += kept != expected
    denominator = max(1, answerable)
    cases_count = len(cases)
    return {
        "budget": budget,
        "policy": "exact_serialized_greedy",
        "cases": cases_count,
        "answerable_cases": answerable,
        "any_gold_in_context": round(any_gold / denominator, 4),
        "all_gold_in_context": round(all_gold / denominator, 4),
        "mean_hits_kept": round(kept_total / max(1, cases_count), 4),
        "mean_tokens": round(token_total / max(1, cases_count), 1),
        "truncated_cases": truncated,
    }


def _artifact_path(path: Path) -> str:
    """Keep repository artifacts portable while allowing explicit external output dirs."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"saved retrieval run is missing or unsafe: {path}")
    content = resolved.read_bytes()
    return {
        "path": _artifact_path(resolved),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def retrieval_implementation_fingerprint() -> dict[str, Any]:
    """Bind benchmark evidence to the exact retrieval implementation."""

    paths = [
        REPO_ROOT / "scripts" / "_longmemeval_common.py",
        REPO_ROOT / "scripts" / "_longmemeval_tokenizer.py",
        REPO_ROOT / "scripts" / "evaluate_retrieval_runs.py",
        REPO_ROOT / "scripts" / "run_retrieval_eval.py",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "uv.lock",
    ]
    paths.extend((REPO_ROOT / "src" / "swarmbrain").rglob("*.py"))
    files = {
        path.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
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
        "artifact_type": REPORT_ARTIFACT_TYPE,
        "schema_version": RETRIEVAL_ARTIFACT_SCHEMA_VERSION,
        "run": _artifact_path(run_path),
        "run_artifact": _file_identity(run_path),
        "execution": {
            "protocol_version": payload.get("protocol_version"),
            "implementation": payload.get("implementation"),
            "track": payload.get("track"),
            "granularity": payload.get("granularity"),
            "recall_limit": payload.get("recall_limit"),
            "saved_ranking_depth": payload.get("saved_ranking_depth"),
            "dense_lane_enabled": payload.get("dense_lane_enabled"),
            "temporal_query_routing": payload.get("temporal_query_routing"),
            "embedding": payload.get("embedding"),
            "embedding_call_accounting": payload.get("embedding_call_accounting"),
            "item_token_accounting": payload.get("item_token_accounting")
            or context_token_accounting(),
            # Schema-v2 runs created before this disclosure field existed still
            # used the same stable proxy.  Make that limitation explicit in
            # every rebuilt report instead of silently treating it as exact.
            "context_token_accounting": payload.get("context_token_accounting")
            or context_token_accounting(),
        },
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
    # Quality per token: what the caller is handed once the floor is applied,
    # rather than what the ranking could have found at unbounded depth.
    report["bundle_by_floor"] = {
        f"k={k}": {
            f"floor={floor}": bundle_metrics(cases, k=k, floor=floor) for floor in BUNDLE_FLOORS
        }
        for k in k_values
    }
    # Answer-in-context: exact runs aggregate full-prompt decision observations;
    # development runs retain the legacy additive chars/4 item replay.
    token_accounting = report["execution"]["context_token_accounting"]
    if isinstance(token_accounting, dict) and token_accounting.get("exact_model_tokenizer") is True:
        if not all(isinstance(case.get("exact_context_packing"), dict) for case in cases):
            raise ValueError("exact token accounting requires every case's packing observations")
        report["answer_in_context"] = {
            f"k={k}": {
                _budget_label(budget): exact_answer_in_context_metrics(
                    cases,
                    k=k,
                    budget=budget,
                )
                for budget in ANSWER_IN_CONTEXT_BUDGETS
            }
            for k in k_values
        }
    elif any(case.get("final_tokens") for case in cases):
        report["answer_in_context"] = {
            f"k={k}": {
                ("budget=none" if budget is None else f"budget={budget}"): (
                    answer_in_context_metrics(cases, k=k, budget=budget)
                )
                for budget in ANSWER_IN_CONTEXT_BUDGETS
            }
            for k in k_values
        }
    degraded = sorted({lane for case in cases for lane in case.get("degraded_lanes", ())})
    report["degraded_lanes"] = degraded
    return report


def build_longmemeval_report(
    payload: dict[str, Any],
    run_path: Path,
    *,
    k_values: Sequence[int] = DEFAULT_K,
) -> dict[str, Any]:
    """Build the canonical retrieval-only LongMemEval-S report from one saved run."""

    report = build_report(payload, run_path, k_values=k_values)
    report["dataset"] = payload["dataset"]
    report["by_abstention"] = {
        f"k={k}": {
            "abstention_questions": slice_metrics(
                payload["cases"],
                k=k,
                selector=lambda case: bool(case["abstention_question"]),
            ),
            "answerable_questions": slice_metrics(
                payload["cases"],
                k=k,
                selector=lambda case: not case["abstention_question"],
            ),
        }
        for k in k_values
    }
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
    parser.add_argument(
        "--min-score-sweep",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8],
        help=(
            "abstention sweep: re-execute every query at each RecallQuery.min_score "
            "floor (a floor on calibrated relevance) and score the result; pass no "
            "values to skip the sweep"
        ),
    )
    parser.add_argument("--lme-path", type=Path, default=None)
    parser.add_argument("--lme-download", action="store_true")
    parser.add_argument("--lme-sample", type=int, default=150)
    parser.add_argument("--lme-seed", type=int, default=20260807)
    parser.add_argument(
        "--temporal-query-routing",
        action="store_true",
        help=(
            "experimental A/B: parse closed referenced-time expressions against each "
            "LongMemEval question_date and enable the temporal lane; disabled by default"
        ),
    )
    parser.add_argument(
        "--embeddings",
        choices=("deterministic", "openai"),
        default="deterministic",
        help="dense-lane embedder; openai targets any /v1/embeddings-compatible server",
    )
    parser.add_argument("--embeddings-base-url", default=None)
    parser.add_argument("--embeddings-model", default=None)
    parser.add_argument(
        "--embeddings-api-key-env",
        default="SWARMBRAIN_EMBEDDINGS_API_KEY",
        help="environment variable holding the embedding API key; never pass the key itself",
    )
    parser.add_argument("--context-tokenizer-executable", type=Path, default=None)
    parser.add_argument("--context-tokenizer-executable-sha256", default=None)
    parser.add_argument("--context-tokenizer-artifact", type=Path, default=None)
    parser.add_argument("--context-tokenizer-artifact-sha256", default=None)
    parser.add_argument("--context-tokenizer-model", default=None)
    parser.add_argument("--context-tokenizer-revision", default=None)
    return parser


def _configured_context_tokenizer(args: argparse.Namespace) -> JsonlExactTokenizer | None:
    values = (
        args.context_tokenizer_executable,
        args.context_tokenizer_executable_sha256,
        args.context_tokenizer_artifact,
        args.context_tokenizer_artifact_sha256,
        args.context_tokenizer_model,
        args.context_tokenizer_revision,
    )
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise ExactTokenizerError("exact context tokenizer configuration must provide all six pins")
    return JsonlExactTokenizer(
        executable=args.context_tokenizer_executable,
        executable_sha256=args.context_tokenizer_executable_sha256,
        artifact=args.context_tokenizer_artifact,
        artifact_sha256=args.context_tokenizer_artifact_sha256,
        model=args.context_tokenizer_model,
        revision=args.context_tokenizer_revision,
        repo_root=REPO_ROOT,
    )


async def _main(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    k_values = tuple(sorted(set(args.k)))
    configure_embeddings(
        args.embeddings,
        base_url=args.embeddings_base_url,
        model_id=args.embeddings_model,
        api_key=(
            os.getenv(args.embeddings_api_key_env, "").strip() or None
            if args.embeddings == "openai"
            else None
        ),
    )
    # Non-default embedders write to their own file family so the checked-in
    # deterministic baselines stay untouched for comparison.
    suffix = "-nodense" if args.no_dense else ""
    if args.embeddings != "deterministic":
        suffix = f"-{args.embeddings}{suffix}"
    lme_suffix = f"{suffix}-temporal" if args.temporal_query_routing else suffix
    use_dense = not args.no_dense

    if args.track in {"swarm", "both"}:
        corpus = load_corpus(args.corpus)
        relevance_thresholds = tuple(sorted(set(args.min_score_sweep)))
        payload, ann, relevance_sweep = await run_swarm_track(
            corpus,
            backend=args.backend,
            limit=args.limit,
            scope_seed=args.scope_seed if args.backend == "memory" else None,
            database_url=_eval_database_url(),
            k_values=k_values,
            use_dense=use_dense,
            relevance_thresholds=relevance_thresholds,
        )
        run_path = args.out_dir / f"swarm-native-{args.backend}{suffix}-run.json"
        run_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = build_report(payload, run_path, k_values=k_values)
        report["corpus_version"] = corpus.corpus_version
        report["judgments_revision"] = corpus.judgments_revision
        report["backend"] = args.backend
        # Kept unchanged so the original 2026-08-07 measurement stays
        # reproducible: this sweeps the rank-anchored public score, which is
        # what ``min_score`` used to filter on.
        report["no_answer_public_score_sweep"] = {
            f"k={k}": no_answer_min_score_sweep(
                payload["cases"],
                k=k,
                thresholds=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
            )
            for k in k_values
        }
        # What ``RecallQuery.min_score`` filters on now: calibrated relevance,
        # measured by re-executing every query at each floor.
        if relevance_sweep:
            report["no_answer_min_score_sweep"] = relevance_sweep
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
        context_tokenizer = _configured_context_tokenizer(args)
        try:
            payload = await run_longmemeval(
                dataset,
                sample=args.lme_sample if args.lme_sample > 0 else None,
                seed=args.lme_seed,
                limit=args.limit,
                use_dense=use_dense,
                temporal_query_routing=args.temporal_query_routing,
                context_tokenizer=context_tokenizer,
            )
        finally:
            if context_tokenizer is not None:
                context_tokenizer.close()
        run_path = args.out_dir / f"longmemeval-s-memory{lme_suffix}-run.json"
        run_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = build_longmemeval_report(payload, run_path, k_values=k_values)
        report_path = args.out_dir / f"longmemeval-s-memory{lme_suffix}-report.json"
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
    # Re-exported from ``_longmemeval_common`` so callers and tests that already
    # treat this script as the Track 2 entry point keep working after the split.
    "_session_text",
    "build_longmemeval_report",
    "build_report",
    "context_token_accounting",
    "exact_answer_in_context_metrics",
    "exact_context_packing_evidence",
    "exact_context_prompt_material",
    "exact_context_serializer_metadata",
    "ensure_longmemeval",
    "evaluate_saved_run",
    "load_corpus",
    "run_longmemeval",
    "run_swarm_track",
    "slice_metrics",
]
