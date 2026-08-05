"""CockroachDB-native exact, FTS, and trigram candidate lanes."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from inspect import isawaitable
from time import perf_counter
from typing import Any
from uuid import UUID

from swarmbrain.application.memory_policy import memory_text_sha256
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.memory import Memory, RecallQuery, Visibility
from swarmbrain.domain.retrieval import (
    Candidate,
    CandidateBatch,
    RetrievalPlan,
    RetrievalSignal,
)
from swarmbrain.retrieval.projection import (
    FUZZY_SIMILARITY_THRESHOLD,
    MAX_EXACT_TERM,
    MAX_QUERY_CHARS,
    MAX_QUERY_TOKENS,
    RETRIEVAL_PROJECTION_ID,
    domain_lane,
    exact_terms,
    lookup_text,
    normalize_term,
    projection_scope_key,
    search_text,
)

from .database import CockroachDatabase

_FTS_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


@asynccontextmanager
async def cockroach_retrieval_snapshot(database: Any) -> AsyncIterator[None]:
    """Use the database snapshot hook while retaining light-weight test doubles."""

    factory = getattr(database, "retrieval_snapshot", None)
    if factory is None:
        yield
        return
    context = factory()
    if isawaitable(context):
        context = await context
    async with context:
        yield


@asynccontextmanager
async def cockroach_retrieval_connection(database: Any) -> AsyncIterator[Any]:
    """Reuse one retrieval transaction connection when a snapshot is active."""

    factory = getattr(database, "retrieval_connection", None)
    if factory is None:
        async with database.pool.connection() as connection:
            yield connection
        return
    context = factory()
    if isawaitable(context):
        context = await context
    async with context as connection:
        yield connection


async def cockroach_retrieval_now(database: Any, connection: Any | None = None) -> Any:
    """Read the stable snapshot clock, with a fallback for adapter test doubles."""

    clock = getattr(database, "retrieval_now", None)
    if clock is not None:
        value = clock(connection)
        return await value if isawaitable(value) else value
    if connection is None:
        async with cockroach_retrieval_connection(database) as leased:
            return await cockroach_retrieval_now(database, leased)
    cursor = await connection.execute("SELECT now() AS database_now")
    row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("database clock query returned no row")
    return row["database_now"]


@dataclass(frozen=True, slots=True)
class RecallPredicates:
    clauses: tuple[str, ...]
    parameters: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class CandidateTrustFilter:
    joins: str = ""
    clause: str = ""
    parameters: tuple[Any, ...] = ()


def build_recall_predicates(
    actor: ActorContext,
    query: RecallQuery,
    *,
    now: Any,
    alias: str = "m",
    candidate_ids: Sequence[str] = (),
    apply_trust: bool = True,
) -> RecallPredicates:
    """Build the one canonical scope/lifecycle/trust predicate set."""

    clauses = [
        f"{alias}.tenant_id = %s",
        f"{alias}.project_id = %s",
        f"{alias}.repository_id = %s",
    ]
    parameters: list[Any] = [actor.tenant_id, actor.project_id, actor.repository_id]
    visibility_clauses: list[str] = []
    if Visibility.REPOSITORY in query.visibilities:
        visibility_clauses.append(f"{alias}.visibility = 'repository'")
    if Visibility.RUN in query.visibilities:
        visibility_clauses.append(
            f"({alias}.visibility = 'run' AND {alias}.swarm_id = %s AND {alias}.run_id = %s)"
        )
        parameters.extend((actor.swarm_id, actor.run_id))
    if Visibility.TASK in query.visibilities and query.task_id is not None:
        visibility_clauses.append(
            f"({alias}.visibility = 'task' AND {alias}.swarm_id = %s "
            f"AND {alias}.run_id = %s AND {alias}.task_id = %s)"
        )
        parameters.extend((actor.swarm_id, actor.run_id, UUID(query.task_id)))
    clauses.append("(" + " OR ".join(visibility_clauses) + ")" if visibility_clauses else "false")

    clauses.append(f"{alias}.state = ANY(%s::STRING[])")
    parameters.append(sorted(state.value for state in query.effective_states))
    if query.memory_ids:
        clauses.append(f"{alias}.id = ANY(%s::UUID[])")
        parameters.append([UUID(item) for item in sorted(query.memory_ids)])
    if candidate_ids:
        clauses.append(f"{alias}.id = ANY(%s::UUID[])")
        parameters.append([UUID(item) for item in dict.fromkeys(candidate_ids)])
    if query.kinds:
        clauses.append(f"{alias}.kind = ANY(%s::STRING[])")
        parameters.append(sorted(str(kind) for kind in query.kinds))

    if query.recorded_at is not None:
        clauses.extend(
            (
                f"{alias}.recorded_from <= %s",
                f"({alias}.recorded_to IS NULL OR %s < {alias}.recorded_to)",
            )
        )
        parameters.extend((query.recorded_at, query.recorded_at))
    elif not query.include_superseded:
        clauses.append(f"{alias}.recorded_to IS NULL")

    world_at = query.world_at or now
    clauses.extend(
        (
            f"{alias}.valid_from <= %s",
            f"({alias}.valid_to IS NULL OR %s < {alias}.valid_to)",
        )
    )
    parameters.extend((world_at, world_at))

    if apply_trust and not query.include_refuted:
        clauses.append(
            f"""
            (
                NOT EXISTS (
                    SELECT 1 FROM memory_evidence AS me0
                    WHERE me0.memory_id = {alias}.id
                )
                OR EXISTS (
                    SELECT 1
                    FROM memory_evidence AS me1
                    JOIN evidence AS e1 ON e1.id = me1.evidence_id
                    JOIN sources AS s1 ON s1.id = e1.source_id
                    WHERE me1.memory_id = {alias}.id
                      AND s1.review_state != 'rejected'
                      AND s1.trust_label != 'untrusted'
                      AND s1.tenant_id = {alias}.tenant_id
                      AND s1.project_id = {alias}.project_id
                      AND s1.repository_id = {alias}.repository_id
                )
            )
            """
        )
    return RecallPredicates(tuple(clauses), tuple(parameters))


def build_candidate_trust_filter(
    query: RecallQuery,
    *,
    alias: str = "m",
) -> CandidateTrustFilter:
    """Candidate-driven trust check that preserves lookup joins and avoids global scans."""

    if query.include_refuted:
        return CandidateTrustFilter()
    return CandidateTrustFilter(
        joins=f"""
            LEFT JOIN LATERAL (
                SELECT true AS present
                FROM memory_evidence@primary AS me0
                WHERE me0.memory_id = {alias}.id
                LIMIT 1
            ) AS any_evidence ON true
            LEFT JOIN LATERAL (
                SELECT true AS present
                FROM memory_evidence@primary AS me1
                INNER LOOKUP JOIN evidence@primary AS e1 ON e1.id = me1.evidence_id
                INNER LOOKUP JOIN sources@primary AS s1 ON s1.id = e1.source_id
                WHERE me1.memory_id = {alias}.id
                  AND s1.review_state != 'rejected'
                  AND s1.trust_label != 'untrusted'
                  AND s1.tenant_id = {alias}.tenant_id
                  AND s1.project_id = {alias}.project_id
                  AND s1.repository_id = {alias}.repository_id
                LIMIT 1
            ) AS good_evidence ON true
        """,
        clause="(any_evidence.present IS NULL OR good_evidence.present)",
    )


async def upsert_memory_retrieval_projection(connection: Any, memory: Memory) -> None:
    """Synchronously maintain the deterministic v7 lexical projection."""

    digest = memory_text_sha256(memory.content)
    scope_key = projection_scope_key(
        memory.visibility,
        repository_id=memory.repository_id,
        run_id=memory.run_id,
        task_id=memory.task_id,
    )
    projected_search = search_text(
        title=memory.title,
        content=memory.content,
        tags=memory.tags,
        metadata=memory.metadata,
    )
    projected_lookup = lookup_text(
        memory_id=memory.memory_id,
        content_sha256=digest,
        title=memory.title,
        content=memory.content,
        tags=memory.tags,
        metadata=memory.metadata,
    )
    key = (
        memory.tenant_id,
        memory.project_id,
        memory.repository_id,
        RETRIEVAL_PROJECTION_ID,
        scope_key,
    )
    await connection.execute(
        """
        UPSERT INTO retrieval_documents (
            tenant_id, project_id, repository_id, projection_id, scope_key,
            resource_type, resource_id, resource_version, canonical_id,
            domain_lane, search_text, lookup_text, content_sha256, indexed_at
        ) VALUES (
            %s, %s, %s, %s, %s, 'memory', %s, %s, %s, %s, %s, %s, %s, now()
        )
        """,
        (
            *key,
            UUID(memory.memory_id),
            memory.version,
            UUID(memory.memory_id),
            domain_lane(memory.kind, memory.metadata),
            projected_search,
            projected_lookup,
            bytes.fromhex(digest),
        ),
    )
    await connection.execute(
        """
        DELETE FROM retrieval_exact_terms
        WHERE tenant_id = %s
          AND project_id = %s
          AND repository_id = %s
          AND projection_id = %s
          AND scope_key = %s
          AND resource_type = 'memory'
          AND resource_id = %s
        """,
        (*key, UUID(memory.memory_id)),
    )
    terms = exact_terms(
        memory_id=memory.memory_id,
        content_sha256=digest,
        title=memory.title,
        content=memory.content,
        tags=memory.tags,
        metadata=memory.metadata,
    )
    if terms:
        values_sql = ", ".join(
            "(%s, %s, %s, %s, %s, %s, %s, 'memory', %s, %s, now())" for _ in terms
        )
        parameters: list[Any] = []
        for term in terms:
            parameters.extend((*key, term.value, term.kind, UUID(memory.memory_id), memory.version))
        await connection.execute(
            f"""
            INSERT INTO retrieval_exact_terms (
                tenant_id, project_id, repository_id, projection_id, scope_key,
                normalized_term, term_kind, resource_type, resource_id,
                resource_version, indexed_at
            ) VALUES {values_sql}
            ON CONFLICT DO NOTHING
            """,
            tuple(parameters),
        )


class CockroachRetrievalGateway:
    def __init__(self, database: CockroachDatabase, signal: RetrievalSignal) -> None:
        if signal not in {
            RetrievalSignal.EXACT,
            RetrievalSignal.LEXICAL,
            RetrievalSignal.FUZZY,
        }:
            raise ValueError(f"unsupported CockroachDB retrieval signal: {signal}")
        self.database = database
        self._signal = signal

    @property
    def signal(self) -> RetrievalSignal:
        return self._signal

    async def retrieve(
        self,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
    ) -> CandidateBatch:
        started = perf_counter()
        if self.signal is RetrievalSignal.LEXICAL and not _safe_or_tsquery(query.text):
            return CandidateBatch(
                lane=self.signal,
                examined_count=0,
                latency_ms=(perf_counter() - started) * 1000.0,
                projection_watermark=RETRIEVAL_PROJECTION_ID,
            )
        fuzzy_query = normalize_term(query.text[:MAX_QUERY_CHARS])
        if self.signal is RetrievalSignal.FUZZY and not 3 <= len(fuzzy_query) <= 256:
            return CandidateBatch(
                lane=self.signal,
                examined_count=0,
                latency_ms=(perf_counter() - started) * 1000.0,
                projection_watermark=RETRIEVAL_PROJECTION_ID,
            )
        budget = plan.lane_budgets[self.signal.value]
        rows: list[dict[str, Any]] = []
        branch_truncated = False
        async with cockroach_retrieval_connection(self.database) as connection:
            now = await cockroach_retrieval_now(self.database, connection)
            if self.signal is RetrievalSignal.FUZZY:
                # The `%` operator reads this session value. Pin it per
                # transaction so cluster/session customization cannot change
                # retrieval semantics or adapter parity.
                await connection.execute(
                    "SET LOCAL pg_trgm.similarity_threshold = %s",
                    (FUZZY_SIMILARITY_THRESHOLD,),
                )
            for scope_key in _scope_keys(plan):
                branch = await self._retrieve_scope(
                    connection,
                    actor,
                    plan,
                    query,
                    scope_key,
                    now,
                    budget,
                )
                rows.extend(branch)
                branch_truncated = branch_truncated or len(branch) >= budget

        unique: dict[str, dict[str, Any]] = {}
        for row in rows:
            memory_id = str(row["id"])
            current = unique.get(memory_id)
            if current is None or float(row["lane_score"]) > float(current["lane_score"]):
                unique[memory_id] = row
        ordered = sorted(
            unique.values(),
            key=lambda row: (
                -float(row["lane_score"]),
                -row["recorded_from"].timestamp(),
                str(row["id"]),
            ),
        )
        candidates = tuple(
            Candidate(
                resource_type="memory",
                resource_id=str(row["id"]),
                resource_version=int(row["version"]),
                canonical_id=str(row["id"]),
                domain_lane=str(
                    row.get("domain_lane")
                    or domain_lane(str(row["kind"]), row.get("metadata") or {})
                ),
                signal=self.signal,
                rank=rank,
                raw_score=float(row["lane_score"]),
                projection_id=RETRIEVAL_PROJECTION_ID,
                projection_version="cockroach-v7",
                reasons=(
                    {
                        RetrievalSignal.EXACT: "exact_term",
                        RetrievalSignal.LEXICAL: "fts_simple",
                        RetrievalSignal.FUZZY: "trigram_similarity",
                    }[self.signal],
                ),
            )
            for rank, row in enumerate(ordered[:budget], start=1)
        )
        return CandidateBatch(
            lane=self.signal,
            candidates=candidates,
            examined_count=len(rows),
            latency_ms=(perf_counter() - started) * 1000.0,
            truncated=branch_truncated or len(ordered) > budget,
            projection_watermark=RETRIEVAL_PROJECTION_ID,
        )

    async def _retrieve_scope(
        self,
        connection: Any,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
        scope_key: str,
        now: Any,
        budget: int,
    ) -> list[dict[str, Any]]:
        predicates = build_recall_predicates(actor, query, now=now, apply_trust=False)
        trust = build_candidate_trust_filter(query)
        if self.signal is RetrievalSignal.EXACT:
            return await self._exact_scope(
                connection,
                actor,
                plan,
                query,
                scope_key,
                predicates,
                trust,
                budget,
            )
        fts_query = _safe_or_tsquery(query.text)
        score_sql = (
            "ts_rank(d.search_tsv, to_tsquery('simple', %s::STRING))"
            if self.signal is RetrievalSignal.LEXICAL
            else "similarity(d.lookup_text, %s::STRING)"
        )
        match_sql = (
            "d.search_tsv @@ to_tsquery('simple', %s::STRING)"
            if self.signal is RetrievalSignal.LEXICAL
            else "d.lookup_text %% %s::STRING"
        )
        lane_query = (
            fts_query
            if self.signal is RetrievalSignal.LEXICAL
            else normalize_term(query.text[:MAX_QUERY_CHARS])
        )
        index_name = (
            "retrieval_documents_fts"
            if self.signal is RetrievalSignal.LEXICAL
            else "retrieval_documents_lookup_trgm"
        )
        sql = f"""
            SELECT m.id, m.version, m.kind, m.metadata, m.recorded_from, d.domain_lane,
                   {score_sql} AS lane_score
            FROM retrieval_documents@{index_name} AS d
            INNER LOOKUP JOIN memories@primary AS m ON m.id = d.canonical_id
            {trust.joins}
            WHERE d.tenant_id = %s
              AND d.project_id = %s
              AND d.repository_id = %s
              AND d.projection_id = %s
              AND d.scope_key = %s
              AND d.resource_type = 'memory'
              AND {" AND ".join(predicates.clauses)}
              {f"AND {trust.clause}" if trust.clause else ""}
              AND {match_sql}
            ORDER BY lane_score DESC, m.recorded_from DESC, m.id
            LIMIT %s
        """
        cursor = await connection.execute(
            sql,
            (
                lane_query,
                *trust.parameters,
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                RETRIEVAL_PROJECTION_ID,
                scope_key,
                *predicates.parameters,
                lane_query,
                budget,
            ),
        )
        return list(await cursor.fetchall())

    async def _exact_scope(
        self,
        connection: Any,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
        scope_key: str,
        predicates: RecallPredicates,
        trust: CandidateTrustFilter,
        budget: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        direct_clauses: list[str] = []
        direct_parameters: list[Any] = []
        raw_query = query.text[:MAX_QUERY_CHARS].strip()
        try:
            query_uuid = UUID(raw_query)
        except ValueError:
            query_uuid = None
        if query_uuid is not None:
            direct_clauses.append("m.id = %s")
            direct_parameters.append(query_uuid)
        if plan.seed_memory_ids:
            direct_clauses.append("m.id = ANY(%s::UUID[])")
            direct_parameters.append([UUID(item) for item in plan.seed_memory_ids])

        if direct_clauses:
            direct_cursor = await connection.execute(
                f"""
                SELECT DISTINCT m.id, m.version, m.kind, m.metadata, m.recorded_from,
                       NULL AS domain_lane, 1.0::FLOAT8 AS lane_score
                FROM memories AS m
                {trust.joins}
                WHERE {" AND ".join(predicates.clauses)}
                  {f"AND {trust.clause}" if trust.clause else ""}
                  AND ({" OR ".join(direct_clauses)})
                ORDER BY m.recorded_from DESC, m.id
                LIMIT %s
                """,
                (
                    *trust.parameters,
                    *predicates.parameters,
                    *direct_parameters,
                    budget,
                ),
            )
            rows.extend(await direct_cursor.fetchall())

        normalized_query = normalize_term(query.text[:MAX_QUERY_CHARS])
        if normalized_query and len(normalized_query) <= MAX_EXACT_TERM:
            term_cursor = await connection.execute(
                f"""
                SELECT DISTINCT m.id, m.version, m.kind, m.metadata, m.recorded_from,
                       NULL AS domain_lane, 1.0::FLOAT8 AS lane_score
                FROM retrieval_exact_terms@retrieval_exact_terms_pkey AS t
                INNER LOOKUP JOIN memories@primary AS m ON m.id = t.resource_id
                {trust.joins}
                WHERE t.tenant_id = %s
                  AND t.project_id = %s
                  AND t.repository_id = %s
                  AND t.projection_id = %s
                  AND t.scope_key = %s
                  AND t.normalized_term = %s
                  AND t.resource_type = 'memory'
                  AND {" AND ".join(predicates.clauses)}
                  {f"AND {trust.clause}" if trust.clause else ""}
                ORDER BY m.recorded_from DESC, m.id
                LIMIT %s
                """,
                (
                    *trust.parameters,
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    RETRIEVAL_PROJECTION_ID,
                    scope_key,
                    normalized_query,
                    *predicates.parameters,
                    budget,
                ),
            )
            rows.extend(await term_cursor.fetchall())
        return rows


def cockroach_retrieval_gateways(
    database: CockroachDatabase,
) -> tuple[CockroachRetrievalGateway, ...]:
    return tuple(
        CockroachRetrievalGateway(database, signal)
        for signal in (
            RetrievalSignal.EXACT,
            RetrievalSignal.LEXICAL,
            RetrievalSignal.FUZZY,
        )
    )


def _scope_keys(plan: RetrievalPlan) -> tuple[str, ...]:
    scope = plan.hard_scope
    keys: list[str] = []
    if Visibility.REPOSITORY in scope.visibilities:
        keys.append(f"repository:{scope.repository_id}")
    if Visibility.RUN in scope.visibilities:
        keys.append(f"run:{scope.run_id}")
    if Visibility.TASK in scope.visibilities and scope.task_id is not None:
        keys.append(f"task:{scope.task_id}")
    return tuple(keys)


def _safe_or_tsquery(value: str) -> str:
    """Build TSQUERY syntax exclusively from normalized lexer tokens."""

    normalized = normalize_term(value[:MAX_QUERY_CHARS])
    tokens = tuple(dict.fromkeys(_FTS_TOKEN.findall(normalized)))[:MAX_QUERY_TOKENS]
    return " | ".join(f"'{token}'" for token in tokens)


__all__ = [
    "CockroachRetrievalGateway",
    "CandidateTrustFilter",
    "RecallPredicates",
    "build_candidate_trust_filter",
    "build_recall_predicates",
    "cockroach_retrieval_connection",
    "cockroach_retrieval_gateways",
    "cockroach_retrieval_now",
    "cockroach_retrieval_snapshot",
    "upsert_memory_retrieval_projection",
]
