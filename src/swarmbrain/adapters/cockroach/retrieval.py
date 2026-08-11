"""CockroachDB-native exact, FTS, trigram, and dense candidate lanes."""

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
    DenseQuery,
    FusedCandidate,
    RetrievalPlan,
    RetrievalSignal,
)
from swarmbrain.retrieval.dense import DENSE_PROJECTION_ID, dense_projection_signature
from swarmbrain.retrieval.graph import (
    GRAPH_FANOUT_SCAN_MULTIPLIER,
    GRAPH_PROJECTION_ID,
    GRAPH_PROJECTION_VERSION,
    GRAPH_RELATION_WEIGHTS,
    GraphTraversalState,
    graph_query_gate,
    graph_state_is_better,
    graph_step_activation,
    select_graph_seeds,
)
from swarmbrain.retrieval.planner import (
    OCCURRENCE_TEMPORAL_PROJECTION_ID,
    OCCURRENCE_TEMPORAL_PROJECTION_VERSION,
    TEMPORAL_PROJECTION_ID,
    TEMPORAL_PROJECTION_VERSION,
    temporal_query_target,
    temporal_valid_from_score,
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
from .vector import COCKROACH_VECTOR_DIMENSIONS, vector_literal

_FTS_TOKEN = re.compile(r"\w+", flags=re.UNICODE)
_GRAPH_RELATION_PRIORITY_SQL = (
    "CASE "
    + " ".join(
        f"WHEN incident.link_type = '{relation}' THEN {priority}"
        for priority, (relation, _weight) in enumerate(
            sorted(GRAPH_RELATION_WEIGHTS.items(), key=lambda item: (-item[1], item[0])),
            start=1,
        )
    )
    + " ELSE 0 END"
)


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

    if query.referenced_valid_from is not None:
        assert query.referenced_valid_to is not None
        clauses.extend(
            (
                f"{alias}.valid_from < %s",
                f"({alias}.valid_to IS NULL OR %s < {alias}.valid_to)",
            )
        )
        parameters.extend((query.referenced_valid_to, query.referenced_valid_from))
    else:
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
        dense_query: DenseQuery | None = None,
    ) -> CandidateBatch:
        del dense_query
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


class CockroachTemporalRetrievalGateway:
    """Single-query explicit temporal lane over canonical memory rows."""

    signal = RetrievalSignal.TEMPORAL

    def __init__(self, database: CockroachDatabase) -> None:
        self.database = database

    async def retrieve(
        self,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
        dense_query: DenseQuery | None = None,
    ) -> CandidateBatch:
        del dense_query
        started = perf_counter()
        target_with_kind = temporal_query_target(query)
        if target_with_kind is None:
            return CandidateBatch(
                lane=self.signal,
                examined_count=0,
                latency_ms=(perf_counter() - started) * 1000.0,
                projection_watermark=TEMPORAL_PROJECTION_VERSION,
            )

        target, target_kind = target_with_kind
        budget = plan.lane_budgets[self.signal.value]
        occurrence_prior = query.occurrence_time_prior_from is not None
        trust = build_candidate_trust_filter(query)
        async with cockroach_retrieval_connection(self.database) as connection:
            # A valid-time target is itself the canonical world-time selector.
            # An occurrence target is only a ranking hint, so current recall
            # must still use the stable database clock for canonical validity.
            predicate_now = (
                await cockroach_retrieval_now(self.database, connection)
                if occurrence_prior
                else target
            )
            predicates = build_recall_predicates(
                actor,
                query,
                now=predicate_now,
                apply_trust=False,
            )
            temporal_column = "occurred_at" if occurrence_prior else "valid_from"
            known_occurrence_clause = "AND m.occurred_at IS NOT NULL" if occurrence_prior else ""
            cursor = await connection.execute(
                f"""
                SELECT m.id, m.version, m.kind, m.metadata, m.{temporal_column},
                       m.recorded_from
                FROM memories@primary AS m
                {trust.joins}
                WHERE {" AND ".join(predicates.clauses)}
                  {f"AND {trust.clause}" if trust.clause else ""}
                  {known_occurrence_clause}
                ORDER BY abs(
                    EXTRACT(EPOCH FROM m.{temporal_column})
                    - EXTRACT(EPOCH FROM %s::TIMESTAMPTZ)
                ), m.{temporal_column}, m.id
                LIMIT %s
                """,
                (
                    *trust.parameters,
                    *predicates.parameters,
                    target,
                    budget + 1,
                ),
            )
            rows = list(await cursor.fetchall())

        scored = [(temporal_valid_from_score(row[temporal_column], target), row) for row in rows]
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1][temporal_column],
                str(item[1]["id"]),
            )
        )
        projection_id = (
            OCCURRENCE_TEMPORAL_PROJECTION_ID if occurrence_prior else TEMPORAL_PROJECTION_ID
        )
        projection_version = (
            OCCURRENCE_TEMPORAL_PROJECTION_VERSION
            if occurrence_prior
            else TEMPORAL_PROJECTION_VERSION
        )
        reason = (
            "temporal_occurrence_distance" if occurrence_prior else "temporal_valid_from_distance"
        )
        candidates = tuple(
            Candidate(
                resource_type="memory",
                resource_id=str(row["id"]),
                resource_version=int(row["version"]),
                canonical_id=str(row["id"]),
                domain_lane=domain_lane(str(row["kind"]), row.get("metadata") or {}),
                signal=self.signal,
                rank=rank,
                raw_score=raw_score,
                projection_id=projection_id,
                projection_version=projection_version,
                reasons=(
                    reason,
                    f"temporal_target:{target_kind}",
                ),
            )
            for rank, (raw_score, row) in enumerate(scored[:budget], start=1)
        )
        return CandidateBatch(
            lane=self.signal,
            candidates=candidates,
            examined_count=len(rows),
            latency_ms=(perf_counter() - started) * 1000.0,
            truncated=len(rows) > budget,
            projection_watermark=projection_version,
        )


class CockroachDenseRetrievalGateway:
    """Scope-prefixed current-memory ANN with an exact evaluation oracle."""

    signal = RetrievalSignal.DENSE

    def __init__(
        self,
        database: CockroachDatabase,
        *,
        min_similarity: float = 0.0,
        beam_size: int = 32,
        overfetch_factor: int = 4,
    ) -> None:
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("dense min_similarity must be between 0 and 1")
        if not 1 <= beam_size <= 1024:
            raise ValueError("dense beam_size must be between 1 and 1024")
        if not 1 <= overfetch_factor <= 20:
            raise ValueError("dense overfetch_factor must be between 1 and 20")
        self.database = database
        self.min_similarity = min_similarity
        self.beam_size = beam_size
        self.overfetch_factor = overfetch_factor

    async def retrieve(
        self,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
        dense_query: DenseQuery | None = None,
    ) -> CandidateBatch:
        return await self._retrieve(actor, plan, query, dense_query, exact=False)

    async def retrieve_exact(
        self,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
        dense_query: DenseQuery,
    ) -> CandidateBatch:
        """Exact-rank the canonically eligible relation as ANN ground truth."""

        return await self._retrieve(actor, plan, query, dense_query, exact=True)

    async def _retrieve(
        self,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
        dense_query: DenseQuery | None,
        *,
        exact: bool,
    ) -> CandidateBatch:
        started = perf_counter()
        if dense_query is None or dense_query.values is None:
            return CandidateBatch(
                lane=self.signal,
                examined_count=0,
                latency_ms=(perf_counter() - started) * 1000.0,
                degraded=True,
                degradation_reason=(
                    dense_query.unavailable_reason
                    if dense_query is not None
                    else "query_embedding_unavailable"
                ),
            )
        if dense_query.dimensions != COCKROACH_VECTOR_DIMENSIONS:
            return CandidateBatch(
                lane=self.signal,
                examined_count=0,
                latency_ms=(perf_counter() - started) * 1000.0,
                degraded=True,
                degradation_reason="vector_dimension_mismatch",
            )

        literal = vector_literal(dense_query.values)
        signature = dense_projection_signature(dense_query.model, dense_query.dimensions)
        budget = plan.lane_budgets[self.signal.value]
        fetch_limit = min(2000, budget * self.overfetch_factor)
        rows: list[dict[str, Any]] = []
        examined_count = 0
        branch_truncated = False
        async with cockroach_retrieval_connection(self.database) as connection:
            now = await cockroach_retrieval_now(self.database, connection)
            if not exact:
                # CockroachDB rejects a bound SET value because psycopg sends it as
                # a string. ``beam_size`` is range-checked in __init__, so rendering
                # the integer literal is both required by Cockroach and safe here.
                await connection.execute(f"SET LOCAL vector_search_beam_size = {self.beam_size}")
            for scope_key in _scope_keys(plan):
                if exact:
                    branch = await self._retrieve_scope_exact(
                        connection,
                        actor,
                        query,
                        scope_key=scope_key,
                        signature=signature,
                        literal=literal,
                        model=dense_query.model.strip(),
                        dimensions=dense_query.dimensions,
                        now=now,
                        limit=fetch_limit,
                    )
                    examined_count += len(branch)
                    may_have_more = len(branch) >= fetch_limit
                else:
                    branch, branch_examined, may_have_more = await self._retrieve_scope_ann(
                        connection,
                        actor,
                        query,
                        scope_key=scope_key,
                        signature=signature,
                        literal=literal,
                        model=dense_query.model.strip(),
                        dimensions=dense_query.dimensions,
                        now=now,
                        target=budget,
                        initial_limit=fetch_limit,
                    )
                    examined_count += branch_examined
                rows.extend(branch)
                branch_truncated = branch_truncated or may_have_more

        unique: dict[str, dict[str, Any]] = {}
        for row in rows:
            score = max(0.0, min(1.0, 1.0 - float(row["distance"])))
            if score < self.min_similarity:
                continue
            row["lane_score"] = score
            memory_id = str(row["id"])
            current = unique.get(memory_id)
            if current is None or score > float(current["lane_score"]):
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
                resource_version=int(row["resource_version"]),
                canonical_id=str(row["id"]),
                domain_lane=str(row["domain_lane"]),
                signal=self.signal,
                rank=rank,
                raw_score=float(row["lane_score"]),
                projection_id=str(row["projection_id"]),
                projection_version=str(row["projection_signature"]),
                reasons=(
                    "semantic_match",
                    "dense_cosine_exact" if exact else "dense_cosine_ann",
                ),
            )
            for rank, row in enumerate(ordered[:budget], start=1)
        )
        indexed_values = [row.get("indexed_at") for row in rows if row.get("indexed_at")]
        watermark = max(indexed_values).isoformat() if indexed_values else signature
        return CandidateBatch(
            lane=self.signal,
            candidates=candidates,
            examined_count=examined_count,
            latency_ms=(perf_counter() - started) * 1000.0,
            truncated=branch_truncated or len(ordered) > budget,
            projection_watermark=watermark,
        )

    async def _retrieve_scope_ann(
        self,
        connection: Any,
        actor: ActorContext,
        query: RecallQuery,
        *,
        scope_key: str,
        signature: str,
        literal: str,
        model: str,
        dimensions: int,
        now: Any,
        target: int,
        initial_limit: int,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        """ANN first, then canonical validation with adaptive filtered widening.

        CockroachDB vector indexes only accelerate predicates over the complete
        prefix. Lifecycle, bitemporal, kind, digest, and trust predicates must
        therefore run after ANN. We keep both phases in the same retrieval
        snapshot and widen geometrically when filtering under-fills the lane,
        mirroring iterative filtered scans without ever broadening auth scope.
        """

        fetch_limit = initial_limit
        examined_count = 0
        validated_by_id: dict[str, dict[str, Any]] = {}
        while True:
            raw = await self._retrieve_ann_candidates(
                connection,
                actor,
                scope_key=scope_key,
                signature=signature,
                literal=literal,
                limit=fetch_limit,
            )
            examined_count += len(raw)
            validated = await self._validate_ann_candidates(
                connection,
                actor,
                query,
                raw,
                scope_key=scope_key,
                model=model,
                dimensions=dimensions,
                now=now,
            )
            for row in validated:
                memory_id = str(row["id"])
                current = validated_by_id.get(memory_id)
                if current is None or float(row["distance"]) < float(current["distance"]):
                    validated_by_id[memory_id] = row
            accumulated = sorted(
                validated_by_id.values(),
                key=lambda row: (
                    float(row["distance"]),
                    -row["recorded_from"].timestamp(),
                    str(row["id"]),
                ),
            )
            saturated = len(raw) >= fetch_limit
            if len(accumulated) >= target or not saturated or fetch_limit >= 2000:
                return accumulated, examined_count, saturated
            fetch_limit = min(2000, fetch_limit * 2)

    @staticmethod
    async def _retrieve_ann_candidates(
        connection: Any,
        actor: ActorContext,
        *,
        scope_key: str,
        signature: str,
        literal: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        cursor = await connection.execute(
            """
            SELECT v.canonical_id AS id, v.resource_id, v.resource_version,
                   v.domain_lane, v.model, v.dimensions,
                   v.projection_id, v.projection_signature, v.indexed_at,
                   v.content_sha256, v.embedding <=> %s::VECTOR AS distance
            FROM retrieval_vectors_1024@retrieval_vectors_1024_ann_v2 AS v
            WHERE v.tenant_id = %s
              AND v.project_id = %s
              AND v.repository_id = %s
              AND v.resource_type = 'memory'
              AND v.projection_id = %s
              AND v.projection_signature = %s
              AND v.scope_key = %s
            ORDER BY v.embedding <=> %s::VECTOR
            LIMIT %s
            """,
            (
                literal,
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                DENSE_PROJECTION_ID,
                signature,
                scope_key,
                literal,
                limit,
            ),
        )
        return list(await cursor.fetchall())

    @staticmethod
    async def _validate_ann_candidates(
        connection: Any,
        actor: ActorContext,
        query: RecallQuery,
        rows: Sequence[dict[str, Any]],
        *,
        scope_key: str,
        model: str,
        dimensions: int,
        now: Any,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        predicates = build_recall_predicates(
            actor,
            query,
            now=now,
            candidate_ids=tuple(str(row["id"]) for row in rows),
        )
        cursor = await connection.execute(
            f"""
            SELECT m.id, m.repository_id, m.run_id, m.task_id, m.visibility,
                   m.version, m.kind, m.metadata, m.recorded_from,
                   m.normalized_sha256
            FROM memories@primary AS m
            WHERE {" AND ".join(predicates.clauses)}
            """,
            predicates.parameters,
        )
        canonical = {str(row["id"]): row for row in await cursor.fetchall()}
        validated: list[dict[str, Any]] = []
        for row in rows:
            memory = canonical.get(str(row["id"]))
            if memory is None:
                continue
            if str(row["resource_id"]) != str(row["id"]):
                continue
            canonical_scope_key = projection_scope_key(
                Visibility(str(memory["visibility"])),
                repository_id=str(memory["repository_id"]),
                run_id=str(memory["run_id"]),
                task_id=(None if memory["task_id"] is None else str(memory["task_id"])),
            )
            if canonical_scope_key != scope_key:
                continue
            if str(row["model"]).strip() != model or int(row["dimensions"]) != dimensions:
                continue
            if int(row["resource_version"]) != int(memory["version"]):
                continue
            if bytes(row["content_sha256"]) != bytes(memory["normalized_sha256"]):
                continue
            merged = dict(row)
            merged.update(
                kind=memory["kind"],
                metadata=memory["metadata"],
                recorded_from=memory["recorded_from"],
            )
            validated.append(merged)
        return validated

    async def _retrieve_scope_exact(
        self,
        connection: Any,
        actor: ActorContext,
        query: RecallQuery,
        *,
        scope_key: str,
        signature: str,
        literal: str,
        model: str,
        dimensions: int,
        now: Any,
        limit: int,
    ) -> list[dict[str, Any]]:
        predicates = build_recall_predicates(actor, query, now=now, apply_trust=False)
        trust = build_candidate_trust_filter(query)
        cursor = await connection.execute(
            f"""
            SELECT m.id, v.resource_version, m.kind, m.metadata, m.recorded_from,
                   v.domain_lane, v.projection_id, v.projection_signature,
                   v.indexed_at, v.embedding <=> %s::VECTOR AS distance
            FROM retrieval_vectors_1024@primary AS v
            INNER JOIN memories@primary AS m ON m.id = v.canonical_id
            {trust.joins}
            WHERE v.tenant_id = %s
              AND v.project_id = %s
              AND v.repository_id = %s
              AND v.projection_signature = %s
              AND v.scope_key = %s
              AND v.resource_type = 'memory'
              AND v.projection_id = %s
              AND v.resource_id = v.canonical_id
              AND v.model = %s
              AND v.dimensions = %s
              AND v.resource_version = m.version
              AND v.content_sha256 = m.normalized_sha256
              AND v.scope_key = CASE m.visibility
                  WHEN 'repository' THEN concat('repository:', m.repository_id)
                  WHEN 'run' THEN concat('run:', m.run_id)
                  ELSE concat('task:', m.task_id::STRING)
              END
              AND {" AND ".join(predicates.clauses)}
              {f"AND {trust.clause}" if trust.clause else ""}
            ORDER BY v.embedding <=> %s::VECTOR, m.recorded_from DESC, m.id
            LIMIT %s
            """,
            (
                literal,
                *trust.parameters,
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                signature,
                scope_key,
                DENSE_PROJECTION_ID,
                model,
                dimensions,
                *predicates.parameters,
                literal,
                limit,
            ),
        )
        return list(await cursor.fetchall())


class CockroachGraphRetrievalGateway:
    """Index-prefixed, bounded graph expansion under the retrieval snapshot."""

    signal = RetrievalSignal.GRAPH

    def __init__(self, database: CockroachDatabase) -> None:
        self.database = database

    async def expand(
        self,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
        seeds: Sequence[FusedCandidate],
        dense_query: DenseQuery | None = None,
    ) -> CandidateBatch:
        del dense_query
        started = perf_counter()
        selected_seeds = select_graph_seeds(plan, seeds)
        if (
            plan.max_graph_hops == 0
            or plan.graph_seed_limit == 0
            or plan.graph_max_fanout == 0
            or plan.graph_edge_budget == 0
            or not selected_seeds
        ):
            return CandidateBatch(
                lane=self.signal,
                examined_count=0,
                latency_ms=(perf_counter() - started) * 1000.0,
                projection_watermark=GRAPH_PROJECTION_VERSION,
            )

        allowed_relations = tuple(
            sorted(frozenset(plan.graph_link_types) & frozenset(GRAPH_RELATION_WEIGHTS))
        )
        if not allowed_relations:
            return CandidateBatch(
                lane=self.signal,
                examined_count=0,
                latency_ms=(perf_counter() - started) * 1000.0,
                projection_watermark=GRAPH_PROJECTION_VERSION,
            )

        examined_count = 0
        truncated = False
        traversed_at: list[Any] = []
        async with cockroach_retrieval_connection(self.database) as connection:
            now = await cockroach_retrieval_now(self.database, connection)
            seed_rows = await self._load_eligible_nodes(
                connection,
                actor,
                query,
                now=now,
                candidate_ids=tuple(memory_id for memory_id, _score in selected_seeds),
            )
            nodes = {str(row["id"]): row for row in seed_rows}
            eligible_seeds = tuple(
                (memory_id, score) for memory_id, score in selected_seeds if memory_id in nodes
            )
            if not eligible_seeds:
                return CandidateBatch(
                    lane=self.signal,
                    examined_count=0,
                    latency_ms=(perf_counter() - started) * 1000.0,
                    projection_watermark=GRAPH_PROJECTION_VERSION,
                )

            seed_ids = frozenset(memory_id for memory_id, _score in eligible_seeds)
            frontier = tuple(
                GraphTraversalState(
                    node_id=memory_id,
                    activation=score,
                    depth=0,
                    seed_id=memory_id,
                    node_path=(memory_id,),
                    trace_path=(f"memory:{memory_id}",),
                    relation_path=(),
                )
                for memory_id, score in eligible_seeds
            )
            best: dict[str, GraphTraversalState] = {state.node_id: state for state in frontier}
            edge_cutoff = query.recorded_at or now

            for _hop in range(1, plan.max_graph_hops + 1):
                raw_hop_edges: list[tuple[GraphTraversalState, dict[str, Any]]] = []
                frontier_index = 0
                while frontier_index < len(frontier) and examined_count < plan.graph_edge_budget:
                    remaining = plan.graph_edge_budget - examined_count
                    scan_cap = max(
                        plan.graph_max_fanout + 1,
                        plan.graph_max_fanout * GRAPH_FANOUT_SCAN_MULTIPLIER,
                    )
                    if remaining >= scan_cap:
                        chunk_size = min(
                            len(frontier) - frontier_index,
                            max(1, remaining // scan_cap),
                        )
                        per_node_limit = scan_cap
                    else:
                        chunk_size = 1
                        per_node_limit = remaining
                    chunk = frontier[frontier_index : frontier_index + chunk_size]
                    rows = await self._load_incident_edges(
                        connection,
                        frontier_ids=tuple(state.node_id for state in chunk),
                        link_types=allowed_relations,
                        edge_cutoff=edge_cutoff,
                        per_node_limit=per_node_limit,
                    )
                    examined_count += len(rows)
                    traversed_at.extend(row["created_at"] for row in rows)
                    by_frontier: dict[str, list[dict[str, Any]]] = {}
                    for row in rows:
                        by_frontier.setdefault(str(row["frontier_id"]), []).append(row)
                    for state in chunk:
                        incident = by_frontier.get(state.node_id, [])
                        if len(incident) >= per_node_limit:
                            truncated = True
                        raw_hop_edges.extend(
                            (state, row)
                            for row in incident
                            if str(row["other_id"]) not in state.node_path
                        )
                    frontier_index += chunk_size
                if frontier_index < len(frontier):
                    truncated = True
                if not raw_hop_edges:
                    break

                target_ids = tuple(
                    dict.fromkeys(str(row["other_id"]) for _state, row in raw_hop_edges)
                )
                target_rows = await self._load_eligible_nodes(
                    connection,
                    actor,
                    query,
                    now=now,
                    candidate_ids=target_ids,
                )
                eligible_targets = {str(row["id"]): row for row in target_rows}
                nodes.update(eligible_targets)
                next_frontier: dict[str, GraphTraversalState] = {}
                accepted_counts: dict[str, int] = {}
                accepted_neighbors: dict[str, set[str]] = {}
                for state, edge in raw_hop_edges:
                    target_id = str(edge["other_id"])
                    target = eligible_targets.get(target_id)
                    if target is None:
                        continue
                    seen_neighbors = accepted_neighbors.setdefault(state.node_id, set())
                    if target_id in seen_neighbors:
                        continue
                    seen_neighbors.add(target_id)
                    accepted = accepted_counts.get(state.node_id, 0)
                    if accepted >= plan.graph_max_fanout:
                        truncated = True
                        continue
                    accepted_counts[state.node_id] = accepted + 1
                    relation = str(edge["link_type"])
                    reverse = bool(edge["reverse"])
                    content = (
                        target.get("content_json")
                        if target.get("content_json") is not None
                        else str(target["content"])
                    )
                    activation = graph_step_activation(
                        state.activation,
                        relation,
                        reverse=reverse,
                        query_gate=graph_query_gate(
                            query.text,
                            title=target.get("title"),
                            content=content,
                            tags=tuple(str(tag) for tag in (target.get("tags") or ())),
                            metadata=dict(target.get("metadata") or {}),
                        ),
                    )
                    direction = "in" if reverse else "out"
                    candidate = GraphTraversalState(
                        node_id=target_id,
                        activation=activation,
                        depth=state.depth + 1,
                        seed_id=state.seed_id,
                        node_path=(*state.node_path, target_id),
                        trace_path=(
                            *state.trace_path,
                            f"edge:{edge['id']}:{direction}:{relation}",
                            f"memory:{target_id}",
                        ),
                        relation_path=(*state.relation_path, relation),
                    )
                    if not graph_state_is_better(candidate, best.get(target_id)):
                        continue
                    best[target_id] = candidate
                    if graph_state_is_better(candidate, next_frontier.get(target_id)):
                        next_frontier[target_id] = candidate
                if not next_frontier:
                    break
                frontier = tuple(
                    sorted(
                        next_frontier.values(),
                        key=lambda state: (-state.activation, state.depth, state.trace_path),
                    )
                )

        ordered = sorted(
            (
                state
                for memory_id, state in best.items()
                if memory_id not in seed_ids and state.depth > 0
            ),
            key=lambda state: (
                -state.activation,
                state.depth,
                -nodes[state.node_id]["recorded_from"].timestamp(),
                state.node_id,
            ),
        )
        budget = plan.lane_budgets[self.signal.value]
        candidates = tuple(
            Candidate(
                resource_type="memory",
                resource_id=state.node_id,
                resource_version=int(nodes[state.node_id]["version"]),
                canonical_id=state.node_id,
                domain_lane=domain_lane(
                    str(nodes[state.node_id]["kind"]),
                    dict(nodes[state.node_id].get("metadata") or {}),
                ),
                signal=self.signal,
                rank=rank,
                raw_score=state.activation,
                projection_id=GRAPH_PROJECTION_ID,
                projection_version=GRAPH_PROJECTION_VERSION,
                reasons=(
                    "graph_expand",
                    "graph_query_gated",
                    f"graph_hops:{state.depth}",
                    f"graph_path:{'>'.join(state.relation_path)}",
                ),
                path=state.trace_path,
            )
            for rank, state in enumerate(ordered[:budget], start=1)
        )
        watermark = max(traversed_at).isoformat() if traversed_at else GRAPH_PROJECTION_VERSION
        return CandidateBatch(
            lane=self.signal,
            candidates=candidates,
            examined_count=examined_count,
            latency_ms=(perf_counter() - started) * 1000.0,
            truncated=truncated or len(ordered) > budget,
            projection_watermark=watermark,
        )

    @staticmethod
    async def _load_eligible_nodes(
        connection: Any,
        actor: ActorContext,
        query: RecallQuery,
        *,
        now: Any,
        candidate_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        if not candidate_ids:
            return []
        predicates = build_recall_predicates(
            actor,
            query,
            now=now,
            candidate_ids=candidate_ids,
            apply_trust=False,
        )
        trust = build_candidate_trust_filter(query)
        cursor = await connection.execute(
            f"""
            SELECT m.id, m.version, m.kind, m.metadata, m.title, m.content,
                   m.content_json, m.tags, m.recorded_from
            FROM memories@primary AS m
            {trust.joins}
            WHERE {" AND ".join(predicates.clauses)}
              {f"AND {trust.clause}" if trust.clause else ""}
            ORDER BY m.id
            """,
            (*trust.parameters, *predicates.parameters),
        )
        return list(await cursor.fetchall())

    @staticmethod
    async def _load_incident_edges(
        connection: Any,
        *,
        frontier_ids: Sequence[str],
        link_types: Sequence[str],
        edge_cutoff: Any,
        per_node_limit: int,
    ) -> list[dict[str, Any]]:
        if not frontier_ids or per_node_limit <= 0:
            return []
        cursor = await connection.execute(
            f"""
            WITH frontier (node_id) AS (
                SELECT unnest(%s::UUID[])
            )
            SELECT frontier.node_id AS frontier_id, edge.id, edge.source_memory_id,
                   edge.target_memory_id, edge.other_id, edge.link_type,
                   edge.created_at, edge.reverse
            FROM frontier
            INNER JOIN LATERAL (
                SELECT incident.*, {_GRAPH_RELATION_PRIORITY_SQL} AS relation_priority
                FROM (
                    SELECT ml.id, ml.source_memory_id, ml.target_memory_id,
                           ml.target_memory_id AS other_id, ml.link_type,
                           ml.created_at, false AS reverse
                    FROM memory_links@memory_links_from_type AS ml
                    WHERE ml.source_memory_id = frontier.node_id
                      AND ml.link_type = ANY(%s::STRING[])
                      AND ml.created_at <= %s
                    UNION ALL
                    SELECT ml.id, ml.source_memory_id, ml.target_memory_id,
                           ml.source_memory_id AS other_id, ml.link_type,
                           ml.created_at, true AS reverse
                    FROM memory_links@memory_links_to_type AS ml
                    WHERE ml.target_memory_id = frontier.node_id
                      AND ml.link_type = ANY(%s::STRING[])
                      AND ml.created_at <= %s
                ) AS incident
                ORDER BY relation_priority, incident.created_at DESC,
                         incident.id, incident.other_id
                LIMIT %s
            ) AS edge ON true
            ORDER BY frontier.node_id, edge.relation_priority,
                     edge.created_at DESC, edge.id, edge.other_id
            """,
            (
                [UUID(memory_id) for memory_id in frontier_ids],
                list(link_types),
                edge_cutoff,
                list(link_types),
                edge_cutoff,
                per_node_limit,
            ),
        )
        return list(await cursor.fetchall())


def cockroach_retrieval_gateways(
    database: CockroachDatabase,
) -> tuple[
    CockroachRetrievalGateway | CockroachTemporalRetrievalGateway | CockroachGraphRetrievalGateway,
    ...,
]:
    primary = tuple(
        CockroachRetrievalGateway(database, signal)
        for signal in (
            RetrievalSignal.EXACT,
            RetrievalSignal.LEXICAL,
            RetrievalSignal.FUZZY,
        )
    )
    return (
        *primary,
        CockroachTemporalRetrievalGateway(database),
        CockroachGraphRetrievalGateway(database),
    )


def cockroach_hybrid_retrieval_gateways(
    database: CockroachDatabase,
    *,
    dense_min_similarity: float = 0.0,
    dense_beam_size: int = 32,
) -> tuple[
    CockroachRetrievalGateway
    | CockroachTemporalRetrievalGateway
    | CockroachDenseRetrievalGateway
    | CockroachGraphRetrievalGateway,
    ...,
]:
    return (
        *(
            gateway
            for gateway in cockroach_retrieval_gateways(database)
            if gateway.signal is not RetrievalSignal.GRAPH
        ),
        CockroachDenseRetrievalGateway(
            database,
            min_similarity=dense_min_similarity,
            beam_size=dense_beam_size,
        ),
        CockroachGraphRetrievalGateway(database),
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
    "CockroachDenseRetrievalGateway",
    "CockroachGraphRetrievalGateway",
    "CockroachRetrievalGateway",
    "CockroachTemporalRetrievalGateway",
    "CandidateTrustFilter",
    "RecallPredicates",
    "build_candidate_trust_filter",
    "build_recall_predicates",
    "cockroach_hybrid_retrieval_gateways",
    "cockroach_retrieval_connection",
    "cockroach_retrieval_gateways",
    "cockroach_retrieval_now",
    "cockroach_retrieval_snapshot",
    "upsert_memory_retrieval_projection",
]
