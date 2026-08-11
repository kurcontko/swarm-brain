"""Server-owned, deterministic retrieval plans for the v1 query profiles."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import datetime
from uuid import UUID

from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import MemoryId
from swarmbrain.domain.memory import RecallQuery
from swarmbrain.domain.retrieval import (
    RetrievalPlan,
    RetrievalPurpose,
    RetrievalScope,
    RetrievalSignal,
)

from .graph import GRAPH_LINK_TYPES
from .projection import MAX_QUERY_CHARS

_HEX_DIGEST = re.compile(r"(?i)^[0-9a-f]{64}$")
_CODE_LOOKUP = re.compile(r"(?:[/\\]|::|\btest_[A-Za-z0-9_]+\b|\bSQLSTATE\b|\b[0-9a-fA-F]{7,40}\b)")
TEMPORAL_PROJECTION_ID = "memory-valid-time"
TEMPORAL_PROJECTION_VERSION = "valid-from-distance-days-v1"
OCCURRENCE_TEMPORAL_PROJECTION_ID = "memory-event-occurrence-time"
OCCURRENCE_TEMPORAL_PROJECTION_VERSION = "event-occurrence-distance-days-v1"
TEMPORAL_SCORE_SCALE_SECONDS = 86_400.0
_QUERY_IDENTIFIER = re.compile(
    r"(?P<uuid>\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b)"
    r"|(?P<sqlstate>\bSQLSTATE\s+[0-9A-Z]{5}\b)"
    r"|(?P<path>(?<![\w./-])(?:[\w.@+-]+/)+[\w.@+-]+)"
    r"|(?P<symbol>\b[A-Za-z_]\w*(?:(?:::|\.)[A-Za-z_]\w*)+\b)"
    r"|(?P<test>\btest_[A-Za-z0-9_]+\b)"
    r"|(?P<hash>\b[0-9a-fA-F]{7,64}\b)",
    flags=re.IGNORECASE,
)


def parse_query_identifiers(value: str) -> tuple[str, ...]:
    """Extract bounded, normalized identifiers for internal retrieval traces."""

    found: list[str] = []
    for match in _QUERY_IDENTIFIER.finditer(value[:MAX_QUERY_CHARS]):
        identifier = match.group(0)
        if match.lastgroup == "uuid":
            identifier = str(UUID(identifier))
        elif match.lastgroup == "hash":
            identifier = identifier.lower()
        elif match.lastgroup == "sqlstate":
            identifier = " ".join(identifier.upper().split())
        found.append(identifier)
    return tuple(dict.fromkeys(found))


def temporal_query_target(query: RecallQuery) -> tuple[datetime, str] | None:
    """Return an explicit temporal ranking target and its trace label.

    System time (``recorded_at``) is intentionally absent.  It controls which
    recorded version is canonically eligible but is never an event-time proxy.
    A caller-selected occurrence prior takes precedence because it is the only
    target that ranks the distinct provenance-backed ``Memory.occurred_at``
    signal.  It remains orthogonal to hard world-validity selection.
    """

    if query.occurrence_time_prior_from is not None:
        assert query.occurrence_time_prior_to is not None
        center = (
            query.occurrence_time_prior_from
            + (query.occurrence_time_prior_to - query.occurrence_time_prior_from) / 2
        )
        return center, "occurrence_interval_center"
    if query.world_at is not None:
        return query.world_at, "world_at"
    if query.referenced_valid_from is None:
        return None
    assert query.referenced_valid_to is not None
    center = (
        query.referenced_valid_from + (query.referenced_valid_to - query.referenced_valid_from) / 2
    )
    return center, "interval_center"


def temporal_valid_from_score(valid_from: datetime, target: datetime) -> float:
    """Rank-independent closeness of ``valid_from`` to an explicit target.

    One day of distance scores ``0.5``.  Python datetimes have a finite range,
    so the reciprocal remains finite and strictly positive for every canonical
    memory while an exact target scores ``1.0``.
    """

    distance_seconds = abs((valid_from - target).total_seconds())
    return 1.0 / (1.0 + distance_seconds / TEMPORAL_SCORE_SCALE_SECONDS)


class RetrievalPlanner:
    def plan(
        self,
        actor: ActorContext,
        query: RecallQuery,
        *,
        purpose: RetrievalPurpose,
        available_signals: Iterable[RetrievalSignal],
        seed_memory_ids: Sequence[MemoryId] = (),
    ) -> RetrievalPlan:
        available = frozenset(available_signals)
        intent = self._intent(query)
        primary = frozenset(
            signal
            for signal in (
                RetrievalSignal.EXACT,
                RetrievalSignal.LEXICAL,
                RetrievalSignal.FUZZY,
                RetrievalSignal.DENSE,
            )
            if signal in available
        )
        if RetrievalSignal.TEMPORAL in available and temporal_query_target(query) is not None:
            primary |= frozenset({RetrievalSignal.TEMPORAL})
        max_graph_hops = self._graph_hops(purpose) if RetrievalSignal.GRAPH in available else 0
        selected = primary | (
            frozenset({RetrievalSignal.GRAPH}) if max_graph_hops > 0 else frozenset()
        )
        weights = self._weights(purpose, intent, selected)
        base_budget = min(2000, max(32, query.limit * 8))
        budgets = {
            RetrievalSignal.EXACT.value: min(2000, max(32, query.limit * 4)),
            RetrievalSignal.LEXICAL.value: base_budget,
            RetrievalSignal.FUZZY.value: min(2000, max(24, query.limit * 5)),
            # The compatibility embedding port and public recall limit are
            # both bounded at 100.  Scope branches may over-fetch internally,
            # but the fused dense lane remains deliberately small.
            RetrievalSignal.DENSE.value: min(100, max(32, query.limit * 8)),
            RetrievalSignal.TEMPORAL.value: min(200, max(16, query.limit * 4)),
            RetrievalSignal.GRAPH.value: min(200, max(16, query.limit * 4)),
        }
        graph_budget = budgets[RetrievalSignal.GRAPH.value]
        selected_seeds = tuple(dict.fromkeys((*seed_memory_ids, *sorted(query.memory_ids))))
        rerank_alpha = self._rerank_alpha(purpose)
        # Four times the public limit, floored at 32.  The measured curve is
        # flat from 20 upward on both tracks, so the window is set by what the
        # caller asks for rather than tuned to a peak; below roughly twice the
        # limit the stage has nothing to promote from.
        rerank_window = min(512, max(32, query.limit * 4)) if rerank_alpha > 0.0 else 0
        return RetrievalPlan(
            purpose=purpose,
            intent=intent,
            domain_lanes=self._domain_lanes(purpose),
            signal_lanes=selected,
            world_at=query.world_at,
            referenced_valid_from=query.referenced_valid_from,
            referenced_valid_to=query.referenced_valid_to,
            occurrence_time_prior_from=query.occurrence_time_prior_from,
            occurrence_time_prior_to=query.occurrence_time_prior_to,
            recorded_at=query.recorded_at,
            hard_scope=RetrievalScope(
                tenant_id=actor.tenant_id,
                project_id=actor.project_id,
                repository_id=actor.repository_id,
                swarm_id=actor.swarm_id,
                run_id=actor.run_id,
                task_id=query.task_id,
                visibilities=query.visibilities,
            ),
            lane_budgets={signal.value: budgets[signal.value] for signal in selected},
            lane_weights=weights,
            seed_memory_ids=selected_seeds,
            max_graph_hops=max_graph_hops,
            graph_seed_limit=min(16, max(4, query.limit)) if max_graph_hops else 0,
            graph_max_fanout=8 if max_graph_hops else 0,
            graph_edge_budget=(min(2000, max(64, graph_budget * 4)) if max_graph_hops else 0),
            graph_link_types=GRAPH_LINK_TYPES if max_graph_hops else frozenset(),
            rerank=rerank_alpha > 0.0,
            rerank_alpha=rerank_alpha,
            rerank_window=rerank_window,
            diversify=purpose in {RetrievalPurpose.PLANNING, RetrievalPurpose.TASK_BOOTSTRAP},
            token_budget=self._token_budget(purpose),
        )

    @staticmethod
    def _rerank_alpha(purpose: RetrievalPurpose) -> float:
        """Weight on calibrated relevance when reordering the fused head.

        Zero everywhere: the stage is implemented, tested and measured, and it
        is **not shipped**, because the independent track said not to.

        Measured 2026-08-09 at ``alpha = 0.5`` (see
        ``docs/retrieval-benchmark.md``).  On the 40-query swarm corpus it is a
        clear win — Recall@10 0.927 → 0.941, MRR 0.912 → 0.927 — and it is
        positive in all three lane configurations there.  On the 500-question
        LongMemEval-S track it is a small loss: Recall@10 0.976 → 0.971, with
        five of six question types regressing.

        The difference is structural rather than a tuning accident.  Reranking
        repairs a *consensus* pathology: candidates that one strong lane ranks
        well and several weak lanes ignore get buried by candidates that many
        lanes rank mid-field.  That pathology grows with the number of lanes
        actually returning candidates.  The swarm corpus fires all five; on
        LongMemEval only lexical and dense return anything (its session
        memories carry no identifiers and no links), fused Recall@10 is already
        0.976, and reordering can mostly only push gold out of the top ten.

        Enabling this by lane count is the obvious next hypothesis and is
        deliberately not implemented here: there is no third judged corpus left
        to validate such a gate on, and inventing one from the two datasets
        that motivated it is how a retriever gets fitted to its own benchmark.
        """

        return 0.0

    @staticmethod
    def _intent(query: RecallQuery) -> str:
        text = query.text[:MAX_QUERY_CHARS].strip()
        try:
            UUID(text)
        except ValueError:
            pass
        else:
            return "identifier"
        if _HEX_DIGEST.fullmatch(text) or _CODE_LOOKUP.search(text):
            return "code_lookup"
        if (
            query.world_at is not None
            or query.referenced_valid_from is not None
            or query.occurrence_time_prior_from is not None
            or query.recorded_at is not None
        ):
            return "historical"
        return "general"

    @staticmethod
    def _weights(
        purpose: RetrievalPurpose,
        intent: str,
        signals: frozenset[RetrievalSignal],
    ) -> dict[str, float]:
        if purpose is RetrievalPurpose.TASK_BOOTSTRAP:
            configured = {
                RetrievalSignal.EXACT: 6.0,
                RetrievalSignal.LEXICAL: 3.0,
                RetrievalSignal.FUZZY: 1.0,
                RetrievalSignal.DENSE: 3.0,
                RetrievalSignal.TEMPORAL: 2.0,
                RetrievalSignal.GRAPH: 1.5,
            }
        elif purpose is RetrievalPurpose.HISTORICAL_AUDIT:
            configured = {
                RetrievalSignal.EXACT: 5.0,
                RetrievalSignal.LEXICAL: 4.0,
                RetrievalSignal.FUZZY: 0.75,
                RetrievalSignal.DENSE: 2.0,
                RetrievalSignal.TEMPORAL: 4.0,
                RetrievalSignal.GRAPH: 1.25,
            }
        else:
            # Interactive graph weight is deliberately small.  Measured on the
            # swarm corpus with a semantic embedder (2026-08-09): at 1.75 the
            # graph lane lowered final MRR@10 by 0.11 versus direct fusion and
            # improved no query, because strong direct lanes already surface
            # linked evidence and expansion mostly promotes connected decoys.
            # Purposes built around traversal (bootstrap, handoff, planning,
            # conflict review) keep their higher weights and two hops.
            if purpose is RetrievalPurpose.CONFLICT_REVIEW:
                graph_weight = 2.5
            elif purpose in {RetrievalPurpose.PLANNING, RetrievalPurpose.HANDOFF_RECOVERY}:
                graph_weight = 1.5
            else:
                graph_weight = 0.5
            configured = {
                RetrievalSignal.EXACT: 5.0,
                RetrievalSignal.LEXICAL: 3.0,
                RetrievalSignal.FUZZY: 1.0,
                RetrievalSignal.DENSE: 4.0,
                RetrievalSignal.TEMPORAL: 2.0,
                RetrievalSignal.GRAPH: graph_weight,
            }
        # Literal identifiers are better served by exact and fuzzy indexes;
        # dense still contributes for mixed natural-language/code queries but
        # cannot displace a direct lookup merely because its ANN rank is high.
        if intent in {"identifier", "code_lookup"}:
            configured[RetrievalSignal.DENSE] = 1.0
            configured[RetrievalSignal.GRAPH] = min(configured[RetrievalSignal.GRAPH], 1.25)
        return {signal.value: configured[signal] for signal in signals}

    @staticmethod
    def _graph_hops(purpose: RetrievalPurpose) -> int:
        if purpose in {
            RetrievalPurpose.TASK_BOOTSTRAP,
            RetrievalPurpose.HANDOFF_RECOVERY,
            RetrievalPurpose.PLANNING,
            RetrievalPurpose.CONFLICT_REVIEW,
        }:
            return 2
        return 1

    @staticmethod
    def _token_budget(purpose: RetrievalPurpose) -> int | None:
        """Bound server-injected context while preserving interactive v1 recall.

        Interactive recall remains unbounded for wire compatibility.  Internal
        activation purposes are server-owned and therefore safe to budget.
        """

        if purpose in {RetrievalPurpose.TASK_BOOTSTRAP, RetrievalPurpose.HANDOFF_RECOVERY}:
            return 2048
        if purpose in {RetrievalPurpose.PLANNING, RetrievalPurpose.CONFLICT_REVIEW}:
            return 4096
        return None

    @staticmethod
    def _domain_lanes(purpose: RetrievalPurpose) -> frozenset[str]:
        if purpose is RetrievalPurpose.TASK_BOOTSTRAP:
            return frozenset({"handoff", "playbook", "execution_history", "knowledge"})
        if purpose is RetrievalPurpose.HANDOFF_RECOVERY:
            return frozenset({"handoff", "execution_history", "playbook"})
        return frozenset({"knowledge", "playbook", "execution_history", "handoff"})


__all__ = [
    "OCCURRENCE_TEMPORAL_PROJECTION_ID",
    "OCCURRENCE_TEMPORAL_PROJECTION_VERSION",
    "TEMPORAL_PROJECTION_ID",
    "TEMPORAL_PROJECTION_VERSION",
    "TEMPORAL_SCORE_SCALE_SECONDS",
    "RetrievalPlanner",
    "parse_query_identifiers",
    "temporal_query_target",
    "temporal_valid_from_score",
]
