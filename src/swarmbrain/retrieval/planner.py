"""Server-owned, deterministic retrieval plans for the v1 query profiles."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
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

from .projection import MAX_QUERY_CHARS

_HEX_DIGEST = re.compile(r"(?i)^[0-9a-f]{64}$")
_CODE_LOOKUP = re.compile(r"(?:[/\\]|::|\btest_[A-Za-z0-9_]+\b|\bSQLSTATE\b|\b[0-9a-fA-F]{7,40}\b)")
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
        selected = frozenset(
            signal
            for signal in (
                RetrievalSignal.EXACT,
                RetrievalSignal.LEXICAL,
                RetrievalSignal.FUZZY,
            )
            if signal in available
        )
        weights = self._weights(purpose, selected)
        base_budget = min(2000, max(32, query.limit * 8))
        budgets = {
            RetrievalSignal.EXACT.value: min(2000, max(32, query.limit * 4)),
            RetrievalSignal.LEXICAL.value: base_budget,
            RetrievalSignal.FUZZY.value: min(2000, max(24, query.limit * 5)),
        }
        selected_seeds = tuple(dict.fromkeys((*seed_memory_ids, *sorted(query.memory_ids))))
        return RetrievalPlan(
            purpose=purpose,
            intent=self._intent(query),
            domain_lanes=self._domain_lanes(purpose),
            signal_lanes=selected,
            world_at=query.world_at,
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
            max_graph_hops=0,
            rerank=False,
            diversify=purpose in {RetrievalPurpose.PLANNING, RetrievalPurpose.TASK_BOOTSTRAP},
        )

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
        if query.world_at is not None or query.recorded_at is not None:
            return "historical"
        return "general"

    @staticmethod
    def _weights(
        purpose: RetrievalPurpose,
        signals: frozenset[RetrievalSignal],
    ) -> dict[str, float]:
        if purpose is RetrievalPurpose.TASK_BOOTSTRAP:
            configured = {
                RetrievalSignal.EXACT: 6.0,
                RetrievalSignal.LEXICAL: 3.0,
                RetrievalSignal.FUZZY: 1.0,
            }
        elif purpose is RetrievalPurpose.HISTORICAL_AUDIT:
            configured = {
                RetrievalSignal.EXACT: 5.0,
                RetrievalSignal.LEXICAL: 4.0,
                RetrievalSignal.FUZZY: 0.75,
            }
        else:
            configured = {
                RetrievalSignal.EXACT: 5.0,
                RetrievalSignal.LEXICAL: 3.0,
                RetrievalSignal.FUZZY: 1.0,
            }
        return {signal.value: configured[signal] for signal in signals}

    @staticmethod
    def _domain_lanes(purpose: RetrievalPurpose) -> frozenset[str]:
        if purpose is RetrievalPurpose.TASK_BOOTSTRAP:
            return frozenset({"handoff", "playbook", "execution_history", "knowledge"})
        if purpose is RetrievalPurpose.HANDOFF_RECOVERY:
            return frozenset({"handoff", "execution_history", "playbook"})
        return frozenset({"knowledge", "playbook", "execution_history", "handoff"})


__all__ = ["RetrievalPlanner", "parse_query_identifiers"]
