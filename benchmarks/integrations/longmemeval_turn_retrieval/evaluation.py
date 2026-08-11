"""Gold-session diagnostics kept strictly downstream of candidate fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import TurnRetrievalError, sha256_json
from .fusion import FusedTurnCandidate, TurnFusionResult


def _session_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TurnRetrievalError("gold session IDs must be non-empty strings")
    if value != value.strip():
        raise TurnRetrievalError("gold session IDs cannot have surrounding whitespace")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise TurnRetrievalError("gold session IDs cannot contain control characters")
    return value


@dataclass(frozen=True, slots=True)
class RecallStage:
    candidate_turns: int
    candidate_sessions: int
    recalled_gold_sessions: int
    gold_session_recall: float
    any_gold_session_recalled: bool
    all_gold_sessions_recalled: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "candidate_turns": self.candidate_turns,
            "candidate_sessions": self.candidate_sessions,
            "recalled_gold_sessions": self.recalled_gold_sessions,
            "gold_session_recall": self.gold_session_recall,
            "any_gold_session_recalled": self.any_gold_session_recalled,
            "all_gold_sessions_recalled": self.all_gold_sessions_recalled,
        }


@dataclass(frozen=True, slots=True)
class GoldSessionRecallEvaluation:
    question_id: str
    fusion_trace_sha256: str
    gold_session_count: int
    gold_session_ids_sha256: str
    pre_cap: RecallStage
    post_cap: RecallStage

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "fusion_trace_sha256": self.fusion_trace_sha256,
            "gold_session_count": self.gold_session_count,
            "gold_session_ids_sha256": self.gold_session_ids_sha256,
            "pre_cap": self.pre_cap.as_dict(),
            "post_cap": self.post_cap.as_dict(),
            "candidate_generation": {
                "gold_fields_used": False,
                "evaluation_is_post_hoc": True,
            },
            "reader_or_judge_executed": False,
            "qa_improvement_proven": False,
        }


def _stage(
    candidates: tuple[FusedTurnCandidate, ...],
    *,
    gold_sessions: frozenset[str],
) -> RecallStage:
    candidate_sessions = {item.turn.parent_session_id for item in candidates}
    recalled = candidate_sessions & gold_sessions
    count = len(recalled)
    return RecallStage(
        candidate_turns=len(candidates),
        candidate_sessions=len(candidate_sessions),
        recalled_gold_sessions=count,
        gold_session_recall=count / len(gold_sessions),
        any_gold_session_recalled=bool(recalled),
        all_gold_sessions_recalled=recalled == gold_sessions,
    )


def evaluate_gold_session_recall(
    result: TurnFusionResult,
    *,
    gold_session_ids: tuple[str, ...],
) -> GoldSessionRecallEvaluation:
    """Measure cap loss after fusion; gold IDs cannot alter either ranking."""

    if not isinstance(result, TurnFusionResult):
        raise TurnRetrievalError("result must be a TurnFusionResult")
    if not isinstance(gold_session_ids, tuple) or not gold_session_ids:
        raise TurnRetrievalError("gold_session_ids must be a non-empty immutable tuple")
    checked = tuple(_session_id(value) for value in gold_session_ids)
    if len(set(checked)) != len(checked):
        raise TurnRetrievalError("gold_session_ids cannot contain duplicates")
    available_sessions = {turn.parent_session_id for turn in result.question_turns}
    unknown = set(checked) - available_sessions
    if unknown:
        raise TurnRetrievalError("gold_session_ids contain a session outside the question corpus")
    gold = frozenset(checked)
    canonical_gold = sorted(gold)
    return GoldSessionRecallEvaluation(
        question_id=result.question_id,
        fusion_trace_sha256=result.trace_sha256,
        gold_session_count=len(gold),
        gold_session_ids_sha256=sha256_json(canonical_gold),
        pre_cap=_stage(result.pre_cap_candidates, gold_sessions=gold),
        post_cap=_stage(result.candidates, gold_sessions=gold),
    )


__all__ = [
    "GoldSessionRecallEvaluation",
    "RecallStage",
    "evaluate_gold_session_recall",
]
