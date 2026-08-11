"""Pure weighted-RRF fusion and content-free E1-A trace generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from benchmarks.integrations.longmemeval_turns import (
    LongMemEvalTurnId,
    TurnProjection,
    TurnProjectionCorpus,
)

from .contracts import (
    ARTIFACT_TYPE,
    E1A_PROTOCOL,
    SCHEMA_VERSION,
    RankedLaneObservation,
    RetrievalLane,
    TurnRetrievalError,
    TurnRetrievalProtocol,
    canonical_json_bytes,
    sha256_json,
)


def _turn_id_payload(turn_id: LongMemEvalTurnId) -> list[str | int]:
    return list(turn_id.as_tuple())


@dataclass(frozen=True, slots=True)
class QueryDigest:
    question_id: str
    utf8_bytes: int
    sha256: str

    @classmethod
    def from_text(cls, *, question_id: str, query_text: str) -> QueryDigest:
        if not isinstance(query_text, str) or not query_text:
            raise TurnRetrievalError("query_text must be a non-empty string")
        try:
            raw = query_text.encode("utf-8")
        except UnicodeError as exc:
            raise TurnRetrievalError("query_text must be valid UTF-8") from exc
        return cls(
            question_id=question_id,
            utf8_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "utf8_bytes": self.utf8_bytes,
            "sha256": self.sha256,
            "normalization": "none-exact-utf8",
            "query_text_stored": False,
        }


@dataclass(frozen=True, slots=True)
class LaneContribution:
    lane: RetrievalLane
    rank: int
    lane_weight: float
    raw_score: float
    rrf_contribution: float

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "lane": self.lane.value,
            "rank": self.rank,
            "lane_weight": self.lane_weight,
            "raw_score": self.raw_score,
            "rrf_contribution": self.rrf_contribution,
        }


@dataclass(frozen=True, slots=True)
class FusedTurnCandidate:
    turn: TurnProjection
    fused_rank: int
    raw_rrf: float
    contributions: tuple[LaneContribution, ...]

    @property
    def turn_id(self) -> LongMemEvalTurnId:
        return self.turn.turn_id

    @property
    def candidate_payload_sha256(self) -> str:
        return self.turn.serialized_document_utf8.sha256

    @property
    def candidate_payload_utf8_bytes(self) -> int:
        return self.turn.serialized_document_utf8.bytes

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "turn_id": _turn_id_payload(self.turn_id),
            "fused_rank": self.fused_rank,
            "raw_rrf": self.raw_rrf,
            "parent_session_id": self.turn.parent_session_id,
            "candidate_payload_utf8": self.turn.serialized_document_utf8.as_dict(),
            "contributions": [item.content_free_binding() for item in self.contributions],
        }


@dataclass(frozen=True, slots=True)
class LaneTrace:
    lane: RetrievalLane
    requested_depth: int
    weight: float
    returned_count: int
    examined_count: int
    query_sha256: str
    turn_corpus_projection_sha256: str
    identity: dict[str, Any]
    candidate_ids_sha256: str
    raw_observations_sha256: str
    candidate_payloads_sha256: str
    candidate_payload_utf8_bytes: int
    lane_trace_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane.value,
            "requested_depth": self.requested_depth,
            "weight": self.weight,
            "returned_count": self.returned_count,
            "examined_count": self.examined_count,
            "query_sha256": self.query_sha256,
            "turn_corpus_projection_sha256": self.turn_corpus_projection_sha256,
            "identity": self.identity,
            "candidate_ids_sha256": self.candidate_ids_sha256,
            "raw_observations_sha256": self.raw_observations_sha256,
            "candidate_payloads_sha256": self.candidate_payloads_sha256,
            "candidate_payload_utf8_bytes": self.candidate_payload_utf8_bytes,
            "lane_trace_sha256": self.lane_trace_sha256,
        }


@dataclass(frozen=True, slots=True)
class TurnFusionResult:
    """Hydrated fused candidates plus a trace that never contains their text."""

    question_id: str
    question_turns: tuple[TurnProjection, ...]
    pre_cap_candidates: tuple[FusedTurnCandidate, ...]
    candidates: tuple[FusedTurnCandidate, ...]
    _trace_canonical_json: str

    @property
    def trace_sha256(self) -> str:
        return hashlib.sha256(self._trace_canonical_json.encode("utf-8")).hexdigest()

    def content_free_trace(self) -> dict[str, Any]:
        # Decode a fresh object so mutating a caller's exported artifact can
        # never mutate the frozen result or change a later digest.
        payload = json.loads(self._trace_canonical_json)
        return {**payload, "trace_sha256": self.trace_sha256}


def _question_turns(
    corpus: TurnProjectionCorpus,
    question_id: str,
) -> tuple[TurnProjection, ...]:
    if not isinstance(corpus, TurnProjectionCorpus):
        raise TurnRetrievalError("turn corpus must be an immutable TurnProjectionCorpus")
    question = next((item for item in corpus.questions if item.question_id == question_id), None)
    if question is None:
        raise TurnRetrievalError("question_id does not exist in the immutable turn corpus")
    turns = tuple(turn for turn in corpus.turns if turn.turn_id.question_id == question_id)
    if len(turns) != question.turns:
        raise TurnRetrievalError("question turn count differs from its immutable corpus binding")
    return turns


def _lane_trace(
    observation: RankedLaneObservation,
    *,
    weight: float,
    turns_by_id: dict[LongMemEvalTurnId, TurnProjection],
) -> LaneTrace:
    ids = [_turn_id_payload(item.turn_id) for item in observation.candidates]
    raw = [
        item.content_free_binding(rank=rank)
        for rank, item in enumerate(observation.candidates, start=1)
    ]
    payloads = [
        {
            "turn_id": _turn_id_payload(item.turn_id),
            "serialized_document_utf8": turns_by_id[
                item.turn_id
            ].serialized_document_utf8.as_dict(),
        }
        for item in observation.candidates
    ]
    base = {
        "lane": observation.lane.value,
        "requested_depth": observation.requested_depth,
        "weight": weight,
        "returned_count": len(observation.candidates),
        "examined_count": observation.examined_count,
        "query_sha256": observation.query_sha256,
        "turn_corpus_projection_sha256": observation.turn_corpus_projection_sha256,
        "identity": observation.identity.content_free_binding(),
        "candidate_ids_sha256": sha256_json(ids),
        "raw_observations_sha256": sha256_json(raw),
        "candidate_payloads_sha256": sha256_json(payloads),
        "candidate_payload_utf8_bytes": sum(
            turns_by_id[item.turn_id].serialized_document_utf8.bytes
            for item in observation.candidates
        ),
    }
    return LaneTrace(
        lane=observation.lane,
        requested_depth=observation.requested_depth,
        weight=weight,
        returned_count=len(observation.candidates),
        examined_count=observation.examined_count,
        query_sha256=observation.query_sha256,
        turn_corpus_projection_sha256=observation.turn_corpus_projection_sha256,
        identity=observation.identity.content_free_binding(),
        candidate_ids_sha256=base["candidate_ids_sha256"],
        raw_observations_sha256=base["raw_observations_sha256"],
        candidate_payloads_sha256=base["candidate_payloads_sha256"],
        candidate_payload_utf8_bytes=base["candidate_payload_utf8_bytes"],
        lane_trace_sha256=sha256_json(base),
    )


def fuse_question_turns(
    corpus: TurnProjectionCorpus,
    *,
    question_id: str,
    query_text: str,
    lexical: RankedLaneObservation,
    dense: RankedLaneObservation,
    protocol: TurnRetrievalProtocol = E1A_PROTOCOL,
) -> TurnFusionResult:
    """Fuse externally ranked lexical and dense turns without executing either lane."""

    if not isinstance(protocol, TurnRetrievalProtocol):
        raise TurnRetrievalError("protocol must be a registered TurnRetrievalProtocol")
    turns = _question_turns(corpus, question_id)
    turns_by_id = {turn.turn_id: turn for turn in turns}
    query = QueryDigest.from_text(question_id=question_id, query_text=query_text)

    observations = (lexical, dense)
    expected_lanes = (RetrievalLane.LEXICAL, RetrievalLane.DENSE)
    if tuple(item.lane for item in observations) != expected_lanes:
        raise TurnRetrievalError("fusion requires exactly one lexical lane then one dense lane")

    contributions_by_id: dict[LongMemEvalTurnId, list[LaneContribution]] = {}
    lane_traces: list[LaneTrace] = []
    for observation in observations:
        expected_depth = protocol.lane_depth(observation.lane)
        if observation.requested_depth != expected_depth:
            raise TurnRetrievalError(
                f"{observation.lane.value} requested_depth differs from the protocol hard depth"
            )
        if observation.query_sha256 != query.sha256:
            raise TurnRetrievalError(
                f"{observation.lane.value} observation is bound to a different query"
            )
        if observation.turn_corpus_projection_sha256 != corpus.projection_sha256:
            raise TurnRetrievalError(
                f"{observation.lane.value} observation is bound to a different turn projection"
            )
        if observation.examined_count > len(turns):
            raise TurnRetrievalError(
                f"{observation.lane.value} examined_count exceeds the question turn corpus"
            )
        for candidate in observation.candidates:
            if candidate.turn_id not in turns_by_id:
                raise TurnRetrievalError(
                    f"{observation.lane.value} lane contains an unknown or cross-question turn ID"
                )
        weight = protocol.lane_weight(observation.lane)
        lane_traces.append(_lane_trace(observation, weight=weight, turns_by_id=turns_by_id))
        for rank, candidate in enumerate(observation.candidates, start=1):
            contributions_by_id.setdefault(candidate.turn_id, []).append(
                LaneContribution(
                    lane=observation.lane,
                    rank=rank,
                    lane_weight=weight,
                    raw_score=candidate.raw_score,
                    rrf_contribution=weight / (protocol.rrf_k + rank),
                )
            )

    ordered = sorted(
        contributions_by_id,
        key=lambda turn_id: (
            -sum(item.rrf_contribution for item in contributions_by_id[turn_id]),
            turn_id,
        ),
    )
    pre_cap_candidates = tuple(
        FusedTurnCandidate(
            turn=turns_by_id[turn_id],
            fused_rank=rank,
            raw_rrf=sum(item.rrf_contribution for item in contributions_by_id[turn_id]),
            contributions=tuple(
                sorted(contributions_by_id[turn_id], key=lambda item: item.lane.value)
            ),
        )
        for rank, turn_id in enumerate(ordered, start=1)
    )
    candidates = pre_cap_candidates[: protocol.fused_candidate_cap]

    question_projection_payload = [turn.content_free_binding() for turn in turns]
    pre_cap_payloads = [
        {
            "turn_id": _turn_id_payload(item.turn_id),
            "serialized_document_utf8": item.turn.serialized_document_utf8.as_dict(),
        }
        for item in pre_cap_candidates
    ]
    post_cap_payloads = pre_cap_payloads[: protocol.fused_candidate_cap]
    fused_bindings = [item.content_free_binding() for item in pre_cap_candidates]
    post_cap_bindings = fused_bindings[: protocol.fused_candidate_cap]
    trace_payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol.as_dict(),
        "question": {
            "question_id": question_id,
            "source_record": next(
                item.source_record.as_dict()
                for item in corpus.questions
                if item.question_id == question_id
            ),
            "question_turn_count": len(turns),
            "question_turn_projection_sha256": sha256_json(question_projection_payload),
            "whole_corpus_projection_sha256": corpus.projection_sha256,
        },
        "query": query.content_free_binding(),
        "lanes": [item.as_dict() for item in lane_traces],
        "fusion": {
            "method": "weighted-reciprocal-rank-fusion",
            "rrf_k": protocol.rrf_k,
            "tie_break": "canonical-turn-id-ascending",
            "pre_cap_count": len(pre_cap_candidates),
            "post_cap_count": len(candidates),
            "dropped_by_cap": len(pre_cap_candidates) - len(candidates),
            "pre_cap_order_sha256": sha256_json(fused_bindings),
            "post_cap_order_sha256": sha256_json(post_cap_bindings),
            "pre_cap_candidate_payloads_sha256": sha256_json(pre_cap_payloads),
            "post_cap_candidate_payloads_sha256": sha256_json(post_cap_payloads),
            "pre_cap_candidates": fused_bindings,
            "post_cap_candidates": post_cap_bindings,
        },
        "accounting": {
            "question_corpus_turns": len(turns),
            "lexical_returned": len(lexical.candidates),
            "dense_returned": len(dense.candidates),
            "external_raw_scores": len(lexical.candidates) + len(dense.candidates),
            "cross_lane_overlap": (
                len(lexical.candidates) + len(dense.candidates) - len(pre_cap_candidates)
            ),
            "unique_pre_cap_candidates": len(pre_cap_candidates),
            "unique_post_cap_candidates": len(candidates),
            "pre_cap_candidate_payload_utf8_bytes": sum(
                item.candidate_payload_utf8_bytes for item in pre_cap_candidates
            ),
            "post_cap_candidate_payload_utf8_bytes": sum(
                item.candidate_payload_utf8_bytes for item in candidates
            ),
            "local_lexical_calls": 0,
            "local_embedding_calls": 0,
            "local_model_calls": 0,
            "local_database_calls": 0,
            "local_network_calls": 0,
        },
        "claims": {
            "external_lane_identities_verified": False,
            "lexical_scores_recomputed": False,
            "dense_vectors_recomputed": False,
            "dense_scores_recomputed": False,
            "gold_fields_present_in_fusion_input": False,
            "reader_or_judge_executed": False,
            "qa_improvement_proven": False,
        },
    }
    # A final serialization check also prevents an accidental NaN from entering
    # a durable evidence artifact through derived arithmetic.
    trace_canonical_json = canonical_json_bytes(trace_payload).decode("utf-8")
    return TurnFusionResult(
        question_id=question_id,
        question_turns=turns,
        pre_cap_candidates=pre_cap_candidates,
        candidates=candidates,
        _trace_canonical_json=trace_canonical_json,
    )


__all__ = [
    "FusedTurnCandidate",
    "LaneContribution",
    "LaneTrace",
    "QueryDigest",
    "TurnFusionResult",
    "fuse_question_turns",
]
