"""Pure E1-B/E1-C/E1-D ranking over a frozen E1-A turn pool."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from benchmarks.integrations.longmemeval_turn_retrieval import (
    ARTIFACT_TYPE as E1A_ARTIFACT_TYPE,
)
from benchmarks.integrations.longmemeval_turn_retrieval import (
    E1A_PROTOCOL,
    FUSED_CANDIDATE_CAP,
    FusedTurnCandidate,
    TurnFusionResult,
)
from benchmarks.integrations.longmemeval_turns import (
    LongMemEvalTurnId,
    TurnProjection,
)

from .contracts import (
    ARTIFACT_TYPE,
    E1_PROTOCOL,
    SCHEMA_VERSION,
    E1Cell,
    E1SelectionError,
    E1SelectionProtocol,
    PoolScoreObservation,
    ScoreChannel,
    canonical_json_bytes,
    checked_sha256,
    finite_score,
    sha256_json,
)


def _turn_id_payload(turn_id: LongMemEvalTurnId) -> list[str | int]:
    return list(turn_id.as_tuple())


def stable_sigmoid(logit: float) -> float:
    """Compute sigmoid without overflowing for any finite binary64 input."""

    value = finite_score(logit, label="CrossEncoder logit")
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass(frozen=True, slots=True)
class E1SelectedCandidate:
    """One hydrated output with all numeric facts needed to audit its rank."""

    turn: TurnProjection
    selected_rank: int
    e1a_fused_rank: int
    cross_encoder_logit: float
    cross_encoder_sigmoid: float
    cross_encoder_rank: int
    colbert_score: float | None = None
    colbert_rank: int | None = None
    weighted_rrf: float | None = None
    e1c_rank: int | None = None

    @property
    def turn_id(self) -> LongMemEvalTurnId:
        return self.turn.turn_id

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "turn_id": _turn_id_payload(self.turn_id),
            "candidate_payload_utf8": self.turn.serialized_document_utf8.as_dict(),
            "selected_rank": self.selected_rank,
            "e1a_fused_rank": self.e1a_fused_rank,
            "cross_encoder_logit": self.cross_encoder_logit,
            "cross_encoder_sigmoid": self.cross_encoder_sigmoid,
            "cross_encoder_rank": self.cross_encoder_rank,
            "colbert_score": self.colbert_score,
            "colbert_rank": self.colbert_rank,
            "weighted_rrf": self.weighted_rrf,
            "e1c_rank": self.e1c_rank,
        }


@dataclass(frozen=True, slots=True)
class E1SelectionResult:
    """Hydrated cell output with an immutable, content-free trace."""

    cell: E1Cell
    question_id: str
    source_pool_count: int
    candidates: tuple[E1SelectedCandidate, ...]
    _trace_canonical_json: str

    @property
    def turns(self) -> tuple[TurnProjection, ...]:
        return tuple(candidate.turn for candidate in self.candidates)

    @property
    def trace_sha256(self) -> str:
        return hashlib.sha256(self._trace_canonical_json.encode("utf-8")).hexdigest()

    def content_free_trace(self) -> dict[str, Any]:
        payload = json.loads(self._trace_canonical_json)
        return {**payload, "trace_sha256": self.trace_sha256}


@dataclass(frozen=True, slots=True)
class E1APoolBinding:
    """Public content-free values required to bind external score evidence."""

    question_id: str
    query_sha256: str
    turn_corpus_projection_sha256: str
    e1a_trace_sha256: str
    e1a_pool_sha256: str
    pool_count: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "question_id": self.question_id,
            "query_sha256": self.query_sha256,
            "turn_corpus_projection_sha256": self.turn_corpus_projection_sha256,
            "e1a_trace_sha256": self.e1a_trace_sha256,
            "e1a_pool_sha256": self.e1a_pool_sha256,
            "pool_count": self.pool_count,
        }


@dataclass(frozen=True, slots=True)
class _SourceContext:
    question_id: str
    query_sha256: str
    turn_corpus_projection_sha256: str
    e1a_trace_sha256: str
    pool_sha256: str
    pool: tuple[FusedTurnCandidate, ...]

    @property
    def pool_by_id(self) -> dict[LongMemEvalTurnId, FusedTurnCandidate]:
        return {candidate.turn_id: candidate for candidate in self.pool}


@dataclass(frozen=True, slots=True)
class _RankFacts:
    candidate: FusedTurnCandidate
    cross_encoder_logit: float
    cross_encoder_sigmoid: float
    cross_encoder_rank: int
    colbert_score: float | None
    colbert_rank: int | None
    weighted_rrf: float | None
    e1c_rank: int | None


def _validate_source(
    source: TurnFusionResult,
    *,
    protocol: E1SelectionProtocol,
) -> _SourceContext:
    if not isinstance(source, TurnFusionResult):
        raise E1SelectionError("source must be a frozen E1-A TurnFusionResult")
    if not isinstance(protocol, E1SelectionProtocol):
        raise E1SelectionError("protocol must be a registered E1SelectionProtocol")
    if not isinstance(source.question_turns, tuple):
        raise E1SelectionError("source E1-A question_turns must be an immutable tuple")
    if not isinstance(source.pre_cap_candidates, tuple) or not isinstance(source.candidates, tuple):
        raise E1SelectionError("source E1-A candidate pools must be immutable tuples")
    if len(source.candidates) > FUSED_CANDIDATE_CAP:
        raise E1SelectionError("source E1-A pool exceeds the immutable 128-candidate cap")
    if len(source.candidates) > protocol.source_pool_cap:
        raise E1SelectionError("source E1-A pool exceeds the registered E1 source cap")

    trace = source.content_free_trace()
    trace_sha256 = trace.pop("trace_sha256", None)
    checked_sha256(trace_sha256, label="source E1-A trace_sha256")
    if trace_sha256 != source.trace_sha256:
        raise E1SelectionError("source E1-A trace digest is internally inconsistent")
    if trace.get("artifact_type") != E1A_ARTIFACT_TYPE:
        raise E1SelectionError("source artifact is not an E1-A turn-fusion trace")
    if trace.get("protocol") != E1A_PROTOCOL.as_dict():
        raise E1SelectionError("source must use the exact frozen E1-A protocol")
    if protocol.source_e1a_protocol_version != E1A_PROTOCOL.protocol_version:
        raise E1SelectionError("selection protocol is not bound to the frozen E1-A version")

    question = trace.get("question")
    query = trace.get("query")
    fusion = trace.get("fusion")
    if (
        not isinstance(question, dict)
        or not isinstance(query, dict)
        or not isinstance(fusion, dict)
    ):
        raise E1SelectionError("source E1-A trace is missing required bindings")
    if question.get("question_id") != source.question_id:
        raise E1SelectionError("source question_id differs from its E1-A trace")
    if (
        query.get("question_id") != source.question_id
        or query.get("query_text_stored") is not False
    ):
        raise E1SelectionError("source E1-A query binding is inconsistent or not content-free")
    query_sha256 = checked_sha256(query.get("sha256"), label="source query_sha256")
    corpus_sha256 = checked_sha256(
        question.get("whole_corpus_projection_sha256"),
        label="source turn corpus projection_sha256",
    )
    if fusion.get("post_cap_count") != len(source.candidates):
        raise E1SelectionError("source E1-A post-cap count differs from its hydrated pool")

    question_turns_by_id: dict[LongMemEvalTurnId, TurnProjection] = {}
    for turn in source.question_turns:
        if not isinstance(turn, TurnProjection):
            raise E1SelectionError("source question_turns contains a non-turn value")
        if turn.turn_id.question_id != source.question_id:
            raise E1SelectionError("source question_turns contains a cross-question turn")
        if turn.turn_id in question_turns_by_id:
            raise E1SelectionError("source question_turns repeats a turn ID")
        question_turns_by_id[turn.turn_id] = turn

    seen: set[LongMemEvalTurnId] = set()
    for expected_rank, candidate in enumerate(source.candidates, start=1):
        if not isinstance(candidate, FusedTurnCandidate):
            raise E1SelectionError("source E1-A pool contains a non-candidate value")
        if candidate.turn_id in seen:
            raise E1SelectionError("source E1-A pool repeats a turn candidate")
        seen.add(candidate.turn_id)
        if candidate.turn_id.question_id != source.question_id:
            raise E1SelectionError("source E1-A pool contains a cross-question turn")
        if candidate.fused_rank != expected_rank:
            raise E1SelectionError("source E1-A fused ranks must be contiguous and one-based")
        if question_turns_by_id.get(candidate.turn_id) != candidate.turn:
            raise E1SelectionError("source E1-A candidate differs from its hydrated question turn")
        finite_score(candidate.raw_rrf, label="source E1-A raw RRF score")

    expected_bindings = [candidate.content_free_binding() for candidate in source.candidates]
    if fusion.get("post_cap_candidates") != expected_bindings:
        raise E1SelectionError("source E1-A trace candidate bindings differ from the hydrated pool")
    if fusion.get("post_cap_order_sha256") != sha256_json(expected_bindings):
        raise E1SelectionError("source E1-A post-cap order digest is inconsistent")

    pool_binding = [
        {
            "turn_id": _turn_id_payload(candidate.turn_id),
            "e1a_fused_rank": candidate.fused_rank,
            "e1a_raw_rrf": candidate.raw_rrf,
            "candidate_payload_utf8": candidate.turn.serialized_document_utf8.as_dict(),
        }
        for candidate in source.candidates
    ]
    return _SourceContext(
        question_id=source.question_id,
        query_sha256=query_sha256,
        turn_corpus_projection_sha256=corpus_sha256,
        e1a_trace_sha256=trace_sha256,
        pool_sha256=sha256_json(pool_binding),
        pool=source.candidates,
    )


def bind_e1a_pool(
    source: TurnFusionResult,
    *,
    protocol: E1SelectionProtocol = E1_PROTOCOL,
) -> E1APoolBinding:
    """Validate E1-A and return the exact content-free scorer bindings."""

    context = _validate_source(source, protocol=protocol)
    return E1APoolBinding(
        question_id=context.question_id,
        query_sha256=context.query_sha256,
        turn_corpus_projection_sha256=context.turn_corpus_projection_sha256,
        e1a_trace_sha256=context.e1a_trace_sha256,
        e1a_pool_sha256=context.pool_sha256,
        pool_count=len(context.pool),
    )


def _validate_observation(
    observation: PoolScoreObservation,
    *,
    channel: ScoreChannel,
    source: _SourceContext,
) -> dict[LongMemEvalTurnId, float]:
    if not isinstance(observation, PoolScoreObservation):
        raise E1SelectionError(f"{channel.value} evidence must be PoolScoreObservation")
    if observation.channel is not channel:
        raise E1SelectionError(f"expected {channel.value} evidence")
    if observation.question_id != source.question_id:
        raise E1SelectionError(f"{channel.value} evidence is bound to a different question")
    if observation.query_sha256 != source.query_sha256:
        raise E1SelectionError(f"{channel.value} evidence is bound to a different query")
    if observation.turn_corpus_projection_sha256 != source.turn_corpus_projection_sha256:
        raise E1SelectionError(f"{channel.value} evidence is bound to a different turn corpus")
    if observation.e1a_trace_sha256 != source.e1a_trace_sha256:
        raise E1SelectionError(f"{channel.value} evidence is bound to a different E1-A trace")
    if observation.e1a_pool_sha256 != source.pool_sha256:
        raise E1SelectionError(f"{channel.value} evidence is bound to a different E1-A pool")
    if observation.pool_count != len(source.pool):
        raise E1SelectionError(f"{channel.value} evidence pool_count differs from E1-A")

    expected_ids = {candidate.turn_id for candidate in source.pool}
    observed_ids = {item.turn_id for item in observation.scores}
    missing = expected_ids - observed_ids
    extra = observed_ids - expected_ids
    if missing or extra:
        raise E1SelectionError(
            f"{channel.value} evidence must cover the fixed E1-A pool exactly once "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    return {item.turn_id: item.raw_score for item in observation.scores}


def _rank_facts(
    source: _SourceContext,
    *,
    cross_encoder: PoolScoreObservation,
    colbert: PoolScoreObservation | None,
    protocol: E1SelectionProtocol,
) -> tuple[tuple[_RankFacts, ...], tuple[dict[str, Any], ...]]:
    ce_scores = _validate_observation(
        cross_encoder,
        channel=ScoreChannel.CROSS_ENCODER_LOGIT,
        source=source,
    )
    colbert_scores = (
        _validate_observation(
            colbert,
            channel=ScoreChannel.COLBERT_SCORE,
            source=source,
        )
        if colbert is not None
        else None
    )

    pool_by_id = source.pool_by_id
    sigmoid_by_id = {turn_id: stable_sigmoid(score) for turn_id, score in ce_scores.items()}
    ce_order = sorted(
        pool_by_id,
        key=lambda turn_id: (
            -sigmoid_by_id[turn_id],
            pool_by_id[turn_id].fused_rank,
            turn_id,
        ),
    )
    ce_rank_by_id = {turn_id: rank for rank, turn_id in enumerate(ce_order, start=1)}

    colbert_rank_by_id: dict[LongMemEvalTurnId, int] | None = None
    rrf_by_id: dict[LongMemEvalTurnId, float] | None = None
    e1c_rank_by_id: dict[LongMemEvalTurnId, int] | None = None
    if colbert_scores is not None:
        colbert_order = sorted(
            pool_by_id,
            key=lambda turn_id: (
                -colbert_scores[turn_id],
                pool_by_id[turn_id].fused_rank,
                turn_id,
            ),
        )
        colbert_rank_by_id = {turn_id: rank for rank, turn_id in enumerate(colbert_order, start=1)}
        rrf_by_id = {
            turn_id: (
                protocol.cross_encoder_weight / (protocol.rrf_k + ce_rank_by_id[turn_id])
                + protocol.colbert_weight / (protocol.rrf_k + colbert_rank_by_id[turn_id])
            )
            for turn_id in pool_by_id
        }
        e1c_order = sorted(
            pool_by_id,
            key=lambda turn_id: (
                -rrf_by_id[turn_id],
                pool_by_id[turn_id].fused_rank,
                turn_id,
            ),
        )
        e1c_rank_by_id = {turn_id: rank for rank, turn_id in enumerate(e1c_order, start=1)}

    facts = tuple(
        _RankFacts(
            candidate=candidate,
            cross_encoder_logit=ce_scores[candidate.turn_id],
            cross_encoder_sigmoid=sigmoid_by_id[candidate.turn_id],
            cross_encoder_rank=ce_rank_by_id[candidate.turn_id],
            colbert_score=(
                colbert_scores[candidate.turn_id] if colbert_scores is not None else None
            ),
            colbert_rank=(
                colbert_rank_by_id[candidate.turn_id] if colbert_rank_by_id is not None else None
            ),
            weighted_rrf=rrf_by_id[candidate.turn_id] if rrf_by_id is not None else None,
            e1c_rank=(e1c_rank_by_id[candidate.turn_id] if e1c_rank_by_id is not None else None),
        )
        for candidate in source.pool
    )
    observations = [cross_encoder.content_free_binding()]
    if colbert is not None:
        observations.append(colbert.content_free_binding())
    return facts, tuple(observations)


def _selected(fact: _RankFacts, *, selected_rank: int) -> E1SelectedCandidate:
    return E1SelectedCandidate(
        turn=fact.candidate.turn,
        selected_rank=selected_rank,
        e1a_fused_rank=fact.candidate.fused_rank,
        cross_encoder_logit=fact.cross_encoder_logit,
        cross_encoder_sigmoid=fact.cross_encoder_sigmoid,
        cross_encoder_rank=fact.cross_encoder_rank,
        colbert_score=fact.colbert_score,
        colbert_rank=fact.colbert_rank,
        weighted_rrf=fact.weighted_rrf,
        e1c_rank=fact.e1c_rank,
    )


def _base_trace(
    *,
    cell: E1Cell,
    source: _SourceContext,
    protocol: E1SelectionProtocol,
    observations: tuple[dict[str, Any], ...],
    selection: dict[str, Any],
) -> str:
    pool_ids = [_turn_id_payload(candidate.turn_id) for candidate in source.pool]
    pool_payloads = [
        {
            "turn_id": _turn_id_payload(candidate.turn_id),
            "candidate_payload_utf8": candidate.turn.serialized_document_utf8.as_dict(),
        }
        for candidate in source.pool
    ]
    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "cell": cell.value,
        "protocol": protocol.as_dict(),
        "source_e1a": {
            "question_id": source.question_id,
            "query_sha256": source.query_sha256,
            "turn_corpus_projection_sha256": source.turn_corpus_projection_sha256,
            "trace_sha256": source.e1a_trace_sha256,
            "fixed_pool_count": len(source.pool),
            "fixed_pool_cap": protocol.source_pool_cap,
            "fixed_pool_sha256": source.pool_sha256,
            "fixed_pool_order_ids_sha256": sha256_json(pool_ids),
            "fixed_pool_candidate_payloads_sha256": sha256_json(pool_payloads),
        },
        "observations": list(observations),
        "selection": selection,
        "accounting": {
            "fixed_pool_candidates": len(source.pool),
            "external_score_observations": sum(
                int(observation["pool_count"]) for observation in observations
            ),
            "local_model_calls": 0,
            "local_embedding_calls": 0,
            "local_database_calls": 0,
            "local_network_calls": 0,
        },
        "claims": {
            "external_scorer_model_revision_and_artifact_identity_verified": False,
            "external_observation_artifact_verified": False,
            "external_scores_recomputed": False,
            "gold_fields_present_in_selection_input": False,
            "prompt_packing_executed": False,
            "reader_or_judge_executed": False,
            "qa_improvement_proven": False,
            "SmartSearch_model_reproduction_proven": False,
        },
    }
    return canonical_json_bytes(payload).decode("utf-8")


def _result(
    *,
    cell: E1Cell,
    source: _SourceContext,
    candidates: tuple[E1SelectedCandidate, ...],
    protocol: E1SelectionProtocol,
    observations: tuple[dict[str, Any], ...],
    selection_details: dict[str, Any],
) -> E1SelectionResult:
    bindings = [candidate.content_free_binding() for candidate in candidates]
    payloads = [
        {
            "turn_id": _turn_id_payload(candidate.turn_id),
            "candidate_payload_utf8": candidate.turn.serialized_document_utf8.as_dict(),
        }
        for candidate in candidates
    ]
    selection = {
        **selection_details,
        "output_count": len(candidates),
        "output_candidates": bindings,
        "output_candidates_sha256": sha256_json(bindings),
        "output_candidate_payloads_sha256": sha256_json(payloads),
        "tie_break": ["prior-E1-A-fused-rank", "canonical-turn-id"],
    }
    trace = _base_trace(
        cell=cell,
        source=source,
        protocol=protocol,
        observations=observations,
        selection=selection,
    )
    return E1SelectionResult(
        cell=cell,
        question_id=source.question_id,
        source_pool_count=len(source.pool),
        candidates=candidates,
        _trace_canonical_json=trace,
    )


def select_e1b(
    source: TurnFusionResult,
    *,
    cross_encoder: PoolScoreObservation,
    protocol: E1SelectionProtocol = E1_PROTOCOL,
) -> E1SelectionResult:
    """Rank every fixed E1-A candidate by sigmoid(raw CrossEncoder logit)."""

    context = _validate_source(source, protocol=protocol)
    facts, observations = _rank_facts(
        context,
        cross_encoder=cross_encoder,
        colbert=None,
        protocol=protocol,
    )
    ordered = sorted(
        facts,
        key=lambda fact: (
            -fact.cross_encoder_sigmoid,
            fact.candidate.fused_rank,
            fact.candidate.turn_id,
        ),
    )
    candidates = tuple(
        _selected(fact, selected_rank=rank) for rank, fact in enumerate(ordered, start=1)
    )
    return _result(
        cell=E1Cell.CROSS_ENCODER,
        source=context,
        candidates=candidates,
        protocol=protocol,
        observations=observations,
        selection_details={
            "method": "rank-descending-sigmoid-of-raw-cross-encoder-logit",
            "sigmoid": "stable-piecewise-binary64",
            "input_count": len(context.pool),
            "every_fixed_pool_candidate_ranked": True,
        },
    )


def validate_e1b_result(
    result: E1SelectionResult,
    *,
    source: TurnFusionResult,
    cross_encoder: PoolScoreObservation,
    protocol: E1SelectionProtocol = E1_PROTOCOL,
) -> E1SelectionResult:
    """Recompute E1-B from its source evidence and require byte-exact equality.

    ``E1SelectionResult`` is a transport value, not an authentication token.
    Downstream experiment compilers must call this boundary with the original
    E1-A pool and CrossEncoder observations instead of trusting a
    self-consistent result/trace assembled by a caller.
    """

    if not isinstance(result, E1SelectionResult):
        raise E1SelectionError("E1-B result must be an E1SelectionResult")
    expected = select_e1b(
        source,
        cross_encoder=cross_encoder,
        protocol=protocol,
    )
    if result != expected:
        raise E1SelectionError(
            "E1-B result differs from deterministic replay of its source evidence"
        )
    return result


def _e1c_order(facts: tuple[_RankFacts, ...]) -> list[_RankFacts]:
    return sorted(
        facts,
        key=lambda fact: (
            -finite_score(fact.weighted_rrf, label="E1-C weighted RRF"),
            fact.candidate.fused_rank,
            fact.candidate.turn_id,
        ),
    )


def select_e1c(
    source: TurnFusionResult,
    *,
    cross_encoder: PoolScoreObservation,
    colbert: PoolScoreObservation,
    protocol: E1SelectionProtocol = E1_PROTOCOL,
) -> E1SelectionResult:
    """Fuse CrossEncoder and ColBERT ranks over exactly the E1-A pool."""

    context = _validate_source(source, protocol=protocol)
    facts, observations = _rank_facts(
        context,
        cross_encoder=cross_encoder,
        colbert=colbert,
        protocol=protocol,
    )
    ordered = _e1c_order(facts)
    candidates = tuple(
        _selected(fact, selected_rank=rank) for rank, fact in enumerate(ordered, start=1)
    )
    return _result(
        cell=E1Cell.CROSS_ENCODER_COLBERT,
        source=context,
        candidates=candidates,
        protocol=protocol,
        observations=observations,
        selection_details={
            "method": "weighted-reciprocal-rank-fusion",
            "input_count": len(context.pool),
            "rrf_k": protocol.rrf_k,
            "weights": {
                ScoreChannel.CROSS_ENCODER_LOGIT.value: protocol.cross_encoder_weight,
                ScoreChannel.COLBERT_SCORE.value: protocol.colbert_weight,
            },
            "formula": (
                f"{protocol.cross_encoder_weight}/({protocol.rrf_k}+cross_encoder_rank)+"
                f"{protocol.colbert_weight}/({protocol.rrf_k}+colbert_rank)"
            ),
            "every_fixed_pool_candidate_ranked": True,
        },
    )


def select_e1d(
    source: TurnFusionResult,
    *,
    cross_encoder: PoolScoreObservation,
    colbert: PoolScoreObservation,
    protocol: E1SelectionProtocol = E1_PROTOCOL,
) -> E1SelectionResult:
    """Take E1-C top 60, then retain candidates meeting the relative CE gate."""

    context = _validate_source(source, protocol=protocol)
    facts, observations = _rank_facts(
        context,
        cross_encoder=cross_encoder,
        colbert=colbert,
        protocol=protocol,
    )
    fused = _e1c_order(facts)
    head = fused[: protocol.adaptive_fused_head]
    # SmartSearch describes the adaptive threshold after top-K preselection.
    # The denominator is consequently scoped to this E1-C head, never to a
    # candidate discarded before the threshold operation.
    maximum = max((fact.cross_encoder_sigmoid for fact in head), default=None)
    threshold = protocol.adaptive_ce_relative_threshold * maximum if maximum is not None else None
    retained = [
        fact for fact in head if threshold is not None and fact.cross_encoder_sigmoid >= threshold
    ]
    candidates = tuple(
        _selected(fact, selected_rank=rank) for rank, fact in enumerate(retained, start=1)
    )
    selected_rank_by_id = {candidate.turn_id: candidate.selected_rank for candidate in candidates}
    head_trace = []
    for fact in head:
        binding = _selected(
            fact,
            selected_rank=fact.e1c_rank or 0,
        ).content_free_binding()
        binding["retained_by_e1d"] = fact.candidate.turn_id in selected_rank_by_id
        binding["e1d_selected_rank"] = selected_rank_by_id.get(fact.candidate.turn_id)
        head_trace.append(binding)
    return _result(
        cell=E1Cell.ADAPTIVE_THRESHOLD,
        source=context,
        candidates=candidates,
        protocol=protocol,
        observations=observations,
        selection_details={
            "method": "E1-C-head-then-relative-cross-encoder-threshold",
            "input_count": len(context.pool),
            "e1c_fused_head_limit": protocol.adaptive_fused_head,
            "pre_threshold_head_count": len(head),
            "relative_threshold": protocol.adaptive_ce_relative_threshold,
            "threshold_denominator_scope": "E1-C-preselection-head-only",
            "maximum_cross_encoder_sigmoid_over_e1c_head": maximum,
            "absolute_cross_encoder_sigmoid_threshold": threshold,
            "keep_comparison": ">=",
            "pre_threshold_head": head_trace,
            "pre_threshold_head_sha256": sha256_json(head_trace),
        },
    )


__all__ = [
    "E1APoolBinding",
    "E1SelectedCandidate",
    "E1SelectionResult",
    "bind_e1a_pool",
    "select_e1b",
    "select_e1c",
    "select_e1d",
    "stable_sigmoid",
    "validate_e1b_result",
]
