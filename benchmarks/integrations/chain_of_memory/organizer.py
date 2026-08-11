"""Pure/offline Chain-of-Memory organizer for the preregistered E2 cells.

This module never embeds text, retrieves candidates, calls a model, answers a
question, or judges an answer.  It deterministically organizes one externally
scored, fixed top-20 turn pool and emits a content-free audit trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.integrations.longmemeval_turns import LongMemEvalTurnId, TurnProjection

from .contracts import (
    ARTIFACT_TYPE,
    BETA,
    CELL_PROTOCOLS,
    CHAIN_CONTEXT_SERIALIZER_VERSION,
    MAX_CHAIN_DECISIONS,
    MAX_CHAIN_LENGTH,
    MAX_CONTEXT_SCORE_EVALUATIONS,
    MAX_PARITY_RENDERED_TURNS,
    MAX_TOTAL_DECISIONS,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    CandidateScoreTrace,
    ChainCandidate,
    ChainDecision,
    ChainOfMemoryError,
    ContextSimilaritySource,
    E2Cell,
    EvidenceChain,
    ExternalSimilarityEvidence,
    K,
    L,
    PrecomputedContextSimilarities,
    chain_context_text,
    finite_score,
    normalized_cosine,
    sha256_json,
    turn_id_payload,
)


@dataclass(frozen=True, slots=True)
class OrganizationResult:
    """One deterministic E2 result; raw turn text is absent from its trace payload."""

    cell: E2Cell
    candidate_pool: tuple[ChainCandidate, ...]
    chains: tuple[EvidenceChain, ...]
    decisions: tuple[ChainDecision, ...]
    similarity_evidence: ExternalSimilarityEvidence
    context_similarity_mode: str
    context_similarity_table_sha256: str | None
    context_similarity_calls: int

    def __post_init__(self) -> None:
        if len(self.candidate_pool) != K:
            raise ChainOfMemoryError(f"organization result must preserve exactly K={K} candidates")
        if len({candidate.turn_id for candidate in self.candidate_pool}) != K:
            raise ChainOfMemoryError("organization result candidate IDs must be unique")
        if len(self.decisions) > MAX_TOTAL_DECISIONS:
            raise ChainOfMemoryError("organization result exceeds the hard decision bound")
        if not 0 <= self.context_similarity_calls <= MAX_CONTEXT_SCORE_EVALUATIONS:
            raise ChainOfMemoryError("organization result exceeds the context-score call bound")
        protocol = CELL_PROTOCOLS[self.cell]
        if protocol.evolves_chains:
            if len(self.chains) != L:
                raise ChainOfMemoryError(f"chain cells must contain exactly L={L} chains")
        elif self.chains or self.decisions or self.context_similarity_calls:
            raise ChainOfMemoryError("the retrieval-order control cannot contain chain decisions")
        pool_ids = {candidate.turn_id for candidate in self.candidate_pool}
        for chain in self.chains:
            if any(candidate.turn_id not in pool_ids for candidate in chain.turns):
                raise ChainOfMemoryError("evidence chain contains a turn outside the fixed pool")
        if len(self.rendered_candidates()) > MAX_PARITY_RENDERED_TURNS:
            raise ChainOfMemoryError("rendered evidence exceeds the hard parity-size bound")

    @property
    def protocol(self):
        return CELL_PROTOCOLS[self.cell]

    def rendered_chains(self) -> tuple[tuple[TurnProjection, ...], ...]:
        """Return the exact block membership after the cell's frozen dedup policy."""

        if not self.protocol.evolves_chains:
            return (tuple(candidate.turn for candidate in self.candidate_pool),)
        if not self.protocol.deduplicate_cross_chain_rendering:
            return tuple(
                tuple(candidate.turn for candidate in chain.turns) for chain in self.chains
            )

        seen: set[LongMemEvalTurnId] = set()
        rendered: list[tuple[TurnProjection, ...]] = []
        for chain in self.chains:
            block: list[TurnProjection] = []
            for candidate in chain.turns:
                if candidate.turn_id in seen:
                    continue
                seen.add(candidate.turn_id)
                block.append(candidate.turn)
            rendered.append(tuple(block))
        return tuple(rendered)

    def rendered_candidates(self) -> tuple[TurnProjection, ...]:
        return tuple(turn for chain in self.rendered_chains() for turn in chain)

    def render_evidence(self) -> str:
        """Render exact serialized turns; decision traces remain separately content-free."""

        blocks = self.rendered_chains()
        if not self.protocol.evolves_chains:
            return "\n\n".join(turn.serialized_text for turn in blocks[0])
        rendered: list[str] = []
        for chain_number, turns in enumerate(blocks, start=1):
            if not turns:
                continue
            body = "\n\n".join(turn.serialized_text for turn in turns)
            rendered.append(f"=== Evidence Chain {chain_number} ===\n{body}")
        return "\n\n".join(rendered)

    def _candidate_pool_binding(self) -> list[dict[str, Any]]:
        return [
            {
                "retrieval_rank": rank,
                "query_cosine": candidate.query_cosine,
                "turn": candidate.turn.content_free_binding(),
            }
            for rank, candidate in enumerate(self.candidate_pool, start=1)
        ]

    def content_free_trace(self) -> dict[str, Any]:
        """Return provenance, scores, choices, and digests without question/turn text."""

        candidate_pool = self._candidate_pool_binding()
        decisions = [decision.content_free_binding() for decision in self.decisions]
        chains = [chain.content_free_binding() for chain in self.chains]
        rendered_chains = [
            [
                {
                    "turn_id": turn_id_payload(turn.turn_id),
                    "serialized_document_utf8": turn.serialized_document_utf8.as_dict(),
                    "source_turn": turn.source_turn.as_dict(),
                }
                for turn in chain
            ]
            for chain in self.rendered_chains()
        ]
        used_context_scores = [
            score.content_free_binding()
            for decision in self.decisions
            for score in decision.scorecard
            if score.context_cosine is not None
        ]
        return {
            "artifact_type": ARTIFACT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "cell": self.cell.value,
            "cell_protocol": self.protocol.as_dict(),
            "frozen_parameters": {
                "K": K,
                "L": L,
                "beta": BETA,
                "primary_candidate_source": "fixed-E1-A-RRF-top-20",
                "anchor_membership": "first-L-candidates-in-frozen-input-order",
                "chain_block_order": "anchor-query-cosine-descending-then-retrieval-rank",
                "paper_text_diagnostic": (
                    "global-query-cosine-top-20-input; first-L-therefore-query-cosine-top-L"
                ),
                "context_mode": "concatenated-chain-serialized-turns",
                "chain_context_serializer": CHAIN_CONTEXT_SERIALIZER_VERSION,
                "product_gate": "query_cosine*chain_context_cosine",
                "apt_stop_rule": "best_score < beta * previous_appended_score",
                "candidate_removal": "within-current-chain-only",
                "cross_chain_reuse": True,
                "tie_break": "retrieval_rank-then-canonical-turn-id",
                "max_chain_length": MAX_CHAIN_LENGTH,
                "max_total_decisions": MAX_TOTAL_DECISIONS,
                "max_context_score_evaluations": MAX_CONTEXT_SCORE_EVALUATIONS,
                "max_parity_rendered_turns": MAX_PARITY_RENDERED_TURNS,
            },
            "claims": {
                "embedding_values_are_external_evidence": True,
                "organizer_executes_embeddings": False,
                "paper_parity": False,
                "qa_improvement": False,
            },
            "limitations": [
                "The organizer does not verify the external embedding/model execution.",
                "A deterministic callback is caller-attested; its outputs are bound below.",
                "Organization output alone is not end-to-end QA evidence.",
                "This paper-inspired transfer cell is not an exact CoM reproduction.",
            ],
            "similarity_evidence": self.similarity_evidence.content_free_binding(),
            "candidate_pool": candidate_pool,
            "candidate_pool_sha256": sha256_json(candidate_pool),
            "context_similarity": {
                "mode": self.context_similarity_mode,
                "precomputed_table_sha256": self.context_similarity_table_sha256,
                "calls": self.context_similarity_calls,
                "used_scores_sha256": sha256_json(used_context_scores),
            },
            "decisions": decisions,
            "decisions_sha256": sha256_json(decisions),
            "chains": chains,
            "chains_sha256": sha256_json(chains),
            "rendered_chains": rendered_chains,
            "rendered_chains_sha256": sha256_json(rendered_chains),
            "rendered_turn_count": sum(len(chain) for chain in rendered_chains),
        }

    @property
    def trace_sha256(self) -> str:
        return sha256_json(self.content_free_trace())

    def content_free_artifact(self) -> dict[str, Any]:
        trace = self.content_free_trace()
        return {**trace, "trace_sha256": sha256_json(trace)}


def _validate_pool(candidates: tuple[ChainCandidate, ...]) -> None:
    if len(candidates) != K:
        raise ChainOfMemoryError(f"E2 requires exactly K={K} ordered candidates")
    if any(not isinstance(candidate, ChainCandidate) for candidate in candidates):
        raise ChainOfMemoryError("every E2 candidate must be a ChainCandidate")
    ids = [candidate.turn_id for candidate in candidates]
    if len(set(ids)) != K:
        raise ChainOfMemoryError("the fixed E2 candidate pool cannot repeat a turn ID")
    if len({turn_id.question_id for turn_id in ids}) != 1:
        raise ChainOfMemoryError("all fixed E2 candidates must belong to one question")
    if len({candidate.turn.source_record for candidate in candidates}) != 1:
        raise ChainOfMemoryError("all fixed E2 candidates must share one source-record binding")


def _rank_by_id(candidates: tuple[ChainCandidate, ...]) -> dict[LongMemEvalTurnId, int]:
    return {candidate.turn_id: rank for rank, candidate in enumerate(candidates, start=1)}


def _anchor_key(
    candidate: ChainCandidate,
    retrieval_rank: dict[LongMemEvalTurnId, int],
) -> tuple[float, int, str]:
    return (
        -candidate.query_cosine,
        retrieval_rank[candidate.turn_id],
        candidate.turn_id.canonical_id,
    )


def _best_score_key(score: CandidateScoreTrace) -> tuple[float, int, str]:
    return (-score.gate_score, score.retrieval_rank, score.candidate_turn_id.canonical_id)


def _context_score(
    source: ContextSimilaritySource,
    chain: tuple[ChainCandidate, ...],
    candidate: ChainCandidate,
) -> float:
    turns = tuple(item.turn for item in chain)
    if isinstance(source, PrecomputedContextSimilarities):
        return source.lookup(turns, candidate.turn)
    if not callable(source):
        raise ChainOfMemoryError("product cells require a context-similarity table or callback")
    value = source(chain_context_text(turns), candidate.turn.serialized_text)
    return normalized_cosine(value, label="context similarity callback result")


def organize_e2(
    candidates: tuple[ChainCandidate, ...],
    *,
    cell: E2Cell,
    similarity_evidence: ExternalSimilarityEvidence,
    context_similarities: ContextSimilaritySource | None = None,
) -> OrganizationResult:
    """Organize one fixed top-20 pool according to exactly one frozen E2 cell.

    Candidate tuple order is the prior retrieval order and is never changed in
    E2-A.  It is also the first stable tie-break for anchor and successor
    selection in E2-B..E.  Product cells receive exact concatenated-chain
    similarities through an immutable table or a deterministic callback.
    """

    if not isinstance(candidates, tuple):
        raise ChainOfMemoryError("E2 candidate order must be frozen in a tuple")
    _validate_pool(candidates)
    if not isinstance(cell, E2Cell):
        raise ChainOfMemoryError("cell must be an E2Cell")
    if not isinstance(similarity_evidence, ExternalSimilarityEvidence):
        raise ChainOfMemoryError("external similarity evidence binding is required")
    protocol = CELL_PROTOCOLS[cell]
    needs_context = protocol.score_mode == "product"
    if needs_context and context_similarities is None:
        raise ChainOfMemoryError("product cells require external context similarities")
    if not needs_context and context_similarities is not None:
        raise ChainOfMemoryError("retrieval/query-only cells cannot consume context similarities")

    if not protocol.evolves_chains:
        return OrganizationResult(
            cell=cell,
            candidate_pool=candidates,
            chains=(),
            decisions=(),
            similarity_evidence=similarity_evidence,
            context_similarity_mode="not-required",
            context_similarity_table_sha256=None,
            context_similarity_calls=0,
        )

    retrieval_rank = _rank_by_id(candidates)
    # The primary transfer input is already the frozen E1-A RRF head.  Anchor
    # membership must therefore preserve its first L candidates.  This differs
    # from the paper-text diagnostic, whose entire input is query-cosine sorted
    # and consequently has the same first-L/top-L membership.  Chain blocks
    # are still ordered by anchor-query cosine after evolution.
    anchors = candidates[:L]
    chains: list[EvidenceChain] = []
    decisions: list[ChainDecision] = []
    context_calls = 0

    for anchor in anchors:
        chain: list[ChainCandidate] = [anchor]
        marginal_scores: list[float] = [anchor.query_cosine]
        remaining = [candidate for candidate in candidates if candidate.turn_id != anchor.turn_id]
        previous_score = anchor.query_cosine

        for iteration in range(1, MAX_CHAIN_DECISIONS + 1):
            if not remaining:
                break
            scorecard: list[CandidateScoreTrace] = []
            chain_tuple = tuple(chain)
            for candidate in remaining:
                context_cosine: float | None = None
                if protocol.score_mode == "product":
                    if context_calls >= MAX_CONTEXT_SCORE_EVALUATIONS:
                        raise ChainOfMemoryError("context-score evaluation bound would be exceeded")
                    assert context_similarities is not None
                    context_cosine = _context_score(
                        context_similarities,
                        chain_tuple,
                        candidate,
                    )
                    context_calls += 1
                    gate_score = finite_score(
                        candidate.query_cosine * context_cosine,
                        label="product gate score",
                    )
                else:
                    gate_score = candidate.query_cosine
                scorecard.append(
                    CandidateScoreTrace(
                        candidate_turn_id=candidate.turn_id,
                        retrieval_rank=retrieval_rank[candidate.turn_id],
                        query_cosine=candidate.query_cosine,
                        context_cosine=context_cosine,
                        gate_score=gate_score,
                    )
                )

            best = min(scorecard, key=_best_score_key)
            threshold = BETA * previous_score if protocol.adaptive_path_truncation else None
            append = threshold is None or best.gate_score >= threshold
            decisions.append(
                ChainDecision(
                    anchor_turn_id=anchor.turn_id,
                    iteration=iteration,
                    previous_appended_score=previous_score,
                    threshold=threshold,
                    scorecard=tuple(scorecard),
                    best_candidate_turn_id=best.candidate_turn_id,
                    best_score=best.gate_score,
                    appended=append,
                    reason=("selected" if append else "below-beta-times-previous-score"),
                )
            )
            if len(decisions) > MAX_TOTAL_DECISIONS:
                raise ChainOfMemoryError("decision bound would be exceeded")
            if not append:
                break

            selected = next(
                candidate for candidate in remaining if candidate.turn_id == best.candidate_turn_id
            )
            chain.append(selected)
            marginal_scores.append(best.gate_score)
            remaining.remove(selected)
            previous_score = best.gate_score

        chains.append(
            EvidenceChain(
                anchor=anchor,
                turns=tuple(chain),
                marginal_scores=tuple(marginal_scores),
            )
        )

    chains.sort(key=lambda chain: _anchor_key(chain.anchor, retrieval_rank))
    context_mode = (
        "precomputed-exact-prefix-table"
        if isinstance(context_similarities, PrecomputedContextSimilarities)
        else "deterministic-callback"
        if needs_context
        else "not-required"
    )
    table_sha256 = (
        context_similarities.table_sha256
        if isinstance(context_similarities, PrecomputedContextSimilarities)
        else None
    )
    return OrganizationResult(
        cell=cell,
        candidate_pool=candidates,
        chains=tuple(chains),
        decisions=tuple(decisions),
        similarity_evidence=similarity_evidence,
        context_similarity_mode=context_mode,
        context_similarity_table_sha256=table_sha256,
        context_similarity_calls=context_calls,
    )


__all__ = ["OrganizationResult", "organize_e2"]
