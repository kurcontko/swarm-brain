from __future__ import annotations

import hashlib
import json
import math

import pytest
from benchmarks.integrations.chain_of_memory import (
    BETA,
    MAX_CHAIN_LENGTH,
    MAX_CONTEXT_SCORE_EVALUATIONS,
    MAX_PARITY_RENDERED_TURNS,
    MAX_TOTAL_DECISIONS,
    ChainCandidate,
    ChainOfMemoryError,
    ContextCosine,
    E2Cell,
    ExternalSimilarityEvidence,
    K,
    L,
    PrecomputedContextSimilarities,
    chain_context_sha256,
    organize_e2,
)
from benchmarks.integrations.longmemeval_turns import compile_dataset_bytes


def _turn_pool(
    query_cosines: list[float] | None = None,
    *,
    content_prefix: str = "PRIVATE-TURN-CONTENT",
) -> tuple[ChainCandidate, ...]:
    turns = [
        {
            "role": "user" if position % 2 == 0 else "assistant",
            "content": f"{content_prefix}-{position:02d}",
            "has_answer": position == 17,
        }
        for position in range(K)
    ]
    records = [
        {
            "question_id": "question-private-id",
            "question_type": "single-session-user",
            "question": "PRIVATE-QUESTION-TEXT",
            "answer": "PRIVATE-ANSWER-TEXT",
            "question_date": "2025/01/04 (Sat) 11:00",
            "haystack_session_ids": ["session-provenance-id"],
            "haystack_dates": ["2025/01/03 (Fri) 09:07"],
            "haystack_sessions": [turns],
            "answer_session_ids": ["session-provenance-id"],
        }
    ]
    raw = (json.dumps(records, separators=(",", ":")) + "\n").encode()
    corpus = compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="synthetic-chain-of-memory.json",
    )
    scores = query_cosines or [1.0 - position * 0.02 for position in range(K)]
    return tuple(
        ChainCandidate(turn=turn, query_cosine=score)
        for turn, score in zip(corpus.turns, scores, strict=True)
    )


def _evidence() -> ExternalSimilarityEvidence:
    return ExternalSimilarityEvidence(
        producer="offline-test-fixture",
        model_id="synthetic-normalized-vectors",
        model_revision="fixture-v1",
        artifact_sha256=hashlib.sha256(b"external-similarity-evidence").hexdigest(),
    )


def _constant_context(value: float):
    def callback(chain_text: str, candidate_text: str) -> float:
        assert chain_text
        assert candidate_text
        return value

    return callback


def _chain_ids(result) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(candidate.turn_id.as_tuple() for candidate in chain.turns) for chain in result.chains
    )


def test_e2_a_preserves_fixed_retrieval_order_without_chain_evolution() -> None:
    pool = _turn_pool()
    result = organize_e2(
        pool,
        cell=E2Cell.RETRIEVAL_ORDER,
        similarity_evidence=_evidence(),
    )

    assert result.chains == ()
    assert result.decisions == ()
    assert result.context_similarity_calls == 0
    assert result.rendered_candidates() == tuple(candidate.turn for candidate in pool)
    assert [turn.turn_id for turn in result.rendered_candidates()] == [
        candidate.turn_id for candidate in pool
    ]
    assert result.render_evidence().startswith(pool[0].turn.serialized_text)


def test_query_only_cell_seeds_beta_with_anchor_query_score_and_never_uses_context() -> None:
    scores = [1.0, 0.8, 0.6, *([0.2] * (K - 3))]
    pool = _turn_pool(scores)
    result = organize_e2(
        pool,
        cell=E2Cell.QUERY_ONLY_APT,
        similarity_evidence=_evidence(),
    )

    assert [chain.anchor.turn_id for chain in result.chains] == [
        pool[0].turn_id,
        pool[1].turn_id,
        pool[2].turn_id,
    ]
    first = result.decisions[0]
    assert first.previous_appended_score == 1.0
    assert first.threshold == BETA * pool[0].query_cosine
    assert first.best_candidate_turn_id == pool[1].turn_id
    assert first.best_score == pool[1].query_cosine
    assert first.appended is True
    assert all(score.context_cosine is None for score in first.scorecard)
    assert result.context_similarity_calls == 0
    assert all(len(chain.turns) == 3 for chain in result.chains)
    assert [chain.marginal_scores for chain in result.chains] == [
        (1.0, 0.8, 0.6),
        (0.8, 1.0, 0.6),
        (0.6, 1.0, 0.8),
    ]

    with pytest.raises(ChainOfMemoryError, match="cannot consume context"):
        organize_e2(
            pool,
            cell=E2Cell.QUERY_ONLY_APT,
            similarity_evidence=_evidence(),
            context_similarities=_constant_context(1.0),
        )


def test_product_no_apt_hits_exact_hard_bounds_and_reuses_across_chains() -> None:
    pool = _turn_pool([0.8] * K)
    result = organize_e2(
        pool,
        cell=E2Cell.PRODUCT_NO_APT,
        similarity_evidence=_evidence(),
        context_similarities=_constant_context(1.0),
    )

    assert len(result.chains) == L
    assert all(len(chain.turns) == MAX_CHAIN_LENGTH for chain in result.chains)
    assert len(result.decisions) == MAX_TOTAL_DECISIONS
    assert result.context_similarity_calls == MAX_CONTEXT_SCORE_EVALUATIONS
    assert len(result.rendered_candidates()) == MAX_PARITY_RENDERED_TURNS
    assert all(result.chains.count(chain) == 1 for chain in result.chains)
    for candidate in pool:
        assert (
            sum(
                candidate.turn_id in {member.turn_id for member in chain.turns}
                for chain in result.chains
            )
            == L
        )
    assert result.chains[0].turns[1].turn_id == pool[1].turn_id
    assert result.chains[1].turns[1].turn_id == pool[0].turn_id
    assert all(decision.threshold is None for decision in result.decisions)


def test_stable_anchor_successor_and_chain_ties_use_retrieval_order() -> None:
    pool = _turn_pool([0.5] * K)
    result = organize_e2(
        pool,
        cell=E2Cell.PRODUCT_NO_APT,
        similarity_evidence=_evidence(),
        context_similarities=_constant_context(1.0),
    )

    assert [chain.anchor.turn_id for chain in result.chains] == [
        pool[0].turn_id,
        pool[1].turn_id,
        pool[2].turn_id,
    ]
    assert result.chains[0].turns[1].turn_id == pool[1].turn_id
    assert result.chains[1].turns[1].turn_id == pool[0].turn_id
    assert [score.retrieval_rank for score in result.decisions[0].scorecard] == list(
        range(2, K + 1)
    )


def test_primary_rrf_head_uses_first_l_anchors_then_orders_blocks_by_anchor_cosine() -> None:
    scores = [0.2, 0.9, 0.1, *([0.3] * 7), 1.0, *([0.3] * (K - 11))]
    pool = _turn_pool(scores)
    result = organize_e2(
        pool,
        cell=E2Cell.PRODUCT_APT,
        similarity_evidence=_evidence(),
        context_similarities=_constant_context(0.0),
    )

    # Rank 11 has the highest query cosine, but cannot replace a frozen RRF
    # anchor. The three selected anchors are then ordered for rendering by
    # their own query cosine.
    assert pool[10].query_cosine == 1.0
    assert [chain.anchor.turn_id for chain in result.chains] == [
        pool[1].turn_id,
        pool[0].turn_id,
        pool[2].turn_id,
    ]
    assert pool[10].turn_id not in {chain.anchor.turn_id for chain in result.chains}
    assert [decision.anchor_turn_id for decision in result.decisions] == [
        pool[0].turn_id,
        pool[1].turn_id,
        pool[2].turn_id,
    ]
    frozen = result.content_free_trace()["frozen_parameters"]
    assert frozen["anchor_membership"] == "first-L-candidates-in-frozen-input-order"
    assert frozen["chain_block_order"] == ("anchor-query-cosine-descending-then-retrieval-rank")
    assert frozen["paper_text_diagnostic"].startswith("global-query-cosine-top-20")


def test_product_apt_accepts_threshold_equality_then_stops() -> None:
    pool = _turn_pool([1.0] * K)

    def equality_then_zero(chain_text: str, candidate_text: str) -> float:
        del candidate_text
        return 0.5 if chain_text.count('"timestamp"') == 1 else 0.0

    result = organize_e2(
        pool,
        cell=E2Cell.PRODUCT_APT,
        similarity_evidence=_evidence(),
        context_similarities=equality_then_zero,
    )

    first, second = result.decisions[:2]
    assert first.previous_appended_score == 1.0
    assert first.threshold == 0.5
    assert first.best_score == 0.5
    assert first.appended is True
    assert second.previous_appended_score == 0.5
    assert second.threshold == 0.25
    assert second.best_score == 0.0
    assert second.appended is False
    assert all(len(chain.turns) == 2 for chain in result.chains)


def test_negative_cosines_are_preserved_with_raw_product_and_stop_semantics() -> None:
    scores = [-0.1 - position * 0.01 for position in range(K)]
    pool = _turn_pool(scores)
    result = organize_e2(
        pool,
        cell=E2Cell.PRODUCT_APT,
        similarity_evidence=_evidence(),
        context_similarities=_constant_context(1.0),
    )

    first = result.decisions[0]
    assert first.previous_appended_score == -0.1
    assert first.threshold == -0.05
    assert first.best_score == -0.11
    assert first.appended is False
    assert first.scorecard[0].query_cosine == -0.11
    assert first.scorecard[0].context_cosine == 1.0
    assert first.scorecard[0].gate_score == -0.11
    assert all(len(chain.turns) == 1 for chain in result.chains)

    positive_product = organize_e2(
        pool,
        cell=E2Cell.PRODUCT_NO_APT,
        similarity_evidence=_evidence(),
        context_similarities=_constant_context(-0.5),
    )
    assert positive_product.decisions[0].scorecard[0].gate_score == pytest.approx(0.055)


def test_e2_e_changes_only_rendering_and_preserves_product_apt_evolution() -> None:
    pool = _turn_pool([0.8] * K)
    parity = organize_e2(
        pool,
        cell=E2Cell.PRODUCT_APT,
        similarity_evidence=_evidence(),
        context_similarities=_constant_context(1.0),
    )
    dedup = organize_e2(
        pool,
        cell=E2Cell.PRODUCT_APT_DEDUP,
        similarity_evidence=_evidence(),
        context_similarities=_constant_context(1.0),
    )

    assert _chain_ids(dedup) == _chain_ids(parity)
    assert [chain.marginal_scores for chain in dedup.chains] == [
        chain.marginal_scores for chain in parity.chains
    ]
    assert len(parity.rendered_candidates()) == K * L
    assert len(dedup.rendered_candidates()) == K
    assert len({turn.turn_id for turn in dedup.rendered_candidates()}) == K
    assert len(dedup.rendered_chains()[0]) == K
    assert dedup.rendered_chains()[1:] == ((), ())
    assert dedup.render_evidence().count("=== Evidence Chain") == 1


def test_precomputed_context_table_binds_exact_chain_and_candidate_documents() -> None:
    pool = _turn_pool([0.8] * K)
    entries = [
        ContextCosine.from_turns((anchor.turn,), candidate.turn, 0.0)
        for anchor in pool[:L]
        for candidate in pool
        if candidate.turn_id != anchor.turn_id
    ]
    table = PrecomputedContextSimilarities(entries)
    result = organize_e2(
        pool,
        cell=E2Cell.PRODUCT_APT,
        similarity_evidence=_evidence(),
        context_similarities=table,
    )

    assert result.context_similarity_mode == "precomputed-exact-prefix-table"
    assert result.context_similarity_table_sha256 == table.table_sha256
    assert result.context_similarity_calls == L * (K - 1)
    assert all(len(chain.turns) == 1 for chain in result.chains)
    assert (
        result.content_free_trace()["context_similarity"]["precomputed_table_sha256"]
        == table.table_sha256
    )

    with pytest.raises(ChainOfMemoryError, match="missing"):
        organize_e2(
            pool,
            cell=E2Cell.PRODUCT_APT,
            similarity_evidence=_evidence(),
            context_similarities=PrecomputedContextSimilarities(()),
        )

    with pytest.raises(ChainOfMemoryError, match="repeats"):
        PrecomputedContextSimilarities((entries[0], entries[0]))
    with pytest.raises(ChainOfMemoryError, match="hard score-evidence bound"):
        PrecomputedContextSimilarities([entries[0]] * (MAX_CONTEXT_SCORE_EVALUATIONS + 1))


def test_precomputed_context_rejects_tampered_context_and_candidate_digests() -> None:
    pool = _turn_pool([0.8] * K)
    valid = ContextCosine.from_turns((pool[0].turn,), pool[1].turn, 0.0)
    tampered_context = ContextCosine(
        chain_turn_ids=valid.chain_turn_ids,
        candidate_turn_id=valid.candidate_turn_id,
        chain_context_sha256="0" * 64,
        candidate_document_sha256=valid.candidate_document_sha256,
        cosine=0.0,
    )
    with pytest.raises(ChainOfMemoryError, match="context digest differs"):
        PrecomputedContextSimilarities((tampered_context,)).lookup((pool[0].turn,), pool[1].turn)

    tampered_candidate = ContextCosine(
        chain_turn_ids=valid.chain_turn_ids,
        candidate_turn_id=valid.candidate_turn_id,
        chain_context_sha256=chain_context_sha256((pool[0].turn,)),
        candidate_document_sha256="f" * 64,
        cosine=0.0,
    )
    with pytest.raises(ChainOfMemoryError, match="candidate digest differs"):
        PrecomputedContextSimilarities((tampered_candidate,)).lookup((pool[0].turn,), pool[1].turn)


def test_callback_receives_exact_concatenated_chain_and_candidate_serialization() -> None:
    pool = _turn_pool([0.8] * K)
    observed: list[tuple[str, str]] = []

    def capture(chain_text: str, candidate_text: str) -> float:
        observed.append((chain_text, candidate_text))
        return 0.0

    organize_e2(
        pool,
        cell=E2Cell.PRODUCT_APT,
        similarity_evidence=_evidence(),
        context_similarities=capture,
    )

    assert observed[0] == (pool[0].turn.serialized_text, pool[1].turn.serialized_text)
    assert all("PRIVATE-TURN-CONTENT" in candidate_text for _, candidate_text in observed)


def test_content_free_trace_preserves_ids_provenance_scores_and_digests_only() -> None:
    pool = _turn_pool([0.8] * K)
    result = organize_e2(
        pool,
        cell=E2Cell.PRODUCT_APT,
        similarity_evidence=_evidence(),
        context_similarities=_constant_context(0.0),
    )
    artifact = result.content_free_artifact()
    encoded = json.dumps(artifact, ensure_ascii=False, sort_keys=True)

    assert "PRIVATE-TURN-CONTENT" not in encoded
    assert "PRIVATE-QUESTION-TEXT" not in encoded
    assert "PRIVATE-ANSWER-TEXT" not in encoded
    assert "session-provenance-id" in encoded
    assert artifact["candidate_pool"][0]["turn"]["turn_id"] == [
        "question-private-id",
        0,
        0,
    ]
    assert artifact["candidate_pool"][0]["turn"]["source_turn"]["sha256"]
    assert artifact["decisions"][0]["scorecard_sha256"]
    assert artifact["decisions_sha256"]
    assert artifact["rendered_chains_sha256"]
    assert artifact["trace_sha256"] == result.trace_sha256
    assert artifact["claims"] == {
        "embedding_values_are_external_evidence": True,
        "organizer_executes_embeddings": False,
        "paper_parity": False,
        "qa_improvement": False,
    }
    assert "PRIVATE-TURN-CONTENT" in result.render_evidence()


@pytest.mark.parametrize("size", [K - 1, K + 1])
def test_candidate_pool_size_fails_closed(size: int) -> None:
    pool = _turn_pool()
    expanded = pool + (pool[-1],)
    candidate_pool = pool[:size] if size < K else expanded
    with pytest.raises(ChainOfMemoryError, match="exactly K=20"):
        organize_e2(
            candidate_pool,
            cell=E2Cell.RETRIEVAL_ORDER,
            similarity_evidence=_evidence(),
        )


def test_candidate_identity_order_and_context_requirements_fail_closed() -> None:
    pool = _turn_pool()
    duplicate = (*pool[:-1], pool[0])
    with pytest.raises(ChainOfMemoryError, match="cannot repeat"):
        organize_e2(
            duplicate,
            cell=E2Cell.RETRIEVAL_ORDER,
            similarity_evidence=_evidence(),
        )
    with pytest.raises(ChainOfMemoryError, match="frozen in a tuple"):
        organize_e2(  # type: ignore[arg-type]
            list(pool),
            cell=E2Cell.RETRIEVAL_ORDER,
            similarity_evidence=_evidence(),
        )
    with pytest.raises(ChainOfMemoryError, match="require external context"):
        organize_e2(
            pool,
            cell=E2Cell.PRODUCT_APT,
            similarity_evidence=_evidence(),
        )

    other_record = _turn_pool(content_prefix="OTHER-SOURCE-RECORD")
    mixed_record = (*pool[: K // 2], *other_record[K // 2 :])
    with pytest.raises(ChainOfMemoryError, match="source-record binding"):
        organize_e2(
            mixed_record,
            cell=E2Cell.RETRIEVAL_ORDER,
            similarity_evidence=_evidence(),
        )


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf, -1.0001, 1.0001, True])
def test_query_cosine_must_be_finite_normalized_and_is_not_clamped(score: object) -> None:
    pool = _turn_pool()
    with pytest.raises(ChainOfMemoryError, match="finite normalized cosine"):
        ChainCandidate(turn=pool[0].turn, query_cosine=score)  # type: ignore[arg-type]

    negative = ChainCandidate(turn=pool[0].turn, query_cosine=-0.75)
    assert negative.query_cosine == -0.75


@pytest.mark.parametrize("score", [math.nan, math.inf, -1.01, 1.01, True])
def test_callback_cosine_must_be_finite_normalized(score: object) -> None:
    pool = _turn_pool([0.8] * K)

    def invalid(chain_text: str, candidate_text: str):
        del chain_text, candidate_text
        return score

    with pytest.raises(ChainOfMemoryError, match="finite normalized cosine"):
        organize_e2(
            pool,
            cell=E2Cell.PRODUCT_APT,
            similarity_evidence=_evidence(),
            context_similarities=invalid,  # type: ignore[arg-type]
        )
