"""Construction, validation, and deterministic application of learned scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from uuid import uuid4

from swarmbrain.domain.memory import Memory
from swarmbrain.domain.reranking import (
    LearnedRerankCandidate,
    LearnedRerankerIdentity,
    LearnedRerankPolicy,
    LearnedRerankReceipt,
    LearnedRerankRequest,
    LearnedRerankResult,
    LearnedRerankScore,
    LearnedRerankUsage,
    canonical_rerank_json,
    learned_rerank_candidate_pool_payload,
    learned_rerank_request_payload,
    learned_rerank_response_payload,
    rerank_sha256_json,
    rerank_sha256_text,
)
from swarmbrain.domain.retrieval import FusedCandidate

from .projection import search_text

SWARMBRAIN_MEMORY_RERANK_SERIALIZER_REVISION = "swarmbrain-memory-rerank-v1"


class LearnedRerankValidationError(ValueError):
    """A provider result or request violated the score-only reranker contract."""


def utf8_prefix(value: str, maximum_bytes: int) -> str:
    """Return the longest valid UTF-8 prefix within a hard byte budget."""

    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value
    return encoded[:maximum_bytes].decode("utf-8", errors="ignore")


def make_learned_rerank_candidate(
    candidate_id: str,
    document: str,
    temporal_context: str,
) -> LearnedRerankCandidate:
    """Build one self-verifying candidate after caller-owned size checks."""

    return LearnedRerankCandidate(
        candidate_id=candidate_id,
        document=document,
        document_sha256=rerank_sha256_text(document),
        temporal_context=temporal_context,
        temporal_sha256=rerank_sha256_text(temporal_context),
    )


def build_learned_rerank_request(
    policy: LearnedRerankPolicy,
    *,
    serializer_revision: str,
    query: str,
    candidates: Sequence[tuple[str, str, str] | LearnedRerankCandidate],
    request_id: str | None = None,
) -> LearnedRerankRequest:
    """Build and byte-bound one deterministic request.

    ``candidates`` are ``(candidate_id, document, temporal_context)`` triples
    in the pre-learned baseline order.  This helper rejects oversize input
    instead of silently changing benchmark text; the Swarm Memory serializer
    applies an explicit UTF-8 prefix before calling it.
    """

    if len(candidates) > policy.window:
        raise LearnedRerankValidationError("candidate count exceeds learned rerank window")
    if not candidates:
        raise LearnedRerankValidationError("learned rerank request requires candidates")
    if len(query) > policy.max_query_characters:
        raise LearnedRerankValidationError("query exceeds learned rerank character bound")
    if len(query.encode("utf-8")) > policy.max_query_bytes:
        raise LearnedRerankValidationError("query exceeds learned rerank byte bound")

    built: list[LearnedRerankCandidate] = []
    for value in candidates:
        candidate = (
            value
            if isinstance(value, LearnedRerankCandidate)
            else make_learned_rerank_candidate(*value)
        )
        if len(candidate.document) > policy.max_document_characters:
            raise LearnedRerankValidationError(
                f"candidate {candidate.candidate_id!r} exceeds document character bound"
            )
        if len(candidate.document.encode("utf-8")) > policy.max_document_bytes:
            raise LearnedRerankValidationError(
                f"candidate {candidate.candidate_id!r} exceeds document byte bound"
            )
        if len(candidate.temporal_context) > policy.max_temporal_characters:
            raise LearnedRerankValidationError(
                f"candidate {candidate.candidate_id!r} exceeds temporal character bound"
            )
        if len(candidate.temporal_context.encode("utf-8")) > policy.max_temporal_bytes:
            raise LearnedRerankValidationError(
                f"candidate {candidate.candidate_id!r} exceeds temporal byte bound"
            )
        built.append(candidate)

    candidate_tuple = tuple(built)
    query_sha256 = rerank_sha256_text(query)
    candidate_pool_sha256 = rerank_sha256_json(
        learned_rerank_candidate_pool_payload(candidate_tuple)
    )
    resolved_request_id = request_id or str(uuid4())
    request_sha256 = rerank_sha256_json(
        learned_rerank_request_payload(
            request_id=resolved_request_id,
            identity=policy.identity,
            serializer_revision=serializer_revision,
            query_sha256=query_sha256,
            candidate_pool_sha256=candidate_pool_sha256,
        )
    )
    request = LearnedRerankRequest(
        request_id=resolved_request_id,
        identity=policy.identity,
        serializer_revision=serializer_revision,
        query=query,
        query_sha256=query_sha256,
        candidates=candidate_tuple,
        candidate_pool_sha256=candidate_pool_sha256,
        request_sha256=request_sha256,
    )
    request_bytes = len(canonical_rerank_json(request.model_dump(mode="json")).encode("utf-8"))
    if request_bytes > policy.max_request_bytes:
        raise LearnedRerankValidationError("serialized learned rerank request exceeds byte bound")
    return request


def canonical_memory_rerank_input(
    memory: Memory,
    policy: LearnedRerankPolicy,
) -> tuple[str, str]:
    """Project canonical memory text and all temporal axes for the provider."""

    document = search_text(
        title=memory.title,
        content=memory.content,
        tags=memory.tags,
        metadata=memory.metadata,
    ).strip()
    if not document:
        document = "(empty memory)"
    document = document[: policy.max_document_characters]
    document = utf8_prefix(document, policy.max_document_bytes)
    temporal_context = canonical_rerank_json(
        {
            "occurred_at": _isoformat(memory.occurred_at),
            "recorded_from": _isoformat(memory.recorded_from),
            "recorded_to": _isoformat(memory.recorded_to),
            "valid_from": _isoformat(memory.valid_from),
            "valid_to": _isoformat(memory.valid_to),
        }
    )
    if (
        len(temporal_context) > policy.max_temporal_characters
        or len(temporal_context.encode("utf-8")) > policy.max_temporal_bytes
    ):
        raise LearnedRerankValidationError("canonical memory temporal context exceeds policy")
    return document, temporal_context


def build_swarm_memory_rerank_request(
    policy: LearnedRerankPolicy,
    *,
    query: str,
    candidates: Sequence[tuple[FusedCandidate, Memory]],
    request_id: str | None = None,
) -> LearnedRerankRequest:
    """Build the core request from canonically hydrated candidates."""

    bounded_query = query[: policy.max_query_characters]
    bounded_query = utf8_prefix(bounded_query, policy.max_query_bytes)
    values = tuple(
        (
            candidate.canonical_id,
            *canonical_memory_rerank_input(memory, policy),
        )
        for candidate, memory in candidates[: policy.window]
    )
    return build_learned_rerank_request(
        policy,
        serializer_revision=SWARMBRAIN_MEMORY_RERANK_SERIALIZER_REVISION,
        query=bounded_query,
        candidates=values,
        request_id=request_id,
    )


def validate_learned_rerank_result(
    request: LearnedRerankRequest,
    result: LearnedRerankResult,
    *,
    expected_identity: LearnedRerankerIdentity,
) -> None:
    """Revalidate every provider-controlled field before scores are trusted."""

    if request.identity != expected_identity:
        raise LearnedRerankValidationError("request identity does not match configured provider")
    if result.receipt.identity != expected_identity:
        raise LearnedRerankValidationError("provider identity mismatch")
    if result.receipt.request_sha256 != request.request_sha256:
        raise LearnedRerankValidationError("provider receipt does not bind the request")
    input_ids = tuple(candidate.candidate_id for candidate in request.candidates)
    result_ids = tuple(score.candidate_id for score in result.scores)
    if result_ids != input_ids:
        raise LearnedRerankValidationError(
            "provider scores must cover exactly the input IDs in input order"
        )
    usage = result.receipt.usage
    expected_accounting = {
        "candidate_count": len(request.candidates),
        "query_characters": len(request.query),
        "document_characters": sum(len(item.document) for item in request.candidates),
        "temporal_characters": sum(len(item.temporal_context) for item in request.candidates),
        "query_bytes": len(request.query.encode("utf-8")),
        "document_bytes": sum(len(item.document.encode("utf-8")) for item in request.candidates),
        "temporal_bytes": sum(
            len(item.temporal_context.encode("utf-8")) for item in request.candidates
        ),
        "request_bytes": len(
            canonical_rerank_json(request.model_dump(mode="json")).encode("utf-8")
        ),
    }
    for field, expected in expected_accounting.items():
        if getattr(usage, field) != expected:
            raise LearnedRerankValidationError(
                f"provider usage {field} does not match the exact request"
            )


def learned_score_reranked(
    baseline: tuple[FusedCandidate, ...],
    scores: Mapping[str, float],
    *,
    alpha: float,
    window: int,
) -> tuple[FusedCandidate, ...]:
    """Reorder scored slots only, with baseline position as the stable tie-break.

    Candidates absent from ``scores`` retain their exact position and score.
    That invariant matters when a fused reference fails canonical hydration:
    the provider never sees it and therefore cannot indirectly move it.  Any
    provider failure bypasses this function entirely, preserving the complete
    pre-learned tuple byte-for-byte.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("learned rerank alpha must be in (0, 1]")
    if not 1 <= window <= 128:
        raise ValueError("learned rerank window must be in [1, 128]")
    if not baseline or not scores:
        return baseline
    head = baseline[:window]
    tail = baseline[window:]
    scored = [(position, item) for position, item in enumerate(head) if item.canonical_id in scores]
    if not scored:
        return baseline
    anchor = max(item.raw_rrf for _position, item in scored)
    if anchor <= 0.0:
        return baseline
    ordered: list[tuple[float, int, FusedCandidate]] = []
    for position, item in scored:
        provider_score = float(scores[item.canonical_id])
        if not 0.0 <= provider_score <= 1.0:
            raise ValueError("learned rerank scores must be finite values in [0, 1]")
        blended = (1.0 - alpha) * (item.raw_rrf / anchor) + alpha * provider_score
        ordered.append((blended, position, item))
    ordered.sort(key=lambda value: (-value[0], value[1]))
    replacements = iter(
        item.model_copy(
            update={
                "reasons": tuple(dict.fromkeys((*item.reasons, "reranker:learned"))),
            }
        )
        for blended, _position, item in ordered
    )
    scored_ids = set(scores)
    reranked_head = tuple(
        next(replacements) if item.canonical_id in scored_ids else item for item in head
    )
    reranked = (*reranked_head, *tail)
    # ``normalized_score`` remains a public rank statement, not a calibrated
    # cross-encoder probability.  Assign the original monotone score ladder by
    # output position so learned order survives consumers that sort by score,
    # including when hydration gaps leave fixed unscored slots or the caller
    # requests results beyond the learned window.  On provider degradation this
    # function is never called, so the original objects and scores are retained
    # exactly.
    score_ladder = tuple(item.normalized_score for item in baseline)
    return tuple(
        item
        if item.normalized_score == score_ladder[position]
        else item.model_copy(update={"normalized_score": score_ladder[position]})
        for position, item in enumerate(reranked)
    )


def build_learned_rerank_result(
    request: LearnedRerankRequest,
    *,
    scores: Sequence[float],
    usage: LearnedRerankUsage,
    provider_request_id: str,
    identity: LearnedRerankerIdentity | None = None,
) -> LearnedRerankResult:
    """Construct a response with its canonical digest (useful to adapters/tests)."""

    if len(scores) != len(request.candidates):
        raise LearnedRerankValidationError("one score is required for every request candidate")
    score_models = tuple(
        LearnedRerankScore(candidate_id=candidate.candidate_id, score=score)
        for candidate, score in zip(request.candidates, scores, strict=True)
    )
    receipt = LearnedRerankReceipt(
        identity=identity or request.identity,
        request_sha256=request.request_sha256,
        provider_request_id=provider_request_id,
        usage=usage,
        response_sha256="0" * 64,
    )
    response_sha256 = rerank_sha256_json(
        learned_rerank_response_payload(scores=score_models, receipt=receipt)
    )
    return LearnedRerankResult(
        scores=score_models,
        receipt=receipt.model_copy(update={"response_sha256": response_sha256}),
    )


def request_usage_dimensions(request: LearnedRerankRequest) -> dict[str, int]:
    """Return exact non-token usage values a provider receipt must echo."""

    return {
        "candidate_count": len(request.candidates),
        "query_characters": len(request.query),
        "document_characters": sum(len(item.document) for item in request.candidates),
        "temporal_characters": sum(len(item.temporal_context) for item in request.candidates),
        "query_bytes": len(request.query.encode("utf-8")),
        "document_bytes": sum(len(item.document.encode("utf-8")) for item in request.candidates),
        "temporal_bytes": sum(
            len(item.temporal_context.encode("utf-8")) for item in request.candidates
        ),
        "request_bytes": len(
            canonical_rerank_json(request.model_dump(mode="json")).encode("utf-8")
        ),
    }


def _isoformat(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


__all__ = [
    "SWARMBRAIN_MEMORY_RERANK_SERIALIZER_REVISION",
    "LearnedRerankValidationError",
    "build_learned_rerank_request",
    "build_learned_rerank_result",
    "build_swarm_memory_rerank_request",
    "canonical_memory_rerank_input",
    "learned_score_reranked",
    "make_learned_rerank_candidate",
    "request_usage_dimensions",
    "utf8_prefix",
    "validate_learned_rerank_result",
]
