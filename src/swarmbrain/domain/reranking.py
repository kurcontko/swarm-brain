"""Strict contracts for the optional learned retrieval reranker.

The reranker is intentionally a *score-only* boundary.  It receives a bounded
ordered set of canonically hydrated candidates and must return one finite score
for every input ID, in the same order.  It never owns candidate generation,
scope, hydration, or filtering.

All content crossing the provider boundary is represented in the persisted
trace only by a digest.  A successful receipt binds the exact request to an
immutable model/tokenizer/deployment/adapter identity and provider-reported
usage.  The application independently revalidates those bindings before using
any score.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Self

from pydantic import (
    Field,
    FiniteFloat,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from .common import ContractModel, SemanticLabel, UUIDString
from .evidence import Sha256

LEARNED_RERANK_REQUEST_SCHEMA = "swarmbrain.learned-rerank.request.v1"
LEARNED_RERANK_RESPONSE_SCHEMA = "swarmbrain.learned-rerank.response.v1"

CandidateId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]
ImmutableRevision = Annotated[
    str,
    StringConstraints(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"),
]


def canonical_rerank_json(value: Any) -> str:
    """Serialize a request/receipt fingerprint without NaN or key-order drift."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def rerank_sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rerank_sha256_json(value: Any) -> str:
    return rerank_sha256_text(canonical_rerank_json(value))


class LearnedRerankerComponent(ContractModel):
    """One ordered scorer component in a single-score learned ranker."""

    role: SemanticLabel
    model: SemanticLabel
    revision: ImmutableRevision
    model_artifact_sha256: Sha256
    tokenizer_revision: ImmutableRevision
    tokenizer_artifact_sha256: Sha256
    weight: FiniteFloat = Field(gt=0.0, le=1.0)

    @field_validator("weight", mode="before")
    @classmethod
    def weight_is_numeric_not_boolean(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("learned reranker component weight must be numeric")
        return value


def learned_reranker_model_bundle_payload(
    components: tuple[LearnedRerankerComponent, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "role": item.role,
            "model": item.model,
            "revision": item.revision,
            "model_artifact_sha256": item.model_artifact_sha256,
            "weight": item.weight,
        }
        for item in components
    ]


def learned_reranker_tokenizer_bundle_payload(
    components: tuple[LearnedRerankerComponent, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "role": item.role,
            "model": item.model,
            "tokenizer_revision": item.tokenizer_revision,
            "tokenizer_artifact_sha256": item.tokenizer_artifact_sha256,
        }
        for item in components
    ]


class LearnedRerankerIdentity(ContractModel):
    """Immutable execution identity required on both request and receipt.

    The top-level artifact hashes are bundle digests over the ordered component
    identities (and, for the model bundle, their fusion weights).  This makes a
    single Qwen3 cross-encoder and a SmartSearch-style CE+ColBERT ``.7/.3``
    scorer equally representable while the port still returns one score.
    """

    provider: SemanticLabel
    model: SemanticLabel
    revision: ImmutableRevision
    components: tuple[LearnedRerankerComponent, ...] = Field(min_length=1, max_length=4)
    model_artifact_sha256: Sha256
    tokenizer_artifact_sha256: Sha256
    deployment_manifest_sha256: Sha256
    adapter_artifact_sha256: Sha256
    runtime_environment_sha256: Sha256
    protocol_revision: SemanticLabel

    @model_validator(mode="after")
    def bundles_bind_ordered_components_and_weights(self) -> Self:
        roles = tuple(item.role for item in self.components)
        if len(roles) != len(set(roles)):
            raise ValueError("learned reranker component roles must be unique")
        if abs(sum(float(item.weight) for item in self.components) - 1.0) > 1e-9:
            raise ValueError("learned reranker component weights must sum to 1.0")
        model_bundle = rerank_sha256_json(learned_reranker_model_bundle_payload(self.components))
        if self.model_artifact_sha256 != model_bundle:
            raise ValueError("model_artifact_sha256 does not bind scorer components")
        tokenizer_bundle = rerank_sha256_json(
            learned_reranker_tokenizer_bundle_payload(self.components)
        )
        if self.tokenizer_artifact_sha256 != tokenizer_bundle:
            raise ValueError("tokenizer_artifact_sha256 does not bind scorer components")
        return self


class LearnedRerankPolicy(ContractModel):
    """Application-owned hard bounds for the opt-in learned stage."""

    identity: LearnedRerankerIdentity
    window: int = Field(default=50, ge=1, le=128)
    alpha: FiniteFloat = Field(default=1.0, gt=0.0, le=1.0)
    timeout_seconds: FiniteFloat = Field(default=20.0, ge=0.05, le=60.0)
    max_query_characters: int = Field(default=8_192, ge=1, le=32_768)
    max_document_characters: int = Field(default=32_768, ge=1, le=262_144)
    max_temporal_characters: int = Field(default=4_096, ge=2, le=8_192)
    max_query_bytes: int = Field(default=32_768, ge=1, le=131_072)
    max_document_bytes: int = Field(default=131_072, ge=1, le=1_048_576)
    max_temporal_bytes: int = Field(default=8_192, ge=2, le=32_768)
    max_request_bytes: int = Field(default=8_388_608, ge=1_024, le=67_108_864)


class LearnedRerankCandidate(ContractModel):
    """One candidate whose ID and canonical projections are self-verifying."""

    candidate_id: CandidateId
    document: str = Field(min_length=1, max_length=262_144)
    document_sha256: Sha256
    temporal_context: str = Field(min_length=2, max_length=8_192)
    temporal_sha256: Sha256

    @model_validator(mode="after")
    def digests_match_content(self) -> Self:
        if rerank_sha256_text(self.document) != self.document_sha256:
            raise ValueError("candidate document_sha256 does not match document")
        if rerank_sha256_text(self.temporal_context) != self.temporal_sha256:
            raise ValueError("candidate temporal_sha256 does not match temporal_context")
        return self


def learned_rerank_candidate_pool_payload(
    candidates: tuple[LearnedRerankCandidate, ...],
) -> list[dict[str, str]]:
    return [
        {
            "candidate_id": candidate.candidate_id,
            "document_sha256": candidate.document_sha256,
            "temporal_sha256": candidate.temporal_sha256,
        }
        for candidate in candidates
    ]


def learned_rerank_request_payload(
    *,
    request_id: str,
    identity: LearnedRerankerIdentity,
    serializer_revision: str,
    query_sha256: str,
    candidate_pool_sha256: str,
) -> dict[str, Any]:
    """Return the content-free canonical payload bound by ``request_sha256``."""

    return {
        "protocol_schema": LEARNED_RERANK_REQUEST_SCHEMA,
        "request_id": request_id,
        "identity": identity.model_dump(mode="json"),
        "serializer_revision": serializer_revision,
        "query_sha256": query_sha256,
        "candidate_pool_sha256": candidate_pool_sha256,
    }


class LearnedRerankRequest(ContractModel):
    protocol_schema: Literal[LEARNED_RERANK_REQUEST_SCHEMA] = LEARNED_RERANK_REQUEST_SCHEMA
    request_id: UUIDString
    identity: LearnedRerankerIdentity
    serializer_revision: SemanticLabel
    query: str = Field(min_length=1, max_length=32_768)
    query_sha256: Sha256
    candidates: tuple[LearnedRerankCandidate, ...] = Field(min_length=1, max_length=128)
    candidate_pool_sha256: Sha256
    request_sha256: Sha256

    @model_validator(mode="after")
    def request_is_self_verifying(self) -> Self:
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("learned rerank candidate IDs must be unique")
        if rerank_sha256_text(self.query) != self.query_sha256:
            raise ValueError("query_sha256 does not match query")
        expected_pool = rerank_sha256_json(learned_rerank_candidate_pool_payload(self.candidates))
        if expected_pool != self.candidate_pool_sha256:
            raise ValueError("candidate_pool_sha256 does not match candidates")
        expected_request = rerank_sha256_json(
            learned_rerank_request_payload(
                request_id=self.request_id,
                identity=self.identity,
                serializer_revision=self.serializer_revision,
                query_sha256=self.query_sha256,
                candidate_pool_sha256=self.candidate_pool_sha256,
            )
        )
        if expected_request != self.request_sha256:
            raise ValueError("request_sha256 does not match the rerank request")
        return self


class LearnedRerankScore(ContractModel):
    candidate_id: CandidateId
    score: FiniteFloat = Field(ge=0.0, le=1.0)

    @field_validator("score", mode="before")
    @classmethod
    def score_is_numeric_not_boolean(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("learned rerank score must be numeric")
        return value


class LearnedRerankUsage(ContractModel):
    """Provider-reported accounting bound into the signed response shape."""

    provider_reported: Literal[True] = True
    candidate_count: StrictInt = Field(ge=1, le=128)
    query_characters: StrictInt = Field(ge=1, le=32_768)
    document_characters: StrictInt = Field(ge=1, le=33_554_432)
    temporal_characters: StrictInt = Field(ge=2, le=1_048_576)
    query_bytes: StrictInt = Field(ge=1, le=131_072)
    document_bytes: StrictInt = Field(ge=1, le=134_217_728)
    temporal_bytes: StrictInt = Field(ge=2, le=4_194_304)
    request_bytes: StrictInt = Field(ge=1, le=67_108_864)
    input_tokens: StrictInt = Field(ge=1, le=1_000_000_000)
    output_tokens: StrictInt = Field(ge=0, le=1_000_000_000)
    total_tokens: StrictInt = Field(ge=1, le=1_000_000_000)
    tokenized_input_sha256: Sha256

    @model_validator(mode="after")
    def total_is_exact(self) -> Self:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens plus output_tokens")
        return self


class LearnedRerankReceipt(ContractModel):
    identity: LearnedRerankerIdentity
    request_sha256: Sha256
    provider_request_id: CandidateId
    usage: LearnedRerankUsage
    response_sha256: Sha256


def learned_rerank_response_payload(
    *,
    scores: tuple[LearnedRerankScore, ...],
    receipt: LearnedRerankReceipt,
) -> dict[str, Any]:
    """Return the canonical provider response excluding its own digest."""

    return {
        "protocol_schema": LEARNED_RERANK_RESPONSE_SCHEMA,
        "scores": [score.model_dump(mode="json") for score in scores],
        "receipt": {
            "identity": receipt.identity.model_dump(mode="json"),
            "request_sha256": receipt.request_sha256,
            "provider_request_id": receipt.provider_request_id,
            "usage": receipt.usage.model_dump(mode="json"),
        },
    }


class LearnedRerankResult(ContractModel):
    protocol_schema: Literal[LEARNED_RERANK_RESPONSE_SCHEMA] = LEARNED_RERANK_RESPONSE_SCHEMA
    scores: tuple[LearnedRerankScore, ...] = Field(min_length=1, max_length=128)
    receipt: LearnedRerankReceipt

    @model_validator(mode="after")
    def response_is_self_verifying(self) -> Self:
        candidate_ids = tuple(score.candidate_id for score in self.scores)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("learned rerank result IDs must be unique")
        expected = rerank_sha256_json(
            learned_rerank_response_payload(scores=self.scores, receipt=self.receipt)
        )
        if expected != self.receipt.response_sha256:
            raise ValueError("response_sha256 does not match the rerank response")
        return self


class LearnedRerankTrace(ContractModel):
    """Content-free evidence for one optional learned-rerank decision."""

    policy: LearnedRerankPolicy
    identity: LearnedRerankerIdentity
    attempted: bool
    applied: bool
    degraded: bool
    serializer_revision: SemanticLabel | None = None
    request_id: UUIDString | None = None
    provider_request_id: CandidateId | None = None
    request_sha256: Sha256 | None = None
    query_sha256: Sha256 | None = None
    candidate_pool_sha256: Sha256 | None = None
    candidate_document_sha256: dict[CandidateId, Sha256] = Field(
        default_factory=dict, max_length=128
    )
    candidate_temporal_sha256: dict[CandidateId, Sha256] = Field(
        default_factory=dict, max_length=128
    )
    input_ids: tuple[CandidateId, ...] = Field(default=(), max_length=128)
    output_ids: tuple[CandidateId, ...] = Field(default=(), max_length=128)
    scores: tuple[LearnedRerankScore, ...] = Field(default=(), max_length=128)
    usage: LearnedRerankUsage | None = None
    response_sha256: Sha256 | None = None
    latency_ms: FiniteFloat = Field(ge=0.0)
    degradation_reason: SemanticLabel | None = None

    @model_validator(mode="after")
    def state_is_unambiguous(self) -> Self:
        if self.identity != self.policy.identity:
            raise ValueError("learned rerank trace identity must equal the policy identity")
        request_fields = (
            self.serializer_revision,
            self.request_id,
            self.request_sha256,
            self.query_sha256,
            self.candidate_pool_sha256,
        )
        has_any_request_identity = any(value is not None for value in request_fields)
        has_complete_request_identity = all(value is not None for value in request_fields)
        if has_any_request_identity != has_complete_request_identity:
            raise ValueError("learned rerank request identity must be complete or absent")
        if len(self.input_ids) != len(set(self.input_ids)):
            raise ValueError("learned rerank trace input IDs must be unique")
        if len(self.output_ids) != len(set(self.output_ids)):
            raise ValueError("learned rerank trace output IDs must be unique")
        if self.applied:
            if not self.attempted or self.degraded:
                raise ValueError("applied learned rerank must be attempted and not degraded")
            if any(value is None for value in request_fields):
                raise ValueError("applied learned rerank requires complete request identity")
            if (
                self.provider_request_id is None
                or self.usage is None
                or self.response_sha256 is None
                or self.degradation_reason is not None
            ):
                raise ValueError("applied learned rerank requires one complete response receipt")
            if (
                not self.input_ids
                or len(self.input_ids) != len(self.output_ids)
                or set(self.input_ids) != set(self.output_ids)
            ):
                raise ValueError("learned rerank output must be an exact input permutation")
            if tuple(score.candidate_id for score in self.scores) != self.input_ids:
                raise ValueError("learned rerank scores must cover input IDs in input order")
        elif self.degraded:
            if not self.attempted or self.degradation_reason is None:
                raise ValueError("degraded learned rerank must record an attempted failure")
            if (
                self.provider_request_id is not None
                or self.output_ids
                or self.scores
                or self.usage is not None
                or self.response_sha256 is not None
            ):
                raise ValueError("untrusted learned rerank output must not enter a degraded trace")
            if has_complete_request_identity != bool(self.input_ids):
                raise ValueError(
                    "degraded request identity and input IDs must be retained together"
                )
        else:
            if self.attempted or any(value is not None for value in request_fields):
                raise ValueError("an empty learned rerank skip must not claim an attempt")
            if (
                self.provider_request_id is not None
                or self.input_ids
                or self.output_ids
                or self.scores
                or self.usage is not None
                or self.response_sha256 is not None
                or self.degradation_reason is not None
            ):
                raise ValueError("an empty learned rerank skip must not contain response evidence")
        if set(self.candidate_document_sha256) != set(self.input_ids):
            raise ValueError("document digests must cover exactly the learned rerank input IDs")
        if set(self.candidate_temporal_sha256) != set(self.input_ids):
            raise ValueError("temporal digests must cover exactly the learned rerank input IDs")
        return self


__all__ = [
    "LEARNED_RERANK_REQUEST_SCHEMA",
    "LEARNED_RERANK_RESPONSE_SCHEMA",
    "CandidateId",
    "ImmutableRevision",
    "LearnedRerankCandidate",
    "LearnedRerankPolicy",
    "LearnedRerankReceipt",
    "LearnedRerankRequest",
    "LearnedRerankResult",
    "LearnedRerankScore",
    "LearnedRerankTrace",
    "LearnedRerankUsage",
    "LearnedRerankerComponent",
    "LearnedRerankerIdentity",
    "canonical_rerank_json",
    "learned_rerank_candidate_pool_payload",
    "learned_rerank_request_payload",
    "learned_rerank_response_payload",
    "learned_reranker_model_bundle_payload",
    "learned_reranker_tokenizer_bundle_payload",
    "rerank_sha256_json",
    "rerank_sha256_text",
]
