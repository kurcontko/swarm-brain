"""Strict raw-provider evidence for the E6 R1 DeepSeek extraction boundary.

This module is deliberately offline.  It builds the one permitted source-only
request and replays retained OpenAI-compatible request/response bytes into the
existing :class:`DerivedKey` and :class:`ConstructionReceipt` contracts.  It
does not make network calls and does not trust caller-normalized model output.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .contracts import (
    EXTRACTOR_INPUT_FIELDS,
    CanonicalValue,
    ConstructionReceipt,
    DerivedKey,
    ExtractorIdentity,
    KeyFamily,
    RepresentationError,
    canonical_json_bytes,
    derived_key_output_binding,
    extraction_request_sha256,
    opaque_navigation_id,
    sha256_bytes,
    sha256_json,
)

ARTIFACT_TYPE = "swarmbrain-longmemeval-e6-r1-deepseek-extraction-evidence"
SCHEMA_VERSION = 1
PROTOCOL_VERSION = "swarmbrain-longmemeval-e6-r1-deepseek-extraction-evidence-v1"

REQUEST_PARSER = "openai-compatible-e6-r1-deepseek-request-exact-v1"
RESPONSE_PARSER = "openai-compatible-e6-r1-deepseek-response-strict-v1"
TOKEN_USAGE_PROTOCOL = "provider-reported-openai-compatible-usage-v1"
RETRY_COST_POLICY = "success-usage-plus-prior-attempts-at-prompt-and-max-output-v1"

EXTRACTOR_PRODUCER = "swarmbrain.longmemeval.e6.r1.deepseek"
EXTRACTOR_PROTOCOL = "swarmbrain-longmemeval-e6-r1-merged-sfk-deepseek-v1"
DEEPSEEK_MODEL_ID = "deepseek-v4-flash"
DEEPSEEK_DEPLOYMENT_ID = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MAX_TOKENS = 512
DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS = 3
DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS = 4
MAX_RAW_REQUEST_BYTES = 2_097_152
MAX_RAW_RESPONSE_BYTES = 1_048_576
MAX_MERGED_SFK_UTF8_BYTES = 8_192

PROMPT_VERSION = "swarmbrain-e6-r1-merged-sfk-source-only-prompt-v1"
PROMPT_INSTRUCTION = (
    "Create exactly one retrieval navigation key from the provided source_value. "
    "Merge a concise summary, durable facts, and discriminative keywords. Use only "
    "the source_value; do not infer missing facts. Return exactly one JSON object "
    'with exactly one field: {"merged_sfk":"non-empty text"}.'
)
USER_PAYLOAD_VERSION = "swarmbrain-e6-r1-source-value-json-v1"
PROMPT_IDENTITY_SHA256 = sha256_json(
    {
        "prompt_version": PROMPT_VERSION,
        "instruction": PROMPT_INSTRUCTION,
        "user_payload_version": USER_PAYLOAD_VERSION,
        "input_fields": list(EXTRACTOR_INPUT_FIELDS),
        "output_schema": {"merged_sfk": "non-empty UTF-8 text"},
    }
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "artifact_type",
        "schema_version",
        "protocol_version",
        "route",
        "extractor",
        "prompt",
        "execution_policy",
        "application_attempts",
        "pricing",
        "normalized",
    }
)
_RAW_BLOCK_FIELDS = frozenset({"parser", "encoding", "raw_bytes", "raw_sha256", "raw_base64"})
_EXECUTION_POLICY_FIELDS = frozenset(
    {
        "endpoint_url",
        "method",
        "content_type",
        "accept_encoding",
        "maximum_application_attempts",
        "maximum_http_attempts_per_application_attempt",
        "application_retry_condition",
        "latency_source",
    }
)
_APPLICATION_ATTEMPT_FIELDS = frozenset(
    {
        "application_attempt",
        "provider_request",
        "provider_response",
        "http_attempts",
        "http_retry_count",
        "latency_microseconds",
        "response",
        "output_validation",
    }
)
_RESPONSE_REQUIRED_FIELDS = frozenset({"id", "model", "choices", "usage"})
_RESPONSE_OPTIONAL_FIELDS = frozenset({"object", "created", "system_fingerprint", "service_tier"})
_CHOICE_REQUIRED_FIELDS = frozenset({"index", "message", "finish_reason"})
_CHOICE_OPTIONAL_FIELDS = frozenset({"logprobs"})
_MESSAGE_REQUIRED_FIELDS = frozenset({"role", "content"})
_MESSAGE_OPTIONAL_FIELDS = frozenset({"reasoning_content", "refusal", "tool_calls"})
_USAGE_REQUIRED_FIELDS = frozenset({"prompt_tokens", "completion_tokens", "total_tokens"})
_USAGE_OPTIONAL_FIELDS = frozenset(
    {
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
    }
)


class RepresentationEvidenceError(RepresentationError):
    """Retained R1 provider evidence cannot support deterministic replay."""


def _reject_json_constant(value: str) -> None:
    raise RepresentationEvidenceError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise RepresentationEvidenceError(f"duplicate JSON field {key!r} is forbidden")
        output[key] = value
    return output


def _strict_json(raw: bytes, *, label: str) -> Any:
    if not isinstance(raw, bytes) or not raw:
        raise RepresentationEvidenceError(f"{label} must be non-empty bytes")
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_fields,
        )
    except (UnicodeError, json.JSONDecodeError, RepresentationEvidenceError, ValueError):
        raise RepresentationEvidenceError(f"{label} is malformed UTF-8 JSON") from None


def _text(value: Any, *, label: str, maximum_bytes: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RepresentationEvidenceError(
            f"{label} must be non-empty text without surrounding whitespace"
        )
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        raise RepresentationEvidenceError(f"{label} must be valid UTF-8") from None
    if len(encoded) > maximum_bytes:
        raise RepresentationEvidenceError(f"{label} exceeds its UTF-8 byte cap")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise RepresentationEvidenceError(f"{label} contains a control character")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RepresentationEvidenceError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result == 0:
        raise RepresentationEvidenceError(f"{label} must be positive")
    return result


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RepresentationEvidenceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact_fields(
    value: Any,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RepresentationEvidenceError(f"{label} must be an object")
    fields = set(value)
    if not required.issubset(fields) or not fields.issubset(required | optional):
        raise RepresentationEvidenceError(f"{label} fields differ from the frozen schema")
    return value


def _validate_endpoint(value: Any) -> str:
    endpoint = _text(value, label="DeepSeek endpoint", maximum_bytes=2048)
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1/chat/completions"
    ):
        raise RepresentationEvidenceError(
            "DeepSeek endpoint must be one credential-free HTTPS /v1/chat/completions URL"
        )
    return endpoint


@dataclass(frozen=True, slots=True)
class DeepSeekR1PricingIdentity:
    """Caller-pinned pricing used for pessimistic integer-microdollar accounting."""

    version: str
    artifact_sha256: str
    cache_miss_input_microusd_per_million_tokens: int
    output_microusd_per_million_tokens: int
    retry_policy: str = RETRY_COST_POLICY

    def __post_init__(self) -> None:
        _text(self.version, label="pricing version")
        _sha256(self.artifact_sha256, label="pricing artifact")
        _positive_int(
            self.cache_miss_input_microusd_per_million_tokens,
            label="cache-miss input price",
        )
        _positive_int(
            self.output_microusd_per_million_tokens,
            label="output price",
        )
        if self.retry_policy != RETRY_COST_POLICY:
            raise RepresentationEvidenceError("pricing retry policy drifted")

    @property
    def identity_sha256(self) -> str:
        return sha256_json(self.binding_without_digest())

    def binding_without_digest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "artifact_sha256": self.artifact_sha256,
            "cache_miss_input_microusd_per_million_tokens": (
                self.cache_miss_input_microusd_per_million_tokens
            ),
            "output_microusd_per_million_tokens": (self.output_microusd_per_million_tokens),
            "retry_policy": self.retry_policy,
            "input_cache_policy": "all-provider-input-tokens-priced-as-cache-misses",
            "cost_unit": "integer-microusd-ceiling",
        }

    def content_free_binding(self) -> dict[str, Any]:
        return {**self.binding_without_digest(), "identity_sha256": self.identity_sha256}

    def upper_bound_microusd(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        retry_count: int,
        request_max_tokens: int,
    ) -> int:
        input_count = _nonnegative_int(input_tokens, label="priced input tokens")
        output_count = _nonnegative_int(output_tokens, label="priced output tokens")
        retries = _nonnegative_int(retry_count, label="priced retry count")
        maximum = _positive_int(request_max_tokens, label="priced request max_tokens")
        successful = (
            input_count * self.cache_miss_input_microusd_per_million_tokens
            + output_count * self.output_microusd_per_million_tokens
        )
        unseen_retry = (
            input_count * self.cache_miss_input_microusd_per_million_tokens
            + maximum * self.output_microusd_per_million_tokens
        )
        numerator = successful + retries * unseen_retry
        return (numerator + 999_999) // 1_000_000


@dataclass(frozen=True, slots=True)
class DeepSeekR1ProviderAttempt:
    """One completed ChatClient call, including its internal HTTP retries."""

    raw_request: bytes = field(repr=False)
    raw_response: bytes = field(repr=False)
    http_attempts: int
    latency_microseconds: int

    def __post_init__(self) -> None:
        if not isinstance(self.raw_request, bytes) or not self.raw_request:
            raise RepresentationEvidenceError("provider attempt request must be non-empty bytes")
        if not isinstance(self.raw_response, bytes) or not self.raw_response:
            raise RepresentationEvidenceError("provider attempt response must be non-empty bytes")
        attempts = _positive_int(self.http_attempts, label="provider HTTP attempts")
        if attempts > DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS:
            raise RepresentationEvidenceError("provider HTTP attempts exceed the frozen maximum")
        _nonnegative_int(self.latency_microseconds, label="provider attempt latency")


@dataclass(frozen=True, slots=True)
class _ReplayedDeepSeekEnvelope:
    content: str = field(repr=False)
    provider_request_id: str
    response_model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    usage_sha256: str
    system_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class ReplayedDeepSeekR1Response:
    """Strict normalized projection of retained DeepSeek response bytes."""

    merged_sfk: str = field(repr=False)
    provider_request_id: str
    response_model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    usage_sha256: str
    system_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class DeepSeekR1ExtractionEvidence:
    """Replayed content-bearing evidence plus E6-native normalized objects."""

    source: CanonicalValue
    extractor: ExtractorIdentity
    pricing: DeepSeekR1PricingIdentity
    derived_key: DerivedKey
    construction_receipt: ConstructionReceipt
    provider_request_id: str
    response_model: str
    finish_reason: str
    total_tokens: int
    usage_sha256: str
    endpoint_url: str
    selected_application_attempt: int
    application_attempts: tuple[DeepSeekR1ProviderAttempt, ...] = field(repr=False)
    record_sha256: str

    @property
    def raw_request(self) -> bytes:
        return self.application_attempts[self.selected_application_attempt - 1].raw_request

    @property
    def raw_response(self) -> bytes:
        return self.application_attempts[self.selected_application_attempt - 1].raw_response

    @property
    def raw_request_sha256(self) -> str:
        return sha256_bytes(self.raw_request)

    @property
    def raw_response_sha256(self) -> str:
        return sha256_bytes(self.raw_response)

    def content_free_binding(self) -> dict[str, Any]:
        return {
            "record_sha256": self.record_sha256,
            "route": _route_binding(self.source),
            "extractor": self.extractor.content_free_binding(),
            "pricing": self.pricing.content_free_binding(),
            "derived_key": self.derived_key.content_free_binding(),
            "construction_receipt": self.construction_receipt.content_free_binding(),
            "provider_request_id_sha256": sha256_bytes(self.provider_request_id.encode("utf-8")),
            "response_model": self.response_model,
            "finish_reason": self.finish_reason,
            "total_tokens": self.total_tokens,
            "usage_sha256": self.usage_sha256,
            "endpoint_url": self.endpoint_url,
            "selected_application_attempt": self.selected_application_attempt,
            "application_attempt_count": len(self.application_attempts),
            "total_http_attempts": sum(
                attempt.http_attempts for attempt in self.application_attempts
            ),
            "raw_request_sha256": self.raw_request_sha256,
            "raw_response_sha256": self.raw_response_sha256,
        }


def deepseek_r1_extractor_identity(
    *,
    model_revision: str,
    model_artifact_sha256: str,
    identity_artifact_sha256: str,
) -> ExtractorIdentity:
    """Build the only extractor identity admitted by this R1 boundary."""

    return ExtractorIdentity(
        producer=EXTRACTOR_PRODUCER,
        protocol=EXTRACTOR_PROTOCOL,
        model_id=DEEPSEEK_MODEL_ID,
        model_revision=model_revision,
        deployment_id=DEEPSEEK_DEPLOYMENT_ID,
        model_artifact_sha256=model_artifact_sha256,
        prompt_sha256=PROMPT_IDENTITY_SHA256,
        identity_artifact_sha256=identity_artifact_sha256,
    )


def _validate_extractor(extractor: ExtractorIdentity) -> None:
    if not isinstance(extractor, ExtractorIdentity):
        raise RepresentationEvidenceError("R1 evidence requires ExtractorIdentity")
    expected = {
        "producer": EXTRACTOR_PRODUCER,
        "protocol": EXTRACTOR_PROTOCOL,
        "model_id": DEEPSEEK_MODEL_ID,
        "deployment_id": DEEPSEEK_DEPLOYMENT_ID,
        "prompt_sha256": PROMPT_IDENTITY_SHA256,
    }
    for field_name, wanted in expected.items():
        if getattr(extractor, field_name) != wanted:
            raise RepresentationEvidenceError(f"R1 extractor {field_name} drifted")


def render_deepseek_r1_prompt(source: CanonicalValue) -> str:
    """Render the exact prompt whose only dynamic field is ``source_value``."""

    if not isinstance(source, CanonicalValue):
        raise RepresentationEvidenceError("R1 prompt requires CanonicalValue")
    payload = canonical_json_bytes({"source_value": source.raw_value}).decode("utf-8")
    return f"{PROMPT_INSTRUCTION}\nINPUT_JSON:\n{payload}"


def deepseek_r1_request_bytes(
    source: CanonicalValue,
    extractor: ExtractorIdentity,
) -> bytes:
    """Build canonical credential-free request bytes for one R1 source value."""

    _validate_extractor(extractor)
    prompt = render_deepseek_r1_prompt(source)
    raw = canonical_json_bytes(
        {
            "model": extractor.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": DEEPSEEK_MAX_TOKENS,
            "thinking": {"type": "disabled"},
        }
    )
    if len(raw) > MAX_RAW_REQUEST_BYTES:
        raise RepresentationEvidenceError("R1 provider request exceeds its byte cap")
    return raw


def _route_binding(source: CanonicalValue) -> dict[str, Any]:
    return {
        "family": KeyFamily.MERGED_SFK.value,
        "canonical_value": source.content_free_binding(),
    }


def _prompt_binding(source: CanonicalValue) -> dict[str, Any]:
    rendered = render_deepseek_r1_prompt(source).encode("utf-8")
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_identity_sha256": PROMPT_IDENTITY_SHA256,
        "extractor_prompt_sha256": PROMPT_IDENTITY_SHA256,
        "user_payload_version": USER_PAYLOAD_VERSION,
        "input_fields": list(EXTRACTOR_INPUT_FIELDS),
        "rendered_prompt_utf8_bytes": len(rendered),
        "rendered_prompt_sha256": sha256_bytes(rendered),
        "question_id_is_routing_metadata_not_model_input": True,
    }


def _raw_block(raw: bytes, *, parser: str, encoding: str) -> dict[str, Any]:
    return {
        "parser": parser,
        "encoding": encoding,
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "raw_base64": base64.b64encode(raw).decode("ascii"),
    }


def _decode_raw_block(
    value: Any,
    *,
    parser: str,
    encoding: str,
    maximum_bytes: int,
    label: str,
) -> bytes:
    if not isinstance(value, dict) or set(value) != _RAW_BLOCK_FIELDS:
        raise RepresentationEvidenceError(f"{label} binding fields differ from the schema")
    if value.get("parser") != parser or value.get("encoding") != encoding:
        raise RepresentationEvidenceError(f"{label} parser or encoding drifted")
    encoded = value.get("raw_base64")
    if not isinstance(encoded, str) or not encoded:
        raise RepresentationEvidenceError(f"{label} raw bytes must be non-empty base64 text")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise RepresentationEvidenceError(f"{label} raw bytes are invalid base64") from None
    byte_count = _positive_int(value.get("raw_bytes"), label=f"{label} raw byte count")
    if byte_count != len(raw) or len(raw) > maximum_bytes:
        raise RepresentationEvidenceError(f"{label} raw byte count is inconsistent")
    if _sha256(value.get("raw_sha256"), label=f"{label} raw digest") != sha256_bytes(raw):
        raise RepresentationEvidenceError(f"{label} raw digest is inconsistent")
    return raw


def _usage_int(usage: Mapping[str, Any], field_name: str) -> int:
    return _nonnegative_int(
        usage.get(field_name),
        label=f"DeepSeek response usage.{field_name}",
    )


def _replay_deepseek_response_envelope(raw_response: bytes) -> _ReplayedDeepSeekEnvelope:
    """Replay the provider envelope even when application output is invalid."""

    if not isinstance(raw_response, bytes) or not raw_response:
        raise RepresentationEvidenceError("DeepSeek raw response must be non-empty bytes")
    if len(raw_response) > MAX_RAW_RESPONSE_BYTES:
        raise RepresentationEvidenceError("DeepSeek raw response exceeds its byte cap")
    body = _strict_json(raw_response, label="DeepSeek response")
    body = _exact_fields(
        body,
        required=_RESPONSE_REQUIRED_FIELDS,
        optional=_RESPONSE_OPTIONAL_FIELDS,
        label="DeepSeek response",
    )
    provider_request_id = _text(body.get("id"), label="DeepSeek provider request id")
    response_model = _text(body.get("model"), label="DeepSeek response model")
    if response_model != DEEPSEEK_MODEL_ID:
        raise RepresentationEvidenceError("DeepSeek response model differs from the frozen alias")
    if "created" in body:
        _nonnegative_int(body.get("created"), label="DeepSeek response created")
    if "object" in body and body.get("object") != "chat.completion":
        raise RepresentationEvidenceError("DeepSeek response object type drifted")
    system_fingerprint = body.get("system_fingerprint")
    if system_fingerprint is not None:
        system_fingerprint = _text(
            system_fingerprint,
            label="DeepSeek response system fingerprint",
        )
    if "service_tier" in body and body.get("service_tier") is not None:
        _text(body.get("service_tier"), label="DeepSeek response service tier")

    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RepresentationEvidenceError("DeepSeek response must contain exactly one choice")
    choice = _exact_fields(
        choices[0],
        required=_CHOICE_REQUIRED_FIELDS,
        optional=_CHOICE_OPTIONAL_FIELDS,
        label="DeepSeek response choice",
    )
    if isinstance(choice.get("index"), bool) or choice.get("index") != 0:
        raise RepresentationEvidenceError("DeepSeek response choice index must be integer zero")
    if choice.get("logprobs") is not None:
        raise RepresentationEvidenceError("DeepSeek response logprobs must be null or absent")
    finish_reason = _text(choice.get("finish_reason"), label="DeepSeek finish reason")
    message = _exact_fields(
        choice.get("message"),
        required=_MESSAGE_REQUIRED_FIELDS,
        optional=_MESSAGE_OPTIONAL_FIELDS,
        label="DeepSeek response message",
    )
    if message.get("role") != "assistant":
        raise RepresentationEvidenceError("DeepSeek response message role must be assistant")
    if message.get("refusal") is not None or message.get("tool_calls") not in (None, []):
        raise RepresentationEvidenceError("DeepSeek response contains a refusal or tool call")
    if message.get("reasoning_content") not in (None, ""):
        raise RepresentationEvidenceError("thinking-disabled response contains reasoning content")
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise RepresentationEvidenceError("DeepSeek response content must be text or null")

    usage = _exact_fields(
        body.get("usage"),
        required=_USAGE_REQUIRED_FIELDS,
        optional=_USAGE_OPTIONAL_FIELDS,
        label="DeepSeek response usage",
    )
    prompt_tokens = _usage_int(usage, "prompt_tokens")
    completion_tokens = _usage_int(usage, "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    if total_tokens != prompt_tokens + completion_tokens:
        raise RepresentationEvidenceError("DeepSeek response token usage does not reconcile")
    for optional_integer in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
        if optional_integer in usage:
            _usage_int(usage, optional_integer)
    for optional_details in ("prompt_tokens_details", "completion_tokens_details"):
        if optional_details in usage and not isinstance(usage[optional_details], dict):
            raise RepresentationEvidenceError(
                f"DeepSeek response usage.{optional_details} must be an object"
            )
        if optional_details in usage:
            canonical_json_bytes(usage[optional_details])
    return _ReplayedDeepSeekEnvelope(
        content=content,
        provider_request_id=provider_request_id,
        response_model=response_model,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        usage_sha256=sha256_json(usage),
        system_fingerprint=system_fingerprint,
    )


class _MergedSFKOutputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _merged_sfk_from_envelope(replayed: _ReplayedDeepSeekEnvelope) -> str:
    if replayed.finish_reason != "stop":
        raise _MergedSFKOutputError(
            "finish-reason-not-stop",
            "merged-SFK output did not finish with reason stop",
        )
    content = replayed.content
    if not content or content != content.strip():
        raise _MergedSFKOutputError(
            "content-empty-or-surrounded",
            "merged-SFK content is empty or has surrounding whitespace",
        )
    try:
        parsed_content = _strict_json(
            content.encode("utf-8"),
            label="merged-SFK response content",
        )
    except RepresentationEvidenceError:
        raise _MergedSFKOutputError(
            "content-malformed-json",
            "merged-SFK content is malformed JSON",
        ) from None
    if not isinstance(parsed_content, dict) or set(parsed_content) != {"merged_sfk"}:
        raise _MergedSFKOutputError(
            "content-schema-mismatch",
            "merged-SFK content does not contain exactly merged_sfk",
        )
    merged_sfk = parsed_content.get("merged_sfk")
    if not isinstance(merged_sfk, str) or not merged_sfk or merged_sfk != merged_sfk.strip():
        raise _MergedSFKOutputError(
            "merged-sfk-invalid-text",
            "merged_sfk is not non-empty trimmed text",
        )
    try:
        merged_bytes = merged_sfk.encode("utf-8")
    except UnicodeError:
        raise _MergedSFKOutputError(
            "merged-sfk-invalid-utf8",
            "merged_sfk is not valid UTF-8",
        ) from None
    if len(merged_bytes) > MAX_MERGED_SFK_UTF8_BYTES:
        raise _MergedSFKOutputError(
            "merged-sfk-byte-cap",
            "merged_sfk exceeds its UTF-8 byte cap",
        )
    return merged_sfk


def replay_deepseek_r1_response(raw_response: bytes) -> ReplayedDeepSeekR1Response:
    """Strictly parse one valid provider response and nested merged-SFK JSON."""

    replayed = _replay_deepseek_response_envelope(raw_response)
    try:
        merged_sfk = _merged_sfk_from_envelope(replayed)
    except _MergedSFKOutputError as exc:
        raise RepresentationEvidenceError(str(exc)) from None
    return ReplayedDeepSeekR1Response(
        merged_sfk=merged_sfk,
        provider_request_id=replayed.provider_request_id,
        response_model=replayed.response_model,
        finish_reason=replayed.finish_reason,
        prompt_tokens=replayed.prompt_tokens,
        completion_tokens=replayed.completion_tokens,
        total_tokens=replayed.total_tokens,
        usage_sha256=replayed.usage_sha256,
        system_fingerprint=replayed.system_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class _AttemptReplay:
    attempt: DeepSeekR1ProviderAttempt
    envelope: _ReplayedDeepSeekEnvelope
    merged_sfk: str | None = field(repr=False)
    error_code: str | None
    record: dict[str, Any] = field(repr=False)


def _application_attempt_material(
    *,
    source: CanonicalValue,
    extractor: ExtractorIdentity,
    attempt: DeepSeekR1ProviderAttempt,
    application_attempt: int,
) -> _AttemptReplay:
    index = _positive_int(application_attempt, label="application attempt")
    if index > DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS:
        raise RepresentationEvidenceError("application attempt exceeds the frozen maximum")
    if not isinstance(attempt, DeepSeekR1ProviderAttempt):
        raise RepresentationEvidenceError("application attempt has an invalid type")
    expected_request = deepseek_r1_request_bytes(source, extractor)
    if attempt.raw_request != expected_request:
        raise RepresentationEvidenceError(
            "application attempt request differs from the exact source-only request"
        )
    envelope = _replay_deepseek_response_envelope(attempt.raw_response)
    merged_sfk: str | None
    error_code: str | None
    try:
        merged_sfk = _merged_sfk_from_envelope(envelope)
    except _MergedSFKOutputError as exc:
        merged_sfk = None
        error_code = exc.code
    else:
        error_code = None
    response_binding = {
        "provider_request_id": envelope.provider_request_id,
        "response_model": envelope.response_model,
        "system_fingerprint": envelope.system_fingerprint,
        "finish_reason": envelope.finish_reason,
        "usage": {
            "protocol": TOKEN_USAGE_PROTOCOL,
            "prompt_tokens": envelope.prompt_tokens,
            "completion_tokens": envelope.completion_tokens,
            "total_tokens": envelope.total_tokens,
            "usage_sha256": envelope.usage_sha256,
        },
    }
    record = {
        "application_attempt": index,
        "provider_request": _raw_block(
            attempt.raw_request,
            parser=REQUEST_PARSER,
            encoding="base64-exact-canonical-http-request-body",
        ),
        "provider_response": _raw_block(
            attempt.raw_response,
            parser=RESPONSE_PARSER,
            encoding="base64-exact-decoded-http-response-body",
        ),
        "http_attempts": attempt.http_attempts,
        "http_retry_count": attempt.http_attempts - 1,
        "latency_microseconds": attempt.latency_microseconds,
        "response": response_binding,
        "output_validation": {
            "protocol": "strict-merged-sfk-json-schema-v1",
            "accepted": merged_sfk is not None,
            "error_code": error_code,
        },
    }
    return _AttemptReplay(
        attempt=attempt,
        envelope=envelope,
        merged_sfk=merged_sfk,
        error_code=error_code,
        record=record,
    )


def build_deepseek_r1_attempt_record(
    *,
    source: CanonicalValue,
    extractor: ExtractorIdentity,
    attempt: DeepSeekR1ProviderAttempt,
    application_attempt: int,
) -> dict[str, Any]:
    """Normalize one raw attempt without discarding invalid application output."""

    if not isinstance(source, CanonicalValue):
        raise RepresentationEvidenceError("R1 attempt source must be CanonicalValue")
    _validate_extractor(extractor)
    return _application_attempt_material(
        source=source,
        extractor=extractor,
        attempt=attempt,
        application_attempt=application_attempt,
    ).record


def replay_deepseek_r1_attempt_record(
    record: Any,
    *,
    source: CanonicalValue,
    extractor: ExtractorIdentity,
    application_attempt: int,
) -> DeepSeekR1ProviderAttempt:
    """Recover one persisted attempt, whether its application output passed or failed."""

    if not isinstance(source, CanonicalValue):
        raise RepresentationEvidenceError("R1 attempt replay source must be CanonicalValue")
    _validate_extractor(extractor)
    index = _positive_int(application_attempt, label="application attempt")
    if not isinstance(record, dict) or set(record) != _APPLICATION_ATTEMPT_FIELDS:
        raise RepresentationEvidenceError("R1 application attempt fields drifted")
    if record.get("application_attempt") != index:
        raise RepresentationEvidenceError("R1 application attempt index drifted")
    raw_request = _decode_raw_block(
        record.get("provider_request"),
        parser=REQUEST_PARSER,
        encoding="base64-exact-canonical-http-request-body",
        maximum_bytes=MAX_RAW_REQUEST_BYTES,
        label=f"application attempt {index} provider request",
    )
    raw_response = _decode_raw_block(
        record.get("provider_response"),
        parser=RESPONSE_PARSER,
        encoding="base64-exact-decoded-http-response-body",
        maximum_bytes=MAX_RAW_RESPONSE_BYTES,
        label=f"application attempt {index} provider response",
    )
    http_attempts = _positive_int(
        record.get("http_attempts"),
        label=f"application attempt {index} HTTP attempts",
    )
    if http_attempts > DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS:
        raise RepresentationEvidenceError("R1 HTTP attempts exceed the frozen maximum")
    if record.get("http_retry_count") != http_attempts - 1:
        raise RepresentationEvidenceError("R1 HTTP retry count does not reconcile")
    latency = _nonnegative_int(
        record.get("latency_microseconds"),
        label=f"application attempt {index} latency",
    )
    attempt = DeepSeekR1ProviderAttempt(
        raw_request=raw_request,
        raw_response=raw_response,
        http_attempts=http_attempts,
        latency_microseconds=latency,
    )
    expected = _application_attempt_material(
        source=source,
        extractor=extractor,
        attempt=attempt,
        application_attempt=index,
    ).record
    if record != expected:
        raise RepresentationEvidenceError("R1 application attempt differs from exact raw replay")
    return attempt


def deepseek_r1_attempt_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize one in-progress application-attempt ledger as canonical JSONL."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise RepresentationEvidenceError("R1 attempt records must be a sequence")
    if not 1 <= len(records) <= DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS:
        raise RepresentationEvidenceError("R1 attempt record count is outside the frozen bound")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise RepresentationEvidenceError("R1 attempt record must be a mapping")
        item = dict(record)
        if item.get("application_attempt") != index:
            raise RepresentationEvidenceError("R1 attempt record indices must be contiguous")
        normalized.append(item)
    return b"".join(canonical_json_bytes(record) + b"\n" for record in normalized)


def replay_deepseek_r1_attempt_jsonl(
    raw: bytes,
    *,
    source: CanonicalValue,
    extractor: ExtractorIdentity,
) -> tuple[DeepSeekR1ProviderAttempt, ...]:
    """Recover a canonical in-progress ledger without discarding invalid outputs."""

    if not isinstance(raw, bytes) or not raw or not raw.endswith(b"\n"):
        raise RepresentationEvidenceError("R1 attempt JSONL must be non-empty and LF-terminated")
    if b"\r" in raw:
        raise RepresentationEvidenceError("R1 attempt JSONL must use byte-exact LF delimiters")
    lines = raw.split(b"\n")[:-1]
    if not 1 <= len(lines) <= DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS:
        raise RepresentationEvidenceError("R1 attempt JSONL line count is outside the frozen bound")
    attempts: list[DeepSeekR1ProviderAttempt] = []
    accepted_indices: list[int] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise RepresentationEvidenceError(
                f"R1 attempt JSONL contains an empty line at {line_number}"
            )
        record = _strict_json(line, label=f"R1 attempt JSONL line {line_number}")
        if not isinstance(record, dict) or canonical_json_bytes(record) != line:
            raise RepresentationEvidenceError(
                f"R1 attempt JSONL line {line_number} is not canonical"
            )
        attempt = replay_deepseek_r1_attempt_record(
            record,
            source=source,
            extractor=extractor,
            application_attempt=line_number,
        )
        attempts.append(attempt)
        if record["output_validation"]["accepted"]:
            accepted_indices.append(line_number)
    if accepted_indices not in ([], [len(lines)]):
        raise RepresentationEvidenceError(
            "R1 attempt JSONL must stop at its first schema-valid response"
        )
    return tuple(attempts)


def load_deepseek_r1_attempt_artifact(
    path: Path,
    *,
    source: CanonicalValue,
    extractor: ExtractorIdentity,
) -> tuple[DeepSeekR1ProviderAttempt, ...]:
    """Read and replay one canonical in-progress attempt-ledger artifact."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RepresentationEvidenceError(f"cannot read R1 attempt artifact {path}: {exc}") from exc
    return replay_deepseek_r1_attempt_jsonl(
        raw,
        source=source,
        extractor=extractor,
    )


def _receipt_and_key(
    *,
    source: CanonicalValue,
    extractor: ExtractorIdentity,
    pricing: DeepSeekR1PricingIdentity,
    attempt_replays: tuple[_AttemptReplay, ...],
    selected: ReplayedDeepSeekR1Response,
    endpoint_url: str,
) -> tuple[DerivedKey, ConstructionReceipt]:
    raw_request = attempt_replays[0].attempt.raw_request
    request_digest = sha256_bytes(raw_request)
    ledger_digest = sha256_json([item.record for item in attempt_replays])
    response_digest = sha256_json(
        [
            {
                "application_attempt": index,
                "raw_response_sha256": sha256_bytes(item.attempt.raw_response),
            }
            for index, item in enumerate(attempt_replays, start=1)
        ]
    )
    receipt_id = opaque_navigation_id(
        prefix="receipt",
        material={
            "protocol_version": PROTOCOL_VERSION,
            "family": KeyFamily.MERGED_SFK.value,
            "source_value_id": source.value_id,
            "source_version_sha256": source.source_version_sha256,
            "raw_value_sha256": source.raw_value_sha256,
            "extractor_identity_sha256": extractor.identity_sha256,
            "provider_request_sha256": request_digest,
            "application_attempt_ledger_sha256": ledger_digest,
            "provider_response_ledger_sha256": response_digest,
        },
    )
    key_text = selected.merged_sfk
    key_text_bytes = key_text.encode("utf-8")
    key_id = opaque_navigation_id(
        prefix="key",
        material={
            "protocol_version": PROTOCOL_VERSION,
            "family": KeyFamily.MERGED_SFK.value,
            "receipt_id": receipt_id,
            "key_text_sha256": sha256_bytes(key_text_bytes),
            "key_text_utf8_bytes": len(key_text_bytes),
        },
    )
    key = DerivedKey.create(
        key_id=key_id,
        family=KeyFamily.MERGED_SFK,
        source=source,
        key_text=key_text,
        construction_receipt_id=receipt_id,
    )
    output_binding = derived_key_output_binding((key,))
    input_tokens = sum(item.envelope.prompt_tokens for item in attempt_replays)
    output_tokens = sum(item.envelope.completion_tokens for item in attempt_replays)
    total_http_attempts = sum(item.attempt.http_attempts for item in attempt_replays)
    retry_count = total_http_attempts - 1
    latency_microseconds = sum(item.attempt.latency_microseconds for item in attempt_replays)
    cost_microusd = sum(
        pricing.upper_bound_microusd(
            input_tokens=item.envelope.prompt_tokens,
            output_tokens=item.envelope.completion_tokens,
            retry_count=item.attempt.http_attempts - 1,
            request_max_tokens=DEEPSEEK_MAX_TOKENS,
        )
        for item in attempt_replays
    )
    artifact_sha256 = sha256_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "route": _route_binding(source),
            "extractor": extractor.content_free_binding(),
            "prompt": _prompt_binding(source),
            "provider_request": {
                "bytes": len(raw_request),
                "sha256": request_digest,
            },
            "application_attempts": {
                "count": len(attempt_replays),
                "ledger_sha256": ledger_digest,
                "provider_response_ledger_sha256": response_digest,
                "selected_application_attempt": len(attempt_replays),
            },
            "execution_policy": {
                "endpoint_url": endpoint_url,
                "maximum_application_attempts": DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS,
                "maximum_http_attempts_per_application_attempt": (DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS),
                "retry_count": retry_count,
                "latency_microseconds": latency_microseconds,
            },
            "pricing": pricing.content_free_binding(),
            "usage": {
                "protocol": TOKEN_USAGE_PROTOCOL,
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "estimated_cost_upper_bound_microusd": cost_microusd,
            },
            "selected_provider_request_id": selected.provider_request_id,
            "response_model": selected.response_model,
            "finish_reason": selected.finish_reason,
            "output_keys": output_binding,
        }
    )
    receipt = ConstructionReceipt(
        receipt_id=receipt_id,
        family=KeyFamily.MERGED_SFK,
        source_value_id=source.value_id,
        question_id=source.question_id,
        source_artifact_sha256=source.source_artifact_sha256,
        projection_sha256=source.projection_sha256,
        source_version_sha256=source.source_version_sha256,
        raw_value_sha256=source.raw_value_sha256,
        raw_value_utf8_bytes=source.raw_value_utf8_bytes,
        extractor=extractor,
        construction_artifact_sha256=artifact_sha256,
        input_fields=EXTRACTOR_INPUT_FIELDS,
        source_input_sha256=source.raw_value_sha256,
        request_sha256=extraction_request_sha256(
            family=KeyFamily.MERGED_SFK,
            raw_value_sha256=source.raw_value_sha256,
            raw_value_utf8_bytes=source.raw_value_utf8_bytes,
            extractor=extractor,
        ),
        response_sha256=response_digest,
        output_key_ids=(key.key_id,),
        output_keys_sha256=sha256_json(output_binding),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_microseconds=latency_microseconds,
        cost_microusd=cost_microusd,
        retry_count=retry_count,
        cache_hit=False,
        complete=True,
    )
    return key, receipt


def _normalized_binding(
    *,
    selected: ReplayedDeepSeekR1Response,
    attempt_replays: tuple[_AttemptReplay, ...],
    pricing: DeepSeekR1PricingIdentity,
    key: DerivedKey,
    receipt: ConstructionReceipt,
) -> dict[str, Any]:
    total_http_attempts = sum(item.attempt.http_attempts for item in attempt_replays)
    return {
        "selected_application_attempt": len(attempt_replays),
        "selected_provider_request_id": selected.provider_request_id,
        "response_model": selected.response_model,
        "system_fingerprint": selected.system_fingerprint,
        "finish_reason": selected.finish_reason,
        "aggregate_accounting": {
            "protocol": TOKEN_USAGE_PROTOCOL,
            "application_attempts": len(attempt_replays),
            "http_attempts": total_http_attempts,
            "retry_count": total_http_attempts - 1,
            "input_tokens": receipt.input_tokens,
            "output_tokens": receipt.output_tokens,
            "total_tokens": receipt.input_tokens + receipt.output_tokens,
            "latency_microseconds": receipt.latency_microseconds,
            "estimated_cost_upper_bound_microusd": receipt.cost_microusd,
            "pricing_identity_sha256": pricing.identity_sha256,
        },
        "derived_key": key.content_free_binding(),
        "construction_receipt": receipt.content_free_binding(),
    }


def build_deepseek_r1_evidence_record(
    *,
    source: CanonicalValue,
    extractor: ExtractorIdentity,
    pricing: DeepSeekR1PricingIdentity,
    application_attempts: Sequence[DeepSeekR1ProviderAttempt],
    endpoint_url: str = DEEPSEEK_DEPLOYMENT_ID,
) -> dict[str, Any]:
    """Build one row whose last attempt is the first schema-valid response."""

    if not isinstance(source, CanonicalValue):
        raise RepresentationEvidenceError("R1 evidence source must be CanonicalValue")
    _validate_extractor(extractor)
    if not isinstance(pricing, DeepSeekR1PricingIdentity):
        raise RepresentationEvidenceError("R1 evidence requires pricing identity")
    endpoint = _validate_endpoint(endpoint_url)
    if endpoint != extractor.deployment_id:
        raise RepresentationEvidenceError("provider endpoint differs from extractor deployment")
    if not isinstance(application_attempts, Sequence) or isinstance(
        application_attempts, (str, bytes)
    ):
        raise RepresentationEvidenceError("application attempts must be a sequence")
    attempts = tuple(application_attempts)
    if not 1 <= len(attempts) <= DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS:
        raise RepresentationEvidenceError("application attempt count is outside the frozen bound")
    attempt_replays = tuple(
        _application_attempt_material(
            source=source,
            extractor=extractor,
            attempt=attempt,
            application_attempt=index,
        )
        for index, attempt in enumerate(attempts, start=1)
    )
    accepted = [index for index, item in enumerate(attempt_replays) if item.merged_sfk is not None]
    if accepted != [len(attempt_replays) - 1]:
        raise RepresentationEvidenceError(
            "application ledger must stop at its first schema-valid response"
        )
    selected_envelope = attempt_replays[-1].envelope
    selected = ReplayedDeepSeekR1Response(
        merged_sfk=attempt_replays[-1].merged_sfk or "",
        provider_request_id=selected_envelope.provider_request_id,
        response_model=selected_envelope.response_model,
        finish_reason=selected_envelope.finish_reason,
        prompt_tokens=selected_envelope.prompt_tokens,
        completion_tokens=selected_envelope.completion_tokens,
        total_tokens=selected_envelope.total_tokens,
        usage_sha256=selected_envelope.usage_sha256,
        system_fingerprint=selected_envelope.system_fingerprint,
    )
    key, receipt = _receipt_and_key(
        source=source,
        extractor=extractor,
        pricing=pricing,
        attempt_replays=attempt_replays,
        selected=selected,
        endpoint_url=endpoint,
    )
    execution_policy = {
        "endpoint_url": endpoint,
        "method": "POST",
        "content_type": "application/json",
        "accept_encoding": "identity",
        "maximum_application_attempts": DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS,
        "maximum_http_attempts_per_application_attempt": DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS,
        "application_retry_condition": "strict-merged-sfk-output-invalid",
        "latency_source": "caller-observed-monotonic-clock",
    }
    record = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "route": _route_binding(source),
        "extractor": extractor.content_free_binding(),
        "prompt": _prompt_binding(source),
        "execution_policy": execution_policy,
        "application_attempts": [item.record for item in attempt_replays],
        "pricing": pricing.content_free_binding(),
        "normalized": _normalized_binding(
            selected=selected,
            attempt_replays=attempt_replays,
            pricing=pricing,
            key=key,
            receipt=receipt,
        ),
    }
    # Make construction reject its own output if any serializer assumption drifts.
    replay_deepseek_r1_evidence_record(
        record,
        source=source,
        extractor=extractor,
        pricing=pricing,
    )
    return record


def replay_deepseek_r1_evidence_record(
    record: Any,
    *,
    source: CanonicalValue,
    extractor: ExtractorIdentity,
    pricing: DeepSeekR1PricingIdentity,
) -> DeepSeekR1ExtractionEvidence:
    """Rebuild E6-native key/receipt objects from one exact retained row."""

    if not isinstance(record, dict) or set(record) != _TOP_LEVEL_FIELDS:
        raise RepresentationEvidenceError("R1 evidence fields differ from the exact schema")
    expected_identity = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
    }
    for field_name, wanted in expected_identity.items():
        if type(record.get(field_name)) is not type(wanted) or record.get(field_name) != wanted:
            raise RepresentationEvidenceError(f"R1 evidence {field_name} drifted")
    if not isinstance(source, CanonicalValue):
        raise RepresentationEvidenceError("R1 replay source must be CanonicalValue")
    _validate_extractor(extractor)
    if not isinstance(pricing, DeepSeekR1PricingIdentity):
        raise RepresentationEvidenceError("R1 replay requires pricing identity")
    if record.get("route") != _route_binding(source):
        raise RepresentationEvidenceError("R1 route differs from the authoritative source")
    if record.get("extractor") != extractor.content_free_binding():
        raise RepresentationEvidenceError("R1 extractor identity differs from the pinned identity")
    if record.get("prompt") != _prompt_binding(source):
        raise RepresentationEvidenceError("R1 prompt identity differs from source-only rendering")
    if record.get("pricing") != pricing.content_free_binding():
        raise RepresentationEvidenceError("R1 pricing identity differs from the pinned identity")

    execution_policy = record.get("execution_policy")
    if not isinstance(execution_policy, dict) or set(execution_policy) != _EXECUTION_POLICY_FIELDS:
        raise RepresentationEvidenceError("R1 execution policy fields differ from the schema")
    if (
        execution_policy.get("method") != "POST"
        or execution_policy.get("content_type") != "application/json"
        or execution_policy.get("accept_encoding") != "identity"
        or execution_policy.get("latency_source") != "caller-observed-monotonic-clock"
        or execution_policy.get("application_retry_condition") != "strict-merged-sfk-output-invalid"
    ):
        raise RepresentationEvidenceError("R1 execution policy drifted")
    endpoint = _validate_endpoint(execution_policy.get("endpoint_url"))
    if endpoint != extractor.deployment_id:
        raise RepresentationEvidenceError("R1 endpoint differs from extractor deployment")
    if (
        execution_policy.get("maximum_application_attempts")
        != DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS
        or execution_policy.get("maximum_http_attempts_per_application_attempt")
        != DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS
    ):
        raise RepresentationEvidenceError("R1 retry maxima drifted")

    raw_attempt_records = record.get("application_attempts")
    if not isinstance(raw_attempt_records, list) or not (
        1 <= len(raw_attempt_records) <= DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS
    ):
        raise RepresentationEvidenceError("R1 application attempt ledger is invalid")
    attempt_replays: list[_AttemptReplay] = []
    for index, raw_attempt_record in enumerate(raw_attempt_records, start=1):
        attempt = replay_deepseek_r1_attempt_record(
            raw_attempt_record,
            source=source,
            extractor=extractor,
            application_attempt=index,
        )
        replayed_attempt = _application_attempt_material(
            source=source,
            extractor=extractor,
            attempt=attempt,
            application_attempt=index,
        )
        attempt_replays.append(replayed_attempt)
    accepted = [index for index, item in enumerate(attempt_replays) if item.merged_sfk is not None]
    if accepted != [len(attempt_replays) - 1]:
        raise RepresentationEvidenceError(
            "R1 application ledger must stop at its first schema-valid response"
        )
    selected_envelope = attempt_replays[-1].envelope
    selected = ReplayedDeepSeekR1Response(
        merged_sfk=attempt_replays[-1].merged_sfk or "",
        provider_request_id=selected_envelope.provider_request_id,
        response_model=selected_envelope.response_model,
        finish_reason=selected_envelope.finish_reason,
        prompt_tokens=selected_envelope.prompt_tokens,
        completion_tokens=selected_envelope.completion_tokens,
        total_tokens=selected_envelope.total_tokens,
        usage_sha256=selected_envelope.usage_sha256,
        system_fingerprint=selected_envelope.system_fingerprint,
    )
    frozen_attempt_replays = tuple(attempt_replays)
    key, receipt = _receipt_and_key(
        source=source,
        extractor=extractor,
        pricing=pricing,
        attempt_replays=frozen_attempt_replays,
        selected=selected,
        endpoint_url=endpoint,
    )
    if record.get("normalized") != _normalized_binding(
        selected=selected,
        attempt_replays=frozen_attempt_replays,
        pricing=pricing,
        key=key,
        receipt=receipt,
    ):
        raise RepresentationEvidenceError("R1 normalized evidence differs from raw replay")
    return DeepSeekR1ExtractionEvidence(
        source=source,
        extractor=extractor,
        pricing=pricing,
        derived_key=key,
        construction_receipt=receipt,
        provider_request_id=selected.provider_request_id,
        response_model=selected.response_model,
        finish_reason=selected.finish_reason,
        total_tokens=receipt.input_tokens + receipt.output_tokens,
        usage_sha256=sha256_json([item.envelope.usage_sha256 for item in frozen_attempt_replays]),
        endpoint_url=endpoint,
        selected_application_attempt=len(frozen_attempt_replays),
        application_attempts=tuple(item.attempt for item in frozen_attempt_replays),
        record_sha256=sha256_json(record),
    )


def deepseek_r1_evidence_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize records as canonical UTF-8 JSONL without rewriting raw sidecars."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise RepresentationEvidenceError("R1 evidence records must be a sequence")
    return b"".join(canonical_json_bytes(dict(record)) + b"\n" for record in records)


def replay_deepseek_r1_evidence_jsonl(
    raw: bytes,
    *,
    sources: Sequence[CanonicalValue],
    extractor: ExtractorIdentity,
    pricing: DeepSeekR1PricingIdentity,
) -> tuple[DeepSeekR1ExtractionEvidence, ...]:
    """Replay canonical JSONL and reject duplicate or unknown value routes."""

    if not isinstance(raw, bytes) or not raw or not raw.endswith(b"\n"):
        raise RepresentationEvidenceError("R1 evidence JSONL must be non-empty and LF-terminated")
    if b"\r" in raw:
        raise RepresentationEvidenceError("R1 evidence JSONL must use byte-exact LF delimiters")
    source_by_id: dict[str, CanonicalValue] = {}
    for source in sources:
        if not isinstance(source, CanonicalValue):
            raise RepresentationEvidenceError("R1 evidence sources must be CanonicalValue")
        if source.value_id in source_by_id:
            raise RepresentationEvidenceError("R1 evidence sources repeat a value ID")
        source_by_id[source.value_id] = source
    results: list[DeepSeekR1ExtractionEvidence] = []
    seen: set[str] = set()
    lines = raw.split(b"\n")
    for line_number, line in enumerate(lines[:-1], start=1):
        if not line:
            raise RepresentationEvidenceError(
                f"R1 evidence JSONL contains an empty line at {line_number}"
            )
        record = _strict_json(line, label=f"R1 evidence JSONL line {line_number}")
        if not isinstance(record, dict) or canonical_json_bytes(record) != line:
            raise RepresentationEvidenceError(
                f"R1 evidence JSONL line {line_number} is not canonical"
            )
        route = record.get("route")
        canonical_value = route.get("canonical_value") if isinstance(route, dict) else None
        value_id = canonical_value.get("value_id") if isinstance(canonical_value, dict) else None
        if not isinstance(value_id, str) or value_id not in source_by_id:
            raise RepresentationEvidenceError(
                f"R1 evidence JSONL line {line_number} has an unknown source route"
            )
        if value_id in seen:
            raise RepresentationEvidenceError("R1 evidence JSONL repeats a source route")
        seen.add(value_id)
        results.append(
            replay_deepseek_r1_evidence_record(
                record,
                source=source_by_id[value_id],
                extractor=extractor,
                pricing=pricing,
            )
        )
    return tuple(results)


def load_deepseek_r1_evidence_artifact(
    path: Path,
    *,
    sources: Sequence[CanonicalValue],
    extractor: ExtractorIdentity,
    pricing: DeepSeekR1PricingIdentity,
) -> tuple[DeepSeekR1ExtractionEvidence, ...]:
    """Read and replay one canonical R1 JSONL artifact."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RepresentationEvidenceError(
            f"cannot read R1 evidence artifact {path}: {exc}"
        ) from exc
    return replay_deepseek_r1_evidence_jsonl(
        raw,
        sources=sources,
        extractor=extractor,
        pricing=pricing,
    )


__all__ = [
    "ARTIFACT_TYPE",
    "DEEPSEEK_DEPLOYMENT_ID",
    "DEEPSEEK_MAXIMUM_APPLICATION_ATTEMPTS",
    "DEEPSEEK_MAXIMUM_HTTP_ATTEMPTS",
    "DEEPSEEK_MAX_TOKENS",
    "DEEPSEEK_MODEL_ID",
    "DeepSeekR1ExtractionEvidence",
    "DeepSeekR1PricingIdentity",
    "DeepSeekR1ProviderAttempt",
    "EXTRACTOR_PRODUCER",
    "EXTRACTOR_PROTOCOL",
    "PROMPT_IDENTITY_SHA256",
    "PROMPT_VERSION",
    "PROTOCOL_VERSION",
    "REQUEST_PARSER",
    "RESPONSE_PARSER",
    "RepresentationEvidenceError",
    "ReplayedDeepSeekR1Response",
    "SCHEMA_VERSION",
    "build_deepseek_r1_evidence_record",
    "build_deepseek_r1_attempt_record",
    "deepseek_r1_attempt_jsonl_bytes",
    "deepseek_r1_evidence_jsonl_bytes",
    "deepseek_r1_extractor_identity",
    "deepseek_r1_request_bytes",
    "load_deepseek_r1_attempt_artifact",
    "load_deepseek_r1_evidence_artifact",
    "render_deepseek_r1_prompt",
    "replay_deepseek_r1_attempt_jsonl",
    "replay_deepseek_r1_attempt_record",
    "replay_deepseek_r1_evidence_jsonl",
    "replay_deepseek_r1_evidence_record",
    "replay_deepseek_r1_response",
]
