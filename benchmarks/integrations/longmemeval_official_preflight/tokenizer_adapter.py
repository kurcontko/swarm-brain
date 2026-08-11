"""Fail-closed bridge from the pinned JSONL tokenizer to prompt receipts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from scripts._longmemeval_tokenizer import ExactTokenizer, TokenizerObservation

from benchmarks.integrations.longmemeval_turn_prompt import (
    TOKENIZER_PROTOCOL,
    ExactTokenCountReceipt,
    TokenizerIdentity,
)

from .contracts import (
    ExactTokenizerPin,
    LongMemEvalOfficialPreflightError,
    checked_integer,
    checked_sha256,
    sha256_bytes,
)

_EVIDENCE_FIELDS = frozenset(
    {
        "method",
        "provider",
        "exact_model_tokenizer",
        "tokenizer_model",
        "tokenizer_revision",
        "tokenizer_artifact",
        "tokenizer_executable",
        "protocol",
        "response_identity_sha256",
        "observation_accounting",
    }
)
_ACCOUNTING_FIELDS = frozenset(
    {
        "source",
        "requests",
        "responses",
        "unique_provider_request_ids",
        "text_characters",
        "text_utf8_bytes",
        "exact_response_identity_verified",
    }
)
_FILE_FIELDS = frozenset({"path", "bytes", "sha256"})
_PROVIDER_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,256}")


def _exact_mapping(value: Any, expected: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LongMemEvalOfficialPreflightError(f"{label} fields differ from the exact schema")
    return dict(value)


class PinnedPromptTokenizerAdapter:
    """Expose one fresh pinned ``ExactTokenizer`` as ``ExactPromptTokenizer``.

    The underlying boundary alone does not know the question digest expected by
    the prompt packer. This adapter binds that digest to its already verified
    text observation and refuses identity/accounting drift before and after
    every local count.
    """

    def __init__(self, boundary: ExactTokenizer, *, pin: ExactTokenizerPin) -> None:
        if not isinstance(pin, ExactTokenizerPin):
            raise LongMemEvalOfficialPreflightError("tokenizer adapter pin is invalid")
        self._boundary = boundary
        self._pin = pin
        self._calls = 0
        self._text_characters = 0
        self._text_utf8_bytes = 0
        self._provider_request_ids: set[str] = set()
        self._evidence(require_fresh=True)

    @property
    def identity(self) -> TokenizerIdentity:
        return self._pin.identity

    @property
    def evidence(self) -> dict[str, Any]:
        return self._evidence(require_fresh=False)

    def _evidence(self, *, require_fresh: bool) -> dict[str, Any]:
        try:
            value = self._boundary.evidence
        except Exception as exc:  # pragma: no cover - defensive boundary normalization
            raise LongMemEvalOfficialPreflightError(
                "could not read exact-tokenizer runtime evidence"
            ) from exc
        evidence = _exact_mapping(value, _EVIDENCE_FIELDS, label="exact-tokenizer evidence")
        expected = {
            "method": "exact_serialized_reader_prompt",
            "provider": "JsonlExactTokenizer",
            "exact_model_tokenizer": True,
            "tokenizer_model": self._pin.model,
            "tokenizer_revision": self._pin.revision,
            "protocol": TOKENIZER_PROTOCOL,
            "response_identity_sha256": self.identity.identity_sha256,
        }
        for field, wanted in expected.items():
            if type(evidence.get(field)) is not type(wanted) or evidence.get(field) != wanted:
                raise LongMemEvalOfficialPreflightError(
                    f"exact-tokenizer evidence {field} differs from its pin"
                )
        artifact = _exact_mapping(
            evidence.get("tokenizer_artifact"),
            _FILE_FIELDS,
            label="exact-tokenizer artifact",
        )
        executable = _exact_mapping(
            evidence.get("tokenizer_executable"),
            _FILE_FIELDS,
            label="exact-tokenizer executable",
        )
        if artifact.get("sha256") != self._pin.artifact_sha256:
            raise LongMemEvalOfficialPreflightError(
                "exact-tokenizer artifact SHA-256 differs from its pin"
            )
        if executable.get("sha256") != self._pin.executable_sha256:
            raise LongMemEvalOfficialPreflightError(
                "exact-tokenizer executable SHA-256 differs from its pin"
            )
        checked_integer(artifact.get("bytes"), label="tokenizer artifact bytes", minimum=1)
        checked_integer(executable.get("bytes"), label="tokenizer executable bytes", minimum=1)
        accounting = _exact_mapping(
            evidence.get("observation_accounting"),
            _ACCOUNTING_FIELDS,
            label="exact-tokenizer accounting",
        )
        if accounting.get("source") != "provider-observed":
            raise LongMemEvalOfficialPreflightError(
                "exact-tokenizer accounting must be provider-observed"
            )
        expected_accounting = {
            "requests": self._calls,
            "responses": self._calls,
            "unique_provider_request_ids": len(self._provider_request_ids),
            "text_characters": self._text_characters,
            "text_utf8_bytes": self._text_utf8_bytes,
            "exact_response_identity_verified": True,
        }
        for field, wanted in expected_accounting.items():
            if type(accounting.get(field)) is not type(wanted) or accounting.get(field) != wanted:
                state = "fresh" if require_fresh else "current"
                raise LongMemEvalOfficialPreflightError(
                    f"exact-tokenizer {state} accounting {field} is inconsistent"
                )
        return evidence

    def count_prompt(
        self,
        prompt: str,
        *,
        query_sha256: str,
    ) -> ExactTokenCountReceipt:
        if not isinstance(prompt, str) or not prompt:
            raise LongMemEvalOfficialPreflightError("tokenizer prompt must be non-empty text")
        checked_sha256(query_sha256, label="tokenizer query SHA-256")
        self._evidence(require_fresh=False)
        encoded = prompt.encode("utf-8")
        try:
            observation = self._boundary.count(prompt)
        except Exception as exc:
            raise LongMemEvalOfficialPreflightError("exact-tokenizer boundary failed") from exc
        if not isinstance(observation, TokenizerObservation):
            raise LongMemEvalOfficialPreflightError(
                "exact-tokenizer boundary returned the wrong observation type"
            )
        expected_request_id = self._calls + 1
        if (
            checked_integer(
                observation.request_id,
                label="exact-tokenizer request ID",
                minimum=1,
            )
            != expected_request_id
        ):
            raise LongMemEvalOfficialPreflightError(
                "exact-tokenizer request IDs are not fresh and contiguous"
            )
        if observation.provider_request_id in self._provider_request_ids:
            raise LongMemEvalOfficialPreflightError(
                "exact-tokenizer provider request ID was reused"
            )
        prompt_sha256 = sha256_bytes(encoded)
        if observation.response_identity_sha256 != self.identity.identity_sha256:
            raise LongMemEvalOfficialPreflightError("exact-tokenizer response identity drifted")
        if observation.text_sha256 != prompt_sha256:
            raise LongMemEvalOfficialPreflightError(
                "exact-tokenizer observation does not bind the complete prompt"
            )
        checked_integer(
            observation.token_count,
            label="exact-tokenizer token count",
            minimum=1,
        )
        if (
            not isinstance(observation.provider_request_id, str)
            or _PROVIDER_REQUEST_ID_RE.fullmatch(observation.provider_request_id) is None
        ):
            raise LongMemEvalOfficialPreflightError(
                "exact-tokenizer provider request ID is invalid"
            )
        self._calls += 1
        self._text_characters += len(prompt)
        self._text_utf8_bytes += len(encoded)
        self._provider_request_ids.add(observation.provider_request_id)
        self._evidence(require_fresh=False)
        return ExactTokenCountReceipt(
            request_id=observation.request_id,
            provider_request_id=observation.provider_request_id,
            tokenizer_identity_sha256=self.identity.identity_sha256,
            query_sha256=query_sha256,
            prompt_sha256=prompt_sha256,
            prompt_utf8_bytes=len(encoded),
            token_count=observation.token_count,
        )


__all__ = ["PinnedPromptTokenizerAdapter"]
