from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from swarmbrain.domain.agents import ActorContext

TOKEN_PREFIX = "sbv0"


class InvalidTokenError(ValueError):
    """The bearer token is malformed or has an invalid signature/claims."""


class ExpiredTokenError(InvalidTokenError):
    """The bearer token is valid but no longer current."""


class RunTokenClaims(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    actor: ActorContext
    issued_at: datetime
    expires_at: datetime
    token_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    audience: str = "swarmbrain-api"


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode((value + padding).encode("ascii"))
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise InvalidTokenError("invalid base64url token component") from exc


class RunTokenCodec:
    """Small HMAC token codec for the closed stdio-MCP MVP.

    This deliberately is not OAuth. The signed payload is the sole source of
    actor identity and capabilities; HTTP or MCP request bodies cannot replace
    those claims.
    """

    def __init__(self, secret: str | bytes, *, audience: str = "swarmbrain-api") -> None:
        secret_bytes = secret.encode("utf-8") if isinstance(secret, str) else secret
        if len(secret_bytes) < 16:
            raise ValueError("token secret must contain at least 16 bytes")
        self._secret = secret_bytes
        self._audience = audience

    def issue(
        self,
        actor: ActorContext,
        *,
        ttl: timedelta = timedelta(minutes=30),
        now: datetime | None = None,
    ) -> str:
        if ttl <= timedelta(0):
            raise ValueError("token ttl must be positive")
        issued_at = self._utc(now or datetime.now(UTC))
        claims = RunTokenClaims(
            actor=actor,
            issued_at=issued_at,
            expires_at=issued_at + ttl,
            audience=self._audience,
        )
        payload = claims.model_dump_json(exclude_none=True).encode("utf-8")
        encoded_payload = _b64encode(payload)
        signing_input = f"{TOKEN_PREFIX}.{encoded_payload}".encode("ascii")
        signature = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        return f"{TOKEN_PREFIX}.{encoded_payload}.{_b64encode(signature)}"

    def verify(self, token: str, *, now: datetime | None = None) -> ActorContext:
        try:
            prefix, encoded_payload, encoded_signature = token.split(".", 2)
        except ValueError as exc:
            raise InvalidTokenError("token must contain three components") from exc
        if prefix != TOKEN_PREFIX:
            raise InvalidTokenError("unsupported token version")

        signing_input = f"{prefix}.{encoded_payload}".encode("ascii")
        expected = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        supplied = _b64decode(encoded_signature)
        if not hmac.compare_digest(expected, supplied):
            raise InvalidTokenError("invalid token signature")

        try:
            payload = json.loads(_b64decode(encoded_payload))
            claims = RunTokenClaims.model_validate(payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
            raise InvalidTokenError("invalid token claims") from exc

        if claims.audience != self._audience:
            raise InvalidTokenError("invalid token audience")
        current = self._utc(now or datetime.now(UTC))
        if self._utc(claims.expires_at) <= current:
            raise ExpiredTokenError("token has expired")
        if self._utc(claims.issued_at) > current + timedelta(seconds=30):
            raise InvalidTokenError("token issued_at is in the future")
        return claims.actor.model_copy(
            update={
                "token_id": claims.token_id,
                "authenticated_at": claims.issued_at,
                "expires_at": claims.expires_at,
            }
        )

    @staticmethod
    def token_hash(token: str) -> bytes:
        """Return the digest persisted in `agent_tokens`, never the token itself."""

        return hashlib.sha256(token.encode("utf-8")).digest()

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("token timestamps must be timezone-aware")
        return value.astimezone(UTC)
