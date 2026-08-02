from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from urllib.parse import urlsplit


def _env(name: str, *, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if value is not None:
        value = value.strip()
    if required and not value:
        raise RuntimeError(f"required environment variable is not set: {name}")
    return value or None


def _integer_env(name: str, *, default: int) -> int:
    value = _env(name, default=str(default))
    assert value is not None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


class BackendKind(StrEnum):
    MEMORY = "memory"
    COCKROACH = "cockroach"


class EmbeddingsKind(StrEnum):
    NONE = "none"
    DETERMINISTIC = "deterministic"
    BEDROCK = "bedrock"


def _embeddings_kind(value: EmbeddingsKind | str) -> EmbeddingsKind:
    try:
        return EmbeddingsKind(value)
    except ValueError as exc:
        raise ValueError("embeddings must be one of: none, deterministic, bedrock") from exc


def _backend_kind(value: BackendKind | str) -> BackendKind:
    try:
        return BackendKind(value)
    except ValueError as exc:
        raise ValueError("backend must be one of: memory, cockroach") from exc


@dataclass(frozen=True, slots=True)
class ApiSettings:
    backend: BackendKind
    token_secret: str = field(repr=False)
    database_url: str | None = field(default=None, repr=False)
    database_pool_min_size: int = 1
    database_pool_max_size: int = 12
    host: str = "127.0.0.1"
    port: int = 8080
    embeddings: EmbeddingsKind = EmbeddingsKind.NONE
    embeddings_model: str | None = None
    embeddings_dimensions: int = 1024
    aws_region: str | None = None

    def __post_init__(self) -> None:
        backend = _backend_kind(self.backend)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "embeddings", _embeddings_kind(self.embeddings))

        if not self.token_secret:
            raise ValueError("token_secret is required")
        if self.embeddings_dimensions < 2:
            raise ValueError("embeddings dimensions must be at least 2")
        if not 0 <= self.database_pool_min_size <= self.database_pool_max_size:
            raise ValueError("database pool minimum must be between zero and the maximum")
        if self.database_pool_max_size < 1:
            raise ValueError("database pool maximum must be at least one")
        if not 1 <= self.port <= 65535:
            raise ValueError("API port must be between 1 and 65535")

        if backend is BackendKind.MEMORY:
            if self.database_url is not None:
                raise ValueError("database URL is not allowed for the memory backend")
            return

        if self.database_url is None:
            raise ValueError("database URL is required for the cockroach backend")
        parsed = urlsplit(self.database_url)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
            raise ValueError(
                "database URL must use postgres:// or postgresql:// and include a host"
            )

    @classmethod
    def from_env(cls) -> ApiSettings:
        backend_value = _env("SWARMBRAIN_BACKEND", required=True)
        secret = _env("SWARMBRAIN_TOKEN_SECRET", required=True)
        assert backend_value is not None and secret is not None
        try:
            return cls(
                backend=_backend_kind(backend_value.casefold()),
                token_secret=secret,
                database_url=_env("SWARMBRAIN_DATABASE_URL"),
                database_pool_min_size=_integer_env("SWARMBRAIN_DATABASE_POOL_MIN_SIZE", default=1),
                database_pool_max_size=_integer_env(
                    "SWARMBRAIN_DATABASE_POOL_MAX_SIZE", default=12
                ),
                host=_env("SWARMBRAIN_HOST", default="127.0.0.1") or "127.0.0.1",
                port=_integer_env("SWARMBRAIN_PORT", default=8080),
                embeddings=_embeddings_kind(
                    (_env("SWARMBRAIN_EMBEDDINGS", default="none") or "none").casefold()
                ),
                embeddings_model=_env("SWARMBRAIN_EMBEDDINGS_MODEL"),
                embeddings_dimensions=_integer_env(
                    "SWARMBRAIN_EMBEDDINGS_DIMENSIONS", default=1024
                ),
                aws_region=_env("SWARMBRAIN_AWS_REGION"),
            )
        except ValueError as exc:
            raise RuntimeError(f"invalid Swarm Brain API configuration: {exc}") from exc


@dataclass(frozen=True, slots=True)
class BridgeSettings:
    api_url: str
    agent_token: str = field(repr=False)
    expected_run_id: str | None = None
    expected_agent_id: str | None = None
    request_timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        if not self.api_url or not self.agent_token:
            raise ValueError("api_url and agent_token are required")
        if (self.expected_run_id is None) != (self.expected_agent_id is None):
            raise ValueError("expected_run_id and expected_agent_id must be configured together")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")

    @classmethod
    def from_env(cls) -> BridgeSettings:
        api_url = _env("SWARMBRAIN_API_URL", required=True)
        token = _env("SWARMBRAIN_AGENT_TOKEN", required=True)
        assert api_url is not None and token is not None
        try:
            return cls(
                api_url=api_url.rstrip("/"),
                agent_token=token,
                expected_run_id=_env("SWARMBRAIN_RUN_ID"),
                expected_agent_id=_env("SWARMBRAIN_AGENT_ID"),
                request_timeout_seconds=float(
                    _env("SWARMBRAIN_REQUEST_TIMEOUT_SECONDS", default="20") or "20"
                ),
            )
        except ValueError as exc:
            raise RuntimeError(f"invalid Swarm Brain bridge configuration: {exc}") from exc


__all__ = ["ApiSettings", "BackendKind", "BridgeSettings", "EmbeddingsKind"]
