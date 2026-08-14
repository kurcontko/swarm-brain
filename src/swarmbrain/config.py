from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from urllib.parse import urlsplit
from uuid import uuid4

from swarmbrain.domain.evidence import SourceTrust


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


def _float_env(name: str, *, default: float) -> float:
    value = _env(name, default=str(default))
    assert value is not None
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc


def _boolean_env(name: str, *, default: bool) -> bool:
    value = _env(name, default="true" if default else "false")
    assert value is not None
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean")


class BackendKind(StrEnum):
    MEMORY = "memory"
    COCKROACH = "cockroach"


class EmbeddingsKind(StrEnum):
    NONE = "none"
    DETERMINISTIC = "deterministic"
    BEDROCK = "bedrock"
    OPENAI = "openai"


class ExtractionProviderKind(StrEnum):
    NONE = "none"
    OPENAI = "openai"


def _backend_kind(value: BackendKind | str) -> BackendKind:
    try:
        return BackendKind(value)
    except ValueError as exc:
        raise ValueError("backend must be one of: memory, cockroach") from exc


def _embeddings_kind(value: EmbeddingsKind | str) -> EmbeddingsKind:
    try:
        return EmbeddingsKind(value)
    except ValueError as exc:
        raise ValueError("embeddings must be one of: none, deterministic, bedrock, openai") from exc


def _extraction_provider_kind(
    value: ExtractionProviderKind | str,
) -> ExtractionProviderKind:
    try:
        return ExtractionProviderKind(value)
    except ValueError as exc:
        raise ValueError("extraction provider must be one of: none, openai") from exc


def _source_trust(value: SourceTrust | str) -> SourceTrust:
    try:
        return SourceTrust(value)
    except ValueError as exc:
        raise ValueError("ingest trust must be one of: unknown, trusted, untrusted") from exc


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
    embeddings_base_url: str | None = None
    embeddings_api_key: str | None = field(default=None, repr=False)
    retrieval_dense_min_similarity: float = 0.0
    retrieval_dense_ann_beam_size: int = 32
    aws_region: str | None = None
    ingest_trust: SourceTrust = SourceTrust.UNKNOWN
    ingest_use_provider: bool = False
    ingest_chunk_chars: int = 65_536
    ingest_priority: int = 0
    extraction_provider: ExtractionProviderKind = ExtractionProviderKind.NONE
    extraction_model: str | None = None
    extraction_revision: str | None = None
    extraction_base_url: str | None = None
    extraction_api_key: str | None = field(default=None, repr=False)
    extraction_timeout_seconds: float = 20.0
    extraction_max_output_tokens: int = 4096
    consolidation_enabled: bool = False
    consolidation_use_provider: bool = False
    consolidation_max_memories: int = 12
    consolidation_max_actions: int = 4
    consolidation_max_input_bytes: int = 65_536
    # Fail closed: the one-click hosted demo trigger only exists when an
    # operator explicitly sets SWARMBRAIN_CONSOLE_DEMO=enabled. Any other
    # value (including unset, empty, or a typo) leaves the route absent.
    console_demo_enabled: bool = False
    # Work-lease duration for console-triggered demo runs. The 20s default
    # assumes the API and CockroachDB are co-located; an operator running the
    # demo across a slower link (or a cautious hosted deployment) can widen it
    # rather than have the run die on lease fencing mid-beat.
    console_demo_lease_seconds: int = 20
    # Same fail-closed pattern for the anonymous OpenAPI surface: /docs,
    # /redoc, and /openapi.json exist only under SWARMBRAIN_PUBLIC_DOCS=enabled
    # (the hosted /docs pages also pull assets from third-party CDNs).
    public_docs_enabled: bool = False

    def __post_init__(self) -> None:
        backend = _backend_kind(self.backend)
        object.__setattr__(self, "backend", backend)
        embeddings = _embeddings_kind(self.embeddings)
        object.__setattr__(self, "embeddings", embeddings)
        extraction_provider = _extraction_provider_kind(self.extraction_provider)
        object.__setattr__(self, "extraction_provider", extraction_provider)
        object.__setattr__(self, "ingest_trust", _source_trust(self.ingest_trust))

        if not self.token_secret:
            raise ValueError("token_secret is required")
        if self.embeddings_model is not None and not 1 <= len(self.embeddings_model) <= 255:
            raise ValueError("embeddings model must contain between 1 and 255 characters")
        if self.embeddings_dimensions < 2:
            raise ValueError("embeddings dimensions must be at least 2")
        if embeddings is EmbeddingsKind.OPENAI:
            if not self.embeddings_base_url or not self.embeddings_base_url.startswith(
                ("http://", "https://")
            ):
                raise ValueError(
                    "openai embeddings require an http(s) SWARMBRAIN_EMBEDDINGS_BASE_URL"
                )
        elif self.embeddings_base_url is not None:
            raise ValueError("embeddings base URL is only allowed for the openai provider")
        if extraction_provider is ExtractionProviderKind.OPENAI:
            if not self.extraction_model or not 1 <= len(self.extraction_model) <= 255:
                raise ValueError("openai extraction requires an extraction model")
            if not self.extraction_base_url:
                raise ValueError("openai extraction requires SWARMBRAIN_EXTRACTION_BASE_URL")
            parsed_extraction_url = urlsplit(self.extraction_base_url)
            if (
                parsed_extraction_url.scheme not in {"http", "https"}
                or not parsed_extraction_url.hostname
                or parsed_extraction_url.username is not None
                or parsed_extraction_url.password is not None
                or parsed_extraction_url.query
                or parsed_extraction_url.fragment
            ):
                raise ValueError(
                    "extraction base URL must be an http(s) URL without credentials, query, or fragment"
                )
        elif any(
            value is not None
            for value in (
                self.extraction_model,
                self.extraction_revision,
                self.extraction_base_url,
                self.extraction_api_key,
            )
        ):
            raise ValueError("extraction settings are only allowed for a configured provider")
        if self.ingest_use_provider and extraction_provider is ExtractionProviderKind.NONE:
            raise ValueError("provider-backed ingestion requires an extraction provider")
        if self.extraction_revision is not None and not 1 <= len(self.extraction_revision) <= 255:
            raise ValueError("extraction revision must contain between 1 and 255 characters")
        if not 1.0 <= self.extraction_timeout_seconds <= 55.0:
            raise ValueError("extraction timeout must be between 1 and 55 seconds")
        if not 256 <= self.extraction_max_output_tokens <= 16_384:
            raise ValueError("extraction max output tokens must be between 256 and 16384")
        if self.consolidation_use_provider and not self.consolidation_enabled:
            raise ValueError("provider consolidation requires consolidation to be enabled")
        if self.consolidation_use_provider and extraction_provider is ExtractionProviderKind.NONE:
            raise ValueError("provider consolidation requires an extraction provider profile")
        if not 2 <= self.consolidation_max_memories <= 32:
            raise ValueError("consolidation max memories must be between 2 and 32")
        if not 1 <= self.consolidation_max_actions <= 8:
            raise ValueError("consolidation max actions must be between 1 and 8")
        if not 1024 <= self.consolidation_max_input_bytes <= 1_000_000:
            raise ValueError("consolidation max input bytes must be between 1024 and 1000000")
        if not 0.0 <= self.retrieval_dense_min_similarity <= 1.0:
            raise ValueError("dense retrieval minimum similarity must be between 0 and 1")
        if not 1 <= self.retrieval_dense_ann_beam_size <= 1024:
            raise ValueError("dense retrieval ANN beam size must be between 1 and 1024")
        if (
            backend is BackendKind.COCKROACH
            and embeddings is not EmbeddingsKind.NONE
            and self.embeddings_dimensions != 1024
        ):
            raise ValueError("cockroach embeddings require exactly 1024 dimensions")
        if not 1024 <= self.ingest_chunk_chars <= 262_144:
            raise ValueError("ingest chunk chars must be between 1024 and 262144")
        if not -1000 <= self.ingest_priority <= 1000:
            raise ValueError("ingest priority must be between -1000 and 1000")
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
                embeddings_base_url=_env("SWARMBRAIN_EMBEDDINGS_BASE_URL"),
                embeddings_api_key=_env("SWARMBRAIN_EMBEDDINGS_API_KEY"),
                embeddings_dimensions=_integer_env(
                    "SWARMBRAIN_EMBEDDINGS_DIMENSIONS", default=1024
                ),
                retrieval_dense_min_similarity=_float_env(
                    "SWARMBRAIN_RETRIEVAL_DENSE_MIN_SIMILARITY",
                    default=0.0,
                ),
                retrieval_dense_ann_beam_size=_integer_env(
                    "SWARMBRAIN_RETRIEVAL_DENSE_ANN_BEAM_SIZE",
                    default=32,
                ),
                aws_region=_env("SWARMBRAIN_AWS_REGION"),
                ingest_trust=_source_trust(
                    (_env("SWARMBRAIN_INGEST_TRUST", default="unknown") or "unknown").casefold()
                ),
                ingest_use_provider=_boolean_env("SWARMBRAIN_INGEST_USE_PROVIDER", default=False),
                ingest_chunk_chars=_integer_env("SWARMBRAIN_INGEST_CHUNK_CHARS", default=65_536),
                ingest_priority=_integer_env("SWARMBRAIN_INGEST_PRIORITY", default=0),
                extraction_provider=_extraction_provider_kind(
                    (_env("SWARMBRAIN_EXTRACTION_PROVIDER", default="none") or "none").casefold()
                ),
                extraction_model=_env("SWARMBRAIN_EXTRACTION_MODEL"),
                extraction_revision=_env("SWARMBRAIN_EXTRACTION_REVISION"),
                extraction_base_url=_env("SWARMBRAIN_EXTRACTION_BASE_URL"),
                extraction_api_key=_env("SWARMBRAIN_EXTRACTION_API_KEY"),
                extraction_timeout_seconds=_float_env(
                    "SWARMBRAIN_EXTRACTION_TIMEOUT_SECONDS", default=20.0
                ),
                extraction_max_output_tokens=_integer_env(
                    "SWARMBRAIN_EXTRACTION_MAX_OUTPUT_TOKENS", default=4096
                ),
                consolidation_enabled=_boolean_env(
                    "SWARMBRAIN_CONSOLIDATION_ENABLED", default=False
                ),
                consolidation_use_provider=_boolean_env(
                    "SWARMBRAIN_CONSOLIDATION_USE_PROVIDER", default=False
                ),
                consolidation_max_memories=_integer_env(
                    "SWARMBRAIN_CONSOLIDATION_MAX_MEMORIES", default=12
                ),
                consolidation_max_actions=_integer_env(
                    "SWARMBRAIN_CONSOLIDATION_MAX_ACTIONS", default=4
                ),
                consolidation_max_input_bytes=_integer_env(
                    "SWARMBRAIN_CONSOLIDATION_MAX_INPUT_BYTES", default=65_536
                ),
                console_demo_enabled=(
                    (_env("SWARMBRAIN_CONSOLE_DEMO", default="") or "").casefold() == "enabled"
                ),
                console_demo_lease_seconds=_integer_env(
                    "SWARMBRAIN_CONSOLE_DEMO_LEASE_SECONDS", default=20
                ),
                public_docs_enabled=(
                    (_env("SWARMBRAIN_PUBLIC_DOCS", default="") or "").casefold() == "enabled"
                ),
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


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    worker_id: str = field(default_factory=lambda: f"worker-{uuid4()}")
    poll_interval_seconds: float = 0.5
    lease_seconds: int = 60
    retry_delay_seconds: int = 30
    limit: int = 1

    def __post_init__(self) -> None:
        if not 1 <= len(self.worker_id.strip()) <= 255:
            raise ValueError("worker_id must contain between 1 and 255 characters")
        if self.poll_interval_seconds <= 0:
            raise ValueError("worker poll interval must be positive")
        if not 5 <= self.lease_seconds <= 3600:
            raise ValueError("worker lease seconds must be between 5 and 3600")
        if self.retry_delay_seconds < 1:
            raise ValueError("worker retry delay must be positive")
        if self.limit != 1:
            raise ValueError("worker limit must be 1 until extraction leases have heartbeats")

    @classmethod
    def from_env(cls) -> WorkerSettings:
        try:
            return cls(
                worker_id=_env("SWARMBRAIN_WORKER_ID") or f"worker-{uuid4()}",
                poll_interval_seconds=_float_env(
                    "SWARMBRAIN_WORKER_POLL_INTERVAL_SECONDS", default=0.5
                ),
                lease_seconds=_integer_env("SWARMBRAIN_WORKER_LEASE_SECONDS", default=60),
                retry_delay_seconds=_integer_env(
                    "SWARMBRAIN_WORKER_RETRY_DELAY_SECONDS", default=30
                ),
                limit=_integer_env("SWARMBRAIN_WORKER_LIMIT", default=1),
            )
        except ValueError as exc:
            raise RuntimeError(f"invalid Swarm Brain worker configuration: {exc}") from exc


__all__ = [
    "ApiSettings",
    "BackendKind",
    "BridgeSettings",
    "EmbeddingsKind",
    "ExtractionProviderKind",
    "WorkerSettings",
]
