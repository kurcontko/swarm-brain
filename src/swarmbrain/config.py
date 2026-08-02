from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, *, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if value is not None:
        value = value.strip()
    if required and not value:
        raise RuntimeError(f"required environment variable is not set: {name}")
    return value or None


@dataclass(frozen=True, slots=True)
class ApiSettings:
    token_secret: str
    database_url: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080

    @classmethod
    def from_env(cls) -> ApiSettings:
        secret = _env("SWARMBRAIN_TOKEN_SECRET", required=True)
        assert secret is not None
        return cls(
            token_secret=secret,
            database_url=_env("SWARMBRAIN_DATABASE_URL"),
            host=_env("SWARMBRAIN_HOST", default="127.0.0.1") or "127.0.0.1",
            port=int(_env("SWARMBRAIN_PORT", default="8080") or "8080"),
        )


@dataclass(frozen=True, slots=True)
class BridgeSettings:
    api_url: str
    agent_token: str
    expected_run_id: str | None = None
    expected_agent_id: str | None = None
    request_timeout_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> BridgeSettings:
        api_url = _env("SWARMBRAIN_API_URL", required=True)
        token = _env("SWARMBRAIN_AGENT_TOKEN", required=True)
        assert api_url is not None and token is not None
        return cls(
            api_url=api_url.rstrip("/"),
            agent_token=token,
            expected_run_id=_env("SWARMBRAIN_RUN_ID"),
            expected_agent_id=_env("SWARMBRAIN_AGENT_ID"),
            request_timeout_seconds=float(
                _env("SWARMBRAIN_REQUEST_TIMEOUT_SECONDS", default="20") or "20"
            ),
        )
