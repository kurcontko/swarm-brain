from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from benchmarks.integrations.mem2act.contracts import (
    Mem2ActContractError,
    ReaderRequest,
)
from benchmarks.integrations.mem2act.openai_reader import (
    PROMPT_ID,
    PROMPT_SHA256,
    SYSTEM_PROMPT,
    USER_PAYLOAD_PROTOCOL,
    OpenAICompatibleReaderConfig,
    OpenAICompatibleReaderUnavailable,
    OpenAICompatibleToolSelectionReader,
    build_reader,
    request_protocol_evidence,
)
from run_mem2act_bench import _factory

MODEL = "Qwen/Qwen2.5-72B-Instruct"


def _request() -> ReaderRequest:
    return ReaderRequest(
        condition="full_catalog",
        query="Use my remembered city to check the weather.",
        memory_contexts=("The user is visiting Warsaw.",),
        tool_catalog=(
            {
                "schema_id": "weather-schema",
                "schema": {
                    "name": "find_weather",
                    "description": "Find current weather",
                    "parameters": {
                        "type": "dict",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    "required": None,
                },
            },
            {
                "schema_id": "calendar-schema",
                "schema": {
                    "name": "read_calendar",
                    "description": "Read calendar",
                    "parameters": {
                        "type": "dict",
                        "properties": {},
                        "required": [],
                    },
                    "required": None,
                },
            },
        ),
    )


def _provider_response(
    *,
    content: str = '{"name":"find_weather","arguments":{"city":"Warsaw"}}',
    model: str = MODEL,
    prompt_tokens: int = 101,
    completion_tokens: int = 9,
    total_tokens: int = 110,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-fixture",
        "model": model,
        "system_fingerprint": "reader-checkpoint-fixture",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


async def test_reader_sends_pinned_prompt_schema_and_accounts_without_exposing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "unit-test-secret-never-record"
    monkeypatch.setenv("MEM2ACT_TEST_API_KEY", secret)
    observed: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers.get("authorization")
        observed["body"] = json.loads(request.content)
        return httpx.Response(200, json=_provider_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reader = OpenAICompatibleToolSelectionReader(
            OpenAICompatibleReaderConfig(
                base_url="https://reader.invalid/v1/",
                model=MODEL,
                revision="checkpoint-2026-08-09",
                api_key_env="MEM2ACT_TEST_API_KEY",
            ),
            client=client,
        )
        result = await reader.select_tool(_request())

    assert observed["url"] == "https://reader.invalid/v1/chat/completions"
    assert observed["authorization"] == f"Bearer {secret}"
    body = observed["body"]
    assert body["model"] == MODEL
    assert body["temperature"] == 0
    assert body["top_p"] == 1
    assert body["seed"] == 0
    assert body["n"] == 1
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    user_payload = json.loads(body["messages"][1]["content"])
    assert set(user_payload) == {
        "protocol",
        "condition",
        "query",
        "memory_contexts",
        "tool_catalog",
    }
    assert user_payload["protocol"] == USER_PAYLOAD_PROTOCOL
    assert "qa_id" not in body["messages"][1]["content"]
    assert "arm" not in user_payload
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["name"]["enum"] == [
        "find_weather",
        "read_calendar",
    ]

    assert result.model == MODEL
    assert result.prompt_tokens == 101
    assert result.completion_tokens == 9
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert result.metadata["revision"] == "checkpoint-2026-08-09"
    assert result.metadata["prompt_id"] == PROMPT_ID
    assert result.metadata["prompt_sha256"] == PROMPT_SHA256
    expected_protocol = request_protocol_evidence(_request())
    assert all(result.metadata[key] == value for key, value in expected_protocol.items())
    assert len(PROMPT_SHA256) == 64
    assert result.metadata["attempts"] == 1
    assert result.metadata["retries"] == 0
    assert result.metadata["provider_total_tokens"] == 110
    assert secret not in json.dumps(result.metadata, sort_keys=True)
    assert secret not in repr(reader.config)


async def test_reader_retries_only_bounded_transient_failures_with_capped_backoff() -> None:
    statuses = [429, 503, 200]
    sleeps: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        status = statuses.pop(0)
        if status == 200:
            return httpx.Response(200, json=_provider_response())
        return httpx.Response(status, text="provider detail must not escape")

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reader = OpenAICompatibleToolSelectionReader(
            OpenAICompatibleReaderConfig(
                base_url="https://reader.invalid",
                model=MODEL,
                api_key_env=None,
                max_retries=2,
                backoff_initial_seconds=0.25,
                backoff_max_seconds=0.3,
            ),
            client=client,
            sleep=fake_sleep,
        )
        result = await reader.select_tool(_request())

    assert not statuses
    assert sleeps == [0.25, 0.3]
    assert result.metadata["attempts"] == 3
    assert result.metadata["retries"] == 2
    assert result.metadata["retry_events"] == [
        {"attempt": 1, "kind": "http_status", "status": 429},
        {"attempt": 2, "kind": "http_status", "status": 503},
    ]
    assert "provider detail" not in json.dumps(result.metadata)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            _provider_response(
                content=(
                    '{"name":"find_weather","arguments":{"city":"Warsaw"},"explanation":"extra"}'
                )
            ),
            "strict validation",
        ),
        (
            _provider_response(content='{"name":"unknown_tool","arguments":{}}'),
            "outside the request catalog",
        ),
        (
            _provider_response(model="provider-alias"),
            "does not match",
        ),
        (
            _provider_response(total_tokens=999),
            "does not reconcile",
        ),
        (
            _provider_response(finish_reason="length"),
            "required stop reason",
        ),
    ],
)
async def test_reader_fails_closed_without_retrying_protocol_or_extra_output(
    response: dict[str, Any], message: str
) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del request
        calls += 1
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reader = OpenAICompatibleToolSelectionReader(
            OpenAICompatibleReaderConfig(
                base_url="https://reader.invalid",
                model=MODEL,
                api_key_env=None,
                max_retries=5,
            ),
            client=client,
        )
        with pytest.raises(OpenAICompatibleReaderUnavailable, match=message):
            await reader.select_tool(_request())

    assert calls == 1


async def test_nonretryable_http_error_does_not_expose_body_or_retry() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del request
        calls += 1
        return httpx.Response(400, text="sensitive provider response")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reader = OpenAICompatibleToolSelectionReader(
            OpenAICompatibleReaderConfig(
                base_url="https://reader.invalid",
                model=MODEL,
                api_key_env=None,
                max_retries=5,
            ),
            client=client,
        )
        with pytest.raises(OpenAICompatibleReaderUnavailable) as raised:
            await reader.select_tool(_request())

    assert calls == 1
    assert "HTTP 400" in str(raised.value)
    assert "sensitive provider response" not in str(raised.value)


async def test_named_api_key_is_resolved_only_when_a_request_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MEM2ACT_LATE_KEY", raising=False)
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_provider_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reader = OpenAICompatibleToolSelectionReader(
            OpenAICompatibleReaderConfig(
                base_url="https://reader.invalid",
                model=MODEL,
                api_key_env="MEM2ACT_LATE_KEY",
            ),
            client=client,
        )
        assert calls == 0
        with pytest.raises(OpenAICompatibleReaderUnavailable, match="MEM2ACT_LATE_KEY"):
            await reader.select_tool(_request())
        assert calls == 0
        monkeypatch.setenv("MEM2ACT_LATE_KEY", "late-secret")
        result = await reader.select_tool(_request())

    assert calls == 1
    assert result.model == MODEL


async def test_module_factory_reads_only_named_nonsecret_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "MEM2ACT_READER_BASE_URL",
        "MEM2ACT_READER_MODEL",
        "MEM2ACT_READER_REVISION",
        "MEM2ACT_READER_API_KEY_ENV",
        "MEM2ACT_READER_TIMEOUT_SECONDS",
        "MEM2ACT_READER_MAX_RETRIES",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(Mem2ActContractError, match="MEM2ACT_READER_BASE_URL"):
        build_reader()

    monkeypatch.setenv("MEM2ACT_READER_BASE_URL", "https://reader.invalid/v1")
    monkeypatch.setenv("MEM2ACT_READER_MODEL", MODEL)
    monkeypatch.setenv("MEM2ACT_READER_REVISION", "immutable-revision")
    monkeypatch.setenv("MEM2ACT_READER_API_KEY_ENV", "NAMED_SECRET_SLOT")
    monkeypatch.setenv("MEM2ACT_READER_TIMEOUT_SECONDS", "321")
    monkeypatch.setenv("MEM2ACT_READER_MAX_RETRIES", "4")
    monkeypatch.delenv("NAMED_SECRET_SLOT", raising=False)
    reader = build_reader()
    try:
        assert reader.model == MODEL
        assert reader.revision == "immutable-revision"
        assert reader.config.api_key_env == "NAMED_SECRET_SLOT"
        assert reader.config.timeout_seconds == 321
        assert reader.config.max_retries == 4
    finally:
        await reader.close()


def test_reader_config_rejects_credentials_in_url_and_invalid_retry_bounds() -> None:
    with pytest.raises(Mem2ActContractError, match="without credentials"):
        OpenAICompatibleReaderConfig(
            base_url="https://user:secret@reader.invalid/v1",
            model=MODEL,
        )
    with pytest.raises(Mem2ActContractError, match="max_retries"):
        OpenAICompatibleReaderConfig(
            base_url="https://reader.invalid/v1",
            model=MODEL,
            max_retries=6,
        )


def test_canonical_factory_resolves_through_module_callable_convention() -> None:
    resolved = _factory("benchmarks.integrations.mem2act.openai_reader:build_reader")
    assert resolved is build_reader
