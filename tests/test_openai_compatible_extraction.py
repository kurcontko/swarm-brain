from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
import pytest

from conftest import make_actor
from swarmbrain.adapters.extraction import CodingRuleExtractor, InMemoryWorkStore
from swarmbrain.adapters.extraction.openai_compatible import (
    PROMPT_ID,
    PROMPT_SHA256,
    OpenAICompatibleExtractionProvider,
    OpenAICompatibleExtractionUnavailable,
)
from swarmbrain.application.extraction import ExtractionService
from swarmbrain.domain.evidence import EvidenceKind, EvidenceSource
from swarmbrain.domain.extraction import ExtractionInput, IngestRawSourceCommand, SourceChunk
from swarmbrain.domain.memory import MemoryLinkKind
from swarmbrain.domain.work import ClaimWorkCommand


class _Response:
    def __init__(
        self,
        payload: object,
        *,
        status_code: int = 200,
        chunks: tuple[bytes, ...] | None = None,
        delay_seconds: float = 0.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.content = json.dumps(payload).encode("utf-8")
        self.headers = headers or {}
        self._chunks = chunks or (self.content,)
        self._delay_seconds = delay_seconds
        self.chunks_read = 0
        self.closed = False

    async def aiter_bytes(self, *, chunk_size: int | None = None) -> Any:
        del chunk_size
        for chunk in self._chunks:
            if self._delay_seconds:
                await asyncio.sleep(self._delay_seconds)
            self.chunks_read += 1
            yield chunk


class _Stream:
    def __init__(self, response: _Response) -> None:
        self.response = response

    async def __aenter__(self) -> _Response:
        return self.response

    async def __aexit__(self, *args: object) -> None:
        del args
        self.response.closed = True


class _Client:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def stream(self, method: str, url: str, **kwargs: Any) -> _Stream:
        assert method == "POST"
        self.calls.append((url, kwargs))
        return _Stream(self.response)


def _chat_response(envelope: object) -> _Response:
    return _Response(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(envelope),
                        "refusal": None,
                    }
                }
            ]
        }
    )


def _proposal(**updates: object) -> dict[str, object]:
    proposal: dict[str, object] = {
        "candidate_key": "event-1",
        "kind": "event",
        "content": "Nebula was deployed on 2026-08-08.",
        "title": "Nebula deployment",
        "tags": ["deployment"],
        "confidence": 0.94,
        "event_time": "2026-08-08T13:00:00Z",
        "valid_from": "2026-08-08T13:00:00Z",
        "valid_to": None,
        "aliases": ["Nebula", "nebula"],
        "relations": [],
        "metadata_entries": [{"namespace": "code", "key": "path", "value": "deploy/nebula.py"}],
        "quotes": [
            {
                "chunk_index": 1,
                "excerpt": "Deploy Nebula at 2026-08-08.",
                "occurrence": None,
            }
        ],
    }
    proposal.update(updates)
    return proposal


def _request() -> ExtractionInput:
    source_id = str(uuid4())
    first = "Header.\n"
    second = "Deploy Nebula at 2026-08-08.\nrepeat repeat\n"
    raw = first + second
    chunks = (
        SourceChunk(
            chunk_id=str(uuid4()),
            source_id=source_id,
            chunk_index=0,
            content=first,
            content_sha256=hashlib.sha256(first.encode()).hexdigest(),
            char_start=0,
            char_end=len(first),
        ),
        SourceChunk(
            chunk_id=str(uuid4()),
            source_id=source_id,
            chunk_index=1,
            content=second,
            content_sha256=hashlib.sha256(second.encode()).hexdigest(),
            char_start=len(first),
            char_end=len(raw),
        ),
    )
    return ExtractionInput(
        work_id=str(uuid4()),
        attempt=1,
        source=EvidenceSource(
            source_id=source_id,
            run_id=str(uuid4()),
            kind=EvidenceKind.DOCUMENT,
            uri="artifact://deployment-log",
            content_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            observed_at=datetime(2026, 8, 9, 12, 0, tzinfo=UTC),
        ),
        raw_content=raw,
        chunks=chunks,
    )


def _strict_object_schema_issues(value: object, path: str = "$") -> list[str]:
    issues: list[str] = []
    if isinstance(value, dict):
        if value.get("type") == "object" or "properties" in value:
            properties = set(value.get("properties", {}))
            required = set(value.get("required", []))
            if value.get("additionalProperties") is not False:
                issues.append(f"{path}: additionalProperties must be false")
            if properties != required:
                issues.append(f"{path}: every property must be required")
        for key, item in value.items():
            issues.extend(_strict_object_schema_issues(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(_strict_object_schema_issues(item, f"{path}[{index}]"))
    return issues


@pytest.mark.asyncio
async def test_strict_provider_resolves_verbatim_quotes_and_typed_fields() -> None:
    second = _proposal(
        candidate_key="procedure-1",
        kind="procedure",
        content="Use the Nebula deployment path.",
        relations=[
            {
                "target_candidate_key": "event-1",
                "kind": "derived_from",
                "reason": "The deployment established the working path.",
            }
        ],
    )
    client = _Client(_chat_response({"memories": [_proposal(), second]}))
    provider = OpenAICompatibleExtractionProvider(
        base_url="http://model.local:8000/v1",
        model_id="qwen-memory",
        revision="weights-7",
        api_key="test-secret-key",
        client=client,
    )
    request = _request()

    candidates = await provider.extract(request)

    assert len(candidates) == 2
    first = candidates[0]
    assert first.kind == "event"
    assert first.event_time == datetime(2026, 8, 8, 13, 0, tzinfo=UTC)
    assert first.aliases == ("Nebula",)
    assert first.metadata == {"extracted": {"code": {"path": "deploy/nebula.py"}}}
    assert len(first.spans) == 1
    span = first.spans[0]
    assert span.char_start == len(request.chunks[0].content)
    assert request.raw_content[span.char_start : span.char_end] == span.excerpt
    assert candidates[1].relations[0].kind is MemoryLinkKind.DERIVED_FROM

    assert len(client.calls) == 1
    url, call = client.calls[0]
    assert url == "http://model.local:8000/v1/chat/completions"
    assert call["headers"] == {
        "Accept-Encoding": "identity",
        "Authorization": "Bearer test-secret-key",
    }
    response_format = call["json"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert "char_start" not in json.dumps(schema)
    assert _strict_object_schema_issues(schema) == []
    proposal_schema = next(
        value
        for value in schema["$defs"].values()
        if "candidate_key" in value.get("properties", {})
    )
    assert set(proposal_schema["required"]) == set(proposal_schema["properties"])
    assert provider.descriptor.prompt_id == PROMPT_ID
    assert provider.descriptor.prompt_sha256 == PROMPT_SHA256


@pytest.mark.asyncio
async def test_quote_occurrence_selects_an_exact_absolute_span() -> None:
    proposal = _proposal(
        quotes=[{"chunk_index": 1, "excerpt": "repeat", "occurrence": 2}],
    )
    provider = OpenAICompatibleExtractionProvider(
        base_url="http://model.local",
        model_id="memory-model",
        client=_Client(_chat_response({"memories": [proposal]})),
    )
    request = _request()

    candidate = (await provider.extract(request))[0]

    expected = request.raw_content.rindex("repeat")
    assert candidate.spans[0].char_start == expected
    assert candidate.spans[0].char_end == expected + len("repeat")


@pytest.mark.asyncio
async def test_ambiguous_quote_and_invalid_candidate_fail_with_safe_errors() -> None:
    secret = "do-not-leak-this-value"
    ambiguous = _proposal(
        content=secret,
        quotes=[{"chunk_index": 1, "excerpt": "repeat", "occurrence": None}],
    )
    provider = OpenAICompatibleExtractionProvider(
        base_url="http://model.local",
        model_id="memory-model",
        api_key=secret,
        client=_Client(_chat_response({"memories": [ambiguous]})),
    )

    with pytest.raises(OpenAICompatibleExtractionUnavailable) as caught:
        await provider.extract(_request())

    assert secret not in str(caught.value)

    invalid_interval = _proposal(content=secret, valid_from=None, valid_to="2026-08-09T00:00:00Z")
    provider = OpenAICompatibleExtractionProvider(
        base_url="http://model.local",
        model_id="memory-model",
        api_key=secret,
        client=_Client(_chat_response({"memories": [invalid_interval]})),
    )
    with pytest.raises(OpenAICompatibleExtractionUnavailable) as caught:
        await provider.extract(_request())
    assert secret not in str(caught.value)


@pytest.mark.asyncio
async def test_provider_bounds_candidates_and_never_exposes_error_body() -> None:
    secret = "provider-secret-body"
    failed = _Client(_Response({"error": {"message": secret}}, status_code=503))
    provider = OpenAICompatibleExtractionProvider(
        base_url="http://model.local",
        model_id="memory-model",
        api_key=secret,
        client=failed,
    )
    with pytest.raises(OpenAICompatibleExtractionUnavailable) as caught:
        await provider.extract(_request())
    assert "503" in str(caught.value)
    assert secret not in str(caught.value)
    assert failed.response.chunks_read == 0
    assert failed.response.closed is True

    limited = OpenAICompatibleExtractionProvider(
        base_url="http://model.local",
        model_id="memory-model",
        max_candidates=1,
        client=_Client(_chat_response({"memories": [_proposal(), _proposal()]})),
    )
    with pytest.raises(OpenAICompatibleExtractionUnavailable, match="too many"):
        await limited.extract(_request())


@pytest.mark.asyncio
async def test_provider_stops_at_the_cumulative_decoded_response_byte_cap() -> None:
    secret = b"must-not-be-read-or-exposed"
    response = _Response(
        None,
        chunks=(b"a" * 3000, b"b" * 1200, secret),
    )
    provider = OpenAICompatibleExtractionProvider(
        base_url="http://model.local",
        model_id="memory-model",
        max_response_bytes=4096,
        client=_Client(response),
    )

    with pytest.raises(OpenAICompatibleExtractionUnavailable, match="byte limit") as caught:
        await provider.extract(_request())

    assert secret.decode() not in str(caught.value)
    assert response.chunks_read == 2
    assert response.closed is True


@pytest.mark.asyncio
async def test_provider_rejects_compressed_response_before_decoding() -> None:
    response = _Response(
        None,
        chunks=(b"compressed-provider-body",),
        headers={"content-encoding": "gzip"},
    )
    provider = OpenAICompatibleExtractionProvider(
        base_url="http://model.local",
        model_id="memory-model",
        client=_Client(response),
    )

    with pytest.raises(
        OpenAICompatibleExtractionUnavailable,
        match="unsupported content encoding",
    ):
        await provider.extract(_request())

    assert response.chunks_read == 0
    assert response.closed is True


@pytest.mark.asyncio
async def test_provider_outer_deadline_stops_a_drip_fed_response() -> None:
    response = _Response(
        None,
        chunks=(b"{", b"}"),
        delay_seconds=0.6,
    )
    provider = OpenAICompatibleExtractionProvider(
        base_url="http://model.local",
        model_id="memory-model",
        timeout_seconds=1.0,
        client=_Client(response),
    )

    with pytest.raises(OpenAICompatibleExtractionUnavailable, match="timed out"):
        await provider.extract(_request())

    assert response.chunks_read == 1
    assert response.closed is True


@pytest.mark.asyncio
async def test_endpoint_failure_falls_back_to_deterministic_candidates(
    scope_ids: dict[str, str],
) -> None:
    secret = "provider-must-not-leak"
    provider = OpenAICompatibleExtractionProvider(
        base_url="http://model.local",
        model_id="memory-model",
        api_key=secret,
        client=_Client(_Response({"error": secret}, status_code=503)),
    )
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor(), provider=provider)
    actor = make_actor(scope_ids)
    ingested = await service.ingest(
        actor,
        IngestRawSourceCommand(
            idempotency_key="provider-fallback-source",
            kind=EvidenceKind.SOURCE_CODE,
            content="def deploy_nebula():\n    pass\n",
            observed_at=datetime(2026, 8, 9, 12, 0, tzinfo=UTC),
            use_provider=True,
        ),
    )
    lease = (await store.claim_work(ClaimWorkCommand(worker_id="provider-fallback-worker"))).leases[
        0
    ]
    request = await store.load_extraction_input(lease)

    result = await service.extract(request, use_provider=True)

    assert result.status.value == "fallback"
    assert result.fallback_reason == "provider_OpenAICompatibleExtractionUnavailable"
    assert result.candidates == result.deterministic_candidates
    assert result.provider_candidates == ()
    assert result.candidates
    assert secret not in json.dumps(result.model_dump(mode="json"))
    assert request.source.source_id == ingested.source.source_id


def test_provider_construction_is_lazy_and_does_not_open_an_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = 0

    def unexpected_client(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal opened
        opened += 1
        raise AssertionError("construction must not open an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", unexpected_client)

    provider = OpenAICompatibleExtractionProvider(
        base_url="http://model.local",
        model_id="memory-model",
    )

    assert opened == 0
    assert provider.descriptor.model == "memory-model"
