from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from swarmbrain.adapters.reranking.local_jsonl import (
    LOCAL_RERANKER_MANIFEST_SCHEMA,
    LocalJsonlLearnedReranker,
    LocalJsonlRerankerUnavailable,
)
from swarmbrain.domain.reranking import LearnedRerankPolicy
from swarmbrain.retrieval.learned_reranking import build_learned_rerank_request

_MODEL = "Qwen/Qwen3-Reranker-8B"
_REVISION = "a" * 40


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_runner(path: Path, *, mode: str = "ok") -> str:
    source = f"""#!{sys.executable}
import hashlib
import json
import os
import sys
import time
import uuid

MODE = {mode!r}

def canonical(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))

if os.environ.get("SWARMBRAIN_TEST_SECRET"):
    raise SystemExit(77)

for line in sys.stdin:
    raw = line.rstrip("\\n")
    request = json.loads(raw)
    if MODE == "sleep":
        time.sleep(10)
    if MODE == "oversize":
        print("x" * 4096, flush=True)
        continue
    candidates = request["candidates"]
    scores = [
        {{"candidate_id": item["candidate_id"], "score": (index + 1) / len(candidates)}}
        for index, item in enumerate(candidates)
    ]
    if MODE == "reverse_ids":
        scores.reverse()
    query = request["query"]
    input_tokens = max(1, len(query.split()) + sum(len(item["document"].split()) for item in candidates))
    usage = {{
        "provider_reported": True,
        "candidate_count": len(candidates),
        "query_characters": len(query),
        "document_characters": sum(len(item["document"]) for item in candidates),
        "temporal_characters": sum(len(item["temporal_context"]) for item in candidates),
        "query_bytes": len(query.encode("utf-8")),
        "document_bytes": sum(len(item["document"].encode("utf-8")) for item in candidates),
        "temporal_bytes": sum(len(item["temporal_context"].encode("utf-8")) for item in candidates),
        "request_bytes": len(raw.encode("utf-8")),
        "input_tokens": input_tokens,
        "output_tokens": 0,
        "total_tokens": input_tokens,
        "tokenized_input_sha256": hashlib.sha256(canonical({{"request": request["request_sha256"], "tokens": input_tokens}}).encode()).hexdigest(),
    }}
    provider_request_id = (
        "11111111-1111-4111-8111-111111111111" if MODE == "reuse_id" else str(uuid.uuid4())
    )
    receipt_without_digest = {{
        "identity": request["identity"],
        "request_sha256": request["request_sha256"],
        "provider_request_id": provider_request_id,
        "usage": usage,
    }}
    response_payload = {{
        "protocol_schema": "swarmbrain.learned-rerank.response.v1",
        "scores": scores,
        "receipt": receipt_without_digest,
    }}
    receipt = dict(receipt_without_digest)
    receipt["response_sha256"] = hashlib.sha256(canonical(response_payload).encode()).hexdigest()
    response_payload["receipt"] = receipt
    print(canonical(response_payload), flush=True)
"""
    path.write_text(source, encoding="utf-8")
    path.chmod(0o700)
    return _sha256_bytes(source.encode("utf-8"))


def _write_deployment(tmp_path: Path, *, mode: str = "ok") -> dict[str, object]:
    executable = tmp_path / f"runner-{mode}.py"
    executable_sha256 = _write_runner(executable, mode=mode)
    model = tmp_path / "model.safetensors"
    tokenizer = tmp_path / "tokenizer.json"
    model.write_bytes(b"pinned-model-weights")
    tokenizer.write_bytes(b'{"pinned":"tokenizer"}')
    manifest_payload = {
        "schema": LOCAL_RERANKER_MANIFEST_SCHEMA,
        "model": _MODEL,
        "revision": _REVISION,
        "components": [
            {
                "role": "cross_encoder",
                "model": _MODEL,
                "revision": _REVISION,
                "model_artifacts": [
                    {
                        "path": model.name,
                        "sha256": _sha256_bytes(model.read_bytes()),
                        "size_bytes": model.stat().st_size,
                    }
                ],
                "tokenizer_revision": _REVISION,
                "tokenizer_artifacts": [
                    {
                        "path": tokenizer.name,
                        "sha256": _sha256_bytes(tokenizer.read_bytes()),
                        "size_bytes": tokenizer.stat().st_size,
                    }
                ],
                "weight": 1.0,
            }
        ],
    }
    manifest = tmp_path / f"manifest-{mode}.json"
    manifest_bytes = json.dumps(
        manifest_payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest.write_bytes(manifest_bytes)
    return {
        "executable": executable,
        "executable_sha256": executable_sha256,
        "manifest": manifest,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "model": model,
        "tokenizer": tokenizer,
    }


def _adapter(
    deployment: dict[str, object],
    *,
    timeout_seconds: float = 1.0,
    max_response_bytes: int = 1_048_576,
    environment: dict[str, str] | None = None,
) -> LocalJsonlLearnedReranker:
    return LocalJsonlLearnedReranker(
        executable_path=deployment["executable"],  # type: ignore[arg-type]
        executable_sha256=deployment["executable_sha256"],  # type: ignore[arg-type]
        deployment_manifest_path=deployment["manifest"],  # type: ignore[arg-type]
        deployment_manifest_sha256=deployment["manifest_sha256"],  # type: ignore[arg-type]
        required_model=_MODEL,
        required_revision=_REVISION,
        timeout_seconds=timeout_seconds,
        max_response_bytes=max_response_bytes,
        environment=environment,
    )


def _request(adapter: LocalJsonlLearnedReranker):
    return build_learned_rerank_request(
        LearnedRerankPolicy(identity=adapter.identity, window=2),
        serializer_revision="fixture-session-v1",
        query="Which session contains the answer?",
        candidates=(
            (
                "001:first",
                "Conversation session recorded 2026-08-01\nfirst",
                '{"date":"2026-08-01"}',
            ),
            (
                "002:second",
                "Conversation session recorded 2026-08-02\nsecond",
                '{"date":"2026-08-02"}',
            ),
        ),
    )


@pytest.mark.asyncio
async def test_local_jsonl_boundary_is_persistent_receipted_and_does_not_inherit_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _write_deployment(tmp_path)
    monkeypatch.setenv("SWARMBRAIN_TEST_SECRET", "must-not-cross-process-boundary")
    adapter = _adapter(deployment)
    first_request = _request(adapter)

    first = await adapter.rerank(first_request)
    second = await adapter.rerank(_request(adapter))
    await adapter.close()

    assert tuple(score.candidate_id for score in first.scores) == (
        "001:first",
        "002:second",
    )
    assert tuple(score.score for score in first.scores) == (0.5, 1.0)
    assert first.receipt.identity == adapter.identity
    assert first.receipt.request_sha256 == first_request.request_sha256
    assert first.receipt.provider_request_id != first_request.request_id
    assert first.receipt.provider_request_id != second.receipt.provider_request_id
    assert first.receipt.usage.candidate_count == 2
    assert adapter.call_accounting == {
        "attempts": 2,
        "successful_requests": 2,
        "failed_requests": 0,
        "scored_candidates": 4,
        "input_tokens": first.receipt.usage.input_tokens + second.receipt.usage.input_tokens,
        "output_tokens": 0,
        "retained_provider_request_ids": 2,
    }


@pytest.mark.asyncio
async def test_local_adapter_rejects_artifact_drift_before_process_invocation(
    tmp_path: Path,
) -> None:
    deployment = _write_deployment(tmp_path)
    adapter = _adapter(deployment)
    model = deployment["model"]
    assert isinstance(model, Path)
    model.write_bytes(b"tampered-model-weights-with-different-size")

    with pytest.raises(LocalJsonlRerankerUnavailable, match="artifact size changed"):
        await adapter.rerank(_request(adapter))

    assert adapter.call_accounting["successful_requests"] == 0
    assert adapter.call_accounting["failed_requests"] == 1
    await adapter.close()


@pytest.mark.asyncio
async def test_local_adapter_rejects_a_nested_symlink_swap_after_verification(
    tmp_path: Path,
) -> None:
    deployment = _write_deployment(tmp_path)
    model = deployment["model"]
    manifest = deployment["manifest"]
    assert isinstance(model, Path) and isinstance(manifest, Path)
    nested = tmp_path / "nested"
    nested.mkdir()
    nested_model = nested / model.name
    model.rename(nested_model)
    deployment["model"] = nested_model
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["components"][0]["model_artifacts"][0]["path"] = f"nested/{nested_model.name}"
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest.write_bytes(raw)
    deployment["manifest_sha256"] = _sha256_bytes(raw)
    adapter = _adapter(deployment)

    moved = tmp_path / "nested-real"
    nested.rename(moved)
    try:
        nested.symlink_to(moved, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platforms without symlink support
        await adapter.close()
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(LocalJsonlRerankerUnavailable, match="route contains a symlink"):
        await adapter.rerank(_request(adapter))

    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "message"),
    (("reverse_ids", "input IDs"), ("reuse_id", "reused")),
)
async def test_local_adapter_rejects_candidate_authority_and_reused_provider_receipts(
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    deployment = _write_deployment(tmp_path, mode=mode)
    adapter = _adapter(deployment)
    if mode == "reuse_id":
        await adapter.rerank(_request(adapter))

    with pytest.raises(LocalJsonlRerankerUnavailable, match=message):
        await adapter.rerank(_request(adapter))

    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "kwargs", "message"),
    (
        ("sleep", {"timeout_seconds": 0.05}, "timed out"),
        ("oversize", {"max_response_bytes": 1_024}, "byte bound"),
    ),
)
async def test_local_adapter_bounds_time_and_response_bytes(
    tmp_path: Path,
    mode: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    deployment = _write_deployment(tmp_path, mode=mode)
    adapter = _adapter(deployment, **kwargs)  # type: ignore[arg-type]

    with pytest.raises(LocalJsonlRerankerUnavailable, match=message):
        await adapter.rerank(_request(adapter))

    assert adapter.call_accounting["failed_requests"] == 1
    await adapter.close()


def test_local_adapter_rejects_unpinned_environment_and_unsafe_artifact_paths(
    tmp_path: Path,
) -> None:
    deployment = _write_deployment(tmp_path)
    with pytest.raises(ValueError, match="not allowlisted"):
        _adapter(deployment, environment={"UNSAFE_API_KEY": "secret"})

    manifest = deployment["manifest"]
    assert isinstance(manifest, Path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["components"][0]["model_artifacts"][0]["path"] = "../outside.bin"
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest.write_bytes(raw)
    deployment["manifest_sha256"] = _sha256_bytes(raw)

    with pytest.raises(ValueError, match="unsafe relative path"):
        _adapter(deployment)
