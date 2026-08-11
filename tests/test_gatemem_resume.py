from __future__ import annotations

import copy
import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import benchmarks.integrations.gatemem.resume as resume_module
import pytest
import run_gatemem_external as cli
from benchmarks.integrations.gatemem.contracts import GATEMEM_COMMIT, GateMemContractError
from benchmarks.integrations.gatemem.policy import SpeakerOnlyAudiencePolicy
from benchmarks.integrations.gatemem.resume import (
    RESUME_ARTIFACT_TYPE,
    AuthenticatedResumeStore,
    EpisodeResumeSpec,
)
from benchmarks.integrations.gatemem.runner import HarnessRun

_KEY = b"local-test-only-gatemem-resume-key-32-bytes"


def _args(tmp_path: Path) -> Any:
    return SimpleNamespace(
        domain="office",
        gatemem_dir=tmp_path / "pinned-gatemem",
        answer_base_url="https://unused-reader.test/v1",
        answer_model="reader-v1",
        answer_revision="reader-v1-revision",
        answer_api_key_env="UNUSED_ANSWER_KEY",
        answer_timeout_seconds=120.0,
        backend="memory",
        api_url=None,
        audience_manifest=None,
        token_manifest=None,
        recall_limit=20,
        min_score=0.0,
        context_token_budget=4096,
        predictions=tmp_path / "predictions.jsonl",
        completion_manifest=None,
        resume_state=tmp_path / "office.resume.json",
        resume_key_env="GATEMEM_RESUME_HMAC_KEY",
        score_out_dir=None,
    )


def _specs() -> tuple[EpisodeResumeSpec, ...]:
    return (
        EpisodeResumeSpec("episode-1", ("checkpoint-1a", "checkpoint-1b")),
        EpisodeResumeSpec("episode-2", ("checkpoint-2a",)),
    )


def _fingerprint(tmp_path: Path, specs: tuple[EpisodeResumeSpec, ...]) -> dict[str, Any]:
    return cli._resume_fingerprint(
        args=_args(tmp_path),
        audience_policy=SpeakerOnlyAudiencePolicy(),
        specs=specs,
        audit_path=tmp_path / "audit.json",
        completion_path=tmp_path / "predictions.jsonl.completion.json",
        token_manifest_sha256=None,
        implementation_fingerprint=cli._implementation_fingerprint(),
    )


def _prediction(episode_id: str, checkpoint_id: str, *, model: str = "reader-v1") -> dict:
    return {
        "checkpoint_id": checkpoint_id,
        "output": {
            "action": "no_memory",
            "answer": "",
            "answer_structured": {},
            "used_record_ids": [],
            "memory_audit": {
                "schema_version": 1,
                "stage": "prompt_context",
                "context_format": "swarmbrain-json-v1",
                "prompt_context": {
                    "text": "[]",
                    "n_chars": 2,
                    "n_items": 0,
                    "items": [],
                },
            },
            "llm_usage": {
                "input_tokens": 11,
                "output_tokens": 2,
                "total_tokens": 13,
            },
        },
        "swarmbrain_audit": {
            "schema_version": 1,
            "gatemem_commit": GATEMEM_COMMIT,
            "episode_id": episode_id,
            "as_of_turn_id": "turn-1",
            "requester": {
                "principal_id": "principal-1",
                "role": "member",
                "scope_key_sha256": "a" * 64,
            },
            "query_sha256": "b" * 64,
            "retrieval": {
                "limit": 20,
                "min_score": 0.0,
                "total_candidates": 0,
                "returned": 0,
                "packed": 0,
                "dropped_by_token_budget": 0,
                "truncated": False,
                "provenance": [],
            },
            "tokens": {
                "context_budget": 4096,
                "context_estimated": 0,
                "request_estimated": 11,
                "provider_input": 11,
                "provider_output": 2,
                "provider_usage_reported": True,
                "usage_source": "provider",
            },
            "answer_model": {
                "provider": "openai-compatible",
                "model": model,
                "revision": "reader-v1-revision",
            },
            "latency_ms": {
                "incremental_ingest": 0.1,
                "recall": 0.2,
                "answer": 0.3,
                "query_total": 0.6,
            },
            "incremental_ingest": {"turns": 1, "operations": 1},
        },
    }


def _episode_run(spec: EpisodeResumeSpec, *, model: str = "reader-v1") -> HarnessRun:
    predictions = tuple(
        _prediction(spec.episode_id, checkpoint_id, model=model)
        for checkpoint_id in spec.checkpoint_ids
    )
    return HarnessRun(
        predictions=predictions,
        audit={
            "schema_version": 1,
            "benchmark": "GateMem",
            "gatemem_commit": GATEMEM_COMMIT,
            "adapter": "swarmbrain-gatemem-external",
            "config": {
                "recall_limit": 20,
                "min_score": 0.0,
                "context_token_budget": 4096,
            },
            "audience_policy": {
                "type": "SpeakerOnlyAudiencePolicy",
                "manifest_sha256": None,
            },
            "turn_interpreter": "DeterministicTurnInterpreter",
            "episodes": 1,
            "checkpoints": len(predictions),
            "ingest_operations": [
                {
                    "episode_id": spec.episode_id,
                    "source_turn_id": "turn-1",
                    "principal_id": "principal-1",
                    "action": "safe_noop",
                }
            ],
            "latency_ms": {"total": 1.25},
        },
    )


def _store(tmp_path: Path, *, fingerprint: dict[str, Any] | None = None, key: bytes = _KEY):
    specs = _specs()
    return AuthenticatedResumeStore(
        path=tmp_path / "office.resume.json",
        key=key,
        fingerprint=fingerprint or _fingerprint(tmp_path, specs),
        episodes=specs,
    )


def test_resume_rejects_durable_http_backend() -> None:
    with pytest.raises(GateMemContractError, match="requires --backend memory"):
        cli._validate_resume_backend(
            backend="http",
            resume_state=Path("office.resume.json"),
        )

    cli._validate_resume_backend(
        backend="memory",
        resume_state=Path("office.resume.json"),
    )
    cli._validate_resume_backend(backend="http", resume_state=None)


def test_resume_store_round_trip_preserves_official_order_and_paired_evidence(
    tmp_path: Path,
) -> None:
    specs = _specs()
    store = _store(tmp_path)
    with store.locked():
        completed = store.load_or_initialize()
        assert completed == ()
        completed = store.append_episode(completed, _episode_run(specs[0]))

    raw = json.loads(store.path.read_text(encoding="utf-8"))
    assert raw["artifact_type"] == RESUME_ARTIFACT_TYPE
    assert raw["payload"]["status"] == "partial"
    assert raw["payload"]["completed_episodes"][0]["episode_id"] == "episode-1"
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert _KEY.decode() not in store.path.read_text(encoding="utf-8")
    assert not (tmp_path / "predictions.jsonl").exists()
    assert not (tmp_path / "audit.json").exists()

    reopened = _store(tmp_path)
    with reopened.locked():
        completed = reopened.load_or_initialize()
        assert len(completed) == 1
        completed = reopened.append_episode(completed, _episode_run(specs[1]))
        result = reopened.combine_complete(completed)
        payload_digest = reopened.authenticated_payload_sha256(completed)

    assert [row["checkpoint_id"] for row in result.predictions] == [
        "checkpoint-1a",
        "checkpoint-1b",
        "checkpoint-2a",
    ]
    assert [item["episode_id"] for item in result.audit["ingest_operations"]] == [
        "episode-1",
        "episode-2",
    ]
    assert result.audit["episodes"] == 2
    assert result.audit["checkpoints"] == 3
    persisted = json.loads(reopened.path.read_text())
    assert persisted["payload"]["status"] == "complete"
    assert (
        payload_digest
        == hashlib.sha256(
            json.dumps(
                persisted["payload"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )


def test_resume_store_rejects_tampering_wrong_key_and_authenticated_reordering(
    tmp_path: Path,
) -> None:
    specs = _specs()
    store = _store(tmp_path)
    with store.locked():
        completed = store.load_or_initialize()
        completed = store.append_episode(completed, _episode_run(specs[0]))
        store.append_episode(completed, _episode_run(specs[1]))

    original = json.loads(store.path.read_text())
    tampered = copy.deepcopy(original)
    tampered["payload"]["completed_episodes"][0]["predictions"][0]["output"]["answer"] = "tampered"
    store.path.write_text(json.dumps(tampered), encoding="utf-8")
    tampered_store = _store(tmp_path)
    with (
        tampered_store.locked(),
        pytest.raises(GateMemContractError, match="authentication failed"),
    ):
        tampered_store.load_or_initialize()

    store.path.write_text(json.dumps(original), encoding="utf-8")
    wrong_key_store = _store(tmp_path, key=b"a different local key with at least 32 bytes")
    with wrong_key_store.locked(), pytest.raises(GateMemContractError, match="another key"):
        wrong_key_store.load_or_initialize()

    reordered = copy.deepcopy(original)
    reordered["payload"]["completed_episodes"].reverse()
    authenticated = {
        "artifact_type": reordered["artifact_type"],
        "schema_version": reordered["schema_version"],
        "payload": reordered["payload"],
        "authentication": {
            "algorithm": reordered["authentication"]["algorithm"],
            "key_id": reordered["authentication"]["key_id"],
        },
    }
    reordered["authentication"]["tag"] = resume_module._authentication_tag(_KEY, authenticated)
    store.path.write_text(json.dumps(reordered), encoding="utf-8")
    reordered_store = _store(tmp_path)
    with reordered_store.locked(), pytest.raises(GateMemContractError, match="official prefix"):
        reordered_store.load_or_initialize()


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("dataset", "combined_sha256"), "0" * 64),
        (("dataset", "repository_commit"), "different-commit"),
        (("dataset", "domain"), "medical"),
        (("audience_policy", "manifest_sha256"), "1" * 64),
        (("answer_model", "model"), "different-model"),
        (("protocol", "answer_prompt_sha256"), "2" * 64),
        (("protocol", "official_scorer_sha256"), "3" * 64),
        (("protocol", "answer_decoding", "temperature"), 0.5),
        (("implementation", "tree_sha256"), "4" * 64),
        (("run_parameters", "recall_limit"), 99),
    ],
)
def test_resume_store_rejects_every_provenance_or_run_parameter_drift(
    tmp_path: Path, path: tuple[str, ...], replacement: Any
) -> None:
    specs = _specs()
    fingerprint = _fingerprint(tmp_path, specs)
    store = _store(tmp_path, fingerprint=fingerprint)
    with store.locked():
        store.load_or_initialize()

    drifted = copy.deepcopy(fingerprint)
    target = drifted
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    drifted_store = _store(tmp_path, fingerprint=drifted)
    with (
        drifted_store.locked(),
        pytest.raises(GateMemContractError, match="does not exactly match"),
    ):
        drifted_store.load_or_initialize()


def test_atomic_replace_failure_leaves_previous_authenticated_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = _specs()
    store = _store(tmp_path)
    with store.locked():
        completed = store.load_or_initialize()
        completed = store.append_episode(completed, _episode_run(specs[0]))
        before = store.path.read_bytes()

        def fail_replace(source: str, destination: Path) -> None:
            del source, destination
            raise OSError("simulated crash before replace")

        monkeypatch.setattr(resume_module.os, "replace", fail_replace)
        with pytest.raises(GateMemContractError, match="atomically write"):
            store.append_episode(completed, _episode_run(specs[1]))
        assert store.path.read_bytes() == before


def test_oversized_update_leaves_previous_authenticated_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = _specs()
    store = _store(tmp_path)
    with store.locked():
        completed = store.load_or_initialize()
        completed = store.append_episode(completed, _episode_run(specs[0]))
        before = store.path.read_bytes()

        monkeypatch.setattr(resume_module, "_MAX_RESUME_BYTES", len(before))
        with pytest.raises(GateMemContractError, match="exceeds the size limit"):
            store.append_episode(completed, _episode_run(specs[1]))
        assert store.path.read_bytes() == before


def test_resume_paths_cannot_overlap_lock_or_input_artifacts(tmp_path: Path) -> None:
    state = tmp_path / "office.resume.json"
    with pytest.raises(GateMemContractError, match="must use distinct paths"):
        cli._validate_output_paths(
            predictions=tmp_path / "office.resume.json.lock",
            audit=tmp_path / "audit.json",
            completion=tmp_path / "completion.json",
            resume_state=state,
            audience_manifest=None,
            token_manifest=None,
        )
    with pytest.raises(GateMemContractError, match="audience manifest cannot overlap"):
        cli._validate_output_paths(
            predictions=tmp_path / "predictions.jsonl",
            audit=tmp_path / "audit.json",
            completion=tmp_path / "completion.json",
            resume_state=state,
            audience_manifest=state,
            token_manifest=None,
        )


def test_resume_store_rejects_a_concurrent_invocation(tmp_path: Path) -> None:
    first = _store(tmp_path)
    second = _store(tmp_path)
    with (
        first.locked(),
        pytest.raises(GateMemContractError, match="another process"),
        second.locked(),
    ):
        raise AssertionError("the second process lock must not be acquired")


def test_canonical_completion_records_explicit_uninterrupted_lineage(tmp_path: Path) -> None:
    args = _args(tmp_path)
    audit_path = tmp_path / "audit.json"
    completion_path = tmp_path / "predictions.jsonl.completion.json"
    implementation = cli._implementation_fingerprint()
    assert (
        cli._write_completed_result(
            args=args,
            checkout=object(),
            episodes=({"episode_id": "episode-1"},),
            audit_path=audit_path,
            completion_path=completion_path,
            result=_episode_run(_specs()[0]),
            resumed_episodes=None,
            authenticated_state_payload_sha256=None,
            implementation_fingerprint=implementation,
        )
        == 0
    )

    lineage = json.loads(completion_path.read_text(encoding="utf-8"))["execution_lineage"]
    assert lineage == {
        "schema_version": 1,
        "mode": "uninterrupted",
        "resume_enabled": False,
        "resume_used": False,
        "completed_prefix_episodes": 0,
        "completed_episodes": 1,
        "authenticated_state_payload_sha256": None,
        "implementation": implementation,
    }


@pytest.mark.asyncio
async def test_cli_resume_after_interruption_skips_completed_episode_and_withholds_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = _specs()
    episodes = tuple(
        {
            "episode_id": spec.episode_id,
            "turns": [{"turn_id": "turn-1"}],
        }
        for spec in specs
    )
    checkpoints = tuple(
        {
            "checkpoint_id": checkpoint_id,
            "episode_id": spec.episode_id,
            "as_of_turn_id": "turn-1",
        }
        for spec in specs
        for checkpoint_id in spec.checkpoint_ids
    )

    class FakeCheckout:
        def __init__(self, path: str) -> None:
            del path

        def verify(self, *, domain: str) -> None:
            assert domain == "office"

        def load(self, domain: str) -> Any:
            assert domain == "office"
            return SimpleNamespace(episodes=episodes, checkpoints=checkpoints)

    class FakeRuntime:
        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

    class NoEndpointAnswerModel:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def close(self) -> None:
            return None

        async def answer(self, request: Any) -> Any:
            del request
            raise AssertionError("the endpoint boundary must not be called by this test")

    calls: list[str] = []
    fail_second = True

    class FakeHarness:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def run(self, *, episodes: Any, checkpoints: Any) -> HarnessRun:
            nonlocal fail_second
            episode_id = episodes[0]["episode_id"]
            calls.append(episode_id)
            if episode_id == "episode-2" and fail_second:
                raise RuntimeError("simulated interruption")
            spec = next(item for item in specs if item.episode_id == episode_id)
            assert tuple(item["checkpoint_id"] for item in checkpoints) == spec.checkpoint_ids
            return _episode_run(spec)

    monkeypatch.setattr(cli, "GateMemCheckout", FakeCheckout)
    monkeypatch.setattr(cli, "build_in_memory_runtime", lambda secret: FakeRuntime())
    monkeypatch.setattr(cli, "OpenAICompatibleAnswerModel", NoEndpointAnswerModel)
    monkeypatch.setattr(cli, "GateMemHarness", FakeHarness)
    monkeypatch.setenv("GATEMEM_RESUME_HMAC_KEY", _KEY.decode())

    predictions = tmp_path / "canonical.jsonl"
    audit = tmp_path / "canonical.audit.json"
    completion = tmp_path / "canonical.jsonl.completion.json"
    state = tmp_path / "office.resume.json"
    args = cli._parser().parse_args(
        [
            "--domain",
            "office",
            "--predictions",
            str(predictions),
            "--audit-output",
            str(audit),
            "--answer-base-url",
            "https://unused-reader.test/v1",
            "--answer-model",
            "reader-v1",
            "--answer-revision",
            "reader-v1-revision",
            "--resume-state",
            str(state),
        ]
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        await cli._run(args)
    assert calls == ["episode-1", "episode-2"]
    assert not predictions.exists()
    assert not audit.exists()
    assert not completion.exists()
    assert json.loads(state.read_text())["payload"]["status"] == "partial"

    fail_second = False
    calls.clear()
    assert await cli._run(args) == 0
    assert calls == ["episode-2"]
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert [row["checkpoint_id"] for row in rows] == [
        "checkpoint-1a",
        "checkpoint-1b",
        "checkpoint-2a",
    ]
    assert json.loads(audit.read_text())["episodes"] == 2
    manifest = json.loads(completion.read_text())
    persisted = json.loads(state.read_text())
    expected_payload_digest = hashlib.sha256(
        json.dumps(
            persisted["payload"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["artifact_type"] == "swarmbrain-gatemem-completion"
    assert manifest["execution_lineage"] == {
        "schema_version": 1,
        "mode": "resumed",
        "resume_enabled": True,
        "resume_used": True,
        "completed_prefix_episodes": 1,
        "completed_episodes": 2,
        "authenticated_state_payload_sha256": expected_payload_digest,
        "implementation": persisted["payload"]["fingerprint"]["implementation"],
    }
    serialized_lineage = json.dumps(manifest["execution_lineage"], sort_keys=True)
    assert "authentication" not in serialized_lineage
    assert _KEY.decode() not in serialized_lineage
    assert persisted["payload"]["status"] == "complete"

    class MustNotBeConstructed:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            raise AssertionError("a complete resume must not construct an endpoint client")

    monkeypatch.setattr(cli, "OpenAICompatibleAnswerModel", MustNotBeConstructed)
    calls.clear()
    assert await cli._run(args) == 0
    assert calls == []
    replay_manifest = json.loads(completion.read_text())
    assert replay_manifest["execution_lineage"]["mode"] == "complete_replay"
    assert replay_manifest["execution_lineage"]["completed_prefix_episodes"] == 2
    assert (
        replay_manifest["execution_lineage"]["authenticated_state_payload_sha256"]
        == expected_payload_digest
    )
