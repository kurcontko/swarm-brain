from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_longmemeval_qa as qa
import run_retrieval_eval as retrieval_eval
from _longmemeval_common import (
    SessionMemory,
    build_official_reader_prompt,
    render_reader_history,
    render_reader_session,
)
from _longmemeval_tokenizer import (
    TOKENIZER_PROTOCOL,
    ExactTokenizerError,
    JsonlExactTokenizer,
    TokenizerObservation,
    tokenizer_response_identity_sha256,
)
from build_longmemeval_retrieval_report import (
    RetrievalReportError,
    _validate_exact_context_evidence,
)


class RecordingTokenizer:
    def __init__(self, *, response_identity_sha256: str = "0" * 64) -> None:
        self.requests: list[str] = []
        self.response_identity_sha256 = response_identity_sha256

    def count(self, text: str) -> TokenizerObservation:
        self.requests.append(text)
        request_id = len(self.requests)
        return TokenizerObservation(
            request_id=request_id,
            provider_request_id=f"fixture-{request_id}",
            response_identity_sha256=self.response_identity_sha256,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            token_count=len(text),
        )


def _write_tokenizer_fixture(root: Path) -> tuple[Path, str, Path, str]:
    artifact = root / "evidence" / "tokenizer.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"fixture":"tokenizer"}\n', encoding="utf-8")
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    executable = root / "evidence" / "tokenizer-provider"
    executable.write_text(
        """#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--artifact", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--revision", required=True)
parser.add_argument("--protocol", required=True)
args = parser.parse_args()
if "OPENAI_API_KEY" in os.environ:
    raise SystemExit(91)
artifact_sha = hashlib.sha256(open(args.artifact, "rb").read()).hexdigest()
for line in sys.stdin:
    request = json.loads(line)
    text = request["text"]
    response = {
        "protocol": args.protocol,
        "request_id": request["request_id"],
        "provider_request_id": f"provider-{request['request_id']}",
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_count": max(1, len(text.split())),
        "tokenizer_model": args.model,
        "tokenizer_revision": args.revision,
        "tokenizer_artifact_sha256": artifact_sha,
    }
    print(json.dumps(response, separators=(",", ":")), flush=True)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    executable_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
    return (
        Path("evidence/tokenizer-provider"),
        executable_sha,
        Path("evidence/tokenizer.json"),
        artifact_sha,
    )


def _session(position: int, date: str, content: str) -> SessionMemory:
    return SessionMemory(
        position=position,
        session_id=f"session-{position}",
        date=date,
        turns=({"role": "user", "content": content},),
        memory_id=f"memory-{position}",
    )


def test_jsonl_boundary_pins_local_files_and_observes_response_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, executable_sha, artifact, artifact_sha = _write_tokenizer_fixture(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-tokenizer")

    with JsonlExactTokenizer(
        executable=executable,
        executable_sha256=executable_sha,
        artifact=artifact,
        artifact_sha256=artifact_sha,
        model="Qwen/Qwen3.5-9B",
        revision="fixture-revision",
        repo_root=tmp_path,
    ) as tokenizer:
        observation = tokenizer.count("hello exact tokenizer")
        evidence = tokenizer.evidence

    assert observation.token_count == 3
    assert observation.provider_request_id == "provider-1"
    assert observation.response_identity_sha256 == evidence["response_identity_sha256"]
    assert evidence["protocol"] == TOKENIZER_PROTOCOL
    assert evidence["tokenizer_artifact"] == {
        "path": "evidence/tokenizer.json",
        "bytes": (tmp_path / artifact).stat().st_size,
        "sha256": artifact_sha,
    }
    assert evidence["tokenizer_executable"]["sha256"] == executable_sha
    assert evidence["observation_accounting"] == {
        "source": "provider-observed",
        "requests": 1,
        "responses": 1,
        "unique_provider_request_ids": 1,
        "text_characters": 21,
        "text_utf8_bytes": 21,
        "exact_response_identity_verified": True,
    }


def test_jsonl_boundary_rejects_unpinned_or_symlinked_files(tmp_path: Path) -> None:
    executable, executable_sha, artifact, artifact_sha = _write_tokenizer_fixture(tmp_path)
    with pytest.raises(ExactTokenizerError, match="operator-pinned SHA-256"):
        JsonlExactTokenizer(
            executable=executable,
            executable_sha256=executable_sha,
            artifact=artifact,
            artifact_sha256="0" * 64,
            model="model",
            revision="revision",
            repo_root=tmp_path,
        )
    link = tmp_path / "evidence" / "tokenizer-link.json"
    link.symlink_to(tmp_path / artifact)
    with pytest.raises(ExactTokenizerError, match="symbolic links"):
        JsonlExactTokenizer(
            executable=executable,
            executable_sha256=executable_sha,
            artifact=Path("evidence/tokenizer-link.json"),
            artifact_sha256=artifact_sha,
            model="model",
            revision="revision",
            repo_root=tmp_path,
        )


def test_exact_packing_counts_complete_official_prompts_and_traces_greedy_decisions() -> None:
    later = _session(0, "2025/01/02 (Thu) 10:00", "later memory")
    earlier = _session(1, "2025/01/01 (Wed) 10:00", "earlier memory")
    record = {
        "question_id": "fixture",
        "question_date": "2025/01/03 (Fri) 10:00",
        "question": "What happened?",
    }
    retrieved = SimpleNamespace(
        record=record,
        retrieved_sessions=lambda: ((later, 1.0), (earlier, 0.9)),
    )
    one_prompt = build_official_reader_prompt(record, [(later.date, later.turns)])
    tokenizer = RecordingTokenizer()

    evidence = retrieval_eval.exact_context_packing_evidence(
        retrieved,
        tokenizer,
        k_values=(2,),
        budgets=(None, len(one_prompt)),
    )

    bounded = evidence["k=2"][f"budget={len(one_prompt)}"]
    assert bounded["kept_ids"] == [later.key]
    assert [decision["accepted"] for decision in bounded["decisions"]] == [True, False]
    assert (
        bounded["final_observation"]["text_sha256"]
        == hashlib.sha256(one_prompt.encode("utf-8")).hexdigest()
    )
    unbounded = evidence["k=2"]["budget=none"]
    expected_full = build_official_reader_prompt(
        record,
        [(later.date, later.turns), (earlier.date, earlier.turns)],
    )
    assert (
        unbounded["final_observation"]["text_sha256"]
        == hashlib.sha256(expected_full.encode("utf-8")).hexdigest()
    )
    assert len(tokenizer.requests) == 3


def test_qa_and_retrieval_share_one_reader_serializer() -> None:
    sessions = [
        ("2025/01/02 (Thu) 10:00", ({"role": "assistant", "content": " second "},)),
        ("2025/01/01 (Wed) 10:00", ({"role": "user", "content": " first "},)),
    ]
    assert qa.render_session(0, *sessions[0]) == render_reader_session(0, *sessions[0])
    assert qa.render_history(sessions) == render_reader_history(sessions)


@pytest.mark.asyncio
async def test_exact_run_preserves_minimal_material_for_offline_prompt_replay(
    tmp_path: Path,
) -> None:
    executable, executable_sha, artifact, artifact_sha = _write_tokenizer_fixture(tmp_path)
    dataset = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "fixtures"
        / "retrieval_eval_corpus"
        / "longmemeval_sample.json"
    )
    with JsonlExactTokenizer(
        executable=executable,
        executable_sha256=executable_sha,
        artifact=artifact,
        artifact_sha256=artifact_sha,
        model="Qwen/Qwen3.5-9B",
        revision="fixture-revision",
        repo_root=tmp_path,
    ) as tokenizer:
        payload = await retrieval_eval.run_longmemeval(
            dataset,
            sample=1,
            seed=1,
            limit=10,
            use_dense=False,
            context_tokenizer=tokenizer,
        )

    case = payload["cases"][0]
    material = case["exact_context_material"]
    assert [item["session_id"] for item in material["ranked_sessions"]] == case["rankings"][
        "final"
    ][:10]
    assert set(material) == {"question", "question_date", "ranked_sessions"}
    _validate_exact_context_evidence(
        payload,
        payload["context_token_accounting"],
        repo_root=tmp_path,
    )


@pytest.mark.asyncio
async def test_full_run_rejects_an_injected_estimator_as_publishable(tmp_path: Path) -> None:
    with pytest.raises(ExactTokenizerError, match="pinned JsonlExactTokenizer"):
        await retrieval_eval.run_longmemeval(
            tmp_path / "not-read.json",
            sample=None,
            seed=1,
            limit=10,
            context_tokenizer=RecordingTokenizer(),
        )


def test_compiler_verifies_exact_paths_accounting_and_full_prompt_traces(tmp_path: Path) -> None:
    executable, executable_sha, artifact, artifact_sha = _write_tokenizer_fixture(tmp_path)
    first = _session(0, "2025/01/01 (Wed) 10:00", "memory")
    record = {
        "question_id": "fixture",
        "question_date": "2025/01/03 (Fri) 10:00",
        "question": "What happened?",
    }
    retrieved = SimpleNamespace(
        record=record,
        retrieved_sessions=lambda: ((first, 1.0),),
    )
    response_identity = tokenizer_response_identity_sha256(
        model="Qwen/Qwen3.5-9B",
        revision="fixture-revision",
        artifact_sha256=artifact_sha,
    )
    tokenizer = RecordingTokenizer(response_identity_sha256=response_identity)
    packing = retrieval_eval.exact_context_packing_evidence(retrieved, tokenizer)
    prompt_material = retrieval_eval.exact_context_prompt_material(retrieved)
    text_characters = sum(len(text) for text in tokenizer.requests)
    text_utf8_bytes = sum(len(text.encode("utf-8")) for text in tokenizer.requests)
    metadata = {
        "method": "exact_serialized_reader_prompt",
        "provider": "JsonlExactTokenizer",
        "mode": "publishable-exact",
        "counted_surface": "complete_official_reader_prompt",
        "packing_observation_source": "provider_observed_full_prompt_decisions",
        "exact_model_tokenizer": True,
        "tokenizer_model": "Qwen/Qwen3.5-9B",
        "tokenizer_revision": "fixture-revision",
        "tokenizer_artifact": {
            "path": artifact.as_posix(),
            "bytes": (tmp_path / artifact).stat().st_size,
            "sha256": artifact_sha,
        },
        "tokenizer_executable": {
            "path": executable.as_posix(),
            "bytes": (tmp_path / executable).stat().st_size,
            "sha256": executable_sha,
        },
        "protocol": TOKENIZER_PROTOCOL,
        "response_identity_sha256": response_identity,
        "serializer": retrieval_eval.exact_context_serializer_metadata(),
        "observation_accounting": {
            "source": "provider-observed",
            "requests": len(tokenizer.requests),
            "responses": len(tokenizer.requests),
            "unique_provider_request_ids": len(tokenizer.requests),
            "text_characters": text_characters,
            "text_utf8_bytes": text_utf8_bytes,
            "exact_response_identity_verified": True,
        },
    }
    payload = {
        "cases": [
            {
                "rankings": {"final": [first.key]},
                "exact_context_packing": packing,
                "exact_context_material": prompt_material,
            }
        ]
    }

    _validate_exact_context_evidence(payload, metadata, repo_root=tmp_path)

    partial = json.loads(json.dumps(metadata))
    partial["counted_surface"] = "individual_session_fragment"
    with pytest.raises(RetrievalReportError, match="counted_surface"):
        _validate_exact_context_evidence(payload, partial, repo_root=tmp_path)
    os.chmod(tmp_path / executable, 0o644)
    with pytest.raises(RetrievalReportError, match="not executable"):
        _validate_exact_context_evidence(payload, metadata, repo_root=tmp_path)


def test_compiler_rejects_a_forged_prompt_digest_and_accounting(
    tmp_path: Path,
) -> None:
    executable, executable_sha, artifact, artifact_sha = _write_tokenizer_fixture(tmp_path)
    first = _session(0, "2025/01/01 (Wed) 10:00", "mémoire")
    record = {
        "question_id": "fixture",
        "question_date": "2025/01/03 (Fri) 10:00",
        "question": "What happened?",
    }
    retrieved = SimpleNamespace(
        record=record,
        retrieved_sessions=lambda: ((first, 1.0),),
    )
    response_identity = tokenizer_response_identity_sha256(
        model="Qwen/Qwen3.5-9B",
        revision="fixture-revision",
        artifact_sha256=artifact_sha,
    )
    tokenizer = RecordingTokenizer(response_identity_sha256=response_identity)
    packing = retrieval_eval.exact_context_packing_evidence(retrieved, tokenizer)
    payload = {
        "cases": [
            {
                "rankings": {"final": [first.key]},
                "exact_context_packing": packing,
                "exact_context_material": retrieval_eval.exact_context_prompt_material(retrieved),
            }
        ]
    }
    metadata = {
        "method": "exact_serialized_reader_prompt",
        "provider": "JsonlExactTokenizer",
        "mode": "publishable-exact",
        "counted_surface": "complete_official_reader_prompt",
        "packing_observation_source": "provider_observed_full_prompt_decisions",
        "exact_model_tokenizer": True,
        "tokenizer_model": "Qwen/Qwen3.5-9B",
        "tokenizer_revision": "fixture-revision",
        "tokenizer_artifact": {
            "path": artifact.as_posix(),
            "bytes": (tmp_path / artifact).stat().st_size,
            "sha256": artifact_sha,
        },
        "tokenizer_executable": {
            "path": executable.as_posix(),
            "bytes": (tmp_path / executable).stat().st_size,
            "sha256": executable_sha,
        },
        "protocol": TOKENIZER_PROTOCOL,
        "response_identity_sha256": response_identity,
        "serializer": retrieval_eval.exact_context_serializer_metadata(),
        "observation_accounting": {
            "source": "provider-observed",
            "requests": len(tokenizer.requests),
            "responses": len(tokenizer.requests),
            "unique_provider_request_ids": len(tokenizer.requests),
            "text_characters": sum(len(text) for text in tokenizer.requests),
            "text_utf8_bytes": sum(len(text.encode("utf-8")) for text in tokenizer.requests),
            "exact_response_identity_verified": True,
        },
    }

    forged_payload = json.loads(json.dumps(payload))
    forged_payload["cases"][0]["exact_context_packing"]["k=5"]["budget=none"]["final_observation"][
        "text_sha256"
    ] = "f" * 64
    with pytest.raises(RetrievalReportError, match="reconstructed official reader prompt"):
        _validate_exact_context_evidence(forged_payload, metadata, repo_root=tmp_path)

    for field in ("text_characters", "text_utf8_bytes"):
        forged_metadata = json.loads(json.dumps(metadata))
        forged_metadata["observation_accounting"][field] += 1
        with pytest.raises(RetrievalReportError, match=f"accounting {field}"):
            _validate_exact_context_evidence(payload, forged_metadata, repo_root=tmp_path)


def test_compiler_rejects_inconsistent_observation_reuse(tmp_path: Path) -> None:
    executable, executable_sha, artifact, artifact_sha = _write_tokenizer_fixture(tmp_path)
    first = _session(0, "2025/01/01 (Wed) 10:00", "memory")
    record = {
        "question_id": "fixture",
        "question_date": "2025/01/03 (Fri) 10:00",
        "question": "What happened?",
    }
    retrieved = SimpleNamespace(record=record, retrieved_sessions=lambda: ((first, 1.0),))
    response_identity = tokenizer_response_identity_sha256(
        model="Qwen/Qwen3.5-9B",
        revision="fixture-revision",
        artifact_sha256=artifact_sha,
    )
    tokenizer = RecordingTokenizer(response_identity_sha256=response_identity)
    packing = retrieval_eval.exact_context_packing_evidence(retrieved, tokenizer)
    payload = {
        "cases": [
            {
                "rankings": {"final": [first.key]},
                "exact_context_packing": packing,
                "exact_context_material": retrieval_eval.exact_context_prompt_material(retrieved),
            }
        ]
    }
    metadata = {
        "method": "exact_serialized_reader_prompt",
        "provider": "JsonlExactTokenizer",
        "mode": "publishable-exact",
        "counted_surface": "complete_official_reader_prompt",
        "packing_observation_source": "provider_observed_full_prompt_decisions",
        "exact_model_tokenizer": True,
        "tokenizer_model": "Qwen/Qwen3.5-9B",
        "tokenizer_revision": "fixture-revision",
        "tokenizer_artifact": {
            "path": artifact.as_posix(),
            "bytes": (tmp_path / artifact).stat().st_size,
            "sha256": artifact_sha,
        },
        "tokenizer_executable": {
            "path": executable.as_posix(),
            "bytes": (tmp_path / executable).stat().st_size,
            "sha256": executable_sha,
        },
        "protocol": TOKENIZER_PROTOCOL,
        "response_identity_sha256": response_identity,
        "serializer": retrieval_eval.exact_context_serializer_metadata(),
        "observation_accounting": {
            "source": "provider-observed",
            "requests": len(tokenizer.requests),
            "responses": len(tokenizer.requests),
            "unique_provider_request_ids": len(tokenizer.requests),
            "text_characters": sum(len(text) for text in tokenizer.requests),
            "text_utf8_bytes": sum(len(text.encode("utf-8")) for text in tokenizer.requests),
            "exact_response_identity_verified": True,
        },
    }
    inconsistent = json.loads(json.dumps(payload))
    observation = inconsistent["cases"][0]["exact_context_packing"]["k=5"]["budget=16000"][
        "initial_observation"
    ]
    observation["request_id"] = len(tokenizer.requests) + 1
    observation["provider_request_id"] = "fixture-inconsistent"

    with pytest.raises(RetrievalReportError, match="inconsistent observations"):
        _validate_exact_context_evidence(inconsistent, metadata, repo_root=tmp_path)
