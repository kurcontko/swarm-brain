"""Integrity gate for the checked-in retrieval evaluation corpus and its loader.

This is a fast structural regression, not a benchmark.  The measured numbers
live in ``docs/retrieval-benchmark.md``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "retrieval_eval_corpus"
SCRIPT = REPO_ROOT / "scripts" / "run_retrieval_eval.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("swarmbrain_eval_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


@pytest.fixture(scope="module")
def raw() -> tuple[dict[str, object], dict[str, object]]:
    corpus = json.loads((CORPUS_DIR / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((CORPUS_DIR / "queries.json").read_text(encoding="utf-8"))
    return corpus, queries


def test_corpus_is_versioned_and_ids_are_unique(raw: tuple[dict, dict]) -> None:
    corpus, queries = raw
    assert corpus["corpus_version"] == queries["corpus_version"]
    assert corpus["judgments_revision"] == queries["judgments_revision"]

    keys = [str(item["key"]) for item in corpus["memories"]]
    assert len(keys) == len(set(keys))
    assert 60 <= len(keys) <= 120

    case_ids = [str(item["case_id"]) for item in queries["queries"]]
    assert len(case_ids) == len(set(case_ids))
    assert 25 <= len(case_ids) <= 40


def test_judgments_reference_existing_memories_and_cover_no_answer(raw: tuple[dict, dict]) -> None:
    corpus, queries = raw
    keys = {str(item["key"]) for item in corpus["memories"]}
    no_answer = 0
    categories: Counter[str] = Counter()
    for case in queries["queries"]:
        relevant = [str(value) for value in case.get("relevant", ())]
        assert set(relevant) <= keys, case["case_id"]
        assert len(relevant) == len(set(relevant)), case["case_id"]
        categories[str(case["category"])] += 1
        if not relevant:
            no_answer += 1
            assert str(case["category"]) == "no_answer", case["case_id"]
    assert no_answer >= 5
    for required in ("identifier_exact", "fuzzy_typo", "paraphrase", "decoy_heavy", "no_answer"):
        assert categories[required] >= 1, required


def test_memory_links_only_point_backwards(raw: tuple[dict, dict]) -> None:
    corpus, _ = raw
    published: set[str] = set()
    for item in corpus["memories"]:
        key = str(item["key"])
        for target in item.get("related", ()):
            assert str(target) in published, f"{key} links forward to {target}"
            assert str(target) != key
        published.add(key)


def test_loader_round_trips_the_checked_in_corpus(raw: tuple[dict, dict]) -> None:
    corpus, queries = raw
    loaded = runner.load_corpus(CORPUS_DIR)

    assert loaded.corpus_version == corpus["corpus_version"]
    assert loaded.judgments_revision == corpus["judgments_revision"]
    assert list(loaded.keys) == [str(item["key"]) for item in corpus["memories"]]
    assert [case.case_id for case in loaded.queries] == [
        str(item["case_id"]) for item in queries["queries"]
    ]
    first = loaded.memories[0]
    assert first.title == str(corpus["memories"][0]["title"])
    assert first.content == str(corpus["memories"][0]["content"])
    assert first.tags == tuple(str(tag) for tag in corpus["memories"][0]["tags"])
    assert any(case.relevant == () for case in loaded.queries)
    assert all(target in set(loaded.keys) for case in loaded.queries for target in case.relevant)


def test_loader_rejects_a_judgment_for_an_unknown_memory(tmp_path: Path) -> None:
    corpus = json.loads((CORPUS_DIR / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((CORPUS_DIR / "queries.json").read_text(encoding="utf-8"))
    queries["queries"][0]["relevant"] = ["memory-that-does-not-exist"]
    (tmp_path / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    (tmp_path / "queries.json").write_text(json.dumps(queries), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown memories"):
        runner.load_corpus(tmp_path)


def test_loader_rejects_a_judgments_revision_mismatch(tmp_path: Path) -> None:
    corpus = json.loads((CORPUS_DIR / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((CORPUS_DIR / "queries.json").read_text(encoding="utf-8"))
    queries["judgments_revision"] = "r0"
    (tmp_path / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    (tmp_path / "queries.json").write_text(json.dumps(queries), encoding="utf-8")

    with pytest.raises(ValueError, match="judgments revision"):
        runner.load_corpus(tmp_path)


def test_longmemeval_sample_fixture_maps_sessions_to_relevance() -> None:
    """The tiny checked-in sample proves the mapper without the 265 MiB release."""

    records = json.loads(
        (CORPUS_DIR / "longmemeval_sample.json").read_text(encoding="utf-8"),
    )
    assert records
    for record in records:
        session_ids = [str(value) for value in record["haystack_session_ids"]]
        assert len(session_ids) == len(record["haystack_sessions"])
        assert len(session_ids) == len(record["haystack_dates"])
        answers = {str(value) for value in record["answer_session_ids"]}
        assert answers <= set(session_ids)
        relevant = [
            f"{index:03d}:{session_id}"
            for index, session_id in enumerate(session_ids)
            if session_id in answers
        ]
        assert relevant
        for turns in record["haystack_sessions"]:
            rendered = runner._session_text(turns)
            assert rendered.strip()
            assert rendered.startswith(("user:", "assistant:"))


def test_longmemeval_download_helper_verifies_the_pinned_digest(tmp_path: Path) -> None:
    corrupt = tmp_path / "longmemeval_s.json"
    corrupt.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned LongMemEval-S digest"):
        runner.ensure_longmemeval(corrupt, download=False)

    missing = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError):
        runner.ensure_longmemeval(missing, download=False)


def test_artifact_path_allows_external_output_directories(tmp_path: Path) -> None:
    assert runner._artifact_path(REPO_ROOT / "benchmarks" / "result.json") == (
        "benchmarks/result.json"
    )
    assert runner._artifact_path(tmp_path / "result.json") == str(
        (tmp_path / "result.json").resolve()
    )


def test_report_binds_execution_protocol_and_exact_saved_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_path = tmp_path / "run.json"
    payload = {
        "artifact_type": runner.RUN_ARTIFACT_TYPE,
        "schema_version": runner.RETRIEVAL_ARTIFACT_SCHEMA_VERSION,
        "protocol_version": runner.RETRIEVAL_PROTOCOL_VERSION,
        "implementation": {"tree_sha256": "a" * 64, "files": {}},
        "track": "longmemeval-s",
        "granularity": "one memory per haystack session",
        "recall_limit": 10,
        "saved_ranking_depth": 50,
        "dense_lane_enabled": True,
        "temporal_query_routing": {"enabled": False, "parser": None},
        "embedding": {
            "provider": "OpenAICompatibleEmbeddingProvider",
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "dimensions": 1024,
        },
        "embedding_call_accounting": {
            "document_inputs": 23867,
            "document_batch_calls": 500,
            "query_calls": 500,
        },
        "cases": [],
    }
    run_path.write_text(json.dumps(payload), encoding="utf-8")

    def fake_evaluate_saved_run(path: Path, k: int) -> dict[str, object]:
        assert path == run_path
        assert k == 10
        return {"final": {"cases": 0}}

    monkeypatch.setattr(runner, "evaluate_saved_run", fake_evaluate_saved_run)
    report = runner.build_report(payload, run_path, k_values=(10,))

    assert report["artifact_type"] == runner.REPORT_ARTIFACT_TYPE
    assert report["schema_version"] == 2
    assert report["run_artifact"]["path"] == str(run_path.resolve())
    assert report["run_artifact"]["bytes"] == run_path.stat().st_size
    assert len(report["run_artifact"]["sha256"]) == 64
    assert report["execution"]["protocol_version"] == runner.RETRIEVAL_PROTOCOL_VERSION
    assert report["execution"]["embedding_call_accounting"]["query_calls"] == 500
    assert report["execution"]["context_token_accounting"] == runner.context_token_accounting()
    assert report["execution"]["context_token_accounting"]["exact_model_tokenizer"] is False


def test_longmemeval_call_accounting_reconciles_observed_http_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {"haystack_sessions": [[], []]},
        {"haystack_sessions": [[]]},
    ]
    observed = {
        "document_inputs": 3,
        "document_batch_calls": 2,
        "query_calls": 2,
        "successful_http_calls": 4,
        "http_attempts": 5,
    }
    monkeypatch.setattr(runner, "observed_embedding_call_accounting", lambda: observed)

    result = runner._longmemeval_embedding_call_accounting(records, use_dense=True)

    assert result == {**observed, "source": "provider-observed"}
    observed["query_calls"] = 1
    with pytest.raises(RuntimeError, match="query_calls"):
        runner._longmemeval_embedding_call_accounting(records, use_dense=True)
