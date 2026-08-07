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
    """The tiny checked-in sample proves the mapper without the 278 MB release."""

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
