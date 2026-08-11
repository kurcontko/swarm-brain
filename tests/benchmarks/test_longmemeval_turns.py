from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from benchmarks.integrations.longmemeval_turns import (
    OFFICIAL_LONGMEMEVAL_S_SHA256,
    TURN_SERIALIZER_VERSION,
    LongMemEvalTurnId,
    LongMemEvalTurnProjectionError,
    compile_dataset_bytes,
    compile_dataset_file,
    implementation_fingerprint,
)


def _record(
    *,
    question_id: str = "q-001",
    dates: list[object] | None = None,
    session_ids: list[object] | None = None,
    sessions: list[object] | None = None,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": "QUESTION-TEXT-MUST-NOT-ENTER-MANIFEST",
        "answer": "ANSWER-TEXT-MUST-NOT-ENTER-MANIFEST",
        "question_date": "2025/01/04 (Sat) 11:00",
        "haystack_session_ids": session_ids or ["session-a"],
        "haystack_dates": dates or ["2025/01/03 (Fri) 09:07"],
        "haystack_sessions": sessions
        or [
            [
                {
                    "role": "user",
                    "content": "  ORIGINAL-TURN-TEXT-MUST-NOT-ENTER-MANIFEST\nCafé e\u0301  ",
                    "has_answer": True,
                },
                {"role": "assistant", "content": "Acknowledged."},
            ]
        ],
        "answer_session_ids": ["session-a"],
    }


def _raw(records: list[dict[str, object]]) -> bytes:
    return (
        json.dumps(records, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _compile(records: list[dict[str, object]]):
    raw = _raw(records)
    return compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="synthetic-longmemeval.json",
    )


def test_projection_preserves_exact_turn_fields_and_binds_every_layer() -> None:
    record = _record()
    corpus = _compile([record])

    assert len(corpus.questions) == 1
    assert len(corpus.sessions) == 1
    assert len(corpus.turns) == 2
    turn = corpus.turns[0]
    expected_content = "  ORIGINAL-TURN-TEXT-MUST-NOT-ENTER-MANIFEST\nCafé e\u0301  "
    expected_text = (
        '{"timestamp":"2025/01/03 (Fri) 09:07","role":"user",'
        '"content":"  ORIGINAL-TURN-TEXT-MUST-NOT-ENTER-MANIFEST\\nCafé é  "}'
    )

    assert turn.turn_id.as_tuple() == ("q-001", 0, 0)
    assert turn.turn_id.canonical_id == '["q-001",0,0]'
    assert turn.parent_session_id == "session-a"
    assert turn.parent_session_date == "2025/01/03 (Fri) 09:07"
    assert turn.parent_session_date_utc == "2025-01-03T09:07:00Z"
    assert turn.role == "user"
    assert turn.original_content == expected_content
    assert turn.serialized_text == expected_text
    assert json.loads(turn.serialized_text) == {
        "timestamp": turn.parent_session_date,
        "role": "user",
        "content": expected_content,
    }
    assert turn.original_content_utf8.bytes == len(expected_content.encode("utf-8"))
    assert (
        turn.original_content_utf8.sha256
        == hashlib.sha256(expected_content.encode("utf-8")).hexdigest()
    )
    assert turn.serialized_document_utf8.bytes == len(expected_text.encode("utf-8"))
    assert (
        turn.serialized_document_utf8.sha256
        == hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
    )
    assert turn.source_turn.sha256 != turn.serialized_document_utf8.sha256
    assert turn.source_session == corpus.sessions[0].source_session
    assert turn.source_record == corpus.questions[0].source_record
    assert corpus.by_id()[LongMemEvalTurnId("q-001", 0, 0)] is turn

    with pytest.raises(FrozenInstanceError):
        turn.original_content = "mutated"  # type: ignore[misc]


def test_projection_and_content_free_manifest_are_deterministic() -> None:
    raw = _raw([_record()])
    digest = hashlib.sha256(raw).hexdigest()
    first = compile_dataset_bytes(raw, expected_sha256=digest, source_label="source.json")
    second = compile_dataset_bytes(raw, expected_sha256=digest, source_label="source.json")

    assert first == second
    assert first.projection_sha256 == second.projection_sha256
    assert first.fingerprint() == second.fingerprint()
    assert first.content_free_manifest() == second.content_free_manifest()
    manifest = first.content_free_manifest()
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True)

    assert "QUESTION-TEXT-MUST-NOT-ENTER-MANIFEST" not in encoded
    assert "ANSWER-TEXT-MUST-NOT-ENTER-MANIFEST" not in encoded
    assert "ORIGINAL-TURN-TEXT-MUST-NOT-ENTER-MANIFEST" not in encoded
    assert "Acknowledged." not in encoded
    assert manifest["serializer_version"] == TURN_SERIALIZER_VERSION
    assert manifest["fingerprint"]["question_count"] == 1
    assert manifest["fingerprint"]["session_count"] == 1
    assert manifest["fingerprint"]["turn_count"] == 2
    assert manifest["official_cleaned_release"] == {
        "required_sha256": OFFICIAL_LONGMEMEVAL_S_SHA256,
        "verified": False,
    }
    assert manifest["turns"][0]["turn_id"] == ["q-001", 0, 0]
    assert manifest["turns"][0]["parent_session_id"] == "session-a"
    assert manifest["turns"][0]["parent_session_date"] == "2025/01/03 (Fri) 09:07"
    assert set(manifest["implementation"]["files_sha256"]) == {
        "pyproject.toml",
        "uv.lock",
        "benchmarks/__init__.py",
        "benchmarks/integrations/longmemeval_turns/__init__.py",
        "benchmarks/integrations/longmemeval_turns/compiler.py",
    }


def test_projection_digest_is_cached_after_corpus_validation(monkeypatch) -> None:
    corpus = _compile([_record()])
    expected = corpus.projection_sha256

    def fail_if_recomputed(_self) -> dict[str, object]:
        raise AssertionError("immutable projection payload was recomputed")

    monkeypatch.setattr(type(corpus), "_projection_payload", fail_if_recomputed)

    assert corpus.projection_sha256 == expected


def test_source_file_bytes_and_parsed_records_have_separate_bindings(tmp_path: Path) -> None:
    records = [_record()]
    compact = _raw(records)
    pretty = (json.dumps(records, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    compact_path = tmp_path / "compact.json"
    compact_path.write_bytes(compact)

    compact_corpus = compile_dataset_file(
        compact_path,
        expected_sha256=hashlib.sha256(compact).hexdigest(),
    )
    pretty_corpus = compile_dataset_bytes(
        pretty,
        expected_sha256=hashlib.sha256(pretty).hexdigest(),
        source_label="pretty.json",
    )

    assert compact_corpus.source_artifact.sha256 != pretty_corpus.source_artifact.sha256
    assert compact_corpus.source_artifact.bytes != pretty_corpus.source_artifact.bytes
    assert compact_corpus.parsed_records == pretty_corpus.parsed_records
    assert (
        compact_corpus.turns[0].serialized_document_utf8
        == pretty_corpus.turns[0].serialized_document_utf8
    )
    assert compact_corpus.projection_sha256 != pretty_corpus.projection_sha256


def test_checked_in_longmemeval_shape_compiles_without_network_access() -> None:
    path = Path("tests/fixtures/retrieval_eval_corpus/longmemeval_sample.json")
    raw = path.read_bytes()
    corpus = compile_dataset_file(
        path,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )

    assert len(corpus.questions) == 2
    assert len(corpus.sessions) == 5
    assert len(corpus.turns) == 10
    assert corpus.turns[-1].turn_id.as_tuple() == ("sample_lme_002_abs", 1, 1)


def test_source_digest_is_required_and_verified_before_projection() -> None:
    raw = _raw([_record()])

    with pytest.raises(LongMemEvalTurnProjectionError, match="lowercase hexadecimal"):
        compile_dataset_bytes(raw, expected_sha256="unverified")
    with pytest.raises(LongMemEvalTurnProjectionError, match="differs"):
        compile_dataset_bytes(raw, expected_sha256="0" * 64)

    assert len(OFFICIAL_LONGMEMEVAL_S_SHA256) == 64


@pytest.mark.parametrize(
    "timestamp",
    [
        "2025/01/03 (Fri) 09:07 ",
        "2025/1/03 (Fri) 09:07",
        "2025/01/03 (Thu) 09:07",
        "2025/02/30 (Sun) 09:07",
        "2025-01-03T09:07:00Z",
        None,
    ],
)
def test_unstable_or_invalid_timestamps_fail_closed(timestamp: object) -> None:
    with pytest.raises(LongMemEvalTurnProjectionError, match="timestamp|date|calendar|weekday"):
        _compile([_record(dates=[timestamp])])


def test_misaligned_haystack_arrays_fail_closed() -> None:
    record = _record()
    record["haystack_dates"] = [
        "2025/01/03 (Fri) 09:07",
        "2025/01/04 (Sat) 09:07",
    ]

    with pytest.raises(LongMemEvalTurnProjectionError, match="misaligned"):
        _compile([record])


@pytest.mark.parametrize("content", [None, 7, True, ["text"], {"text": "value"}])
def test_non_string_content_fails_closed(content: object) -> None:
    sessions: list[object] = [[{"role": "user", "content": content}]]

    with pytest.raises(LongMemEvalTurnProjectionError, match="content must be a string"):
        _compile([_record(sessions=sessions)])


def test_missing_or_unstable_role_fails_closed() -> None:
    missing_role: list[object] = [[{"content": "text"}]]
    padded_role: list[object] = [[{"role": " user ", "content": "text"}]]

    with pytest.raises(LongMemEvalTurnProjectionError, match="role is missing"):
        _compile([_record(sessions=missing_role)])
    with pytest.raises(LongMemEvalTurnProjectionError, match="leading or trailing"):
        _compile([_record(sessions=padded_role)])


def test_duplicate_question_ids_and_json_keys_fail_closed() -> None:
    duplicate_records = [_record(), _record()]
    with pytest.raises(LongMemEvalTurnProjectionError, match="duplicate question_id"):
        _compile(duplicate_records)

    raw = (
        b'[{"question_id":"q","question_id":"q2","haystack_session_ids":[],'
        b'"haystack_dates":[],"haystack_sessions":[]}]'
    )
    with pytest.raises(LongMemEvalTurnProjectionError, match="repeats object key"):
        compile_dataset_bytes(raw, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_repeated_parent_session_ids_remain_unambiguous_by_position() -> None:
    sessions: list[object] = [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]
    corpus = _compile(
        [
            _record(
                session_ids=["repeated", "repeated"],
                dates=["2025/01/03 (Fri) 09:07", "2025/01/04 (Sat) 10:08"],
                sessions=sessions,
            )
        ]
    )

    assert [turn.turn_id.as_tuple() for turn in corpus.turns] == [
        ("q-001", 0, 0),
        ("q-001", 1, 0),
    ]
    assert len(corpus.by_id()) == 2


def test_public_immutable_contract_rejects_tampered_documents_and_order() -> None:
    corpus = _compile(
        [
            _record(
                session_ids=["first", "second"],
                dates=["2025/01/03 (Fri) 09:07", "2025/01/04 (Sat) 10:08"],
                sessions=[
                    [{"role": "user", "content": "first"}],
                    [{"role": "assistant", "content": "second"}],
                ],
            )
        ]
    )

    with pytest.raises(LongMemEvalTurnProjectionError, match="serialized turn document"):
        replace(corpus.turns[0], serialized_text="tampered")
    with pytest.raises(LongMemEvalTurnProjectionError, match="contiguous|source order"):
        replace(corpus, sessions=tuple(reversed(corpus.sessions)))
    with pytest.raises(LongMemEvalTurnProjectionError, match="source order"):
        replace(corpus, turns=tuple(reversed(corpus.turns)))
    with pytest.raises(LongMemEvalTurnProjectionError, match="repeats a turn ID"):
        replace(corpus, turns=(corpus.turns[0], corpus.turns[0], corpus.turns[1]))


def test_source_turn_digest_binds_ignored_source_metadata() -> None:
    first = _record()
    second = _record()
    second_sessions = second["haystack_sessions"]
    assert isinstance(second_sessions, list)
    assert isinstance(second_sessions[0], list)
    assert isinstance(second_sessions[0][0], dict)
    second_sessions[0][0]["has_answer"] = False

    first_turn = _compile([first]).turns[0]
    second_turn = _compile([second]).turns[0]
    assert first_turn.serialized_text == second_turn.serialized_text
    assert first_turn.serialized_document_utf8 == second_turn.serialized_document_utf8
    assert first_turn.source_turn != second_turn.source_turn
    assert first_turn.source_session != second_turn.source_session
    assert first_turn.source_record != second_turn.source_record


def test_implementation_fingerprint_is_content_free_and_repeatable() -> None:
    first = implementation_fingerprint()
    second = implementation_fingerprint()

    assert first == second
    assert len(first["tree_sha256"]) == 64
    assert all(len(digest) == 64 for digest in first["files_sha256"].values())
