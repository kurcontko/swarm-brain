from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.integrations.longmemeval_turns import compile_dataset_bytes
from scripts.run_longmemeval_e1_external import SelectedQuestion, seal_artifact
from scripts.run_longmemeval_e2_external import (
    E2Context,
    ExternalE2Error,
    _compute_organization,
    _qa_arm_order,
    replay_organization_row,
)


class _Scalar(float):
    def item(self) -> float:
        return float(self)


class _Vector(tuple[float, ...]):
    def __new__(cls, values):
        return super().__new__(cls, values)

    def __matmul__(self, other: _Vector) -> _Scalar:
        return _Scalar(sum(left * right for left, right in zip(self, other, strict=True)))


class _Scores(list[float]):
    def tolist(self) -> list[float]:
        return list(self)


class _Matrix(tuple[_Vector, ...]):
    def __new__(cls, values):
        return super().__new__(cls, values)

    def __matmul__(self, other: _Vector) -> _Scores:
        return _Scores(vector @ other for vector in self)


class _FakeBatch:
    def __init__(self, vectors: list[_Vector], texts: list[str]) -> None:
        self.vectors = _Matrix(vectors)
        self.token_counts = tuple(max(1, len(text.split())) for text in texts)
        self.truncated = tuple(False for _ in texts)
        self.singleton_retry_positions = ()
        self.batch_plan = (tuple(range(len(texts))),)
        self.model_batches = 1
        self.padded_tokens = sum(self.token_counts)
        self.padded_attention_cells = sum(value * value for value in self.token_counts)
        self.elapsed_ms = 1.0


class _FakeEmbedder:
    torch = SimpleNamespace(__version__="fixture-torch")
    transformers_version = "fixture-transformers"
    device = "cpu"
    dtype_name = "float32"
    batch_size = 8

    def embed(self, texts: list[str]) -> _FakeBatch:
        vectors = [
            _Vector((1.0, 0.0)) if text.startswith("Instruct:") else _Vector((0.8, 0.6))
            for text in texts
        ]
        return _FakeBatch(vectors, texts)


def _fixture() -> tuple[E2Context, SelectedQuestion, dict, object]:
    record = {
        "question_id": "q-e2-external",
        "question_type": "multi-session",
        "question": "Which evidence matters?",
        "answer": "gold",
        "question_date": "2025/02/01 (Sat) 12:00",
        "haystack_session_ids": ["session"],
        "haystack_dates": ["2025/01/31 (Fri) 09:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": f"unique evidence {position:02d}"}
                for position in range(20)
            ]
        ],
        "answer_session_ids": ["session"],
    }
    raw = (json.dumps([record], separators=(",", ":")) + "\n").encode()
    corpus = compile_dataset_bytes(
        raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        source_label="synthetic-e2-external.json",
    )
    question = SelectedQuestion(position=0, record=record, turns=corpus.turns)
    e1a = SimpleNamespace(
        candidates=tuple(SimpleNamespace(turn=turn, turn_id=turn.turn_id) for turn in corpus.turns),
        trace_sha256="c" * 64,
    )
    dense_row = {
        "artifact_sha256": "d" * 64,
        "e1a_trace": {"trace_sha256": e1a.trace_sha256},
        "dense": {
            "observations": [
                {"turn_id": list(turn.turn_id.as_tuple()), "raw_cosine": 0.8}
                for turn in corpus.turns
            ]
        },
    }
    context = E2Context(
        e1=SimpleNamespace(),
        output_dir=Path("/unused"),
        manifest={"artifact_sha256": "b" * 64},
    )
    return context, question, dense_row, e1a


def test_real_model_boundary_row_replays_without_reexecuting_embedder(monkeypatch) -> None:
    context, question, dense_row, e1a = _fixture()
    monkeypatch.setattr(
        "scripts.run_longmemeval_e2_external._dense_source",
        lambda _context, _question: (dense_row, e1a),
    )
    monkeypatch.setattr(
        "scripts.run_longmemeval_e2_external._snapshot_artifact",
        lambda _context, _name: "a" * 64,
    )

    row = _compute_organization(context, question, embedder=_FakeEmbedder())
    pool, results = replay_organization_row(context, question, row)

    assert len(pool) == 20
    assert results["E2-D"].rendered_candidates()
    assert len(results["E2-D"].rendered_candidates()) == 60
    assert len(results["E2-E"].rendered_candidates()) == 20
    assert row["similarity_observation"]["query_replay"]["maximum_absolute_delta"] == pytest.approx(
        0.0
    )


def test_similarity_replay_rejects_resealed_internal_score_inconsistency(monkeypatch) -> None:
    context, question, dense_row, e1a = _fixture()
    monkeypatch.setattr(
        "scripts.run_longmemeval_e2_external._dense_source",
        lambda _context, _question: (dense_row, e1a),
    )
    monkeypatch.setattr(
        "scripts.run_longmemeval_e2_external._snapshot_artifact",
        lambda _context, _name: "a" * 64,
    )
    row = _compute_organization(context, question, embedder=_FakeEmbedder())
    payload = {key: value for key, value in row.items() if key != "artifact_sha256"}
    observation = dict(payload["similarity_observation"])
    observation_payload = {
        key: value for key, value in observation.items() if key != "artifact_sha256"
    }
    replay = dict(observation_payload["query_replay"])
    rows = [dict(value) for value in replay["observations"]]
    rows[0]["reexecuted_raw_cosine"] = 0.7
    replay["observations"] = rows
    observation_payload["query_replay"] = replay
    observation = {
        **observation_payload,
        "artifact_sha256": hashlib.sha256(
            json.dumps(
                observation_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }
    payload["similarity_observation"] = observation
    tampered = seal_artifact(payload)

    with pytest.raises(ExternalE2Error, match="absolute delta"):
        replay_organization_row(context, question, tampered)


def test_e2_qa_order_rotates_by_question_position() -> None:
    _, question, _, _ = _fixture()

    assert _qa_arm_order(question) == ("E2-A", "E2-E")
    odd = SelectedQuestion(position=1, record=question.record, turns=question.turns)
    assert _qa_arm_order(odd) == ("E2-E", "E2-A")
