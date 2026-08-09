"""The demo's hidden gate depends only on the explicitly delivered memory hits."""

from __future__ import annotations

import hashlib
from uuid import uuid4

from swarmbrain.demo.verification import verify_memory_context


def _expected(guard: str, procedure: str) -> str:
    return hashlib.sha256(f"{guard}\n{procedure}".encode()).hexdigest()


def _hit(
    slot: str,
    token: str,
    *,
    state: str = "confirmed",
    recorded_to: str | None = None,
    marker: bool = True,
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if marker:
        metadata["demo_verification"] = {"slot": slot, "token": token}
    return {
        "score": 1.0,
        "memory": {
            "memory_id": str(uuid4()),
            "state": state,
            "recorded_to": recorded_to,
            "content": f"visible text mentions {token}",
            "metadata": metadata,
        },
    }


def test_empty_context_fails_the_same_gate_that_current_memories_pass() -> None:
    guard = uuid4().hex
    procedure = uuid4().hex
    expected = _expected(guard, procedure)

    disabled = verify_memory_context((), expected_sha256=expected)
    enabled = verify_memory_context(
        (_hit("guard", guard), _hit("procedure", procedure)),
        expected_sha256=expected,
    )

    assert disabled.passed is False
    assert disabled.delivered_memory_ids == ()
    assert set(disabled.missing_slots) == {"guard", "procedure"}
    assert enabled.passed is True
    assert len(enabled.accepted_memory_ids) == 2
    assert enabled.answer_sha256 == expected


def test_missing_or_incorrect_memory_cannot_satisfy_the_gate() -> None:
    guard = uuid4().hex
    procedure = uuid4().hex
    expected = _expected(guard, procedure)

    missing = verify_memory_context((_hit("guard", guard),), expected_sha256=expected)
    incorrect = verify_memory_context(
        (_hit("guard", guard), _hit("procedure", "wrong-token")),
        expected_sha256=expected,
    )

    assert missing.passed is False
    assert missing.missing_slots == ("procedure",)
    assert incorrect.passed is False
    assert incorrect.missing_slots == ()
    assert incorrect.answer_sha256 != expected


def test_superseded_ambiguous_and_unstructured_content_are_rejected() -> None:
    guard = uuid4().hex
    procedure = uuid4().hex
    expected = _expected(guard, procedure)
    superseded = _hit(
        "guard",
        guard,
        state="superseded",
        recorded_to="2026-08-09T12:00:00Z",
    )
    content_only = _hit("guard", guard, marker=False)
    ambiguous = (
        _hit("guard", guard),
        _hit("guard", "poison-token"),
        _hit("procedure", procedure),
    )

    assert not verify_memory_context(
        (superseded, _hit("procedure", procedure)), expected_sha256=expected
    ).passed
    assert not verify_memory_context(
        (content_only, _hit("procedure", procedure)), expected_sha256=expected
    ).passed
    result = verify_memory_context(ambiguous, expected_sha256=expected)
    assert result.passed is False
    assert "guard" in result.missing_slots
