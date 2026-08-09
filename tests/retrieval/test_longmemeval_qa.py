"""Protocol gate for the LongMemEval-S QA harness.

The measured accuracy lives in the benchmark reports; what is asserted here is
the part that has to be *exactly* right for those numbers to mean anything:
the reader context the official generation script would have produced, the
hypothesis file format `evaluate_qa.py` consumes, and the per-question-type
judge prompt selection copied from that same file.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

# ``scripts`` is on the pytest pythonpath (pyproject.toml), so the harness
# imports like any other module.
import run_longmemeval_qa as qa

SESSIONS: list[tuple[str, list[dict[str, str]]]] = [
    (
        "2026/02/06 (Fri) 08:12",
        [
            {"role": "user", "content": " I moved the key.  "},
            {"role": "assistant", "content": "Noted."},
        ],
    ),
    (
        "2026/02/01 (Sun) 11:02",
        [{"role": "user", "content": "I replaced the hallway bulb."}],
    ),
]

RECORD: dict[str, Any] = {
    "question_id": "sample_lme_001",
    "question_type": "single-session-user",
    "question": "Where did I put the spare apartment key?",
    "answer": "under the blue plant pot",
    "question_date": "2026/02/10 (Tue) 09:15",
}


# --------------------------------------------------------------------------- #
# context rendering
# --------------------------------------------------------------------------- #


def test_history_is_ordered_chronologically_not_by_rank() -> None:
    history = qa.render_history(SESSIONS)
    early = history.index("2026/02/01")
    late = history.index("2026/02/06")
    assert early < late
    # Renumbered after sorting, exactly like the reference script.
    assert history.index("### Session 1:") < history.index("### Session 2:")
    assert "hallway bulb" in history.split("### Session 2:")[0]


def test_history_ties_keep_retrieval_rank_order() -> None:
    same_date = [
        ("2026/02/01 (Sun) 11:02", [{"role": "user", "content": "first"}]),
        ("2026/02/01 (Sun) 11:02", [{"role": "user", "content": "second"}]),
    ]
    history = qa.render_history(same_date)
    assert history.index("first") < history.index("second")


def test_official_prompt_matches_the_reference_template_byte_for_byte() -> None:
    """Reconstructed independently from LongMemEval ``run_generation.py``."""

    expected_history = (
        "\n### Session 1:\nSession Date: 2026/02/01 (Sun) 11:02\nSession Content:\n"
        "\n\nuser: I replaced the hallway bulb.\n"
        "\n### Session 2:\nSession Date: 2026/02/06 (Fri) 08:12\nSession Content:\n"
        "\n\nuser: I moved the key.\n\nassistant: Noted.\n"
    )
    expected = (
        "I will give you several history chats between you and a user. Please answer "
        "the question based on the relevant chat history.\n\n\nHistory Chats:\n\n"
        f"{expected_history}\n\nCurrent Date: 2026/02/10 (Tue) 09:15\n"
        "Question: Where did I put the spare apartment key?\nAnswer:"
    )
    assert qa.build_reader_prompt(RECORD, SESSIONS, style="official") == expected


def test_swarm_style_adds_only_the_evidence_only_instruction() -> None:
    official = qa.build_reader_prompt(RECORD, SESSIONS, style="official")
    swarm = qa.build_reader_prompt(RECORD, SESSIONS, style="swarm")
    assert swarm != official
    assert "the information is not available" in swarm
    # Everything from the history onwards is untouched.
    assert swarm.split("History Chats:")[1] == official.split("History Chats:")[1]


def test_an_empty_bundle_is_surfaced_to_the_reader() -> None:
    prompt = qa.build_reader_prompt(RECORD, [], style="swarm", requested=10, floored=True)
    assert qa.EMPTY_CONTEXT_NOTE in prompt
    assert "### Session" not in prompt
    assert RECORD["question"] in prompt


def test_a_thin_bundle_is_announced_only_when_a_floor_was_applied() -> None:
    floored = qa.build_reader_prompt(RECORD, SESSIONS, style="swarm", requested=10, floored=True)
    assert "only 2 of the 10 requested sessions" in floored

    unfloored = qa.build_reader_prompt(RECORD, SESSIONS, style="swarm", requested=10, floored=False)
    assert "requested sessions" not in unfloored

    # The official style stays byte-identical to the reference under a floor.
    official = qa.build_reader_prompt(RECORD, SESSIONS, style="official", requested=10, floored=True)
    assert official == qa.build_reader_prompt(RECORD, SESSIONS, style="official")


# --------------------------------------------------------------------------- #
# hypothesis file format
# --------------------------------------------------------------------------- #


def test_hypothesis_line_is_exactly_what_evaluate_qa_consumes() -> None:
    line = qa.hypothesis_line("gpt4_4929293a", 'He said "yes"\nthen left.')
    parsed = json.loads(line)
    assert set(parsed) == {"question_id", "hypothesis"}
    assert parsed["question_id"] == "gpt4_4929293a"
    assert parsed["hypothesis"] == 'He said "yes"\nthen left.'
    assert "\n" not in line  # one question per line, always


# --------------------------------------------------------------------------- #
# judge prompt selection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("question_type", "marker"),
    [
        ("single-session-user", "Correct Answer:"),
        ("single-session-assistant", "Correct Answer:"),
        ("multi-session", "Correct Answer:"),
        ("temporal-reasoning", "do not penalize off-by-one errors"),
        ("knowledge-update", "as long as the updated answer is the required answer"),
        ("single-session-preference", "Rubric:"),
    ],
)
def test_each_question_type_gets_its_official_judge_prompt(
    question_type: str, marker: str
) -> None:
    prompt = qa.judge_prompt(question_type, "q", "a", "r")
    assert marker in prompt
    assert prompt.endswith("Is the model response correct? Answer yes or no only.")


def test_temporal_and_knowledge_update_differ_from_the_default_prompt() -> None:
    default = qa.judge_prompt("multi-session", "q", "a", "r")
    assert qa.judge_prompt("temporal-reasoning", "q", "a", "r") != default
    assert qa.judge_prompt("knowledge-update", "q", "a", "r") != default
    assert qa.judge_prompt("single-session-preference", "q", "a", "r") != default


def test_abstention_questions_take_the_abstention_path() -> None:
    assert qa.is_abstention_question("3e5fea0e_abs_1")
    assert not qa.is_abstention_question("gpt4_4929293a")

    prompt = qa.judge_prompt("multi-session", "q", "a", "r", abstention=True)
    assert "unanswerable" in prompt
    assert "Explanation:" in prompt
    assert prompt.endswith("Answer yes or no only.")
    # The abstention prompt overrides the question type, as in evaluate_qa.py.
    assert prompt == qa.judge_prompt("temporal-reasoning", "q", "a", "r", abstention=True)


def test_an_unknown_question_type_fails_loudly() -> None:
    with pytest.raises(NotImplementedError):
        qa.judge_prompt("something-new", "q", "a", "r")


def test_judge_label_follows_the_official_rule() -> None:
    assert qa.judge_label("yes")
    assert qa.judge_label("  Yes.\n")
    assert not qa.judge_label("no")
    assert not qa.judge_label("")


# --------------------------------------------------------------------------- #
# chat client behaviour around the reasoning-model quirk
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.requests.append({"url": url, **kwargs})
        return self._responses.pop(0)

    async def aclose(self) -> None:
        return None


def _completion(content: str | None, *, reasoning: str = "", finish: str = "stop") -> _FakeResponse:
    return _FakeResponse(
        200,
        {
            "choices": [
                {
                    "message": {"content": content, "reasoning_content": reasoning},
                    "finish_reason": finish,
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        },
    )


def _client(responses: list[_FakeResponse], **kwargs: Any) -> tuple[qa.ChatClient, _FakeClient]:
    fake = _FakeClient(responses)
    return (
        qa.ChatClient(
            base_url="https://api.example.com",
            model="reasoner-1",
            api_key="secret",
            temperature=0.0,
            max_tokens=4096,
            client=fake,
            **kwargs,
        ),
        fake,
    )


async def test_reader_reads_content_and_ignores_reasoning_content() -> None:
    client, fake = _client([_completion("under the blue plant pot", reasoning="thinking...")])
    result = await client.complete("prompt")
    assert result.content == "under the blue plant pot"
    assert result.prompt_tokens == 11
    assert result.attempts == 1
    assert fake.requests[0]["url"] == "https://api.example.com/v1/chat/completions"
    assert fake.requests[0]["json"]["max_tokens"] == 4096
    assert fake.requests[0]["json"]["temperature"] == 0.0


async def test_empty_content_from_a_spent_reasoning_budget_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qa, "_BACKOFF_SECONDS", (0.0, 0.0, 0.0, 0.0))
    client, _ = _client(
        [
            _completion("", reasoning="all budget spent here", finish="length"),
            _completion("recovered"),
        ]
    )
    result = await client.complete("prompt")
    assert result.content == "recovered"
    assert result.attempts == 2


async def test_rate_limits_are_retried_and_exhaustion_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qa, "_BACKOFF_SECONDS", (0.0, 0.0, 0.0, 0.0))
    client, _ = _client([_FakeResponse(429) for _ in range(4)], attempts=4)
    with pytest.raises(qa.ChatUnavailable, match="after 4 attempts"):
        await client.complete("prompt")


async def test_a_protocol_error_fails_fast() -> None:
    client, fake = _client([_FakeResponse(400, {"error": "bad model"})])
    with pytest.raises(qa.ChatProtocolError, match="HTTP 400"):
        await client.complete("prompt")
    assert len(fake.requests) == 1


def test_the_base_url_tolerates_a_trailing_v1() -> None:
    client, _ = _client([])
    assert client.base_url == "https://api.example.com"
    stripped = qa.ChatClient(
        base_url="https://api.example.com/v1/",
        model="m",
        api_key=None,
        temperature=0.0,
        max_tokens=16,
        client=_FakeClient([]),
    )
    assert stripped.base_url == "https://api.example.com"


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _outcome(
    question_id: str,
    question_type: str,
    label: bool | None,
    *,
    retrieved: tuple[str, ...] = ("000:a",),
    gold: tuple[str, ...] = ("000:a",),
) -> qa.QuestionOutcome:
    return qa.QuestionOutcome(
        record={"question_id": question_id, "question_type": question_type},
        retrieved_keys=retrieved,
        retrieved_relevance=tuple(0.5 for _ in retrieved),
        gold_keys=gold,
        hypothesis="answer" if label is not None else "",
        reader=qa.ChatResult(
            content="answer",
            prompt_tokens=20000,
            completion_tokens=40,
            finish_reason="stop",
            attempts=1,
            latency_ms=1200.0,
        ),
        reader_error=None,
        judge=(
            None
            if label is None
            else qa.ChatResult(
                content="yes" if label else "no",
                prompt_tokens=300,
                completion_tokens=1,
                finish_reason="stop",
                attempts=1,
                latency_ms=400.0,
            )
        ),
        judge_error=None if label is not None else "judge died",
        label=label,
        retrieval_ms=140.0,
        total_ms=1400.0,
    )


def test_report_keys_the_accuracy_as_a_dev_judge_number_and_slices_abstention(
    tmp_path: Any,
) -> None:
    outcomes = [
        _outcome("q1", "multi-session", True),
        _outcome("q2", "multi-session", False),
        _outcome("q3_abs", "temporal-reasoning", True),
        _outcome("q4", "temporal-reasoning", None),
    ]
    report = qa.build_report(
        outcomes,
        metadata={"harness": "test"},
        hypothesis_path=qa.REPO_ROOT / "benchmarks" / "retrieval" / "x-hypotheses.jsonl",
        run_path=qa.REPO_ROOT / "benchmarks" / "retrieval" / "x-run.json",
    )

    assert report["overall"]["dev_judge_accuracy"] == 0.6667
    assert report["overall"]["judged_questions"] == 3
    assert report["overall"]["unjudged_questions"] == 1
    assert "accuracy" not in report  # never an unqualified accuracy key
    assert report["by_question_type"]["multi-session"]["dev_judge_accuracy"] == 0.5
    assert report["by_slice"]["abstention_questions"]["questions"] == 1
    assert report["by_slice"]["answerable_questions"]["questions"] == 3
    assert report["failures"]["judge_failures"] == 1
    assert report["reader_tokens"]["prompt_total"] == 80000
    assert report["judge_tokens"]["prompt_total"] == 900  # the unjudged question contributes none


def test_run_record_carries_the_evidence_for_every_question() -> None:
    run = qa.build_run(
        [_outcome("q1", "multi-session", True, retrieved=("000:a", "001:b"), gold=("000:a",))],
        metadata={"harness": "test"},
        hypothesis_path=qa.REPO_ROOT / "benchmarks" / "retrieval" / "x-hypotheses.jsonl",
    )
    entry = run["questions"][0]
    assert entry["question_id"] == "q1"
    assert entry["retrieved_session_keys"] == ["000:a", "001:b"]
    assert entry["gold_sessions_in_bundle"] == 1
    assert entry["dev_judge_label"] is True
    assert entry["reader_prompt_tokens"] == 20000
