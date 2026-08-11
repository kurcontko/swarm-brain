from __future__ import annotations

import json
from pathlib import Path

import pytest
import run_longmemeval_official_judge as official
import run_longmemeval_qa as qa


class _JudgeClient:
    def __init__(self, responses: tuple[str, ...], *, repeated_id: bool = False) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.repeated_id = repeated_id

    async def complete(self, prompt: str) -> qa.ChatResult:
        index = len(self.prompts)
        self.prompts.append(prompt)
        content = self.responses.pop(0)
        request_id = "judge-repeated" if self.repeated_id else f"judge-{index}"
        raw = json.dumps(
            {
                "id": request_id,
                "model": qa.OFFICIAL_JUDGE_MODEL,
                "system_fingerprint": "official-judge-fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100 + index,
                    "completion_tokens": 1,
                    "total_tokens": 101 + index,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return qa.chat_result_from_raw_response(
            raw,
            prompt=prompt,
            attempts=1,
            latency_ms=10.0,
            request_model=qa.OFFICIAL_JUDGE_MODEL,
            request_temperature=0.0,
            request_max_tokens=10,
            endpoint_url="https://api.openai.com/v1/chat/completions",
        )


def _references() -> list[dict[str, str]]:
    return [
        {
            "question_id": "q-answerable",
            "question_type": "multi-session",
            "question": "Where is the key?",
            "answer": "Under the plant pot.",
        },
        {
            "question_id": "q_abs",
            "question_type": "temporal-reasoning",
            "question": "When did it happen?",
            "answer": "The history does not say.",
        },
    ]


def _hypotheses() -> tuple[dict[str, str], ...]:
    return (
        {"question_id": "q-answerable", "hypothesis": "Under the plant pot."},
        {"question_id": "q_abs", "hypothesis": "That information is unavailable."},
    )


async def test_official_runner_uses_exact_prompts_and_emits_replayable_receipts() -> None:
    client = _JudgeClient(("Yes", "no"))
    outcomes = await official.run_official_judge(
        _references(),
        _hypotheses(),
        client=client,  # type: ignore[arg-type]
        concurrency=2,
    )

    assert client.prompts == [
        qa.judge_prompt(
            "multi-session",
            "Where is the key?",
            "Under the plant pot.",
            "Under the plant pot.",
        ),
        qa.judge_prompt(
            "temporal-reasoning",
            "When did it happen?",
            "The history does not say.",
            "That information is unavailable.",
            abstention=True,
        ),
    ]
    assert [outcome.label for outcome in outcomes] == [True, False]
    labels = official.official_label_records(outcomes)
    assert labels[0] == {
        "question_id": "q-answerable",
        "hypothesis": "Under the plant pot.",
        "autoeval_label": {"model": qa.OFFICIAL_JUDGE_MODEL, "label": True},
    }
    receipts = official.official_receipt_records(outcomes)
    assert [record["call_role"] for record in receipts] == [
        "official_judge",
        "official_judge",
    ]
    replayed = qa.validate_chat_receipt_record(receipts[0])
    assert replayed.response_model == qa.OFFICIAL_JUDGE_MODEL
    assert replayed.request.model == qa.OFFICIAL_JUDGE_MODEL
    assert replayed.request.max_tokens == 10
    assert replayed.content == "Yes"
    assert replayed.prompt_bytes.decode() == client.prompts[0]


async def test_official_runner_rejects_coverage_and_reused_provider_ids() -> None:
    client = _JudgeClient(("yes",))
    with pytest.raises(official.OfficialJudgeRunError, match="coverage differs"):
        await official.run_official_judge(
            _references(),
            _hypotheses()[:1],
            client=client,  # type: ignore[arg-type]
            concurrency=1,
        )

    repeated = _JudgeClient(("yes", "no"), repeated_id=True)
    with pytest.raises(official.OfficialJudgeRunError, match="not unique"):
        await official.run_official_judge(
            _references(),
            _hypotheses(),
            client=repeated,  # type: ignore[arg-type]
            concurrency=1,
        )


def test_hypothesis_loader_is_strict_and_order_preserving(tmp_path: Path) -> None:
    path = tmp_path / "hypotheses.jsonl"
    path.write_text(
        '{"question_id":"q2","hypothesis":"second"}\n{"question_id":"q1","hypothesis":"first"}\n',
        encoding="utf-8",
    )
    assert official.load_hypotheses(path) == (
        {"question_id": "q2", "hypothesis": "second"},
        {"question_id": "q1", "hypothesis": "first"},
    )

    path.write_text(
        '{"question_id":"q1","question_id":"shadow","hypothesis":"x"}\n',
        encoding="utf-8",
    )
    with pytest.raises(official.OfficialJudgeRunError, match="malformed JSON"):
        official.load_hypotheses(path)
