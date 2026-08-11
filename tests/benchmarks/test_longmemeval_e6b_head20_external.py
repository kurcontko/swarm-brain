from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from scripts import run_longmemeval_e6b_head20_external as runner
from scripts.run_longmemeval_e1_external import SelectedQuestion


def _question(
    *,
    position: int,
    question_id: str,
    question_type: str = "single-session-user",
) -> SelectedQuestion:
    return SelectedQuestion(
        position=position,
        record={
            "question_id": question_id,
            "question": "Question?",
            "question_type": question_type,
            "answer": "Answer.",
            "question_date": "2025/01/01 (Wed) 00:00",
        },
        turns=(),
    )


def test_qa_arm_order_uses_frozen_run_position_not_dataset_position() -> None:
    first = _question(position=101, question_id="first")
    second = _question(position=200, question_id="second")
    context = SimpleNamespace(e1=SimpleNamespace(selected=(first, second)))

    assert runner._qa_arm_order(context, first) == runner.E6_CELLS
    assert runner._qa_arm_order(context, second) == tuple(reversed(runner.E6_CELLS))

    outsider = _question(position=300, question_id="outsider")
    with pytest.raises(runner.ExternalE6Error, match="outside the frozen run order"):
        runner._qa_arm_order(context, outsider)


def _g1_diagnostic() -> dict[str, Any]:
    noninferiority_axes = (
        "candidate_any_gold_in_context",
        "candidate_all_gold_in_context",
        "candidate_answer_session_recall",
        "prompt_any_gold_in_context",
        "prompt_all_gold_in_context",
        "prompt_answer_session_recall",
    )
    r0 = {axis: 0.75 for axis in noninferiority_axes}
    r0.update(
        {
            "candidate_answer_session_mrr": 0.50,
            "prompt_answer_session_mrr": 0.50,
        }
    )
    r1 = dict(r0)
    r1.update(
        {
            "candidate_answer_session_mrr": 0.51,
            "prompt_answer_session_mrr": 0.51,
        }
    )
    cases = [
        {
            "arms": {
                cell: {
                    "context": {
                        "candidate_value_count": runner.HEAD_MATCHED_VALUE_COUNT,
                        "prompt_value_count": runner.HEAD_MATCHED_VALUE_COUNT,
                    }
                }
                for cell in runner.E6_CELLS
            }
        }
        for _ in range(runner.E6B_SAMPLE)
    ]
    return {
        "context_quality": {
            "available": True,
            "gold_eligible_cases": runner.E6B_SAMPLE - runner.E6B_ABS_COUNT,
            "arms": {"R0": r0, "R1": r1},
        },
        "efficiency": {
            "arms": {
                "R0": {"prompt_tokens": {"total": 16_000, "p95": 100.0}},
                "R1": {"prompt_tokens": {"total": 16_000, "p95": 100.0}},
            }
        },
        "cases": cases,
    }


def test_g1_requires_strict_mrr_token_nonregression_and_150_gold_cases() -> None:
    diagnostic = _g1_diagnostic()
    gate = runner._e6b_context_gate_evidence(diagnostic)
    assert gate["passed"] is True
    assert gate["gold_eligible_non_abstention_cases"] == 150
    assert gate["required_gold_eligible_non_abstention_cases"] == 150
    assert gate["prompt_token_noninferiority"]["total"]["passed"] is True
    assert gate["prompt_token_noninferiority"]["p95"]["passed"] is True

    equal_mrr = deepcopy(diagnostic)
    equal_mrr["context_quality"]["arms"]["R1"]["candidate_answer_session_mrr"] = 0.50
    equal_mrr_gate = runner._e6b_context_gate_evidence(equal_mrr)
    assert equal_mrr_gate["passed"] is False
    assert (
        equal_mrr_gate["strict_candidate_and_prompt_mrr_improvement"][
            "candidate_answer_session_mrr"
        ]["passed"]
        is False
    )

    for denominator in (149, 151):
        wrong_denominator = deepcopy(diagnostic)
        wrong_denominator["context_quality"]["gold_eligible_cases"] = denominator
        assert runner._e6b_context_gate_evidence(wrong_denominator)["passed"] is False

    for field, regression in (("total", 16_001), ("p95", 100.1)):
        token_regression = deepcopy(diagnostic)
        token_regression["efficiency"]["arms"]["R1"]["prompt_tokens"][field] = regression
        token_gate = runner._e6b_context_gate_evidence(token_regression)
        assert token_gate["passed"] is False
        assert token_gate["prompt_token_noninferiority"][field]["passed"] is False


def _qa_arm(*, correct: bool, abstention: bool, latency: int, cost: int) -> dict[str, Any]:
    return {
        "qa_correct": correct,
        "context": {
            "prompt": {
                "any_gold_in_context": None if abstention else True,
                "all_gold_in_context": None if abstention else True,
                "answer_session_mrr": None if abstention else 1.0,
            },
            "prompt_tokens": 100,
        },
        "accounting": {
            "construction_plus_query": {
                "latency_microseconds": latency,
                "cost_microusd": cost,
            }
        },
    }


def _g2_diagnostic() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    case_index = 0
    for question_type, count in runner.E6B_ACTUAL_TYPE_COUNTS.items():
        for _ in range(count):
            abstention = case_index < runner.E6B_ABS_COUNT
            question_id = f"case-{case_index:03d}{'_abs' if abstention else ''}"
            cases.append(
                {
                    "question_id": question_id,
                    "question_type": question_type,
                    "arms": {
                        "R0": _qa_arm(
                            correct=False,
                            abstention=abstention,
                            latency=100,
                            cost=10,
                        ),
                        "R1": _qa_arm(
                            correct=not abstention,
                            abstention=abstention,
                            latency=110,
                            cost=11,
                        ),
                    },
                }
            )
            case_index += 1
    assert len(cases) == runner.E6B_SAMPLE
    return {
        "qa": {
            "available": True,
            "complete_case_coverage": True,
            "paired_cases": runner.E6B_SAMPLE,
        },
        "efficiency": {
            "arms": {
                "R0": {
                    "prompt_tokens": {"p95": 100.0},
                    "operational_latency_microseconds": {"p95": 100.0},
                    "construction_plus_query_cost_microusd": {"total": 1_600},
                },
                "R1": {
                    "prompt_tokens": {"p95": 90.0},
                    "operational_latency_microseconds": {"p95": 110.0},
                    "construction_plus_query_cost_microusd": {"total": 1_760},
                },
            }
        },
        "cases": cases,
    }


def _fast_paired_qa_summary(cases: tuple[Any, ...]) -> dict[str, Any]:
    baseline_correct = sum(case.baseline.correct for case in cases)
    candidate_correct = sum(case.candidate.correct for case in cases)
    delta = (candidate_correct - baseline_correct) / len(cases)
    deltas = [int(case.candidate.correct) - int(case.baseline.correct) for case in cases]
    return {
        "baseline": {
            "questions": len(cases),
            "correct": baseline_correct,
            "accuracy": baseline_correct / len(cases),
        },
        "candidate": {
            "questions": len(cases),
            "correct": candidate_correct,
            "accuracy": candidate_correct / len(cases),
        },
        "paired_delta": {
            "delta": delta,
            "ci_low": delta,
            "ci_high": delta,
            "improved_questions": sum(value > 0 for value in deltas),
            "regressed_questions": sum(value < 0 for value in deltas),
            "tied_questions": sum(value == 0 for value in deltas),
        },
    }


def test_g2_depends_on_g0_and_enforces_type_and_abstention_margins(monkeypatch) -> None:
    monkeypatch.setattr(runner, "paired_qa_summary", _fast_paired_qa_summary)
    diagnostic = _g2_diagnostic()

    passed = runner._e6b_qa_gate_evidence(
        diagnostic,
        integrity_gate_passed=True,
        context_gate_passed=True,
    )
    assert passed["passed"] is True
    assert passed["abstention_subgroup"]["paired_delta"] == 0.0
    assert passed["all_question_types_noninferior_at_margin"] is True

    failed_g0 = runner._e6b_qa_gate_evidence(
        diagnostic,
        integrity_gate_passed=False,
        context_gate_passed=True,
    )
    assert failed_g0["passed"] is False
    assert failed_g0["G0_integrity_gate_passed"] is False

    type_regression = deepcopy(diagnostic)
    preference_cases = [
        case
        for case in type_regression["cases"]
        if case["question_type"] == "single-session-preference"
    ]
    for case in preference_cases:
        case["arms"]["R0"]["qa_correct"] = False
        case["arms"]["R1"]["qa_correct"] = False
    preference_cases[0]["arms"]["R0"]["qa_correct"] = True
    type_gate = runner._e6b_qa_gate_evidence(
        type_regression,
        integrity_gate_passed=True,
        context_gate_passed=True,
    )
    assert type_gate["passed"] is False
    assert type_gate["by_question_type"]["single-session-preference"]["paired_delta"] == -0.1
    assert type_gate["by_question_type"]["single-session-preference"]["passed"] is False
    assert type_gate["abstention_subgroup"]["passed"] is True

    abs_regression = deepcopy(diagnostic)
    abstention_case = next(
        case for case in abs_regression["cases"] if "_abs" in case["question_id"]
    )
    abstention_case["arms"]["R0"]["qa_correct"] = True
    abstention_case["arms"]["R1"]["qa_correct"] = False
    abs_gate = runner._e6b_qa_gate_evidence(
        abs_regression,
        integrity_gate_passed=True,
        context_gate_passed=True,
    )
    assert abs_gate["passed"] is False
    assert abs_gate["all_question_types_noninferior_at_margin"] is True
    assert abs_gate["abstention_subgroup"]["paired_delta"] == -0.1
    assert abs_gate["abstention_subgroup"]["passed"] is False


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["all", "--limit", "39"], "frozen 40-question operational tranche"),
        (
            ["diagnose", "--limit", "40"],
            "aggregate diagnostics and QA are forbidden on the operational tranche",
        ),
    ],
)
def test_limited_cli_rejects_arbitrary_prefix_and_diagnostics(
    argv: list[str],
    message: str,
) -> None:
    with pytest.raises(SystemExit, match=message):
        runner.main(argv)


def test_all_resume_reuses_frozen_context_diagnostic_during_partial_qa(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    question = _question(position=7, question_id="resume-case")
    context = SimpleNamespace(
        output_dir=tmp_path,
        e1=SimpleNamespace(selected=(question,)),
    )
    qa_path = runner.e6_phase_path(context, "qa", question)
    qa_path.parent.mkdir()
    qa_path.write_text("{}", encoding="utf-8")
    runner.write_json(
        tmp_path / "diagnostic.json",
        runner.seal_artifact({"qa": {"available": False}}),
    )

    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "run_extraction_phase",
        lambda *_args, **_kwargs: calls.append("extract"),
    )
    monkeypatch.setattr(
        runner,
        "run_rank_phase",
        lambda *_args, **_kwargs: calls.append("rank"),
    )
    monkeypatch.setattr(
        runner,
        "run_pack_phase",
        lambda *_args, **_kwargs: calls.append("pack"),
    )

    def unexpected_diagnostic(*_args, **_kwargs):
        raise AssertionError("partial QA resume must not rebuild the context diagnostic")

    monkeypatch.setattr(runner, "build_diagnostic_report", unexpected_diagnostic)

    def resume_qa(*_args, **_kwargs):
        calls.append("qa")
        return {"stage": "qa-resumed"}

    def final_report(*_args, **_kwargs):
        calls.append("report")
        return {"stage": "report"}

    monkeypatch.setattr(runner, "run_qa_phase", resume_qa)
    monkeypatch.setattr(runner, "build_report", final_report)
    args = SimpleNamespace(
        phase="all",
        limit=None,
        base_url="https://api.deepseek.com",
        api_key_env="FIXTURE_API_KEY",
        device=runner.E6B_QWEN_DEVICE,
        qwen_batch_size=runner.E6B_QWEN_BATCH_SIZE,
    )

    assert runner._execute_run(context, args) == 0
    assert calls == ["extract", "rank", "pack", "qa", "report"]
    assert "qa-resumed" in capsys.readouterr().out


def test_concurrent_extraction_preserves_cost_cap_exception(monkeypatch, tmp_path) -> None:
    question = _question(position=7, question_id="cost-cap")
    context = SimpleNamespace(
        output_dir=tmp_path,
        manifest={"extraction": {"concurrency": 2}},
    )

    class Client:
        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(runner, "_ExtractionChatClient", lambda **_kwargs: Client())
    monkeypatch.setattr(
        runner,
        "_reconcile_extraction_journals",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(runner, "_external_journal_cost", lambda _context: (0, 0))
    monkeypatch.setattr(runner, "_source_values", lambda *_args: (object(), object()))

    async def fail_one(*_args, **_kwargs):
        raise runner.ExternalCostCapExceeded("fixture cap")

    monkeypatch.setattr(runner, "_extract_one_value", fail_one)
    with pytest.raises(runner.ExternalCostCapExceeded, match="fixture cap"):
        asyncio.run(
            runner._run_extraction_async(
                context,
                selected=(question,),
                tokenizer=SimpleNamespace(),
                base_url="https://api.deepseek.com",
                api_key="fixture",
            )
        )
