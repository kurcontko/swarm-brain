#!/usr/bin/env python3
"""Run the pinned official LongMemEval GPT-4o judge with raw receipts.

The upstream evaluator intentionally emits only ``{model, label}``, discarding
the provider response and usage.  This compatibility runner uses the same
prompt function, model, temperature, max-token limit, label rule, and output
shape while additionally writing an exact request/prompt/response receipt
sidecar without retaining the API credential.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _longmemeval_common import (
    default_longmemeval_path,
    ensure_longmemeval,
)
from run_longmemeval_qa import (
    LONGMEMEVAL_S_QUESTION_COUNT,
    OFFICIAL_JUDGE_MODEL,
    ChatClient,
    ChatResult,
    artifact_identity,
    chat_receipt_record,
    is_abstention_question,
    judge_label,
    judge_prompt,
)

OFFICIAL_BASE_URL = "https://api.openai.com"
OFFICIAL_TEMPERATURE = 0.0
OFFICIAL_MAX_TOKENS = 10


class OfficialJudgeRunError(ValueError):
    """Inputs cannot support the frozen official-judge protocol."""


@dataclass(frozen=True, slots=True)
class OfficialJudgeOutcome:
    question_id: str
    hypothesis: str
    result: ChatResult
    label: bool


def _reject_constant(value: str) -> None:
    del value
    raise OfficialJudgeRunError("non-standard JSON constants are forbidden")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise OfficialJudgeRunError("duplicate JSON fields are forbidden")
        output[key] = value
    return output


def _strict_loads(value: str, *, label: str) -> Any:
    try:
        return json.loads(
            value,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (json.JSONDecodeError, OfficialJudgeRunError, ValueError):
        raise OfficialJudgeRunError(f"{label} is malformed JSON") from None


def load_hypotheses(path: Path) -> tuple[dict[str, str], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise OfficialJudgeRunError(
            f"cannot read hypothesis artifact: {type(exc).__name__}"
        ) from exc
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise OfficialJudgeRunError(f"hypothesis artifact has an empty line at {line_number}")
        value = _strict_loads(line, label=f"hypothesis line {line_number}")
        if not isinstance(value, dict) or set(value) != {"question_id", "hypothesis"}:
            raise OfficialJudgeRunError("hypothesis fields differ from the official schema")
        question_id = value.get("question_id")
        hypothesis = value.get("hypothesis")
        if (
            not isinstance(question_id, str)
            or not question_id
            or question_id != question_id.strip()
        ):
            raise OfficialJudgeRunError("hypothesis question_id is invalid")
        if not isinstance(hypothesis, str):
            raise OfficialJudgeRunError("hypothesis must be text")
        if question_id in seen:
            raise OfficialJudgeRunError(f"hypothesis artifact repeats {question_id!r}")
        seen.add(question_id)
        records.append({"question_id": question_id, "hypothesis": hypothesis})
    return tuple(records)


def validate_reference_records(
    references: Any,
    hypotheses: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(references, list):
        raise OfficialJudgeRunError("LongMemEval reference artifact must be a JSON array")
    indexed: dict[str, dict[str, Any]] = {}
    required = {"question_id", "question_type", "question", "answer"}
    for record in references:
        if not isinstance(record, dict) or not required <= set(record):
            raise OfficialJudgeRunError("LongMemEval reference record is malformed")
        question_id = record.get("question_id")
        if (
            not isinstance(question_id, str)
            or not question_id
            or question_id != question_id.strip()
        ):
            raise OfficialJudgeRunError("LongMemEval reference question_id is invalid")
        if question_id in indexed:
            raise OfficialJudgeRunError(f"LongMemEval references repeat {question_id!r}")
        for field in ("question_type", "question", "answer"):
            if not isinstance(record.get(field), str):
                raise OfficialJudgeRunError(f"LongMemEval reference {field} must be text")
        indexed[question_id] = record
    hypothesis_ids = {record["question_id"] for record in hypotheses}
    if hypothesis_ids != set(indexed):
        raise OfficialJudgeRunError("hypothesis coverage differs from the reference dataset")
    return indexed


async def run_official_judge(
    references: Any,
    hypotheses: Sequence[Mapping[str, str]],
    *,
    client: ChatClient,
    concurrency: int,
) -> tuple[OfficialJudgeOutcome, ...]:
    """Judge hypotheses with the exact upstream prompt and Boolean rule."""

    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise OfficialJudgeRunError("concurrency must be a positive integer")
    indexed = validate_reference_records(references, hypotheses)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(hypothesis_record: Mapping[str, str]) -> OfficialJudgeOutcome:
        question_id = hypothesis_record["question_id"]
        hypothesis = hypothesis_record["hypothesis"]
        reference = indexed[question_id]
        prompt = judge_prompt(
            reference["question_type"],
            reference["question"],
            reference["answer"],
            hypothesis,
            abstention=is_abstention_question(question_id),
        )
        async with semaphore:
            result = await client.complete(prompt)
        return OfficialJudgeOutcome(
            question_id=question_id,
            hypothesis=hypothesis,
            result=result,
            label=judge_label(result.content),
        )

    outcomes = tuple(await asyncio.gather(*(one(record) for record in hypotheses)))
    request_ids = [outcome.result.request_id for outcome in outcomes]
    if any(request_id is None for request_id in request_ids):
        raise OfficialJudgeRunError("official judge response has no provider request ID")
    if len(set(request_ids)) != len(request_ids):
        raise OfficialJudgeRunError("official judge provider request IDs are not unique")
    return outcomes


def official_label_records(
    outcomes: Sequence[OfficialJudgeOutcome],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "question_id": outcome.question_id,
            "hypothesis": outcome.hypothesis,
            "autoeval_label": {
                "model": OFFICIAL_JUDGE_MODEL,
                "label": outcome.label,
            },
        }
        for outcome in outcomes
    )


def official_receipt_records(
    outcomes: Sequence[OfficialJudgeOutcome],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        chat_receipt_record(outcome.question_id, "official_judge", outcome.result)
        for outcome in outcomes
    )


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for record in records
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--lme-path", type=Path, default=default_longmemeval_path())
    parser.add_argument("--lme-download", action="store_true")
    parser.add_argument("--out-labels", type=Path, required=True)
    parser.add_argument("--out-receipts", type=Path, required=True)
    parser.add_argument(
        "--api-key-env",
        default="LONGMEMEVAL_OFFICIAL_JUDGE_API_KEY",
        help=(
            "environment variable holding the OpenAI judge credential; the dedicated "
            "default prevents forwarding another provider's OPENAI_API_KEY"
        ),
    )
    parser.add_argument("--concurrency", type=int, default=8)
    return parser


async def _main(args: argparse.Namespace) -> int:
    dataset = ensure_longmemeval(args.lme_path, download=args.lme_download)
    try:
        references = _strict_loads(
            dataset.read_text(encoding="utf-8"),
            label="LongMemEval reference artifact",
        )
    except (OSError, UnicodeError) as exc:
        raise OfficialJudgeRunError(
            f"cannot read LongMemEval reference artifact: {type(exc).__name__}"
        ) from exc
    hypotheses = load_hypotheses(args.hypotheses)
    if len(hypotheses) != LONGMEMEVAL_S_QUESTION_COUNT:
        raise OfficialJudgeRunError(
            f"official judging requires exactly {LONGMEMEVAL_S_QUESTION_COUNT} hypotheses"
        )
    key = os.getenv(args.api_key_env, "").strip()
    if not key:
        raise OfficialJudgeRunError(
            f"official judge API key environment variable {args.api_key_env!r} is unset"
        )
    client = ChatClient(
        base_url=OFFICIAL_BASE_URL,
        model=OFFICIAL_JUDGE_MODEL,
        api_key=key,
        temperature=OFFICIAL_TEMPERATURE,
        max_tokens=OFFICIAL_MAX_TOKENS,
        required_response_model=OFFICIAL_JUDGE_MODEL,
        require_request_id=True,
    )
    try:
        outcomes = await run_official_judge(
            references,
            hypotheses,
            client=client,
            concurrency=args.concurrency,
        )
    finally:
        await client.aclose()
    labels = official_label_records(outcomes)
    receipts = official_receipt_records(outcomes)
    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    args.out_receipts.parent.mkdir(parents=True, exist_ok=True)
    args.out_labels.write_bytes(_jsonl_bytes(labels))
    args.out_receipts.write_bytes(_jsonl_bytes(receipts))
    label_identity = artifact_identity(args.out_labels)
    receipt_identity = artifact_identity(args.out_receipts)
    total_tokens = sum(outcome.result.total_tokens for outcome in outcomes)
    accuracy = sum(outcome.label for outcome in outcomes) / len(outcomes)
    print(
        f"wrote {label_identity['path']} and {receipt_identity['path']}; "
        f"official accuracy={accuracy:.4f}, provider tokens={total_tokens}"
    )
    return 0


def main() -> int:
    try:
        return asyncio.run(_main(_parser().parse_args()))
    except OfficialJudgeRunError as exc:
        print(f"cannot run official LongMemEval judge: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OFFICIAL_BASE_URL",
    "OFFICIAL_MAX_TOKENS",
    "OFFICIAL_TEMPERATURE",
    "OfficialJudgeOutcome",
    "OfficialJudgeRunError",
    "load_hypotheses",
    "official_label_records",
    "official_receipt_records",
    "run_official_judge",
    "validate_reference_records",
]
