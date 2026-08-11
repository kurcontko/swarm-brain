#!/usr/bin/env python3
"""Replay greedy versus facility-location packing on a saved LongMemEval run.

The saved run supplies the frozen fused ranking.  The pinned LongMemEval-S
release supplies candidate text beyond the public top ten, allowing both arms
to use the same 4x over-fetch as the runtime.  No retriever, embedding provider,
reader, or judge is called.  The report therefore measures answer evidence
surviving the packing boundary, not end-to-end QA accuracy.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from math import comb
from pathlib import Path
from typing import Any

from _longmemeval_common import (
    LONGMEMEVAL_S_SHA256,
    default_longmemeval_path,
    ensure_longmemeval,
    session_text,
)

from swarmbrain.domain.retrieval import PackingPolicy
from swarmbrain.retrieval import (
    RRF_K,
    build_packing_features,
    estimate_tokens,
    pack_to_budget,
    relevance_query,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO_ROOT / "benchmarks/retrieval/longmemeval-s-memory-openai-run.json"
DEFAULT_BUDGETS = (2048, 4096, 8192, 16384)
SIGNAL_WEIGHTS = {
    "exact": 5.0,
    "lexical": 3.0,
    "fuzzy": 1.0,
    "dense": 4.0,
    "graph": 0.5,
}


def _fused_relevance(case: dict[str, Any], candidate_id: str) -> float:
    """Reconstruct the public weighted-RRF score from persisted lane ranks."""

    raw = 0.0
    rankings = case.get("rankings") or {}
    for lane, weight in SIGNAL_WEIGHTS.items():
        lane_ranking = rankings.get(lane) or ()
        try:
            rank = lane_ranking.index(candidate_id) + 1
        except ValueError:
            continue
        raw += weight / (RRF_K + rank)
    anchor = max(SIGNAL_WEIGHTS.values()) / (RRF_K + 1)
    return min(1.0, max(0.0, raw / anchor))


def _candidate_text(record: dict[str, Any], key: str) -> tuple[str, str]:
    raw_position, session_id = key.split(":", 1)
    position = int(raw_position)
    expected_id = str(record["haystack_session_ids"][position])
    if expected_id != session_id:
        raise ValueError(f"saved key {key!r} does not match dataset session {expected_id!r}")
    dates = record.get("haystack_dates") or ()
    date = str(dates[position]) if position < len(dates) else ""
    title = f"Conversation session recorded {date}" if date else "Conversation session"
    content = session_text(record["haystack_sessions"][position]).strip() or "(empty session)"
    return f"{title}\n{content}", f"{title}\n{content}\nlongmemeval\nsession"


def _exact_binomial_p_value(wins: int, losses: int) -> float:
    """One-sided paired sign-test p-value under an equal win/loss null."""

    discordant = wins + losses
    if discordant == 0 or wins <= losses:
        return 1.0
    numerator = sum(comb(discordant, value) for value in range(wins, discordant + 1))
    return numerator / (2**discordant)


def _metrics(outcomes: Sequence[dict[str, Any]], policy: PackingPolicy) -> dict[str, Any]:
    answerable = [outcome for outcome in outcomes if outcome["gold"]]
    any_values = [bool(outcome[policy.value]["any_gold"]) for outcome in answerable]
    all_values = [bool(outcome[policy.value]["all_gold"]) for outcome in answerable]
    return {
        "answerable_cases": len(answerable),
        "any_gold_in_context": round(sum(any_values) / len(any_values), 4),
        "all_gold_in_context": round(sum(all_values) / len(all_values), 4),
        "mean_hits_kept": round(
            sum(outcome[policy.value]["kept"] for outcome in outcomes) / len(outcomes),
            4,
        ),
        "mean_tokens": round(
            sum(outcome[policy.value]["tokens"] for outcome in outcomes) / len(outcomes),
            1,
        ),
    }


def _paired_metrics(outcomes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [outcome for outcome in outcomes if outcome["gold"]]

    def compare(metric: str) -> dict[str, Any]:
        greedy = [bool(outcome[PackingPolicy.GREEDY.value][metric]) for outcome in answerable]
        facility = [
            bool(outcome[PackingPolicy.FACILITY_LOCATION.value][metric]) for outcome in answerable
        ]
        wins = sum(new and not old for old, new in zip(greedy, facility, strict=True))
        losses = sum(old and not new for old, new in zip(greedy, facility, strict=True))
        return {
            "delta": round((sum(facility) - sum(greedy)) / len(answerable), 4),
            "wins": wins,
            "losses": losses,
            "ties": len(answerable) - wins - losses,
            "one_sided_sign_test_p": round(_exact_binomial_p_value(wins, losses), 6),
        }

    return {"any_gold": compare("any_gold"), "all_gold": compare("all_gold")}


def evaluate(
    run: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    budgets: Sequence[int],
    limit: int,
    candidate_pool: int,
) -> dict[str, Any]:
    if run.get("track") != "longmemeval-s":
        raise ValueError("packing A/B requires a LongMemEval-S saved run")
    by_question = {str(record["question_id"]): record for record in records}
    prepared: list[dict[str, Any]] = []
    token_checks = 0
    for case in run["cases"]:
        case_id = str(case["case_id"])
        record = by_question.get(case_id)
        if record is None:
            raise ValueError(f"saved case {case_id!r} is absent from the pinned dataset")
        ranking = list((case.get("rankings") or {}).get("fused") or ())[:candidate_pool]
        query_terms = relevance_query(str(record["question"])).tokens
        sizes: list[int] = []
        features = []
        for candidate_id in ranking:
            reader_text, feature_text = _candidate_text(record, str(candidate_id))
            sizes.append(estimate_tokens(reader_text))
            features.append(
                build_packing_features(
                    feature_text,
                    query_terms=query_terms,
                    relevance=_fused_relevance(case, str(candidate_id)),
                    diversity_labels=(
                        "kind:observation",
                        "visibility:repository",
                        "tag:longmemeval",
                        "tag:session",
                    ),
                )
            )
        recorded_tokens = case.get("final_tokens") or ()
        if recorded_tokens:
            expected = sizes[: len(recorded_tokens)]
            if expected != [int(value) for value in recorded_tokens]:
                raise ValueError(f"token reconstruction drifted for case {case_id}")
            token_checks += 1
        prepared.append(
            {
                "ids": ranking,
                "sizes": sizes,
                "features": features,
                "gold": frozenset(str(value) for value in case.get("relevant_ids", ())),
            }
        )

    by_budget: dict[str, Any] = {}
    for budget in budgets:
        outcomes: list[dict[str, Any]] = []
        for case in prepared:
            outcome: dict[str, Any] = {"gold": case["gold"]}
            for policy in (PackingPolicy.GREEDY, PackingPolicy.FACILITY_LOCATION):
                packed = pack_to_budget(
                    case["sizes"],
                    budget,
                    policy=policy,
                    features=(
                        case["features"] if policy is PackingPolicy.FACILITY_LOCATION else None
                    ),
                    max_items=limit,
                )
                kept_ids = {case["ids"][index] for index in packed.kept_indices}
                gold = case["gold"]
                overlap = kept_ids & gold
                outcome[policy.value] = {
                    "any_gold": bool(overlap),
                    "all_gold": bool(gold) and overlap == gold,
                    "kept": len(packed.kept_indices),
                    "tokens": packed.used_tokens,
                }
            outcomes.append(outcome)
        by_budget[str(budget)] = {
            PackingPolicy.GREEDY.value: _metrics(outcomes, PackingPolicy.GREEDY),
            PackingPolicy.FACILITY_LOCATION.value: _metrics(
                outcomes, PackingPolicy.FACILITY_LOCATION
            ),
            "paired": _paired_metrics(outcomes),
        }

    activation_budgets = [value for value in (2048, 4096) if str(value) in by_budget]
    supported = len(activation_budgets) == 2 and all(
        by_budget[str(budget)]["paired"]["any_gold"]["delta"] >= 0.01
        and by_budget[str(budget)]["paired"]["any_gold"]["one_sided_sign_test_p"] < 0.05
        and by_budget[str(budget)]["paired"]["all_gold"]["delta"] >= 0.0
        for budget in activation_budgets
    )
    return {
        "schema_version": "packing-ab-v1",
        "cases": len(prepared),
        "candidate_pool": candidate_pool,
        "public_limit": limit,
        "token_estimator": "chars/4-ceil-v1",
        "token_reconstruction_checks": token_checks,
        "relevance_signal": "weighted-rrf reconstructed from saved lane ranks",
        "budgets": by_budget,
        "enablement_gate": {
            "supported": supported,
            "rule": (
                "At both 2048 and 4096 tokens: any-gold delta >= 0.01, paired "
                "one-sided sign-test p < 0.05, and all-gold does not regress."
            ),
            "decision": (
                "enable facility_location" if supported else "keep greedy as the runtime default"
            ),
        },
        "scope": (
            "Packing-only replay over a frozen full-500 ranking; this is not a "
            "LongMemEval QA score."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dataset", type=Path, default=default_longmemeval_path())
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=40)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if not 1 <= args.candidate_pool <= 128:
        raise SystemExit("--candidate-pool must be between 1 and 128")
    if any(budget < 1 for budget in args.budgets):
        raise SystemExit("--budgets must be positive")
    dataset = ensure_longmemeval(args.dataset, download=False)
    records = json.loads(dataset.read_text(encoding="utf-8"))
    run = json.loads(args.run.read_text(encoding="utf-8"))
    report = {
        "run": str(args.run),
        "dataset": {"path": str(dataset), "sha256": LONGMEMEVAL_S_SHA256},
        **evaluate(
            run,
            records,
            budgets=args.budgets,
            limit=args.limit,
            candidate_pool=args.candidate_pool,
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
