#!/usr/bin/env python3
"""Evaluate saved retrieval-lane runs without model or database dependencies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from swarmbrain.retrieval.evaluation import evaluate_lanes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--k", type=int, default=10)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload: dict[str, Any] = json.loads(args.run.read_text(encoding="utf-8"))
    relevant = {
        str(case["case_id"]): tuple(str(value) for value in case.get("relevant_ids", ()))
        for case in payload["cases"]
    }
    lane_names = tuple(
        dict.fromkeys(
            str(lane) for case in payload["cases"] for lane in (case.get("rankings") or {})
        )
    )
    rankings = {
        lane: {
            str(case["case_id"]): tuple(
                str(value) for value in (case.get("rankings") or {}).get(lane, ())
            )
            for case in payload["cases"]
        }
        for lane in lane_names
    }
    metrics = evaluate_lanes(relevant, rankings, k=args.k)
    print(
        json.dumps(
            {lane: asdict(value) for lane, value in metrics.items()},
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
