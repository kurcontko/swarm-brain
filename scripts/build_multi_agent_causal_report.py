#!/usr/bin/env python3
"""Compile measured 1/2/4-agent raw evidence into the canonical SOTA report."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.integrations.causal_scaling import (
    CausalScalingError,
    build_causal_scaling_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate measured paired causal-scaling evidence and build "
            "benchmarks/sota/multi-agent-1-2-4-report.json"
        )
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        required=True,
        help="separately reviewed task/public-prompt/hidden-verifier digest manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "sota" / "multi-agent-1-2-4-report.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output only after the new evidence validates",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        report = build_causal_scaling_report(
            args.evidence,
            workload_manifest=args.workload_manifest,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if args.overwrite else "x"
        with args.output.open(mode, encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except (CausalScalingError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "status": "multi_agent_causal_report_written",
                "output": str(args.output.resolve()),
                "records": report["raw"]["record_count"],
                "clusters": report["raw"]["task_cluster_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
