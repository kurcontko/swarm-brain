#!/usr/bin/env python3
"""Compile four complete GateMem scorer runs into the SOTA gate report."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.integrations.gatemem.contracts import GateMemCheckout, GateMemContractError
from benchmarks.integrations.gatemem.report import DomainEvidence, build_gatemem_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate official GateMem outputs and build benchmarks/sota/gatemem-report.json"
    )
    parser.add_argument("--gatemem-dir", default="/private/tmp/swarmbrain-gatemem")
    parser.add_argument(
        "--evidence",
        action="append",
        nargs=4,
        metavar=("DOMAIN", "PREDICTIONS", "RUN_AUDIT", "SCORER_DIR"),
        required=True,
        help="repeat once for education, household, medical, and office",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "sota" / "gatemem-report.json",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        evidence = tuple(
            DomainEvidence.create(
                domain=domain,
                predictions_path=predictions,
                audit_path=audit,
                scorer_dir=scorer_dir,
            )
            for domain, predictions, audit, scorer_dir in args.evidence
        )
        report = build_gatemem_report(checkout=GateMemCheckout(args.gatemem_dir), evidence=evidence)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except GateMemContractError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "status": "gatemem_report_written",
                "output": str(args.output.resolve()),
                "episodes": report["dataset"]["episodes"],
                "checkpoints": report["dataset"]["checkpoints"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
