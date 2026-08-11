#!/usr/bin/env python3
"""Recompute a Mem2Act report offline from a repository-local bound run artifact."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.integrations.mem2act.report import (
    Mem2ActReportError,
    compile_mem2act_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="repository-local *-run.json produced by run_mem2act_bench.py",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="repository-local directory containing the three pinned official dataset files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace an existing report")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        compile_mem2act_report(
            args.run,
            args.output,
            dataset_dir=args.dataset_dir,
            artifact_root=REPO_ROOT,
            code_root=REPO_ROOT,
            enforce_repository_local=True,
            overwrite=args.force,
        )
    except (FileExistsError, Mem2ActReportError, OSError) as exc:
        raise SystemExit(f"Mem2Act offline compilation failed closed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
