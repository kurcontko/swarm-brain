#!/usr/bin/env python3
"""Recompute a paired LongMemEval-S reranker report from raw core traces."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from benchmarks.integrations.longmemeval_reranker import (
    LongMemEvalRerankerEvidenceError,
    compile_longmemeval_reranker_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="repository-local paired run manifest",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="byte-pinned official longmemeval_s_cleaned.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace an existing report")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        compile_longmemeval_reranker_report(
            args.run,
            args.dataset,
            args.output,
            artifact_root=REPO_ROOT,
            code_root=REPO_ROOT,
            overwrite=args.force,
        )
    except (FileExistsError, LongMemEvalRerankerEvidenceError, OSError) as exc:
        raise SystemExit(f"LongMemEval reranker compilation failed closed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
