#!/usr/bin/env python3
"""Compile frozen paired LongMemEval selection QA evidence without external calls."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from benchmarks.integrations.longmemeval_selection_report import (
    LongMemEvalSelectionEvidenceError,
    compile_longmemeval_selection_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="repository-local run manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--heldout-confirmation",
        type=Path,
        help="optional separately preregistered held-out confirmation manifest",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing report")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        compile_longmemeval_selection_report(
            args.run,
            args.output,
            artifact_root=REPO_ROOT,
            code_root=REPO_ROOT,
            heldout_confirmation_path=args.heldout_confirmation,
            overwrite=args.force,
        )
    except (FileExistsError, LongMemEvalSelectionEvidenceError, OSError) as exc:
        raise SystemExit(f"LongMemEval selection QA compilation failed closed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
