#!/usr/bin/env python3
"""Fail when a repository-local Markdown link points at a missing path."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "data:", "app://")


def markdown_files() -> tuple[Path, ...]:
    files = [ROOT / "README.md"]
    files.extend(sorted((ROOT / "docs").rglob("*.md")))
    return tuple(path for path in files if path.is_file())


def local_target(source: Path, raw: str) -> Path | None:
    value = raw.strip().split(maxsplit=1)[0].strip("<>")
    if not value or value.startswith(("#", *EXTERNAL_PREFIXES)):
        return None
    path_value = unquote(value.split("#", 1)[0])
    if not path_value:
        return None
    return (source.parent / path_value).resolve()


def main() -> int:
    failures: list[str] = []
    files = markdown_files()
    for source in files:
        text = source.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            target = local_target(source, match.group(1))
            if target is None or target.exists():
                continue
            line = text.count("\n", 0, match.start()) + 1
            try:
                display = target.relative_to(ROOT)
            except ValueError:
                display = target
            failures.append(f"{source.relative_to(ROOT)}:{line}: missing {display}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"checked {len(files)} Markdown files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
