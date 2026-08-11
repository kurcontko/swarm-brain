"""Immutable runtime provenance for offline Mem2Act report compilation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .contracts import Mem2ActContractError
from .dataset import canonical_json

RUN_ARTIFACT_TYPE = "swarmbrain-mem2act-run"
RUN_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "mem2act-three-arm-two-condition-v1"

IMPLEMENTATION_FILES = (
    "pyproject.toml",
    "uv.lock",
    "benchmarks/integrations/mem2act/__init__.py",
    "benchmarks/integrations/mem2act/contracts.py",
    "benchmarks/integrations/mem2act/dataset.py",
    "benchmarks/integrations/mem2act/metrics.py",
    "benchmarks/integrations/mem2act/openai_reader.py",
    "benchmarks/integrations/mem2act/provenance.py",
    "benchmarks/integrations/mem2act/report.py",
    "benchmarks/integrations/mem2act/runner.py",
    "benchmarks/integrations/mem2act/runtime_bridge.py",
    "scripts/build_mem2act_report.py",
    "scripts/run_mem2act_bench.py",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def implementation_fingerprint(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Hash every implementation file that can affect raw rows or compilation."""

    root = (repo_root or repository_root()).resolve()
    relative_files = set(IMPLEMENTATION_FILES)
    relative_files.update(
        path.relative_to(root).as_posix() for path in (root / "src" / "swarmbrain").rglob("*.py")
    )
    files: dict[str, str] = {}
    for relative in sorted(relative_files):
        path = root
        unsafe = False
        for part in Path(relative).parts:
            path /= part
            unsafe = unsafe or path.is_symlink()
        if unsafe or not path.is_file():
            raise Mem2ActContractError(
                f"Mem2Act implementation file is missing or unsafe: {relative}"
            )
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "tree_sha256": hashlib.sha256(canonical_json(files).encode("utf-8")).hexdigest(),
        "files_sha256": files,
    }


__all__ = [
    "IMPLEMENTATION_FILES",
    "PROTOCOL_VERSION",
    "RUN_ARTIFACT_TYPE",
    "RUN_SCHEMA_VERSION",
    "implementation_fingerprint",
    "repository_root",
]
