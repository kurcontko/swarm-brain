"""Strict external-prediction integration for the GateMem benchmark."""

from .completion import (
    build_execution_lineage,
    default_completion_path,
    validate_completion_manifest,
    write_completion_manifest,
)
from .contracts import (
    GATEMEM_COMMIT,
    GateMemCheckout,
    PrincipalScope,
    PublicCheckpoint,
    PublicEpisode,
    PublicTurn,
    ScopeFactory,
)
from .resume import AuthenticatedResumeStore, EpisodeResumeSpec
from .runner import GateMemHarness, HarnessConfig, HarnessRun

__all__ = [
    "GATEMEM_COMMIT",
    "AuthenticatedResumeStore",
    "EpisodeResumeSpec",
    "GateMemCheckout",
    "GateMemHarness",
    "HarnessConfig",
    "HarnessRun",
    "PrincipalScope",
    "PublicCheckpoint",
    "PublicEpisode",
    "PublicTurn",
    "ScopeFactory",
    "build_execution_lineage",
    "default_completion_path",
    "validate_completion_manifest",
    "write_completion_manifest",
]
