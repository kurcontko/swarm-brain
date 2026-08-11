"""Strict 1/2/4-agent causal-scaling benchmark integration."""

from .contracts import (
    AgentUsage,
    CausalScalingError,
    CausalTask,
    EvaluationProvenance,
    ExecutionKind,
    ExecutionResult,
    ModelProfile,
    PublicTask,
    RuntimeEventEnvelope,
    ScoreResult,
    TotalBudget,
)
from .report import build_causal_scaling_report
from .runner import CausalScalingConfig, CausalScalingRunner, write_raw_evidence
from .workload import WorkloadManifest, WorkloadTaskEntry, load_workload_manifest

__all__ = [
    "AgentUsage",
    "CausalScalingConfig",
    "CausalScalingError",
    "CausalScalingRunner",
    "CausalTask",
    "EvaluationProvenance",
    "ExecutionKind",
    "ExecutionResult",
    "ModelProfile",
    "PublicTask",
    "RuntimeEventEnvelope",
    "ScoreResult",
    "TotalBudget",
    "WorkloadManifest",
    "WorkloadTaskEntry",
    "build_causal_scaling_report",
    "load_workload_manifest",
    "write_raw_evidence",
]
