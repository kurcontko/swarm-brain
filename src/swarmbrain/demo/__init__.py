"""Deterministic swarm demo scenario, HTTP-level runner, and A/B comparison.

The demo drives the canonical HTTP API — never application services directly —
so every beat it proves is a beat the production transport supports. Simulated
agents speak the same protocol a real Claude Code, Codex, Gemini, or Qwen
worker speaks through the MCP bridge; the roster labels record the harness mix
the scenario stands in for.

``ab`` adds the measured-versus-simulated comparison: Arm B is the real run,
Arm A is an explicitly labelled model of the same fleet with nothing shared.
"""

from .ab import (
    BASELINE_KIND,
    SWARM_KIND,
    WORKLOAD,
    ABReport,
    ArmMetrics,
    Workload,
    WorkloadTask,
    build_workload,
    render_table,
    run_comparison,
    simulate_uncoordinated_baseline,
)
from .runner import BeatCheck, BeatReport, DemoAssertionError, DemoReport, DemoRunner
from .scenario import DemoAgent, DemoScenario, DemoTask, actor_context, build_scenario

__all__ = [
    "BASELINE_KIND",
    "SWARM_KIND",
    "WORKLOAD",
    "ABReport",
    "ArmMetrics",
    "BeatCheck",
    "BeatReport",
    "DemoAgent",
    "DemoAssertionError",
    "DemoReport",
    "DemoRunner",
    "DemoScenario",
    "DemoTask",
    "Workload",
    "WorkloadTask",
    "actor_context",
    "build_scenario",
    "build_workload",
    "render_table",
    "run_comparison",
    "simulate_uncoordinated_baseline",
]
