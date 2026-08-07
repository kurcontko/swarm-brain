"""Deterministic swarm demo scenario and HTTP-level runner.

The demo drives the canonical HTTP API — never application services directly —
so every beat it proves is a beat the production transport supports. Simulated
agents speak the same protocol a real Claude Code, Codex, Gemini, or Qwen
worker speaks through the MCP bridge; the roster labels record the harness mix
the scenario stands in for.
"""

from .runner import BeatCheck, BeatReport, DemoAssertionError, DemoReport, DemoRunner
from .scenario import DemoAgent, DemoScenario, DemoTask, build_scenario

__all__ = [
    "BeatCheck",
    "BeatReport",
    "DemoAgent",
    "DemoAssertionError",
    "DemoReport",
    "DemoRunner",
    "DemoScenario",
    "DemoTask",
    "build_scenario",
]
