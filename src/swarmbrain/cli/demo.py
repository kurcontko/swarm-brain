"""Run the scripted swarm demo end to end and write an evidence artifact.

Zero-configuration default: an in-process in-memory backend with a controllable
clock, so crash handoff is demonstrated without waiting out a real lease.
With ``SWARMBRAIN_BACKEND=cockroach`` the same beats run against the durable
backend over the same HTTP application, and lease expiry elapses in real time.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx

from swarmbrain.application.runtime import SwarmBrainRuntime, build_in_memory_runtime
from swarmbrain.config import ApiSettings, BackendKind
from swarmbrain.demo import DemoReport, DemoRunner, build_scenario


class ControllableClock:
    """A ticking UTC clock the demo can jump forward to expire leases instantly.

    Each read advances one millisecond so bitemporal writes that order
    intervals strictly (supersession stamps ``valid_from`` just after "now")
    become visible to the next read, exactly as they do under a real clock.
    """

    def __init__(self) -> None:
        self._now = datetime.now(UTC)

    def __call__(self) -> datetime:
        self._now += timedelta(milliseconds=1)
        return self._now

    def advance_past(self, moment: datetime) -> None:
        if self._now <= moment:
            self._now = moment + timedelta(seconds=1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Swarm Brain demo scenario over the canonical HTTP API"
    )
    parser.add_argument(
        "--evidence-dir",
        default="evidence",
        help="directory for the JSON evidence artifact (default: evidence/)",
    )
    parser.add_argument(
        "--lease-seconds",
        type=int,
        default=None,
        help=(
            "work lease duration; defaults to 120 with the controllable clock "
            "and 20 when lease expiry must elapse in real time"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="suppress narration lines")
    return parser


def _compose() -> tuple[SwarmBrainRuntime, ControllableClock | None]:
    import os

    backend = (os.getenv("SWARMBRAIN_BACKEND") or "memory").strip().casefold()
    if backend == "memory":
        secret = os.getenv("SWARMBRAIN_TOKEN_SECRET") or f"swarmbrain-demo-{uuid4()}"
        clock = ControllableClock()
        return build_in_memory_runtime(secret, clock=clock), clock
    from swarmbrain.application.runtime import build_runtime

    settings = ApiSettings.from_env()
    if settings.backend is not BackendKind.COCKROACH:
        raise SystemExit(f"unsupported demo backend: {backend}")
    return build_runtime(settings), None


async def _run(args: argparse.Namespace) -> DemoReport:
    from swarmbrain.transports.http import create_app

    runtime, clock = _compose()
    if runtime.coordination_store is None:
        raise SystemExit("the composed runtime does not expose a coordination store")

    if clock is not None:

        async def expire(expires_at: datetime) -> None:
            clock.advance_past(expires_at)

        lease_seconds = args.lease_seconds or 120
    else:

        async def expire(expires_at: datetime) -> None:
            remaining = (expires_at - datetime.now(UTC)).total_seconds() + 1.0
            if remaining > 0:
                print(f"waiting {remaining:.0f}s for the crashed worker's lease to expire...")
                await asyncio.sleep(remaining)

        lease_seconds = args.lease_seconds or 20

    narrate = (lambda _line: None) if args.quiet else print
    app = create_app(runtime)
    await runtime.start()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://swarm-demo") as client:
            runner = DemoRunner(
                client=client,
                tokens=runtime.tokens,
                store=runtime.coordination_store,
                scenario=build_scenario(),
                expire_leases=expire,
                narrate=narrate,
                lease_seconds=lease_seconds,
                now=clock if clock is not None else None,
            )
            return await runner.run()
    finally:
        await runtime.close()


def main() -> None:
    args = _parser().parse_args()
    report = asyncio.run(_run(args))
    evidence_dir = Path(args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact = evidence_dir / f"{stamp}-swarm-demo.json"
    artifact.write_text(report.to_json() + "\n", encoding="utf-8")
    verdict = "PASSED" if report.ok else "FAILED"
    print(f"\ndemo {verdict}; evidence written to {artifact}")
    if not report.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
