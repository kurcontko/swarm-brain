from __future__ import annotations

import argparse
import asyncio
import signal

from swarmbrain.application.runtime import build_runtime
from swarmbrain.config import ApiSettings, BackendKind, WorkerSettings
from swarmbrain.workers.runtime import WorkerSupervisor


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run durable Swarm Brain workers")
    parser.add_argument("--once", action="store_true", help="process one bounded queue cycle")
    return parser.parse_args()


async def _run(*, once: bool) -> None:
    api_settings = ApiSettings.from_env()
    if api_settings.backend is not BackendKind.COCKROACH:
        raise RuntimeError("swarmbrain-worker requires SWARMBRAIN_BACKEND=cockroach")
    supervisor = WorkerSupervisor(build_runtime(api_settings), WorkerSettings.from_env())
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    await supervisor.serve(stop, once=once)


def main() -> None:
    arguments = _arguments()
    asyncio.run(_run(once=arguments.once))


if __name__ == "__main__":
    main()
