"""Run the durable embedding worker against the configured backend."""

from __future__ import annotations

import asyncio
import os
import socket

from swarmbrain.application.runtime import build_runtime
from swarmbrain.config import ApiSettings


def main() -> None:
    asyncio.run(_run())


async def _run() -> None:
    settings = ApiSettings.from_env()
    runtime = build_runtime(settings)
    worker = runtime.embedding_worker()
    worker_id = os.getenv("SWARMBRAIN_WORKER_ID") or f"embed-{socket.gethostname()}-{os.getpid()}"
    poll_seconds = float(os.getenv("SWARMBRAIN_WORKER_POLL_SECONDS", "2"))
    await runtime.start()
    try:
        while True:
            completed = await worker.run_once(worker_id)
            if not completed:
                await asyncio.sleep(poll_seconds)
    finally:
        await runtime.close()


if __name__ == "__main__":
    main()


__all__ = ["main"]
