from __future__ import annotations

import argparse
import asyncio
import os

from swarmbrain.adapters.cockroach.schema import install_schema, verify_schema


def _database_url() -> str:
    value = os.getenv("SWARMBRAIN_DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError("SWARMBRAIN_DATABASE_URL is required")
    return value


async def _run(action: str) -> None:
    database_url = _database_url()
    if action == "install":
        count = await install_schema(database_url)
        await verify_schema(database_url)
        print(f"installed and verified Swarm Brain schema ({count} statements)")
        return
    await verify_schema(database_url)
    print("Swarm Brain schema is installed and current")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="swarmbrain-schema",
        description="Explicitly install or verify the Swarm Brain CockroachDB schema.",
    )
    parser.add_argument("action", choices=("install", "verify"))
    args = parser.parse_args()
    asyncio.run(_run(args.action))


if __name__ == "__main__":
    main()
