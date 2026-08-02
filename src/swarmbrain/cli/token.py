from __future__ import annotations

import argparse
import os
from datetime import timedelta

from swarmbrain.adapters.auth import RunTokenCodec
from swarmbrain.domain.agents import ActorContext


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Issue a local Swarm Brain run token")
    parser.add_argument("--tenant", required=True, help="tenant UUID")
    parser.add_argument("--project", required=True, help="project UUID")
    parser.add_argument("--repository", required=True, help="repository UUID")
    parser.add_argument("--swarm", required=True, help="swarm UUID")
    parser.add_argument("--run", required=True, help="run UUID")
    parser.add_argument("--agent", required=True, help="agent UUID")
    parser.add_argument("--harness", default="codex-cli")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--capability", action="append", default=[])
    parser.add_argument("--ttl-seconds", type=int, default=1800)
    return parser


def main() -> None:
    args = _parser().parse_args()
    secret = os.getenv("SWARMBRAIN_TOKEN_SECRET")
    if not secret:
        raise SystemExit("SWARMBRAIN_TOKEN_SECRET is required")
    actor = ActorContext(
        tenant_id=args.tenant,
        project_id=args.project,
        repository_id=args.repository,
        swarm_id=args.swarm,
        run_id=args.run,
        agent_id=args.agent,
        harness=args.harness,
        provider=args.provider,
        model=args.model,
        capabilities=frozenset(args.capability),
    )
    print(RunTokenCodec(secret).issue(actor, ttl=timedelta(seconds=args.ttl_seconds)))


if __name__ == "__main__":
    main()
