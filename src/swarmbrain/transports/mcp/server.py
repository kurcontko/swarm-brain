"""Exactly six model-visible tools; all behavior remains in the API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from swarmbrain.config import BridgeSettings
from swarmbrain.domain.common import MemoryContent

from .client import SwarmBrainHttpClient


def create_server(
    settings: BridgeSettings,
    *,
    http_client: SwarmBrainHttpClient | None = None,
) -> FastMCP:
    holder: dict[str, SwarmBrainHttpClient] = {}

    @asynccontextmanager
    async def lifespan(_: FastMCP) -> AsyncIterator[None]:
        client = http_client or SwarmBrainHttpClient(settings)
        holder["client"] = client
        try:
            await client.join()
            yield
        finally:
            holder.pop("client", None)
            if http_client is None:
                await client.close()

    server = FastMCP(
        "swarmbrain",
        instructions=(
            "Shared task coordination and temporal memory. Actor scope and "
            "permissions are derived from the local bridge token."
        ),
        lifespan=lifespan,
    )

    def client() -> SwarmBrainHttpClient:
        try:
            return holder["client"]
        except KeyError as exc:
            raise RuntimeError("MCP bridge lifespan is not active") from exc

    @server.tool()
    async def claim_task(
        idempotency_key: str,
        task_id: str | None = None,
        required_tags: list[str] | None = None,
        required_capabilities: list[str] | None = None,
        lease_seconds: int = 120,
        expected_task_version: int | None = None,
    ) -> dict[str, Any]:
        """Claim one eligible task and return its lease, checkpoint, and memory."""

        return await client().claim_task(
            task_id=task_id,
            required_tags=required_tags,
            required_capabilities=required_capabilities,
            lease_seconds=lease_seconds,
            expected_task_version=expected_task_version,
            idempotency_key=idempotency_key,
        )

    @server.tool()
    async def recall_memory(
        text: str,
        task_id: str | None = None,
        kinds: list[str] | None = None,
        visibilities: list[str] | None = None,
        states: list[str] | None = None,
        include_refuted: bool = False,
        include_superseded: bool = False,
        world_at: str | None = None,
        recorded_at: str | None = None,
        min_score: float = 0.0,
        limit: int = 10,
        include_evidence: bool = True,
        include_lineage: bool = False,
    ) -> dict[str, Any]:
        """Recall scoped current memory; historical/refuted rows are opt-in."""

        return await client().recall_memory(
            text=text,
            task_id=task_id,
            kinds=kinds,
            visibilities=visibilities,
            states=states,
            include_refuted=include_refuted,
            include_superseded=include_superseded,
            world_at=world_at,
            recorded_at=recorded_at,
            min_score=min_score,
            limit=limit,
            include_evidence=include_evidence,
            include_lineage=include_lineage,
        )

    @server.tool()
    async def publish_memory(
        idempotency_key: str,
        content: MemoryContent,
        kind: str = "observation",
        desired_state: str = "tentative",
        visibility: str = "run",
        task_id: str | None = None,
        title: str | None = None,
        tags: list[str] | None = None,
        evidence: list[dict[str, Any]] | None = None,
        supersedes_memory_id: str | None = None,
        related_memory_ids: list[str] | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a flexible, source-preserving temporal memory proposal."""

        return await client().publish_memory(
            content=content,
            kind=kind,
            desired_state=desired_state,
            visibility=visibility,
            task_id=task_id,
            title=title,
            tags=tags,
            evidence=evidence,
            supersedes_memory_id=supersedes_memory_id,
            related_memory_ids=related_memory_ids,
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=confidence,
            metadata=metadata,
            idempotency_key=idempotency_key,
        )

    @server.tool()
    async def checkpoint_task(
        idempotency_key: str,
        task_id: str,
        summary: str,
        discoveries: list[str] | None = None,
        completed_work: list[str] | None = None,
        remaining_work: list[str] | None = None,
        memory_ids: list[str] | None = None,
        artifact_ids: list[str] | None = None,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist durable handoff state for the bridge-owned current lease."""

        return await client().checkpoint_task(
            task_id=task_id,
            summary=summary,
            discoveries=discoveries,
            completed_work=completed_work,
            remaining_work=remaining_work,
            memory_ids=memory_ids,
            artifact_ids=artifact_ids,
            state=state,
            idempotency_key=idempotency_key,
        )

    @server.tool()
    async def complete_task(
        idempotency_key: str,
        task_id: str,
        summary: str,
        outcome: str = "succeeded",
        memory_ids: list[str] | None = None,
        artifact_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Complete the bridge-owned current lease exactly once."""

        return await client().complete_task(
            task_id=task_id,
            summary=summary,
            outcome=outcome,
            memory_ids=memory_ids,
            artifact_ids=artifact_ids,
            idempotency_key=idempotency_key,
        )

    @server.tool()
    async def report_conflict(
        idempotency_key: str,
        memory_ids: list[str],
        description: str,
        evidence: list[dict[str, Any]] | None = None,
        task_id: str | None = None,
        severity: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Report contradictory memories for later evidence-backed resolution."""

        return await client().report_conflict(
            memory_ids=memory_ids,
            description=description,
            evidence=evidence,
            task_id=task_id,
            severity=severity,
            metadata=metadata,
            idempotency_key=idempotency_key,
        )

    return server


def main() -> None:
    create_server(BridgeSettings.from_env()).run(transport="stdio")


if __name__ == "__main__":
    main()


__all__ = ["create_server", "main"]
