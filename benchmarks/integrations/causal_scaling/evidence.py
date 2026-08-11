"""Public-runtime evidence readers and activation/citation projection."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

import httpx

from .contracts import (
    CausalScalingError,
    ExecutionResult,
    MemoryUseProof,
    RolloutRequest,
    RuntimeEventEnvelope,
    sha256_json,
)

PUBLIC_EVENT_SOURCE = "swarmbrain_public_run_events_v1"
TokenResolver = Callable[[RolloutRequest, ExecutionResult], str | Awaitable[str]]


class PublicHttpRuntimeEvidenceReader:
    """Page the ordinary authenticated run-event endpoint to exhaustion."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        token_resolver: TokenResolver,
        *,
        page_limit: int = 200,
        max_pages: int = 10_000,
    ) -> None:
        if not 1 <= page_limit <= 200:
            raise CausalScalingError("event page_limit must be in [1, 200]")
        if max_pages < 1:
            raise CausalScalingError("event max_pages must be positive")
        self._client = client
        self._token_resolver = token_resolver
        self._page_limit = page_limit
        self._max_pages = max_pages

    async def read_events(
        self,
        request: RolloutRequest,
        result: ExecutionResult,
    ) -> RuntimeEventEnvelope:
        token_value = self._token_resolver(request, result)
        token = await token_value if inspect.isawaitable(token_value) else token_value
        if not isinstance(token, str) or not token.strip():
            raise CausalScalingError("event token resolver returned an empty token")

        events: list[dict[str, Any]] = []
        cursor: str | None = None
        for page_number in range(1, self._max_pages + 1):
            params: dict[str, Any] = {"limit": self._page_limit}
            if cursor is not None:
                params["cursor"] = cursor
            response = await self._client.get(
                f"/v1/runs/{result.run_id}/events",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
            page = payload.get("events")
            if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
                raise CausalScalingError("public run-event response has invalid events")
            events.extend(page)
            next_cursor = payload.get("next_cursor")
            if next_cursor is None:
                return RuntimeEventEnvelope(
                    source=PUBLIC_EVENT_SOURCE,
                    run_id=result.run_id,
                    complete=True,
                    page_count=page_number,
                    events=tuple(events),
                )
            if not isinstance(next_cursor, str) or not next_cursor:
                raise CausalScalingError("public run-event response has an invalid cursor")
            cursor = next_cursor
        raise CausalScalingError("public run-event pagination did not terminate")


def project_memory_use(
    envelope: RuntimeEventEnvelope,
    *,
    request: RolloutRequest,
    result: ExecutionResult,
) -> MemoryUseProof:
    """Prove use by an exact task/lease/consumer/memory event intersection.

    A recall result alone is never evidence of use.  The same memory ID must be
    present in a durable ``memory.activated`` event and a checkpoint/completion
    citation for the same task, lease, and consuming agent.
    """

    if envelope.source != PUBLIC_EVENT_SOURCE:
        raise CausalScalingError("runtime evidence did not come from the public event surface")
    if not envelope.complete:
        raise CausalScalingError("runtime event stream is incomplete")
    if envelope.run_id != result.run_id:
        raise CausalScalingError("runtime event stream belongs to another rollout")

    activation_keys: set[tuple[str, str, str, str]] = set()
    citation_keys: set[tuple[str, str, str, str]] = set()
    activation_ids: list[str] = []
    citation_ids: list[str] = []
    event_ids: set[str] = set()

    for event in envelope.events:
        event_id = _required_string(event, "event_id")
        if event_id in event_ids:
            raise CausalScalingError("runtime event stream contains duplicate event IDs")
        event_ids.add(event_id)
        if _required_string(event, "run_id") != result.run_id:
            raise CausalScalingError("runtime event stream mixes run IDs")
        if event.get("task_id") != request.task.task_id:
            continue
        event_type = event.get("event_type")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise CausalScalingError("runtime task event payload must be an object")
        if event_type == "memory.activated":
            activation_ids.append(event_id)
            activation_keys.update(_memory_keys(event, payload))
        elif event_type in {"task.checkpointed", "task.completed"}:
            keys = _memory_keys(event, payload)
            if keys:
                citation_ids.append(event_id)
                citation_keys.update(keys)

    matched = activation_keys & citation_keys
    matched_memory_ids = tuple(sorted({key[3] for key in matched}))
    absence = not activation_ids and not citation_keys
    if request.memory_enabled:
        if not activation_ids or not citation_ids or not matched:
            raise CausalScalingError(
                "memory-enabled rollout lacks a matching activation and durable citation"
            )
    elif not absence:
        raise CausalScalingError(
            "no-memory rollout contains a memory activation or durable memory citation"
        )

    return MemoryUseProof(
        source=envelope.source,
        event_stream_sha256=sha256_json(asdict(envelope)),
        activation_event_ids=tuple(sorted(activation_ids)),
        citation_event_ids=tuple(sorted(citation_ids)),
        matched_memory_ids=matched_memory_ids,
        matched_activation_citations=len(matched),
        memory_absence_proven=absence,
    )


def _memory_keys(event: dict[str, Any], payload: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    task_id = _required_string(event, "task_id")
    agent_id = _required_string(event, "agent_id")
    lease_id = _required_string(payload, "lease_id")
    memory_ids = payload.get("memory_ids", [])
    if not isinstance(memory_ids, list) or any(
        not isinstance(memory_id, str) or not memory_id for memory_id in memory_ids
    ):
        raise CausalScalingError("runtime memory_ids must be a list of non-empty strings")
    if len(memory_ids) != len(set(memory_ids)):
        raise CausalScalingError("runtime event contains duplicate memory IDs")
    return {(task_id, lease_id, agent_id, memory_id) for memory_id in memory_ids}


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise CausalScalingError(f"runtime event {key} must be a non-empty string")
    return result


__all__ = [
    "PUBLIC_EVENT_SOURCE",
    "PublicHttpRuntimeEvidenceReader",
    "TokenResolver",
    "project_memory_use",
]
