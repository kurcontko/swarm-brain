from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx

from swarmbrain.config import BridgeSettings


class BridgeHttpError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        code: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retryable = retryable


@dataclass(slots=True)
class LeaseBinding:
    task_id: str
    lease_id: str
    task_version: int
    lease_version: int
    extension_seconds: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SwarmBrainHttpClient:
    """Thin HTTP client used by the stdio bridge.

    The bridge owns heartbeat mechanics. Authenticated scope exists only in
    the bearer token; no tenant/run/agent identity is copied into tool bodies.
    """

    def __init__(
        self,
        settings: BridgeSettings,
        *,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        renewal_interval_fraction: float = 1 / 3,
    ) -> None:
        if renewal_interval_fraction <= 0:
            raise ValueError("renewal_interval_fraction must be positive")
        self.settings = settings
        self._sleep = sleep
        self._renewal_interval_fraction = renewal_interval_fraction
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.api_url,
            timeout=settings.request_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.agent_token}"},
        )
        self._leases: dict[str, LeaseBinding] = {}
        self._renewals: dict[str, asyncio.Task[None]] = {}
        self._inline_evidence: dict[str, list[dict[str, Any]]] = {}

    async def close(self) -> None:
        renewals = list(self._renewals.values())
        self._renewals.clear()
        for renewal in renewals:
            renewal.cancel()
        if renewals:
            await asyncio.gather(*renewals, return_exceptions=True)
        if self._owns_client:
            await self._client.aclose()

    async def join(self) -> dict[str, Any] | None:
        """Join the configured run when the bridge was given an expected run ID."""

        if self.settings.expected_run_id is None:
            return None
        result = await self._request(
            "POST",
            f"/v1/runs/{self.settings.expected_run_id}/agents:join",
            json={},
        )
        if (
            result.get("run_id") != self.settings.expected_run_id
            or result.get("agent_id") != self.settings.expected_agent_id
        ):
            raise BridgeHttpError(
                409,
                "joined actor does not match the bridge's expected run and agent",
                code="actor_identity_mismatch",
            )
        return result

    async def claim_task(
        self,
        *,
        task_id: str | None = None,
        required_tags: list[str] | None = None,
        required_capabilities: list[str] | None = None,
        lease_seconds: int = 120,
        expected_task_version: int | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        key = idempotency_key or str(uuid4())
        result = await self._request(
            "POST",
            "/v1/tasks:claim",
            json={
                "task_id": task_id,
                "required_tags": required_tags or [],
                "required_capabilities": required_capabilities or [],
                "lease_seconds": lease_seconds,
                "expected_task_version": expected_task_version,
            },
            headers={"Idempotency-Key": key},
        )
        task = result["task"]
        lease = result["lease"]
        if self.settings.expected_run_id is not None and (
            task.get("run_id") != self.settings.expected_run_id
            or lease.get("owner_agent_id") != self.settings.expected_agent_id
        ):
            raise BridgeHttpError(
                409,
                "claimed lease does not match the bridge's expected run and agent",
                code="actor_identity_mismatch",
            )
        binding = LeaseBinding(
            task_id=str(task["task_id"]),
            lease_id=str(lease["lease_id"]),
            task_version=int(task["version"]),
            lease_version=int(lease["version"]),
            extension_seconds=lease_seconds,
        )
        self._leases[binding.task_id] = binding
        self._start_renewal(binding)
        return result

    async def recall_memory(
        self,
        *,
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
        payload: dict[str, Any] = {
            "text": text,
            "task_id": task_id,
            "kinds": kinds or [],
            "visibilities": visibilities or ["task", "run", "repository"],
            "include_refuted": include_refuted,
            "include_superseded": include_superseded,
            "world_at": world_at,
            "recorded_at": recorded_at,
            "min_score": min_score,
            "limit": limit,
            "include_evidence": include_evidence,
            "include_lineage": include_lineage,
        }
        if states is not None:
            payload["states"] = states
        return await self._request("POST", "/v1/memories:recall", json=payload)

    async def publish_memory(
        self,
        *,
        content: str,
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
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        key = idempotency_key or str(uuid4())
        return await self._request(
            "POST",
            "/v1/memories",
            json={
                "content": content,
                "kind": kind,
                "desired_state": desired_state,
                "visibility": visibility,
                "task_id": task_id,
                "title": title,
                "tags": tags or [],
                "evidence": await self._resolve_evidence(evidence, parent_key=key),
                "supersedes_memory_id": supersedes_memory_id,
                "related_memory_ids": related_memory_ids or [],
                "valid_from": valid_from,
                "valid_to": valid_to,
                "confidence": confidence,
                "metadata": metadata or {},
            },
            headers={"Idempotency-Key": key},
        )

    async def checkpoint_task(
        self,
        *,
        task_id: str,
        summary: str,
        discoveries: list[str] | None = None,
        completed_work: list[str] | None = None,
        remaining_work: list[str] | None = None,
        memory_ids: list[str] | None = None,
        artifact_ids: list[str] | None = None,
        state: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        binding = self._binding(task_id)
        async with binding.lock:
            result = await self._request(
                "POST",
                f"/v1/tasks/{task_id}/checkpoints",
                json={
                    "lease_id": binding.lease_id,
                    "expected_task_version": binding.task_version,
                    "expected_lease_version": binding.lease_version,
                    "summary": summary,
                    "discoveries": discoveries or [],
                    "completed_work": completed_work or [],
                    "remaining_work": remaining_work or [],
                    "memory_ids": memory_ids or [],
                    "artifact_ids": artifact_ids or [],
                    "state": state or {},
                },
                headers={"Idempotency-Key": idempotency_key or str(uuid4())},
            )
            self._update_binding(binding, result)
            return result

    async def complete_task(
        self,
        *,
        task_id: str,
        summary: str,
        outcome: str = "succeeded",
        memory_ids: list[str] | None = None,
        artifact_ids: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        binding = self._binding(task_id)
        async with binding.lock:
            response = await self._request(
                "POST",
                f"/v1/tasks/{task_id}:complete",
                json={
                    "lease_id": binding.lease_id,
                    "expected_task_version": binding.task_version,
                    "expected_lease_version": binding.lease_version,
                    "outcome": outcome,
                    "summary": summary,
                    "memory_ids": memory_ids or [],
                    "artifact_ids": artifact_ids or [],
                },
                headers={"Idempotency-Key": idempotency_key or str(uuid4())},
            )
        await self._drop_binding(task_id)
        return response

    async def report_conflict(
        self,
        *,
        memory_ids: list[str],
        description: str,
        evidence: list[dict[str, Any]] | None = None,
        task_id: str | None = None,
        severity: str = "medium",
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        key = idempotency_key or str(uuid4())
        return await self._request(
            "POST",
            "/v1/conflicts",
            json={
                "memory_ids": memory_ids,
                "description": description,
                "evidence": await self._resolve_evidence(evidence, parent_key=key),
                "task_id": task_id,
                "severity": severity,
                "metadata": metadata or {},
            },
            headers={"Idempotency-Key": key},
        )

    async def _resolve_evidence(
        self,
        evidence: list[dict[str, Any]] | None,
        *,
        parent_key: str,
    ) -> list[dict[str, Any]]:
        """Register inline evidence material so refs always cite preserved rows.

        Items already carrying ``evidence_id`` and ``source_id`` pass through
        untouched. Anything else is treated as new material. Retried tool
        calls converge on the same rows twice over: this process caches the
        resolution per parent mutation key, and the server deduplicates
        sources on a deterministic occurrence key, so a bridge restart between
        registration and publish cannot mint a second source.
        """

        if not evidence:
            return []
        cached = self._inline_evidence.get(parent_key)
        if cached is not None:
            return cached
        resolved: list[dict[str, Any]] = []
        for index, item in enumerate(evidence):
            if "evidence_id" in item and "source_id" in item:
                resolved.append(item)
                continue
            resolved.append(
                await self._register_inline_evidence(dict(item), key=f"{parent_key}:ev{index}")
            )
        self._inline_evidence[parent_key] = resolved
        return resolved

    async def _register_inline_evidence(
        self,
        material: dict[str, Any],
        *,
        key: str,
    ) -> dict[str, Any]:
        kind = material.pop("kind", None)
        if not kind:
            raise ValueError(
                "inline evidence requires 'kind'; cite existing rows with "
                "evidence_id and source_id instead"
            )
        excerpt = material.pop("excerpt", None)
        content_sha256 = material.pop("content_sha256", None)
        if content_sha256 is None:
            if not excerpt:
                raise ValueError("inline evidence requires content_sha256 or a non-empty excerpt")
            content_sha256 = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        source_payload = {
            "kind": kind,
            "content_sha256": content_sha256,
            "observed_at": material.pop("observed_at", None) or datetime.now(UTC).isoformat(),
            "uri": material.pop("uri", None),
            # Deterministic across retries: the server returns the existing
            # source for a repeated occurrence instead of minting another.
            "occurrence_key": material.pop("occurrence_key", None) or f"swarmbrain-mcp:{key}",
            "task_id": material.pop("task_id", None),
            "metadata": material.pop("metadata", None) or {},
        }
        locator = material.pop("locator", None)
        if material:
            raise ValueError(f"unsupported inline evidence fields: {sorted(material)}")
        # Fresh idempotency keys: observed_at makes the payload time-varying,
        # so replay identity comes from the occurrence key above instead.
        source = await self._request(
            "POST",
            "/v1/evidence/sources",
            json=source_payload,
            headers={"Idempotency-Key": str(uuid4())},
        )
        row = await self._request(
            "POST",
            "/v1/evidence",
            json={
                "source_id": source["source_id"],
                "kind": kind,
                "locator": locator,
                "excerpt": excerpt,
                "content_sha256": content_sha256,
                "metadata": source_payload["metadata"],
            },
            headers={"Idempotency-Key": str(uuid4())},
        )
        return {
            "evidence_id": row["evidence_id"],
            "source_id": row["source_id"],
            "locator": row.get("locator"),
            "excerpt": row.get("excerpt"),
            "content_sha256": row.get("content_sha256"),
            "metadata": row.get("metadata") or {},
        }

    async def _renew(self, binding: LeaseBinding) -> None:
        interval = max(0.001, binding.extension_seconds * self._renewal_interval_fraction)
        try:
            while self._leases.get(binding.task_id) is binding:
                await self._sleep(interval)
                renewal_key = str(uuid4())
                while self._leases.get(binding.task_id) is binding:
                    try:
                        async with binding.lock:
                            response = await self._request(
                                "POST",
                                f"/v1/leases/{binding.lease_id}:renew",
                                json={
                                    "expected_version": binding.lease_version,
                                    "extension_seconds": binding.extension_seconds,
                                },
                                headers={"Idempotency-Key": renewal_key},
                            )
                            binding.lease_version = int(response["lease"]["version"])
                        break
                    except BridgeHttpError as exc:
                        if not exc.retryable and exc.status_code < 500:
                            return
                    except httpx.TransportError:
                        pass
                    await self._sleep(min(1.0, max(0.05, interval / 10)))
        except asyncio.CancelledError:
            raise
        except Exception:
            # A model-visible task mutation will receive the authoritative
            # fencing error; the bridge never fabricates continued ownership.
            return

    def _start_renewal(self, binding: LeaseBinding) -> None:
        previous = self._renewals.pop(binding.task_id, None)
        if previous is not None:
            previous.cancel()
        self._renewals[binding.task_id] = asyncio.create_task(self._renew(binding))

    async def _drop_binding(self, task_id: str) -> None:
        self._leases.pop(task_id, None)
        renewal = self._renewals.pop(task_id, None)
        if renewal is not None:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)

    def _binding(self, task_id: str) -> LeaseBinding:
        binding = self._leases.get(task_id)
        if binding is None:
            raise ValueError(
                "no live bridge lease for task; claim the task in this MCP session first"
            )
        return binding

    @staticmethod
    def _update_binding(binding: LeaseBinding, response: dict[str, Any]) -> None:
        binding.task_version = int(response["task"]["version"])
        binding.lease_version = int(response["lease"]["version"])

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._client.request(method, path, **kwargs)
        if response.is_success:
            data = response.json()
            if not isinstance(data, dict):
                raise BridgeHttpError(response.status_code, "API returned a non-object response")
            return data

        message = response.text
        code: str | None = None
        retryable = False
        try:
            detail = response.json().get("error", {})
            message = str(detail.get("message") or message)
            code = detail.get("code")
            retryable = bool(detail.get("retryable", False))
        except (AttributeError, ValueError):
            pass
        raise BridgeHttpError(
            response.status_code,
            message,
            code=code,
            retryable=retryable,
        )


__all__ = ["BridgeHttpError", "LeaseBinding", "SwarmBrainHttpClient"]
