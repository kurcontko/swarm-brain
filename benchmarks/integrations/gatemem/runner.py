"""Incremental GateMem runner that emits official external predictions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .answering import AnswerModel, AnswerRequest, ContextMemory, estimate_tokens
from .contracts import (
    GATEMEM_COMMIT,
    GateMemContractError,
    PrincipalScope,
    PublicCheckpoint,
    PublicEpisode,
    PublicTurn,
    ScopeFactory,
    assert_hidden_fields_absent,
    content_sha256,
)
from .gateway import MemoryGateway, MemoryWrite, RecalledMemory, RecallRequest
from .policy import (
    AudiencePolicy,
    DeterministicTurnInterpreter,
    KnownMemory,
    SpeakerOnlyAudiencePolicy,
    TurnInterpreter,
)

HARNESS_SCHEMA_VERSION = 1
TURN_SCHEMA = "gatemem-public-turn-v1"
DELETION_SCHEMA = "gatemem-active-forgetting-v1"


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    recall_limit: int = 20
    min_score: float = 0.0
    context_token_budget: int = 4096

    def __post_init__(self) -> None:
        if not 1 <= self.recall_limit <= 100:
            raise GateMemContractError("recall_limit must be between 1 and 100")
        if not 0.0 <= self.min_score <= 1.0:
            raise GateMemContractError("min_score must be between 0 and 1")
        if not 1 <= self.context_token_budget <= 1_000_000:
            raise GateMemContractError("context_token_budget must be positive")


@dataclass(frozen=True, slots=True)
class HarnessRun:
    predictions: tuple[dict[str, Any], ...]
    audit: dict[str, Any]

    def write_predictions(self, path: str | Path) -> None:
        destination = Path(path)
        _atomic_write_text(
            destination,
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                for row in self.predictions
            ),
        )

    def write_audit(self, path: str | Path) -> None:
        destination = Path(path)
        _atomic_write_text(
            destination,
            json.dumps(self.audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )


class GateMemHarness:
    """Map official incremental episodes onto principal-scoped Swarm memory."""

    def __init__(
        self,
        *,
        gateway: MemoryGateway,
        answer_model: AnswerModel,
        audience_policy: AudiencePolicy | None = None,
        turn_interpreter: TurnInterpreter | None = None,
        scope_factory: ScopeFactory | None = None,
        config: HarnessConfig | None = None,
    ) -> None:
        self.gateway = gateway
        self.answer_model = answer_model
        self.audience_policy = audience_policy or SpeakerOnlyAudiencePolicy()
        self.turn_interpreter = turn_interpreter or DeterministicTurnInterpreter()
        self.scope_factory = scope_factory or ScopeFactory()
        self.config = config or HarnessConfig()

    async def run(
        self,
        *,
        episodes: Sequence[dict[str, Any]],
        checkpoints: Sequence[dict[str, Any]],
    ) -> HarnessRun:
        public_episodes = tuple(PublicEpisode.from_raw(item) for item in episodes)
        public_checkpoints = tuple(PublicCheckpoint.from_raw(item) for item in checkpoints)
        if len({item.episode_id for item in public_episodes}) != len(public_episodes):
            raise GateMemContractError("GateMem episode IDs must be unique")
        if len({item.checkpoint_id for item in public_checkpoints}) != len(public_checkpoints):
            raise GateMemContractError("GateMem checkpoint IDs must be unique")

        checkpoints_by_episode: dict[str, list[PublicCheckpoint]] = {}
        for checkpoint in public_checkpoints:
            checkpoints_by_episode.setdefault(checkpoint.episode_id, []).append(checkpoint)
        known_episode_ids = {item.episode_id for item in public_episodes}
        unknown = set(checkpoints_by_episode).difference(known_episode_ids)
        if unknown:
            raise GateMemContractError(f"checkpoints reference unknown episodes: {sorted(unknown)}")

        predictions: list[dict[str, Any]] = []
        ingest_events: list[dict[str, Any]] = []
        catalogs: dict[str, dict[str, KnownMemory]] = {}
        deletion_fences: dict[str, set[str]] = {}
        started = perf_counter()
        for episode in public_episodes:
            episode_checkpoints = checkpoints_by_episode.get(episode.episode_id, [])
            if not episode_checkpoints:
                continue
            episode_predictions, episode_ingest = await self._run_episode(
                episode=episode,
                checkpoints=episode_checkpoints,
                catalogs=catalogs,
                deletion_fences=deletion_fences,
            )
            predictions.extend(episode_predictions)
            ingest_events.extend(episode_ingest)

        elapsed_ms = (perf_counter() - started) * 1000.0
        for prediction in predictions:
            assert_hidden_fields_absent(prediction)
        audit = {
            "schema_version": HARNESS_SCHEMA_VERSION,
            "benchmark": "GateMem",
            "gatemem_commit": GATEMEM_COMMIT,
            "adapter": "swarmbrain-gatemem-external",
            "config": asdict(self.config),
            "audience_policy": {
                "type": type(self.audience_policy).__name__,
                "manifest_sha256": getattr(self.audience_policy, "manifest_sha256", None),
            },
            "turn_interpreter": type(self.turn_interpreter).__name__,
            "episodes": len({item.episode_id for item in public_checkpoints}),
            "checkpoints": len(predictions),
            "ingest_operations": ingest_events,
            "latency_ms": {"total": elapsed_ms},
        }
        assert_hidden_fields_absent(audit)
        return HarnessRun(predictions=tuple(predictions), audit=audit)

    async def _run_episode(
        self,
        *,
        episode: PublicEpisode,
        checkpoints: Sequence[PublicCheckpoint],
        catalogs: dict[str, dict[str, KnownMemory]],
        deletion_fences: dict[str, set[str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        turn_positions = {turn.turn_id: index for index, turn in enumerate(episode.turns)}
        ordered: list[tuple[int, int, PublicCheckpoint]] = []
        for source_index, checkpoint in enumerate(checkpoints):
            if checkpoint.episode_id != episode.episode_id:
                raise GateMemContractError("checkpoint crossed episode boundary")
            if checkpoint.as_of_turn_id not in turn_positions:
                raise GateMemContractError(
                    f"checkpoint {checkpoint.checkpoint_id} has unknown as_of_turn_id"
                )
            if checkpoint.asker.principal_id not in episode.principal_ids:
                raise GateMemContractError(
                    f"checkpoint requester {checkpoint.asker.principal_id!r} is unknown"
                )
            known_role = episode.role_for(checkpoint.asker.principal_id)
            if known_role != checkpoint.asker.role:
                raise GateMemContractError(
                    f"checkpoint requester role changed for {checkpoint.asker.principal_id!r}"
                )
            ordered.append((turn_positions[checkpoint.as_of_turn_id], source_index, checkpoint))
        ordered.sort(key=lambda item: (item[0], item[1]))

        predictions: list[dict[str, Any]] = []
        ingest_events: list[dict[str, Any]] = []
        ingested_upto = -1
        for target_position, _, checkpoint in ordered:
            ingest_started = perf_counter()
            new_turn_count = 0
            new_operation_count = 0
            for index in range(ingested_upto + 1, target_position + 1):
                events = await self._ingest_turn(
                    episode=episode,
                    turn=episode.turns[index],
                    catalogs=catalogs,
                    deletion_fences=deletion_fences,
                )
                ingest_events.extend(events)
                new_turn_count += 1
                new_operation_count += len(events)
                ingested_upto = index
            ingest_ms = (perf_counter() - ingest_started) * 1000.0
            predictions.append(
                await self._query(
                    episode=episode,
                    checkpoint=checkpoint,
                    catalog=catalogs,
                    ingest_ms=ingest_ms,
                    ingested_turns=new_turn_count,
                    ingest_operations=new_operation_count,
                )
            )
        return predictions, ingest_events

    async def _ingest_turn(
        self,
        *,
        episode: PublicEpisode,
        turn: PublicTurn,
        catalogs: dict[str, dict[str, KnownMemory]],
        deletion_fences: dict[str, set[str]],
    ) -> list[dict[str, Any]]:
        audiences = self.audience_policy.audiences(episode, turn)
        if not audiences:
            raise GateMemContractError(
                f"audience policy returned no principal for {episode.episode_id}/{turn.turn_id}"
            )
        unknown = audiences.difference(episode.principal_ids)
        if unknown:
            raise GateMemContractError(
                f"audience policy returned unknown principals: {sorted(unknown)}"
            )

        events: list[dict[str, Any]] = []
        for principal_id in sorted(audiences):
            role = episode.role_for(principal_id)
            assert role is not None
            scope = self.scope_factory.for_principal(
                domain=episode.domain,
                episode_id=episode.episode_id,
                principal_id=principal_id,
                principal_role=role,
            )
            catalog = catalogs.setdefault(scope.key, {})
            fences = deletion_fences.setdefault(scope.key, set())
            plan = self.turn_interpreter.plan(
                turn,
                tuple(catalog.values()),
                frozenset(fences),
            )
            fences.update(plan.new_fence_hashes)

            if plan.remember:
                text_digest = content_sha256(turn.text)
                stored_content = {
                    "schema": TURN_SCHEMA,
                    "episode_id": episode.episode_id,
                    "turn_id": turn.turn_id,
                    "timestamp": turn.timestamp,
                    "speaker": {
                        "principal_id": turn.speaker.principal_id,
                        "role": turn.speaker.role,
                    },
                    "turn_kind": turn.turn_kind,
                    "text": turn.text,
                    "record_refs": list(turn.record_refs),
                    "content_sha256": text_digest,
                }
                published = await self.gateway.publish(
                    scope,
                    MemoryWrite(
                        idempotency_key=_idempotency_key(scope, turn.turn_id, "remember"),
                        content=stored_content,
                        title=f"GateMem {episode.domain} {turn.turn_id}",
                        tags=("gatemem", episode.domain, "turn"),
                        metadata=_memory_metadata(
                            scope=scope,
                            turn=turn,
                            content_digest=text_digest,
                            operation="remember",
                        ),
                    ),
                )
                if published.content != stored_content:
                    raise GateMemContractError("Swarm Brain changed GateMem turn content")
                catalog[published.memory_id] = KnownMemory(
                    memory_id=published.memory_id,
                    version=published.version,
                    source_turn_id=turn.turn_id,
                    text=turn.text,
                    content_sha256=text_digest,
                )
                events.append(
                    _ingest_event(
                        scope=scope,
                        turn=turn,
                        action="remember",
                        memory_id=published.memory_id,
                        memory_version=published.version,
                        content_digest=text_digest,
                        latency_ms=published.latency_ms,
                    )
                )

            for target_id in plan.forget_memory_ids:
                target = catalog.get(target_id)
                if target is None:
                    raise GateMemContractError(
                        f"turn interpreter targeted non-current memory {target_id}"
                    )
                tombstone = {
                    "schema": DELETION_SCHEMA,
                    "episode_id": episode.episode_id,
                    "turn_id": turn.turn_id,
                    "status": "deleted",
                    "target_memory_id": target.memory_id,
                    "target_content_sha256": target.content_sha256,
                }
                published = await self.gateway.publish(
                    scope,
                    MemoryWrite(
                        idempotency_key=_idempotency_key(
                            scope, turn.turn_id, f"forget:{target.memory_id}"
                        ),
                        content=tombstone,
                        title=f"GateMem deletion {turn.turn_id}",
                        tags=("gatemem", episode.domain, "deletion"),
                        metadata=_memory_metadata(
                            scope=scope,
                            turn=turn,
                            content_digest=target.content_sha256,
                            operation="forget",
                            deletion_directive_sha256=content_sha256(turn.text),
                        ),
                        supersedes_memory_id=target.memory_id,
                    ),
                )
                if published.content != tombstone:
                    raise GateMemContractError("Swarm Brain changed GateMem tombstone content")
                del catalog[target_id]
                events.append(
                    _ingest_event(
                        scope=scope,
                        turn=turn,
                        action="forget",
                        memory_id=published.memory_id,
                        memory_version=published.version,
                        content_digest=target.content_sha256,
                        latency_ms=published.latency_ms,
                        superseded_memory_id=target.memory_id,
                    )
                )

            if not plan.remember and not plan.forget_memory_ids:
                events.append(
                    _ingest_event(
                        scope=scope,
                        turn=turn,
                        action="safe_noop",
                        content_digest=content_sha256(turn.text),
                        reason=plan.reason,
                    )
                )
        return events

    async def _query(
        self,
        *,
        episode: PublicEpisode,
        checkpoint: PublicCheckpoint,
        catalog: dict[str, dict[str, KnownMemory]],
        ingest_ms: float,
        ingested_turns: int,
        ingest_operations: int,
    ) -> dict[str, Any]:
        query_started = perf_counter()
        scope = self.scope_factory.for_principal(
            domain=episode.domain,
            episode_id=episode.episode_id,
            principal_id=checkpoint.asker.principal_id,
            principal_role=checkpoint.asker.role,
        )
        response = await self.gateway.recall(
            scope,
            RecallRequest(
                text=checkpoint.query_text,
                limit=self.config.recall_limit,
                min_score=self.config.min_score,
            ),
        )
        active_ids = frozenset(catalog.get(scope.key, {}))
        candidates = tuple(
            self._context_memory(scope, episode, item, active_ids) for item in response.memories
        )
        candidates = tuple(item for item in candidates if item is not None)
        packed, dropped, context_tokens = _pack_context(
            candidates, token_budget=self.config.context_token_budget
        )
        request = AnswerRequest(
            checkpoint_id=checkpoint.checkpoint_id,
            episode_id=episode.episode_id,
            requester_principal_id=checkpoint.asker.principal_id,
            requester_role=checkpoint.asker.role,
            relationship_facts_json=episode.relationship_facts_for(checkpoint.asker.principal_id),
            query_text=checkpoint.query_text,
            memories=packed,
        )
        payload = request.agent_payload()
        answer_started = perf_counter()
        answer = await self.answer_model.answer(request)
        answer_ms = (perf_counter() - answer_started) * 1000.0
        if (
            answer.token_usage_source != "provider"
            or answer.input_tokens is None
            or answer.output_tokens is None
        ):
            raise GateMemContractError(
                "GateMem evidence requires provider-reported answer-call token usage"
            )
        provider_total_tokens = answer.input_tokens + answer.output_tokens
        allowed_record_ids = {record_id for memory in packed for record_id in memory.record_refs}
        unknown_record_ids = set(answer.used_record_ids).difference(allowed_record_ids)
        if unknown_record_ids:
            raise GateMemContractError(
                f"answer model cited record IDs outside recalled context: {sorted(unknown_record_ids)}"
            )
        total_ms = (perf_counter() - query_started) * 1000.0
        output = {
            "action": answer.action,
            "answer": answer.answer,
            "answer_structured": answer.answer_structured,
            "used_record_ids": list(answer.used_record_ids),
            "memory_audit": _official_memory_audit(payload["recalled_memory"]),
            "llm_usage": {
                "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens,
                "total_tokens": provider_total_tokens,
            },
        }
        audit = {
            "schema_version": HARNESS_SCHEMA_VERSION,
            "gatemem_commit": GATEMEM_COMMIT,
            "episode_id": episode.episode_id,
            "as_of_turn_id": checkpoint.as_of_turn_id,
            "requester": {
                "principal_id": checkpoint.asker.principal_id,
                "role": checkpoint.asker.role,
                "scope_key_sha256": content_sha256(scope.key),
            },
            "query_sha256": content_sha256(checkpoint.query_text),
            "retrieval": {
                "limit": self.config.recall_limit,
                "min_score": self.config.min_score,
                "total_candidates": response.total_candidates,
                "returned": len(response.memories),
                "packed": len(packed),
                "dropped_by_token_budget": dropped,
                "truncated": response.truncated,
                "provenance": [_provenance(item) for item in packed],
            },
            "tokens": {
                "context_budget": self.config.context_token_budget,
                "context_estimated": context_tokens,
                "request_estimated": estimate_tokens(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True)
                ),
                "provider_input": answer.input_tokens,
                "provider_output": answer.output_tokens,
                "provider_usage_reported": True,
                "usage_source": answer.token_usage_source,
            },
            "answer_model": {
                "provider": answer.provider,
                "model": answer.model,
                "revision": answer.revision,
            },
            "latency_ms": {
                "incremental_ingest": ingest_ms,
                "recall": response.latency_ms,
                "answer": answer_ms,
                "query_total": total_ms,
            },
            "incremental_ingest": {
                "turns": ingested_turns,
                "operations": ingest_operations,
            },
        }
        prediction = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "output": output,
            "swarmbrain_audit": audit,
        }
        assert_hidden_fields_absent(prediction)
        return prediction

    @staticmethod
    def _context_memory(
        scope: PrincipalScope,
        episode: PublicEpisode,
        recalled: RecalledMemory,
        active_memory_ids: frozenset[str],
    ) -> ContextMemory | None:
        content = recalled.content
        if not isinstance(content, dict):
            raise GateMemContractError("GateMem recall returned non-object content")
        schema = content.get("schema")
        if schema == DELETION_SCHEMA:
            return None
        if schema != TURN_SCHEMA:
            raise GateMemContractError(f"unexpected memory schema in GateMem scope: {schema!r}")
        if recalled.memory_id not in active_memory_ids:
            raise GateMemContractError(
                "Swarm Brain recalled a memory removed by the active-forgetting catalog"
            )
        if recalled.state not in {"tentative", "confirmed"}:
            raise GateMemContractError(f"Swarm Brain recalled disallowed state {recalled.state!r}")
        if content.get("episode_id") != episode.episode_id:
            raise GateMemContractError("recalled GateMem turn crossed episode boundary")
        if recalled.metadata.get("principal_scope_key") != scope.key:
            raise GateMemContractError("recalled GateMem turn crossed principal boundary")
        if recalled.metadata.get("gatemem_commit") != GATEMEM_COMMIT:
            raise GateMemContractError("recalled GateMem turn has unpinned provenance")
        text = content.get("text")
        digest = content.get("content_sha256")
        if not isinstance(text, str) or not text or digest != content_sha256(text):
            raise GateMemContractError("recalled GateMem turn failed content provenance")
        speaker = content.get("speaker")
        if not isinstance(speaker, dict):
            raise GateMemContractError("recalled GateMem turn speaker is malformed")
        speaker_id = speaker.get("principal_id")
        speaker_role = speaker.get("role")
        if not isinstance(speaker_id, str) or not isinstance(speaker_role, str):
            raise GateMemContractError("recalled GateMem turn speaker fields are malformed")
        turn_id = content.get("turn_id")
        timestamp = content.get("timestamp")
        refs = content.get("record_refs") or []
        if not isinstance(turn_id, str) or not turn_id:
            raise GateMemContractError("recalled GateMem turn ID is malformed")
        if timestamp is not None and not isinstance(timestamp, str):
            raise GateMemContractError("recalled GateMem timestamp is malformed")
        if not isinstance(refs, list) or any(not isinstance(item, str) for item in refs):
            raise GateMemContractError("recalled GateMem record refs are malformed")
        return ContextMemory(
            memory_id=recalled.memory_id,
            version=recalled.version,
            source_turn_id=turn_id,
            speaker_principal_id=speaker_id,
            speaker_role=speaker_role,
            timestamp=timestamp,
            text=text,
            record_refs=tuple(refs),
            content_sha256=digest,
            score=recalled.score,
        )


def _memory_metadata(
    *,
    scope: PrincipalScope,
    turn: PublicTurn,
    content_digest: str,
    operation: str,
    deletion_directive_sha256: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "benchmark": "GateMem",
        "gatemem_commit": GATEMEM_COMMIT,
        "adapter_schema_version": HARNESS_SCHEMA_VERSION,
        "domain": scope.domain,
        "episode_id": scope.episode_id,
        "source_turn_id": turn.turn_id,
        "source_speaker_principal_id": turn.speaker.principal_id,
        "source_speaker_role": turn.speaker.role,
        "principal_scope_key": scope.key,
        "principal_scope_id_sha256": content_sha256(scope.key),
        "operation": operation,
        "content_sha256": content_digest,
    }
    if deletion_directive_sha256 is not None:
        metadata["deletion_directive_sha256"] = deletion_directive_sha256
    return metadata


def _idempotency_key(scope: PrincipalScope, turn_id: str, operation: str) -> str:
    source = "|".join(
        [
            GATEMEM_COMMIT,
            str(HARNESS_SCHEMA_VERSION),
            scope.key,
            turn_id,
            operation,
        ]
    )
    return f"gatemem:{hashlib.sha256(source.encode()).hexdigest()}"


def _ingest_event(
    *,
    scope: PrincipalScope,
    turn: PublicTurn,
    action: str,
    content_digest: str,
    memory_id: str | None = None,
    memory_version: int | None = None,
    latency_ms: float = 0.0,
    superseded_memory_id: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "episode_id": scope.episode_id,
        "source_turn_id": turn.turn_id,
        "principal_id": scope.principal_id,
        "principal_scope_key_sha256": content_sha256(scope.key),
        "action": action,
        "content_sha256": content_digest,
        "latency_ms": latency_ms,
    }
    if memory_id is not None:
        event["memory_id"] = memory_id
    if memory_version is not None:
        event["memory_version"] = memory_version
    if superseded_memory_id is not None:
        event["superseded_memory_id"] = superseded_memory_id
    if reason is not None:
        event["reason"] = reason
    return event


def _render_context_memory(memory: ContextMemory) -> str:
    return (
        f"turn={memory.source_turn_id} speaker={memory.speaker_principal_id} "
        f"role={memory.speaker_role} timestamp={memory.timestamp or 'unknown'}\n{memory.text}"
    )


def _pack_context(
    candidates: tuple[ContextMemory, ...], *, token_budget: int
) -> tuple[tuple[ContextMemory, ...], int, int]:
    selected: list[ContextMemory] = []
    used = 0
    for memory in candidates:
        cost = estimate_tokens(_render_context_memory(memory))
        if used + cost > token_budget:
            continue
        selected.append(memory)
        used += cost
    return tuple(selected), len(candidates) - len(selected), used


def _provenance(memory: ContextMemory) -> dict[str, Any]:
    return {
        "memory_id": memory.memory_id,
        "memory_version": memory.version,
        "source_turn_id": memory.source_turn_id,
        "source_speaker_principal_id": memory.speaker_principal_id,
        "source_speaker_role": memory.speaker_role,
        "content_sha256": memory.content_sha256,
        "record_refs": list(memory.record_refs),
        "score": memory.score,
    }


def _official_memory_audit(recalled_memory: list[dict[str, Any]]) -> dict[str, Any]:
    """GateMem's prompt-exposure schema over the exact model-visible memory list."""

    text = json.dumps(
        recalled_memory,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": 1,
        "stage": "prompt_context",
        "context_format": "swarmbrain-json-v1",
        "prompt_context": {
            "text": text,
            "n_chars": len(text),
            "n_items": len(recalled_memory),
            "items": recalled_memory,
        },
    }


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace one completed canonical artifact without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor = -1
    temporary_name: str | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            file_descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise GateMemContractError(f"cannot atomically write GateMem artifact: {path}") from exc
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                Path(temporary_name).unlink()


__all__ = [
    "DELETION_SCHEMA",
    "GateMemHarness",
    "HARNESS_SCHEMA_VERSION",
    "HarnessConfig",
    "HarnessRun",
    "TURN_SCHEMA",
]
