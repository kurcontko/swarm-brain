"""CockroachDB-backed task coordination with transactional fencing and events."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb

from swarmbrain.application.errors import (
    InvalidState,
    LeaseLost,
    NoTaskAvailable,
    ResourceNotFound,
)
from swarmbrain.domain.agents import ActorContext, Agent, AgentStatus
from swarmbrain.domain.common import ContractModel, RunId
from swarmbrain.domain.events import AggregateType, EventPage, EventType, RunMetrics, SwarmEvent
from swarmbrain.domain.leases import LeaseStatus, RenewLeaseCommand, RenewLeaseResult, TaskLease
from swarmbrain.domain.tasks import (
    CheckpointCommand,
    CheckpointResult,
    ClaimTaskCommand,
    ClaimTaskResult,
    CompleteTaskCommand,
    CompletionResult,
    DependencyKind,
    ReleaseResult,
    ReleaseTaskCommand,
    Task,
    TaskCheckpoint,
    TaskCompletion,
    TaskDependency,
    TaskStatus,
)

from .database import CockroachDatabase


class _ClaimContention(RuntimeError):
    """Ask the shared transaction runner to retry a transient SKIP LOCKED miss."""

    sqlstate = "40001"


TASK_COLUMNS = """
    id, tenant_id, project_id, repository_id, swarm_id, run_id,
    title, description, state, priority, version, required_capabilities,
    tags, created_by_agent_id, claimed_by_agent_id, active_lease_id,
    available_at, metadata, created_at, updated_at, completed_at
"""
TASK_ALIAS_COLUMNS = """
    t.id, t.tenant_id, t.project_id, t.repository_id, t.swarm_id, t.run_id,
    t.title, t.description, t.state, t.priority, t.version,
    t.required_capabilities, t.tags, t.created_by_agent_id,
    t.claimed_by_agent_id, t.active_lease_id, t.available_at, t.metadata,
    t.created_at, t.updated_at, t.completed_at
"""
LEASE_COLUMNS = """
    id, task_id, tenant_id, run_id, agent_id, status, version,
    claimed_at, renewed_at, expires_at, released_at, completed_at, metadata
"""
CHECKPOINT_COLUMNS = """
    id, task_id, lease_id, tenant_id, run_id, agent_id, sequence, summary,
    discoveries, completed_work, remaining_work, memory_ids, artifact_ids,
    state, created_at
"""
COMPLETION_COLUMNS = """
    id, task_id, lease_id, tenant_id, run_id, agent_id, outcome, summary,
    memory_ids, artifact_ids, completed_at
"""
EVENT_COLUMNS = """
    id, tenant_id, project_id, repository_id, swarm_id, run_id, agent_id,
    task_id, event_type, aggregate_type, aggregate_id, aggregate_version,
    payload, occurred_at, recorded_at, correlation_id, causation_id,
    idempotency_key
"""

INSERT_RUN_SQL = """
    INSERT INTO runs (
        tenant_id, id, project_id, repository_id, swarm_id, state, metadata
    ) VALUES (%s, %s, %s, %s, %s, 'active', %s)
    ON CONFLICT (tenant_id, id) DO NOTHING
    RETURNING tenant_id, id
"""
SELECT_RUN_SCOPE_SQL = """
    SELECT tenant_id, id, project_id, repository_id, swarm_id, state
    FROM runs
    WHERE tenant_id = %s
      AND id = %s
      AND project_id = %s
      AND repository_id = %s
      AND swarm_id = %s
    FOR UPDATE
"""
INSERT_AGENT_SQL = """
    INSERT INTO agents (
        tenant_id, run_id, id, project_id, repository_id, swarm_id,
        harness, provider, model, status, capabilities, metadata,
        joined_at, last_seen_at, version
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active',
        %s::STRING[], %s, now(), now(), 1
    )
    ON CONFLICT (tenant_id, run_id, id) DO NOTHING
    RETURNING
        tenant_id, run_id, id, project_id, repository_id, swarm_id,
        harness, provider, model, status, capabilities, metadata,
        joined_at, last_seen_at, version
"""
UPDATE_AGENT_SQL = """
    UPDATE agents
    SET harness = %s,
        provider = %s,
        model = %s,
        capabilities = %s::STRING[],
        metadata = %s,
        last_seen_at = now(),
        version = version + 1
    WHERE tenant_id = %s
      AND run_id = %s
      AND id = %s
      AND project_id = %s
      AND repository_id = %s
      AND swarm_id = %s
    RETURNING
        tenant_id, run_id, id, project_id, repository_id, swarm_id,
        harness, provider, model, status, capabilities, metadata,
        joined_at, last_seen_at, version
"""
INSERT_TASK_SQL = f"""
    INSERT INTO tasks (
        id, tenant_id, project_id, repository_id, swarm_id, run_id,
        title, description, state, priority, version, required_capabilities,
        tags, created_by_agent_id, claimed_by_agent_id, active_lease_id,
        available_at, metadata, created_at, updated_at, completed_at
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s::STRING[], %s::STRING[], %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT (id) DO NOTHING
    RETURNING {TASK_COLUMNS}
"""
LOCK_DEPENDENCY_TASKS_SQL = """
    SELECT id, tenant_id, project_id, repository_id, swarm_id, run_id
    FROM tasks
    WHERE id = ANY(%s::UUID[])
    ORDER BY id
    FOR UPDATE
"""
DEPENDENCY_CYCLE_SQL = """
    WITH RECURSIVE dependency_walk(task_id) AS (
        VALUES (%s::UUID)
        UNION
        SELECT dependency.depends_on_task_id
        FROM task_dependencies AS dependency
        JOIN dependency_walk AS walk ON dependency.task_id = walk.task_id
    )
    SELECT EXISTS (
        SELECT 1
        FROM dependency_walk
        WHERE task_id = %s::UUID
    ) AS creates_cycle
"""
INSERT_DEPENDENCY_SQL = """
    INSERT INTO task_dependencies (
        task_id, depends_on_task_id, kind, created_at
    ) VALUES (%s, %s, %s, %s)
    ON CONFLICT (task_id, depends_on_task_id) DO NOTHING
    RETURNING task_id, depends_on_task_id, kind, created_at
"""
SELECT_DEPENDENCY_SQL = """
    SELECT task_id, depends_on_task_id, kind, created_at
    FROM task_dependencies
    WHERE task_id = %s
      AND depends_on_task_id = %s
"""
EXPIRE_LEASES_SQL = """
    UPDATE task_leases AS lease
    SET status = 'expired',
        version = lease.version + 1
    WHERE lease.tenant_id = %s
      AND lease.run_id = %s
      AND lease.status = 'active'
      AND lease.expires_at <= now()
      AND EXISTS (
          SELECT 1
          FROM tasks AS task
          WHERE task.id = lease.task_id
            AND task.tenant_id = %s
            AND task.project_id = %s
            AND task.repository_id = %s
            AND task.swarm_id = %s
            AND task.run_id = %s
      )
    RETURNING lease.id, lease.task_id
"""
RELEASE_EXPIRED_TASKS_SQL = """
    UPDATE tasks
    SET state = 'ready',
        claimed_by_agent_id = NULL,
        active_lease_id = NULL,
        updated_at = now(),
        version = version + 1
    WHERE tenant_id = %s
      AND project_id = %s
      AND repository_id = %s
      AND swarm_id = %s
      AND run_id = %s
      AND active_lease_id = ANY(%s::UUID[])
    RETURNING id
"""
CLAIM_CANDIDATE_SQL = f"""
    SELECT {TASK_ALIAS_COLUMNS}
    FROM tasks AS t
    WHERE t.tenant_id = %s
      AND t.project_id = %s
      AND t.repository_id = %s
      AND t.swarm_id = %s
      AND t.run_id = %s
      AND (%s::UUID IS NULL OR t.id = %s::UUID)
      AND t.state IN ('pending', 'ready')
      AND t.active_lease_id IS NULL
      AND (t.available_at IS NULL OR t.available_at <= now())
      AND t.tags @> %s::STRING[]
      AND t.required_capabilities @> %s::STRING[]
      AND %s::STRING[] @> t.required_capabilities
      AND NOT EXISTS (
          SELECT 1
          FROM task_dependencies AS dependency
          LEFT JOIN tasks AS prerequisite
            ON prerequisite.id = dependency.depends_on_task_id
           AND prerequisite.tenant_id = t.tenant_id
           AND prerequisite.project_id = t.project_id
           AND prerequisite.repository_id = t.repository_id
           AND prerequisite.swarm_id = t.swarm_id
           AND prerequisite.run_id = t.run_id
          WHERE dependency.task_id = t.id
            AND dependency.kind = 'blocks'
            AND (prerequisite.id IS NULL OR prerequisite.state != 'completed')
      )
    ORDER BY
        t.priority DESC,
        COALESCE(t.available_at, t.created_at),
        t.created_at,
        t.id
    LIMIT 1
    FOR UPDATE SKIP LOCKED
"""
CLAIM_ELIGIBLE_SQL = """
    SELECT EXISTS (
        SELECT 1
        FROM tasks AS t
        WHERE t.tenant_id = %s
          AND t.project_id = %s
          AND t.repository_id = %s
          AND t.swarm_id = %s
          AND t.run_id = %s
          AND (%s::UUID IS NULL OR t.id = %s::UUID)
          AND t.state IN ('pending', 'ready')
          AND t.active_lease_id IS NULL
          AND (t.available_at IS NULL OR t.available_at <= now())
          AND t.tags @> %s::STRING[]
          AND t.required_capabilities @> %s::STRING[]
          AND %s::STRING[] @> t.required_capabilities
          AND NOT EXISTS (
              SELECT 1
              FROM task_dependencies AS dependency
              LEFT JOIN tasks AS prerequisite
                ON prerequisite.id = dependency.depends_on_task_id
               AND prerequisite.tenant_id = t.tenant_id
               AND prerequisite.project_id = t.project_id
               AND prerequisite.repository_id = t.repository_id
               AND prerequisite.swarm_id = t.swarm_id
               AND prerequisite.run_id = t.run_id
              WHERE dependency.task_id = t.id
                AND dependency.kind = 'blocks'
                AND (prerequisite.id IS NULL OR prerequisite.state != 'completed')
          )
    ) AS eligible
"""
SELECT_CLAIM_TASK_STATE_SQL = """
    SELECT id, state, version
    FROM tasks
    WHERE id = %s
      AND tenant_id = %s
      AND project_id = %s
      AND repository_id = %s
      AND swarm_id = %s
      AND run_id = %s
"""
MARK_TASK_CLAIMED_SQL = f"""
    UPDATE tasks
    SET state = 'claimed',
        claimed_by_agent_id = %s,
        active_lease_id = %s,
        updated_at = now(),
        version = version + 1
    WHERE id = %s
      AND tenant_id = %s
      AND project_id = %s
      AND repository_id = %s
      AND swarm_id = %s
      AND run_id = %s
      AND version = %s
      AND state IN ('pending', 'ready')
      AND active_lease_id IS NULL
    RETURNING {TASK_COLUMNS}
"""
INSERT_LEASE_SQL = f"""
    INSERT INTO task_leases (
        id, task_id, tenant_id, run_id, agent_id, status, version,
        claimed_at, renewed_at, expires_at, metadata
    ) VALUES (
        %s, %s, %s, %s, %s, 'active', 1,
        now(), now(), now() + %s, %s
    )
    ON CONFLICT DO NOTHING
    RETURNING {LEASE_COLUMNS}
"""
LATEST_CHECKPOINT_SQL = f"""
    SELECT {CHECKPOINT_COLUMNS}
    FROM task_checkpoints AS checkpoint
    WHERE checkpoint.task_id = %s
      AND checkpoint.tenant_id = %s
      AND checkpoint.run_id = %s
      AND EXISTS (
          SELECT 1
          FROM tasks AS task
          WHERE task.id = checkpoint.task_id
            AND task.tenant_id = %s
            AND task.project_id = %s
            AND task.repository_id = %s
            AND task.swarm_id = %s
            AND task.run_id = %s
      )
    ORDER BY checkpoint.sequence DESC
    LIMIT 1
"""
RENEW_LEASE_SQL = f"""
    UPDATE task_leases AS lease
    SET renewed_at = now(),
        expires_at = greatest(lease.expires_at, now() + %s),
        version = lease.version + 1
    WHERE lease.id = %s
      AND lease.tenant_id = %s
      AND lease.run_id = %s
      AND lease.agent_id = %s
      AND lease.status = 'active'
      AND lease.expires_at > now()
      AND lease.version = %s
      AND EXISTS (
          SELECT 1
          FROM tasks AS task
          WHERE task.id = lease.task_id
            AND task.tenant_id = %s
            AND task.project_id = %s
            AND task.repository_id = %s
            AND task.swarm_id = %s
            AND task.run_id = %s
            AND task.state = 'claimed'
            AND task.active_lease_id = lease.id
      )
    RETURNING {LEASE_COLUMNS}
"""
LOAD_TASK_FOR_UPDATE_SQL = f"""
    SELECT {TASK_COLUMNS}
    FROM tasks
    WHERE id = %s
      AND tenant_id = %s
      AND project_id = %s
      AND repository_id = %s
      AND swarm_id = %s
      AND run_id = %s
    FOR UPDATE
"""
LOAD_LEASE_FOR_UPDATE_SQL = f"""
    SELECT {LEASE_COLUMNS}
    FROM task_leases
    WHERE id = %s
      AND task_id = %s
      AND tenant_id = %s
      AND run_id = %s
      AND agent_id = %s
      AND status = 'active'
      AND expires_at > now()
    FOR UPDATE
"""
SCOPED_MEMORY_IDS_SQL = """
    SELECT id
    FROM memories
    WHERE id = ANY(%s::UUID[])
      AND tenant_id = %s
      AND project_id = %s
      AND repository_id = %s
      AND (
          visibility = 'repository'
          OR (swarm_id = %s AND run_id = %s)
      )
"""
NEXT_CHECKPOINT_SEQUENCE_SQL = """
    SELECT sequence
    FROM task_checkpoints
    WHERE task_id = %s
      AND tenant_id = %s
      AND run_id = %s
    ORDER BY sequence DESC
    LIMIT 1
"""
INSERT_CHECKPOINT_SQL = f"""
    INSERT INTO task_checkpoints (
        id, task_id, lease_id, tenant_id, run_id, agent_id, sequence,
        summary, discoveries, completed_work, remaining_work, memory_ids,
        artifact_ids, state, created_at
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s::UUID[], %s::UUID[], %s, now()
    )
    RETURNING {CHECKPOINT_COLUMNS}
"""
ADVANCE_TASK_SQL = f"""
    UPDATE tasks
    SET updated_at = now(),
        version = version + 1
    WHERE id = %s
      AND tenant_id = %s
      AND project_id = %s
      AND repository_id = %s
      AND swarm_id = %s
      AND run_id = %s
      AND state = 'claimed'
      AND active_lease_id = %s
      AND version = %s
    RETURNING {TASK_COLUMNS}
"""
ADVANCE_LEASE_SQL = f"""
    UPDATE task_leases
    SET version = version + 1
    WHERE id = %s
      AND task_id = %s
      AND tenant_id = %s
      AND run_id = %s
      AND agent_id = %s
      AND status = 'active'
      AND expires_at > now()
      AND version = %s
    RETURNING {LEASE_COLUMNS}
"""
INSERT_COMPLETION_SQL = f"""
    INSERT INTO task_completions (
        id, task_id, lease_id, tenant_id, run_id, agent_id, outcome,
        summary, memory_ids, artifact_ids, completed_at
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s,
        %s::UUID[], %s::UUID[], now()
    )
    ON CONFLICT (task_id) DO NOTHING
    RETURNING {COMPLETION_COLUMNS}
"""
COMPLETE_TASK_SQL = f"""
    UPDATE tasks
    SET state = %s,
        claimed_by_agent_id = NULL,
        active_lease_id = NULL,
        updated_at = now(),
        completed_at = now(),
        version = version + 1
    WHERE id = %s
      AND tenant_id = %s
      AND project_id = %s
      AND repository_id = %s
      AND swarm_id = %s
      AND run_id = %s
      AND state = 'claimed'
      AND active_lease_id = %s
      AND version = %s
    RETURNING {TASK_COLUMNS}
"""
COMPLETE_LEASE_SQL = f"""
    UPDATE task_leases
    SET status = 'completed',
        released_at = now(),
        completed_at = now(),
        version = version + 1
    WHERE id = %s
      AND task_id = %s
      AND tenant_id = %s
      AND run_id = %s
      AND agent_id = %s
      AND status = 'active'
      AND expires_at > now()
      AND version = %s
    RETURNING {LEASE_COLUMNS}
"""
DEPENDENCIES_READY_SQL = """
    SELECT NOT EXISTS (
        SELECT 1
        FROM task_dependencies AS dependency
        LEFT JOIN tasks AS prerequisite
          ON prerequisite.id = dependency.depends_on_task_id
         AND prerequisite.tenant_id = %s
         AND prerequisite.project_id = %s
         AND prerequisite.repository_id = %s
         AND prerequisite.swarm_id = %s
         AND prerequisite.run_id = %s
        WHERE dependency.task_id = %s
          AND dependency.kind = 'blocks'
          AND (prerequisite.id IS NULL OR prerequisite.state != 'completed')
    ) AS dependencies_ready
"""
RELEASE_TASK_SQL = f"""
    UPDATE tasks
    SET state = %s,
        claimed_by_agent_id = NULL,
        active_lease_id = NULL,
        updated_at = now(),
        version = version + 1
    WHERE id = %s
      AND tenant_id = %s
      AND project_id = %s
      AND repository_id = %s
      AND swarm_id = %s
      AND run_id = %s
      AND state = 'claimed'
      AND active_lease_id = %s
      AND version = %s
    RETURNING {TASK_COLUMNS}
"""
RELEASE_LEASE_SQL = f"""
    UPDATE task_leases
    SET status = 'released',
        released_at = now(),
        version = version + 1
    WHERE id = %s
      AND task_id = %s
      AND tenant_id = %s
      AND run_id = %s
      AND agent_id = %s
      AND status = 'active'
      AND expires_at > now()
      AND version = %s
    RETURNING {LEASE_COLUMNS}
"""
INSERT_SWARM_EVENT_SQL = """
    INSERT INTO swarm_events (
        id, tenant_id, project_id, repository_id, swarm_id, run_id,
        agent_id, task_id, event_type, aggregate_type, aggregate_id,
        aggregate_version, payload, occurred_at, recorded_at,
        correlation_id, causation_id, idempotency_key
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s
    )
    RETURNING id
"""
INSERT_OUTBOX_EVENT_SQL = """
    INSERT INTO outbox_events (
        id, event_id, tenant_id, run_id, aggregate_type, aggregate_id,
        aggregate_version, event_type, dedupe_key, payload, status,
        attempts, available_at, occurred_at, version
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        'pending', 0, %s, %s, 1
    )
    RETURNING id
"""
EVENT_PAGE_FIRST_SQL = f"""
    SELECT {EVENT_COLUMNS}
    FROM swarm_events
    WHERE tenant_id = %s
      AND project_id = %s
      AND repository_id = %s
      AND swarm_id = %s
      AND run_id = %s
    ORDER BY occurred_at, id
    LIMIT %s
"""
EVENT_PAGE_AFTER_SQL = f"""
    SELECT {EVENT_COLUMNS}
    FROM swarm_events
    WHERE tenant_id = %s
      AND project_id = %s
      AND repository_id = %s
      AND swarm_id = %s
      AND run_id = %s
      AND (occurred_at, id) > (%s, %s)
    ORDER BY occurred_at, id
    LIMIT %s
"""
RUN_METRICS_SQL = """
    WITH scoped_run AS (
        SELECT tenant_id, id, project_id, repository_id, swarm_id
        FROM runs
        WHERE tenant_id = %s
          AND id = %s
          AND project_id = %s
          AND repository_id = %s
          AND swarm_id = %s
    )
    SELECT
        now() AS measured_at,
        (SELECT count(*) FROM tasks AS task
         WHERE task.tenant_id = run.tenant_id
           AND task.project_id = run.project_id
           AND task.repository_id = run.repository_id
           AND task.swarm_id = run.swarm_id
           AND task.run_id = run.id) AS tasks_total,
        (SELECT count(*) FROM tasks AS task
         WHERE task.tenant_id = run.tenant_id
           AND task.project_id = run.project_id
           AND task.repository_id = run.repository_id
           AND task.swarm_id = run.swarm_id
           AND task.run_id = run.id
           AND task.state = 'claimed') AS tasks_claimed,
        (SELECT count(*) FROM tasks AS task
         WHERE task.tenant_id = run.tenant_id
           AND task.project_id = run.project_id
           AND task.repository_id = run.repository_id
           AND task.swarm_id = run.swarm_id
           AND task.run_id = run.id
           AND task.state = 'completed') AS tasks_completed,
        (SELECT count(*) FROM tasks AS task
         WHERE task.tenant_id = run.tenant_id
           AND task.project_id = run.project_id
           AND task.repository_id = run.repository_id
           AND task.swarm_id = run.swarm_id
           AND task.run_id = run.id
           AND task.state = 'failed') AS tasks_failed,
        (SELECT count(*)
         FROM task_leases AS lease
         JOIN tasks AS task ON task.id = lease.task_id
         WHERE task.tenant_id = run.tenant_id
           AND task.project_id = run.project_id
           AND task.repository_id = run.repository_id
           AND task.swarm_id = run.swarm_id
           AND task.run_id = run.id
           AND lease.tenant_id = run.tenant_id
           AND lease.run_id = run.id
           AND lease.status = 'active'
           AND lease.expires_at > now()) AS active_leases,
        (SELECT count(*)
         FROM task_checkpoints AS checkpoint
         JOIN tasks AS task ON task.id = checkpoint.task_id
         WHERE task.tenant_id = run.tenant_id
           AND task.project_id = run.project_id
           AND task.repository_id = run.repository_id
           AND task.swarm_id = run.swarm_id
           AND task.run_id = run.id
           AND checkpoint.tenant_id = run.tenant_id
           AND checkpoint.run_id = run.id) AS checkpoints,
        (SELECT count(*)
         FROM task_leases AS lease
         JOIN tasks AS task ON task.id = lease.task_id
         WHERE task.tenant_id = run.tenant_id
           AND task.project_id = run.project_id
           AND task.repository_id = run.repository_id
           AND task.swarm_id = run.swarm_id
           AND task.run_id = run.id
           AND lease.tenant_id = run.tenant_id
           AND lease.run_id = run.id
           AND lease.status = 'expired'
           AND EXISTS (
               SELECT 1 FROM task_checkpoints AS checkpoint
               WHERE checkpoint.task_id = lease.task_id
           )) AS crash_handoffs,
        (SELECT count(*) FROM memories AS memory
         WHERE memory.tenant_id = run.tenant_id
           AND memory.project_id = run.project_id
           AND memory.repository_id = run.repository_id
           AND memory.run_id = run.id
           AND memory.recorded_to IS NULL) AS memories_published,
        (SELECT count(*) FROM memory_conflicts AS conflict
         WHERE conflict.tenant_id = run.tenant_id
           AND conflict.project_id = run.project_id
           AND conflict.repository_id = run.repository_id
           AND conflict.swarm_id = run.swarm_id
           AND conflict.run_id = run.id
           AND conflict.state = 'open') AS conflicts_open,
        (SELECT count(*) FROM memory_conflicts AS conflict
         WHERE conflict.tenant_id = run.tenant_id
           AND conflict.project_id = run.project_id
           AND conflict.repository_id = run.repository_id
           AND conflict.swarm_id = run.swarm_id
           AND conflict.run_id = run.id
           AND conflict.state = 'resolved') AS conflicts_resolved,
        COALESCE((
            SELECT sum(replay.replay_count)
            FROM (
                SELECT greatest(count(*) - 1, 0) AS replay_count
                FROM action_attempts AS attempt
                WHERE attempt.tenant_id = run.tenant_id
                  AND attempt.run_id = run.id
                  AND attempt.status = 'succeeded'
                  AND attempt.idempotency_record_id IS NOT NULL
                GROUP BY attempt.idempotency_record_id
            ) AS replay
        ), 0) AS duplicate_mutations_replayed
    FROM scoped_run AS run
"""


class _WriteAck(ContractModel):
    ok: bool = True


class CockroachCoordinationStore:
    """Typed CockroachDB implementation of the coordination repository port."""

    def __init__(self, database: CockroachDatabase) -> None:
        self.database = database

    async def ensure_run(self, actor: ActorContext) -> None:
        """Explicit administrative bootstrap used before joining a new run."""

        async def transaction(connection: Any) -> _WriteAck:
            await _ensure_run_scope(
                connection,
                tenant_id=actor.tenant_id,
                project_id=actor.project_id,
                repository_id=actor.repository_id,
                swarm_id=actor.swarm_id,
                run_id=actor.run_id,
            )
            return _WriteAck()

        await self.database.run(transaction)

    async def join_agent(self, actor: ActorContext) -> Agent:
        event_id = str(uuid4())
        outbox_id = str(uuid4())

        async def transaction(connection: Any) -> Agent:
            await _ensure_run_scope(
                connection,
                tenant_id=actor.tenant_id,
                project_id=actor.project_id,
                repository_id=actor.repository_id,
                swarm_id=actor.swarm_id,
                run_id=actor.run_id,
            )
            insert_cursor = await connection.execute(
                INSERT_AGENT_SQL,
                (
                    actor.tenant_id,
                    actor.run_id,
                    actor.agent_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.harness,
                    actor.provider,
                    actor.model,
                    sorted(actor.capabilities),
                    Jsonb(actor.metadata),
                ),
            )
            row = await insert_cursor.fetchone()
            inserted = row is not None
            if row is None:
                update_cursor = await connection.execute(
                    UPDATE_AGENT_SQL,
                    (
                        actor.harness,
                        actor.provider,
                        actor.model,
                        sorted(actor.capabilities),
                        Jsonb(actor.metadata),
                        actor.tenant_id,
                        actor.run_id,
                        actor.agent_id,
                        actor.project_id,
                        actor.repository_id,
                        actor.swarm_id,
                    ),
                )
                row = await update_cursor.fetchone()
            if row is None:
                raise ResourceNotFound("agent", actor.agent_id)
            agent = _agent_from_row(row)
            if inserted:
                await _append_event(
                    connection,
                    actor,
                    event_id=event_id,
                    outbox_id=outbox_id,
                    event_type=EventType.AGENT_JOINED,
                    aggregate_type=AggregateType.AGENT,
                    aggregate_id=agent.agent_id,
                    aggregate_version=agent.version,
                    occurred_at=agent.joined_at,
                    payload={},
                )
            return agent

        return await self.database.run(transaction)

    async def add_task(self, task: Task) -> Task:
        async def transaction(connection: Any) -> Task:
            await _ensure_run_scope(
                connection,
                tenant_id=task.tenant_id,
                project_id=task.project_id,
                repository_id=task.repository_id,
                swarm_id=task.swarm_id,
                run_id=task.run_id,
            )
            cursor = await connection.execute(
                INSERT_TASK_SQL,
                (
                    task.task_id,
                    task.tenant_id,
                    task.project_id,
                    task.repository_id,
                    task.swarm_id,
                    task.run_id,
                    task.title,
                    task.description,
                    task.status.value,
                    task.priority,
                    task.version,
                    sorted(task.required_capabilities),
                    list(task.tags),
                    task.created_by_agent_id,
                    task.claimed_by_agent_id,
                    task.active_lease_id,
                    task.available_at,
                    Jsonb(task.metadata),
                    task.created_at,
                    task.updated_at,
                    task.completed_at,
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                raise InvalidState("task already exists", task_id=task.task_id)
            return _task_from_row(row)

        return await self.database.run(transaction)

    async def add_dependency(self, dependency: TaskDependency) -> TaskDependency:
        async def transaction(connection: Any) -> TaskDependency:
            lock_cursor = await connection.execute(
                LOCK_DEPENDENCY_TASKS_SQL,
                ([dependency.task_id, dependency.depends_on_task_id],),
            )
            rows = await lock_cursor.fetchall()
            known = {str(row["id"]): row for row in rows}
            for task_id in (dependency.task_id, dependency.depends_on_task_id):
                if task_id not in known:
                    raise ResourceNotFound("task", task_id)
            scope_fields = (
                "tenant_id",
                "project_id",
                "repository_id",
                "swarm_id",
                "run_id",
            )
            task_scope = tuple(str(known[dependency.task_id][field]) for field in scope_fields)
            prerequisite_scope = tuple(
                str(known[dependency.depends_on_task_id][field]) for field in scope_fields
            )
            if task_scope != prerequisite_scope:
                raise InvalidState("task dependencies must stay in one authenticated scope")

            cycle_cursor = await connection.execute(
                DEPENDENCY_CYCLE_SQL,
                (dependency.depends_on_task_id, dependency.task_id),
            )
            cycle_row = await cycle_cursor.fetchone()
            if cycle_row is not None and bool(cycle_row["creates_cycle"]):
                raise InvalidState("task dependency would create a cycle")

            insert_cursor = await connection.execute(
                INSERT_DEPENDENCY_SQL,
                (
                    dependency.task_id,
                    dependency.depends_on_task_id,
                    dependency.kind.value,
                    dependency.created_at,
                ),
            )
            row = await insert_cursor.fetchone()
            if row is None:
                existing_cursor = await connection.execute(
                    SELECT_DEPENDENCY_SQL,
                    (dependency.task_id, dependency.depends_on_task_id),
                )
                row = await existing_cursor.fetchone()
            if row is None:
                raise InvalidState("task dependency could not be persisted")
            persisted = _dependency_from_row(row)
            if persisted.kind is not dependency.kind:
                raise InvalidState("task dependency already exists with another kind")
            return persisted

        return await self.database.run(transaction)

    async def claim_task(
        self,
        actor: ActorContext,
        command: ClaimTaskCommand,
    ) -> ClaimTaskResult:
        lease_id = str(uuid4())
        event_id = str(uuid4())
        outbox_id = str(uuid4())

        async def transaction(connection: Any) -> ClaimTaskResult:
            expired_cursor = await connection.execute(
                EXPIRE_LEASES_SQL,
                (
                    actor.tenant_id,
                    actor.run_id,
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                ),
            )
            expired = await expired_cursor.fetchall()
            expired_lease_ids = [str(row["id"]) for row in expired]
            if expired_lease_ids:
                await connection.execute(
                    RELEASE_EXPIRED_TASKS_SQL,
                    (
                        actor.tenant_id,
                        actor.project_id,
                        actor.repository_id,
                        actor.swarm_id,
                        actor.run_id,
                        expired_lease_ids,
                    ),
                )

            candidate_cursor = await connection.execute(
                CLAIM_CANDIDATE_SQL,
                _claim_filter_params(actor, command),
            )
            candidate_row = await candidate_cursor.fetchone()
            if candidate_row is None:
                await _raise_claim_miss(connection, actor, command)
            candidate = _task_from_row(candidate_row)
            if (
                command.expected_task_version is not None
                and candidate.version != command.expected_task_version
            ):
                raise InvalidState(
                    "task version changed before claim",
                    task_id=candidate.task_id,
                    expected_version=command.expected_task_version,
                    actual_version=candidate.version,
                )

            task_cursor = await connection.execute(
                MARK_TASK_CLAIMED_SQL,
                (
                    actor.agent_id,
                    lease_id,
                    candidate.task_id,
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                    candidate.version,
                ),
            )
            task_row = await task_cursor.fetchone()
            if task_row is None:
                raise NoTaskAvailable()
            task = _task_from_row(task_row)

            lease_cursor = await connection.execute(
                INSERT_LEASE_SQL,
                (
                    lease_id,
                    task.task_id,
                    actor.tenant_id,
                    actor.run_id,
                    actor.agent_id,
                    timedelta(seconds=command.lease_seconds),
                    Jsonb({}),
                ),
            )
            lease_row = await lease_cursor.fetchone()
            if lease_row is None:
                raise NoTaskAvailable()
            lease = _lease_from_row(lease_row)
            checkpoint = await _latest_checkpoint(connection, actor, task.task_id)
            await _append_event(
                connection,
                actor,
                event_id=event_id,
                outbox_id=outbox_id,
                event_type=EventType.TASK_CLAIMED,
                aggregate_type=AggregateType.TASK,
                aggregate_id=task.task_id,
                aggregate_version=task.version,
                occurred_at=lease.acquired_at,
                task_id=task.task_id,
                payload={"lease_id": lease.lease_id},
                idempotency_key=command.idempotency_key,
            )
            return ClaimTaskResult(task=task, lease=lease, checkpoint=checkpoint)

        return await self.database.run_idempotent(
            actor,
            "task.claim",
            command,
            ClaimTaskResult,
            transaction,
        )

    async def renew_lease(
        self,
        actor: ActorContext,
        command: RenewLeaseCommand,
    ) -> RenewLeaseResult:
        event_id = str(uuid4())
        outbox_id = str(uuid4())

        async def transaction(connection: Any) -> RenewLeaseResult:
            cursor = await connection.execute(
                RENEW_LEASE_SQL,
                (
                    timedelta(seconds=command.extension_seconds),
                    command.lease_id,
                    actor.tenant_id,
                    actor.run_id,
                    actor.agent_id,
                    command.expected_version,
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                raise LeaseLost(command.lease_id)
            lease = _lease_from_row(row)
            await _append_event(
                connection,
                actor,
                event_id=event_id,
                outbox_id=outbox_id,
                event_type=EventType.LEASE_RENEWED,
                aggregate_type=AggregateType.LEASE,
                aggregate_id=lease.lease_id,
                aggregate_version=lease.version,
                occurred_at=lease.renewed_at or lease.acquired_at,
                task_id=lease.task_id,
                payload={"expires_at": lease.expires_at.isoformat()},
                idempotency_key=command.idempotency_key,
            )
            return RenewLeaseResult(lease=lease)

        return await self.database.run_idempotent(
            actor,
            "lease.renew",
            command,
            RenewLeaseResult,
            transaction,
        )

    async def checkpoint_task(
        self,
        actor: ActorContext,
        command: CheckpointCommand,
    ) -> CheckpointResult:
        checkpoint_id = str(uuid4())
        event_id = str(uuid4())
        outbox_id = str(uuid4())

        async def transaction(connection: Any) -> CheckpointResult:
            await _validate_memory_ids(connection, actor, command.memory_ids)
            task, lease = await _load_owned_task_and_lease(
                connection,
                actor,
                task_id=command.task_id,
                lease_id=command.lease_id,
                expected_task_version=command.expected_task_version,
                expected_lease_version=command.expected_lease_version,
            )
            sequence_cursor = await connection.execute(
                NEXT_CHECKPOINT_SEQUENCE_SQL,
                (command.task_id, actor.tenant_id, actor.run_id),
            )
            sequence_row = await sequence_cursor.fetchone()
            sequence = 1 if sequence_row is None else int(sequence_row["sequence"]) + 1
            checkpoint_cursor = await connection.execute(
                INSERT_CHECKPOINT_SQL,
                (
                    checkpoint_id,
                    command.task_id,
                    command.lease_id,
                    actor.tenant_id,
                    actor.run_id,
                    actor.agent_id,
                    sequence,
                    command.summary,
                    Jsonb(list(command.discoveries)),
                    Jsonb(list(command.completed_work)),
                    Jsonb(list(command.remaining_work)),
                    list(command.memory_ids),
                    list(command.artifact_ids),
                    Jsonb(command.state),
                ),
            )
            checkpoint_row = await checkpoint_cursor.fetchone()
            if checkpoint_row is None:
                raise InvalidState("checkpoint could not be persisted", task_id=command.task_id)
            checkpoint = _checkpoint_from_row(checkpoint_row)
            task = await _advance_task(connection, actor, task, lease.lease_id)
            lease = await _advance_lease(connection, actor, task.task_id, lease)
            await _append_event(
                connection,
                actor,
                event_id=event_id,
                outbox_id=outbox_id,
                event_type=EventType.TASK_CHECKPOINTED,
                aggregate_type=AggregateType.TASK,
                aggregate_id=task.task_id,
                aggregate_version=task.version,
                occurred_at=checkpoint.created_at,
                task_id=task.task_id,
                payload={"checkpoint_id": checkpoint.checkpoint_id},
                idempotency_key=command.idempotency_key,
            )
            return CheckpointResult(task=task, lease=lease, checkpoint=checkpoint)

        return await self.database.run_idempotent(
            actor,
            "task.checkpoint",
            command,
            CheckpointResult,
            transaction,
        )

    async def complete_task(
        self,
        actor: ActorContext,
        command: CompleteTaskCommand,
    ) -> CompletionResult:
        completion_id = str(uuid4())
        event_id = str(uuid4())
        outbox_id = str(uuid4())

        async def transaction(connection: Any) -> CompletionResult:
            await _validate_memory_ids(connection, actor, command.memory_ids)
            task, lease = await _load_owned_task_and_lease(
                connection,
                actor,
                task_id=command.task_id,
                lease_id=command.lease_id,
                expected_task_version=command.expected_task_version,
                expected_lease_version=command.expected_lease_version,
            )
            completion_cursor = await connection.execute(
                INSERT_COMPLETION_SQL,
                (
                    completion_id,
                    task.task_id,
                    lease.lease_id,
                    actor.tenant_id,
                    actor.run_id,
                    actor.agent_id,
                    command.outcome.value,
                    command.summary,
                    list(command.memory_ids),
                    list(command.artifact_ids),
                ),
            )
            completion_row = await completion_cursor.fetchone()
            if completion_row is None:
                raise InvalidState("task already has a completion", task_id=task.task_id)
            completion = _completion_from_row(completion_row)
            task_status = (
                TaskStatus.COMPLETED if command.outcome.value == "succeeded" else TaskStatus.FAILED
            )
            task_cursor = await connection.execute(
                COMPLETE_TASK_SQL,
                (
                    task_status.value,
                    task.task_id,
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                    lease.lease_id,
                    task.version,
                ),
            )
            task_row = await task_cursor.fetchone()
            if task_row is None:
                raise LeaseLost(lease.lease_id, "task version changed")
            task = _task_from_row(task_row)
            lease_cursor = await connection.execute(
                COMPLETE_LEASE_SQL,
                (
                    lease.lease_id,
                    task.task_id,
                    actor.tenant_id,
                    actor.run_id,
                    actor.agent_id,
                    lease.version,
                ),
            )
            lease_row = await lease_cursor.fetchone()
            if lease_row is None:
                raise LeaseLost(lease.lease_id, "lease version changed")
            lease = _lease_from_row(lease_row)
            await _append_event(
                connection,
                actor,
                event_id=event_id,
                outbox_id=outbox_id,
                event_type=EventType.TASK_COMPLETED,
                aggregate_type=AggregateType.TASK,
                aggregate_id=task.task_id,
                aggregate_version=task.version,
                occurred_at=completion.completed_at,
                task_id=task.task_id,
                payload={"outcome": command.outcome.value},
                idempotency_key=command.idempotency_key,
            )
            return CompletionResult(task=task, lease=lease, completion=completion)

        return await self.database.run_idempotent(
            actor,
            "task.complete",
            command,
            CompletionResult,
            transaction,
        )

    async def release_task(
        self,
        actor: ActorContext,
        command: ReleaseTaskCommand,
    ) -> ReleaseResult:
        event_id = str(uuid4())
        outbox_id = str(uuid4())

        async def transaction(connection: Any) -> ReleaseResult:
            task, lease = await _load_owned_task_and_lease(
                connection,
                actor,
                task_id=command.task_id,
                lease_id=command.lease_id,
                expected_task_version=command.expected_task_version,
                expected_lease_version=command.expected_lease_version,
            )
            ready_cursor = await connection.execute(
                DEPENDENCIES_READY_SQL,
                (
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                    task.task_id,
                ),
            )
            ready_row = await ready_cursor.fetchone()
            ready = ready_row is not None and bool(ready_row["dependencies_ready"])
            next_status = TaskStatus.READY if ready else TaskStatus.BLOCKED
            task_cursor = await connection.execute(
                RELEASE_TASK_SQL,
                (
                    next_status.value,
                    task.task_id,
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                    lease.lease_id,
                    task.version,
                ),
            )
            task_row = await task_cursor.fetchone()
            if task_row is None:
                raise LeaseLost(lease.lease_id, "task version changed")
            task = _task_from_row(task_row)
            lease_cursor = await connection.execute(
                RELEASE_LEASE_SQL,
                (
                    lease.lease_id,
                    task.task_id,
                    actor.tenant_id,
                    actor.run_id,
                    actor.agent_id,
                    lease.version,
                ),
            )
            lease_row = await lease_cursor.fetchone()
            if lease_row is None:
                raise LeaseLost(lease.lease_id, "lease version changed")
            lease = _lease_from_row(lease_row)
            await _append_event(
                connection,
                actor,
                event_id=event_id,
                outbox_id=outbox_id,
                event_type=EventType.TASK_RELEASED,
                aggregate_type=AggregateType.TASK,
                aggregate_id=task.task_id,
                aggregate_version=task.version,
                occurred_at=lease.released_at or task.updated_at,
                task_id=task.task_id,
                payload={"reason": command.reason},
                idempotency_key=command.idempotency_key,
            )
            return ReleaseResult(task=task, lease=lease)

        return await self.database.run_idempotent(
            actor,
            "task.release",
            command,
            ReleaseResult,
            transaction,
        )

    async def list_run_events(
        self,
        actor: ActorContext,
        run_id: RunId,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> EventPage:
        if run_id != actor.run_id:
            raise ResourceNotFound("run", run_id)
        if not 1 <= limit <= 1000:
            raise ValueError("event page limit must be between 1 and 1000")
        decoded_cursor = _decode_event_cursor(cursor) if cursor is not None else None

        async def transaction(connection: Any) -> EventPage:
            params: tuple[Any, ...] = (
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                run_id,
            )
            if decoded_cursor is None:
                query = EVENT_PAGE_FIRST_SQL
                query_params = (*params, limit + 1)
            else:
                query = EVENT_PAGE_AFTER_SQL
                query_params = (*params, decoded_cursor[0], decoded_cursor[1], limit + 1)
            event_cursor = await connection.execute(query, query_params)
            rows = await event_cursor.fetchall()
            has_more = len(rows) > limit
            page_rows = rows[:limit]
            events = tuple(_event_from_row(row) for row in page_rows)
            next_cursor = None
            if has_more and events:
                last = events[-1]
                next_cursor = _encode_event_cursor(last.occurred_at, last.event_id)
            return EventPage(events=events, next_cursor=next_cursor)

        return await self.database.run(transaction)

    async def get_run_metrics(
        self,
        actor: ActorContext,
        run_id: RunId,
    ) -> RunMetrics:
        if run_id != actor.run_id:
            raise ResourceNotFound("run", run_id)

        async def transaction(connection: Any) -> RunMetrics:
            cursor = await connection.execute(
                RUN_METRICS_SQL,
                (
                    actor.tenant_id,
                    run_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                raise ResourceNotFound("run", run_id)
            return RunMetrics(
                run_id=run_id,
                measured_at=row["measured_at"],
                tasks_total=int(row["tasks_total"]),
                tasks_claimed=int(row["tasks_claimed"]),
                tasks_completed=int(row["tasks_completed"]),
                tasks_failed=int(row["tasks_failed"]),
                active_leases=int(row["active_leases"]),
                checkpoints=int(row["checkpoints"]),
                crash_handoffs=int(row["crash_handoffs"]),
                memories_published=int(row["memories_published"]),
                memories_reused=0,
                conflicts_open=int(row["conflicts_open"]),
                conflicts_resolved=int(row["conflicts_resolved"]),
                duplicate_claims_prevented=0,
                duplicate_mutations_replayed=int(row["duplicate_mutations_replayed"]),
            )

        return await self.database.run(transaction)


async def _ensure_run_scope(
    connection: Any,
    *,
    tenant_id: str,
    project_id: str,
    repository_id: str,
    swarm_id: str,
    run_id: str,
) -> None:
    await connection.execute(
        INSERT_RUN_SQL,
        (tenant_id, run_id, project_id, repository_id, swarm_id, Jsonb({})),
    )
    cursor = await connection.execute(
        SELECT_RUN_SCOPE_SQL,
        (tenant_id, run_id, project_id, repository_id, swarm_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise InvalidState("run identity exists under another authenticated scope", run_id=run_id)
    if str(row["state"]) != "active":
        raise InvalidState("run is not active", run_id=run_id, state=str(row["state"]))


async def _raise_claim_miss(
    connection: Any,
    actor: ActorContext,
    command: ClaimTaskCommand,
) -> None:
    if command.task_id is not None:
        cursor = await connection.execute(
            SELECT_CLAIM_TASK_STATE_SQL,
            (
                command.task_id,
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ),
        )
        row = await cursor.fetchone()
        if row is None:
            raise ResourceNotFound("task", command.task_id)
        if command.expected_task_version is not None and int(row["version"]) != (
            command.expected_task_version
        ):
            raise InvalidState(
                "task version changed before claim",
                task_id=command.task_id,
                expected_version=command.expected_task_version,
                actual_version=int(row["version"]),
            )

    cursor = await connection.execute(
        CLAIM_ELIGIBLE_SQL,
        _claim_filter_params(actor, command),
    )
    row = await cursor.fetchone()
    if row is not None and bool(row["eligible"]):
        raise _ClaimContention("eligible task was temporarily locked by another claimant")
    raise NoTaskAvailable()


def _claim_filter_params(
    actor: ActorContext,
    command: ClaimTaskCommand,
) -> tuple[Any, ...]:
    return (
        actor.tenant_id,
        actor.project_id,
        actor.repository_id,
        actor.swarm_id,
        actor.run_id,
        command.task_id,
        command.task_id,
        list(command.required_tags),
        sorted(command.required_capabilities),
        sorted(actor.capabilities),
    )


async def _latest_checkpoint(
    connection: Any,
    actor: ActorContext,
    task_id: str,
) -> TaskCheckpoint | None:
    cursor = await connection.execute(
        LATEST_CHECKPOINT_SQL,
        (
            task_id,
            actor.tenant_id,
            actor.run_id,
            actor.tenant_id,
            actor.project_id,
            actor.repository_id,
            actor.swarm_id,
            actor.run_id,
        ),
    )
    row = await cursor.fetchone()
    return None if row is None else _checkpoint_from_row(row)


async def _load_owned_task_and_lease(
    connection: Any,
    actor: ActorContext,
    *,
    task_id: str,
    lease_id: str,
    expected_task_version: int,
    expected_lease_version: int,
) -> tuple[Task, TaskLease]:
    task_cursor = await connection.execute(
        LOAD_TASK_FOR_UPDATE_SQL,
        (
            task_id,
            actor.tenant_id,
            actor.project_id,
            actor.repository_id,
            actor.swarm_id,
            actor.run_id,
        ),
    )
    task_row = await task_cursor.fetchone()
    if task_row is None:
        raise ResourceNotFound("task", task_id)
    task = _task_from_row(task_row)
    lease_cursor = await connection.execute(
        LOAD_LEASE_FOR_UPDATE_SQL,
        (lease_id, task_id, actor.tenant_id, actor.run_id, actor.agent_id),
    )
    lease_row = await lease_cursor.fetchone()
    if lease_row is None:
        raise LeaseLost(lease_id)
    lease = _lease_from_row(lease_row)
    if (
        task.status is not TaskStatus.CLAIMED
        or task.active_lease_id != lease_id
        or task.claimed_by_agent_id != actor.agent_id
    ):
        raise LeaseLost(lease_id)
    if task.version != expected_task_version:
        raise LeaseLost(lease_id, "task version changed")
    if lease.version != expected_lease_version:
        raise LeaseLost(lease_id, "lease version changed")
    return task, lease


async def _validate_memory_ids(
    connection: Any,
    actor: ActorContext,
    memory_ids: tuple[str, ...],
) -> None:
    if not memory_ids:
        return
    cursor = await connection.execute(
        SCOPED_MEMORY_IDS_SQL,
        (
            list(memory_ids),
            actor.tenant_id,
            actor.project_id,
            actor.repository_id,
            actor.swarm_id,
            actor.run_id,
        ),
    )
    rows = await cursor.fetchall()
    found = {str(row["id"]) for row in rows}
    missing = next((memory_id for memory_id in memory_ids if memory_id not in found), None)
    if missing is not None:
        raise ResourceNotFound("memory", missing)


async def _advance_task(
    connection: Any,
    actor: ActorContext,
    task: Task,
    lease_id: str,
) -> Task:
    cursor = await connection.execute(
        ADVANCE_TASK_SQL,
        (
            task.task_id,
            actor.tenant_id,
            actor.project_id,
            actor.repository_id,
            actor.swarm_id,
            actor.run_id,
            lease_id,
            task.version,
        ),
    )
    row = await cursor.fetchone()
    if row is None:
        raise LeaseLost(lease_id, "task version changed")
    return _task_from_row(row)


async def _advance_lease(
    connection: Any,
    actor: ActorContext,
    task_id: str,
    lease: TaskLease,
) -> TaskLease:
    cursor = await connection.execute(
        ADVANCE_LEASE_SQL,
        (
            lease.lease_id,
            task_id,
            actor.tenant_id,
            actor.run_id,
            actor.agent_id,
            lease.version,
        ),
    )
    row = await cursor.fetchone()
    if row is None:
        raise LeaseLost(lease.lease_id, "lease version changed")
    return _lease_from_row(row)


async def _append_event(
    connection: Any,
    actor: ActorContext,
    *,
    event_id: str,
    outbox_id: str,
    event_type: EventType,
    aggregate_type: AggregateType,
    aggregate_id: str,
    aggregate_version: int,
    occurred_at: datetime,
    task_id: str | None = None,
    payload: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> SwarmEvent:
    event = SwarmEvent(
        event_id=event_id,
        event_type=event_type,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        tenant_id=actor.tenant_id,
        project_id=actor.project_id,
        repository_id=actor.repository_id,
        swarm_id=actor.swarm_id,
        run_id=actor.run_id,
        agent_id=actor.agent_id,
        task_id=task_id,
        aggregate_version=aggregate_version,
        payload=payload or {},
        occurred_at=occurred_at,
        recorded_at=occurred_at,
        idempotency_key=idempotency_key,
    )
    event_cursor = await connection.execute(
        INSERT_SWARM_EVENT_SQL,
        (
            event.event_id,
            event.tenant_id,
            event.project_id,
            event.repository_id,
            event.swarm_id,
            event.run_id,
            event.agent_id,
            event.task_id,
            str(event.event_type),
            event.aggregate_type.value,
            event.aggregate_id,
            event.aggregate_version,
            Jsonb(event.payload),
            event.occurred_at,
            event.recorded_at,
            event.correlation_id,
            event.causation_id,
            event.idempotency_key,
        ),
    )
    if await event_cursor.fetchone() is None:
        raise InvalidState("swarm event could not be persisted", event_id=event.event_id)
    outbox_cursor = await connection.execute(
        INSERT_OUTBOX_EVENT_SQL,
        (
            outbox_id,
            event.event_id,
            event.tenant_id,
            event.run_id,
            event.aggregate_type.value,
            event.aggregate_id,
            event.aggregate_version,
            str(event.event_type),
            event.event_id,
            Jsonb(event.model_dump(mode="json")),
            event.occurred_at,
            event.occurred_at,
        ),
    )
    if await outbox_cursor.fetchone() is None:
        raise InvalidState("outbox event could not be persisted", event_id=event.event_id)
    return event


def _task_from_row(row: Mapping[str, Any]) -> Task:
    return Task(
        task_id=row["id"],
        tenant_id=row["tenant_id"],
        project_id=row["project_id"],
        repository_id=row["repository_id"],
        swarm_id=row["swarm_id"],
        run_id=row["run_id"],
        title=row["title"],
        description=row["description"],
        status=TaskStatus(str(row["state"])),
        priority=int(row["priority"]),
        tags=tuple(row["tags"] or ()),
        required_capabilities=frozenset(row["required_capabilities"] or ()),
        created_by_agent_id=row["created_by_agent_id"],
        claimed_by_agent_id=row["claimed_by_agent_id"],
        active_lease_id=row["active_lease_id"],
        available_at=row["available_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
        version=int(row["version"]),
        metadata=_json_object(row["metadata"]),
    )


def _lease_from_row(row: Mapping[str, Any]) -> TaskLease:
    renewed_at = row["renewed_at"]
    if renewed_at == row["claimed_at"]:
        renewed_at = None
    return TaskLease(
        lease_id=row["id"],
        task_id=row["task_id"],
        run_id=row["run_id"],
        owner_agent_id=row["agent_id"],
        status=LeaseStatus(str(row["status"])),
        acquired_at=row["claimed_at"],
        renewed_at=renewed_at,
        expires_at=row["expires_at"],
        released_at=row["released_at"],
        version=int(row["version"]),
    )


def _checkpoint_from_row(row: Mapping[str, Any]) -> TaskCheckpoint:
    return TaskCheckpoint(
        checkpoint_id=row["id"],
        task_id=row["task_id"],
        lease_id=row["lease_id"],
        run_id=row["run_id"],
        agent_id=row["agent_id"],
        sequence=int(row["sequence"]),
        summary=row["summary"],
        discoveries=_json_sequence(row["discoveries"]),
        completed_work=_json_sequence(row["completed_work"]),
        remaining_work=_json_sequence(row["remaining_work"]),
        memory_ids=_uuid_sequence(row["memory_ids"]),
        artifact_ids=_uuid_sequence(row["artifact_ids"]),
        state=_json_object(row["state"]),
        created_at=row["created_at"],
    )


def _completion_from_row(row: Mapping[str, Any]) -> TaskCompletion:
    return TaskCompletion(
        completion_id=row["id"],
        task_id=row["task_id"],
        lease_id=row["lease_id"],
        run_id=row["run_id"],
        agent_id=row["agent_id"],
        outcome=str(row["outcome"]),
        summary=row["summary"],
        memory_ids=_uuid_sequence(row["memory_ids"]),
        artifact_ids=_uuid_sequence(row["artifact_ids"]),
        completed_at=row["completed_at"],
    )


def _dependency_from_row(row: Mapping[str, Any]) -> TaskDependency:
    return TaskDependency(
        task_id=row["task_id"],
        depends_on_task_id=row["depends_on_task_id"],
        kind=DependencyKind(str(row["kind"])),
        created_at=row["created_at"],
    )


def _agent_from_row(row: Mapping[str, Any]) -> Agent:
    return Agent(
        tenant_id=row["tenant_id"],
        project_id=row["project_id"],
        repository_id=row["repository_id"],
        swarm_id=row["swarm_id"],
        run_id=row["run_id"],
        agent_id=row["id"],
        harness=row["harness"],
        provider=row["provider"],
        model=row["model"],
        capabilities=frozenset(row["capabilities"] or ()),
        status=AgentStatus(str(row["status"])),
        joined_at=row["joined_at"],
        last_seen_at=row["last_seen_at"],
        version=int(row["version"]),
        metadata=_json_object(row["metadata"]),
    )


def _event_from_row(row: Mapping[str, Any]) -> SwarmEvent:
    return SwarmEvent(
        event_id=row["id"],
        tenant_id=row["tenant_id"],
        project_id=row["project_id"],
        repository_id=row["repository_id"],
        swarm_id=row["swarm_id"],
        run_id=row["run_id"],
        agent_id=row["agent_id"],
        task_id=row["task_id"],
        event_type=str(row["event_type"]),
        aggregate_type=AggregateType(str(row["aggregate_type"])),
        aggregate_id=row["aggregate_id"],
        aggregate_version=int(row["aggregate_version"]),
        payload=_json_object(row["payload"]),
        occurred_at=row["occurred_at"],
        recorded_at=row["recorded_at"],
        correlation_id=row["correlation_id"],
        causation_id=row["causation_id"],
        idempotency_key=row["idempotency_key"],
    )


def _json_object(value: Any) -> dict[str, Any]:
    decoded = _json_value(value)
    if decoded is None:
        return {}
    if not isinstance(decoded, dict):
        raise ValueError("expected a JSON object from CockroachDB")
    return decoded


def _json_sequence(value: Any) -> tuple[str, ...]:
    decoded = _json_value(value)
    if decoded is None:
        return ()
    if not isinstance(decoded, (list, tuple)):
        raise ValueError("expected a JSON array from CockroachDB")
    return tuple(str(item) for item in decoded)


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def _uuid_sequence(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in (value or ()))


def _encode_event_cursor(occurred_at: datetime, event_id: str) -> str:
    payload = json.dumps(
        [occurred_at.astimezone(UTC).isoformat(), str(UUID(event_id))],
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_event_cursor(value: str) -> tuple[datetime, str]:
    try:
        padding = "=" * (-len(value) % 4)
        decoded = base64.urlsafe_b64decode((value + padding).encode("ascii"))
        payload = json.loads(decoded)
        if not isinstance(payload, list) or len(payload) != 2:
            raise ValueError
        occurred_at = datetime.fromisoformat(str(payload[0]))
        if occurred_at.tzinfo is None:
            raise ValueError
        return occurred_at.astimezone(UTC), str(UUID(str(payload[1])))
    except (ValueError, TypeError, json.JSONDecodeError, binascii.Error) as exc:
        raise InvalidState("invalid event cursor") from exc


__all__ = ["CockroachCoordinationStore"]
