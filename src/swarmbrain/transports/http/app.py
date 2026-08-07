"""Canonical authenticated HTTP routes over application services."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Annotated, Any, TypeVar
from uuid import UUID, uuid4

import httpx
from fastapi import Body, Depends, FastAPI, Header, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ValidationError

from swarmbrain.adapters.auth import ExpiredTokenError, InvalidTokenError
from swarmbrain.application.errors import (
    AuthenticationRequired,
    ErrorCode,
    InvalidAuthentication,
    InvalidState,
    ResourceNotFound,
    SwarmBrainError,
)
from swarmbrain.application.runtime import SwarmBrainRuntime
from swarmbrain.cli.demo import (
    REAL_TIME_LEASE_SECONDS,
    real_time_lease_expiry,
    run_demo_scenario,
)
from swarmbrain.demo import build_scenario
from swarmbrain.demo.scenario import DemoScenario
from swarmbrain.domain.agents import ActorContext, Agent, Capability
from swarmbrain.domain.conflicts import (
    ReportConflictCommand,
    ReportConflictResult,
    ResolveConflictCommand,
    ResolveConflictResult,
)
from swarmbrain.domain.events import EventPage, RunMetrics
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    Evidence,
    EvidenceSource,
    RegisterEvidenceSourceCommand,
    SourceTrust,
)
from swarmbrain.domain.extraction import IngestRawSourceCommand, SourceIngestResult
from swarmbrain.domain.leases import RenewLeaseCommand, RenewLeaseResult
from swarmbrain.domain.memory import (
    MemoryLineage,
    RecallBundle,
    RecallQuery,
    RememberCommand,
    RememberResult,
)
from swarmbrain.domain.tasks import (
    CheckpointCommand,
    CheckpointResult,
    ClaimTaskCommand,
    ClaimTaskResult,
    CompleteTaskCommand,
    CompletionResult,
    ReleaseResult,
    ReleaseTaskCommand,
)
from swarmbrain.domain.work import SourceExtractionStatus

from .console import console_html
from .contracts import (
    AddEvidenceBody,
    CheckpointBody,
    ClaimTaskBody,
    CompleteTaskBody,
    IngestSourceBody,
    RegisterEvidenceSourceBody,
    ReleaseTaskBody,
    RememberBody,
    RenewLeaseBody,
    ReportConflictBody,
    ResolveConflictBody,
)

CommandT = TypeVar("CommandT", bound=BaseModel)
JsonBody = Annotated[dict[str, Any], Body(default_factory=dict)]

# One-click demo guard rails. The trigger takes no body and no secret, so the
# only thing standing between it and a scenario storm is this in-process gate.
DEMO_COOLDOWN_SECONDS = 30.0
DEMO_VIEWER_TTL = timedelta(minutes=30)
# A viewer token is a read-only credential for exactly the run the trigger just
# created: it can read that run's events and metrics and recall its memories,
# and nothing else. Handing that to whoever pressed the button is safe because
# they caused the run to exist; the grant reveals no pre-existing data, cannot
# mutate anything, and expires in thirty minutes.
DEMO_VIEWER_CAPABILITIES = frozenset(
    {
        Capability.EVENTS_READ.value,
        Capability.METRICS_READ.value,
        Capability.MEMORY_RECALL.value,
    }
)


@dataclass(slots=True)
class _DemoGate:
    """At most one scenario in flight, and no restart inside the cooldown."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task[Any] | None = None
    last_started_at: float | None = None

    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def cooldown_remaining(self) -> float:
        if self.last_started_at is None:
            return 0.0
        return max(0.0, DEMO_COOLDOWN_SECONDS - (time.monotonic() - self.last_started_at))


async def _authenticated_actor(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> ActorContext:
    runtime: SwarmBrainRuntime = request.app.state.runtime
    if not authorization:
        raise AuthenticationRequired()
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.casefold() != "bearer" or not token.strip():
        raise AuthenticationRequired("Authorization must use the Bearer scheme")
    try:
        return runtime.tokens.verify(token.strip())
    except ExpiredTokenError as exc:
        raise InvalidAuthentication("the bearer token has expired") from exc
    except InvalidTokenError as exc:
        raise InvalidAuthentication() from exc


ActorDependency = Annotated[ActorContext, Depends(_authenticated_actor)]
IdempotencyHeader = Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=255)]


def create_app(
    runtime: SwarmBrainRuntime,
    *,
    console_demo: bool = False,
    public_docs: bool = False,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(
        title="Swarm Brain",
        version="0.1.0",
        lifespan=lifespan,
        # The interactive docs pages load assets from third-party CDNs and the
        # schema is served anonymously, so the whole surface is opt-in.
        docs_url="/docs" if public_docs else None,
        redoc_url="/redoc" if public_docs else None,
        openapi_url="/openapi.json" if public_docs else None,
    )
    app.state.runtime = runtime
    app.state.console_demo = console_demo
    demo_gate = _DemoGate()
    app.state.console_demo_gate = demo_gate

    @app.exception_handler(SwarmBrainError)
    async def handle_domain_error(request: Request, exc: SwarmBrainError) -> JSONResponse:
        return _error_response(
            request, exc.status_code, exc.code, str(exc), exc.retryable, exc.details
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = {"fields": [_bounded_validation_error(item) for item in exc.errors()]}
        return _error_response(
            request,
            422,
            "validation_error",
            "request validation failed",
            False,
            details,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, _exc: Exception) -> JSONResponse:
        return _error_response(
            request,
            500,
            "internal_error",
            "an internal error occurred",
            False,
            {},
        )

    @app.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/console", include_in_schema=False)
    async def console() -> HTMLResponse:
        # Static document only; it carries no run data and no credentials, and
        # fetches everything from the bearer-authenticated routes in the browser.
        return HTMLResponse(console_html(), headers={"Cache-Control": "no-store"})

    @app.get("/console/demo/status", include_in_schema=False)
    async def console_demo_status() -> JSONResponse:
        # Absent unless the operator enabled it, so the console can probe for
        # the trigger and hide the button when a deployment has it switched off.
        if not console_demo:
            raise ResourceNotFound("route", "/console/demo/status")
        return JSONResponse(
            content={
                "enabled": True,
                "running": demo_gate.running(),
                "cooldown_seconds": round(demo_gate.cooldown_remaining(), 1),
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/console/demo", include_in_schema=False, status_code=202)
    async def console_demo_trigger(request: Request) -> JSONResponse:
        if not console_demo:
            raise ResourceNotFound("route", "/console/demo")

        async with demo_gate.lock:
            if demo_gate.running():
                return _error_response(
                    request,
                    429,
                    "demo_already_running",
                    "a demo scenario is already running; watch it or retry when it finishes",
                    True,
                    {"cooldown_seconds": round(demo_gate.cooldown_remaining(), 1)},
                )
            remaining = demo_gate.cooldown_remaining()
            if remaining > 0.0:
                return _error_response(
                    request,
                    429,
                    "demo_cooldown",
                    "the demo trigger accepts at most one scenario every 30 seconds",
                    True,
                    {"cooldown_seconds": round(remaining, 1)},
                )
            if runtime.coordination_store is None:
                raise InvalidState("the running backend cannot host the demo scenario")

            # The run identity always comes from a freshly built scenario, so the
            # trigger can never be aimed at a caller-chosen run: it takes no body.
            scenario = build_scenario()
            viewer = ActorContext(
                **scenario.scope(),
                agent_id=str(uuid4()),
                harness="swarm-console",
                provider="swarmbrain",
                model="viewer",
                capabilities=DEMO_VIEWER_CAPABILITIES,
            )
            viewer_token = runtime.tokens.issue(viewer, ttl=DEMO_VIEWER_TTL)
            demo_gate.last_started_at = time.monotonic()
            demo_gate.task = asyncio.create_task(_drive_console_demo(app, runtime, scenario))

        return JSONResponse(
            status_code=202,
            content={
                "run_id": scenario.run_id,
                "viewer_token": viewer_token,
                "expires_in": int(DEMO_VIEWER_TTL.total_seconds()),
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/readyz", include_in_schema=False)
    async def readiness() -> JSONResponse:
        try:
            ready = await runtime.ready()
        except Exception:
            ready = False
        if not ready:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return JSONResponse(content={"status": "ready"})

    @app.post("/v1/runs/{run_id}/agents:join", response_model=Agent)
    async def join_agent(run_id: str, actor: ActorDependency, body: JsonBody) -> Agent:
        if body:
            raise InvalidState("agent join body must be empty")
        if run_id != actor.run_id:
            raise ResourceNotFound("run", run_id)
        return await runtime.coordination.join(actor)

    @app.post("/v1/tasks:claim", response_model=ClaimTaskResult)
    async def claim_task(
        body: ClaimTaskBody,
        actor: ActorDependency,
        idempotency_key: IdempotencyHeader,
        response: Response,
    ) -> ClaimTaskResult:
        command = _command(ClaimTaskCommand, body, idempotency_key=idempotency_key)
        return _mark_replay(response, await runtime.coordination.claim(actor, command))

    @app.post("/v1/leases/{lease_id}:renew", response_model=RenewLeaseResult)
    async def renew_lease(
        lease_id: str,
        body: RenewLeaseBody,
        actor: ActorDependency,
        idempotency_key: IdempotencyHeader,
        response: Response,
    ) -> RenewLeaseResult:
        command = _command(
            RenewLeaseCommand,
            body,
            idempotency_key=idempotency_key,
            lease_id=lease_id,
        )
        return _mark_replay(response, await runtime.coordination.renew(actor, command))

    @app.post("/v1/memories", response_model=RememberResult)
    async def publish_memory(
        body: RememberBody,
        actor: ActorDependency,
        idempotency_key: IdempotencyHeader,
        response: Response,
    ) -> RememberResult:
        command = _command(RememberCommand, body, idempotency_key=idempotency_key)
        return _mark_replay(response, await runtime.memory.publish(actor, command))

    @app.post("/v1/memories:recall", response_model=RecallBundle)
    async def recall_memory(body: RecallQuery, actor: ActorDependency) -> RecallBundle:
        return await runtime.memory.recall(actor, body)

    @app.get("/v1/memories/{memory_id}/lineage", response_model=MemoryLineage)
    async def memory_lineage(memory_id: str, actor: ActorDependency) -> MemoryLineage:
        return await runtime.memory.lineage(actor, memory_id)

    @app.post(
        "/v1/sources:ingest",
        response_model=SourceIngestResult,
        status_code=202,
    )
    async def ingest_source(
        body: IngestSourceBody,
        actor: ActorDependency,
        idempotency_key: IdempotencyHeader,
        response: Response,
    ) -> SourceIngestResult:
        if runtime.extraction is None:
            raise InvalidState("durable source ingestion requires the cockroach backend")
        profile = runtime.ingestion_profile
        command = _command(
            IngestRawSourceCommand,
            body,
            idempotency_key=idempotency_key,
            observed_at=body.observed_at,
            trust=profile.trust,
            use_provider=profile.use_provider,
            chunk_chars=profile.chunk_chars,
            priority=profile.priority,
        )
        return _mark_replay(response, await runtime.extraction.ingest(actor, command))

    @app.get(
        "/v1/sources/{source_id}/extraction",
        response_model=SourceExtractionStatus,
    )
    async def source_extraction_status(
        source_id: UUID,
        actor: ActorDependency,
    ) -> SourceExtractionStatus:
        if runtime.extraction is None:
            raise InvalidState("durable source ingestion requires the cockroach backend")
        return await runtime.extraction.status(actor, str(source_id))

    @app.post("/v1/evidence/sources", response_model=EvidenceSource)
    async def register_evidence_source(
        body: RegisterEvidenceSourceBody,
        actor: ActorDependency,
        idempotency_key: IdempotencyHeader,
        response: Response,
    ) -> EvidenceSource:
        command = _command(
            RegisterEvidenceSourceCommand,
            body,
            idempotency_key=idempotency_key,
            trust=SourceTrust.UNKNOWN,
        )
        return _mark_replay(response, await runtime.evidence.register_source(actor, command))

    @app.post("/v1/evidence", response_model=Evidence)
    async def add_evidence(
        body: AddEvidenceBody,
        actor: ActorDependency,
        idempotency_key: IdempotencyHeader,
        response: Response,
    ) -> Evidence:
        command = _command(AddEvidenceCommand, body, idempotency_key=idempotency_key)
        return _mark_replay(response, await runtime.evidence.add_evidence(actor, command))

    @app.post("/v1/tasks/{task_id}/checkpoints", response_model=CheckpointResult)
    async def checkpoint_task(
        task_id: str,
        body: CheckpointBody,
        actor: ActorDependency,
        idempotency_key: IdempotencyHeader,
        response: Response,
    ) -> CheckpointResult:
        command = _command(
            CheckpointCommand,
            body,
            idempotency_key=idempotency_key,
            task_id=task_id,
        )
        return _mark_replay(response, await runtime.coordination.checkpoint(actor, command))

    @app.post("/v1/tasks/{task_id}:complete", response_model=CompletionResult)
    async def complete_task(
        task_id: str,
        body: CompleteTaskBody,
        actor: ActorDependency,
        idempotency_key: IdempotencyHeader,
        response: Response,
    ) -> CompletionResult:
        command = _command(
            CompleteTaskCommand,
            body,
            idempotency_key=idempotency_key,
            task_id=task_id,
        )
        return _mark_replay(response, await runtime.coordination.complete(actor, command))

    @app.post("/v1/tasks/{task_id}:release", response_model=ReleaseResult)
    async def release_task(
        task_id: str,
        body: ReleaseTaskBody,
        actor: ActorDependency,
        idempotency_key: IdempotencyHeader,
        response: Response,
    ) -> ReleaseResult:
        command = _command(
            ReleaseTaskCommand,
            body,
            idempotency_key=idempotency_key,
            task_id=task_id,
        )
        return _mark_replay(response, await runtime.coordination.release(actor, command))

    @app.post("/v1/conflicts", response_model=ReportConflictResult)
    async def report_conflict(
        body: ReportConflictBody,
        actor: ActorDependency,
        idempotency_key: IdempotencyHeader,
        response: Response,
    ) -> ReportConflictResult:
        command = _command(ReportConflictCommand, body, idempotency_key=idempotency_key)
        return _mark_replay(response, await runtime.conflicts.report(actor, command))

    @app.post("/v1/conflicts/{conflict_id}:resolve", response_model=ResolveConflictResult)
    async def resolve_conflict(
        conflict_id: str,
        body: ResolveConflictBody,
        actor: ActorDependency,
        idempotency_key: IdempotencyHeader,
        response: Response,
    ) -> ResolveConflictResult:
        command = _command(
            ResolveConflictCommand,
            body,
            idempotency_key=idempotency_key,
            conflict_id=conflict_id,
        )
        return _mark_replay(response, await runtime.conflicts.resolve(actor, command))

    @app.get("/v1/runs/{run_id}/events", response_model=EventPage)
    async def run_events(
        run_id: str,
        actor: ActorDependency,
        cursor: Annotated[str | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> EventPage:
        if run_id != actor.run_id:
            raise ResourceNotFound("run", run_id)
        return await runtime.coordination.events(actor, cursor=cursor, limit=limit)

    @app.get("/v1/runs/{run_id}/metrics", response_model=RunMetrics)
    async def run_metrics(run_id: str, actor: ActorDependency) -> RunMetrics:
        if run_id != actor.run_id:
            raise ResourceNotFound("run", run_id)
        return await runtime.coordination.metrics(actor)

    return app


async def _drive_console_demo(
    app: FastAPI,
    runtime: SwarmBrainRuntime,
    scenario: DemoScenario,
) -> None:
    """Run the scripted scenario against this very app, in this very process.

    The beats are driven over the canonical HTTP surface through an in-process
    ASGI transport, exactly as ``swarmbrain-demo`` drives it, so the board a
    judge is watching fills from real authenticated requests. Nothing is
    written to disk: the hosted path produces no evidence artifact.
    """

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://swarm-console-demo",
            timeout=60.0,
        ) as client:
            await run_demo_scenario(
                client=client,
                runtime=runtime,
                scenario=scenario,
                expire_leases=real_time_lease_expiry(lambda _line: None),
                lease_seconds=REAL_TIME_LEASE_SECONDS,
                narrate=None,
                now=None,
            )
    except Exception:
        # A failed beat leaves the run's durable events in place; the console
        # keeps reading them. Never let the background task escape as noise.
        return


def _command(
    model: type[CommandT],
    body: BaseModel | dict[str, Any],
    *,
    idempotency_key: str,
    **injected: Any,
) -> CommandT:
    values = body.model_dump(exclude_unset=True) if isinstance(body, BaseModel) else dict(body)
    supplied_key = values.pop("idempotency_key", None)
    if supplied_key is not None and supplied_key != idempotency_key:
        raise InvalidState("body idempotency_key does not match Idempotency-Key header")
    for name, value in injected.items():
        supplied = values.pop(name, None)
        if supplied is not None and supplied != value:
            raise InvalidState(f"body {name} does not match path")
        values[name] = value
    values["idempotency_key"] = idempotency_key
    return _validate(model, values)


def _validate(model: type[CommandT], values: dict[str, Any]) -> CommandT:
    try:
        return model.model_validate(values)
    except ValidationError as exc:
        details = {"fields": [_bounded_validation_error(item) for item in exc.errors()]}
        error = SwarmBrainError(
            ErrorCode.INVALID_STATE,
            "request validation failed",
            status_code=422,
            details=details,
        )
        error.code = "validation_error"  # transport-stable code outside the domain enum
        raise error from exc


def _bounded_validation_error(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "location": [str(part) for part in item.get("loc", ())][:8],
        "type": str(item.get("type", "validation_error"))[:120],
        "message": str(item.get("msg", "invalid value"))[:300],
    }


def _mark_replay(response: Response, result: CommandT) -> CommandT:
    if bool(getattr(result, "replayed", False)):
        response.headers["Idempotency-Replayed"] = "true"
    return result


def _error_response(
    request: Request,
    status_code: int,
    code: ErrorCode | str,
    message: str,
    retryable: bool,
    details: dict[str, Any],
) -> JSONResponse:
    supplied_request_id = request.headers.get("X-Request-ID")
    try:
        request_id = str(UUID(supplied_request_id)) if supplied_request_id else str(uuid4())
    except ValueError:
        request_id = str(uuid4())
    headers = {"X-Request-ID": request_id}
    if status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": str(code),
                "message": message,
                "request_id": request_id,
                "retryable": retryable,
                "details": details,
            }
        },
        headers=headers,
    )


__all__ = ["create_app"]
