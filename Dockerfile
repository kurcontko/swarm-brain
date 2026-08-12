# syntax=docker/dockerfile:1
#
# Swarm Brain — one image, two processes, no state.
#
# The container holds no durable state. CockroachDB is the sole store: there is
# no volume, no local database file, no cached model, and nothing on disk that a
# replacement task needs in order to resume a run. That is the property the
# crash-handoff demo turns on, so the image must not quietly acquire a
# dependency on its own disk.
#
# ONE IMAGE, TWO SERVICES. The default CMD serves the HTTP API. The durable
# worker is the *same* image with the command overridden:
#
#     docker run ... swarm-brain:TAG                    -> swarmbrain-api
#     docker run ... swarm-brain:TAG swarmbrain-worker  -> swarmbrain-worker
#
# ECS does this with `command` in the container definition; see
# deploy/ecs-task-definition.worker.json. Two images would mean two things to
# build, push, scan and keep in step, for two entry points into the same
# package — the split buys nothing and costs drift.
#
# THE CONSOLE IS NOT A BUILD STEP. `/console` is a single self-contained HTML
# document that ships inside the Python package
# (src/swarmbrain/transports/http/console/index.html) and is served by the API
# process. There is no Node, no bundler, and no separate asset stage: nothing to
# install, nothing to pin, nothing to go stale.
#
# Nothing here bakes in a secret, an account identifier, a region, or a DSN.
# Every credential arrives at run time — from the ECS task role for AWS, and
# from Secrets Manager for the database URL and the token secret. See
# deploy/README.md.
#
# BuildKit is not required; nothing below uses a BuildKit-only feature.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.11.6

# ---------------------------------------------------------------------------
# Stage: uv — the resolver, pinned, as a named stage
# ---------------------------------------------------------------------------
# A named stage rather than `COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION}`:
# only `FROM` expands build args, so the inline form silently requires a
# hard-coded version.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------------------
# Stage: build — resolve and install the Python environment
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS build
COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app

# Manifests first so the dependency layer survives a source-only edit.
# README.md and LICENSE are build inputs, not documentation shipped for their
# own sake: pyproject.toml names them as `readme` and `license`, and hatchling
# reads both to build metadata.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/

# `--frozen`        the lock is the authority; a drifted pyproject fails the
#                   build rather than silently resolving something else.
# `--no-dev`        pytest and ruff have no business in a deployed image. This
#                   is what keeps the image lean: the `dev` extra pulls the test
#                   toolchain and is never installed here.
# `--extra serve`   fastapi + uvicorn — the API process.
# `--extra crdb`    psycopg[binary] + psycopg-pool — the CockroachDB backend.
#                   The binary wheel bundles libpq, so no apt package is needed.
# `--extra aws`     boto3 — the Bedrock embedding provider.
# `--no-editable`   copies the package into site-packages, so the runtime stage
#                   needs neither /app/src nor a .pth pointing at it.
#
# The `mcp` extra is deliberately absent. The MCP bridge is a stdio transport
# an operator runs beside their agent; it is not a server process, and shipping
# it here would install a dependency nothing in the container can reach.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1
RUN uv sync --frozen --no-dev --no-editable \
      --extra serve --extra crdb --extra aws

# ---------------------------------------------------------------------------
# Stage: runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Same base and same interpreter version as the build stage, and the venv keeps
# the same absolute path, so its interpreter symlink and its console scripts
# (swarmbrain-api, swarmbrain-worker, swarmbrain-schema, swarmbrain-token) stay
# valid.

LABEL org.opencontainers.image.title="Swarm Brain" \
      org.opencontainers.image.description="Vendor-neutral temporal memory and coordination kernel for agent swarms, on CockroachDB" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/OWNER/swarm-brain"

# Unprivileged, no shell, no home to write to. Nothing in the image is chowned
# to it: both processes read the app and write nothing to disk.
RUN groupadd --system --gid 10001 swarmbrain \
    && useradd --system --uid 10001 --gid 10001 \
       --home-dir /nonexistent --shell /usr/sbin/nologin swarmbrain

WORKDIR /app
COPY --from=build /app/.venv /app/.venv

# MIT requires the licence to travel with a distributed copy.
COPY LICENSE /app/licenses/swarm-brain-MIT.txt

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Bind to every interface: the process is alone in its network namespace and
# ECS awsvpc gives the task its own ENI, so 127.0.0.1 would be unreachable from
# the load balancer. The port is fixed here and matched by the task definition's
# portMappings and the ALB target group.
#
# SWARMBRAIN_BACKEND, SWARMBRAIN_DATABASE_URL and SWARMBRAIN_TOKEN_SECRET are
# deliberately NOT set. Configuration is fail-closed: a container started
# without them exits with a named error rather than quietly serving an
# in-memory backend that looks fine and stores nothing.
ENV SWARMBRAIN_HOST=0.0.0.0 \
    SWARMBRAIN_PORT=8080

USER 10001:10001
EXPOSE 8080

# No curl or wget in a slim image, and adding one to health-check ourselves is
# a poor trade. The interpreter is already here.
#
# This probe is for the API. ECS ignores an image's HEALTHCHECK entirely — it
# uses only the container definition's `healthCheck` — so the worker task
# definition simply omits one and this instruction never applies to it. Under a
# plain `docker run` of the worker the container will report `unhealthy`; that
# is cosmetic and expected, since a queue worker serves no HTTP.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4).status == 200 else 1)"]

# One uvicorn worker per task. Concurrency comes from more tasks, not more
# processes: the API is stateless and CockroachDB is what serialises it.
CMD ["swarmbrain-api"]
