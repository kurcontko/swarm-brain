#!/usr/bin/env bash
# Prove a running Swarm Brain is alive, ready, serving the console, and
# enforcing token authentication — against ANY base URL.
#
#   scripts/smoke_deploy.sh http://127.0.0.1:8080     # local uvicorn
#   scripts/smoke_deploy.sh http://127.0.0.1:18080    # local container
#   scripts/smoke_deploy.sh https://<public-url>      # the deployed service
#
# One script for all three, on purpose: the rehearsal that passes locally is
# the rehearsal run against the public URL, so a difference between them is a
# deployment difference and not a difference in what was tested.
#
# WHAT IT ASSERTS
#   /healthz    200 {"status":"ok"} — the process is alive.
#   /readyz     200 {"status":"ready"} — the backend answered. On the cockroach
#               backend this is a real round trip to the cluster.
#   /console    200 text/html, a non-trivial self-contained document that
#               contains no bearer token and no database URL.
#   auth        an unauthenticated read of a v1 route is refused with 401, and
#               a malformed bearer token is refused with 401. A deployment that
#               serves run data without a token is a failure, not a warning.
#   metrics     an authenticated read of the run's metrics returns 200 and a
#               RunMetrics body whose run_id is the token's run.
#   scope       the same token reading a *different* run's metrics gets 404.
#               The token cannot see outside the run it was minted for.
#
# WHAT IT WRITES
#   By default it joins one agent to one run in a namespace of freshly
#   generated UUIDs that this script invents and nothing else will ever
#   address. That is one `runs` row and one `agents` row. It is what turns the
#   metrics read into a real 200 with real numbers instead of a 404, and it
#   proves the database is writable — which /readyz does not.
#
#   No task, memory, evidence, conflict or work item is created. Nothing
#   existing is read, modified or deleted. Set SMOKE_JOIN=0 for a strictly
#   read-only run; the metrics read then answers 404 on the cockroach backend
#   (correct for a run that does not exist) and the script reports it as such.
#
# WHAT IT NEEDS
#   SWARMBRAIN_TOKEN_SECRET   the same secret the target is running with.
#                             Tokens are minted locally by swarmbrain-token;
#                             none is ever sent anywhere but the target, and
#                             none is printed.
#   python3                   required — for JSON reads and as the HTTP
#                             fallback transport.
#   curl                      optional — used when present.
#   swarmbrain-token          found on PATH, in ./.venv/bin, via `uv run`, or
#                             named explicitly with SMOKE_TOKEN_CMD.
#
# Environment:
#   SWARMBRAIN_TOKEN_SECRET   required — the target's token secret
#   SMOKE_JOIN                1 (default) join first; 0 read-only
#   SMOKE_TIMEOUT             per-request timeout in seconds (default: 30)
#   SMOKE_TOKEN_CMD           override the swarmbrain-token invocation

set -uo pipefail

BASE_URL="${1:-}"
SMOKE_JOIN="${SMOKE_JOIN:-1}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-30}"

if [ -z "$BASE_URL" ]; then
  echo "usage: $0 <base-url>    e.g. $0 http://127.0.0.1:8080" >&2
  exit 2
fi
BASE_URL="${BASE_URL%/}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "smoke: required tool not found: python3" >&2
  exit 2
fi

# curl is preferred — it is the idiomatic tool here and gives better TLS
# diagnostics against a public HTTPS URL — but it is not required. A minimal
# container or a locked-down CI runner may have python3 and nothing else, and
# this script has to be the same script everywhere or it is not evidence of
# anything. The urllib fallback speaks the same request/response contract.
if command -v curl >/dev/null 2>&1; then
  HTTP_TRANSPORT="curl"
else
  HTTP_TRANSPORT="python3"
fi

if [ -z "${SWARMBRAIN_TOKEN_SECRET:-}" ]; then
  echo "smoke: SWARMBRAIN_TOKEN_SECRET is not set." >&2
  echo "smoke: it must match the secret the target is running with." >&2
  echo "smoke: load it with:  set -a; source .env; set +a" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

failures=0
body=""
status=""

pass() { printf '  ok    %s\n' "$1"; }
info() { printf '  --    %s\n' "$1"; }
fail() {
  printf '  FAIL  %s\n' "$1" >&2
  failures=$((failures + 1))
}
section() { printf '\n== %s\n' "$1"; }

# ---------------------------------------------------------------------------
# token minting — local, never over the network
# ---------------------------------------------------------------------------

# Resolve swarmbrain-token without assuming how the operator installed it: an
# activated venv, a `uv run` project checkout, or an explicit override all work.
resolve_token_cmd() {
  if [ -n "${SMOKE_TOKEN_CMD:-}" ]; then
    echo "$SMOKE_TOKEN_CMD"
    return 0
  fi
  if command -v swarmbrain-token >/dev/null 2>&1; then
    echo "swarmbrain-token"
    return 0
  fi
  if [ -x "$repo_root/.venv/bin/swarmbrain-token" ]; then
    echo "$repo_root/.venv/bin/swarmbrain-token"
    return 0
  fi
  if command -v uv >/dev/null 2>&1 && [ -f "$repo_root/pyproject.toml" ]; then
    echo "uv run --quiet --project $repo_root swarmbrain-token"
    return 0
  fi
  return 1
}

TOKEN_CMD="$(resolve_token_cmd)" || {
  echo "smoke: swarmbrain-token not found." >&2
  echo "smoke: install the project, activate its venv, or set SMOKE_TOKEN_CMD." >&2
  exit 2
}

uuid() { python3 -c 'import uuid; print(uuid.uuid4())'; }

TENANT_ID="$(uuid)"
PROJECT_ID="$(uuid)"
REPOSITORY_ID="$(uuid)"
SWARM_ID="$(uuid)"
RUN_ID="$(uuid)"
AGENT_ID="$(uuid)"
OTHER_RUN_ID="$(uuid)"

# A viewer token: it may join the run and read its metrics and events, and it
# holds no capability that could publish memory, claim a task, resolve a
# conflict, or mutate anything a judge is looking at.
mint_token() {
  # shellcheck disable=SC2086
  $TOKEN_CMD \
    --tenant "$TENANT_ID" \
    --project "$PROJECT_ID" \
    --repository "$REPOSITORY_ID" \
    --swarm "$SWARM_ID" \
    --run "$RUN_ID" \
    --agent "$AGENT_ID" \
    --harness "smoke-deploy" \
    --provider "swarmbrain" \
    --model "smoke" \
    --capability run:join \
    --capability metrics:read \
    --capability events:read \
    --ttl-seconds 300 2>/dev/null
}

TOKEN="$(mint_token)"
if [ -z "$TOKEN" ]; then
  echo "smoke: could not mint a token with: $TOKEN_CMD" >&2
  echo "smoke: check SWARMBRAIN_TOKEN_SECRET is at least 16 bytes." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

# Both transports emit the response body, a newline, then the numeric status —
# so everything downstream parses one shape regardless of which one ran.
# A transport-level failure (refused, timed out, TLS error) is status "000",
# which is distinguishable from every real HTTP status.
http_via_python() {
  local method="$1" url="$2" header="$3" payload="$4"
  SMOKE_METHOD="$method" SMOKE_URL="$url" SMOKE_AUTH="$header" \
    SMOKE_PAYLOAD="$payload" SMOKE_TIMEOUT="$SMOKE_TIMEOUT" python3 -c '
import os, sys, urllib.error, urllib.request

payload = os.environ["SMOKE_PAYLOAD"]
data = payload.encode("utf-8") if payload else None
request = urllib.request.Request(
    os.environ["SMOKE_URL"], data=data, method=os.environ["SMOKE_METHOD"]
)
if os.environ["SMOKE_AUTH"]:
    request.add_header("Authorization", os.environ["SMOKE_AUTH"])
if data is not None:
    request.add_header("Content-Type", "application/json")

try:
    with urllib.request.urlopen(request, timeout=float(os.environ["SMOKE_TIMEOUT"])) as response:
        body, code = response.read(), response.status
except urllib.error.HTTPError as exc:
    # A 4xx/5xx is a real answer from the service, not a transport failure.
    body, code = exc.read(), exc.code
except Exception:
    body, code = b"", 0

sys.stdout.write(body.decode("utf-8", "replace"))
sys.stdout.write("\n%03d" % code)
'
}

# request METHOD URL [AUTH] [JSON-BODY] -> sets $status and $body
# AUTH: "none" (default) or "bearer" or a literal token string.
request() {
  local method="$1" url="$2" auth="${3:-none}" payload="${4:-}"
  local raw header=""
  case "$auth" in
  none) ;;
  bearer) header="Bearer $TOKEN" ;;
  *) header="Bearer $auth" ;;
  esac

  if [ "$HTTP_TRANSPORT" = "curl" ]; then
    local -a args=(--silent --show-error --max-time "$SMOKE_TIMEOUT"
      --write-out '\n%{http_code}' --request "$method")
    if [ -n "$header" ]; then
      args+=(--header "Authorization: $header")
    fi
    if [ -n "$payload" ]; then
      args+=(--header 'content-type: application/json' --data "$payload")
    fi
    raw="$(curl "${args[@]}" "$url" 2>/dev/null)" || {
      status="000"
      body=""
      return 1
    }
  else
    raw="$(http_via_python "$method" "$url" "$header" "$payload")" || {
      status="000"
      body=""
      return 1
    }
  fi

  status="${raw##*$'\n'}"
  body="${raw%$'\n'*}"
  if [ "$status" = "000" ]; then
    return 1
  fi
  return 0
}

# jsonq DOTTED.PATH -- read $body as JSON and print the value at that path.
# Prints nothing and returns 1 when the path is absent, so a caller can tell
# "missing" apart from "empty".
jsonq() {
  BODY="$body" QUERY="$1" python3 -c '
import json, os, sys

try:
    doc = json.loads(os.environ["BODY"])
except Exception:
    sys.exit(1)

node = doc
for part in os.environ["QUERY"].split("."):
    if part == "":
        continue
    if isinstance(node, list):
        try:
            node = node[int(part)]
        except (ValueError, IndexError):
            sys.exit(1)
    elif isinstance(node, dict) and part in node:
        node = node[part]
    else:
        sys.exit(1)

if isinstance(node, (dict, list)):
    print(json.dumps(node, separators=(",", ":"), sort_keys=True))
elif isinstance(node, bool):
    print("true" if node else "false")
elif node is None:
    print("null")
else:
    print(node)
'
}

expect_status() {
  local want="$1" what="$2"
  if [ "$status" = "$want" ]; then
    pass "$what -> $status"
    return 0
  fi
  fail "$what -> $status (expected $want)"
  return 1
}

expect_field() {
  local path="$1" want="$2" what="$3" got
  got="$(jsonq "$path")" || {
    fail "$what: $path is missing"
    return 1
  }
  if [ "$got" = "$want" ]; then
    pass "$what: $path = $got"
    return 0
  fi
  fail "$what: $path = $got (expected $want)"
  return 1
}

# ---------------------------------------------------------------------------

echo "smoke: $BASE_URL"
info "run:   $RUN_ID (freshly generated for this smoke)"

# ---------------------------------------------------------------------------
section "liveness and readiness"
# ---------------------------------------------------------------------------

request GET "$BASE_URL/healthz"
expect_status 200 "GET /healthz"
expect_field "status" "ok" "healthz"

request GET "$BASE_URL/readyz"
if [ "$status" = "200" ]; then
  pass "GET /readyz -> 200"
  expect_field "status" "ready" "readyz"
else
  # 503 is the honest answer from a process whose store is unreachable. It is
  # still a failure of the deployment, so it is reported as one.
  fail "GET /readyz -> $status (expected 200; 503 means the backend did not answer)"
fi

# /readyz reports readiness, not identity: it does not name which store
# answered. Which backend this deployment runs is fixed by SWARMBRAIN_BACKEND
# in the task definition and visible in the task's logs — confirm it there, not
# from this output.

# ---------------------------------------------------------------------------
section "console"
# ---------------------------------------------------------------------------

request GET "$BASE_URL/console"
expect_status 200 "GET /console"
console_bytes="${#body}"
if [ "$console_bytes" -gt 2000 ]; then
  pass "console document: $console_bytes bytes"
else
  fail "console document is only $console_bytes bytes; expected a full page"
fi
case "$body" in
*"<html"* | *"<!doctype"* | *"<!DOCTYPE"*) pass "console is an HTML document" ;;
*) fail "console response does not look like HTML" ;;
esac
# The page is static and fetches run data in the browser with a token the
# operator pastes in. If a credential ever appears in the served bytes, that is
# the single worst thing this deployment could do, so it is checked explicitly.
case "$body" in
*"$SWARMBRAIN_TOKEN_SECRET"*) fail "the console body contains the token secret" ;;
*) pass "console body carries no token secret" ;;
esac
case "$body" in
*"postgresql://"* | *"postgres://"*) fail "the console body contains a database URL" ;;
*) pass "console body carries no database URL" ;;
esac

# ---------------------------------------------------------------------------
section "authentication is enforced"
# ---------------------------------------------------------------------------

METRICS_URL="$BASE_URL/v1/runs/$RUN_ID/metrics"

request GET "$METRICS_URL"
expect_status 401 "GET metrics without a token"

request GET "$METRICS_URL" "not-a-real-token"
expect_status 401 "GET metrics with a malformed token"

# ---------------------------------------------------------------------------
section "authenticated read"
# ---------------------------------------------------------------------------

if [ "$SMOKE_JOIN" = "1" ]; then
  request POST "$BASE_URL/v1/runs/$RUN_ID/agents:join" bearer '{}'
  expect_status 200 "POST agents:join"
  expect_field "run_id" "$RUN_ID" "join"
  agent_status="$(jsonq "status" || echo "?")"
  info "agent joined: $AGENT_ID (status=$agent_status)"
else
  info "join skipped (SMOKE_JOIN=0); this run has no rows"
fi

request GET "$METRICS_URL" bearer
case "$status" in
200)
  pass "GET metrics -> 200"
  expect_field "run_id" "$RUN_ID" "metrics"
  for metric in tasks_total tasks_claimed active_leases memories_published \
    duplicate_claims_prevented; do
    value="$(jsonq "$metric")" || {
      fail "metrics: $metric is missing"
      continue
    }
    info "metrics.$metric = $value"
  done
  ;;
404)
  if [ "$SMOKE_JOIN" = "1" ]; then
    fail "GET metrics -> 404 after a successful join; the run did not persist"
  else
    code="$(jsonq "error.code" || echo "?")"
    info "GET metrics -> 404 $code (expected with SMOKE_JOIN=0: the run has no rows)"
    pass "the token authenticated and was authorized; the run simply does not exist"
  fi
  ;;
401 | 403)
  fail "GET metrics -> $status; the minted token was rejected. Does SWARMBRAIN_TOKEN_SECRET match the target's?"
  ;;
*)
  fail "GET metrics -> $status (expected 200)"
  ;;
esac

# ---------------------------------------------------------------------------
section "run scope is enforced"
# ---------------------------------------------------------------------------

# The same valid token, pointed at a run it was not minted for. A deployment
# that answers this with anything but 404 is leaking across runs.
request GET "$BASE_URL/v1/runs/$OTHER_RUN_ID/metrics" bearer
expect_status 404 "GET another run's metrics with this token"

# ---------------------------------------------------------------------------
section "summary"
# ---------------------------------------------------------------------------

info "base url: $BASE_URL"
info "console:  $console_bytes bytes"
info "join:     $([ "$SMOKE_JOIN" = 1 ] && echo "performed" || echo "skipped")"

echo
if [ "$failures" -eq 0 ]; then
  echo "smoke: PASSED"
  exit 0
fi
echo "smoke: FAILED ($failures check(s))" >&2
exit 1
