#!/usr/bin/env bash
# Provision the CockroachDB Cloud Basic cluster that backs the hosted demo,
# using the `ccloud` CLI.
#
#   scripts/provision_cloud.sh --dry-run   # print the plan, call nothing
#   scripts/provision_cloud.sh             # create, confirming each step
#   scripts/provision_cloud.sh --yes       # create without prompting
#
# KEY-GATED. With no `ccloud` on PATH, or no authenticated session, this exits
# **0** after saying exactly what is missing and how to fix it. It is safe to
# run at any time — including right now, before any account exists — and safe
# to wire into a checklist that must not fail a pipeline.
#
# It creates and it never destroys. There is no delete path in this file; the
# teardown commands live in deploy/README.md where a human reads them.
#
# COMMAND VERIFICATION. `ccloud` is not installed on the machine this script
# was written on, so nothing here was executed against a real CLI. Every
# command is written against the official CockroachDB Cloud documentation and
# carries a confidence marker in the output:
#
#   [doc]    documented in the official ccloud command reference
#   [?]      not in the official docs, or documented ambiguously — the script
#            probes for it and degrades rather than assuming
#
# Read the printed command before approving it. If one is wrong, the fix is a
# one-line edit here, not a debugging session against a half-made cluster.
#
# Sources:
#   https://www.cockroachlabs.com/docs/cockroachcloud/ccloud-reference
#   https://www.cockroachlabs.com/docs/cockroachcloud/create-a-basic-cluster
#   https://www.cockroachlabs.com/docs/cockroachcloud/connect-to-a-basic-cluster
#
# Environment:
#   SB_CLUSTER_NAME   cluster name        (default: swarm-brain-demo)
#   SB_CLOUD          cloud provider      (default: AWS)
#   SB_REGION         region              (default: us-east-1)
#   SB_SQL_USER       SQL user to create  (default: swarmbrain_app)
#   SB_DATABASE       logical database    (default: defaultdb)

set -euo pipefail

CLUSTER_NAME="${SB_CLUSTER_NAME:-swarm-brain-demo}"
CLOUD="${SB_CLOUD:-AWS}"
REGION="${SB_REGION:-us-east-1}"
SQL_USER="${SB_SQL_USER:-swarmbrain_app}"
DATABASE="${SB_DATABASE:-defaultdb}"

DRY_RUN=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
  --dry-run) DRY_RUN=1 ;;
  --yes | -y) ASSUME_YES=1 ;;
  -h | --help)
    sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    echo "provision_cloud: unknown argument: $arg" >&2
    echo "provision_cloud: usage: $0 [--dry-run] [--yes]" >&2
    exit 2
    ;;
  esac
done

say() { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
note() { printf '  --  %s\n' "$*"; }
warn() { printf '  !!  %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# the key gate
# ---------------------------------------------------------------------------

say "provision_cloud: CockroachDB Cloud Basic — $CLUSTER_NAME ($CLOUD/$REGION)"

if ! command -v ccloud >/dev/null 2>&1; then
  step "ccloud is not installed"
  say "  The CockroachDB Cloud CLI is not on PATH, so nothing can be provisioned."
  say ""
  say "  Install it:"
  say "    brew install cockroachdb/tap/ccloud"
  say "    # or see https://www.cockroachlabs.com/docs/cockroachcloud/ccloud-get-started"
  say ""
  say "  Then authenticate and re-run:"
  say "    ccloud auth login"
  say "    scripts/provision_cloud.sh --dry-run"
  say ""
  say "provision_cloud: nothing to do (exit 0)"
  exit 0
fi

# `ccloud auth whoami` is the documented "which org am I in" command. [doc]
# That it exits nonzero when unauthenticated is NOT documented, so an empty
# stdout is treated as unauthenticated too rather than trusting the status. [?]
authenticated=1
whoami_output=""
if ! whoami_output="$(ccloud auth whoami 2>/dev/null)"; then
  authenticated=0
elif [ -z "${whoami_output//[[:space:]]/}" ]; then
  authenticated=0
fi

if [ "$authenticated" -eq 0 ]; then
  step "ccloud is not authenticated"
  say "  ccloud not authenticated — run ccloud auth login"
  say ""
  say "  There is no API-key or environment-variable login for this CLI: the"
  say "  documented flows are all human-in-the-loop. On a headless machine use"
  say "  the code-paste flow:"
  say ""
  say "    ccloud auth login                 # opens a browser"
  say "    ccloud auth login --no-redirect    # prints a URL to open elsewhere"
  say ""
  say "  Then re-run:  scripts/provision_cloud.sh --dry-run"
  say ""
  say "provision_cloud: nothing to do (exit 0)"
  exit 0
fi

note "authenticated as: $whoami_output"

# ---------------------------------------------------------------------------
# JSON support is uncertain — probe once, never assume
# ---------------------------------------------------------------------------

# A Cockroach Labs blog post states `-o json` is a global flag on every ccloud
# command. The official command reference does not mention it anywhere. Rather
# than pick a side, probe it and remember the answer. [?]
JSON_FLAG=""
if ccloud cluster list -o json >/dev/null 2>&1; then
  JSON_FLAG="-o json"
  note "JSON output: supported (-o json)"
else
  note "JSON output: NOT supported by this ccloud build; using text output"
fi

# ---------------------------------------------------------------------------
# run — print, confirm, execute
# ---------------------------------------------------------------------------

confirm() {
  local prompt="$1"
  [ "$ASSUME_YES" -eq 1 ] && return 0
  if [ ! -t 0 ]; then
    warn "not a terminal and --yes not given; refusing to run unattended"
    return 1
  fi
  local reply=""
  printf '  ?? %s [y/N] ' "$prompt"
  read -r reply
  case "$reply" in
  y | Y | yes | YES) return 0 ;;
  *) return 1 ;;
  esac
}

# run CONFIDENCE DESCRIPTION -- COMMAND...
run() {
  local confidence="$1" description="$2"
  shift 2
  [ "${1:-}" = "--" ] && shift

  printf '  %s %s\n' "$confidence" "$description"
  printf '      $ %s\n' "$*"

  if [ "$DRY_RUN" -eq 1 ]; then
    note "dry run: not executed"
    return 0
  fi
  if ! confirm "run it?"; then
    note "skipped"
    return 0
  fi
  "$@"
}

if [ "$DRY_RUN" -eq 1 ]; then
  step "DRY RUN — nothing below will be executed"
fi

# ---------------------------------------------------------------------------
step "1. Does the cluster already exist?"
# ---------------------------------------------------------------------------

# Creating a cluster that already exists is the one mistake here that costs
# money twice, so this is a read before every write.
existing=""
# shellcheck disable=SC2086
if existing="$(ccloud cluster list $JSON_FLAG 2>/dev/null)"; then
  if printf '%s' "$existing" | grep -q -- "$CLUSTER_NAME"; then
    warn "a cluster named '$CLUSTER_NAME' already appears in 'ccloud cluster list'"
    warn "skipping creation. Set SB_CLUSTER_NAME to use a different name."
    CLUSTER_EXISTS=1
  else
    note "no cluster named '$CLUSTER_NAME' found"
    CLUSTER_EXISTS=0
  fi
else
  warn "could not list clusters; treating the cluster as absent"
  CLUSTER_EXISTS=0
fi

# ---------------------------------------------------------------------------
step "2. Create the Basic cluster"
# ---------------------------------------------------------------------------

# The plan is a SUBCOMMAND (`basic` | `standard` | `dedicated`), not a --plan
# flag, and the region is POSITIONAL, not --region. Both are easy to get wrong
# from memory. [doc]
#
# --spend-limit 0 is the free tier. The *units* of --spend-limit are not
# documented anywhere; 0 is the only value the docs ever show, and 0 is the
# only value this script will ever pass. [?]
if [ "$CLUSTER_EXISTS" -eq 1 ]; then
  note "skipped — cluster already exists"
else
  run "[doc]" "create the Basic cluster on the free tier" -- \
    ccloud cluster create basic "$CLUSTER_NAME" "$REGION" \
    --cloud "$CLOUD" --spend-limit 0
fi

# ---------------------------------------------------------------------------
step "3. Confirm it is ready"
# ---------------------------------------------------------------------------

# `info`, not `describe` — there is no describe subcommand. [doc]
# Basic clusters provision in under a minute.
run "[doc]" "show cluster state and plan" -- \
  ccloud cluster info "$CLUSTER_NAME"

# ---------------------------------------------------------------------------
step "4. Create the SQL user"
# ---------------------------------------------------------------------------

# The password is PROMPTED interactively and is never echoed back. There is no
# documented --password flag and no stdin mechanism, so this step cannot be
# automated — which is fine, because a database password that a script could
# print is a password in a log. [doc]
#
# New users are granted the admin SQL role by default. That is more than this
# application needs; step 6 shows how to narrow it.
run "[doc]" "create the SQL user (password is prompted, never printed)" -- \
  ccloud cluster user create "$CLUSTER_NAME" "$SQL_USER"

# ---------------------------------------------------------------------------
step "5. The connection URL"
# ---------------------------------------------------------------------------

# `ccloud cluster sql --connection-url` is the documented way to print it.
# There is no `ccloud cluster connection-string`. [doc]
run "[doc]" "print the cluster's connection URL" -- \
  ccloud cluster sql --connection-url "$CLUSTER_NAME"

cat <<EOF

  The printed URL carries NO username and NO password. Swarm Brain needs a
  complete URL, so build it from the printed host:

    postgresql://$SQL_USER:<PASSWORD>@<HOST>:26257/$DATABASE?sslmode=verify-full

  where <HOST> looks like:

    $CLUSTER_NAME-<4CHAR>.$(printf '%s' "$CLOUD" | tr '[:upper:]' '[:lower:]')-$REGION.cockroachlabs.cloud

  The 4-character routing suffix is assigned at creation and is only knowable
  from the output above.

  Three things that are easy to get wrong:

    * sslmode MUST be verify-full. Anything weaker authenticates the client to
      the server but not the server to the client.
    * The routing ID lives in the HOSTNAME. There is no options=--cluster=...
      parameter; that form is legacy and does not apply to a Basic cluster.
    * A Basic cluster's certificate is signed by Let's Encrypt — publicly
      trusted — so NO CA file is needed. verify-full works against the system
      trust store, in the container and on your laptop alike.

      Only if a client ignores the OS trust store:
        curl --create-dirs -o ~/.postgresql/root.crt \\
          https://cockroachlabs.cloud/clusters/<CLUSTER_ID>/cert

EOF

# ---------------------------------------------------------------------------
step "6. Install the schema"
# ---------------------------------------------------------------------------

cat <<EOF
  Swarm Brain never applies DDL on its own: the runtime VERIFIES the schema at
  start-up and refuses to serve without it. Installing is a separate, explicit
  act — run it now, before any task starts, or the first deploy will fail its
  readiness check and look like a networking problem.

    export SWARMBRAIN_DATABASE_URL='postgresql://$SQL_USER:<PASSWORD>@<HOST>:26257/$DATABASE?sslmode=verify-full'

    swarmbrain-schema install     # applies the DDL, then verifies it
    swarmbrain-schema verify      # idempotent; safe to re-run any time

  'install' prints the statement count and then verifies. If 'verify' passes on
  a fresh cluster you are done; if it fails, read the error before re-running —
  it names the missing object.

  Then narrow the SQL user, which ccloud created with the admin role:

    REVOKE admin FROM $SQL_USER;
    GRANT CONNECT ON DATABASE $DATABASE TO $SQL_USER;
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO $SQL_USER;

  Do this AFTER 'swarmbrain-schema install' — the install needs DDL rights the
  running application must not keep.

EOF

# ---------------------------------------------------------------------------
step "Next"
# ---------------------------------------------------------------------------

cat <<'EOF'
  1. Store the completed URL in AWS Secrets Manager as swarm-brain/database-url
  2. Follow deploy/README.md from step 3 onward
  3. Teardown, when the judging window closes:  ccloud cluster delete <name>
     (the full inventory is in deploy/README.md — nothing in this script
     deletes anything)
EOF

say ""
if [ "$DRY_RUN" -eq 1 ]; then
  say "provision_cloud: DRY RUN complete — nothing was created"
else
  say "provision_cloud: done"
fi
