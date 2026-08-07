# Deployment

> **Nothing has been deployed, and nothing in this repository deploys itself.**
> Every cloud resource described below is a template awaiting operator
> approval. No credential has been available to the work that produced these
> files, and no AWS or CockroachDB Cloud account has been touched.

Swarm Brain deploys as **one image running two processes**: the HTTP API
(default `CMD`) and the durable worker (`command: ["swarmbrain-worker"]`).
CockroachDB is the only durable store, so either task can be destroyed and
replaced at any moment without losing a run.

## Where things live

| | |
|---|---|
| [`../deploy/README.md`](../deploy/README.md) | **The runbook.** ECR, Secrets Manager, IAM, ECS services, the ALB, health checks, the cost envelope, tags, and the full teardown inventory. Read this before running anything. |
| [`../deploy/iam/README.md`](../deploy/iam/README.md) | Why each IAM grant exists, per statement, and what is deliberately absent. |
| [`../deploy/secrets.example.env`](../deploy/secrets.example.env) | Every secret and parameter the deployment needs — names only, no values, ever. |
| [`../Dockerfile`](../Dockerfile) | The image. Multi-stage, non-root, no volume, no secret, no Node. |
| [`../scripts/build_image.sh`](../scripts/build_image.sh) | Build it locally. Never pushes, never logs in to a registry. |
| [`../scripts/smoke_deploy.sh`](../scripts/smoke_deploy.sh) | Verify a running instance at **any** base URL — local, container, or public. |
| [`../scripts/provision_cloud.sh`](../scripts/provision_cloud.sh) | Create the CockroachDB Cloud Basic cluster via `ccloud`. Key-gated; exits 0 doing nothing when not authenticated. |
| [`../scripts/bedrock_smoke.py`](../scripts/bedrock_smoke.py) | Prove the Bedrock embedding path end to end. Key-gated; exits non-zero with one clean line when credentials are missing. |
| [`../.env.example`](../.env.example) | Local configuration, including the "activation keys" block the operator fills in. |

## Activation order, once the keys land

Run these in order. Each step is cheap to verify and expensive to skip, and the
sequence is chosen so that **every failure happens locally, before anything
costs money**.

```bash
# 0. Load the environment. Nothing auto-loads .env.
set -a; source .env; set +a
```

```bash
# 1. Prove Bedrock works — BEFORE building or deploying anything.
python3 scripts/bedrock_smoke.py
```

Embeds two fixed sentences through the real adapter, asserts 1024 dimensions,
unit norm, and that unrelated sentences are actually distinguishable. With
`SWARMBRAIN_DATABASE_URL` also set it publishes one memory, drains the durable
embedding queue with the real worker, and proves the vector round trip with a
dense recall. Writes redacted evidence to `evidence/`.

Use `--stub` to exercise the wiring with no credentials and no network — it
proves the plumbing, and its evidence artifact says `mode: stubbed-provider`
so it can never be mistaken for the real thing.

```bash
# 2. Create the CockroachDB Cloud Basic cluster.
scripts/provision_cloud.sh --dry-run   # read the plan first
ccloud auth login
scripts/provision_cloud.sh
```

```bash
# 3. Install the schema. The application verifies it and never applies DDL,
#    so this must happen before the first task starts.
export SWARMBRAIN_DATABASE_URL='postgresql://<USER>:<PASSWORD>@<HOST>:26257/<DB>?sslmode=verify-full'
swarmbrain-schema install
swarmbrain-schema verify
```

```bash
# 4. Build the image and rehearse locally against the same smoke script that
#    will later run against the public URL.
scripts/build_image.sh "swarm-brain:$(git rev-parse --short HEAD)"
docker run -d --name sb-api -p 8080:8080 --env-file .env "swarm-brain:$(git rev-parse --short HEAD)"
scripts/smoke_deploy.sh http://127.0.0.1:8080
docker rm -f sb-api
```

```bash
# 5. Push to ECR, then apply the ECS templates.
#    Every command is in deploy/README.md, and every one needs approval.
```

```bash
# 6. Verify the deployment with the same script from step 4.
scripts/smoke_deploy.sh https://<PUBLIC_URL>
```

A difference between the step 4 run and the step 6 run is a *deployment*
difference, not a difference in what was tested. That is the entire reason
there is one script rather than two.

## What is verified, and what is not

| | Status |
|---|---|
| The image builds, runs, and serves against a real CockroachDB | **verified by running it** |
| `smoke_deploy.sh` passes against a running container | **verified by running it** |
| `bedrock_smoke.py` fails cleanly with no credentials | **verified by running it** |
| The Bedrock adapter's wiring, via a stubbed client | **verified by running it** |
| The live Bedrock call | **not verified** — no credentials exist yet. This is exactly what step 1 is for. |
| The durable vector round trip against a real database | **not verified** — it writes, so it waits for a database the operator nominates |
| Every `aws` and `ccloud` command | **not verified** — no account, no CLI. Confidence markers are printed per command. |
| The cost estimate | **published rates, not re-verified.** The briefed shape overruns the ~$40 budget; see the levers in [`../deploy/README.md`](../deploy/README.md#cost-envelope). |
