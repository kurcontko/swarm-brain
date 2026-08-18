# Deploying Swarm Brain

> ## This has been applied — see the record.
>
> **On 2026-08-16 this procedure was executed** against a private AWS account
> in `us-east-1`. The resources described here now exist and are
> billing. An as-built record listing every one of them by identifier, the order
> they were created in, the five divergences from this plan, and the
> verification that followed is kept privately, outside this repository.
>
> The files in this directory are still **templates** with `<PLACEHOLDER>`
> tokens — the substituted copies were rendered outside the repository so no
> account identifier or secret ARN is committed here. No script in this
> repository creates, modifies, or pays for anything on its own. Creating a
> cluster, storing a secret, pushing an image, enabling a Bedrock model, and
> incurring provider cost **each require explicit operator approval**. This
> document is the plan you approve — not a script that runs.
>
> The one script that touches a provider,
> [`../scripts/provision_cloud.sh`](../scripts/provision_cloud.sh), is
> key-gated: with no authenticated `ccloud` session it prints what it would do
> and exits 0 without calling anything.

One image, two processes, no volume. CockroachDB is the only durable store, so
either task can be destroyed and replaced at any moment and the run survives —
which is the crash-handoff beat, and the reason the deployment is shaped this
way rather than it being an implementation detail.

- [The shape](#the-shape)
- [Open decisions](#open-decisions)
- [What gets created](#what-gets-created)
- [Procedure](#procedure)
- [Health and readiness](#health-and-readiness)
- [The hosted-demo switch](#the-hosted-demo-switch)
- [Cost envelope](#cost-envelope)
- [Tags](#tags)
- [Teardown inventory](#teardown-inventory)
- [Known gaps](#known-gaps)

---

## The shape

| Piece | What it is |
|---|---|
| **`swarm-brain-api`** | ECS service, 1 task, behind an ALB. Serves `/healthz`, `/readyz`, `/console` and the authenticated `/v1` routes. This is the demo URL. |
| **`swarm-brain-worker`** | ECS service, 1 task, **no load balancer**. Drains the durable work queue: extraction and Bedrock embedding. Reachable by nothing. |
| **One image** | [`../Dockerfile`](../Dockerfile). The API is the default `CMD`; the worker is the same image with `command: ["swarmbrain-worker"]`. |
| **CockroachDB Cloud Basic** | `us-east-1`. The only durable state in the system. |
| **Bedrock** | `amazon.titan-embed-text-v2:0`, 1024-dim, reached with the task role. No keys. |

The console is **not** a build artifact. `/console` is a single self-contained
HTML document that ships inside the Python package and is served by the API
process. There is no Node, no bundler, no CDN, and no second origin.

---

## Open decisions

Three things are deliberately not decided here.

### 1. Region — recommended `us-east-1`, not yet fixed

`us-east-1` is the recommendation and the assumption throughout, for three
reasons that happen to agree: it is the cheapest region for Fargate and
CloudWatch ingestion, Titan Text Embeddings V2 is available there as a plain
in-region foundation model (no inference profile, which would change the IAM
ARNs — see [`iam/README.md`](iam/README.md)), and the CockroachDB Cloud Basic
cluster is planned for the same region.

**The binding constraint is the cluster.** Put the tasks in the cluster's
region: every request is a round trip to it, and cross-region adds latency to a
live demo and data-transfer cost to the bill. If the cluster ends up somewhere
else, move the tasks, not the cluster.

### 2. Whether to run an ALB at all — **this is the budget decision**

The brief is an ALB in front of the API. The ALB and its public IPv4 addresses
are **more than half the total bill** and are what pushes this deployment past
the ~$40 budget. See [Cost envelope](#cost-envelope) for the numbers and the
levers. Decide it there, deliberately, before creating anything.

### 3. Task sizes — recommended, not fixed

The templates ship **API 0.5 vCPU / 1 GB** and **worker 0.25 vCPU / 0.5 GB**,
both **ARM64**.

- The API gets the larger size because botocore alone loads tens of megabytes
  of service data, and a 512 MB ceiling leaves no margin for a
  garbage-collection spike during a live demo. Saving ~$8 over the whole window
  to risk an OOM on camera is a bad trade.
- The worker gets the Fargate minimum. It does one leased item at a time
  (`SWARMBRAIN_WORKER_LIMIT` is pinned to 1 until extraction leases heartbeat),
  and its work is a Bedrock round trip plus a short fenced transaction — it is
  waiting, not computing.
- ARM64 is **~20% cheaper than x86_64** on both vCPU and memory, the image is
  pure Python plus manylinux wheels, and on Apple silicon it is also the native
  local build with no emulation.

**If you build on x86**, either build with `PLATFORM=linux/arm64` or change
`cpuArchitecture` to `X86_64` in both task definitions. An image architecture
that disagrees with the task platform fails at task start with
`exec format error`, *after* the image pull, so it reads as a runtime problem
and is not one.

---

## What gets created

You create all of it. Nothing here uses a service that provisions on your
behalf, which means nothing appears on the bill that is not on this list — and
the [teardown inventory](#teardown-inventory) is therefore complete rather than
approximate.

| Resource | Count | Note |
|---|---|---|
| CockroachDB Cloud Basic cluster | 1 | outside AWS, billed separately |
| CockroachDB SQL user | 1 | scoped to the demo database, not an admin |
| ECR repository + image | 1 | one image, both services |
| Secrets Manager secrets | 2 | database URL, token secret |
| IAM execution role + inline policy | 1 | shared by both services |
| IAM task role + inline policy | 1 | shared by both services |
| CloudWatch log groups | 2 | `/ecs/swarm-brain/api`, `/ecs/swarm-brain/worker` |
| ECS cluster | 1 | `swarm-brain` |
| ECS task definitions | 2 | api, worker |
| ECS services | 2 | api (with ALB), worker (no LB) |
| Application Load Balancer + listener + target group | 1 | **the budget decision** |
| Security groups | 2 | one for the ALB, one for the tasks |
| S3 evidence bucket | 0 | **not yet** — see [Known gaps](#known-gaps) |

**Do not create a NAT Gateway.** At $0.045/hour it is ~$36 over the deployment
window on its own — more than everything else combined, and it would blow the
budget by itself. Put the tasks in **public subnets with
`assignPublicIp=ENABLED`** so they can pull the image and reach Bedrock and
CockroachDB Cloud directly. The task security group still allows no inbound
traffic except from the ALB security group, so a public IP is not a public
service.

---

## Procedure

Every step is **approval-gated**. Read the whole thing before running any of
it, and keep the [teardown inventory](#teardown-inventory) open alongside.

Set these once. `ACCOUNT_ID` is used only to build ARNs locally; it is never
baked into an image or committed.

```bash
export AWS_REGION=us-east-1
export ACCOUNT_ID=<ACCOUNT_ID>
export REPO=swarm-brain
export IMAGE_TAG=$(git rev-parse --short HEAD)
```

### 0. Rehearse locally first

Not optional politeness — it is what makes the public rehearsal meaningful,
because it is the same script against the same image.

```bash
scripts/build_image.sh swarm-brain:$IMAGE_TAG

docker run -d --name sb-api -p 8080:8080 \
  -e SWARMBRAIN_BACKEND=cockroach \
  -e "SWARMBRAIN_DATABASE_URL=postgresql://root@host.docker.internal:26257/swarmbrain_demo?sslmode=disable" \
  -e SWARMBRAIN_TOKEN_SECRET=local-demo-secret-0123456789 \
  swarm-brain:$IMAGE_TAG

SWARMBRAIN_TOKEN_SECRET=local-demo-secret-0123456789 \
  scripts/smoke_deploy.sh http://127.0.0.1:8080

docker rm -f sb-api
```

The worker is the same image with the command overridden:

```bash
docker run --rm -e SWARMBRAIN_BACKEND=cockroach \
  -e "SWARMBRAIN_DATABASE_URL=..." -e SWARMBRAIN_TOKEN_SECRET=... \
  swarm-brain:$IMAGE_TAG swarmbrain-worker
```

### 1. Create the CockroachDB Cloud cluster

[`../scripts/provision_cloud.sh`](../scripts/provision_cloud.sh) drives
`ccloud`. It exits 0 with a clear message when `ccloud` is absent or not
authenticated, and it prints every command before running it. Run it with
`--dry-run` first to read the plan.

```bash
scripts/provision_cloud.sh --dry-run
ccloud auth login          # approval-gated: creates a session
scripts/provision_cloud.sh # approval-gated: creates the cluster
```

It creates the Basic cluster in `us-east-1`, creates a SQL user, and prints the
connection URL shape. **Basic clusters use a publicly trusted CA** (Let's
Encrypt, chaining to ISRG Root X1), so no cluster-specific CA file is needed —
but the DSN must say **where** the trust store is:
`&sslrootcert=/etc/ssl/certs/ca-certificates.crt`. Verified in a container on
2026-08-13: with no `sslrootcert`, libpq looks for `~/.postgresql/root.crt` in
the unprivileged user's nonexistent home; `sslrootcert=system` also fails
because psycopg's bundled OpenSSL carries its own default CA path that does not
exist in this image. The explicit Debian bundle path works and is what the
Secrets Manager DSN must carry.

### 2. Install the schema

The application never applies DDL. `runtime.start()` *verifies* the schema and
refuses to serve without it; installing is an explicit, separate act.

```bash
export SWARMBRAIN_DATABASE_URL='postgresql://<USER>:<PASSWORD>@<HOST>:26257/<DATABASE>?sslmode=verify-full'
swarmbrain-schema install
swarmbrain-schema verify
```

Run this **before** the first task starts. A task pointed at an uninitialised
database fails its readiness check and never becomes healthy, which reads as a
networking problem and is not one.

### 3. Store the two secrets

**Never on a command line** — argv is visible in shell history, in `ps`, and in
CloudTrail request logging. Paste and press Ctrl-D.

```bash
aws secretsmanager create-secret \
  --name swarm-brain/database-url \
  --description "Swarm Brain CockroachDB URL (verify-full)" \
  --secret-string file:///dev/stdin \
  --tags Key=project,Value=swarm-brain Key=event,Value=crdb-aws-hackathon-2026

aws secretsmanager create-secret \
  --name swarm-brain/token-secret \
  --description "Swarm Brain run-token HMAC secret" \
  --secret-string file:///dev/stdin \
  --tags Key=project,Value=swarm-brain Key=event,Value=crdb-aws-hackathon-2026
```

The database URL **must** carry `sslmode=verify-full` **and**
`sslrootcert=/etc/ssl/certs/ca-certificates.crt` (see step 1 for why the
explicit path is required in this image). Anything weaker than `verify-full` —
`require`, `prefer`, or nothing — authenticates the client to the server
without authenticating the server to the client, which leaves the connection
open to an active man-in-the-middle and makes "accountable memory" a claim the
transport does not support.

Generate the token secret; never type it:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Record both returned ARNs **including the six-character suffix** Secrets
Manager appends. Those are `<DATABASE_URL_SECRET_ARN>` and
`<TOKEN_SECRET_ARN>`; an ARN without the suffix is a prefix, and a prefix grant
silently widens the IAM permission.

See [`secrets.example.env`](secrets.example.env) for the full names-only
inventory.

### 4. Create the two roles

Substitute the placeholders in the JSON files first. See
[`iam/README.md`](iam/README.md) for what each grant is and why the roles are
separate.

```bash
# Execution role — used by the ECS agent, before the container starts.
aws iam create-role --role-name swarmBrainExecutionRole \
  --assume-role-policy-document file://deploy/iam/trust-policy-ecs-tasks.json
aws iam attach-role-policy --role-name swarmBrainExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam put-role-policy --role-name swarmBrainExecutionRole \
  --policy-name read-swarm-brain-secrets \
  --policy-document file://deploy/iam/execution-role-secrets.json

# Task role — used by the application, at run time.
aws iam create-role --role-name swarmBrainTaskRole \
  --assume-role-policy-document file://deploy/iam/trust-policy-ecs-tasks.json
aws iam put-role-policy --role-name swarmBrainTaskRole \
  --policy-name bedrock-and-evidence \
  --policy-document file://deploy/iam/task-role-policy.json
```

**Delete the two `s3:` statements** from `task-role-policy.json` before
applying it unless an S3 artifact adapter exists — today none does. See
[Known gaps](#known-gaps).

Enabling `amazon.titan-embed-text-v2:0` in the Bedrock console is a separate,
account-level action. An IAM grant on a model you have not enabled still fails.

### 5. Push the image

```bash
aws ecr create-repository --repository-name $REPO \
  --image-scanning-configuration scanOnPush=true \
  --tags Key=project,Value=swarm-brain Key=event,Value=crdb-aws-hackathon-2026

aws ecr get-login-password | docker login --username AWS --password-stdin \
  $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker tag swarm-brain:$IMAGE_TAG \
  $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:$IMAGE_TAG
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:$IMAGE_TAG
```

Push an **immutable tag** — the commit SHA, not `latest`. A forced
redeployment of `latest` cannot tell you which build a task is running, and the
crash-handoff beat turns on knowing exactly which process answered.

`get-login-password` mints a short-lived token from your own identity. No
registry credential is stored anywhere.

### 6. Create the log groups

Owning them explicitly makes teardown a checklist rather than a hunt, and
removes the need for `logs:CreateLogGroup` in the execution role.

```bash
for name in api worker; do
  aws logs create-log-group --log-group-name /ecs/swarm-brain/$name \
    --tags project=swarm-brain,component=$name,event=crdb-aws-hackathon-2026
  aws logs put-retention-policy --log-group-name /ecs/swarm-brain/$name \
    --retention-in-days 7
done
```

Set a retention policy. Log groups default to **never expire**, which is a
storage line that outlives the demo by years.

### 7. Register the task definitions and create the services

Fill in `<ACCOUNT_ID>`, `<REGION>`, `<ECR_IMAGE_URI>`,
`<DATABASE_URL_SECRET_ARN>`, `<TOKEN_SECRET_ARN>` and `<OWNER>` in both
[`ecs-task-definition.api.json`](ecs-task-definition.api.json) and
[`ecs-task-definition.worker.json`](ecs-task-definition.worker.json), then:

```bash
aws ecs create-cluster --cluster-name swarm-brain \
  --tags key=project,value=swarm-brain key=event,value=crdb-aws-hackathon-2026

aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-definition.api.json
aws ecs register-task-definition --cli-input-json file://deploy/ecs-task-definition.worker.json
```

The API service attaches to the ALB target group; the worker service takes no
`--load-balancers` argument at all, because nothing should ever route to it.

```bash
# API — behind the ALB, health-checked on /readyz by the target group.
aws ecs create-service --cluster swarm-brain --service-name swarm-brain-api \
  --task-definition swarm-brain-api --desired-count 1 --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<PUBLIC_SUBNET_A>,<PUBLIC_SUBNET_B>],securityGroups=[<TASK_SG>],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=<TARGET_GROUP_ARN>,containerName=swarm-brain-api,containerPort=8080" \
  --health-check-grace-period-seconds 60 \
  --tags key=project,value=swarm-brain key=component,value=api

# Worker — no load balancer, no inbound anything.
aws ecs create-service --cluster swarm-brain --service-name swarm-brain-worker \
  --task-definition swarm-brain-worker --desired-count 1 --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<PUBLIC_SUBNET_A>],securityGroups=[<TASK_SG>],assignPublicIp=ENABLED}" \
  --tags key=project,value=swarm-brain key=component,value=worker
```

The target group is `protocol=HTTP, port=8080, target-type=ip`, with its health
check on **`/readyz`** — see [Health and readiness](#health-and-readiness) for
why that differs from the container's own probe.

`--health-check-grace-period-seconds 60` matters: the API verifies the schema
against CockroachDB Cloud during startup, and without a grace period the ALB
can fail the target before the first successful check.

### 8. Verify before announcing the URL

```bash
SWARMBRAIN_TOKEN_SECRET=<the deployed token secret> \
  scripts/smoke_deploy.sh https://<PUBLIC_URL>
```

The same script that passed locally in step 0. A difference between the two
runs is a deployment difference, not a difference in what was tested.

---

## Health and readiness

| Route | Meaning | Used by |
|---|---|---|
| `GET /healthz` | the process is alive. Always `200 {"status":"ok"}`; touches no dependency. | container `healthCheck` |
| `GET /readyz` | the backend answered. `200 {"status":"ready"}` or **`503 {"status":"not_ready"}`**. On the cockroach backend this is a real round trip to the cluster. | ALB target group, the smoke script |
| `GET /console` | the static console document | judges |

**The two probes are deliberately pointed at different routes.**

- The **ALB target group checks `/readyz`.** A task that cannot reach
  CockroachDB cannot answer a single useful request, so the load balancer
  should stop sending it traffic. That is what a readiness probe is for.
- The **container `healthCheck` checks `/healthz`.** If it checked `/readyz`, a
  CockroachDB Cloud blip would make ECS *kill and replace* a process that is
  perfectly healthy and would have recovered the moment the store returned —
  and the replacement would fail the same check, producing a crash loop out of
  a transient network event.

Liveness and readiness are not the same signal. Stop routing to a task that
cannot serve; do not destroy it.

Give the target group forgiving thresholds — `interval=30`,
`unhealthyThresholdCount=3`, `healthyThresholdCount=2` — so a single slow
round trip to the cluster does not deregister the only task.

`/readyz` reports readiness but **not identity**: it does not name which store
answered. Which backend a deployment runs is fixed by `SWARMBRAIN_BACKEND` in
the task definition and visible in the task's startup logs. Confirm it there.
The smoke script says so rather than inferring it.

---

## The hosted-demo switch

`SWARMBRAIN_CONSOLE_DEMO=enabled` is set in
[`ecs-task-definition.api.json`](ecs-task-definition.api.json). It is the
hosted-demo trigger: it turns on the console's "run the scenario" control so a
judge can drive the scripted swarm from the page without a shell, which is the
difference between a demo URL and a screenshot.

Two things to know:

- **The flag has landed and the name is verified** against
  [`../src/swarmbrain/config.py`](../src/swarmbrain/config.py): the demo
  trigger is on only when the value case-folds to exactly `enabled`. **Any
  other value — including `true`, `1` and `yes` — leaves it off.** That is
  fail-closed on purpose, and it is the opposite of how the other boolean
  settings parse, so it is worth reading twice.
- **Leave it unset on any deployment you do not want a stranger driving.** The
  console is otherwise a read-only page that fetches run data with a token the
  viewer pastes in; the demo trigger is the one control that causes writes.

---

## Cost envelope

Rates are **published `us-east-1` on-demand prices**. They are stable
line items, but they are **not re-verified against the pricing pages as part of
this document** — check the four that matter (Fargate ARM64, ALB, public IPv4,
Secrets Manager) before approving spend. The conclusion below is robust to
±20% on any of them.

**Window priced:** deploy ~Aug 13 through the Sep 15 judging tail = **33 days =
792 hours**.

| Line item | Rate | Shape A | Shape B |
|---|---|---:|---:|
| Fargate ARM64, API task (0.5 vCPU / 1 GB) | $0.03238/vCPU-hr + $0.00356/GB-hr | $15.64 | $15.64 |
| Fargate ARM64, worker task (0.25 vCPU / 0.5 GB) | same | $7.82 | $0.59 |
| Application Load Balancer, hours | $0.0225/hr | $17.82 | $17.82 |
| ALB LCUs | $0.008/LCU-hr | ~$1.00 | ~$1.00 |
| Public IPv4 — ALB nodes (2 AZs) | $0.005/addr-hr | $7.92 | $7.92 |
| Public IPv4 — tasks | $0.005/addr-hr | $7.92 | $4.26 |
| Secrets Manager, 2 secrets | $0.40/secret-month | $0.88 | $0.88 |
| CloudWatch Logs, 7-day retention | $0.50/GB ingest | ~$0.30 | ~$0.30 |
| ECR storage (~360 MB image) | $0.10/GB-month | ~$0.04 | ~$0.04 |
| **Total over the window** | | **≈ $59** | **≈ $48** |

- **Shape A** — both services running continuously, as the templates ship.
- **Shape B** — the worker scaled to 0 except during rehearsals and the demo
  (~60 hours). Embedding work simply waits in the durable queue while it is
  down, which is exactly what a durable queue is for, and drains when it comes
  back.

**Neither shape fits a ~$40 budget.** That is the honest finding, and the
reason is a single line:

> **The load balancer and its public IPv4 addresses are ~$27 of the bill —
> more than both Fargate tasks combined.**

### Levers, in the order worth taking

| Lever | Saves | Costs you |
|---|---:|---|
| Share an existing ALB (add a host/path listener rule) | **~$26** | nothing, if one exists in the VPC |
| Drop the ALB; expose the API task's public IP directly | **~$27** | HTTPS, and a stable URL — the IP changes whenever the task is replaced. **This is disqualifying for a demo URL that must live to Sep 15.** |
| Run the worker as a second **container in the API task** rather than a second service | ~$8 | the "two services" shape; you keep two processes and two containers, but one task and one public IP |
| Scale the worker to 0 between rehearsals | ~$7 | nothing — the queue is durable |
| Drop the API to 0.25 vCPU / 0.5 GB | ~$8 | headroom; risks an OOM on camera |

**Recommendation:** take levers 3 and 4 together. One task carrying both
containers, worker scaled with the demo, keeps the ALB and the real HTTPS URL
and lands at **≈ $40** — at the budget rather than over it. Combine with lever 1
if any ALB already exists and it lands at ≈ $14.

Do **not** take lever 2 to save money. A demo URL that has to survive to Sep 15
needs stable DNS and HTTPS; an ephemeral public IP on port 8080 is not a
submission.

### CockroachDB Cloud, and Bedrock

Both are outside this table.

- **CockroachDB Cloud Basic** is billed by CockroachDB Cloud, not AWS, and the
  free tier is expected to cover this workload. Confirm against your credit
  before assuming $0.
- **Bedrock embeddings are noise.** Titan Text Embeddings V2 is $0.02 per 1M
  input tokens. The whole demo corpus is a few thousand tokens; a full run
  embeds well under 5,000, so it is **under $0.0001 per run**. It will not
  appear on the bill. There is no chat model in this deployment at all — Swarm
  Brain runs no LLM of its own.

---

## Tags

Every resource created for this deployment carries all six. They are what makes
the teardown inventory verifiable instead of a memory exercise.

| Tag | Value | Purpose |
|---|---|---|
| `project` | `swarm-brain` | the cost-allocation key; activate it in Billing |
| `component` | `api`, `worker`, `image`, `secret`, `iam`, `logs`, `alb` | which piece |
| `event` | `crdb-aws-hackathon-2026` | separates this from anything else in the account |
| `owner` | `<OWNER>` | who to ask before deleting |
| `teardown-after` | `2026-09-15` | the judging commitment; after this date, deletion needs no further discussion |
| `managed-by` | `manual` | everything here is created by hand; nothing provisions on your behalf |

Find everything at teardown time:

```bash
aws resourcegroupstaggingapi get-resources \
  --tag-filters Key=project,Values=swarm-brain \
  --query 'ResourceTagMappingList[].ResourceARN' --output table
```

IAM roles are not returned by that API. Treat the list below — not the tag
query — as the authoritative inventory.

---

## Teardown inventory

Every resource this deployment brings into existence, and the command that
removes it. Work top to bottom; the order matters, because a resource in use
cannot be deleted.

| # | Resource | Delete with |
|---|---|---|
| 1 | ECS service `swarm-brain-worker` | `aws ecs delete-service --cluster swarm-brain --service swarm-brain-worker --force` |
| 2 | ECS service `swarm-brain-api` | `aws ecs delete-service --cluster swarm-brain --service swarm-brain-api --force` |
| 3 | ALB listener | `aws elbv2 delete-listener --listener-arn <ARN>` |
| 4 | Application Load Balancer | `aws elbv2 delete-load-balancer --load-balancer-arn <ARN>` — **the $18 line; verify it is gone** |
| 5 | Target group | `aws elbv2 delete-target-group --target-group-arn <ARN>` |
| 6 | ACM certificate (if you requested one for HTTPS) | `aws acm delete-certificate --certificate-arn <ARN>` — only after the listener is gone |
| 7 | Public IPv4 addresses | released when the ALB and the last task are gone |
| 8 | Task security group | `aws ec2 delete-security-group --group-id <TASK_SG>` — after the tasks stop |
| 9 | ALB security group | `aws ec2 delete-security-group --group-id <ALB_SG>` — after the ALB is gone |
| 10 | Task definition revisions | `aws ecs deregister-task-definition --task-definition swarm-brain-api:<N>` per revision, then `aws ecs delete-task-definitions --task-definitions swarm-brain-api:<N>`; repeat for `swarm-brain-worker` |
| 11 | ECS cluster | `aws ecs delete-cluster --cluster swarm-brain` |
| 12 | CloudWatch log groups | `aws logs delete-log-group --log-group-name /ecs/swarm-brain/api` and `.../worker` — **export any evidence first** |
| 13 | ECR images | `aws ecr batch-delete-image --repository-name swarm-brain --image-ids imageTag=<TAG>` |
| 14 | ECR repository | `aws ecr delete-repository --repository-name swarm-brain --force` |
| 15 | Secrets Manager, database URL | `aws secretsmanager delete-secret --secret-id swarm-brain/database-url --recovery-window-in-days 7` |
| 16 | Secrets Manager, token secret | `aws secretsmanager delete-secret --secret-id swarm-brain/token-secret --recovery-window-in-days 7` |
| 17 | Task role inline policy | `aws iam delete-role-policy --role-name swarmBrainTaskRole --policy-name bedrock-and-evidence` |
| 18 | Task role | `aws iam delete-role --role-name swarmBrainTaskRole` |
| 19 | Execution role inline policy | `aws iam delete-role-policy --role-name swarmBrainExecutionRole --policy-name read-swarm-brain-secrets` |
| 20 | Execution role managed-policy attachment | `aws iam detach-role-policy --role-name swarmBrainExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy` |
| 21 | Execution role | `aws iam delete-role --role-name swarmBrainExecutionRole` |
| 22 | ECS service-linked roles | leave them. Account-wide, cost nothing, and deleting them breaks other ECS usage. |
| 23 | Bedrock model access | an account setting, not a resource. Costs nothing when idle; revoke only if you want to. |

Secrets keep billing at $0.40/month until the recovery window closes. That is
deliberate — a recovery window rather than
`--force-delete-without-recovery`, in case teardown was premature.

**Outside AWS:** the CockroachDB Cloud cluster, its SQL user, and the read-only
Managed MCP connection are deleted in the CockroachDB Cloud console or with
`ccloud cluster delete`. No `aws` command touches them.

Confirm the account is clean:

```bash
aws ecs list-services --cluster swarm-brain
aws ecs list-clusters
aws elbv2 describe-load-balancers --query 'LoadBalancers[].LoadBalancerName'
aws ecr describe-repositories --query 'repositories[].repositoryName'
aws secretsmanager list-secrets --query 'SecretList[].Name'
aws iam list-roles --query "Roles[?starts_with(RoleName,'swarmBrain')].RoleName"
aws logs describe-log-groups --log-group-name-prefix /ecs/swarm-brain
```

Then check Cost Explorer filtered on `project=swarm-brain` two days later. A
tagged resource you forgot shows up there before it shows up on a statement.

---

## Known gaps

Honest at time of writing. None is a defect in this directory; each is a
dependency that has to land or a decision that has to be made.

| Gap | Consequence |
|---|---|
| ~~**Nothing here has been applied.**~~ **Resolved 2026-08-16** — applied in full; recorded in the private as-built record. | none |
| ~~**The Bedrock path has never been called.**~~ **Resolved 2026-08-16** — called live twice: with operator credentials (`evidence/20260816T225339Z-bedrock-smoke.json`) and from the ECS task role, which stamped `amazon.titan-embed-text-v2:0` vectors during a demo run on the public URL. | none |
| **This directory does not create its own networking.** [Step 7](#7-register-the-task-definitions-and-create-the-services) consumes `<PUBLIC_SUBNET_A>`, `<PUBLIC_SUBNET_B>`, `<TASK_SG>` and `<TARGET_GROUP_ARN>`, but nothing here creates the ALB, the target group, the listener, or either security group. | The real deployment had to write those commands from scratch. They are recorded in the private as-built record and should be folded back in here. |
| **The deployed URL is HTTP, not HTTPS.** ACM will not issue a public certificate for the ALB's own `*.elb.amazonaws.com` hostname, and no domain was available. | Judges see a plain `http://` link. Fixable without touching any task, service, or image: request a certificate for a domain and add a 443 listener. |
| **No S3 artifact adapter.** `ports/artifacts.py` is defined; nothing implements it. | The two `s3:` statements in `iam/task-role-policy.json` are unexercised. They were deleted before the task role was applied, and no bucket exists. |
| **The `ccloud` commands are not verified against a local CLI.** It is not installed here. | [`../scripts/provision_cloud.sh`](../scripts/provision_cloud.sh) marks each command with its confidence and prints before it runs. Read its output before approving. |
| ~~`SWARMBRAIN_CONSOLE_DEMO` is unverified.~~ **Resolved** — the flag has landed and the name and its exact-match `enabled` semantics are verified against `config.py`. | none |
| **The budget does not fit the briefed shape.** | See [Cost envelope](#cost-envelope). This needs a decision, not a workaround. |
| **`readonlyRootFilesystem` is `false`.** | A hardening option left off because Fargate does not support `tmpfs` mounts and Python's need for a writable `/tmp` has not been validated. Turn it on only after testing. |
| **No service-health alarms.** A `$70`/month AWS Budget with email notifications was created on 2026-08-16, but that watches spend, not health. | A demo URL that must survive to Sep 15 should have at least an ALB `UnHealthyHostCount` alarm and a 5xx alarm. Neither exists yet. |
