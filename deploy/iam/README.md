# IAM for the Swarm Brain tasks

JSON has no comments, so the reasoning lives here. Every file in this directory
is a **template**: the `<PLACEHOLDER>` tokens are filled in at approval time and
**nothing here has been applied to an AWS account**. No script in this
repository applies them.

## Two roles, never one

ECS gives a task two distinct identities, and collapsing them is the most
common way a demo ends up over-permissioned.

| Role | Used by | Holds |
|---|---|---|
| **Execution role** | the ECS agent, *before* the container starts | pull the image from ECR, create/write the log group, read the two secrets |
| **Task role** | the application process, at run time | Bedrock and S3, and nothing else |

The application can never pull an image or read a secret, because those
permissions belong to a role it does not assume. The ECS agent can never call
Bedrock. Keep it that way.

Both roles use the same trust policy shape,
[`trust-policy-ecs-tasks.json`](trust-policy-ecs-tasks.json). The
`aws:SourceAccount` and `aws:SourceArn` conditions are not decoration: without
them the role is assumable by the ECS service on behalf of *any* account, which
is the confused-deputy case AWS documents for service principals.

**Both ECS services share both roles.** The API and the worker run the same
image and need the same access — the worker is arguably the *only* thing that
calls Bedrock, since embedding happens on the durable queue, but the API calls
`embed_query` on the recall path too. Two role pairs would be two things to
keep in step for no reduction in blast radius.

## Execution role

Attach the AWS-managed `AmazonECSTaskExecutionRolePolicy` — it covers the ECR
pull and `logs:CreateLogStream` / `logs:PutLogEvents` — plus the inline policy
in [`execution-role-secrets.json`](execution-role-secrets.json).

Three notes on that file:

- It grants `secretsmanager:GetSecretValue` on **two named secret ARNs**, not on
  `*`. Use the **full ARN including the six-character suffix** Secrets Manager
  appends (`...:secret:swarm-brain/database-url-AbCdEf`). An ARN without the
  suffix is a prefix, and a prefix grant quietly widens to any secret whose name
  starts the same way.
- If the secrets use a customer-managed KMS key rather than
  `aws/secretsmanager`, the execution role also needs `kms:Decrypt` on that key
  ARN. With the AWS managed key it does not.
- `logs:CreateLogGroup` is **not** in the managed policy, and the task
  definitions set `awslogs-create-group: "true"`. Either create the two log
  groups ahead of time — the teardown inventory is simpler if you do, because
  you then own them explicitly — or keep the second statement here, which is
  scoped to exactly those two group names.

## Task role — Bedrock

[`task-role-policy.json`](task-role-policy.json) grants `bedrock:InvokeModel` on
**one model**:

- `amazon.titan-embed-text-v2:0`. This is not a guess and not a free choice: it
  is the default `model_id` in
  [`../../src/swarmbrain/adapters/embeddings/bedrock.py`](../../src/swarmbrain/adapters/embeddings/bedrock.py),
  and the dense projection stamps every stored vector with the model that
  produced it. Changing the model invalidates stored vectors — it is a reindex,
  not a config change.

Foundation-model ARNs have **no account ID** —
`arn:aws:bedrock:<REGION>::foundation-model/<id>`, with the empty account
segment. That is correct, not a typo.

Deliberately absent:

- **`bedrock:InvokeModelWithResponseStream`.** The provider calls
  `invoke_model` and reads the whole body. Granting the streaming variant would
  be permission the application cannot use.
- **`bedrock:Converse` and any chat model.** Swarm Brain runs no LLM of its
  own. It is a memory and coordination kernel; the agents that call it bring
  their own models and their own credentials. There is no chat-model line in
  this policy because there is no chat-model call in the code.
- **No inference profile.** Titan embeddings are available as a plain in-region
  foundation model in `us-east-1`, so there is no `us.`/`global.` profile ARN
  to grant. If you deploy to a region where that is not true, the profile ARN
  *does* carry an account ID and must be added alongside the foundation-model
  ARN in **every region the profile can route to**.

### Model access is separate from IAM

Enabling a model in the Bedrock console is an account-level action, not a
policy. An IAM grant on a model you have not enabled still fails, with an
`AccessDeniedException` that reads like a policy problem and is not. Enabling
models requires operator approval like every other account change.

## Task role — S3

**This grant is currently written and unexercised. Read this before creating a
bucket.**

`ports/artifacts.py` defines `ArtifactReader` / `ArtifactWriter`, but **no S3
adapter implements them** — `grep boto3 src/` returns only the Bedrock provider.
The two `s3:` statements in `task-role-policy.json` are the shape the evidence
export *will* need, kept here so the policy is reviewed once rather than
bolted on under time pressure.

Until an adapter exists: **do not create the bucket, and delete both `s3:`
statements before applying the policy.** A grant on a bucket that does not exist
is harmless; a bucket nobody wrote an adapter for is a teardown line and a
$0-value resource on the inventory.

When it does land, the shape is deliberate:

- `s3:PutObject` / `s3:GetObject` on **one prefix inside one bucket**, never
  `arn:aws:s3:::<bucket>/*`.
- `s3:ListBucket` is a *bucket*-level action, so its resource is the bucket ARN
  with no `/*`. Without the `s3:prefix` condition that grant would list the
  whole bucket, so the condition is what actually scopes it.
- No `s3:DeleteObject`. Evidence the application can delete is weaker evidence,
  and teardown is a human action against a bucket the human owns.

## What is deliberately not here

- **No static access keys, anywhere.** Not in the image, not in the task
  definitions, not in an environment variable, not in this repository. The task
  role is the only credential path in the deployment, and boto3 discovers it
  from the container credentials endpoint by itself —
  `BedrockEmbeddingProvider` constructs no session and reads no credential.

  The `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` pair in `.env.example` is
  for an **operator's laptop** running `scripts/bedrock_smoke.py` before the
  deployment exists. It has no place in the deployed task.
- **No `secretsmanager:*` on the task role.** ECS injects the secrets as
  environment variables using the *execution* role, before the process starts.
  The application reads `os.getenv` and has no idea Secrets Manager exists, so
  granting it a read would be permission nothing calls.
- **No `iam:*`, no `ecs:*`, no `logs:*` on the task role.** The application
  never manages infrastructure and never writes a log through the API — stdout
  is collected by the log driver, which is the execution role's job.
- **No CockroachDB permissions**, because CockroachDB is not an AWS resource.
  Its authorisation is the SQL user inside `SWARMBRAIN_DATABASE_URL`, which
  should be a least-privilege user scoped to the demo database — not a cluster
  admin. See [`../README.md`](../README.md).
