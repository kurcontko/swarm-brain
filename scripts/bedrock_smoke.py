#!/usr/bin/env python3
"""Prove the Amazon Bedrock embedding path works, the moment credentials land.

The Bedrock adapter (``src/swarmbrain/adapters/embeddings/bedrock.py``) is
committed, covered by unit tests against a fake client, and named in the ECS
task role — but it has never been called against the real service. This script
is the activation proof, and it is deliberately the *first* thing to run when
AWS credentials arrive, before any image is pushed or any ECS service exists.
A failure here is a five-second fix; the same failure discovered from a
CloudWatch log inside a Fargate task is an afternoon.

WHAT IT PROVES, in order:

  1. Provider  Composes the real ``BedrockEmbeddingProvider``
               (amazon.titan-embed-text-v2:0, 1024 dimensions) and embeds two
               fixed, deliberately unrelated sentences. Asserts both vectors
               have exactly 1024 dimensions, are unit-normalised (Titan is
               asked for ``normalize: true``), and are NOT near-duplicates of
               each other. Two identical vectors would mean the provider is
               returning a constant, which is the failure mode a dimension
               check alone cannot see.

  2. Round trip (only when SWARMBRAIN_DATABASE_URL is set)
               Composes the REAL in-process runtime against that database —
               the same ``build_runtime`` the API and the worker call — then:
                 publishes one memory with embeddings enabled,
                 drains the durable embedding queue with the real worker,
                 runs a dense recall and asserts the memory comes back with a
                 ``semantic_match`` reason.
               That chain is the whole vector story end to end: Bedrock ->
               outbox work item -> VECTOR(1024) column -> ANN index -> recall.

  3. Evidence  Writes a redacted JSON artifact to ``evidence/``. No credential,
               no account identifier, no database host, and no raw vector — a
               model name, dimensions, cosines, digests and timings.

USAGE

    set -a; source .env; set +a          # nothing auto-loads .env
    python3 scripts/bedrock_smoke.py

    python3 scripts/bedrock_smoke.py --stub    # no credentials, no network
    python3 scripts/bedrock_smoke.py --no-database

WITHOUT CREDENTIALS it exits non-zero with one precise line naming what is
missing. It never prints a stack trace: a traceback from a missing environment
variable teaches nothing and buries the answer.

COST. Titan Text Embeddings V2 is $0.02 per 1M input tokens. This script embeds
three short strings — well under 100 tokens, so under $0.000002 per run. It is
free in every sense that matters, but it is a real paid API call and it is only
made when real credentials are present.

WHAT IT WRITES TO THE DATABASE. The round trip creates one run, one agent, one
memory and one work item, all under freshly generated UUIDs that nothing else
will ever address. It reads, modifies and deletes nothing that already exists.
Skip it entirely with --no-database.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 1024
EVIDENCE_DIR = REPO_ROOT / "evidence"

# Two fixed sentences from unrelated domains. "Unrelated" is the load-bearing
# property: a provider stuck on a constant vector, or one silently ignoring its
# input, produces cosine 1.0 here and is caught. Sentences about the same topic
# would score high legitimately and prove nothing.
SENTENCE_A = (
    "The payment reconciliation worker exhausted the database connection pool "
    "during the nightly batch and stalled every downstream settlement job."
)
SENTENCE_B = (
    "Wandering albatrosses cross the southern ocean by dynamic soaring, "
    "extracting energy from the wind gradient above the waves."
)

# Above this, the two unrelated sentences are suspiciously alike and the
# provider is not doing its job. Real Titan scores these well under 0.5.
MAX_UNRELATED_COSINE = 0.90
# Below this, the round trip did not retrieve what it stored.
MIN_ROUND_TRIP_COSINE = 0.50


class SmokeFailure(RuntimeError):
    """A named, expected failure. Printed as one line, never as a traceback."""


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------

_ACCOUNT_ID = re.compile(r"\b\d{12}\b")
_DATABASE_URL = re.compile(r"\b(postgres|postgresql)://[^\s\"']+", re.IGNORECASE)
_AWS_KEY_ID = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")


def redact(text: str) -> str:
    """Strip the three things that must never reach an evidence artifact."""

    text = _DATABASE_URL.sub("<database-url-redacted>", text)
    text = _AWS_KEY_ID.sub("<aws-key-id-redacted>", text)
    return _ACCOUNT_ID.sub("<account-id-redacted>", text)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def say(message: str) -> None:
    print(redact(message), flush=True)


def step(message: str) -> None:
    print(f"\n== {redact(message)}", flush=True)


def ok(message: str) -> None:
    print(f"  ok    {redact(message)}", flush=True)


def note(message: str) -> None:
    print(f"  --    {redact(message)}", flush=True)


# ---------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------


def credential_mode() -> str:
    """Name the credential source boto3 will use, or raise a precise failure.

    boto3 has a long resolution chain, and "it found something" is not the same
    as "it found what you meant". Naming the source up front means a run that
    picks up a stale default profile is visible in the output rather than in a
    confusing AccessDenied twenty seconds later.
    """

    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return "static keys (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)"
    if os.getenv("AWS_PROFILE"):
        return f"shared config profile (AWS_PROFILE={os.environ['AWS_PROFILE']})"
    if os.getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"):
        return "ECS task role (container credentials endpoint)"
    if os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE"):
        return "web identity token (IRSA / OIDC)"
    raise SmokeFailure(
        "no AWS credentials in the environment. "
        "missing: AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE. "
        "Load them with:  set -a; source .env; set +a   "
        "(or re-run with --stub to exercise the wiring without credentials)"
    )


def resolve_region() -> str:
    for name in ("SWARMBRAIN_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    raise SmokeFailure(
        "no AWS region in the environment. "
        "missing: AWS_REGION (or SWARMBRAIN_AWS_REGION). "
        "The deployment targets us-east-1"
    )


# ---------------------------------------------------------------------------
# the stub client
# ---------------------------------------------------------------------------


class StubBedrockClient:
    """A deterministic stand-in for ``bedrock-runtime``, for --stub runs.

    It speaks the exact wire shape ``BedrockEmbeddingProvider._invoke`` expects
    — ``invoke_model(modelId=, body=, accept=, contentType=)`` returning an
    object with ``["body"].read()`` — so a --stub run exercises every line of
    the real adapter except the network call itself. That is what makes it a
    wiring proof rather than a mock of the thing under test.

    The vectors are a hashed bag of words: unrelated sentences land nearly
    orthogonal, and the same sentence embedded twice — or embedded once as raw
    text and once inside a JSON projection — lands nearly parallel. That is the
    behaviour the assertions below depend on, so a --stub run tests the
    assertions too.

    It is NOT a substitute for the real proof, and the evidence artifact says so
    in its `mode` field. A stub can never establish that Bedrock works.
    """

    invocations = 0

    def invoke_model(
        self,
        *,
        modelId: str,  # noqa: N803 - boto3's parameter name, matched exactly
        body: str,
        accept: str,
        contentType: str,  # noqa: N803 - boto3's parameter name, matched exactly
    ) -> dict[str, Any]:
        StubBedrockClient.invocations += 1
        request = json.loads(body)
        vector = self._embed(request["inputText"], int(request["dimensions"]))
        payload = json.dumps({"embedding": list(vector)}).encode("utf-8")
        return {"body": _StubBody(payload)}

    @staticmethod
    def _embed(text: str, dimensions: int) -> tuple[float, ...]:
        values = [0.0] * dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            values[0] = 1.0
            norm = 1.0
        return tuple(value / norm for value in values)


class _StubBody:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


# ---------------------------------------------------------------------------
# vector helpers
# ---------------------------------------------------------------------------


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise SmokeFailure("the provider returned a zero vector")
    return dot / (left_norm * right_norm)


def vector_digest(values: tuple[float, ...]) -> str:
    """A stable digest of a vector, so evidence can cite it without carrying it."""

    joined = ",".join(f"{value:.8f}" for value in values)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. the provider check
# ---------------------------------------------------------------------------


async def check_provider(provider: Any) -> dict[str, Any]:
    step("1. provider — embed two unrelated sentences")

    started = time.monotonic()
    first = await provider.embed_query(SENTENCE_A)
    second = await provider.embed_query(SENTENCE_B)
    elapsed_ms = round((time.monotonic() - started) * 1000)

    for label, vector in (("A", first), ("B", second)):
        if len(vector) != DIMENSIONS:
            raise SmokeFailure(
                f"sentence {label} returned {len(vector)} dimensions, expected {DIMENSIONS}"
            )
    ok(f"both vectors have {DIMENSIONS} dimensions")

    norms = tuple(math.sqrt(sum(value * value for value in vector)) for vector in (first, second))
    for label, norm in zip(("A", "B"), norms, strict=True):
        if not 0.98 <= norm <= 1.02:
            raise SmokeFailure(
                f"sentence {label} has norm {norm:.4f}; the provider asks for "
                "normalize=true, so ~1.0 is expected"
            )
    ok(f"both vectors are unit-normalised (|A|={norms[0]:.4f}, |B|={norms[1]:.4f})")

    similarity = cosine(first, second)
    if similarity > MAX_UNRELATED_COSINE:
        raise SmokeFailure(
            f"two unrelated sentences scored cosine {similarity:.4f}, above "
            f"{MAX_UNRELATED_COSINE}. The provider may be returning a constant "
            "vector or ignoring its input"
        )
    ok(f"unrelated sentences are distinguishable (cosine {similarity:.4f})")
    note(f"provider: {provider.model_name}, {elapsed_ms} ms for two embeddings")

    return {
        "model": provider.model_name,
        "dimensions": DIMENSIONS,
        "sentence_a_sha256": hashlib.sha256(SENTENCE_A.encode("utf-8")).hexdigest(),
        "sentence_b_sha256": hashlib.sha256(SENTENCE_B.encode("utf-8")).hexdigest(),
        "vector_a_sha256": vector_digest(first),
        "vector_b_sha256": vector_digest(second),
        "vector_a_norm": round(norms[0], 6),
        "vector_b_norm": round(norms[1], 6),
        "unrelated_cosine": round(similarity, 6),
        "max_unrelated_cosine": MAX_UNRELATED_COSINE,
        "elapsed_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# 2. the durable round trip
# ---------------------------------------------------------------------------


async def check_round_trip(database_url: str, region: str, stub_client: Any | None) -> dict[str, Any]:
    step("2. round trip — publish, embed, recall through the real runtime")

    from swarmbrain.application.memory_service import SEMANTIC_MATCH_REASON
    from swarmbrain.application.runtime import build_runtime
    from swarmbrain.config import ApiSettings, BackendKind, EmbeddingsKind
    from swarmbrain.domain.agents import ActorContext
    from swarmbrain.domain.memory import RecallQuery, RememberCommand, Visibility

    settings = ApiSettings(
        backend=BackendKind.COCKROACH,
        token_secret="bedrock-smoke-local-only-secret",
        database_url=database_url,
        embeddings=EmbeddingsKind.BEDROCK,
        embeddings_model=MODEL_ID,
        embeddings_dimensions=DIMENSIONS,
        aws_region=region,
        database_pool_min_size=1,
        database_pool_max_size=4,
    )
    runtime = build_runtime(settings)

    # A --stub run swaps only the boto3 client, deep inside the composed
    # provider. Everything above it — runtime composition, the work queue, the
    # worker, the projection, the ANN read — is the real code path.
    if stub_client is not None:
        if runtime.embeddings is None:
            raise SmokeFailure("the composed runtime has no embedding provider")
        runtime.embeddings._client = stub_client  # noqa: SLF001 - deliberate test seam

    # A namespace nothing else will ever address.
    actor = ActorContext(
        tenant_id=str(uuid4()),
        project_id=str(uuid4()),
        repository_id=str(uuid4()),
        swarm_id=str(uuid4()),
        run_id=str(uuid4()),
        agent_id=str(uuid4()),
        harness="bedrock-smoke",
        provider="swarmbrain",
        model="none",
        capabilities=frozenset({"run:join", "memory:publish", "memory:recall"}),
    )

    await runtime.start()
    try:
        # join creates the run row the memory's foreign keys need.
        await runtime.coordination.join(actor)
        ok(f"joined a fresh run ({actor.run_id})")

        published = await runtime.memory.publish(
            actor,
            RememberCommand(
                idempotency_key=f"bedrock-smoke-{uuid4()}",
                kind="observation",
                visibility=Visibility.RUN,
                content={"statement": SENTENCE_A},
            ),
        )
        if published.memory is None:
            raise SmokeFailure("publish returned no memory; the policy rejected it")
        memory_id = published.memory.memory_id
        ok(f"published one memory ({memory_id})")

        # Publishing only ENQUEUES the embedding — the provider call happens on
        # the durable queue, outside the memory transaction. Draining it here is
        # exactly what the deployed worker service does.
        started = time.monotonic()
        completed = await runtime.embedding_worker().run_once(
            f"bedrock-smoke-{uuid4()}", limit=4
        )
        embed_ms = round((time.monotonic() - started) * 1000)
        subjects = {item.item.subject_id for item in completed}
        if memory_id not in subjects:
            raise SmokeFailure(
                "the embedding worker did not complete this memory's work item; "
                f"completed {len(completed)} item(s)"
            )
        ok(f"the durable worker embedded it via Bedrock ({embed_ms} ms)")

        # The recall embeds the query text through the same provider, so a hit
        # here means the stored vector and a freshly computed one agree.
        recalled = await runtime.memory.recall(
            actor,
            RecallQuery(text=SENTENCE_A, min_score=0.1, limit=10),
        )
        hits = [hit for hit in recalled.hits if hit.memory.memory_id == memory_id]
        if not hits:
            raise SmokeFailure(
                "dense recall did not return the memory that was just embedded; "
                f"{len(recalled.hits)} other hit(s) came back"
            )
        hit = hits[0]
        if SEMANTIC_MATCH_REASON not in hit.reasons:
            raise SmokeFailure(
                "the memory was recalled, but not by the semantic lane "
                f"(reasons: {', '.join(hit.reasons) or 'none'}). "
                "The vector round trip is unproven"
            )
        ok(f"dense recall returned it with reasons: {', '.join(hit.reasons)}")

        if hit.score < MIN_ROUND_TRIP_COSINE:
            raise SmokeFailure(
                f"round-trip score {hit.score:.4f} is below {MIN_ROUND_TRIP_COSINE}"
            )
        ok(f"round-trip score {hit.score:.4f}")

        return {
            "memory_id": memory_id,
            "run_id": actor.run_id,
            "work_items_completed": len(completed),
            "embed_elapsed_ms": embed_ms,
            "recall_reasons": list(hit.reasons),
            "recall_score": round(hit.score, 6),
            "min_round_trip_cosine": MIN_ROUND_TRIP_COSINE,
            "total_hits": len(recalled.hits),
        }
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------


def write_evidence(artifact: dict[str, Any]) -> Path:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = EVIDENCE_DIR / f"{stamp}-bedrock-smoke.json"
    serialized = json.dumps(artifact, indent=2, sort_keys=True)
    # Belt and braces: the artifact is assembled from named fields that should
    # already be clean, and is redacted again on the way out.
    path.write_text(redact(serialized) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


async def run(arguments: argparse.Namespace) -> int:
    from swarmbrain.adapters.embeddings.bedrock import BedrockEmbeddingProvider

    stub_client: Any | None = None
    if arguments.stub:
        mode = "stubbed-provider"
        region = os.getenv("AWS_REGION") or "unset"
        stub_client = StubBedrockClient()
        say("bedrock_smoke: --stub — no credentials used, no network call made")
        say("bedrock_smoke: this proves the WIRING only. It is not Bedrock evidence.")
    else:
        mode = "live-bedrock"
        source = credential_mode()
        region = resolve_region()
        say(f"bedrock_smoke: credentials from {source}")
        say(f"bedrock_smoke: region {region}, model {MODEL_ID}")

    provider = BedrockEmbeddingProvider(
        model_id=MODEL_ID,
        dimensions=DIMENSIONS,
        region_name=None if arguments.stub else region,
        client=stub_client,
    )

    artifact: dict[str, Any] = {
        "artifact": "bedrock-smoke",
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "region": region,
        "provider": await check_provider(provider),
    }

    database_url = (os.getenv("SWARMBRAIN_DATABASE_URL") or "").strip()
    if arguments.no_database:
        step("2. round trip — skipped (--no-database)")
        artifact["round_trip"] = {"skipped": "--no-database"}
    elif not database_url:
        step("2. round trip — skipped")
        note("SWARMBRAIN_DATABASE_URL is not set, so the durable half is unproven.")
        note("Set it to prove Bedrock -> work queue -> VECTOR(1024) -> ANN -> recall.")
        artifact["round_trip"] = {"skipped": "SWARMBRAIN_DATABASE_URL is not set"}
    else:
        artifact["round_trip"] = await check_round_trip(database_url, region, stub_client)

    if arguments.stub:
        artifact["stub_invocations"] = StubBedrockClient.invocations

    path = write_evidence(artifact)

    step("result")
    ok(f"evidence written to {path.relative_to(REPO_ROOT)}")
    if mode == "stubbed-provider":
        note("mode=stubbed-provider — wiring proven, Bedrock NOT proven")
    else:
        note("mode=live-bedrock — the Bedrock embedding path is live")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="bedrock_smoke.py",
        description="Prove the Bedrock embedding path, end to end.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="use a deterministic offline client; proves wiring, not Bedrock",
    )
    parser.add_argument(
        "--no-database",
        action="store_true",
        help="skip the durable round trip even if SWARMBRAIN_DATABASE_URL is set",
    )
    arguments = parser.parse_args()

    try:
        return asyncio.run(run(arguments))
    except SmokeFailure as failure:
        # One line, no traceback. The whole point of this script is that its
        # failures are readable by someone who has not read its source.
        print(f"bedrock_smoke: {redact(str(failure))}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("bedrock_smoke: interrupted", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - the top-level guard
        print(
            f"bedrock_smoke: {type(error).__name__}: {redact(str(error))}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
