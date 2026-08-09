# Paper-driven memory runtime

The research review now has a concrete runtime slice. This note separates what
is shipped from what the cited work still suggests building; the detailed
evidence and links remain in the
[agent-memory research dump](research/sota-agent-memory-retrieval-2026-08-07.md).

## What this branch ships

| Research signal | Framework change | Runtime invariant |
| --- | --- | --- |
| Mastra Observational Memory and Mem0: distilled, typed observations are more useful than replaying raw history | Optional OpenAI-compatible typed extraction proposes bounded candidates, dates, aliases, metadata, and safe local relations; deterministic extraction remains the fallback | Provider output cannot choose identity, scope, trust, visibility, lifecycle, or invented source offsets |
| MemClaw: shared memory needs scoped, provenance-preserving, policy-governed propagation | Automatic activation retrieves confirmed memories only and reuses the existing scope, trust, temporal, and supersession filters | A task claim cannot inject tentative, stale, refuted, untrusted, or out-of-scope memory |
| MemOS and finite-context work: retrieval and activation are different lifecycle decisions | `MemoryActivationRequest` models task claim, checkpoint resume, dependency unblock, tool error, repeated failure, and explicit triggers; task claim and resume are wired first | Query text and memory content never enter activation telemetry; deterministic IDs make one trigger per lease idempotent |
| Anthropic context engineering and context-rot findings: small relevant context beats indiscriminate history | Canonical memory blocks are greedily packed to a 2,048-token default, with a final result cap and `PackingTrace` | The exact `activation_context` sent over HTTP/MCP is the representation measured against the budget; the raw recall bundle is excluded |
| Answer-in-context and trace-derived evaluation: retrieval is not use | Metrics separately count activation decisions, selected memories, lease-scoped citations, and citations of another agent's memory that match activation for the same task/lease/consumer | Ordinary recall and self-reported citations cannot inflate proven cross-agent use |
| Multi-agent workflow-memory research: the test must be causal | The demo uses exactly four agents and two task waves; fresh opaque facts exist only in Wave-A memories, and the same hidden verifier fails without context and passes with the delivered Wave-B context | Success requires activation, citation, dependency release, and a fenced cross-provider checkpoint handoff |

## End-to-end memory loop

```text
raw source
  -> deterministic + optional typed-provider candidates
  -> exact local evidence validation and fenced materialization
  -> tentative/confirmed governance and bitemporal supersession
  -> trigger-scoped confirmed retrieval
  -> relevance floor and token-bounded canonical packing
  -> agent checkpoint/completion citations
  -> activation/citation/cross-agent outcome metrics
```

The critical boundary is between retrieval and activation. Retrieval may
produce a ranked candidate set; activation decides whether any of it is safe
and useful enough to occupy working context. Before that context is released,
the activation transaction revalidates every selected ID against the current
lifecycle, temporal, visibility, and evidence-trust predicates, and requires the
exact selected memory version to remain current. This version proof also catches
partial evidence revocation that leaves the memory itself recallable; a stale
rendered selection is withheld. Citation then records what the agent says it
used, while proven cross-agent use requires that citation to match the activation
event for the same task lease.

## Highest-value next increments

1. Replace size-only greedy packing with the reviewed facility-location
   objective, while keeping the current answer-in-context and token metrics as
   the acceptance gate.
2. Extract referenced dates separately from observation and system time, then
   include validity ranges in activated blocks and add temporal query routing.
3. Add per-query graph gating and per-hop sufficiency checks so graph expansion
   helps multi-evidence work without adding decoys to simple lookups.
4. Add an asynchronous Observer/Reflector consolidation job that proposes
   append, supersede, or no-op actions without bypassing evidence and review.
5. Turn activation/citation traces into trajectory-derived relevance labels,
   then evaluate on LongMemEval-V2 and report quality per delivered token.
6. Expose iterative search/read-expand tools after the small bootstrap bundle;
   task-claim activation should remain the fast starting context, not the only
   evidence-gathering mechanism.

Those increments should be accepted only when they improve causal task outcomes
or answer-in-context under a fixed token budget. Raw recall count is retained as
deprecated compatibility telemetry, not as evidence of memory value.
