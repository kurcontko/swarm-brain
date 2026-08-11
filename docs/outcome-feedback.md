# Observational memory/outcome feedback

Swarm Brain records a content-free outcome association only when a task
completion cites a memory that was also delivered by a durable activation for
the exact same task, lease, and consuming agent. The association is explicitly
`observational_silver`: it is useful offline evidence, not a causal label and
not proof that the memory caused success or failure.

This closes the first feedback-loop seam suggested by the experience-following
error-propagation analysis in
[Xiong et al. (ACL 2026)](https://aclanthology.org/2026.acl-long.27/) without
letting a noisy outcome signal change production behavior.

## Commit-time proof

Task completion remains authoritative. Inside the same completion transaction,
each backend:

1. deduplicates and bounds completion citations to the first 100 IDs;
2. reads canonical `memory.activated` telemetry from the authenticated
   tenant/project/repository/swarm/run scope;
3. keeps only activations with the same task, lease, and consumer;
4. requires exactly one unambiguous activated version for a cited memory;
5. revalidates that exact version as current, confirmed, visible, temporally
   valid, and supported by non-rejected, non-untrusted evidence; and
6. inserts one deterministic association carrying only scope IDs, memory
   version, `succeeded|failed`, kind, and timestamp.

Recall alone, activation without a completion citation, and a citation without
activation are not treated as use. A previous lease or another agent's
activation cannot prove the current completion. If memory changed, expired,
was superseded/refuted, or lost trusted evidence after activation, the
association is omitted. Invalid or ambiguous proof never prevents task
completion.

The in-memory and CockroachDB adapters expose a bounded, read-only
`list_memory_outcome_associations` store method for offline evaluation. Reads
are constrained to the actor's exact run scope and may be narrowed by task or
memory ID. CockroachDB schema v11 persists the same contract in
`memory_outcome_associations`.

## Deliberate non-effects

The runtime does not consume these observations in retrieval, packing,
activation, consolidation, trust, supersession, or conflict resolution. In
particular, a failed task does not refute or down-rank a cited memory: failures
can arise from the rest of the task, the environment, or incorrect execution.
Any future learner must establish incremental value with a causal or paired
evaluation and remain behind the existing governance boundary before this
silver signal can influence ranking.
