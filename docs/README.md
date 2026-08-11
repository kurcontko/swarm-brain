# Swarm Brain documentation

Start here:

- [Retrieval status](retrieval-status.md) — what is implemented, verified, and
  still open as of 2026-08-06.
- [Standalone retrieval architecture](retrieval-architecture.md) — target
  boundary, contracts, lanes, projections, migration phases, and evaluation.
- [SOTA retrieval research](research/sota-retrieval-postgresql-cockroachdb-2026-08-02.md)
  — PostgreSQL/CockroachDB techniques, trade-offs, and primary sources.
- [Reranking and global-selection protocol audit](research/sota-selection-reranker-protocol-2026-08-09.md)
  — primary-source Mnemis, SmartSearch, Chain-of-Memory, and MAGMA protocols,
  reproducibility gaps, and the frozen experiment order.
- [Memory representation and learned-policy frontier audit](research/sota-memory-representation-policy-audit-2026-08-09.md)
  — storage-form, retrieval-budget, and learned memory-operation frontier, and
  what each would have to prove before promotion.
- [CORAL shared-memory alignment audit](research/coral-shared-memory-2026-08-09.md)
  — the closest published multi-agent shared-memory system: its grader-isolation
  threat model, its note/attempt/skill schema against ours, and where its scalar
  grader stops.
- [Retrieval evaluation](retrieval-evaluation.md) — saved-run format, lane
  ablations, ANN exact-oracle checks, and release evidence.
- [SOTA acceptance gates](sota-acceptance.md) — frozen, machine-checkable
  end-to-end, causal, governance, and memory-to-action evidence requirements.
- [API contracts](api.md) — public HTTP/MCP/domain boundary.
- [Evidence-gated consolidation](consolidation-runtime.md) — asynchronous
  Observer/Reflector staging, immutable evidence, and governed lineage.
- [Observational outcome feedback](outcome-feedback.md) — strict
  activation/citation proof, silver semantics, and ranking isolation.
- [CockroachDB schema](cockroach-schema.md) — current v12 addendum plus the
  historical table inventory.
- [Restart demo](restart-demo.md) — local durability/restart acceptance flow.
- [Resilience demo](resilience-demo.md) — three-node cluster, one node killed
  mid-run, every beat green on the surviving quorum.
- [Extraction history](history/sen-extraction-map.md) — original Sen commits
  mapped to standalone repository history.
- [Submission package](submission/architecture.md) — hackathon architecture
  overview, Devpost draft, video script, and CockroachDB tool feedback.
- [Deployment](deploy.md) — container image, AWS templates, and the key-gated
  activation scripts; nothing applied without operator approval.
- [Security review 2026-08-07](security-review-20260807.md) — pre-publication
  sweep: console XSS, secrets and history, public API abuse posture, licensing,
  and the pre-push and deploy-week checklists.

Implementation context:

- [Architecture](architecture.md)
- [Implementation plan](implementation-plan.md)
- [Source map](source-map.md)
- [Historical issue 1](issue-1.md)
