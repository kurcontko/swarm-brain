# Swarm Brain documentation

Start here:

- [Retrieval status](retrieval-status.md) — what is implemented, verified, and
  still open as of 2026-08-06.
- [Standalone retrieval architecture](retrieval-architecture.md) — target
  boundary, contracts, lanes, projections, migration phases, and evaluation.
- [SOTA retrieval research](research/sota-retrieval-postgresql-cockroachdb-2026-08-02.md)
  — PostgreSQL/CockroachDB techniques, trade-offs, and primary sources.
- [Retrieval evaluation](retrieval-evaluation.md) — saved-run format, lane
  ablations, ANN exact-oracle checks, and release evidence.
- [API contracts](api.md) — public HTTP/MCP/domain boundary.
- [CockroachDB schema](cockroach-schema.md) — current v8 addendum plus the
  historical table inventory.
- [Restart demo](restart-demo.md) — local durability/restart acceptance flow.
- [Extraction history](history/sen-extraction-map.md) — original Sen commits
  mapped to standalone repository history.
- [Submission package](submission/architecture.md) — hackathon architecture
  overview, Devpost draft, video script, and CockroachDB tool feedback.
- [Deployment](deploy.md) — container image, AWS templates, and the key-gated
  activation scripts; nothing applied without operator approval.

Implementation context:

- [Architecture](architecture.md)
- [Implementation plan](implementation-plan.md)
- [Source map](source-map.md)
- [Historical issue 1](issue-1.md)
