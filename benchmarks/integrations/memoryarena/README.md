# MemoryArena integration

This package implements the pinned official memory HTTP seam over an isolated
canonical Swarm Brain runtime. See `docs/memoryarena-benchmark.md` for the
protocol boundary, preflight contract, and current upstream blockers.

It provides an explicitly nonpublishable deterministic validation mode and a
fail-closed OpenAI-compatible semantic embedding mode with provider-observed
call evidence. Semantic evidence also remains nonpublishable because the
standard response attests a model alias, not the immutable served-weights
revision. It does not execute or score the paper benchmark. The public upstream
commit is preview code and does not yet expose enough immutable material for a
strict 766-task SR/PS result compiler.
