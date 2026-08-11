# CORAL shared memory — alignment audit 2026-08-09

[CORAL](https://github.com/Human-Agent-Society/CORAL) (MIT/NUS/Stanford;
[arXiv:2604.01658](https://arxiv.org/abs/2604.01658v1); accepted to COLM 2026)
is the closest published system to Swarm Brain's premise: *multiple long-running
coding agents that get better because they write to, and read from, a shared
persistent memory.* It is worth a careful read for three reasons — it
independently validates our threat model, it converged on a memory schema
strikingly close to ours, and its evaluation contract leaves the exact workload
Swarm Brain targets uncovered.

**Method and limits.** Read at `main` on 2026-08-09: `README.md`, `CLAUDE.md`,
`coral/hub/{notes,attempts,skills,_island}.py`, `coral/grader/{protocol,base}.py`,
`coral/workspace/worktree.py`, `coral/config.py`, and the bundled heartbeat
prompts. The paper was read at abstract level only. CORAL was **not installed or
run**, so every behavioural claim below is source-derived, and the two marked
*(inferred)* are inferences from the permission wiring rather than observed
behaviour.

## 1. The threat model, independently validated

CORAL's [2026-06-24 release note](https://github.com/Human-Agent-Society/CORAL#-news):

> The Docker session now isolates the agent from the grader: each agent runs as
> an unprivileged user (manager and grader stay root), so agents can no longer
> read `.coral/private/` (grader venv, answer keys) — **not even via Bash**. On
> the host this stays opt-in via `agents.isolate_user`.

The "not even via Bash" is the whole story, and the source shows why. Before
that release the only boundary was a Claude Code permission rule written per
worktree (`coral/workspace/worktree.py`):

```python
allow_rules = ["Bash", f"Read(/{worktree_pattern})", ...]
deny_rules  = ["Bash(git *)", f"Read(/{private_pattern})", ...]
```

`Read` on `.coral/private/**` is denied; `Bash` is allowed **unscoped**, and the
only Bash deny is `git *`. A `cat .coral/private/taskdata/answers.json` was
never covered by the deny rule. So a system whose agents are scored on a scalar,
who write to shared memory, and who have a shell, had exactly one thing standing
between them and the answer key: a tool-name-level allowlist that the shell
walked straight around. The fix was to move the boundary into the kernel —
uid separation, agents unprivileged, manager and grader root.

Two details sharpen rather than weaken the lesson:

- The boundary is deliberately **narrow**. The grader's *source* is surfaced
  read-only to agents at `<shared_dir>/grader/` on purpose — "so they can read
  how they're scored." Only `grader.private` data (answer keys, hidden test
  fixtures) is hidden, and `coral validate` errors if a `grader.private` path
  sits inside the visible `grader/` package, because it would be both copied to
  `.coral/private/` *and* leaked through the surfaced source. Transparency about
  the objective, secrecy about the labels.
- The host path is still **opt-in** (`agents.isolate_user: str = ""`, empty
  meaning no isolation), mutually exclusive with the newer `agents.sandbox`
  (srt, OS-level FS/network enforcement). Only the Docker session forces it.

**What this means for us.** Swarm Brain's poisoning guards, append-by-default
writes, explicit supersession, evidence trust, and content-free telemetry are
the *same threat model at a different layer*. CORAL's answer is process
isolation on one host; ours is that the state lives out-of-process in
CockroachDB behind a scoped, authenticated API, so there is no filesystem path
to walk around in the first place. That is a stronger structural position — but
it relocates the risk rather than removing it: our equivalent of "read the
answer key via Bash" is a token whose scope is broader than the agent's job.
The isolation lesson to import is not Docker; it is *the enforcement point must
not be the same layer the agent is free to operate in.*

## 2. CORAL's shared memory, concretely

Run layout (`CLAUDE.md`, `coral/hub/_island.py`):

```
results/<task>/<ts>/
  .coral/
    public/          symlinked into every worktree as .claude/ (or .codex/, ...)
      attempts/      one JSON per commit hash (pending -> final ScoreBundle)
      notes/         agent-written markdown + YAML frontmatter
      skills/        agent-built reusable tools (<name>/SKILL.md)
      agents/        subagent definitions (seeded: deep-researcher, librarian)
      roles/         <agent_id>.md — self-authored, public, everyone reads
      logs/, eval_logs/, heartbeat/, eval_count
    private/         grader venv + grader.private data — denied to agents
    .git/            checkpoint repo versioning .coral/public/
  agents/<agent_id>/ git worktree on branch coral/<agent_id>
```

Multi-island runs (v0.6.0) replace `public/` with `islands/<id>/`, each carrying
its own attempts, notes, skills, and heartbeat state, plus agent migration
between islands. `island_root()` returns `public/` when no `islands/` dir
exists, so the partition is a late, backwards-compatible addition.

The note schema (`coral/hub/notes.py`) is the interesting part — a *structured
trace* so "the framework — not just the reading agent — can filter, relate, and
verify" notes:

```yaml
creator: island-0-agent-2
created: 2026-03-14T17:35:00-00:00
type: experiment          # experiment | hypothesis | dead_end | open_question | synthesis
claim: "matmul inner-loop tiling at tile=32 improves score"
based_on: a3f9c2          # attempt this builds on (provenance)
evidence:
  attempt: 7b1e4d         # the graded artifact behind the claim
  score_delta: -0.03      # 0.42 -> 0.39
  verified: true
confidence: medium        # low | medium | high
status: confirmed         # confirmed | refuted | untested
supersedes: [research/old-idea.md]
touched: [matmul.cu]
```

Every field is optional at parse time (legacy data still loads), and a missing
`creator` becomes the loud sentinel `UNATTRIBUTED_CREATOR = "unknown"` rather
than being silently dropped from team aggregations — `notes_unattributed()`
exists to lint for it.

Consolidation is a **prompt on a timer**, not a service. `HeartbeatAction`s fire
on `interval` or `plateau` triggers; defaults are `reflect` every 1 eval,
`consolidate` every 10 (global), `pivot` after 5 plateau evals, `lint_wiki`
every 10. `consolidate.md` instructs the agent to distil 3+ related notes into
`notes/_synthesis/<topic>.md`, append cross-category patterns to
`_connections.md`, log contradictions in `_open-questions.md`, and audit the
team's `roles/` files for lane and posture coverage.

Durability is filesystem-shaped and thoughtfully done: attempt JSONs are written
tmp + `fsync` + `os.replace` so concurrent pollers see old-complete or
new-complete, never partial; `.coral/public/` is a git repo with lock-protected
checkpoint commits so agents can browse the history of shared state.

## 3. Convergence: CORAL ↔ Swarm Brain

Two teams, no shared code, near-identical ontology. This is the strongest
external evidence that the memory model in `domain/memory.py` is the right shape.

| CORAL | Swarm Brain | Note |
| --- | --- | --- |
| `type: experiment \| hypothesis \| dead_end \| open_question \| synthesis` | `MemoryKind: observation, invariant, hypothesis, decision, attempt, outcome, procedure, warning, handoff` (open via `SemanticLabel`) | Both refused a flat "memory"; both landed on typed epistemic status |
| `status: confirmed \| refuted \| untested` | `MemoryState: tentative, confirmed, refuted, superseded` | Near-identical lifecycle; we add `superseded` as a distinct terminal state |
| `confidence: low \| medium \| high` | `Memory.confidence` | Same axis |
| `supersedes: [...]`, `refutes: [...]` | `MemoryLinkKind.SUPERSEDES`, `.CONTRADICTS` | Same relations, ours typed and traversable |
| `based_on: <attempt hash>` | `MemoryLinkKind.DERIVED_FROM` | Both chose lineage over free-text citation |
| `evidence: {attempt, score_delta, verified}` | Immutable evidence with exact source offsets and trust state | Same slot; enforcement differs (§4) |
| `creator` + `UNATTRIBUTED_CREATOR` sentinel | Authenticated agent/run/swarm scope | Both treat unattributed memory as a defect, not a default |
| `consolidate` heartbeat → `_synthesis/` | Observer/Reflector consolidation, staged replay-safe plans | Same job: bounded synthesis, not unconditional append |
| `.coral/public/` git checkpoints | Append-only versioned, bitemporal memory | Both keep shared-state history browsable |
| `islands/<id>` partitions + migration | `Visibility: TASK \| RUN \| REPOSITORY` | **Not** the same thing — see §4 |
| `skills/<name>/SKILL.md` | `MemoryKind.PROCEDURE` | CORAL promotes validated technique to executable skill; we have the kind but no promotion path |
| `attempts/<hash>.json` (graded artifact) | `MemoryKind.ATTEMPT` / `.OUTCOME` | CORAL's is grader-written and authoritative; ours is agent-published |

## 4. Where the two designs actually diverge

**Governance is advisory in CORAL, enforced in Swarm Brain.** `status:
confirmed`, `confidence: high`, `evidence.verified: true` are YAML written by
the agent that benefits from them, parsed by `yaml.safe_load`, and enforced by
nothing. Our rule is the opposite and is the single sharpest contrast in this
document: provider and agent output *cannot choose* identity, scope, trust,
visibility, lifecycle, or source offsets. Writer-supplied governance is
untrusted input.

**There is no write authorization over shared memory** *(inferred)*. `public/`
is symlinked into every worktree and `Bash` is allowed unscoped with only
`git *` denied, so any agent can rewrite or delete any other agent's note,
role file, or attempt JSON. The `consolidate` prompt's "Do **not** edit other
agents' role files" is a norm, not a control. Falsifying an attempt JSON would
not change the real grade (grading runs in a detached worktree at the commit),
but it would change the leaderboard, `coral log`, and what every other agent
reads while reflecting. Untested, but it follows directly from the permission
wiring.

**Islands are an exploration partition, not an authorization boundary.** They
scope *what you see for diversity*; our `Visibility` and tenant/project/
repository/swarm/run/agent scope decide *what you are allowed to see*. Confusing
the two would be a mistake in either direction.

**CORAL has no retrieval layer at all.** Agents find memory with `ls`, `grep`,
and `Read`; `coral notes --search` is substring matching. No ranking, no
embedding, no token budget, no packing, no relevance floor, no activation
decision. This is the largest single gap and the whole subject of
`retrieval-architecture.md`. It is also why CORAL's approach works at its
current scale and would degrade: a flat notes directory is a fine index at 200
notes and a context-rot generator at 20,000.

**Nothing in CORAL proves a memory was used.** The paper claims gains "arise
from knowledge reuse," but the substrate has no primitive linking a note to the
attempt that consumed it. Swarm Brain's activation events, exact-version
citation matching, cross-agent proven use, and `observational_silver` outcome
associations are precisely that missing primitive. Conversely, CORAL has the
thing *we* lack — see §6.

**Durability and topology.** CORAL is one host, one filesystem, tmp+rename,
symlinks, a PID file for the grader daemon. Swarm Brain is transactional
CockroachDB with leases, fencing, and idempotent mutations across nodes. Neither
is wrong; they are answers to different blast radii.

## 5. The scalar-grader ceiling

`GraderInterface` is described in-repo as "the only interface contract in
CORAL":

```python
async def grade(self, codebase_path: str, tasks: list[Task], **kwargs) -> ScoreBundle: ...
```

You must supply the grader (`grader.entrypoint` is required since 2026-06-13;
legacy `eval/grader.py` auto-discovery was removed), and the loop is driven
end-to-end by its scalar: the leaderboard ranks it, `pivot` fires on its
plateau, `run.stop.max_real_attempts` counts against it. The example set is
exactly what that contract admits — circle packing, Erdős conjectures, VLIW and
GPU kernels, MNIST, two Kaggle competitions. Rubric judges (2026-04-24) extend
this to open-ended prose, but they still collapse to a score.

Two things fall outside:

1. **Work with no scalar objective.** Ordinary software maintenance — a
   dependency migration, a flaky-test hunt, a refactor that must preserve
   behaviour, an incident postmortem, a code review — has no `score_delta` to
   plateau on. You can write a grader for "tests pass," but that is a gate, not
   a gradient, and CORAL's whole control loop (reflect on what moved the score,
   pivot when it stops moving) is built on the gradient.
2. **Containerised deployment when the grader needs containers.** The README
   is explicit: built-in graders using Harbor (SWE-bench, terminal-bench) run
   evaluations inside Docker, DinD is unsupported, so **CORAL itself must not
   run inside Docker** in that configuration. The hardest isolation guarantee
   and the heaviest built-in graders are mutually exclusive.

Swarm Brain's trigger set was designed on the other side of this line:
`task_claim`, `checkpoint_resume`, `dependency_unblocked`, `tool_error`,
`repeated_failure`, `explicit`. Every one is a *task-lifecycle* event. None
requires a scalar. That is not an accident of implementation order — it is the
difference between a system for optimisation runs and a system for ongoing
work, and it is the gap CORAL's contract structurally cannot close.

## 6. What to do with this

Ranked by value, and separating what is genuinely available from what is
speculation.

1. **CORAL is a causal-intervention harness, and we have said in writing that
   we need one.** `outcome-feedback.md` states our outcome signal "remains
   observational and offline" and must not influence ranking "before a causal
   or paired evaluation." CORAL supplies exactly the missing apparatus: a fixed
   task, a fixed agent count, an independent scalar grader the agents cannot
   read the labels for, and a plateau-aware control loop. Ablate the memory
   substrate — CORAL's native `.coral/public/` notes versus Swarm Brain over
   MCP — hold everything else constant, and compare improvement rate per
   evaluation. That is a causal claim about memory quality, on someone else's
   grader, with the answer keys behind a uid boundary. It is the strongest
   experiment available to us that we cannot currently run in-house.
2. **The integration seam is already open on both sides.** CORAL's default
   runtime is `claude_code`, which loads MCP servers; we ship a nine-tool stdio
   MCP bridge (`claim_task`, `recall_memory`, `activate_memory`,
   `read_expand_memory`, `publish_memory`, `ingest_memory_source`,
   `checkpoint_task`, `complete_task`, `report_conflict`). Nothing structural
   blocks running a CORAL task where shared memory is Swarm Brain instead of a
   notes directory. Note that CORAL's plugin is deliberately "skills-first (no
   MCP)" for *driving* CORAL — that is about its own CLI surface, not a
   restriction on what MCP servers the agents may load. Worth confirming
   against `coral/agent/builtin/claude_code.py` before planning on it.
3. **Steal the skill-promotion path.** CORAL promotes a validated technique from
   note → `skills/<name>/SKILL.md`, an executable artifact the next agent runs
   rather than re-derives. We have `MemoryKind.PROCEDURE` and no promotion
   mechanism. Under our governance this is well-defined: promotion requires
   confirmed state, trusted evidence, and `DERIVED_FROM` lineage to the
   supporting memories — all primitives we already have.
4. **Steal `roles/` and the roster audit.** Self-authored, public, evidence-
   backed role files plus a periodic coverage audit ("an all-engineer team is a
   warning sign") is a cheap, legible answer to swarm specialisation. Our
   coordination layer has leases and dependencies but no notion of an agent's
   declared lane.
5. **Cite the convergence, do not claim the paper.** The schema overlap in §3 is
   real external validation of our ontology and belongs in the submission
   narrative. What does *not* belong is any implication that we reproduce or
   beat CORAL's numbers — different task family, different metric, no shared
   protocol. The 1363 → 1103 cycles kernel result is theirs, on their grader.

## 7. Open questions

- Does CORAL's Claude Code permission matching resolve symlinks before matching?
  The in-source comment about the grader symlink ("its target is outside the
  worktree/state root, so neither the worktree nor state-root Read rule covers
  it") implies resolved-path matching, which would mean tool-based writes to
  `public/` are *also* uncovered — and that agents write notes via Bash. This
  determines whether §4's write-authorization inference is a design choice or
  an oversight. Requires running it.
- How does `.coral/public/` behave at 10k+ notes? The consolidate prompt asks an
  agent to "browse `notes/`, especially anything new" — an instruction with no
  bounded cost. Their scale ceiling is our retrieval layer's justification, and
  measuring it would be a fair, citable comparison.
- Does the paper's mechanistic analysis of "knowledge reuse" establish causality,
  or is it correlational? Only the abstract was read; the full text should be
  checked before we cite it either way, because our own standard
  (`outcome-feedback.md`) would reject a correlational version.
- What exactly does `evidence.verified: true` mean operationally — is anything
  checked, or is it agent-asserted? Source suggests the latter; worth
  confirming, since it is the crux of §4.

## Sources

- [CORAL repository](https://github.com/Human-Agent-Society/CORAL) — `README.md`,
  `CLAUDE.md`, `coral/hub/`, `coral/workspace/worktree.py`, `coral/config.py`,
  `coral/grader/`, read at `main`, 2026-08-09.
- [CORAL: Towards Autonomous Multi-Agent Evolution for Open-Ended Discovery](https://arxiv.org/abs/2604.01658v1)
  — Qu et al., arXiv:2604.01658, COLM 2026. Abstract only.
- [CORAL documentation](https://coral.compounding-intelligence.ai/docs/) —
  concepts, eval loop, custom grader, multi-agent guides.
