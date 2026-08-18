# Sen extraction map

Swarm Brain started as a self-contained subproject in the Sen repository. On
2026-08-05 its committed project, documentation, and task evidence were
extracted into this standalone repository.

History was rewritten only because paths moved:

- `swarm-brain/` became the repository root;
- `docs/swarm-brain/` became `docs/`;
- committed `current_tasks/swarm-brain-*.md` files became per-task history
  documents, kept outside the published tree.

Authors, timestamps, messages, file contents, and the original branch topology
were preserved. Git commit IDs necessarily changed because commit trees and
parents are part of each ID.

History was rewritten a second time on 2026-08-18, before this repository was
made public. That pass removed the internal working documents — strategy and
planning notes, the pre-publication security review, and the submission drafts
— from every commit, and redacted the CockroachDB Cloud cluster identifiers
from the committed evidence artifacts. Nothing else changed: no code, test,
benchmark, or evidence result was altered, and the commit sequence, authorship,
and timestamps are intact. Commit IDs in the table below are the post-rewrite
ones, and commit IDs quoted in this repository's other documents are too.

| Sen commit | Standalone commit | Meaning |
| --- | --- | --- |
| `373801c15766c2dff837e0023a4a66de25a553db` | `be09ac98c3f5693950cbd2ecd0591eddad3150f1` | Swarm Brain P0 |
| `e2637f058c8940a533907fee93e675fe705ec28b` | `2a1cc20aa1f236c7f36e0948b9d0025fb8f4181d` | CockroachDB P1 |
| `68aec1a3af534bca1793ce2385200a45e1441771` | `cb9c2ca00a489e15192180c638c90ab54a2cfcca` | Flexible memory contracts |
| `8b153ae17b5e8aaf52f0733b395855a065a238a2` | `f42cbaab402aea623695cdb600e8bffce1f216a3` | Durable ingestion and vector recall |
| `6b0d12d8ad282ddabc43c043eef122924ba4a5f2` | `856c9051b7fda43b48a9d5174abc90ec1341a467` | Divergent P2a prototype |

The current product line is `main`. The divergent P2a prototype remains
available as `archive/p2-vector-evidence`; it was not merged into `main`
because the later ingestion implementation reconciled the useful pieces under
different contracts.

The extraction intentionally excludes uncommitted Sen worktree changes and
unrelated Sen files. This commit adds only standalone-repository maintenance:
the license, corrected local paths, complete test dependencies, and ignore
rules.
