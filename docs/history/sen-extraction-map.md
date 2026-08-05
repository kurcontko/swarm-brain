# Sen extraction map

Swarm Brain started as a self-contained subproject in the Sen repository. On
2026-08-05 its committed project, documentation, and task evidence were
extracted into this standalone repository.

History was rewritten only because paths moved:

- `swarm-brain/` became the repository root;
- `docs/swarm-brain/` became `docs/`;
- committed `current_tasks/swarm-brain-*.md` files became
  `docs/history/tasks/`.

Authors, timestamps, messages, file contents, and the original branch topology
were preserved. Git commit IDs necessarily changed because commit trees and
parents are part of each ID.

| Sen commit | Standalone commit | Meaning |
| --- | --- | --- |
| `373801c15766c2dff837e0023a4a66de25a553db` | `1ab671a656ff8eb9d8bf66dd276859fb29c8372b` | Swarm Brain P0 |
| `e2637f058c8940a533907fee93e675fe705ec28b` | `602fc8f79e458a9b0303abaee3a1fe764c0069b7` | CockroachDB P1 |
| `68aec1a3af534bca1793ce2385200a45e1441771` | `70ae78622660475ee4998c21f088ec5564cda85e` | Flexible memory contracts |
| `8b153ae17b5e8aaf52f0733b395855a065a238a2` | `5c91295716ce66e67e8a2213f1db0078c89ff52f` | Durable ingestion and vector recall |
| `6b0d12d8ad282ddabc43c043eef122924ba4a5f2` | `949db6406c93c70511af004e78bc377fd71fbbd1` | Divergent P2a prototype |

The current product line is `main`. The divergent P2a prototype remains
available as `archive/p2-vector-evidence`; it was not merged into `main`
because the later ingestion implementation reconciled the useful pieces under
different contracts.

The extraction intentionally excludes uncommitted Sen worktree changes and
unrelated Sen files. This commit adds only standalone-repository maintenance:
the license, corrected local paths, complete test dependencies, and ignore
rules.
