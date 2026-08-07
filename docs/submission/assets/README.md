# Submission visual assets

Static, self-contained SVGs for the hackathon demo video and the Devpost page.
Every asset is 1920 × 1080 with a `#0d1117` background so a cutaway matches the
read-only console's dark theme, and every number on them was re-verified against
the repository documents on 2026-08-07.

## Shot map

Shot ids are the headings in [`video-script.md`](../video-script.md).

| Asset | Shot | Window | Recommended on screen | Role in the shot |
| --- | --- | --- | --- | --- |
| `architecture-topology.svg` | **D** — architecture / tools / disclosure | 2:22–2:38 | 8–10 s (centre, dominant) | The centre of the segment-D slide. Replaces the "rendered and simplified" mermaid diagram the script calls for. |
| `tool-checklist.svg` | **D** — architecture / tools / disclosure | 2:22–2:38 | 4–5 s (or left column, held) | The tool checklist. Composite it as the left column of the D slide, or cut to it full-frame for 4–5 s inside the 16 s window. |
| `disclosure-card.svg` | **D** — architecture / tools / disclosure | 2:22–2:38 | ≥ 7 s, full frame | The disclosure sentence. Script rule 3: it is spoken once, verbatim, and must be on screen long enough to read. 7 s is the floor for 78 words at a comfortable reading rate. |
| `memory-dataflow.svg` | **C1** — ANN + `EXPLAIN` | 1:46–2:00 | 4–5 s, optional cutaway | Optional. Shows where the dense lane sits among the other lanes while the narration says "its ranks fuse with exact, full-text, trigram, and graph lanes". Cut it before cutting anything else — C1's required content is the live `EXPLAIN`, not this diagram. |
| `benchmark-card.svg` | *no shot* | — | 6–8 s if a shot is ever added | Devpost page and thumbnail only. There is no retrieval-benchmark shot in the current 2:44 script and none was invented. |
| `ab-card.svg` | *no shot* | — | 6–8 s if a shot is ever added | Devpost page ("Real-World Impact", [hackathon-plan §4](../../hackathon-plan.md)) only. The A/B run is a separate demo mode (`--ab`) and no shot in the current script covers it. |

Segments A, B1–B6, C2, C3 and E are live capture or plain typographic cards;
they get no asset.

### If the video gains an A/B or benchmark beat

Both cards are built to stand alone full-frame for 6–8 seconds. Adding either
costs time the script does not have at 2:44, so a shot would have to be cut
first — see the cut order in the script's time-budget table.

## Where each number comes from

Nothing on these assets may be edited without re-checking its source.

| Asset | Source of record |
| --- | --- |
| `architecture-topology.svg` | [`architecture.md`](../architecture.md) §1 and the "what runs where today" table |
| `memory-dataflow.svg` | [`architecture.md`](../architecture.md) §2 |
| `tool-checklist.svg` | [`devpost-draft.md`](../devpost-draft.md) — "CockroachDB tools used", "AWS services used", and the integrity guard |
| `benchmark-card.svg` | [`docs/retrieval-benchmark.md`](../../retrieval-benchmark.md) — track 2 results table, track 1 headline table, environment and corpus tables |
| `ab-card.svg` | `uv run --extra serve swarmbrain-demo --ab`, and the `*-swarm-ab.json` evidence artifact it writes |
| `disclosure-card.svg` | [`devpost-draft.md`](../devpost-draft.md) — "Pre-existing code disclosure", quoted verbatim |

Two values are deliberately placeholders and must be resolved before the final
take: `<PUBLIC_REPO_URL>` on the disclosure card, and the `185 passing tests`
count the script's segment D calls for (not drawn on any asset here, because it
has to be re-run immediately before the take).

## Regeneration

**These SVGs are hand-authored. There is no generator script — edit the files
directly.** They are plain XML with an inline `<style>` block; open one in an
editor and change the text or the coordinates.

House rules, so an edited asset still matches the others:

- Palette: background `#0d1117`, panel `#161b22`, border `#30363d`, primary text
  `#e6edf3`, secondary `#a9b4c0`, muted `#8b949e` / `#6e7681`. Accents: green
  `#3fb950` (proven / measured), amber `#d29922` (caveat, write path), blue
  `#58a6ff` (read path), purple `#bc8cff` (vector plane), orange `#f0883e`
  (the uncoordinated baseline arm).
- Type: system sans stack only (`-apple-system, BlinkMacSystemFont, "Segoe UI",
  Roboto, Helvetica, Arial, sans-serif`) and the system mono stack for
  identifiers. **No web fonts, no external references of any kind** — these must
  render identically offline and inside a video editor.
- Minimum body size is 23 px in the 1920 × 1080 coordinate space, and 25–28 px
  for anything a judge is expected to read at speed. Headlines are 46–54 px.
- A tick mark, a solid border or a green fill means *proven*. Dashed borders and
  grey text mean *not yet proven*. Never restyle a planned item as a used one
  without changing `devpost-draft.md` first.

### Checks before committing an edit

```bash
# well-formed XML
for f in docs/submission/assets/*.svg; do
  python3 -c "import sys,xml.etree.ElementTree as ET; ET.parse(sys.argv[1])" "$f" || echo "FAIL $f"
done

# no external references (should print nothing but the svg namespace)
grep -n 'href\|@import\|src=\|https\?://' docs/submission/assets/*.svg

# markdown links still resolve
python3 scripts/check_markdown_links.py
```

To eyeball one at 1920 × 1080, open it in a browser at 100 % zoom, or render it:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless \
  --screenshot=/tmp/out.png --window-size=1920,1080 \
  "file://$PWD/docs/submission/assets/architecture-topology.svg"
```
