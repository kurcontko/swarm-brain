# Submission visual assets

Static, self-contained SVGs for the hackathon demo video and the Devpost page.
Every asset is 1920 × 1080 with a `#0d1117` background so a cutaway matches the
read-only console's dark theme, and every number on them was re-verified against
the repository documents on 2026-08-17.

## Shot map

Shot ids are the headings in the demo video script.

| Asset | Shot | Window | Recommended on screen | Role in the shot |
| --- | --- | --- | --- | --- |
| `architecture-topology.svg` | **D** — one database / tools / disclosure | 2:12–2:28 base | 5 s, full frame | Shows the one-database topology. Add the script's lower third naming claims, leases, JSON memory, vectors, lineage, and telemetry. |
| `tool-checklist.svg` | **D** — one database / tools / disclosure | 2:12–2:28 base | 4 s, full frame | Proves all four CockroachDB tools and the exercised AWS services; the narration need not read every line. |
| `disclosure-card.svg` | **D** — one database / tools / disclosure | 2:12–2:28 base | 7 s, full frame | Full disclosure remains visible while the concise demo disclosure is spoken. The full wording also remains in Devpost and README. |
| `memory-dataflow.svg` | **C1** — ANN + `EXPLAIN` | 1:46–2:00 | 4–5 s, optional cutaway | Optional. Shows where the dense lane sits among the other lanes while the narration says "its ranks fuse with exact, full-text, trigram, and graph lanes". Cut it before cutting anything else — C1's required content is the live `EXPLAIN`, not this diagram. |
| `benchmark-card.svg` | *no shot* | — | 6–8 s if a shot is ever added | Devpost page and thumbnail only. There is no retrieval-benchmark shot in the current 2:44 script and none was invented. |
| `impact-card.svg` | **E** — measured impact | 2:28–2:40 | 12 s, full frame | Video-safe summary of the disclosed A/B: 103 modeled baseline steps versus 25 measured live-run steps, with the claim boundary visible. |
| `ab-card.svg` | *no shot* | — | 6–8 s if a shot is ever added | Detailed Devpost-page A/B card. It carries more provenance than can be read in the video window; segment E uses `impact-card.svg` instead. |
| `close-card.svg` | **F** — close | 2:40–2:46 | 6 s, full frame | Product name, one-database thesis, the `/console` demo URL, and the repository + MIT license. |

Segments A, B0, B1–B6, C2 and C3 are live capture or plain typographic cards;
they get no asset.

### Why there are two A/B cards

`impact-card.svg` is the deliberately sparse, ten-second video card.
`ab-card.svg` is the detailed Devpost artifact with the full decomposition and
method caveat; it is too dense for the video window. Adding the separate
retrieval benchmark card would cost time the 2:44 base cut does not have.

## Where each number comes from

Nothing on these assets may be edited without re-checking its source.

| Asset | Source of record |
| --- | --- |
| `architecture-topology.svg` | [`architecture.md`](../architecture.md) §1 and the "what runs where today" table |
| `memory-dataflow.svg` | [`architecture.md`](../architecture.md) §2 |
| `tool-checklist.svg` | The Devpost submission text — "CockroachDB tools used", "AWS services used", and the integrity guard |
| `benchmark-card.svg` | [`docs/retrieval-benchmark.md`](../../retrieval-benchmark.md) — track 2 results table, track 1 headline table, environment and corpus tables |
| `impact-card.svg` | `evidence/20260817T124654Z-swarm-ab.json`; CockroachDB-backed, provenance-stamped, video-safe subset of the detailed A/B card |
| `ab-card.svg` | `SWARMBRAIN_BACKEND=cockroach uv run --extra serve --extra crdb swarmbrain-demo --ab`, and the provenance-stamped `*-swarm-ab.json` artifact it writes |
| `disclosure-card.svg` | The Devpost submission text — "Pre-existing code disclosure", quoted verbatim |

The 2026-08-17 preflight verified both the offline and CockroachDB-backed test
counts, but they are intentionally absent from the video assets. The repository
URL on the disclosure card is resolved to
`github.com/kurcontko/swarm-brain`.

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
  without changing the Devpost submission text first.

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
