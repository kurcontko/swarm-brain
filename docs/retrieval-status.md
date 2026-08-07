# Retrieval v2 — aktualny stan

Stan na: 2026-08-06
Repozytorium: standalone `swarm-brain`

## Wynik

Swarm Brain ma działający, provider-neutral hybrid retrieval v2 dla pamięci:

```text
RecallQuery + ActorContext
          ↓
server-owned purpose + planner
          ↓
exact ─ FTS simple ─ trigram ─ dense current
          ↓
direct weighted RRF (k=60)
          ↓
bounded memory-link graph (1–2 hops)
          ↓
final weighted RRF (k=60)
          ↓
private canonical hydration
          ↓
RecallBundle albo abstention
```

Sen nie jest zależnością runtime, współwłaścicielem schematu ani mutation
authority. Może później zostać opcjonalnym zewnętrznym candidate providerem za
neutralnym `RetrievalGateway`, ale dopiero po benchmarku pokazującym lukę w
lokalnym retrievalu.

## Zaimplementowane

### Kontrakty i orkiestracja

- `RetrievalPurpose`, `RetrievalPlan`, `Candidate`, `CandidateBatch`,
  `RetrievalTrace` i contribution-level weighted RRF;
- osobne porty `RetrievalGateway`, `GraphExpansionGateway` i
  `CanonicalMemoryReader`;
- pełny trace pozostaje wewnętrzny i nie zmienia publicznego `RecallBundle`;
- stabilna normalizacja publicznego score do zakresu `[0,1]`;
- exact/lexical/fuzzy/dense/graph collapse po canonical memory ID;
- awaria pojedynczego lane'u jest oznaczona jako degraded, a nie ukryta.

Query embedding powstaje przed snapshotem bazy. Udany dense lookup jest
`CandidateBatch` z raw cosine score, rankiem, projection signature i wkładem do
weighted RRF; awaria providera albo mismatch wymiaru jest widoczna w trace jako
degraded dense lane. Stary semantic max-score merge pozostaje wyłącznie
compatibility fallbackiem dla ręcznie złożonego `RetrievalService` bez gatewaya
dense; runtime produkcyjny i lokalny składają v2 lane.

### Correctness i bezpieczeństwo

- brak zero-score `scope_match`; niezwiązane zapytanie zwraca `hits=[]`;
- finalna hydracja powtarza tenant/project/repository, visibility, run/task,
  state/currentness, kind, valid-time, recorded-time oraz source trust/review;
- filtry są również częścią każdego CockroachDB candidate query przed `LIMIT`;
- prywatne candidate ID nie są przenoszone przez `RecallQuery.model_copy()`;
- publiczne `memory_ids` pozostaje tylko jako oznaczony deprecated selector
  kompatybilności HTTP v1;
- mixed-trust memory pozostaje recallable, gdy ma co najmniej jedno dobre
  evidence; odrzucone/untrusted cytowania nie trafiają do wyniku także przy
  audytowym `include_refuted=true`;
- repository-visible memory zachowuje zaakceptowane cytowania także przy recallu
  z innego runu w tym samym tenant/project/repository;
- candidate lanes i canonical hydration korzystają z jednego snapshotu
  adaptera; CockroachDB obejmuje nim także ANN lookup, współdzieli jedną
  transakcję i zegar `now()`, serializuje pracę na connection i izoluje każdy
  lane savepointem, a adapter lokalny blokuje canonical publication do końca
  recallu;
- explicit historical queries nie używają current-only ANN.

### Purpose-specific task bootstrap

Claim wybiera po stronie serwera `task_bootstrap`, a query uwzględnia:

- title i description;
- tags i required capabilities;
- latest checkpoint summary;
- discoveries i remaining work;
- checkpoint memory IDs jako prywatne exact seeds.

Publiczny v1 nadal zwraca płaski `RecallBundle`; sekcje są oznaczane reasons
(`section:handoff`, `section:playbook`, `section:prior_attempts`,
`section:knowledge`). Strukturalne sekcje wymagają wersjonowanego response v2.

### CockroachDB schema v8

`retrieval_documents` przechowuje:

- `projection_id=memory-simple-v1`;
- repository/run/task `scope_key`;
- deterministic `search_text`;
- bounded `lookup_text`;
- stored `TSVECTOR` z konfiguracją `simple`;
- scope-prefixed FTS i `gin_trgm_ops` indexes.

`retrieval_exact_terms` przechowuje znormalizowane:

- memory ID i content digest;
- title i tags;
- paths, symbols, test names, commands, commits, error codes;
- jawne aliasy z retrieval metadata.

Nowy publish/merge aktualizuje canonical memory i obie projekcje w jednej
transakcji. Installer najpierw nakłada minimalny SQL overlay, a następnie
przebudowuje rekordy sprzed v7 tym samym ograniczonym projektorem aplikacyjnym
NFKC/casefold co nowe zapisy. Rebuild działa jako jeden SERIALIZABLE snapshot z
bounded retry i końcowym sweepem starych scope/version rows, więc udany install
nie publikuje częściowo przebudowanej projekcji. Search projection nie jest
granicą auth: lane query robi canonical lookup join, a final hydration sprawdza
wynik ponownie.

FTS używa bezpiecznie generowanego OR-`TSQUERY` z tokenów aplikacyjnego lexera.
Nie przekazuje surowej składni użytkownika do `to_tsquery`; dzięki temu zachowuje
partial lexical recall (`database SQL port` → `CockroachDB SQL port`) i nadal
korzysta z inverted index. Trigram przypina
`pg_trgm.similarity_threshold` do wspólnej wartości adapterów w każdej
transakcji odczytu, używa indeksowalnego operatora `%`, a `similarity()`
wyłącznie do rankingu. Query trafiające do fuzzy SQL jest najpierw ograniczone,
znormalizowane NFKC/casefold i whitespace-collapsed tak samo jak projekcja.

Verifier sprawdza metodę krytycznych indeksów, opclass, kolejność scope prefix,
primary key exact terms, computed `TSVECTOR` oraz kształt projection primary
key — nie tylko nazwy obiektów.

`retrieval_vectors_1024` jest nową projekcją dense, oddzieloną od legacy
`memory_vector_embeddings`. Każdy row zapisuje:

- canonical resource ID i wersję;
- content SHA-256;
- `scope_key` repository/run/task i `domain_lane`;
- input renderer, current/history mode, cosine metric, provider
  normalization/truncation policy, model i dimensions w
  `projection_signature`;
- fixed-width `VECTOR(1024)` oraz `indexed_at`.

Vector index ma pełny prefix `(tenant, project, repository, resource_type,
projection_id, signature, scope_key)`. Gateway uruchamia osobną equality-bound
gałąź ANN dla każdego dozwolonego scope. Ponieważ Cockroach przyspiesza tylko
filtry z pełnego prefixu, state/currentness, kind, bitemporal, version/digest i
source trust/review są walidowane na canonical rows w tym samym snapshotcie po
ANN. Gdy filtracja nie wypełnia lane budget, okno ANN rośnie geometrycznie do
twardego limitu. Final hydration powtarza wszystkie predykaty. Current-only
dense nie jest uruchamiany dla explicit historical, refuted ani superseded
query.

Fenced worker zapisuje legacy vector, projekcję v2 i completion w jednej
transakcji; utracony lease nie może opublikować żadnego z nich. Installer kopiuje
zgodne istniejące wektory do jawnie podpisanej projekcji v2. Schema verifier
sprawdza wymagane kolumny, primary-key shape, kolejność pełnego vector prefixu i
`vector_cosine_ops`.

Cockroach dense gateway udostępnia ANN przez
`retrieval_vectors_1024_ann_v2` z adaptive canonical validation oraz exact
oracle wymuszony przez `@primary`, który filtruje eligible set przed exact
sortem. `vector_search_beam_size` i raw cosine floor są operator-owned settings.
Wspólny evaluator liczy ANN Recall@k, Recall@k, MRR, nDCG oraz no-answer
precision/recall dla zapisanych lane ablations.

Raw cosine floor domyślnie wynosi `0.0`: nie istnieje jeden poprawny cutoff dla
wszystkich modeli, a niezmierzony `0.2` powodował regresję na istniejącym live
gate z długim query. Zero/ujemne wkłady i tak są odrzucane przez fusion; dodatni
floor wymaga pomiaru na docelowym modelu i korpusie.

### Bounded graph lane

`memory_links` jest teraz drugim etapem retrievalu, a nie równoległym,
niezaseedowanym skanem. Najpierw exact/FTS/trigram/dense tworzą direct RRF;
najsilniejsze wyniki i jawne prywatne seedy uruchamiają graph expansion, po czym
wszystkie lane'y przechodzą finalny RRF.

Plan zapisuje pełną politykę ograniczeń:

- maksymalnie 16 seedów;
- jeden hop dla interactive/historical/orientation i dwa dla
  bootstrap/handoff/planning/conflict review;
- osiem eligible sąsiadów na node;
- osobny candidate budget i całkowity edge-examination budget;
- allowlistę siedmiu wbudowanych relacji;
- fail-safe degraded lane przy błędzie.

Ranking graph stosuje typed-relation weight, karę dla odwrotnego przejścia po
relacji asymetrycznej, decay `0.85` na krok i bounded query-text gate z floorem
`0.60`. To score wewnątrz lane'u; contribution do wspólnego rankingu nadal
jest audytowalnym weighted RRF. Najlepsza deterministyczna ścieżka przechowuje
każdy memory ID, edge ID, kierunek i relation type w `Candidate.path`.

Adapter CockroachDB wykonuje stałą liczbę iteracji nad covering indexes
`memory_links_from_type` i `memory_links_to_type`. Bounded `LATERAL` scan może
pobrać najwyżej czterokrotność fan-out na frontier node; dopiero canonical
validation nadaje sloty fan-out, więc hidden/stale/refuted/untrusted endpoint
nie wypiera poprawnego sąsiada. Target musi przejść tenant/project/repository,
visibility, run/task, state, kind, valid/recorded time i trust/review przed
wejściem do następnego frontieru, a final hydration sprawdza wszystko ponownie.
Edge `created_at` jest ograniczony przez `recorded_at` albo snapshot `now()`.
Całość działa w tym samym retrieval snapshotcie co direct lanes.

Nie używamy recursive lineage CTE na critical path. PostgreSQL daje path arrays
i `CYCLE`, ale kolejność wykonania nie jest kontraktem; CockroachDB wymaga
jawnego warunku końca i ostrzega, że część recursive CTE nie jest jeszcze
optymalizowana. Fixed-depth application traversal daje jawny hop/fan-out/edge
budget i łatwy `EXPLAIN` obu kierunków. Duże PPR/community graph pozostaje
asynchroniczną projekcją przyszłości.

### Trwały licznik reuse (schema v9)

CockroachDB ma teraz audytowalny licznik reuse zapisywany poza read-only
snapshotem recallu. Schema v9 dodaje `retrieval_reuse_counters` z kluczem
głównym `(tenant_id, run_id)`, kolumnami `reuse_count` i `recall_count`,
właścicielskim scope'em project/repository/swarm oraz czasem pierwszego i
ostatniego zapisu.

Po zamknięciu snapshotu adapter wykonuje jedną krótką transakcję z pojedynczym
`INSERT ... ON CONFLICT (tenant_id, run_id) DO UPDATE`, który dodaje liczbę
różnych publicznych hitów do `reuse_count` i jeden recall do `recall_count`.
Gałąź konfliktu aktualizuje wiersz tylko wtedy, gdy zapisany
project/repository/swarm zgadza się z uwierzytelnionym scope'em, a klucz obcy do
`runs` kasuje licznik razem z runem. Zapisywane są wyłącznie liczby: ani tekst
zapytania, ani treść pamięci nie trafiają do tej tabeli.

Zapis jest fire-and-forget: recall pozostaje read-only i nieblokujący, a błąd
licznika jest logowany i pomijany, nigdy nie wraca do klienta. `get_run_metrics`
czyta `reuse_count` jednym lookupem po kluczu głównym wewnątrz zapytania już
ograniczonego do runu, więc `memories_reused` ma teraz parytet między adapterem
lokalnym a CockroachDB. Ta sama ścieżka obejmuje bootstrap po zatwierdzonym
claimie, bo przechodzi przez `MemoryService.recall`. Verifier sprawdza kształt
klucza `(tenant_id, run_id)` i kolumny licznika, a `schema install` pozostaje
additive i idempotentny.

## Publiczna kompatybilność

Bez zmian pozostają:

- `POST /v1/memories:recall` i `RecallBundle`;
- strict `RecallQuery` field set;
- siedem MCP tools i dotychczasowy MCP `recall_memory` input;
- capability checks oraz server-derived identity/scope;
- `ClaimTaskResult.memory | null` i fail-open enrichment po zatwierdzonym
  claimie.

Nie dodano do publicznego request/response:

- purpose;
- lane/provider selection;
- budgets lub weights;
- pełnego trace;
- tenant/repository/agent identity.

## Weryfikacja

- pełna macierz na świeżym CockroachDB 26.2.1: `204 passed`;
- fresh schema v9 install i verify na izolowanym CockroachDB v26.2;
- istniejący live memory gate po transactional projection write;
- live exact, FTS, trigram, RRF, abstention i `EXPLAIN` dokładnego SQL
  JOIN/filter/ranking emitowanego przez runtime;
- gold corpus z rekordem starszym niż 2005 nowszych decoyów;
- in-memory/Cockroach canonical-filter tests;
- weighted-RRF contribution/trace test;
- live 42703 regression: błędny SQL lane cofa się do savepointu i nie zatruwa
  poprawnego lane'u ani hydration stanem 25P02;
- public contract snapshot;
- task-bootstrap server-purpose test;
- mixed-trust trust-before-`LIMIT`, audit sanitization i cross-run parity na
  in-memory oraz żywym CockroachDB;
- live parity referencyjnego trigram score z `similarity()` CockroachDB;
- regresja transakcyjnego migracyjnego rebuild i stale-row sweep dla
  znormalizowanego title, tag, path, alias i command;
- focused matrix path/symbol/test/command/hash/error/alias.
- dense v2 unit gate: pełny equality prefix ANN, same-snapshot canonical
  validation, bounded adaptive widening, signature/version/digest propagation,
  ANN/exact routing, RRF contribution, provider degradation i publiczny v1
  compatibility snapshot;
- schema v8 static gate dla oddzielnej signed projection, vector opclass i
  prefix order;
- deterministyczny evaluator smoke dla lane ablations i ANN exact-oracle
  overlap. Fixture smoke nie jest reprezentatywnym benchmarkiem jakości.
- graph unit gate: staged direct→graph→final RRF, one/two-hop policy, cycle
  prevention, relation allowlist, query/relation/hop decay, historical edge
  cutoff, canonical-before-fan-out, hard fan-out/edge budget, path provenance i
  degraded-lane fallback;
- live CockroachDB graph gate: dwa hop'y przez rzeczywiste `memory_links`,
  oba directional covering indexes, brak full scan w runtime `EXPLAIN`,
  canonical validation i schema index `EXPLAIN`;
- reuse unit gate: scoped, zdeduplikowany i wolny od treści UPSERT, brak zapisu
  bez hitów oraz przełknięty i zalogowany błąd licznika;
- live CockroachDB reuse gate: inkrementacja licznika i `memories_reused` w
  metrykach runu, abstention bez wiersza licznika, awaria zapisu nieprzerywająca
  recallu oraz izolacja cross-run i cross-tenant.

Pełne końcowe wyniki komend są raportowane w handoffie gałęzi; ten dokument nie
jest automatycznym substytutem CI evidence.

## Co pozostaje

### Phase 2 — dług ewaluacyjny (mierzony baseline: zrobiony)

Mierzony baseline istnieje. Reprezentatywny korpus `swarm-coding-2026-08-07`
(90 pamięci, 40 zapytań, judgments `r1`, w tym sześć jawnych no-answer)
przeszedł prawdziwą ścieżką retrievalu na adapterze lokalnym i na żywym
CockroachDB 26.2.1 z pełnym ablation exact vs FTS vs trigram vs dense vs
direct-fused vs final-fused, percentylami latency oraz ANN Recall@k względem
exact oracle. Ten sam protokół policzył też oficjalny zewnętrzny zbiór
LongMemEval-S (500 pytań, retrieval-only). Liczby, wersje, seedy i
zastrzeżenia: [retrieval benchmark](retrieval-benchmark.md).

Pomiar odsłonił dwa otwarte problemy, których zielone testy correctness nie
pokazywały: final fusion nie abstainuje na zapytaniu spoza korpusu (no-answer
recall `0.00`, bo publiczny score jest zakotwiczony w rank najlepszego lane'u,
a nie w trafności), a graph lane psuje MRR na decoy-heavy zapytaniach, mimo że
pomaga na multi-evidence. Dense liczby pochodzą z deterministycznego hash
embeddera, więc są dowodem instalacji, nie jakości semantycznej. Check-in gold
set nadal jest deterministyczną regresją correctness, a nie benchmarkiem
jakości.

### P3 — dług ewaluacyjny dense

Implementacja dense v2 i exact oracle są domknięte, ale mały deterministic/live
gate nie dowodzi jakości ani skali. Phase 3 nie jest benchmark-complete, dopóki
nie powstanie reprezentatywny corpus coding-agent memory z wynikami:

- dense-only, lexical-only i hybrid Recall@k/MRR/nDCG/no-answer;
- filtered ANN Recall@k względem exact w bucketach scope selectivity i
  filter–vector correlation;
- p50/p95/p99, CPU, rows/bytes, storage, projection lag i stale rate;
- sweep beam, cosine floor, overfetch i RRF weights;
- osobne wyniki dla świeżych/starych rekordów oraz identifier, conceptual i
  task-bootstrap intent.

Historical dense nadal wymaga bitemporalnego relational prefilteru + exact
ranking albo osobnej signed historical projection. Current ANN nie jest w tym
celu rozszerzany post-filterem.

### Operacyjny dług v2

- `schema install` celowo wykonuje pełny `O(N)` rebuild projekcji jako ścieżkę
  migracji i naprawy; przed dużym wdrożeniem potrzebuje osobnego resumable joba,
  progress telemetry i kontrolowanego rate limitu;
- upgrade pre-v8 → v8 wymaga writer barrier: zatrzymania starych publisherów i
  embedding workers, wykonania `schema install` + `verify`, a dopiero potem
  uruchomienia v8; mixed-version writes podczas backfillu nie są wspierane;
- upgrade v8 → v9 jest additive (nowa pusta tabela licznika, brak rebuildu), ale
  zmienia checksum `schema.sql`, więc istniejąca baza wymaga `schema install`
  przed startem procesów v9;
- trwały trace sink oraz metryki lane latency/underfill/freshness pozostają
  kolejnym slicem.

### Kolejne lane'y

- source chunks i evidence expansion;
- latest handoff/checkpoint jako deterministic context;
- task-dependency graph i open-conflict expansion;
- temporal/entity routing;
- diversity/context packing;
- reranker dopiero po lane ablations;
- trwały, audytowany trace sink i metryki latency/underfill/freshness.

### P4 — dług ewaluacyjny graph

Bounded memory-link graph jest correctness-complete, ale nie jest jeszcze
benchmark-complete. Reprezentatywny corpus musi zmierzyć direct-only vs
direct+graph Recall@k/MRR/nDCG/no-answer, osobno dla associative/multi-hop,
conflict i zwykłych factual queries. Należy też wykonać sweep hopów, seedów,
fan-out, edge budget, query-gate floor, relation decay oraz graph RRF weight,
raportując p50/p95/p99, rows read i odsetek truncation/underfill. Bez tego nie
ma podstaw do twierdzenia o SOTA jakości.

## Definition of done dla implementacji retrieval v2

Ten slice jest gotowy do merge, gdy przechodzą:

```bash
uv run --extra dev python -m pytest -q
uv run --extra dev ruff check src tests
uv run --extra dev ruff format --check src tests
python3 -m compileall -q src tests
python3 scripts/check_markdown_links.py
git diff --check
```

oraz na izolowanym CockroachDB:

```bash
SWARMBRAIN_TEST_DATABASE_URL=... \
  uv run --extra dev python -m pytest \
  tests/test_cockroach_retrieval_live.py \
  tests/test_cockroach_embeddings_live.py \
  tests/test_cockroach_memory_live.py -q
```

Benchmark-complete pozostaje osobnym exit criterion opisanym w
[retrieval evaluation](retrieval-evaluation.md); zielone testy correctness nie
są jego substytutem.
