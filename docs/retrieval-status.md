# Retrieval v1 — aktualny stan

Stan na: 2026-08-05
Repozytorium: standalone `swarm-brain`
Gałąź implementacyjna: `feat/retrieval-v1`

## Wynik

Swarm Brain ma działający, provider-neutral retrieval v1 dla pamięci:

```text
RecallQuery + ActorContext
          ↓
server-owned purpose + planner
          ↓
exact ─ FTS simple ─ trigram
          ↓
weighted RRF (k=60)
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
- osobne porty `RetrievalGateway` i `CanonicalMemoryReader`;
- pełny trace pozostaje wewnętrzny i nie zmienia publicznego `RecallBundle`;
- stabilna normalizacja publicznego score do zakresu `[0,1]`;
- exact/lexical/fuzzy collapse po canonical memory ID;
- awaria pojedynczego lane'u jest oznaczona jako degraded, a nie ukryta.

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

### CockroachDB schema v7

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

## Weryfikacja wykonana w tej gałęzi

- fresh schema v7 install i verify na izolowanym CockroachDB v26.2;
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

Pełne końcowe wyniki komend są raportowane w handoffie gałęzi; ten dokument nie
jest automatycznym substytutem CI evidence.

## Co pozostaje

### Phase 2 — dług ewaluacyjny

Implementacja indexed lexical/identifier jest domknięta, ale wymagany przez
architekturę mierzony baseline lexical-only nie został jeszcze wykonany na
reprezentatywnym korpusie. Check-in gold set jest deterministyczną regresją
correctness (w tym abstention i rekord starszy niż 2005 decoyów), a nie
benchmarkiem jakości. Przed oznaczeniem phase 2 jako benchmark-complete trzeba
opublikować Recall@k, MRR, no-answer precision, percentyle latency, wersję
korpusu oraz ablation exact vs FTS vs trigram vs fused.

### P3 — poprawna integracja dense

Istniejący current-only `VECTOR(1024)` plane nadal jest compatibility path po
lexical bundle. Nie jest jeszcze pełnoprawnym `CandidateBatch`, ponieważ vector
projection nie zapisuje canonical resource version, content digest ani
projection signature. W konsekwencji semantic compatibility merge nadal nie
jest częścią weighted RRF.

Następny slice powinien:

1. dodać resource version/content SHA/projection signature do vector row;
2. filtrować visibility/run/task/state/trust/time przed ANN lane limit albo
   stosować temporal prefilter + exact vector ranking;
3. mierzyć ANN Recall@k względem exact vector oracle;
4. włączyć dense do tego samego RRF i trace;
5. zachować current/historical projections jako osobne polityki.

### Operacyjny dług v1

- `schema install` celowo wykonuje pełny `O(N)` rebuild projekcji jako ścieżkę
  migracji i naprawy; przed dużym wdrożeniem potrzebuje osobnego resumable joba,
  progress telemetry i kontrolowanego rate limitu;
- upgrade pre-v7 → v7 wymaga writer barrier: zatrzymania starych publisherów,
  wykonania `schema install` + `verify`, a dopiero potem uruchomienia v7; mixed
  pre-v7/v7 online writes podczas backfillu nie są wspierane;
- lokalny adapter zlicza `memories_reused`, ale CockroachDB nie ma jeszcze
  trwałego licznika reuse; potrzebny jest audytowalny retrieval event/metric
  zapisywany poza read-only snapshotem;
- trwały trace sink oraz metryki lane latency/underfill/freshness pozostają
  kolejnym slicem.

### Kolejne lane'y

- source chunks i evidence expansion;
- latest handoff/checkpoint jako deterministic context;
- bounded graph/lineage i open conflicts;
- temporal/entity routing;
- diversity/context packing;
- reranker dopiero po lane ablations;
- trwały, audytowany trace sink i metryki latency/underfill/freshness.

## Definition of done dla retrieval v1

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
  tests/test_cockroach_memory_live.py -q
```
