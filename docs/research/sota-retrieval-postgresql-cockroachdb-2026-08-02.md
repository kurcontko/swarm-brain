# SOTA retrieval dla PostgreSQL i CockroachDB

Data researchu: 2026-08-02; aktualizacja źródeł: 2026-08-06

Zakres: retrieval dla pamięci agentowej i systemów RAG, ze szczególnym
uwzględnieniem architektury Sen. Dokument opisuje techniki, retrieval lanes,
ograniczenia baz danych oraz rekomendowaną kolejność wdrożeń.

> Ten dokument jest analizą techniczną, nie źródłem aktualnego statusu produktu
> ani wyniku benchmarkowego Sen. Publikowane wyniki z różnych prac mogą używać
> innych zbiorów danych, readerów, budżetów kontekstu i evaluatorów.

Swarm Brain jest obecnie osobnym projektem. Sposób zastosowania tego researchu
oraz aktualny stan exact/FTS/trigram/RRF opisują
[architektura standalone](../retrieval-architecture.md) i
[status retrieval v2](../retrieval-status.md).

## Executive summary

SOTA retrieval w 2026 roku nie jest pojedynczym retrieverem ani indeksem
wektorowym. Najmocniejszy produkcyjny wzorzec to system wieloetapowy:

```text
hard scope / ACL / tenant / time
              ↓
intent + entity + temporal parsing
              ↓
┌ exact ┬ lexical ┬ fuzzy ┬ dense ┬ temporal ┬ entity/graph ┬ hierarchy ┐
              ↓
collapse do canonical memory ID
              ↓
RRF lub learned fusion
              ↓
cross-encoder / late-interaction rerank
              ↓
dedup + diversity + parent/source expansion
              ↓
token-budget packing + sufficiency check
              ↓
wynik albo corrective retry / abstention
```

Najkrótsza rekomendacja:

- **PostgreSQL** jest lepszym wyborem, gdy priorytetem jest jakość,
  elastyczność i szerokość retrievalu w jednym systemie.
- **CockroachDB** ma sens, gdy pamięć musi być silnie spójna, wieloregionowa
  i lokalna dla użytkownika lub tenanta.
- **Sen** powinien zachować wspólną semantykę lane'ów i fusion, ale używać
  różnych implementacji backendowych. Wymuszanie feature parity obniżyłoby
  jakość PostgreSQL albo nadmiernie skomplikowało CockroachDB.

Najświeższy bezpośredni proof point to
[Hindsight, ACL 2026](https://aclanthology.org/2026.acl-demo.27/): system
oparty o PostgreSQL i pgvector łączy równoległy dense, lexical, graph i
temporal retrieval. Autorzy raportują 83,6% na LongMemEval z modelem 20B
oraz 91,4% z Gemini-3 Pro. Jest to mocny przykład architektoniczny, ale nadal
wynik jednego systemu i konfiguracji, a nie uniwersalny ranking baz danych.

## Aktualizacja 2026-08-06: wnioski dla dense v2

Ponowne sprawdzenie źródeł pierwotnych nie zmienia ogólnej architektury, ale
doprecyzowuje pięć decyzji implementacyjnych.

1. **Filtered ANN wymaga projektowania indeksu, nie tylko `WHERE`.** Oficjalna
   dokumentacja [CockroachDB v26.2 vector indexes](https://www.cockroachlabs.com/docs/v26.2/vector-indexes)
   mówi, że vector index z prefix columns jest używany tylko wtedy, gdy każdy
   prefix ma konkretną wartość equality albo kompletny tuple `IN`. Filtry inne
   niż prefix nie mają akceleracji vector indexu. Dla Swarm Brain oznacza to
   osobne equality-bound branches dla `repository:*`, `run:*` i `task:*`, a
   nie globalny ANN z post-filterem widoczności.
2. **Beam jest parametrem jakości, nie stałą poprawności.** CockroachDB podaje
   domyślny `vector_search_beam_size=32`; większy beam bada więcej partycji,
   podnosząc recall kosztem CPU i latency. Musi być konfigurowalny i mierzony
   względem exact oracle na każdym bucketcie selectivity.
3. **PostgreSQL potrzebuje innej polityki filtered ANN.** Oficjalny
   [pgvector](https://github.com/pgvector/pgvector) wykonuje domyślnie exact
   nearest-neighbor search. HNSW zwykle daje lepszy speed/recall trade-off niż
   IVFFlat, ale zwykłe filtry są nakładane po skanie approximate indexu.
   Iterative scans od 0.8.0 ograniczają underfill; bardzo selektywne filtry
   nadal często wygrywają przez B-tree/partycję i exact vector sort.
4. **RRF jest baseline'em, nie końcem optymalizacji.** Analiza
   [Bruch, Gai i Ingber](https://arxiv.org/abs/2210.11934) pokazuje wrażliwość
   RRF na parametry oraz przewagę uczonej convex combination w ich testach.
   Dlatego Swarm Brain zachowuje audytowalne contribution-level RRF bez zbioru
   treningowego, ale wersjonuje weights i wymaga lane ablations; learned fusion
   ma sens dopiero po zebraniu reprezentatywnych judgments.
5. **Następny skok jakości jest strukturalny.** [Hindsight](https://aclanthology.org/2026.acl-demo.27/)
   łączy vector, keyword, graph i temporal retrieval;
   [APEX-MEM](https://aclanthology.org/2026.acl-long.749/) zachowuje append-only
   temporal evolution w entity-centric property graph; a
   [MRAgent](https://arxiv.org/abs/2606.06036) aktywnie eksploruje i przycina
   Cue–Tag–Content graph. Po poprawnym hybrid dense v2 większy potencjał ma
   bounded source/graph/temporal expansion niż kolejna niezmierzona zmiana
   samego ANN.

[BEIR](https://arxiv.org/abs/2104.08663) pozostaje ostrzeżeniem przed
optymalizacją do jednego rodzaju zapytań: benchmark obejmuje heterogeniczne
domeny i pokazuje istotne różnice generalizacji retrieverów. Produktowy raport
Swarm Brain musi więc osobno mierzyć identifier/code lookup, konceptualne
parafrazy, task bootstrap, temporal/contradiction, multi-evidence i no-answer.

Wynikająca kolejność: signed current vector projection → equality-bound
auth/visibility/projection prefix → ANN → canonical validation w tym samym
snapshotcie z bounded adaptive widening → exact oracle filtrujący eligible set
przed exact sort → dense/lexical/hybrid ablations → dopiero potem learned
fusion, reranker oraz bounded graph. Samo wdrożenie tej architektury nie jest
twierdzeniem o SOTA; takie twierdzenie wymaga reprezentatywnego,
wersjonowanego benchmarku.

## Dwie osie lane'ów

Warto rozdzielić dwa znaczenia słowa „lane”.

### Domain lanes

Domain lane określa, czym semantycznie jest rekord:

- `fact` — atomowy fakt lub wersja faktu;
- `episode/source` — surowe zdarzenie, wiadomość lub fragment źródła;
- `memory` — skonsolidowana pamięć lub obserwacja;
- `document/summary` — większy dokument, temat lub podsumowanie;
- `procedure` — workflow, strategia, gotcha albo wyuczony sposób działania;
- `intention` — cel, plan, zobowiązanie lub stan przyszły;
- `entity/edge` — encja i typowana relacja.

### Retrieval signals

Retrieval signal określa, jak rekord został znaleziony:

- exact/structured;
- lexical;
- fuzzy/trigram;
- dense semantic;
- temporal;
- entity/graph;
- neighborhood/parent;
- hierarchy/summary;
- reranker.

Ten sam fakt może zostać znaleziony jednocześnie przez lexical, dense i graph.
Domain lane powinien sterować polityką oraz prezentacją, a signal powinien
zapewniać provenance, diagnostykę i wkład do fusion.

## Rekomendowane retrieval lanes

| Lane | Co znajduje | Kiedy uruchamiać | Dojrzałość |
|---|---|---|---|
| Hard constraint | tenant, ACL, status, deletion, valid/system time | Zawsze, przed `LIMIT` w każdym retrieverze | Obowiązkowe |
| Exact/structured | ID, nazwy kanoniczne, statusy, liczby, agregacje | Zawsze tanio; dominuje dla pytań strukturalnych | Dojrzałe |
| Lexical sparse | Rzadkie termy, cytaty, nazwy, kody | Zwykle zawsze równolegle z dense | Dojrzałe |
| Fuzzy/trigram | Literówki, aliasy, substringi, symbole | Nazwy, identyfikatory i kod | Dojrzałe |
| Dense semantic | Parafrazy, opisowe podobieństwo, intencje | Zwykle zawsze; ANN lub exact nad małą pulą | Dojrzałe |
| Temporal | Latest-before, as-of, zakresy i kolejność zdarzeń | Po wykryciu temporal intent | Niezbędne dla memory |
| Entity | Wszystko o rozpoznanej osobie, projekcie lub obiekcie | Po entity resolution | Dojrzałe |
| Graph/causal | Multi-hop, zależności, przyczyny i relacje | Bounded 1–2, maksymalnie 3 hops | Warunkowo dojrzałe |
| Parent/hierarchy | Sesja, źródło, rodzeństwo, summary | Aggregation, global i multi-session | Warunkowo dojrzałe |
| Procedure/intention | Workflow, plan, wcześniejsze sukcesy i błędy | Planning i action-oriented queries | Wczesne, obiecujące |
| Deep fallback | Rewrite, decomposition, HyDE, większy ANN beam | Tylko przy niskiej pewności | Warunkowe |

Dense i lexical powinny być produkcyjnym baseline'em.
[BEIR](https://arxiv.org/abs/2104.08663) pokazuje, że nie istnieje jeden
retriever wygrywający wszystkie domeny. Do fusion dobrym startem pozostaje
[Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf),
ponieważ nie wymaga kalibracji skal cosine, `ts_rank`, BM25, trigram i graph
score.

RRF nie jest automatycznie optymalne. Gdy istnieją relevance judgments, uczona
kombinacja score'ów lub learning-to-rank może przewyższyć statyczne RRF.
Najpierw trzeba jednak zgromadzić dane treningowe i zachować surowe sygnały.

## Routing według typu zapytania

### Exact ID, nazwa lub kod

1. B-tree/exact lookup.
2. Aliasy i canonical entity.
3. Trigram lub prefix lookup.
4. Lexical jako fallback.

Dense powinien mieć niewielki wpływ na identyfikatory.

### Semantic fact lookup

1. Hard scope.
2. Lexical i dense równolegle.
3. Entity expansion, jeśli rozpoznano encję.
4. RRF.
5. Cross-encoder nad małą pulą.

### Temporal

1. Rozpoznanie daty, zakresu i relacji typu „przed”, „po”, „wtedy”.
2. Bitemporal SQL: valid/event time oraz observed/system time.
3. Źródła i fakty pasujące do zakresu.
4. Chronologiczne neighborhood expansion.
5. Temporal-aware reranking.

Globalny recency decay nie może usuwać starych faktów. Dla pytania
historycznego stary rekord może być jedyną poprawną odpowiedzią.

### Multi-hop

1. Lexical+dense seeds.
2. Canonical entity resolution.
3. Kontrolowana ekspansja grafu 1–2 hops.
4. Ocena pokrycia wymaganych encji i relacji.
5. Zachowanie kilku niezależnych dowodów podczas diversity.

### Aggregation lub pytanie globalne

1. Topic/collection lane.
2. Hierarchiczne summary.
3. Reprezentatywne child/source evidence.
4. Coverage-aware reranking i packing.

### Planning

1. Aktualne fakty.
2. Intencje i zobowiązania.
3. Procedury, strategie i wcześniejsze gotchas.
4. Jawne rozróżnienie wiedzy od sugestii i opinii.

## PostgreSQL

### Rekomendowany baseline

```text
PostgreSQL 18
+ pgvector HNSW
+ weighted TSVECTOR/GIN albo BM25
+ pg_trgm
+ RRF
+ cross-encoder w aplikacji
```

### Dense retrieval: pgvector

[pgvector](https://github.com/pgvector/pgvector) zapewnia:

- exact scan jako oracle z pełnym recall;
- HNSW jako domyślny wybór quality/latency;
- IVFFlat dla szybszego i tańszego builda oraz specyficznych workloadów;
- cosine, inner product i L2;
- iterative scans dla zapytań z filtrami;
- `halfvec`, binary quantization, sparse vectors i subvectors;
- możliwość generowania kandydatów na tańszej reprezentacji i dokładnego
  rerankingu oryginalnym wektorem.

HNSW zwykle zapewnia lepszy speed/recall niż IVFFlat, ale:

- buduje się dłużej;
- zużywa więcej pamięci;
- wymaga strojenia `m`, `ef_construction` i `ef_search`;
- nie zawsze wygrywa w filtered ANN;
- może wymagać kosztowniejszego maintenance i vacuum.

IVFFlat wymaga reprezentatywnych danych podczas budowy. Liczba list i probes
musi być dobrana do wielkości zbioru oraz oczekiwanego recall.

### Filtered ANN

To najważniejszy problem projektowy w PostgreSQL. W pgvector filtr metadanych
jest często stosowany po pobraniu kandydatów z HNSW lub IVFFlat. Jeżeli tylko
10% rekordów przechodzi filtr, mała pula ANN może nie dostarczyć oczekiwanej
liczby wyników.

Iterative scans od pgvector 0.8+ zwiększają przeszukiwany fragment indeksu,
aż znajdą wystarczająco kandydatów albo osiągną limit. Nie rozwiązują jednak
każdej postaci joinów, subqueries i arbitralnego przecięcia B-tree × HNSW.

Rekomendowana polityka:

- bardzo selektywny filtr: B-tree/BRIN/partycja, potem exact vector sort;
- średnio lub słabo selektywny filtr: HNSW + iterative scan;
- kilka stabilnych kategorii lub tenantów: partial HNSW indexes;
- wiele tenantów albo time buckets: LIST/RANGE partitioning;
- złożone high-cardinality filters: benchmark wyspecjalizowanego filtered ANN.

Tenant, ACL, deletion i time scope muszą być zastosowane wewnątrz każdego lane
przed `LIMIT`. Post-filtering dopiero w aplikacji jest ryzykiem bezpieczeństwa
i recall.

### Wymiar embeddingu Sen

W audytowanym snapshotcie donor repo Sen domyślna konfiguracja używała
embeddingu 2560-dimensional. Pliki źródłowe pozostają w osobnym repozytorium
Sen i celowo nie są zależnością ani linkiem filesystemowym standalone Swarm
Brain.

Zwykły indeks HNSW na typie `vector` w pgvector obsługuje do 2000 wymiarów.
Indeksy `halfvec` obsługują do 4000 wymiarów. Sensowne warianty dla Sen:

1. MRL truncation do 1024 albo 1536 wymiarów.
2. `halfvec(2560)`.
3. Tańszy subvector/quantized candidate generation i rerank pełnym wektorem.

Wyboru należy dokonać na podstawie Recall@k, p95, storage i write cost.

### Natywny lexical retrieval

PostgreSQL FTS oferuje:

- generated lub utrzymywany `tsvector`;
- GIN jako preferowany indeks;
- `websearch_to_tsquery` dla surowego inputu użytkownika;
- frazy i prefix queries;
- weights A–D;
- `ts_rank` i `ts_rank_cd`;
- `ts_headline`.

[Dokumentacja PostgreSQL FTS](https://www.postgresql.org/docs/current/textsearch-controls.html)

Core PostgreSQL FTS nie jest BM25. Ranking nie korzysta z pełnych globalnych
statystyk korpusu/IDF i może być kosztowny przy szerokich matchach. Jest jednak
dobrym, dojrzałym matcherem dla umiarkowanej skali.

### Fuzzy i identifiers

Osobny [`pg_trgm`](https://www.postgresql.org/docs/current/pgtrgm.html) dobrze
obsługuje:

- literówki;
- similarity i word similarity;
- substringi;
- `LIKE`, `ILIKE` i regex;
- nazwy własne i identyfikatory.

To powinien być osobny signal, a nie dodatek ukryty wewnątrz ogólnego lexical
score.

### BM25 w PostgreSQL

Jeżeli natywny FTS okaże się bottleneckiem:

- [`pg_textsearch`](https://github.com/timescale/pg_textsearch) — BM25,
  Block-Max WAND i relatywnie mały zakres funkcji;
- [ParadeDB `pg_search`](https://docs.paradedb.com/welcome/introduction) —
  BM25, phrase, fuzzy, facets, boosting i bogatszy search UX.

`pg_textsearch` jest prostszym wyborem dla szybkiego top-k BM25. `pg_search`
jest bardziej zbliżony do pełnego produktu wyszukiwarkowego, ale oznacza
cięższy indeks, dodatkowe decyzje operacyjne i inne konsekwencje licencyjne.

Najlepiej wybrać jeden lexical stack po benchmarku:

- core FTS — najmniej operacji i największa przenośność;
- `pg_textsearch` — prosty in-DB BM25;
- `pg_search` — bogate search features.

### Frontier vector extensions

Przy bardzo dużej skali warto benchmarkować:

- [pgvectorscale](https://github.com/timescale/pgvectorscale) —
  SSD-oriented StreamingDiskANN, quantization i ograniczone label filters;
- [VectorChord](https://github.com/supervc-stack/VectorChord) —
  disk-friendly IVF/RaBitQ oraz eksperymentalna ścieżka do MaxSim/ColBERT;
- provider-specific AlloyDB ScaNN lub Azure DiskANN.

Nie są automatycznym zamiennikiem HNSW. Wyniki vendor benchmarks powinny być
potwierdzone na realnym corpus, filtrach i rozkładzie tenantów.

## CockroachDB

### Vector search: C-SPANN

CockroachDB v26.2 posiada rozproszony ANN C-SPANN. Vector indexes są GA od
linii v25.4. Implementacja używa hierarchicznych partycji k-means i pomysłów
z SPANN, SPFresh, ScaNN oraz RaBitQ.

[Dokumentacja vector indexes](https://www.cockroachlabs.com/docs/v26.2/vector-indexes)

Obsługiwane są:

| Operator | Opclass | Metryka |
|---|---|---|
| `<->` | `vector_l2_ops` | Euclidean/L2 |
| `<=>` | `vector_cosine_ops` | Cosine distance |
| `<#>` | `vector_ip_ops` | Negative inner product |

Główne parametry:

- `vector_search_beam_size` — większy beam zwiększa recall, CPU i latency;
- `min_partition_size` i `max_partition_size` — większe partycje zwiększają
  pulę kandydatów i koszt odczytu, ale mogą ograniczyć splity podczas ingestu;
- `build_beam_size` — dokumentacja zaleca zwykle stroić search beam zamiast
  build beam.

Exact vector scan nadal jest potrzebny jako ground truth i jako plan dla małej
puli kandydatów.

### Prefix filtering

Vector index może zawierać prefix columns:

```sql
CREATE VECTOR INDEX ON memory_vectors (
    tenant_id,
    namespace,
    embedding vector_cosine_ops
);
```

ANN zostanie użyty tylko wtedy, gdy wszystkie prefix columns mają:

- equality;
- albo pełne tuple `IN`.

Przykład korzystający z indeksu:

```sql
WHERE tenant_id = $1
  AND namespace = $2
ORDER BY embedding <=> $3
LIMIT $4
```

Przykład, który nie użyje vector indexu:

```sql
WHERE tenant_id = $1
  AND namespace >= $2
ORDER BY embedding <=> $3
LIMIT $4
```

Prefix powinien obejmować 1–3 stabilne pola equality obecne w niemal każdym
zapytaniu:

- `tenant_id` lub subject/bank;
- `projection_id` lub namespace;
- ewentualnie stabilny `state` lub locality region.

Dynamiczne range filters, confidence i dokładny event time nie powinny trafiać
do prefixu bez pomiaru. Dla temporal retrieval lepsze są:

- osobne current/history projections;
- coarse time buckets przekazywane przez tuple `IN`;
- relacyjny prefilter i exact vector sort małej puli;
- over-fetch z pomiarem filtered recall.

Sen ma już poprawny wzorzec:

```sql
(projection_id, subject_id, embedding vector_cosine_ops)
```

Implementacja referencyjna pozostaje w osobnym repozytorium Sen; do Swarm Brain
przenoszony jest wzorzec i evidence z benchmarku, nie runtime import.

### Exact retrieval

Bez vector indexu sortowanie według dystansu jest O(N). Exact lane ma sens,
jeśli najpierw ograniczymy zbiór przez:

- tenant lub subject;
- encje;
- status;
- czas;
- metadane;
- wynik FTS.

Dobry wzorzec:

1. B-tree/GIN wybiera setki lub kilka tysięcy kandydatów.
2. Exact cosine/L2 szereguje małą pulę.
3. `EXPLAIN ANALYZE` potwierdza oczekiwany plan.

### Full-text search

CockroachDB ma PostgreSQL-compatible:

- `TSVECTOR` i `TSQUERY`;
- operator `@@`;
- `ts_rank`;
- GIN/GiST/`CREATE INVERTED INDEX`.

[Dokumentacja CockroachDB FTS](https://www.cockroachlabs.com/docs/v26.2/full-text-search)

Ważne różnice względem PostgreSQL:

- `ts_rank` jest frequency-based, nie BM25;
- brak `ts_rank_cd`;
- brak `websearch_to_tsquery`;
- brak `setweight`;
- ranking szerokiego zbioru matchy może być kosztowny;
- GIN i GiST używają tej samej implementacji;
- brak polskiego słownika;
- konfiguracja tekstowa jest znacznie mniej rozszerzalna.

Dla polskiego corpus rekomendowany baseline:

- multilingual embeddings jako główna semantyka;
- FTS `simple` nad znormalizowanymi tokenami;
- osobny trigram dla nazw, symboli i identyfikatorów;
- zewnętrzny BM25 projection dopiero po wykazaniu luki jakościowej.

### Trigram

[CockroachDB trigram indexes](https://www.cockroachlabs.com/docs/v26.2/trigram-indexes)
przyspieszają:

- `LIKE` i `ILIKE`;
- operator podobieństwa `%`;
- substringi;
- literówki;
- nazwy własne i symbole.

Implementacja jest uboższa od PostgreSQL `pg_trgm`: brakuje między innymi
`word_similarity` oraz odpowiadających jej operatorów.

### Relacyjne metadata i graph lanes

JSONB/ARRAY inverted indexes nadają się do mniej regularnych metadanych. Często
filtrowane pola zakresowe powinny być zdenormalizowane do typowanych kolumn
z B-tree.

Graph lane najlepiej modelować jako tabelę typowanych krawędzi:

- source i target entity/memory;
- relation type;
- confidence/weight;
- valid time;
- provenance;
- tenant/subject.

Traversal powinien:

- startować od entity IDs albo top lexical/dense hits;
- kończyć się po 1–2 hops;
- stosować ACL, tenant i time predicates na każdym hopie;
- unikać dużego fan-out przez regiony.

#### Wynik implementacyjny Swarm Brain, 2026-08-06

Po dense v2 wdrożony został pierwszy mierzony structural slice:
`memory_links` jako second-stage lane seedowany wynikiem direct RRF. Nie jest to
pełny GraphRAG ani PPR. To fixed-depth spreading activation z jednym hopem dla
interactive/historical/orientation i dwoma dla
bootstrap/handoff/planning/conflict review.

Decyzja SQL jest świadoma różnic między systemami:

- [PostgreSQL 18 `WITH RECURSIVE`](https://www.postgresql.org/docs/current/queries-with.html)
  dokumentuje depth/path arrays, `SEARCH` i `CYCLE`, ale podkreśla, że kolejność
  odwiedzania jest implementation-dependent i zaleca jawne sortowanie;
- [CockroachDB 26.2 recursive CTE](https://www.cockroachlabs.com/docs/v26.2/common-table-expressions)
  wymaga jawnego warunku końca, odradza poleganie na outer `LIMIT` w produkcji i
  zaznacza, że część recursive CTE nie jest jeszcze zoptymalizowana;
- dlatego critical path używa jawnej pętli 1–2 hopów w jednym retrieval
  snapshotcie, batched `LATERAL` scans i osobnych covering indexes
  `(source_memory_id, link_type, created_at DESC, id)` oraz symetrycznego
  target indexu.

Per-node raw scan ma stały cap równy maksymalnie czterem fan-outom. Endpointy
są następnie walidowane canonicalnie; dopiero eligible node otrzymuje jeden z
ośmiu fan-out slots. To uniknęło dwóch złych planów sprawdzonych live na
CockroachDB 26.2.1: złożony trust predicate wewnątrz wymuszonego `LOOKUP JOIN`
nie dawał legalnego planu, a zwykły join został zhashowany nad szerokim skanem
`memories`. Dwufazowy bounded edge scan → primary-ID validation zachowuje oba
directional indexes, brak full scan i pełną politykę auth/trust/time przed
przejściem do następnego frontieru.

Scoring czerpie z aktualnych wyników graph-memory research, ale pozostaje
deterministyczny. [HippoRAG 2](https://arxiv.org/abs/2502.14802) pokazuje wartość
graph propagation/PPR dla factual, sense-making i associative memory, jednak
globalne PPR nie należy do synchronicznego critical path. Nowsze
[Query-Aware Spreading Activation](https://arxiv.org/abs/2606.30133) raportuje
korzyść z fixed-iteration, per-step query gate względem query-blind traversal.
Swarm Brain stosuje więc seed activation z direct hybrid, relation/direction
weight, `0.85` step decay i bounded lexical query gate z floorem `0.60`.
Semantic dense gate dla target node pozostaje hipotezą do ablation, nie
niezmierzoną zależnością runtime.

Każdy kandydat zapisuje pełny edge/node path, relation sequence, hop count i
cumulative activation; path-local node set blokuje cykle. Graph lane ma niższy
RRF weight niż direct evidence (z wyjątkiem umiarkowanego boostu conflict
review), degraduje się niezależnie i nie omija final canonical hydration.

To jest SOTA-informed architecture, nie dowód SOTA jakości. Release evidence
musi porównać direct-only z direct+graph na reprezentatywnych factual,
associative, multi-hop i no-answer judgments oraz wykonać sweep hop/fan-out,
seed/edge budget, relation decay, gate floor i graph RRF weight.

### Multi-region

Dla pamięci użytkowników naturalną podstawą jest
[`LOCALITY REGIONAL BY ROW`](https://www.cockroachlabs.com/docs/v26.2/table-localities):

- rekordy i indeksy są rozmieszczane według regionu;
- vector index pozostaje współlokowany z rekordem;
- zapytania użytkownika można kierować do home region;
- małe read-mostly tabele polityk mogą używać `GLOBAL`.

Globalny retrieval powinien uruchamiać regionalne kanały i scalać ich top-k,
zamiast domyślnie wykonywać szeroki traversal przez WAN.

### Ograniczenia operacyjne

- Duże batche vector inserts mogą pogorszyć wydajność.
- `IMPORT INTO` nie działa na tabeli mającej vector index; bezpieczny wzorzec
  to import przed utworzeniem indeksu.
- Bardzo duże wektory zwiększają write amplification.
- Brak L1, Hamming i Jaccard vector opclasses.
- Brak automatycznych rekomendacji vector indexów.
- Bieżąca strona vector indexes nadal ostrzega o blokowaniu zapisów podczas
  backfillu indeksu na niepustej tabeli. Release notes v25.4 opisują ulepszenia
  online. Zachowanie należy potwierdzić na konkretnej wersji patchowej w
  stagingu; najbezpieczniej tworzyć projection table i indeks przed ingestem.

## PostgreSQL kontra CockroachDB

| Obszar | PostgreSQL | CockroachDB |
|---|---|---|
| Dense ANN | pgvector HNSW/IVFFlat; szeroki ekosystem | Rozproszony C-SPANN |
| Exact vector | Tak | Tak |
| Filtered ANN | Iterative scans, partial indexes, partycje | Bardzo dobre equality prefix filters; słabe arbitrary/range filters |
| Core lexical | Bogatszy FTS, phrase, weights, `ts_rank_cd` | TSVECTOR/GIN i `ts_rank`, mniej funkcji |
| BM25 | Kilka rozszerzeń in-DB | Brak natywnego BM25 |
| Fuzzy | Bogaty `pg_trgm` | Podstawowy trigram |
| Learned sparse | `sparsevec` lub rozszerzenia/serwis | Zwykle zewnętrzna projekcja |
| Late interaction | VectorChord lub zewnętrzny reranker | Zewnętrzny reranker |
| Graph | Recursive CTE/aplikacja | Bounded CTE/aplikacja, kolokacja po tenant |
| Temporal | Range types, SQL, app bitemporal model | SQL + app bitemporal model; MVCC nie zastępuje event time |
| Multi-region | Wymaga dodatkowej architektury | Główna przewaga produktu |
| Ekosystem extensions | Bardzo szeroki | Celowo węższy |
| Najlepszy fit | Retrieval-first | Distributed-memory-first |

## Zaawansowane techniki

### Learned sparse: SPLADE

[SPLADE v2](https://arxiv.org/abs/2109.10086) generuje uczone, ważone termy
i rozszerzenia słownikowe. Może przewyższać BM25, szczególnie na zapytaniach
z vocabulary mismatch.

Ocena:

- model retrieval jest dojrzały badawczo;
- produkcyjne uruchomienie w standardowym SQL jest trudniejsze;
- PostgreSQL `sparsevec` nie jest pełnym impact-pruned inverted engine;
- CockroachDB wymagałby zewnętrznej projekcji lub własnego formatu;
- nie jest rekomendowanym pierwszym krokiem przed dobrym BM25/FTS+dense.

### Late interaction: ColBERT

[ColBERTv2](https://aclanthology.org/2022.naacl-main.272/) przechowuje wiele
embeddingów tokenowych dla dokumentu i oblicza MaxSim. Zapewnia lepszą
precyzję niż pojedynczy embedding, ale znacznie zwiększa storage i złożoność
indeksowania.

Praktyczne warianty:

- hybrid DB retrieval, potem ColBERT rerank top 50–200;
- osobny PLAID/ColBERT service;
- VectorChord MaxSim jako eksperyment PostgreSQL.

Pełnego row-per-token retrievalu nie warto budować w CockroachDB bez mocnego
benchmarkowego uzasadnienia.

### Hierarchical retrieval

[RAPTOR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html)
buduje drzewo klastrów i abstrakcyjnych podsumowań. Jest użyteczny dla:

- globalnych pytań o corpus;
- aggregation;
- wielosesyjnych tematów;
- wyboru właściwego parent context.

Summary musi zachowywać provenance do children i podlegać versioningowi.
W przeciwnym razie pojawia się summary drift.

### Graph retrieval

[GraphRAG](https://arxiv.org/abs/2404.16130) i
[HippoRAG 2](https://arxiv.org/abs/2502.14802) pokazują wartość grafów dla
global sensemaking i associative multi-hop. Nie powinny zastępować lexical
i dense retrieval.

Najbezpieczniejszy produkcyjny wariant:

- canonical entities;
- typed temporal edges;
- lexical/dense seeds;
- bounded traversal;
- provenance i confidence;
- fusion z innymi lane'ami.

### Query rewriting i adaptive retrieval

- [HyDE](https://aclanthology.org/2023.acl-long.99/) generuje hipotetyczny
  dokument i wyszukuje przez jego embedding.
- [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/) wybiera
  no-retrieval, single-step albo iterative retrieval.
- Query decomposition pomaga przy multi-hop, ale zwiększa koszt i może
  propagować błędy.

Rewriting i decomposition powinny być deep fallbackiem uruchamianym przy
niskiej pewności, nie obowiązkowym kosztem każdego query.

### Cross-encoder i listwise reranking

Cross-encoder wspólnie koduje query i dokument, dlatego zwykle przewyższa
bi-encoder na małej puli. Naturalne miejsce to top 30–100 kandydatów po fusion.

Generic topical reranker może jednak szkodzić:

- temporal relevance;
- wymaganej kolejności zdarzeń;
- multi-hop evidence coverage;
- różnorodności źródeł.

Reranker dla pamięci powinien widzieć również:

- event/valid time;
- observed/system time;
- current/superseded state;
- source provenance;
- entity/path coverage;
- query intent.

Listwise LLM reranking jest droższy i mniej deterministyczny. Powinien być
używany tylko dla trudnych zapytań.

### MMR i diversity

Po rerankingu warto stosować Maximal Marginal Relevance albo podobną politykę,
aby jedna sesja lub dokument nie zdominowały całego kontekstu.

Diversity można liczyć po:

- embedding similarity;
- parent/session ID;
- entity overlap;
- source;
- event time.

Multi-hop wymaga wyjątku: kilka podobnych fragmentów może stanowić niezależne
elementy jednego dowodu.

## Ocena aktualnego Sen

Obecny Sen jest bliżej SOTA niż klasyczne „wrzuć wszystko do vector DB”.

### Mocne strony

- sześć domain lanes: fact, memory, source, document, procedure, intention;
- sparse + semantic + graph signals;
- weighted RRF z klasycznym `k=60`;
- routing intencji;
- world/system time i temporal modes;
- episodic/source neighborhood expansion;
- opcjonalny reranker;
- token-budget packing;
- źródła jako prawda, indeksy jako rebuildable projections.

Kluczowe miejsca w donor repo to pipeline recall, CockroachDB vector projection
i CockroachDB FTS schema. Standalone Swarm Brain nie linkuje ich relatywną
ścieżką: granicę i transfer wzorców opisuje
[architektura retrieval](../retrieval-architecture.md).

### Istotne luki

#### P0: benchmark retrievalowy według intentów

Potrzebne są osobne zbiory:

- exact ID/name;
- semantic paraphrase;
- temporal point/range/as-of;
- knowledge update i contradiction;
- multi-session multi-hop;
- aggregation/global;
- planning/workflow;
- negative/no-answer.

Bez relevance labels lane weights pozostają heurystyką.

#### P0: multilingual lexical retrieval

Schemat CockroachDB hardcoduje `to_tsvector('english', ...)` w kilku
projekcjach. Dla polskiego i multilingual corpus może to obniżać recall.

Najpierw należy porównać:

- `english`;
- `simple`;
- własną normalizację tokenów;
- trigram;
- multilingual dense;
- ewentualny BM25 projection.

#### P1: historyczny semantic retrieval

Persisted semantic ANN dla trybów innych niż `current` jest obecnie wyłączony
poza `source` lane. Może to gubić stare fakty, memories i documents w temporal
recall, szczególnie gdy fallback działa nad ograniczoną pulą.

Możliwe rozwiązania:

- osobna historical vector projection;
- bitemporal projection keys;
- exact semantic rerank kandydatów wybranych temporal SQL;
- current/history namespace z jawnym routingiem.

#### P1: jawny exact/fuzzy signal

Exact entity/name/identifier i trigram powinny być niezależnymi sygnałami,
z osobnym candidate depth i telemetry, zamiast być ukryte w ogólnym sparse
lane.

#### P1: observability

Dla każdego kandydata warto zachować:

- raw lexical/vector/trigram/graph score;
- rank w każdym lane;
- filter selectivity;
- ANN parameters;
- query variants;
- provenance/path;
- collapse/canonical ID;
- reason odrzucenia.

RRF może nadal używać wyłącznie ranków, ale surowe dane są konieczne do
diagnostyki i learned fusion.

#### P1: temporal-aware reranker

Obecne pomijanie generic rerankera dla temporal, aggregation i multi-hop jest
rozsądnym zabezpieczeniem. Kolejnym krokiem powinien być reranker uczony lub
zaprojektowany na temporal relevance i evidence coverage, a nie bezwarunkowe
włączenie topical rerankera.

#### P2: hierarchy i advanced retrieval

Po zamknięciu powyższych luk warto testować:

- session/topic summaries;
- RAPTOR-style hierarchy;
- SPLADE;
- ColBERT rerank;
- query rewriting;
- decomposition;
- calibrated lub learned fusion.

## Rekomendowana kolejność dla Sen

1. Zbudować gold retrieval suite z query intent i wymaganym evidence set.
2. Dodać exact/alias/trigram jako jawny signal.
3. Naprawić multilingual lexical path, szczególnie CockroachDB `english` FTS.
4. Zapewnić historyczny semantic retrieval.
5. Zapisać pełną telemetry lane'ów i ANN recall względem exact oracle.
6. Dostroić PostgreSQL HNSW/halfvec/truncation oraz CockroachDB beam/prefixes.
7. Zbudować temporal-aware reranker i adaptive candidate budgets.
8. Dopiero potem testować BM25 extensions, SPLADE, ColBERT i RAPTOR.

## Jak benchmarkować

### Quality

- Recall@k osobno per lane i po fusion;
- MRR i nDCG po rerankingu;
- all-evidence recall dla multi-hop;
- temporal precision i state-as-of correctness;
- update/contradiction accuracy;
- abstention/no-answer accuracy;
- source/provenance correctness;
- zero cross-tenant i ACL leaks.

### ANN

- Recall@k względem exact oracle;
- corpus size;
- tenant size;
- filter selectivity;
- korelacja filter–embedding;
- świeże kontra stare rekordy;
- różne `k`, beam, `ef_search` i oversampling.

### Performance i operacje

- p50/p95/p99;
- QPS;
- CPU i memory;
- rows/bytes read;
- index bytes per vector;
- build/rebuild time;
- insert/update latency;
- WAL/replication overhead;
- vector projection freshness lag;
- koszt modelu, tokenów i rerankera.

### Ablacje

Każdy wynik powinien obejmować:

- dense-only;
- lexical-only;
- hybrid;
- hybrid + temporal;
- hybrid + entity;
- hybrid + graph;
- hybrid + reranker;
- pełny pipeline.

Należy mierzyć zarówno candidate recall, jak i końcową odpowiedź. Wzrost raw
recall może zniknąć po rerankingu, deduplikacji albo obcięciu token budget.

## Poziom dojrzałości

### Wdrażać teraz

- hard scope/ACL filters;
- exact + lexical + dense;
- trigram dla identifiers;
- RRF;
- cross-encoder nad małą pulą;
- explicit bitemporal schema;
- canonical entities;
- parent-child mapping;
- provenance;
- MMR/dedup;
- exact fallback;
- per-lane observability;
- abstention.

### Wdrażać po lokalnej ewaluacji

- BM25 extension;
- bounded graph lane;
- query rewriting i decomposition;
- hierarchical summaries;
- temporal-aware reranker;
- learned fusion;
- SPLADE;
- ColBERT jako reranker.

### Traktować jako eksperyment

- ColBERT jako główny token-vector retriever wewnątrz SQL;
- HippoRAG/PPR jako podstawowy retriever;
- pełne Self-RAG/CRAG/FLARE loops;
- listwise reasoning-LLM reranking;
- agentic SQL jako domyślny retrieval;
- RL-trained routing i memory policies.

## Decyzja architektoniczna

### Wybrać PostgreSQL, gdy

- retrieval quality jest głównym wyróżnikiem produktu;
- deployment jest single-region lub klasyczne HA wystarcza;
- potrzebne są BM25, bogaty fuzzy search lub eksperymenty z nowymi indeksami;
- zależy nam na jednym systemie dla OLTP i wielu search lanes.

### Wybrać CockroachDB, gdy

- pamięć jest globalnym, silnie spójnym stanem;
- tenant lub użytkownik ma home region;
- disaster tolerance i locality są ważniejsze niż najbogatszy search stack;
- retrieval można oprzeć na C-SPANN + FTS + trigram + aplikacyjnym fusion.

### Rekomendacja dla Sen

Zachować:

- bazę jako źródło prawdy;
- raw sources i bitemporal records;
- retrieval indexes jako rebuildable projections;
- wspólny kontrakt candidate/provenance/fusion;
- backend-specific query planning.

PostgreSQL powinien wykorzystywać bogatszy ekosystem retrievalowy.
CockroachDB powinien wykorzystywać locality, equality-prefix C-SPANN oraz
kontrolowane, relacyjne lane'y. Zewnętrzny search projection ma sens dopiero,
gdy lokalny benchmark pokaże lukę, której nie da się zamknąć bez drugiego
systemu.

## Główne źródła

### Systemy i benchmarki

- [Hindsight, ACL 2026](https://aclanthology.org/2026.acl-demo.27/)
- [LongMemEval](https://arxiv.org/abs/2410.10813)
- [BEIR](https://arxiv.org/abs/2104.08663)

### PostgreSQL

- [pgvector](https://github.com/pgvector/pgvector)
- [PostgreSQL full-text search](https://www.postgresql.org/docs/current/textsearch-controls.html)
- [PostgreSQL pg_trgm](https://www.postgresql.org/docs/current/pgtrgm.html)
- [pg_textsearch](https://github.com/timescale/pg_textsearch)
- [ParadeDB](https://docs.paradedb.com/welcome/introduction)
- [pgvectorscale](https://github.com/timescale/pgvectorscale)
- [VectorChord](https://github.com/supervc-stack/VectorChord)

### CockroachDB

- [Vector indexes](https://www.cockroachlabs.com/docs/v26.2/vector-indexes)
- [VECTOR type](https://www.cockroachlabs.com/docs/v26.2/vector)
- [Full-text search](https://www.cockroachlabs.com/docs/v26.2/full-text-search)
- [Trigram indexes](https://www.cockroachlabs.com/docs/v26.2/trigram-indexes)
- [Table localities](https://www.cockroachlabs.com/docs/v26.2/table-localities)

### Retrieval research

- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [SPLADE v2](https://arxiv.org/abs/2109.10086)
- [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/)
- [RAPTOR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html)
- [GraphRAG](https://arxiv.org/abs/2404.16130)
- [HippoRAG 2](https://arxiv.org/abs/2502.14802)
- [HyDE](https://aclanthology.org/2023.acl-long.99/)
- [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/)
