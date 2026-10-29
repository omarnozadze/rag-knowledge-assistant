# RAG Knowledge Assistant

[![tests](https://github.com/omarnozadze/rag-knowledge-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/omarnozadze/rag-knowledge-assistant/actions/workflows/tests.yml)

Production-style portfolio project for asking questions over private documents.

## Features

- Upload PDF, DOCX, TXT and Markdown files
- Extract and chunk text
- Create OpenAI embeddings
- Store chunks and embeddings in PostgreSQL with the `pgvector` extension
- Hybrid retrieval in one SQL query: vector search (HNSW) + keyword search
  (`tsvector`, GIN), merged with Reciprocal Rank Fusion
- Exact identifier matching (product codes like `ABC-10293`) that embeddings alone miss
- Generate grounded answers with the OpenAI Responses API
- Return citations to source filename and page/chunk
- Simple browser UI
- FastAPI REST endpoints

## Architecture

```text
Documents
   |
   v
Text extraction -> Chunking -> OpenAI Embeddings -> PostgreSQL + pgvector
                                          (HNSW on embedding, GIN on tsvector)
                                                              |
                                  +---------------------------+
                                  |                           |
User question -> Query embedding -> vector top 20     keyword top 20 <- question words
                                  |                           |
                                  +-- Reciprocal Rank Fusion -+
                                                |
                                                v
                                         Relevant chunks (top k)
                                                              |
                                                              v
                                                   OpenAI Responses API
                                                              |
                                                              v
                                                    Answer + citations
```

## Storage

```sql
CREATE TABLE chunks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    text TEXT NOT NULL,
    embedding vector(1536) NOT NULL
);

CREATE INDEX idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
```

Retrieval is a single query: the database ranks chunks and returns only the top `k`.

```sql
SELECT c.*, d.filename, 1 - (c.embedding <=> :query) AS score
FROM chunks c JOIN documents d ON d.id = c.document_id
ORDER BY c.embedding <=> :query
LIMIT :k;
```

Why it changed: v0.1 stored embeddings as JSON in SQLite and loaded every chunk
into Python on each question to compute cosine similarity. That costs O(n) memory
and time per question. With pgvector, the search runs inside the database and the
HNSW index keeps it fast as the corpus grows.

## Hybrid search

Semantic search finds meaning ("annual leave" ~ "vacation"), but blurs exact
identifiers: to an embedding model `ABC-10293` and `ABC-10294` look almost the same.
Keyword search is the opposite. v0.3 runs both and fuses the results:

```sql
ALTER TABLE chunks ADD COLUMN text_search tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED;
CREATE INDEX idx_chunks_text_search ON chunks USING gin (text_search);
```

- **Vector side:** top `RAG_CANDIDATES` (20) chunks by cosine distance.
- **Keyword side, question with identifiers** (tokens with a digit, 3+ chars, e.g.
  `ABC-10293`, `INV2024`): only those are matched, as exact phrases.
- **Keyword side, any other question:** BM25-style scoring that only lets in chunks with
  real word evidence, because RRF rewards *every* chunk on the keyword list:
  1. English and Georgian stopwords are removed (`app/stopwords.py`, synced into a
     `stopwords` table on startup)
  2. words found in at least half of all chunks are dropped (their BM25 IDF is ≤ 0)
  3. a chunk must match at least 2 of the remaining question words
     (`RAG_MIN_SHOULD_MATCH`, like Elasticsearch's `minimum_should_match`)
  4. matches are ranked by the summed IDF of the words they contain
- **Fusion:** Reciprocal Rank Fusion, `score = Σ 1 / (60 + rank)`. It uses ranks only,
  so cosine similarity and `ts_rank_cd` never need to be put on the same scale. A
  chunk found by both searches ranks highest.

The keyword index is per language (`app/languages.py`): English chunks use Postgres's
`english` config, which stems (`requests` → `request`), and Georgian chunks use `simple`
(lowercase only), because Postgres has no Georgian stemmer. A chunk's language is the
script with more letters, computed by Postgres in a generated column; the question's
words go through the config of the question's language. Codes like `ABC-10293` are
tokenised identically by both configs. Georgian word forms (`შვებულება` vs
`შვებულების`) still don't match on the keyword side; the vector side covers that.

Each citation reports `vector_rank` and `keyword_rank` (null = not found by that
search), which shows why a chunk was retrieved. `POST /api/ask` accepts
`"mode": "vector"` to compare against pure semantic search.

Example (from the test suite, with an embedding that cannot see digits):

| Question | `vector` top hit | `hybrid` top hit |
| --- | --- | --- |
| What is the return procedure for ABC-10293? | returns-ABC-10294 (wrong) | returns-ABC-10293 |

The schema is created on startup. The vector dimension comes from
`RAG_EMBEDDING_DIM` and must match the embedding model
(1536 for `text-embedding-3-small`, 3072 for `text-embedding-3-large`).

## Evaluation

`eval/` measures retrieval quality on a fixed, versioned test set, so every change
to chunking, embeddings or ranking can be compared with numbers instead of impressions.

- `eval/corpus/`: 7 policy documents of a fictional company (HR, remote work,
  returns, IT security, expenses in English; warehouse and onboarding in Georgian).
  Product codes like `ABC-10293` / `ABC-10294` are deliberately one digit apart.
- `eval/questions.jsonl`: 40 questions, each with the expected source document and an
  **evidence phrase** that the retrieved chunk must contain. Tags: `semantic`
  (paraphrased, little word overlap), `keyword`, `identifier`, `ka` (Georgian),
  `cross-lingual` (question and document in different languages), `multi-source`.
- `eval/questions_holdout.jsonl`: 14 more questions, written *after* the failure
  analysis below but *before* running the resulting fix, to check the fix wasn't
  tuned to the main set.
- `eval/questions_multilingual.jsonl`: 17 Georgian and cross-lingual questions (see
  "Embedding models" below).
- `eval/run_eval.py`: indexes the corpus into a separate database per embedding config
  (the app's data is never touched), runs every question in each mode and reports:
  - **hit@k**: the answer-bearing chunk is in the top k
  - **MRR**: mean of 1 / rank of the first correct chunk
  - **doc_hit@1**: the top chunk is at least from the right document
  - **latency p50/p95** of the retrieval query (embeddings are pre-computed)

A hit requires the right document *and* the evidence phrase, so "right file, wrong
section" does not count. The runner also checks up front that every evidence phrase fits
inside one chunk; a test case that can never pass fails the run instead of quietly
lowering the score.

```powershell
python -m eval.run_eval                     # vector vs hybrid, top-5
python -m eval.run_eval --questions eval/questions_holdout.jsonl
python -m eval.run_eval --embeddings fake   # offline pipeline check, no OpenAI calls
```

Embeddings are cached in `eval/.cache/`, so re-runs only pay for changed text.
Full per-question reports land in `eval/results/`.

### Results (text-embedding-3-small, top-5)

Main set, 40 questions. `hybrid v0.4` is the first hybrid version (any question word,
ranked by `ts_rank_cd`); `hybrid v0.5` is the current keyword side described above.

| mode | hit@1 | hit@3 | hit@5 | MRR | doc_hit@1 |
| --- | --- | --- | --- | --- | --- |
| vector | 55.0% | 82.5% | **90.0%** | 0.684 | 72.5% |
| hybrid v0.4 | 67.5% | 85.0% | 85.0% | 0.758 | 82.5% |
| **hybrid v0.5** | **75.0%** | **87.5%** | **90.0%** | **0.810** | **85.0%** |

Held-out set, 14 questions:

| mode | hit@1 | hit@3 | hit@5 | MRR |
| --- | --- | --- | --- | --- |
| vector | 71.4% | 85.7% | 92.9% | 0.792 |
| **hybrid v0.5** | **85.7%** | **92.9%** | 92.9% | **0.881** |

Main set by tag (hit@1):

| tag | n | vector | hybrid v0.4 | hybrid v0.5 |
| --- | --- | --- | --- | --- |
| identifier | 6 | 66.7% | 100.0% | **100.0%** |
| ka (Georgian) | 9 | 11.1% | 44.4% | **55.6%** |
| semantic | 17 | 58.8% | 70.6% | **82.4%** |
| keyword | 8 | 100.0% | 87.5% | **100.0%** |
| cross-lingual | 5 | 0.0% | 0.0% | 0.0% |

What the numbers say:

- **Hybrid v0.5 beats vector everywhere and never does worse:** +20 points hit@1 on the
  main set, +14 on the held-out set, with hit@5 equal.
- **The v0.4 regression is fixed.** v0.4 lost hit@5 (90% → 85%) because RRF gives a
  bonus to any chunk on the keyword list, and chunks sharing only filler words with the
  question ("will", "the", "company") made the list. In "Will the company pay for drinks
  at the hotel bar?" the right chunk was vector rank 1 but fell out of the top 5.
- **IDF alone was not enough.** The first attempt used only corpus IDF and scored
  *worse* (hit@1 62.5%): on 14 chunks, "will" appears in 2 chunks, so IDF thought it
  was a rare, meaningful word. Fixed stopword lists and `minimum_should_match` don't
  depend on corpus size; the IDF part becomes more useful as the corpus grows.
- **Identifiers:** exact code matching takes them from 67% to 100%.
- **Georgian:** `text-embedding-3-small` is weak on Georgian (11% hit@1); hybrid lifts
  it to 56%.
- **Cross-lingual is unsolved:** a Georgian question does not find the English policy
  and vice versa, in any mode.

Caveats: 54 questions over 14 chunks is a small test set; treat differences of one or
two questions as noise. The v0.5 fix was designed by looking at main-set failures,
which is why the held-out set exists.

### Latency

Retrieval runs at **~1 ms p50 / ~1.5 ms p95** (both modes), measured end to end from
Python. Two things got it there from ~31 ms:

- **Connection pool** (`psycopg_pool`): the app used to open a new Postgres connection
  per request, ~25-30 ms each on Docker Desktop for Windows.
- **Prepared statements:** on reused connections psycopg prepares a query after 5 runs.
  Without that, a long SQL text plus large parameters (especially multi-byte Georgian
  text) sometimes hit a ~40 ms stall in the Docker Desktop network path, even though
  Postgres itself executed the query in ~1-2 ms (`EXPLAIN ANALYZE`). The stopword list
  also moved from a per-query parameter into a table, keeping messages small.

### Embedding models: small vs large

Does a bigger embedding model fix Georgian and cross-lingual retrieval? Compared on all
three question sets pooled (71 questions), including `eval/questions_multilingual.jsonl`:
17 questions written for this comparison, before running it (9 Georgian question →
English document, 4 English → Georgian, 4 Georgian → Georgian, paraphrased).

| config | vector hit@1 | vector hit@5 | hybrid hit@1 | hybrid hit@5 | hybrid MRR |
| --- | --- | --- | --- | --- | --- |
| `text-embedding-3-small`, 1536 | 46.5% | 80.3% | 63.4% | 80.3% | 0.691 |
| `text-embedding-3-large`, 1536 | 46.5% | 78.9% | 63.4% | 78.9% | 0.685 |
| `text-embedding-3-large`, 3072 | 49.3% | 81.7% | 63.4% | 81.7% | 0.692 |

By direction (vector mode, hit@5):

| direction | n | small-1536 | large-1536 | large-3072 |
| --- | --- | --- | --- | --- |
| Georgian question → English doc | 9 | 0.0% | 11.1% | 11.1% |
| English question → Georgian doc | 4 | 100.0% | 75.0% | 100.0% |
| Georgian → Georgian | 4 | 100.0% | 100.0% | 100.0% |

What it shows:

- **The large model doesn't move the needle.** Hybrid hit@1 is identical (63.4%) for all
  three configs; the vector-only gains are 2 questions out of 71, i.e. noise. Large costs
  6.5× more per token, and at 3072 dims needs `halfvec` and twice the storage.
- **Cross-lingual retrieval fails because of language bias, not model size.** For *every*
  Georgian question, in both models, the vector top 4 were exactly the corpus's 4
  Georgian chunks, whatever the topic. The query's language outweighs its meaning, so a
  Georgian question can't reach the English policy that answers it.
- **Decision:** stay on `text-embedding-3-small`. The fix for cross-lingual questions has
  to change the query or the documents, not the model: translate the question into the
  other corpus language and search with both (multi-query), or index an English
  translation of non-English chunks.

The app supports both models: `RAG_EMBEDDING_DIM` is sent to the API as `dimensions`,
and above 2000 dims (pgvector's HNSW limit for `vector`) embeddings are stored as
`halfvec`.

```powershell
python -m eval.run_eval --model text-embedding-3-large --dim 3072 --questions eval/questions_multilingual.jsonl
python -m eval.compare    # pooled comparison of the saved reports
```

### Query translation

`RAG_QUERY_TRANSLATION=true` (or `"translate": true` in `POST /api/ask`) translates the
question into the corpus's other language with an LLM (`app/query_translation.py`):
codes, amounts and times are kept verbatim, and the language is detected from the
script (Georgian letters → `ka`). Each chunk's language is a generated column, from the
script that dominates its text.

**How the translation is used** took three attempts, each measured on all three sets:

| hybrid, hit@1 / hit@5 | main (40) | holdout (14) | multilingual (17) |
| --- | --- | --- | --- |
| no translation | 75.0% / 90.0% | 85.7% / 92.9% | 17.6% / 47.1% |
| 1. search with both phrasings, fuse lists by rank | 42.5% / 95.0% | 50.0% / 78.6% | 52.9% / 100% |
| 2. language-matched vector + keyword | 72.5% / 97.5% | 85.7% / 85.7% | 88.2% / 100% |
| **3. translation on the keyword side only** | **80.0% / 100%** | **85.7% / 92.9%** | **58.8% / 100%** |

1. **Fuse by rank:** an English question's Georgian translation ranks the Georgian chunks
   1-4 (language bias again), and RRF counts that as strongly as the English hit.
   Semantic hit@1 fell from 78% to 26%.
2. **Language-matched:** each chunk is compared only with the question in its own
   language. Better, but vector distances are still not comparable across languages:
   Georgian texts sit close together in embedding space, so the Georgian translation is
   "near" every Georgian chunk. "I'm sick today, by what time do I have to tell my
   boss?" got 4 warehouse/onboarding chunks ahead of the leave policy.
3. **Keyword side only (current):** the vector side keeps the original question over all
   chunks, exactly as without translation. Each translation feeds the keyword search for
   its own language's chunks, and candidates merge by BM25 score, which counts word
   matches and does not care about language.

Option 3 is the only one that **never pushed a correct chunk out of the top 5**: 0 of 40
main-set questions got worse, 1 of 14 holdout questions moved from rank 3 to 4, and
Georgian → English questions went from 0% to 100% hit@5. Option 2 has a better
multilingual hit@1, but at the cost of the monolingual questions.

Costs and limits:

- **Latency:** one LLM call per question, ~3-4 s with `gpt-5.5` (p50 in the eval; the
  end-to-end `/api/ask` went 3.6 s → 8.0 s). Retrieval itself stays ~1 ms. A smaller
  `RAG_TRANSLATION_MODEL` is the obvious lever.
- **Word forms:** a translation can pick a different form than the document uses.
  English is covered by stemming (below); Georgian is not.
- Off by default (`RAG_QUERY_TRANSLATION=false`): it pays off when users ask in a
  different language than the documents are written in.

```powershell
python -m eval.run_eval --questions eval/questions_multilingual.jsonl --translate
python -m eval.compare --configs text-embedding-3-small-1536,text-embedding-3-small-1536-qt
```

Translations are cached in `eval/.cache/`, so eval re-runs make no LLM calls.

### English stemming on the keyword side

v0.7 switched English chunks from the `simple` config to `english` (stemming + English
stopwords), so `requests` matches `request`. Hybrid mode, before → after:

| set | hit@1 | hit@5 | MRR |
| --- | --- | --- | --- |
| main (40) | 75.0% → 67.5% | 90.0% → 90.0% | 0.810 → 0.773 |
| holdout (14) | 85.7% → 92.9% | 92.9% → 92.9% | 0.881 → 0.929 |
| multilingual (17) | 17.6% → 17.6% | 47.1% → 47.1% | 0.254 → 0.254 |
| main, with translation | 80.0% → 75.0% | 100% → 100% | 0.881 → 0.858 |
| holdout, with translation | 85.7% → 85.7% | 92.9% → 92.9% | 0.875 → 0.893 |
| multilingual, with translation | 58.8% → 58.8% | 100% → 100% | 0.760 → 0.760 |

**On this test set stemming is neutral:** hit@5 (what the LLM sees) is unchanged
everywhere, and hit@1 moves by 1-3 questions in both directions, which is within noise.
It helps where a question uses another form of the document's word ("Can I *install*
Spotify…" 3 → 1, since the policy says "may be *installed*"). It also lets more weak
matches through: "Will the company pay for drinks…" now matches "company **pays** all
shipping costs" in the returns policy, two generic words that IDF cannot recognise as
generic on a 14-chunk corpus. Two of the three hit@1 drops are the right document with
the other chunk ranked first.

Stemming stays, because matching word forms is what a keyword index should do and the
eval shows no cost where it counts (top 5). The weak-match problem is the same one
the BM25 IDF filter addresses, and it fades as the corpus grows.

A related limit, seen in a 4-chunk demo database: when a question shares only one
corpus word with a chunk, `minimum_should_match = 2` keeps it off the keyword list even
if that word is the important one. On a realistic corpus the question usually shares
two or more words with the right chunk; on a tiny one, IDF also drops any word that
appears in half the chunks.

Next steps the eval points to: a faster translation model, then a reranker over the
top 20.

## Setup (Windows PowerShell)

Requires Python 3.11+ and Docker Desktop.

```powershell
docker compose up -d

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
notepad .env
```

Put your OpenAI API key into `.env`. PostgreSQL listens on `localhost:5434`
(user/password/db: `rag`) so it doesn't clash with a locally installed
PostgreSQL on 5432/5433.

Run:

```powershell
uvicorn app.main:app --reload --port 8020
```

Open:

`http://127.0.0.1:8020`

### Migrating v0.1 data

If you have documents indexed in the old `data/rag.db`, copy them over
(embeddings are reused, no OpenAI calls):

```powershell
python -m scripts.migrate_sqlite_to_pg
```

## Streaming answers

The web UI calls `POST /api/ask/stream`, which returns the answer as Server-Sent Events
while the model writes it (OpenAI Responses API with `stream=True`):

```text
event: citations   data: [...]               sources, sent right after retrieval
event: delta       data: {"text": "Five "}   many times, pieces of the answer
event: done        data: {}
```

If generation fails midway, an `error` event (`{"detail": ...}`) replaces `done`: the
HTTP status is already 200 at that point, so the failure has to travel inside the stream.
The internal error goes to the server log, not to the client. `POST /api/ask` still
returns the whole answer as one JSON response, which suits integrations like chat bots.

Measured against the real model (`gpt-5.5`, 3 documents indexed):

| question | sources shown | first words | full answer | `/api/ask` (nothing shown until) |
| --- | --- | --- | --- | --- |
| "How far in advance must I request annual leave?" | 0.50 s | 2.22 s | 2.87 s | 2.99 s |
| "Can a customer return product ABC-10294?" | 0.46 s | 1.69 s | 2.88 s | 5.02 s |

The total time barely changes; what changes is that the reader sees sources after half a
second and starts reading after under two seconds instead of waiting for the whole answer.
With query translation on, the first words come ~3-4 s later (the translation call).

The UI reads the stream with `fetch` (the browser's `EventSource` only supports GET).

### Rendering the answer safely

Answers are Markdown (**bold**, lists, citations like `[1]`). `app/static/markdown.js`
renders the subset answers use, and is re-run on every streamed piece so formatting
appears as the answer is written. It is XSS-safe by construction:

- `parse()` turns text into a plain tree; `render()` builds it with `createElement` and
  `createTextNode` only. No HTML string is ever parsed, so `<script>` or
  `<img onerror=…>` in an answer (or in a document it quotes) is shown as text.
- Markdown links `[text](url)` are not supported, so no URL from the model becomes an
  `href`. The only links are citations, pointing at the page's own `#source-N`.
- No library or CDN: nothing third-party runs in the page.

`tests/js/markdown.test.js` (`node --test`, also run by `pytest`) covers the parser and
checks which elements the renderer creates for hostile input; `tests/test_frontend.py`
fails if any frontend file uses `innerHTML`, `outerHTML`, `insertAdjacentHTML` or
`document.write`. Checked in a real browser too: four injection attempts, nothing executed.

## Tests

```powershell
pip install -r requirements-dev.txt
docker compose up -d
pytest
```

45 tests, ~5 s (plus 9 JavaScript tests run through Node.js, skipped if it isn't installed). They run against a separate `rag_test` database on the docker-compose
Postgres and never call OpenAI: embeddings are a deterministic fake (which, like a
real model, can't tell `ABC-10293` from `ABC-10294`), the LLM answer is stubbed, and
`OPENAI_API_KEY` is cleared so a call that slips through fails instead of costing money.
Database tests are skipped if Postgres isn't running.

**CI:** `.github/workflows/tests.yml` runs the whole suite on every push and pull
request, against the same `pgvector/pgvector:pg17` image as docker-compose, with Node.js
for the JavaScript tests. There, a missing database or Node.js fails the run instead of
skipping tests (`RAG_REQUIRE_DB`, `RAG_REQUIRE_NODE`), so a green badge means every test
actually ran.

- `tests/test_units.py`: chunking and overlap, extraction, identifier and language
  detection, embedding batching and order
- `tests/test_search.py`: the retrieval behaviours the eval measured, pinned down:
  exact codes beat near-duplicates, Georgian keyword matching, English stemming,
  `minimum_should_match`, stopwords, translation feeding only the keyword side
- `tests/test_api.py`: upload → list → ask → delete, cascade delete, file cleanup,
  rejected uploads, input validation; streaming event order, streamed text equal to the
  non-streaming answer, a failure mid-stream ending in `error` without leaking details
- `tests/test_db.py`: an old `simple` index is migrated to stemming in place;
  `init_db()` is idempotent

The tests check that each piece works as designed; `eval/` measures how well retrieval
works with real embeddings.

## API

- `GET /health`
- `GET /api/documents`
- `POST /api/documents/upload`
- `DELETE /api/documents/{document_id}`
- `POST /api/ask`
- `POST /api/ask/stream` (Server-Sent Events, see "Streaming answers")

Example question body:

```json
{
  "question": "What is the return procedure for ABC-10293?",
  "top_k": 5,
  "mode": "hybrid"
}
```

`top_k` and `mode` are optional (defaults: `RAG_TOP_K`, `RAG_RETRIEVAL_MODE`).

## Portfolio roadmap

1. ~~PostgreSQL + pgvector~~ (v0.2)
2. ~~Hybrid keyword + vector retrieval~~ (v0.3)
3. Reranking
4. User authentication and document permissions
5. Evaluation dataset and retrieval metrics (done, v0.4); answer and citation
   correctness (LLM-as-judge) still to add
6. Observability, latency and cost tracking
7. ~~Docker + CI~~ (tests run on GitHub Actions); deployment still to add
8. ~~Streaming answers~~ (v0.8)
