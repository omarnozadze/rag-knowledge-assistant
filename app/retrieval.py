import re

from .db import connect, to_vector
from .openai_client import embed_texts
from . import query_translation
from .languages import TEXT_SEARCH_CONFIG, detect_language
from .config import CANDIDATES, MIN_SHOULD_MATCH, QUERY_TRANSLATION, RETRIEVAL_MODE, RRF_K, TOP_K

# Retrieval is one SQL query built from the pieces below:
#   vector_hits  - nearest chunks by cosine distance (HNSW index)
#   keyword_hits - identifier or BM25-style keyword matches (GIN index)
#   fused        - Reciprocal Rank Fusion: score = sum of 1 / (rrf_k + rank) over both lists
# RRF only uses ranks, so the two very different score scales never need normalising.
# A chunk found by both searches gets both contributions and rises to the top.
#
# With query translation there is one *variant* of the question per corpus language, used
# on the keyword side only: each variant's keywords search chunks in its own language, and
# the candidates merge into one ranking by BM25 score (word matches, comparable across
# languages). The vector side keeps the original question over all chunks. The eval
# showed why: embeddings put same-language text close together whatever it says, so a
# Georgian translation is "near" every Georgian chunk. Fusing per-language vector lists,
# by rank or even by language-matched distance, pulled unrelated Georgian chunks to the
# top of English questions.

# %(q)b sends the query vector in binary: ~6 KB for 1536 floats instead of ~19 KB of text
VECTOR_CTES = """
vector_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY distance) AS rank
    FROM (
        SELECT id, embedding <=> %(q)b AS distance
        FROM chunks
        ORDER BY distance
        LIMIT %(n_vector)s
    ) nearest
),
"""

# Keyword side when the question names exact identifiers: match only those, each as
# a phrase ('ABC-10293' -> 'abc' <-> '-10293'), so common words can't outrank them.
# Codes are the same in every language, so this runs once over all chunks.
IDENTIFIER_KEYWORD_CTES = """
keyword_query AS (
    SELECT string_agg('(' || phraseto_tsquery('simple', t)::text || ')', ' | ')::tsquery AS query
    FROM unnest(%(identifiers)s::text[]) AS t
),
keyword_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(text_search, query) DESC) AS rank
    FROM chunks, keyword_query
    WHERE text_search @@ query
    ORDER BY ts_rank_cd(text_search, query) DESC
    LIMIT %(n_keyword)s
),
"""

# Keyword side for ordinary questions. RRF rewards every chunk that makes the keyword
# list, however weak its match, so the list must only hold chunks with real evidence:
#   1. stopwords (app/stopwords.py, synced into the stopwords table) are removed
#   2. words in at least half of all chunks are dropped too (BM25 IDF <= 0); this
#      catches domain filler like "company" once the corpus is large enough
#   3. a chunk must match at least min_should_match of the remaining words: sharing a
#      single word with the question is too weak to count
#   4. survivors are ranked by the summed BM25 IDF of the words they match, so one rare
#      word ("alcohol") outweighs several common ones
def idf_terms_ctes(i):
    return f"""
query_terms_{i} AS (
    SELECT DISTINCT lexeme, quote_literal(lexeme)::tsquery AS query
    -- Same config as chunks in this language: English words are stemmed on both sides
    FROM unnest(tsvector_to_array(to_tsvector(%(config{i})s::regconfig, %(text{i})s))) AS lexeme
    WHERE NOT EXISTS (SELECT 1 FROM stopwords s WHERE s.word = lexeme)
),
informative_terms_{i} AS (
    SELECT t.query, ln((corpus.n - df.n + 0.5) / (df.n + 0.5)) AS idf
    FROM query_terms_{i} t
    CROSS JOIN corpus
    CROSS JOIN LATERAL (
        SELECT count(*)::float AS n FROM chunks WHERE text_search @@ t.query
    ) df
    WHERE ln((corpus.n - df.n + 0.5) / (df.n + 0.5)) > 0
),
"""

def idf_branch(i, filtered):
    where = f"WHERE c.language = %(lang{i})s" if filtered else ""
    return f"""(
        SELECT c.id, SUM(t.idf) AS score
        FROM chunks c
        JOIN informative_terms_{i} t ON c.text_search @@ t.query
        {where}
        GROUP BY c.id
        -- A one-word question only needs its one word
        HAVING count(*) >= LEAST(%(min_should_match)s, (SELECT count(*) FROM informative_terms_{i}))
        ORDER BY score DESC, c.id
        LIMIT %(n_keyword)s
    )"""

def idf_keyword_ctes(n_variants, filtered):
    terms = "".join(idf_terms_ctes(i) for i in range(n_variants))
    branches = "\n        UNION ALL\n        ".join(idf_branch(i, filtered) for i in range(n_variants))
    return f"""
corpus AS (
    SELECT count(*)::float AS n FROM chunks
),
{terms}
keyword_candidates AS (
    {branches}
),
keyword_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY score DESC, id) AS rank
    FROM keyword_candidates
    ORDER BY score DESC, id
    LIMIT %(n_keyword)s
),
"""

FUSED_SELECT = """
fused AS (
    SELECT
        COALESCE(v.id, k.id) AS id,
        v.rank AS vector_rank,
        k.rank AS keyword_rank,
        COALESCE(1.0 / (%(rrf_k)s + v.rank), 0)
          + COALESCE(1.0 / (%(rrf_k)s + k.rank), 0) AS score
    FROM vector_hits v
    FULL OUTER JOIN keyword_hits k ON k.id = v.id
)
SELECT
    c.id AS chunk_id,
    c.document_id,
    d.filename,
    c.chunk_index,
    c.page_number,
    c.text,
    c.language,
    f.score,
    f.vector_rank,
    f.keyword_rank
FROM fused f
JOIN chunks c ON c.id = f.id
JOIN documents d ON d.id = c.document_id
ORDER BY f.score DESC, f.vector_rank NULLS LAST
LIMIT %(k)s
"""

def build_sql(n_variants, filtered, identifiers):
    keyword = IDENTIFIER_KEYWORD_CTES if identifiers else idf_keyword_ctes(n_variants, filtered)
    return "WITH " + VECTOR_CTES.lstrip() + keyword + FUSED_SELECT

def extract_identifiers(question):
    # Product codes, invoice numbers, versions: tokens with a digit, 3+ chars.
    # Semantic embeddings blur these (ABC-10293 ~ ABC-10294); keyword search doesn't.
    return [t for t in re.findall(r"[\w-]+", question) if len(t) >= 3 and any(ch.isdigit() for ch in t)]

def retrieve(question, top_k=None, mode=None, translate=None):
    top_k = top_k or TOP_K
    mode = mode or RETRIEVAL_MODE
    translate = QUERY_TRANSLATION if translate is None else translate
    candidates = max(CANDIDATES, top_k)

    texts = query_translation.query_variants(question) if translate else [question]
    languages = [detect_language(t) for t in texts]
    # Without translation the one variant searches chunks in every language
    filtered = translate
    identifiers = extract_identifiers(question)

    params = {
        "q": to_vector(embed_texts([question])[0]),
        "identifiers": identifiers,
        "min_should_match": MIN_SHOULD_MATCH,
        "n_vector": candidates,
        # "vector" mode = same query with the keyword side switched off
        "n_keyword": candidates if mode == "hybrid" else 0,
        "rrf_k": RRF_K,
        "k": top_k,
    }
    for i, (text, lang) in enumerate(zip(texts, languages)):
        params.update({f"text{i}": text, f"lang{i}": lang, f"config{i}": TEXT_SEARCH_CONFIG[lang]})

    with connect() as conn:
        rows = conn.execute(build_sql(len(texts), filtered, identifiers), params).fetchall()

    # The phrasing whose keywords matched the chunk: the variant in the chunk's language.
    # Vector-only matches were scored against the original question.
    query_for_language = dict(zip(languages, texts))
    return [
        {**row, "score": float(row["score"]),
         "matched_query": query_for_language.get(row["language"], question)
                          if row["keyword_rank"] else question}
        for row in rows
    ]
