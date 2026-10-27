from contextlib import contextmanager

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from pgvector.utils import HalfVector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import DATABASE_URL, EMBEDDING_DIM
from .stopwords import STOPWORDS

# pgvector's HNSW index supports up to 2000 dimensions for `vector` (32-bit floats) and
# 4000 for `halfvec` (16-bit floats). text-embedding-3-large at full size is 3072, so
# larger embeddings are stored as halfvec: half the storage, negligible accuracy loss.
VECTOR_TYPE = "vector" if EMBEDDING_DIM <= 2000 else "halfvec"

def _configure(conn):
    register_vector(conn)
    conn.commit()  # register_vector runs a query; the pool wants connections idle

# Reusing connections saves the ~25-30 ms connect on every request, and lets psycopg
# turn repeated queries into prepared statements (after 5 runs per connection), so the
# long hybrid SQL is parsed once instead of being resent and re-parsed each time.
# Created on first use; a closed pool can't be reopened, so close_pool() drops it and
# the next connect() builds a fresh one (app restarts in tests, eval runs).
_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            DATABASE_URL,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
            configure=_configure,
            open=False,
        )
        _pool.open(wait=True)
    return _pool

# A chunk's language: whichever script has more letters (mirrors languages.detect_language).
# Generated columns can't reference each other, so text_search repeats this expression.
LANGUAGE_EXPR = """CASE WHEN length(regexp_replace(text, '[^ა-ჿ]', '', 'g'))
                > length(regexp_replace(text, '[^A-Za-z]', '', 'g'))
             THEN 'ka' ELSE 'en' END"""

# Keyword index, per chunk language (see languages.TEXT_SEARCH_CONFIG): English is stemmed
# so "requests" matches "request"; Georgian has no Postgres stemmer and stays 'simple'.
# Codes like ABC-10293 are tokenised the same way by both configs.
TEXT_SEARCH_EXPR = f"""CASE WHEN ({LANGUAGE_EXPR}) = 'ka'
             THEN to_tsvector('simple', text)
             ELSE to_tsvector('english', text) END"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS documents (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    content_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    text TEXT NOT NULL,
    embedding {VECTOR_TYPE}({EMBEDDING_DIM}) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- Approximate nearest-neighbour index for cosine distance (the <=> operator)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding {VECTOR_TYPE}_cosine_ops);

-- Keyword side of hybrid search
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS text_search tsvector
    GENERATED ALWAYS AS ({TEXT_SEARCH_EXPR}) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_text_search
    ON chunks USING gin (text_search);

-- Chunk language, for query translation and per-language text search
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS language TEXT
    GENERATED ALWAYS AS ({LANGUAGE_EXPR}) STORED;

-- Filled from app/stopwords.py on startup; read by the keyword side of hybrid search
CREATE TABLE IF NOT EXISTS stopwords (
    word TEXT PRIMARY KEY
);
"""

def init_db():
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(SCHEMA)

        # Databases created before per-language stemming have text_search =
        # to_tsvector('simple', text). PostgreSQL 17+ can swap a stored generated
        # column's expression in place: rows are recomputed and the GIN index rebuilt.
        current = conn.execute("""
            SELECT pg_get_expr(d.adbin, d.adrelid) FROM pg_attrdef d
            JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
            WHERE d.adrelid = 'chunks'::regclass AND a.attname = 'text_search'
        """).fetchone()[0]
        if "english" not in current:
            conn.execute(f"ALTER TABLE chunks ALTER COLUMN text_search SET EXPRESSION AS ({TEXT_SEARCH_EXPR})")

        # Code is the source of truth for the list; resync it on every startup
        with conn.transaction():
            conn.execute("DELETE FROM stopwords")
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO stopwords(word) VALUES (%s)", [(w,) for w in STOPWORDS])

        existing = conn.execute("""
            SELECT format_type(atttypid, atttypmod) FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
        """).fetchone()[0]

    expected = f"{VECTOR_TYPE}({EMBEDDING_DIM})"
    if existing != expected:
        raise RuntimeError(
            f"chunks.embedding is {existing} but RAG_EMBEDDING_DIM={EMBEDDING_DIM} needs {expected}. "
            "Re-create the table (and re-index documents) or change the embedding settings back."
        )

@contextmanager
def connect():
    # Commits on success, rolls back on error, then returns the connection to the pool
    with get_pool().connection() as conn:
        yield conn

def close_pool():
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None

def to_vector(values):
    if VECTOR_TYPE == "halfvec":
        return HalfVector(values)
    return np.asarray(values, dtype=np.float32)
