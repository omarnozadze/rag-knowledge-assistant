"""Schema setup and migrations run by init_db() on startup."""
from app.db import TEXT_SEARCH_EXPR, connect, init_db
from app.stopwords import STOPWORDS

def text_search_expression():
    with connect() as conn:
        return conn.execute("""
            SELECT pg_get_expr(d.adbin, d.adrelid) AS e FROM pg_attrdef d
            JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum
            WHERE d.adrelid = 'chunks'::regclass AND a.attname = 'text_search'
        """).fetchone()["e"]

def test_old_simple_index_is_migrated_to_stemming(index_text):
    index_text("leave.txt", "Leave requests are submitted early.")
    # Recreate the pre-v0.7 schema: one 'simple' config for every chunk
    with connect() as conn:
        conn.execute("ALTER TABLE chunks ALTER COLUMN text_search SET EXPRESSION AS (to_tsvector('simple', text))")
    assert "english" not in text_search_expression()

    try:
        init_db()
        assert "english" in text_search_expression()
        with connect() as conn:
            stored = conn.execute("SELECT text_search::text AS t FROM chunks").fetchone()["t"]
        assert "'request'" in stored and "'requests'" not in stored  # existing rows recomputed
    finally:
        # Leave the session database in the current schema even if the test fails
        with connect() as conn:
            conn.execute(f"ALTER TABLE chunks ALTER COLUMN text_search SET EXPRESSION AS ({TEXT_SEARCH_EXPR})")

def test_init_db_is_idempotent_and_syncs_stopwords(database):
    init_db()
    init_db()
    with connect() as conn:
        n = conn.execute("SELECT count(*) AS n FROM stopwords").fetchone()["n"]
    assert n == len(STOPWORDS)
