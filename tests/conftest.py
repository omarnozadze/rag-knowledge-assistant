"""Test setup: a separate rag_test database on the docker-compose Postgres, and no OpenAI.

Tests that touch the database need `docker compose up -d`. Embeddings are replaced by a
deterministic bag-of-words that ignores tokens containing digits, mimicking how semantic
embeddings blur exact codes (ABC-10293 ~ ABC-10294). OPENAI_API_KEY is cleared, so any
call that slips past the fakes fails loudly instead of costing money.
"""
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import psycopg
import pytest
from dotenv import load_dotenv
from psycopg.conninfo import conninfo_to_dict, make_conninfo

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Must run before any app module reads its config (app.config reads env at import)
_base = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5434/rag")
TEST_DATABASE_URL = os.getenv("RAG_TEST_DATABASE_URL", make_conninfo(_base, dbname="rag_test"))
os.environ.update({
    "DATABASE_URL": TEST_DATABASE_URL,
    "OPENAI_API_KEY": "",
    "RAG_EMBEDDING_MODEL": "text-embedding-3-small",
    "RAG_EMBEDDING_DIM": "1536",
    "RAG_RETRIEVAL_MODE": "hybrid",
    "RAG_QUERY_TRANSLATION": "false",
    "RAG_TOP_K": "5",
    "RAG_MIN_SHOULD_MATCH": "2",
})

EMBEDDING_DIM = 1536

def fake_embed_texts(texts):
    out = []
    for t in texts:
        v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        for w in re.findall(r"[^\W\d_]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % EMBEDDING_DIM] += 1
        out.append((v / (np.linalg.norm(v) or 1)).tolist())
    return out

@pytest.fixture(scope="session")
def database():
    dbname = conninfo_to_dict(TEST_DATABASE_URL)["dbname"]
    try:
        with psycopg.connect(make_conninfo(TEST_DATABASE_URL, dbname="postgres"),
                             autocommit=True, connect_timeout=3) as conn:
            if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone():
                conn.execute(f'CREATE DATABASE "{dbname}"')
    except psycopg.OperationalError as exc:
        # Locally a missing database skips these tests; CI sets RAG_REQUIRE_DB so a
        # broken database setup fails the run instead of passing with tests skipped
        if os.getenv("RAG_REQUIRE_DB"):
            raise
        pytest.skip(f"Postgres not reachable (docker compose up -d): {exc}")

    from app.db import close_pool, init_db
    init_db()
    yield
    close_pool()

@pytest.fixture
def clean_db(database):
    from app.db import connect
    with connect() as conn:
        conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")

@pytest.fixture
def fake_embeddings(monkeypatch):
    from app import ingest, retrieval
    monkeypatch.setattr(ingest, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retrieval, "embed_texts", fake_embed_texts)

@pytest.fixture
def index_text(clean_db, fake_embeddings, tmp_path):
    """Index a document from a string; returns its document_id."""
    from app.ingest import index_document

    def _index(filename, text):
        path = tmp_path / filename
        path.write_text(text, encoding="utf-8")
        return index_document(path, filename, "text/plain")["document_id"]

    return _index

@pytest.fixture
def client(clean_db, fake_embeddings, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import main

    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "answer_with_context", stub_answer)
    monkeypatch.setattr(main, "stream_answer", stub_stream_answer)
    with TestClient(main.app) as c:
        yield c

def stub_answer(question, sources):
    return f"answer from {sources[0]['filename']} [1]"

def stub_stream_answer(question, sources):
    # Word by word, like the model's deltas
    words = stub_answer(question, sources).split(" ")
    for i, word in enumerate(words):
        yield word if i == 0 else " " + word
