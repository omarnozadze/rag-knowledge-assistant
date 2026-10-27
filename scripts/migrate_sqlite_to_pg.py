"""One-off migration of v0.1 data (data/rag.db) into PostgreSQL + pgvector.

Run from the project root, with the database up (docker compose up -d):

    python -m scripts.migrate_sqlite_to_pg

Embeddings are copied as-is, so no OpenAI calls are made.
"""
import json
import sqlite3

from app.config import DATA_DIR
from app.db import connect, init_db, to_vector

SQLITE_PATH = DATA_DIR / "rag.db"

def main():
    if not SQLITE_PATH.exists():
        print(f"Nothing to migrate: {SQLITE_PATH} not found.")
        return

    init_db()

    src = sqlite3.connect(SQLITE_PATH)
    src.row_factory = sqlite3.Row
    documents = src.execute("SELECT * FROM documents ORDER BY id").fetchall()

    with connect() as conn:
        existing = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        if existing:
            print(f"PostgreSQL already has {existing} documents; aborting to avoid duplicates.")
            return

        total_chunks = 0
        for doc in documents:
            new_id = conn.execute(
                "INSERT INTO documents(filename, stored_path, content_type, created_at) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (doc["filename"], doc["stored_path"], doc["content_type"], doc["created_at"]),
            ).fetchone()["id"]

            chunks = src.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (doc["id"],),
            ).fetchall()

            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO chunks(document_id, chunk_index, page_number, text, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                """, [
                    (
                        new_id,
                        c["chunk_index"],
                        c["page_number"],
                        c["text"],
                        to_vector(json.loads(c["embedding_json"])),
                    )
                    for c in chunks
                ])
            total_chunks += len(chunks)

    src.close()
    print(f"Migrated {len(documents)} documents and {total_chunks} chunks.")

if __name__ == "__main__":
    main()
