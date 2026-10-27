from pathlib import Path

from .chunking import chunk_pages
from .db import connect, to_vector
from .extractors import extract_pages
from .openai_client import embed_texts

class EmptyDocumentError(ValueError):
    pass

def index_document(path: Path, filename: str, content_type: str | None = None):
    """Extract, chunk, embed and store one document. Shared by the upload API and the eval."""
    pages = extract_pages(path)
    chunks = chunk_pages(pages)

    if not chunks:
        raise EmptyDocumentError("No readable text found in document.")

    embeddings = embed_texts([c["text"] for c in chunks])

    with connect() as conn:
        document_id = conn.execute(
            "INSERT INTO documents(filename, stored_path, content_type) "
            "VALUES (%s, %s, %s) RETURNING id",
            (filename, str(path), content_type),
        ).fetchone()["id"]

        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO chunks(
                    document_id, chunk_index, page_number, text, embedding
                ) VALUES (%s, %s, %s, %s, %s)
            """, [
                (
                    document_id,
                    c["chunk_index"],
                    c["page_number"],
                    c["text"],
                    to_vector(embedding),
                )
                for c, embedding in zip(chunks, embeddings)
            ])

    return {
        "document_id": document_id,
        "filename": filename,
        "pages_with_text": len(pages),
        "chunks": len(chunks),
    }
