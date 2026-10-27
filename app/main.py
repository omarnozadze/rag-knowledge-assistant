from contextlib import asynccontextmanager
from pathlib import Path
import json
import logging
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import UPLOAD_DIR
from .db import close_pool, connect, init_db
from .extractors import SUPPORTED_EXTENSIONS
from .ingest import EmptyDocumentError, index_document
from .openai_client import answer_with_context, stream_answer
from .retrieval import retrieve
from .schemas import AskRequest, AskResponse, Citation

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield
    close_pool()

app = FastAPI(
    title="RAG Knowledge Assistant",
    version="0.8.0",
    description="Upload private documents and ask grounded questions with citations.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/documents")
def list_documents():
    with connect() as conn:
        rows = conn.execute("""
            SELECT d.id, d.filename, d.content_type, d.created_at,
                   COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.id DESC
        """).fetchall()
    return [dict(row) for row in rows]

# Plain `def` on purpose: extraction, OpenAI and DB calls are all blocking, so FastAPI
# runs this in its threadpool instead of stalling the event loop for other requests.
@app.post("/api/documents/upload")
def upload_document(file: UploadFile = File(...)):
    original_name = Path(file.filename or "document")
    ext = original_name.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = UPLOAD_DIR / stored_name

    with stored_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        return index_document(stored_path, original_name.name, file.content_type)
    except EmptyDocumentError as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))

@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int):
    with connect() as conn:
        row = conn.execute(
            "SELECT stored_path FROM documents WHERE id = %s",
            (document_id,),
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))

    Path(row["stored_path"]).unlink(missing_ok=True)
    return {"deleted": True, "document_id": document_id}

NO_DOCUMENTS_ANSWER = "No documents have been indexed yet."

def to_citations(sources):
    return [
        Citation(
            document_id=s["document_id"],
            filename=s["filename"],
            page_number=s["page_number"],
            chunk_index=s["chunk_index"],
            score=round(s["score"], 4),
            vector_rank=s["vector_rank"],
            keyword_rank=s["keyword_rank"],
            matched_query=s["matched_query"],
            excerpt=s["text"][:350],
        )
        for s in sources
    ]

@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest):
    sources = retrieve(body.question, body.top_k, body.mode, body.translate)

    if not sources:
        return AskResponse(answer=NO_DOCUMENTS_ANSWER, citations=[])

    answer = answer_with_context(body.question, sources)
    return AskResponse(answer=answer, citations=to_citations(sources))

def sse_event(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

@app.post("/api/ask/stream")
def ask_stream(body: AskRequest):
    """Same as /api/ask, as Server-Sent Events, so the UI can show the answer as it is
    written. Events, in order:
        citations  list of citations (sent before the model starts writing)
        delta      {"text": ...} pieces of the answer, many times
        done       {} after the last delta
    If generation fails midway, an `error` event {"detail": ...} replaces `done`; the HTTP
    status is already 200 by then, so the error has to travel inside the stream.
    """
    # Retrieval runs before the response starts, so its failures are normal HTTP errors
    sources = retrieve(body.question, body.top_k, body.mode, body.translate)
    citations = [c.model_dump() for c in to_citations(sources)]

    def events():
        yield sse_event("citations", citations)
        if not sources:
            yield sse_event("delta", {"text": NO_DOCUMENTS_ANSWER})
        else:
            try:
                for text in stream_answer(body.question, sources):
                    yield sse_event("delta", {"text": text})
            except Exception:
                logger.exception("Streaming answer failed")
                yield sse_event("error", {"detail": "Answer generation failed. Please try again."})
                return
        yield sse_event("done", {})

    # A sync generator: Starlette iterates it in a worker thread, so the blocking
    # OpenAI stream doesn't stall other requests.
    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
