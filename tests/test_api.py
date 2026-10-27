"""HTTP API end to end: FastAPI + real Postgres, fake embeddings and a stub LLM answer."""
import json

def upload(client, filename, content):
    return client.post("/api/documents/upload", files={"file": (filename, content, "text/plain")})

def uploaded_files(client):
    from app import main
    return sorted(p.name for p in main.UPLOAD_DIR.iterdir()) if main.UPLOAD_DIR.exists() else []

def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}

def test_upload_list_ask_delete(client):
    r = upload(client, "returns.txt", b"Customers may return products within fourteen days with a receipt.")
    assert r.status_code == 200
    doc = r.json()
    assert doc["filename"] == "returns.txt" and doc["chunks"] == 1
    assert len(uploaded_files(client)) == 1

    listed = client.get("/api/documents").json()
    assert [(d["id"], d["filename"], d["chunk_count"]) for d in listed] == [(doc["document_id"], "returns.txt", 1)]

    r = client.post("/api/ask", json={"question": "How long can customers return products?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "answer from returns.txt [1]"
    citation = body["citations"][0]
    assert citation["filename"] == "returns.txt"
    assert citation["excerpt"].startswith("Customers may return")
    assert citation["vector_rank"] == 1
    assert citation["matched_query"] == "How long can customers return products?"

    r = client.delete(f"/api/documents/{doc['document_id']}")
    assert r.json() == {"deleted": True, "document_id": doc["document_id"]}
    assert client.get("/api/documents").json() == []
    assert uploaded_files(client) == []  # stored file removed too
    assert client.delete(f"/api/documents/{doc['document_id']}").status_code == 404

def test_delete_cascades_to_chunks(client):
    from app.db import connect
    doc_id = upload(client, "long.txt", ("Policy sentence number one. " * 200).encode()).json()["document_id"]
    with connect() as conn:
        assert conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] > 1
    client.delete(f"/api/documents/{doc_id}")
    with connect() as conn:
        assert conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] == 0

def test_unsupported_file_type_is_rejected(client):
    r = upload(client, "data.csv", b"a,b")
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["detail"]
    assert uploaded_files(client) == []

def test_empty_document_is_rejected_and_not_kept(client):
    r = upload(client, "empty.txt", b"   \n  ")
    assert r.status_code == 400
    assert r.json()["detail"] == "No readable text found in document."
    assert uploaded_files(client) == []

def test_ask_without_documents(client):
    body = client.post("/api/ask", json={"question": "Anything indexed?"}).json()
    assert body == {"answer": "No documents have been indexed yet.", "citations": []}

# --- streaming -------------------------------------------------------------------------

def parse_sse(text):
    """Server-Sent Events -> [(event, data)], the same way the UI parses them."""
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events

def ask_stream(client, question, **extra):
    r = client.post("/api/ask/stream", json={"question": question, **extra})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    return parse_sse(r.text)

def test_stream_sends_citations_then_deltas_then_done(client):
    upload(client, "returns.txt", b"Customers may return products within fourteen days with a receipt.")

    events = ask_stream(client, "How long can customers return products?")
    names = [e for e, _ in events]

    assert names[0] == "citations" and names[-1] == "done"
    assert set(names[1:-1]) == {"delta"} and len(names) > 3  # several pieces, not one blob
    citations = events[0][1]
    assert citations[0]["filename"] == "returns.txt"
    assert "".join(d["text"] for e, d in events if e == "delta") == "answer from returns.txt [1]"

def test_stream_matches_non_streaming_answer(client):
    upload(client, "returns.txt", b"Customers may return products within fourteen days with a receipt.")
    question = "How long can customers return products?"

    plain = client.post("/api/ask", json={"question": question}).json()
    events = ask_stream(client, question)

    assert "".join(d["text"] for e, d in events if e == "delta") == plain["answer"]
    assert events[0][1] == plain["citations"]

def test_stream_without_documents(client):
    events = ask_stream(client, "Anything indexed?")
    assert events == [("citations", []), ("delta", {"text": "No documents have been indexed yet."}), ("done", {})]

def test_stream_reports_failure_inside_the_stream(client, monkeypatch):
    from app import main

    def failing_stream(question, sources):
        yield "Partial "
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(main, "stream_answer", failing_stream)
    upload(client, "returns.txt", b"Customers may return products within fourteen days with a receipt.")

    events = ask_stream(client, "How long can customers return products?")
    assert [e for e, _ in events] == ["citations", "delta", "error"]  # no "done"
    assert events[-1][1] == {"detail": "Answer generation failed. Please try again."}
    assert "model unavailable" not in str(events)  # internal errors stay in the server log

def test_stream_validates_input(client):
    assert client.post("/api/ask/stream", json={"question": "x"}).status_code == 422

def test_ask_validates_input(client):
    assert client.post("/api/ask", json={"question": "x"}).status_code == 422
    assert client.post("/api/ask", json={"question": "valid question", "top_k": 50}).status_code == 422
    assert client.post("/api/ask", json={"question": "valid question", "mode": "magic"}).status_code == 422
