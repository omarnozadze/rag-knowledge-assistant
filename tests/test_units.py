"""Pure-Python units: no database, no network."""
from types import SimpleNamespace

import pytest

from app import openai_client
from app.chunking import chunk_pages
from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.extractors import extract_pages
from app.languages import TEXT_SEARCH_CONFIG, detect_language
from app.retrieval import extract_identifiers

# --- chunking ----------------------------------------------------------------------

def test_short_page_is_one_chunk_with_whitespace_collapsed():
    chunks = chunk_pages([{"page_number": 3, "text": "Leave  policy.\n\nRequest   early."}])
    assert chunks == [{"chunk_index": 0, "page_number": 3, "text": "Leave policy. Request early."}]

def test_long_page_is_split_with_overlap():
    text = "abcdefghij" * 300  # 3000 chars, no whitespace so offsets are exact
    chunks = chunk_pages([{"page_number": None, "text": text}])

    assert len(chunks) > 1
    assert all(len(c["text"]) <= CHUNK_SIZE for c in chunks)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    for a, b in zip(chunks, chunks[1:]):
        assert b["text"][:CHUNK_OVERLAP] == a["text"][-CHUNK_OVERLAP:]
    # Nothing is lost: the last chunk ends where the text ends
    assert text.endswith(chunks[-1]["text"])

def test_blank_pages_produce_no_chunks():
    assert chunk_pages([{"page_number": 1, "text": "   \n "}]) == []

# --- extraction ----------------------------------------------------------------------

def test_markdown_and_text_are_read_as_one_page(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text("# Policy\n\nშვებულება", encoding="utf-8")
    assert extract_pages(path) == [{"page_number": None, "text": "# Policy\n\nშვებულება"}]

def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_pages(path)

# --- identifiers and language ------------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    ("What is the return policy for ABC-10293?", ["ABC-10293"]),
    ("FIN-207 submission deadline", ["FIN-207"]),
    ("Invoice INV2024 and order 10293", ["INV2024", "10293"]),
    ("WH-3381 რა ვადაში უნდა შეივსოს?", ["WH-3381"]),
    ("Within 5 days?", []),  # short numbers are not identifiers
    ("How many paid vacation days do I get?", []),
])
def test_extract_identifiers(question, expected):
    assert extract_identifiers(question) == expected

@pytest.mark.parametrize("text, language", [
    ("How do I request leave?", "en"),
    ("როგორ მოვითხოვო შვებულება?", "ka"),
    ("WH-3381 ფორმა", "ka"),  # any Georgian letters -> Georgian
    ("ABC-10293", "en"),
])
def test_detect_language(text, language):
    assert detect_language(text) == language
    assert language in TEXT_SEARCH_CONFIG

# --- embeddings client -------------------------------------------------------------------

class FakeEmbeddingsAPI:
    def __init__(self):
        self.calls = []

    def create(self, model, input, encoding_format, **kwargs):
        self.calls.append({"n": len(input), **kwargs})
        items = [SimpleNamespace(index=i, embedding=[float(t)]) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(items)))  # out of order on purpose

def test_embed_texts_batches_and_keeps_order(monkeypatch):
    api = FakeEmbeddingsAPI()
    monkeypatch.setattr(openai_client, "get_client", lambda: SimpleNamespace(embeddings=api))
    monkeypatch.setattr(openai_client, "EMBEDDING_BATCH_SIZE", 2)

    result = openai_client.embed_texts(["1", "2", "3", "4", "5"])

    assert [c["n"] for c in api.calls] == [2, 2, 1]
    assert result == [[1.0], [2.0], [3.0], [4.0], [5.0]]

def test_embed_texts_requests_configured_dimensions(monkeypatch):
    api = FakeEmbeddingsAPI()
    monkeypatch.setattr(openai_client, "get_client", lambda: SimpleNamespace(embeddings=api))
    openai_client.embed_texts(["1"])
    assert api.calls[0]["dimensions"] == openai_client.EMBEDDING_DIM

# --- answer streaming ----------------------------------------------------------------------

class FakeResponsesAPI:
    def __init__(self, events):
        self.events = events
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.events)

def event(type_, **fields):
    return SimpleNamespace(type=type_, **fields)

def test_stream_answer_yields_text_deltas_only(monkeypatch):
    api = FakeResponsesAPI([
        event("response.created"),
        event("response.output_text.delta", delta="Five "),
        event("response.output_text.delta", delta="working days [1]."),
        event("response.output_text.done", text="Five working days [1]."),
        event("response.completed"),
    ])
    monkeypatch.setattr(openai_client, "get_client", lambda: SimpleNamespace(responses=api))
    sources = [{"filename": "hr.md", "page_number": 2, "text": "Requests 5 working days before."}]

    assert list(openai_client.stream_answer("How early?", sources)) == ["Five ", "working days [1]."]
    assert api.kwargs["stream"] is True
    assert "[SOURCE 1] hr.md, page 2" in api.kwargs["input"]
    assert api.kwargs["instructions"] == openai_client.ANSWER_INSTRUCTIONS

def test_stream_answer_raises_on_failed_response(monkeypatch):
    api = FakeResponsesAPI([
        event("response.output_text.delta", delta="Partial"),
        event("response.failed", response=SimpleNamespace(error="rate limited")),
    ])
    monkeypatch.setattr(openai_client, "get_client", lambda: SimpleNamespace(responses=api))
    stream = openai_client.stream_answer("q", [{"filename": "a", "text": "b"}])
    assert next(stream) == "Partial"
    with pytest.raises(RuntimeError, match="rate limited"):
        next(stream)

def test_missing_api_key_fails_loudly():
    # conftest clears OPENAI_API_KEY so tests can never reach the real API
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        openai_client.get_client()
