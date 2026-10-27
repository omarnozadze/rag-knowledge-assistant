from typing import Literal

from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=5000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    # None = server default (RAG_RETRIEVAL_MODE)
    mode: Literal["hybrid", "vector"] | None = None
    # Also search with the question translated into the corpus's other languages.
    # None = server default (RAG_QUERY_TRANSLATION)
    translate: bool | None = None

class Citation(BaseModel):
    document_id: int
    filename: str
    page_number: int | None = None
    chunk_index: int
    score: float
    # Position in each search's candidate list; None = not found by that search
    vector_rank: int | None = None
    keyword_rank: int | None = None
    # The phrasing (original or a translation) that ranked this chunk best
    matched_query: str | None = None
    excerpt: str

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
