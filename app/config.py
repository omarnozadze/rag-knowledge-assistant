from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5434/rag")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gpt-5.5")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1536"))
# ~1200-char chunks are ~300-400 tokens, so 100 per request stays well under API limits
EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "100"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
# "hybrid" (vector + keyword, fused with RRF) or "vector"
RETRIEVAL_MODE = os.getenv("RAG_RETRIEVAL_MODE", "hybrid")
# Each search side returns this many candidates before fusion
CANDIDATES = int(os.getenv("RAG_CANDIDATES", "20"))
# Reciprocal Rank Fusion constant; 60 is the value from the original RRF paper
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
# Keyword side: a chunk must match at least this many non-stopword question words
MIN_SHOULD_MATCH = int(os.getenv("RAG_MIN_SHOULD_MATCH", "2"))
# Query translation: also search with the question translated into the corpus's other
# languages. Costs one LLM call per question; see "Query translation" in the README.
QUERY_TRANSLATION = os.getenv("RAG_QUERY_TRANSLATION", "false").lower() == "true"
CORPUS_LANGUAGES = os.getenv("RAG_CORPUS_LANGUAGES", "en,ka").split(",")
TRANSLATION_MODEL = os.getenv("RAG_TRANSLATION_MODEL") or CHAT_MODEL  # empty value = chat model
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
