"""Retrieval evaluation: vector vs hybrid on a fixed corpus and question set.

Run from the project root, with the database up (docker compose up -d):

    python -m eval.run_eval
    python -m eval.run_eval --modes vector,hybrid --k 5
    python -m eval.run_eval --model text-embedding-3-large --dim 3072
    python -m eval.run_eval --embeddings fake     # offline smoke test, no OpenAI calls

Indexes eval/corpus into a separate database per embedding config (e.g.
rag_eval_text_embedding_3_small_1536) so the app's own data is never touched and
configs with different vector sizes don't collide, then asks every question in each mode.

A question counts as a hit at rank r when the r-th retrieved chunk comes from an
expected source document AND contains that source's evidence phrase, i.e. the chunk
really holds the answer, not just the right file.

Embeddings are cached in eval/.cache, so re-runs only pay for new or changed text.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from psycopg.conninfo import conninfo_to_dict, make_conninfo

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = EVAL_DIR.parent
CORPUS_DIR = EVAL_DIR / "corpus"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
CACHE_DIR = EVAL_DIR / ".cache"
RESULTS_DIR = EVAL_DIR / "results"

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", default="vector,hybrid")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embeddings", choices=["openai", "fake"], default="openai")
    parser.add_argument("--model", help="embedding model (default: RAG_EMBEDDING_MODEL)")
    parser.add_argument("--dim", type=int, help="embedding dimensions (default: RAG_EMBEDDING_DIM)")
    parser.add_argument("--questions", default=str(QUESTIONS_PATH),
                        help="question set, e.g. eval/questions_holdout.jsonl")
    parser.add_argument("--translate", action="store_true",
                        help="query translation: also search with the question in the corpus's other languages")
    return parser

# --- Configure the app before any app module reads its config ---------------------

ARGS = build_parser().parse_args()
load_dotenv(BASE_DIR / ".env")
if ARGS.model:
    os.environ["RAG_EMBEDDING_MODEL"] = ARGS.model
if ARGS.dim:
    os.environ["RAG_EMBEDDING_DIM"] = str(ARGS.dim)

_model = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
_dim = os.getenv("RAG_EMBEDDING_DIM", "1536")
EMBEDDING_CONFIG = f"{_model}-{_dim}" if ARGS.embeddings == "openai" else f"fake-{_dim}"
# Results are named per retrieval config; the index only depends on the embeddings
CONFIG_NAME = EMBEDDING_CONFIG + ("-qt" if ARGS.translate else "")
_app_db = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5434/rag")
EVAL_DATABASE_URL = os.getenv(
    "RAG_EVAL_DATABASE_URL",
    make_conninfo(_app_db, dbname="rag_eval_" + re.sub(r"\W", "_", EMBEDDING_CONFIG)),
)
os.environ["DATABASE_URL"] = EVAL_DATABASE_URL

from app import ingest, query_translation, retrieval  # noqa: E402
from app.config import EMBEDDING_DIM, EMBEDDING_MODEL, TRANSLATION_MODEL  # noqa: E402
from app.db import VECTOR_TYPE, close_pool, connect, init_db  # noqa: E402
from app.openai_client import embed_texts as openai_embed_texts  # noqa: E402

def ensure_eval_database():
    params = conninfo_to_dict(EVAL_DATABASE_URL)
    dbname = params["dbname"]
    with psycopg.connect(make_conninfo(EVAL_DATABASE_URL, dbname="postgres"), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{dbname}"')

# --- Embeddings ------------------------------------------------------------------

def fake_embed_texts(texts):
    # Bag-of-words hashing: only for checking the pipeline offline. It behaves like
    # keyword search, so its vector-vs-hybrid numbers say nothing about real models.
    out = []
    for t in texts:
        v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        for w in re.findall(r"\w+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % EMBEDDING_DIM] += 1
        out.append((v / (np.linalg.norm(v) or 1)).tolist())
    return out

class CachedEmbedder:
    def __init__(self, backend, name):
        self.backend = backend
        self.path = CACHE_DIR / f"embeddings-{name}.json"
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.api_texts = 0

    @staticmethod
    def key(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def __call__(self, texts):
        missing = list({self.key(t): t for t in texts if self.key(t) not in self.cache}.values())
        if missing:
            for text, vec in zip(missing, self.backend(missing)):
                self.cache[self.key(text)] = vec
            self.api_texts += len(missing)
            CACHE_DIR.mkdir(exist_ok=True)
            self.path.write_text(json.dumps(self.cache), encoding="utf-8")
        return [self.cache[self.key(t)] for t in texts]

class CachedTranslator:
    """Caches LLM translations so re-runs are free and see the exact same queries."""

    def __init__(self, backend, name):
        self.backend = backend
        self.path = CACHE_DIR / f"translations-{name}.json"
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.api_latencies_ms = []

    def __call__(self, text, target):
        key = f"{target}:{text}"
        if key not in self.cache:
            t0 = time.perf_counter()
            self.cache[key] = self.backend(text, target)
            self.api_latencies_ms.append((time.perf_counter() - t0) * 1000)
            CACHE_DIR.mkdir(exist_ok=True)
            self.path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return self.cache[key]

# --- Eval ------------------------------------------------------------------------

def normalize(text):
    # Same whitespace collapsing as the chunker
    return " ".join(text.split())

def load_questions(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def index_corpus():
    with connect() as conn:
        conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")
    files = sorted(p for p in CORPUS_DIR.iterdir() if p.is_file())
    for path in files:
        info = ingest.index_document(path, path.name, "text/markdown")
        print(f"  indexed {path.name}: {info['chunks']} chunks")
    return len(files)

def check_evidence(questions):
    """Every evidence phrase must sit fully inside at least one chunk of its source,
    otherwise the question can never score a hit and the eval is broken, not the retriever."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT d.filename, c.text FROM chunks c JOIN documents d ON d.id = c.document_id"
        ).fetchall()
    chunks_by_file = defaultdict(list)
    for r in rows:
        chunks_by_file[r["filename"]].append(r["text"])

    problems = []
    for q in questions:
        for exp in q["expected"]:
            evidence = normalize(exp["evidence"])
            if not any(evidence in t for t in chunks_by_file.get(exp["source"], [])):
                problems.append(f"{q['id']}: '{exp['evidence']}' not found in a single chunk of {exp['source']}")
    return problems

def first_hit_rank(question, results):
    for rank, r in enumerate(results, start=1):
        for exp in question["expected"]:
            if r["filename"] == exp["source"] and normalize(exp["evidence"]) in r["text"]:
                return rank
    return None

def first_doc_rank(question, results):
    sources = {exp["source"] for exp in question["expected"]}
    for rank, r in enumerate(results, start=1):
        if r["filename"] in sources:
            return rank
    return None

def summarize(records, k):
    n = len(records)
    ranks = [r["hit_rank"] for r in records]
    latencies = sorted(r["latency_ms"] for r in records)
    return {
        "n": n,
        "hit@1": sum(1 for x in ranks if x == 1) / n,
        "hit@3": sum(1 for x in ranks if x and x <= 3) / n,
        f"hit@{k}": sum(1 for x in ranks if x) / n,
        "mrr": sum(1 / x for x in ranks if x) / n,
        "doc_hit@1": sum(1 for r in records if r["doc_rank"] == 1) / n,
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": latencies[min(n - 1, int(round(0.95 * (n - 1))))],
    }

def fmt_table(rows, columns):
    widths = [max(len(c), *(len(str(r[i])) for r in rows)) for i, c in enumerate(columns)]
    line = lambda vals: "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))
    return "\n".join([line(columns), line("-" * w for w in widths), *(line(r) for r in rows)])

def main():
    args = ARGS
    questions_path = Path(args.questions)
    set_name = questions_path.stem
    modes = args.modes.split(",")

    backend = openai_embed_texts if args.embeddings == "openai" else fake_embed_texts
    embedder = CachedEmbedder(backend, EMBEDDING_CONFIG)
    ingest.embed_texts = embedder
    retrieval.embed_texts = embedder
    translator = CachedTranslator(query_translation.translate, TRANSLATION_MODEL)
    query_translation.translate = translator

    print(f"Eval database: {conninfo_to_dict(EVAL_DATABASE_URL)['dbname']}")
    print(f"Embeddings: {args.embeddings} "
          f"({EMBEDDING_MODEL if args.embeddings == 'openai' else 'bag-of-words'}, "
          f"{EMBEDDING_DIM} dims, stored as {VECTOR_TYPE})")
    ensure_eval_database()
    init_db()

    questions = load_questions(questions_path)
    print(f"Question set: {set_name}")
    print(f"Indexing corpus ({CORPUS_DIR.relative_to(BASE_DIR)}):")
    n_docs = index_corpus()

    problems = check_evidence(questions)
    if problems:
        print("\nBroken test cases:")
        print("\n".join("  " + p for p in problems))
        sys.exit(1)

    # Translate and embed all questions up front so per-question latency below is
    # retrieval only; the translation call's own latency is reported separately.
    variants = {
        q["id"]: query_translation.query_variants(q["question"]) if args.translate else [q["question"]]
        for q in questions
    }
    embedder([q["question"] for q in questions])  # translations only feed keyword search
    print(f"Embedding API calls this run: {embedder.api_texts} texts (rest from cache)")
    if args.translate:
        lat = translator.api_latencies_ms
        print(f"Translation ({TRANSLATION_MODEL}): {len(lat)} API calls this run"
              + (f", p50 {statistics.median(lat):.0f} ms, max {max(lat):.0f} ms" if lat else " (all cached)"))
    print()

    records = {mode: [] for mode in modes}
    for mode in modes:
        for q in questions:
            t0 = time.perf_counter()
            results = retrieval.retrieve(q["question"], top_k=args.k, mode=mode, translate=args.translate)
            latency_ms = (time.perf_counter() - t0) * 1000
            records[mode].append({
                "id": q["id"],
                "question": q["question"],
                "variants": variants[q["id"]],
                "tags": q["tags"],
                "hit_rank": first_hit_rank(q, results),
                "doc_rank": first_doc_rank(q, results),
                "latency_ms": round(latency_ms, 2),
                "retrieved": [
                    {"filename": r["filename"], "chunk_index": r["chunk_index"],
                     "score": round(r["score"], 4),
                     "vector_rank": r["vector_rank"], "keyword_rank": r["keyword_rank"],
                     "matched_query": r["matched_query"]}
                    for r in results
                ],
            })

    overall = {mode: summarize(records[mode], args.k) for mode in modes}
    tags = sorted({t for q in questions for t in q["tags"]})
    by_tag = {
        mode: {tag: summarize([r for r in records[mode] if tag in r["tags"]], args.k) for tag in tags}
        for mode in modes
    }

    pct = lambda x: f"{x * 100:.1f}%"
    print(f"Overall ({len(questions)} questions, {n_docs} documents, top-{args.k})")
    print(fmt_table(
        [[m, pct(s["hit@1"]), pct(s["hit@3"]), pct(s[f"hit@{args.k}"]), f"{s['mrr']:.3f}",
          pct(s["doc_hit@1"]), f"{s['latency_p50_ms']:.1f}", f"{s['latency_p95_ms']:.1f}"]
         for m, s in overall.items()],
        ["mode", "hit@1", "hit@3", f"hit@{args.k}", "MRR", "doc_hit@1", "p50 ms", "p95 ms"],
    ))

    print("\nBy tag (hit@1 / MRR)")
    print(fmt_table(
        [[tag, str(by_tag[modes[0]][tag]["n"]),
          *(f"{pct(by_tag[m][tag]['hit@1'])} / {by_tag[m][tag]['mrr']:.3f}" for m in modes)]
         for tag in tags],
        ["tag", "n", *modes],
    ))

    misses = [
        (q["id"], q["question"], {m: next(r["hit_rank"] for r in records[m] if r["id"] == q["id"]) for m in modes})
        for q in questions
    ]
    misses = [m for m in misses if any(rank != 1 for rank in m[2].values())]
    if misses:
        print("\nNot at rank 1 in some mode (rank per mode, - = not in top-k)")
        for qid, question, ranks in misses:
            print(f"  {qid}  " + "  ".join(f"{m}={ranks[m] or '-'}" for m in modes) + f"  {question}")

    RESULTS_DIR.mkdir(exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question_set": set_name,
        "embeddings": args.embeddings,
        "embedding_model": EMBEDDING_MODEL if args.embeddings == "openai" else "fake-bag-of-words",
        "embedding_dim": EMBEDDING_DIM,
        "config": CONFIG_NAME,
        "query_translation": args.translate,
        "translation_model": TRANSLATION_MODEL if args.translate else None,
        "k": args.k,
        "documents": n_docs,
        "questions": len(questions),
        "overall": overall,
        "by_tag": by_tag,
        "per_question": records,
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"{set_name}-{CONFIG_NAME}"
    for name in [f"{stamp}-{suffix}.json", f"latest-{suffix}.json"]:
        (RESULTS_DIR / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report: {(RESULTS_DIR / f'latest-{suffix}.json').relative_to(BASE_DIR)}")

if __name__ == "__main__":
    try:
        main()
    finally:
        close_pool()
