"""Retrieval against real Postgres + pgvector, with fake embeddings (see conftest).

Each test pins down a behaviour the eval relies on. Keyword matching uses BM25-style
IDF, which drops words found in half the chunks or more, so most tests index a few
unrelated documents to give the corpus realistic word statistics.
"""
import pytest

from app import query_translation
from app.db import connect
from app.retrieval import retrieve

DISTRACTORS = {
    "it-security.txt": "Passwords must be at least fourteen characters. Multi-factor authentication "
                       "is mandatory for email and the VPN. Report phishing to the security team.",
    "expenses.txt": "Travel expenses are reimbursed with an itemized receipt. The daily allowance "
                    "abroad is fifty euro. Alcohol and minibar charges are not reimbursed.",
    "remote-work.txt": "Office employees may work remotely two days per week. Core hours are from "
                       "ten to four. The company provides a laptop and a headset.",
    "onboarding.txt": "New employees meet their mentor weekly during the first three months. "
                      "Mandatory trainings cover information security and workplace safety.",
}

@pytest.fixture
def corpus(index_text):
    for name, text in DISTRACTORS.items():
        index_text(name, text)
    return index_text

def top(results):
    return results[0]["filename"]

def by_file(results, filename):
    return next(r for r in results if r["filename"] == filename)

def test_exact_identifier_beats_near_duplicate(corpus):
    # The fake embedding can't see digits, like a real model blurring 10293 vs 10294
    corpus("returns-10293.txt", "Return procedure for product ABC-10293. Inspect the packaging and refund within five days.")
    corpus("returns-10294.txt", "Return procedure for product ABC-10294. Return procedure: inspect the packaging, return procedure refund.")
    question = "What is the return procedure for ABC-10293?"

    hybrid = retrieve(question, mode="hybrid")
    assert top(hybrid) == "returns-10293.txt"
    assert by_file(hybrid, "returns-10293.txt")["keyword_rank"] == 1

    vector = retrieve(question, mode="vector")
    assert top(vector) == "returns-10294.txt"  # pure semantic search picks the wrong code
    assert all(r["keyword_rank"] is None for r in vector)

def test_georgian_keyword_match(corpus):
    corpus("warehouse-ka.txt", "დაზიანებული საქონელი ინახება წითელი ეტიკეტით მონიშნულ ზონაში.")
    results = retrieve("სად ინახება დაზიანებული საქონელი?")
    assert top(results) == "warehouse-ka.txt"
    assert results[0]["keyword_rank"] == 1
    assert results[0]["language"] == "ka"

def test_english_stemming_matches_other_word_forms(corpus):
    corpus("leave.txt", "Leave requests are submitted through the HR portal and approved by managers.")
    # "request ... submit ... approve" vs "requests ... submitted ... approved"
    result = by_file(retrieve("Who approves a leave request I submit?"), "leave.txt")
    assert result["keyword_rank"] is not None
    assert result["language"] == "en"

def test_single_shared_word_is_not_keyword_evidence(corpus):
    corpus("kettle.txt", "The electric kettle must be returned with its seal unbroken.")
    # Only "kettle" is shared: one word is too weak (minimum_should_match = 2)
    weak = by_file(retrieve("Is the kettle dishwasher safe?"), "kettle.txt")
    assert weak["keyword_rank"] is None
    # Two shared informative words qualify
    strong = by_file(retrieve("Must the kettle seal be unbroken?"), "kettle.txt")
    assert strong["keyword_rank"] is not None

def test_stopwords_alone_never_match(corpus):
    results = retrieve("What is the and of the for?")
    assert all(r["keyword_rank"] is None for r in results)

def test_top_k_is_respected(corpus):
    assert len(retrieve("company policy", top_k=2)) == 2

def test_translation_feeds_keyword_side_in_the_document_language(corpus, monkeypatch):
    corpus("parental.txt", "Parental leave lasts one hundred eighty three days and must be announced to HR early.")
    question = "რამდენ ხანს გრძელდება მშობლის შვებულება?"
    translation = "How long does parental leave last?"
    monkeypatch.setattr(query_translation, "translate", lambda text, target: translation)

    without = by_file(retrieve(question, translate=False), "parental.txt")
    assert without["keyword_rank"] is None  # Georgian words can't match an English chunk

    with_translation = by_file(retrieve(question, translate=True), "parental.txt")
    assert with_translation["keyword_rank"] == 1
    assert with_translation["matched_query"] == translation
    assert with_translation["vector_rank"] == without["vector_rank"]  # vector side unchanged

def test_chunk_language_is_generated_from_its_script(corpus):
    corpus("mixed-ka.txt", "საწყობი WH-3381 ფორმა ინვენტარიზაცია კვარტალში ერთხელ")
    with connect() as conn:
        rows = conn.execute("""
            SELECT d.filename, c.language FROM chunks c JOIN documents d ON d.id = c.document_id
        """).fetchall()
    languages = {r["filename"]: r["language"] for r in rows}
    assert languages["mixed-ka.txt"] == "ka"
    assert languages["expenses.txt"] == "en"
