"""Language handling for the en/ka corpus this project targets.

Python detects a question's language; Postgres detects a chunk's language with the
equivalent SQL expression in db.py. Each language maps to the text-search config used
on both sides, so query words and chunk words are normalised the same way.
"""
import re

GEORGIAN_LETTERS = re.compile(r"[Ⴀ-ჿᲐ-Ჿ]")

# 'english' stems ("requests" -> "request") and drops English stopwords.
# Georgian has no stemmer in Postgres, so it stays 'simple' (lowercase only).
TEXT_SEARCH_CONFIG = {"en": "english", "ka": "simple"}

def detect_language(text):
    # Script-based: enough for an English/Georgian corpus, and free. Anything without
    # Georgian letters counts as English.
    return "ka" if GEORGIAN_LETTERS.search(text) else "en"
