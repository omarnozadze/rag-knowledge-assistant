"""Cross-lingual retrieval by translating the question.

Embeddings rank same-language text first: in the eval, a Georgian question's vector top 4
were always the corpus's 4 Georgian chunks, whatever the topic, so it could not reach the
English policy holding the answer. Translating the question into the corpus's other
languages and searching with every version sidesteps that bias.
"""
from .config import CORPUS_LANGUAGES, TRANSLATION_MODEL
from .languages import detect_language
from .openai_client import get_client

LANGUAGE_NAMES = {"en": "English", "ka": "Georgian"}

def translate(text, target):
    response = get_client().responses.create(
        model=TRANSLATION_MODEL,
        instructions=(
            f"Translate the user's question into {LANGUAGE_NAMES[target]}. "
            "It is a search query over company documents: keep its meaning, use the "
            "wording such documents would use, and copy product codes, form numbers, "
            "amounts and times exactly. Reply with the translation only."
        ),
        input=text,
    )
    return response.output_text.strip()

def query_variants(question):
    """The question itself first, then one translation per other corpus language."""
    source = detect_language(question)
    return [question] + [translate(question, lang) for lang in CORPUS_LANGUAGES if lang != source]
