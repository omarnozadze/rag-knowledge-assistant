from openai import OpenAI
from .config import OPENAI_API_KEY, CHAT_MODEL, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_DIM

def get_client():
    if not OPENAI_API_KEY or OPENAI_API_KEY == "replace_me":
        raise RuntimeError("OPENAI_API_KEY is not configured in .env")
    return OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts):
    if not texts:
        return []

    # The embeddings endpoint caps inputs and tokens per request, so large
    # documents are sent in batches. Results keep the input order.
    client = get_client()
    # text-embedding-3-* can return shortened vectors (e.g. large at 1536 instead of
    # 3072), so the request always asks for RAG_EMBEDDING_DIM. Older models can't.
    dimensions_arg = {"dimensions": EMBEDDING_DIM} if EMBEDDING_MODEL.startswith("text-embedding-3") else {}
    embeddings = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts[start:start + EMBEDDING_BATCH_SIZE],
            encoding_format="float",
            **dimensions_arg,
        )
        embeddings.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return embeddings

ANSWER_INSTRUCTIONS = """
You are a knowledge-base assistant.
Answer only from the supplied SOURCES.
If the sources do not support the answer, clearly say that the available documents do not contain enough information.
Do not invent facts.
Use inline citations like [1], [2].
Keep the answer concise but complete.
""".strip()

def build_answer_prompt(question, sources):
    context_blocks = []
    for i, src in enumerate(sources, start=1):
        location = src["filename"]
        if src.get("page_number"):
            location += f", page {src['page_number']}"
        context_blocks.append(
            f"[SOURCE {i}] {location}\n{src['text']}"
        )

    context = "\n\n".join(context_blocks)

    return f"""QUESTION:
{question}

SOURCES:
{context}
"""

def answer_with_context(question, sources):
    response = get_client().responses.create(
        model=CHAT_MODEL,
        instructions=ANSWER_INSTRUCTIONS,
        input=build_answer_prompt(question, sources),
    )
    return response.output_text

def stream_answer(question, sources):
    """Same answer as answer_with_context, yielded as text pieces while the model writes."""
    stream = get_client().responses.create(
        model=CHAT_MODEL,
        instructions=ANSWER_INSTRUCTIONS,
        input=build_answer_prompt(question, sources),
        stream=True,
    )
    for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta
        elif event.type == "response.failed":
            raise RuntimeError(f"Answer generation failed: {event.response.error}")
        elif event.type == "error":
            raise RuntimeError(f"Answer generation failed: {event.message}")
