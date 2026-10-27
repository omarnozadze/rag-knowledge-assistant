from .config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_pages(pages):
    chunks = []

    for page in pages:
        text = " ".join(page["text"].split())
        if not text:
            continue

        start = 0
        index = 0

        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            part = text[start:end].strip()

            if part:
                chunks.append({
                    "chunk_index": index,
                    "page_number": page.get("page_number"),
                    "text": part,
                })

            if end >= len(text):
                break

            start = max(end - CHUNK_OVERLAP, start + 1)
            index += 1

    return chunks
