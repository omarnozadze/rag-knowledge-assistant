from pathlib import Path
from pypdf import PdfReader
from docx import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

def extract_pages(path: Path):
    ext = path.suffix.lower()

    if ext == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page_number": idx, "text": text})
        return pages

    if ext == ".docx":
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"page_number": None, "text": text}] if text.strip() else []

    if ext in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [{"page_number": None, "text": text}] if text.strip() else []

    raise ValueError(f"Unsupported file type: {ext}")
