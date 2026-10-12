"""Chunking.

Two views of the same document:
- chunk_text: whole document as one text, chunks may cross page boundaries.
  Used for summarization (fewer, fuller chunks means fewer model calls).
- chunk_pages: every page is split on its own, so each chunk has an exact page
  number. Used for retrieval and citations.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def _splitter(size: int | None, overlap: int | None) -> RecursiveCharacterTextSplitter:
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    # Tries paragraph breaks first, then lines, then spaces, then characters,
    # so chunks end at natural boundaries instead of mid-word.
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    if not text or not text.strip():
        return []
    return [c for c in _splitter(size, overlap).split_text(text) if c.strip()]


def chunk_pages(
    pages: list[dict], size: int | None = None, overlap: int | None = None
) -> list[dict]:
    """Return [{'page': 3, 'text': '...'}, ...]; empty pages are skipped."""
    splitter = _splitter(size, overlap)
    chunks = []
    for page in pages:
        if not page["text"].strip():
            continue
        for piece in splitter.split_text(page["text"]):
            if piece.strip():
                chunks.append({"page": page["page"], "text": piece})
    return chunks
