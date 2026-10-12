"""Ask questions about one PDF and get answers that cite page numbers.

Pipeline: chunk per page -> embed with nomic-embed-text -> store in ChromaDB
(one collection per file and chunk size) -> embed the question -> take the
nearest chunks -> answer only from them, citing [p.N].
"""

from __future__ import annotations

from typing import Callable

import chromadb
import ollama

from . import config, llm
from .chunk import chunk_pages

EmbedFn = Callable[[list[str]], list[list[float]]]

QA = (
    "Answer the question using only the excerpts below. After each claim, cite the "
    "page like [p.12]. If the excerpts do not contain the answer, say you cannot "
    "find it in the document.\n\nExcerpts:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"
)

_client = None


def get_client():
    """Persistent on-disk Chroma client, created on first use."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return _client


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    return ollama.embed(model=model or config.EMBED_MODEL, input=texts).embeddings


def collection_name(file_hash_: str, chunk_size: int) -> str:
    return f"pdf_{file_hash_}_{chunk_size}"


def ingest(
    file_hash_: str,
    pages: list[dict],
    chunk_size: int | None = None,
    overlap: int | None = None,
    embed_fn: EmbedFn = embed,
    client=None,
):
    """Index a PDF once; later calls with the same file and chunk size reuse it."""
    client = client or get_client()
    size = chunk_size or config.CHUNK_SIZE
    # Cosine distance suits text embeddings. Chroma's default is squared L2.
    collection = client.get_or_create_collection(
        collection_name(file_hash_, size), metadata={"hnsw:space": "cosine"}
    )
    if collection.count() > 0:
        return collection

    chunks = chunk_pages(pages, size, overlap)
    batch = config.EMBED_BATCH
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        vectors = embed_fn([c["text"] for c in part])
        collection.add(
            ids=[f"{file_hash_}-{start + i}" for i in range(len(part))],
            documents=[c["text"] for c in part],
            embeddings=vectors,
            metadatas=[{"page": c["page"]} for c in part],
        )
    return collection


def retrieve(collection, question: str, k: int | None = None, embed_fn: EmbedFn = embed) -> list[dict]:
    """Return the k most relevant chunks as [{'page', 'text', 'distance'}]."""
    available = collection.count()
    if available == 0:
        return []
    n_results = min(k or config.TOP_K, available)
    query_vector = embed_fn([question])[0]
    result = collection.query(query_embeddings=[query_vector], n_results=n_results)
    return [
        {"page": meta["page"], "text": doc, "distance": dist}
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def build_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(f"[p.{h['page']}] {h['text']}" for h in hits)
    return QA.format(ctx=context, q=question)


def answer_stream(question: str, hits: list[dict], model: str | None = None):
    """Yield the answer piece by piece."""
    if not hits:
        yield "I could not find any text in this document to answer from."
        return
    yield from llm.stream_chat(build_prompt(question, hits), model)
