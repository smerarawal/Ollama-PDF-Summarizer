import re
import uuid
import zlib

import chromadb

from pdfsum import rag

DIM = 64


def fake_embed(texts):
    """Deterministic bag-of-words vectors, so retrieval can be tested without Ollama."""
    vectors = []
    for text in texts:
        vec = [0.0] * DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            vec[zlib.crc32(word.encode()) % DIM] += 1.0
        vectors.append(vec)
    return vectors


PAGES = [
    {"page": 1, "text": "Photosynthesis converts sunlight into chemical energy inside chloroplasts."},
    {"page": 2, "text": "The treaty of Westphalia ended the Thirty Years War in 1648."},
    {"page": 3, "text": "Transistors switch electric current and form the basis of modern computers."},
]


def make_collection():
    client = chromadb.EphemeralClient()
    file_hash = uuid.uuid4().hex[:16]  # unique so tests never share a collection
    return rag.ingest(file_hash, PAGES, chunk_size=500, overlap=50, embed_fn=fake_embed, client=client), file_hash, client


def test_retrieval_returns_the_page_that_matches_the_question():
    col, _, _ = make_collection()
    hits = rag.retrieve(col, "how does photosynthesis use sunlight", k=1, embed_fn=fake_embed)
    assert hits[0]["page"] == 1
    hits = rag.retrieve(col, "the treaty that ended the war", k=1, embed_fn=fake_embed)
    assert hits[0]["page"] == 2


def test_ingest_is_idempotent():
    col, file_hash, client = make_collection()
    before = col.count()
    again = rag.ingest(file_hash, PAGES, chunk_size=500, overlap=50, embed_fn=fake_embed, client=client)
    assert again.count() == before == 3


def test_retrieve_asks_for_no_more_than_available():
    col, _, _ = make_collection()
    hits = rag.retrieve(col, "anything", k=50, embed_fn=fake_embed)
    assert len(hits) == 3


def test_prompt_labels_each_chunk_with_its_page():
    prompt = rag.build_prompt("Q?", [{"page": 7, "text": "alpha"}, {"page": 9, "text": "beta"}])
    assert "[p.7] alpha" in prompt and "[p.9] beta" in prompt and "Question: Q?" in prompt


def test_answer_with_no_hits_does_not_call_the_model():
    out = "".join(rag.answer_stream("Q?", []))
    assert "could not find" in out.lower()
