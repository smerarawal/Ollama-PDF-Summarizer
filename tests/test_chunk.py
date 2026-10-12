import pytest

from pdfsum.chunk import chunk_pages, chunk_text


def test_chunk_text_has_no_empty_chunks_and_respects_size():
    text = ("Sentence number one. " * 200).strip()
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)
    assert all(len(c) <= 300 for c in chunks)


def test_chunk_text_overlap_repeats_content():
    words = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(words, size=200, overlap=60)
    # the start of chunk n+1 appears at the end of chunk n
    first_words_of_second = chunks[1].split()[:2]
    assert all(w in chunks[0] for w in first_words_of_second)


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("hello world", size=100, overlap=100)


def test_chunk_pages_keeps_page_numbers_and_skips_empty_pages():
    pages = [
        {"page": 1, "text": "Alpha " * 100},
        {"page": 2, "text": ""},
        {"page": 3, "text": "Gamma " * 100},
    ]
    chunks = chunk_pages(pages, size=200, overlap=20)
    assert {c["page"] for c in chunks} == {1, 3}
    assert all(c["text"].strip() for c in chunks)
    assert all("Gamma" in c["text"] for c in chunks if c["page"] == 3)
