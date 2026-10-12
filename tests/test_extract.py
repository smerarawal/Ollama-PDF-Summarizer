import io

import pytest

from pdfsum import extract

reportlab = pytest.importorskip("reportlab")
from reportlab.pdfgen import canvas  # noqa: E402


def make_pdf(page_texts):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for text in page_texts:
        if text:
            pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer


def test_extract_pages_keeps_numbers_and_empty_pages():
    pages = extract.extract_pages(make_pdf(["First page text", "", "Third page text"]))
    assert [p["page"] for p in pages] == [1, 2, 3]
    assert "First page" in pages[0]["text"]
    assert pages[1]["text"] == ""
    assert "Third page" in pages[2]["text"]


def test_looks_scanned():
    assert extract.looks_scanned([])
    assert extract.looks_scanned([{"page": 1, "text": ""}, {"page": 2, "text": "tiny"}])
    assert not extract.looks_scanned([{"page": 1, "text": "x" * 500}])
