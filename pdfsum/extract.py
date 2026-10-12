"""PDF text extraction that keeps page numbers, plus an optional OCR fallback."""

from __future__ import annotations

from pypdf import PdfReader


def extract_pages(source) -> list[dict]:
    """Return [{'page': 1, 'text': '...'}, ...] for a path or file-like object.

    Pages without a text layer (scans, pure images) come back with empty text
    instead of being dropped, so page numbers always match the real PDF.
    """
    reader = PdfReader(source)
    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty password; try that first.
        if not reader.decrypt(""):
            raise ValueError("This PDF is password protected.")

    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # a single broken page should not kill the whole file
            text = ""
        pages.append({"page": number, "text": text.strip()})
    return pages


def looks_scanned(pages: list[dict], min_avg_chars: int = 100) -> bool:
    """True when the PDF has (almost) no text layer, i.e. it is probably a scan."""
    if not pages:
        return True
    average = sum(len(p["text"]) for p in pages) / len(pages)
    return average < min_avg_chars


def ocr_pages(pdf_bytes: bytes, dpi: int = 200) -> list[dict]:
    """OCR every page. Optional: needs pdf2image + pytesseract and the
    Poppler and Tesseract programs installed on the machine."""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "OCR needs the packages pdf2image and pytesseract, plus the Poppler "
            "and Tesseract programs. Install them or skip OCR."
        ) from exc

    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    return [
        {"page": number, "text": pytesseract.image_to_string(image).strip()}
        for number, image in enumerate(images, start=1)
    ]
