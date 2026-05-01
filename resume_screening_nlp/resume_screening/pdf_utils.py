"""Extract plain text from PDF resumes using pypdf (no system dependencies)."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Read all pages from a PDF in memory and concatenate extracted text."""
    reader = PdfReader(BytesIO(file_bytes))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n".join(parts).strip()
