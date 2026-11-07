from io import BytesIO

from pypdf import PdfReader

from app.core.ingestion.models import DocumentLoadError, DocumentSection


def load_pdf(filename: str, data: bytes) -> list[DocumentSection]:
    try:
        reader = PdfReader(BytesIO(data))
    except Exception as exc:
        raise DocumentLoadError("PDF is corrupt or cannot be read.") from exc

    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception:
            decrypted = 0
        if not decrypted:
            raise DocumentLoadError("PDF is password-protected.")

    try:
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise DocumentLoadError("PDF is corrupt or cannot be read.") from exc

    sections = [
        DocumentSection(text=text, metadata={"source": filename, "file_type": "pdf", "page": number})
        for number, text in enumerate(pages, start=1)
        if text.strip()
    ]
    if not sections:
        raise DocumentLoadError("No extractable text: the PDF may be scanned or image-only.")
    return sections
