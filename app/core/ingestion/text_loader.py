from app.core.ingestion.models import DocumentLoadError, DocumentSection


def decode_text(data: bytes) -> str:
    if b"\x00" in data:
        raise DocumentLoadError("File appears to be binary, not plain text.")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError("Unsupported text encoding (expected UTF-8 or Windows-1252).")


def load_text(filename: str, data: bytes) -> list[DocumentSection]:
    text = decode_text(data)
    if not text.strip():
        raise DocumentLoadError("File contains no text.")
    return [DocumentSection(text=text, metadata={"source": filename, "file_type": "txt"})]
