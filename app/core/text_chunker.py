import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.ingestion.models import DocumentSection

TABULAR_TYPES = {"csv", "xlsx"}

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")
_TRAILING_SPACES = re.compile(r" +\n")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub(" ", text)
    text = _SPACES.sub(" ", text)
    text = _TRAILING_SPACES.sub("\n", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _row_metadata(first: DocumentSection, last: DocumentSection) -> dict:
    meta = {k: v for k, v in first.metadata.items() if k != "row"}
    meta["row_start"] = first.metadata["row"]
    meta["row_end"] = last.metadata["row"]
    return meta


def _group_rows(rows: list[DocumentSection], chunk_size: int, splitter) -> list[tuple[str, dict]]:
    pieces: list[tuple[str, dict]] = []
    current: list[DocumentSection] = []

    def flush():
        if current:
            pieces.append(("\n".join(r.text for r in current), _row_metadata(current[0], current[-1])))
            current.clear()

    for row in rows:
        if len(row.text) > chunk_size:
            flush()
            meta = _row_metadata(row, row)
            pieces.extend((text, dict(meta)) for text in splitter.split_text(row.text))
            continue
        same_sheet = not current or current[0].metadata.get("sheet") == row.metadata.get("sheet")
        # +1 per row for the joining newline
        projected = sum(len(r.text) + 1 for r in current) + len(row.text)
        if current and (not same_sheet or projected > chunk_size):
            flush()
        current.append(row)
    flush()
    return pieces


def chunk_sections(
    sections: list[DocumentSection],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",
    )

    by_source: dict[str, list[DocumentSection]] = {}
    for section in sections:
        by_source.setdefault(section.metadata["source"], []).append(section)

    chunks: list[Chunk] = []
    for source, items in by_source.items():
        if items[0].metadata["file_type"] in TABULAR_TYPES:
            pieces = _group_rows(items, chunk_size, splitter)
        else:
            pieces = [
                (text, dict(section.metadata))
                for section in items
                for text in splitter.split_text(clean_text(section.text))
            ]
        pieces = [(text.strip(), meta) for text, meta in pieces if text.strip()]

        for index, (text, meta) in enumerate(pieces):
            chunk_id = f"{source}:{index}"
            chunks.append(Chunk(id=chunk_id, text=text, metadata={**meta, "chunk_index": index, "chunk_id": chunk_id}))
    return chunks
