from pathlib import Path

from app.core.ingestion.document_loader import load_documents
from app.core.ingestion.models import DocumentSection
from app.core.text_chunker import chunk_sections, clean_text

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_documents"


def txt_section(text, source="doc.txt"):
    return DocumentSection(text=text, metadata={"source": source, "file_type": "txt"})


def row_section(row, text, source="data.csv", file_type="csv", sheet=None):
    meta = {"source": source, "file_type": file_type, "row": row}
    if sheet:
        meta["sheet"] = sheet
    return DocumentSection(text=text, metadata=meta)


# ---------- clean_text ----------

def test_clean_text_line_endings_and_spaces():
    assert clean_text("port    8080\r\nhost\t\tlocal  \r\n") == "port 8080\nhost local"


def test_clean_text_rejoins_hyphenated_words():
    assert clean_text("the config-\nuration file") == "the configuration file"


def test_clean_text_collapses_blank_lines_keeps_paragraphs():
    assert clean_text("Para one.\n\n\n\n\nPara two.") == "Para one.\n\nPara two."


def test_clean_text_removes_control_characters():
    assert clean_text("page end\x0cnext page\x00") == "page end next page"


# ---------- text chunking ----------

def test_short_text_single_chunk_keeps_metadata():
    chunks = chunk_sections([txt_section("Short document.")], chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "Short document."
    assert chunks[0].metadata == {
        "source": "doc.txt", "file_type": "txt", "chunk_index": 0, "chunk_id": "doc.txt:0",
    }


def test_long_text_split_within_size_with_overlap():
    text = " ".join(f"Sentence number {i} describes a configuration step." for i in range(40))
    chunks = chunk_sections([txt_section(text)], chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 200 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.text[:15] in prev.text, "expected overlap between neighbouring chunks"


def test_pdf_chunks_never_span_pages():
    sections = [
        DocumentSection(text="Alpha " * 60, metadata={"source": "a.pdf", "file_type": "pdf", "page": 1}),
        DocumentSection(text="Beta " * 60, metadata={"source": "a.pdf", "file_type": "pdf", "page": 2}),
    ]
    chunks = chunk_sections(sections, chunk_size=150, chunk_overlap=20)
    for c in chunks:
        if c.metadata["page"] == 1:
            assert "Beta" not in c.text
        else:
            assert "Alpha" not in c.text
    assert {c.metadata["page"] for c in chunks} == {1, 2}


# ---------- tabular chunking ----------

def test_csv_rows_grouped_with_row_range():
    rows = [row_section(i, f"Row {i}: key: value{i}") for i in range(2, 12)]
    chunks = chunk_sections(rows, chunk_size=60, chunk_overlap=10)
    assert all(len(c.text) <= 60 for c in chunks)
    assert "row" not in chunks[0].metadata
    assert chunks[0].metadata["row_start"] == 2
    covered = []
    for c in chunks:
        lines = c.text.split("\n")
        assert len(lines) == c.metadata["row_end"] - c.metadata["row_start"] + 1
        covered.extend(range(c.metadata["row_start"], c.metadata["row_end"] + 1))
    assert covered == list(range(2, 12)), "rows must be in order, none split or lost"


def test_xlsx_sheets_not_mixed():
    rows = [
        row_section(2, "Row 2: a: 1", "b.xlsx", "xlsx", "Config"),
        row_section(3, "Row 3: a: 2", "b.xlsx", "xlsx", "Config"),
        row_section(2, "Row 2: code: E01", "b.xlsx", "xlsx", "Errors"),
    ]
    chunks = chunk_sections(rows, chunk_size=500, chunk_overlap=0)
    assert [c.metadata["sheet"] for c in chunks] == ["Config", "Errors"]
    assert chunks[0].metadata["row_start"] == 2 and chunks[0].metadata["row_end"] == 3


def test_oversized_row_is_split():
    long_row = row_section(5, "Row 5: description: " + "word " * 100)
    chunks = chunk_sections([long_row], chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)
    assert all(c.metadata["row_start"] == 5 and c.metadata["row_end"] == 5 for c in chunks)


# ---------- ids and metadata ----------

def test_chunk_ids_unique_and_deterministic():
    sections = [txt_section("word " * 300, "x.txt"), txt_section("text " * 300, "y.txt")]
    first = chunk_sections(sections, chunk_size=100, chunk_overlap=10)
    second = chunk_sections(sections, chunk_size=100, chunk_overlap=10)
    ids = [c.id for c in first]
    assert len(ids) == len(set(ids))
    assert ids == [c.id for c in second]


def test_chunk_index_sequential_per_source():
    sections = [txt_section("word " * 300, "x.txt"), txt_section("text " * 300, "y.txt")]
    chunks = chunk_sections(sections, chunk_size=100, chunk_overlap=10)
    for source in ("x.txt", "y.txt"):
        indexes = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == source]
        assert indexes == list(range(len(indexes)))


def test_metadata_values_are_chroma_compatible():
    sections = [
        txt_section("Some text here."),
        row_section(2, "Row 2: a: 1", "b.xlsx", "xlsx", "Sheet1"),
        DocumentSection(text="Page text", metadata={"source": "c.pdf", "file_type": "pdf", "page": 1}),
    ]
    for c in chunk_sections(sections):
        assert all(isinstance(v, (str, int)) for v in c.metadata.values())


def test_empty_input():
    assert chunk_sections([]) == []


# ---------- end to end ----------

def test_sample_documents_end_to_end():
    files = [(p.name, p.read_bytes()) for p in SAMPLES.glob("sample_*")]
    result = load_documents(files)
    assert result.errors == []
    chunks = chunk_sections(result.sections, chunk_size=800, chunk_overlap=120)

    csv_chunks = [c for c in chunks if c.metadata["source"] == "sample_api_endpoints.csv"]
    covered = [r for c in csv_chunks for r in range(c.metadata["row_start"], c.metadata["row_end"] + 1)]
    assert covered == list(range(2, 9))
    assert all(len(c.text) <= 800 for c in csv_chunks)

    txt_chunks = [c for c in chunks if c.metadata["source"] == "sample_install_guide.txt"]
    assert len(txt_chunks) >= 2
    assert all(len(c.text) <= 800 for c in txt_chunks)
