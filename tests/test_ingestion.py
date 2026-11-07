from io import BytesIO

import pandas as pd
import pytest
from pypdf import PdfReader, PdfWriter

from app.core.ingestion.document_loader import load_documents, safe_filename


def make_pdf(pages_text: list[str]) -> bytes:
    """Build a minimal text PDF in memory (text must not contain parentheses)."""
    n = len(pages_text)
    page_ids = [4 + 2 * i for i in range(n)]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages_text):
        content_id = page_ids[i] + 1
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    return bytes(out)


def make_xlsx(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


# ---------- Valid files ----------

def test_txt_loads_with_metadata():
    result = load_documents([("guide.txt", b"Run setup.exe to install the service.")])
    assert result.errors == []
    assert len(result.sections) == 1
    assert "setup.exe" in result.sections[0].text
    assert result.sections[0].metadata == {"source": "guide.txt", "file_type": "txt"}


def test_txt_cp1252_decoded():
    data = "Caf\u00e9 configuration \u2013 port 8080".encode("cp1252")
    result = load_documents([("notes.txt", data)])
    assert result.errors == []
    assert "Caf\u00e9" in result.sections[0].text


def test_pdf_pages_have_page_numbers():
    pdf = make_pdf(["Authentication uses OAuth tokens", "Troubleshooting port conflicts"])
    result = load_documents([("arch.pdf", pdf)])
    assert result.errors == []
    assert [s.metadata["page"] for s in result.sections] == [1, 2]
    assert "OAuth" in result.sections[0].text
    assert "port conflicts" in result.sections[1].text
    assert result.sections[0].metadata["source"] == "arch.pdf"


def test_pdf_empty_pages_skipped():
    pdf = make_pdf(["First page text", "", "Third page text"])
    result = load_documents([("doc.pdf", pdf)])
    assert [s.metadata["page"] for s in result.sections] == [1, 3]


def test_csv_rows_include_column_names():
    data = b"endpoint,method,description\n/session,POST,Create a session\n/users,GET,List users\n"
    result = load_documents([("api.csv", data)])
    assert result.errors == []
    assert len(result.sections) == 2
    first = result.sections[0]
    assert first.text == "Row 2: endpoint: /session; method: POST; description: Create a session"
    assert first.metadata == {"source": "api.csv", "file_type": "csv", "row": 2}


def test_xlsx_multiple_sheets():
    data = make_xlsx({
        "Config": pd.DataFrame({"key": ["port", "timeout"], "value": ["8080", "30"]}),
        "Errors": pd.DataFrame({"code": ["E01"], "meaning": ["Service failed to start"]}),
    })
    result = load_documents([("settings.xlsx", data)])
    assert result.errors == []
    sheets = [s.metadata["sheet"] for s in result.sections]
    assert sheets == ["Config", "Config", "Errors"]
    assert result.sections[2].text == "Row 2: code: E01; meaning: Service failed to start"
    assert result.sections[2].metadata["row"] == 2


# ---------- Validation ----------

def test_empty_file_rejected():
    result = load_documents([("empty.txt", b"")])
    assert result.sections == []
    assert result.errors == ["empty.txt: File is empty."]


@pytest.mark.parametrize("name", ["report.docx", "virus.exe", "noextension"])
def test_unsupported_extension_rejected(name):
    result = load_documents([(name, b"some content")])
    assert result.sections == []
    assert "Unsupported file type" in result.errors[0]


def test_renamed_file_rejected():
    result = load_documents([("fake.pdf", b"just plain text, not a pdf")])
    assert "not a valid PDF" in result.errors[0]
    result = load_documents([("fake.xlsx", b"just plain text")])
    assert "not a valid XLSX" in result.errors[0]


def test_oversize_file_rejected():
    data = b"a" * (1024 * 1024 + 1)
    result = load_documents([("big.txt", data)], max_file_mb=1)
    assert "exceeds the 1 MB limit" in result.errors[0]


def test_too_many_files_rejected():
    files = [(f"f{i}.txt", b"text") for i in range(3)]
    result = load_documents(files, max_files=2)
    assert result.sections == []
    assert "Too many files" in result.errors[0]


def test_no_files():
    assert load_documents([]).errors == ["No files provided."]


def test_whitespace_only_txt_rejected():
    result = load_documents([("blank.txt", b"   \n\t ")])
    assert "no text" in result.errors[0]


def test_binary_txt_rejected():
    result = load_documents([("bin.txt", b"abc\x00\x01\x02")])
    assert "binary" in result.errors[0]


# ---------- Corrupt / problematic files ----------

def test_corrupt_pdf_rejected():
    result = load_documents([("broken.pdf", b"%PDF-1.4\n garbage without structure")])
    assert result.sections == []
    assert result.errors and result.errors[0].startswith("broken.pdf:")


def test_scanned_pdf_without_text():
    result = load_documents([("scan.pdf", make_pdf(["", ""]))])
    assert "No extractable text" in result.errors[0]


def test_encrypted_pdf_rejected():
    writer = PdfWriter(clone_from=PdfReader(BytesIO(make_pdf(["Secret text"]))))
    writer.encrypt(user_password="secret", algorithm="RC4-128")
    buffer = BytesIO()
    writer.write(buffer)
    result = load_documents([("locked.pdf", buffer.getvalue())])
    assert "password-protected" in result.errors[0]


def test_csv_header_only_rejected():
    result = load_documents([("headers.csv", b"a,b,c\n")])
    assert "no data rows" in result.errors[0]


def test_corrupt_xlsx_rejected():
    result = load_documents([("broken.xlsx", b"PK\x03\x04 not really a zip")])
    assert "corrupt" in result.errors[0]


# ---------- Mixed and security ----------

def test_partial_success_mixed_upload():
    result = load_documents([("good.txt", b"Valid content"), ("bad.exe", b"MZ")])
    assert len(result.sections) == 1
    assert len(result.errors) == 1
    assert result.errors[0].startswith("bad.exe:")


@pytest.mark.parametrize("raw, expected", [
    ("../../etc/passwd.txt", "passwd.txt"),
    ("..\\..\\Windows\\win.txt", "win.txt"),
    ("normal.csv", "normal.csv"),
])
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected


def test_path_traversal_name_in_metadata():
    result = load_documents([("../../secret/notes.txt", b"hello")])
    assert result.sections[0].metadata["source"] == "notes.txt"


def test_duplicate_names_in_one_upload_rejected():
    result = load_documents([("guide.txt", b"first"), ("GUIDE.txt", b"second"), ("dir/guide.txt", b"third")])
    assert [s.text for s in result.sections] == ["first"]
    assert len(result.errors) == 2
    assert all("Duplicate file name" in e for e in result.errors)
