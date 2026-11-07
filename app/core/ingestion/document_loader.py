from pathlib import PurePosixPath

from app.core.config import settings
from app.core.ingestion.csv_loader import load_csv
from app.core.ingestion.excel_loader import load_excel
from app.core.ingestion.models import DocumentLoadError, IngestionResult
from app.core.ingestion.pdf_loader import load_pdf
from app.core.ingestion.text_loader import load_text

LOADERS = {".pdf": load_pdf, ".txt": load_text, ".csv": load_csv, ".xlsx": load_excel}


def safe_filename(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).name or "unnamed"


def validate_file(filename: str, data: bytes, max_file_mb: int) -> str:
    ext = PurePosixPath(filename).suffix.lower()
    if ext not in LOADERS:
        supported = ", ".join(sorted(LOADERS))
        raise DocumentLoadError(f"Unsupported file type '{ext or 'none'}'. Supported: {supported}.")
    if not data:
        raise DocumentLoadError("File is empty.")
    if len(data) > max_file_mb * 1024 * 1024:
        raise DocumentLoadError(f"File exceeds the {max_file_mb} MB limit.")
    if ext == ".pdf" and b"%PDF-" not in data[:1024]:
        raise DocumentLoadError("File content is not a valid PDF.")
    if ext == ".xlsx" and not data.startswith(b"PK\x03\x04"):
        raise DocumentLoadError("File content is not a valid XLSX workbook.")
    return ext


def load_documents(
    files: list[tuple[str, bytes]],
    max_files: int | None = None,
    max_file_mb: int | None = None,
) -> IngestionResult:
    max_files = max_files or settings.max_files
    max_file_mb = max_file_mb or settings.max_file_mb
    result = IngestionResult()

    if not files:
        result.errors.append("No files provided.")
        return result
    if len(files) > max_files:
        result.errors.append(f"Too many files: {len(files)} uploaded, maximum is {max_files}.")
        return result

    seen: set[str] = set()
    for raw_name, data in files:
        name = safe_filename(raw_name)
        # Windows file names are case-insensitive, so "Guide.txt" and "guide.txt" collide.
        if name.lower() in seen:
            result.errors.append(f"{name}: Duplicate file name in this upload; only the first file was processed.")
            continue
        seen.add(name.lower())
        try:
            ext = validate_file(name, data, max_file_mb)
            result.sections.extend(LOADERS[ext](name, data))
        except DocumentLoadError as exc:
            result.errors.append(f"{name}: {exc}")
    return result
