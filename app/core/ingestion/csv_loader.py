from io import StringIO

import pandas as pd

from app.core.ingestion.models import DocumentLoadError, DocumentSection
from app.core.ingestion.text_loader import decode_text


def _cell(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def dataframe_to_sections(df: pd.DataFrame, base_metadata: dict) -> list[DocumentSection]:
    columns = [str(c).strip() for c in df.columns]
    sections = []
    # Spreadsheet row numbers: row 1 is the header, so data starts at row 2.
    for row_number, values in enumerate(df.itertuples(index=False, name=None), start=2):
        pairs = [f"{col}: {_cell(val)}" for col, val in zip(columns, values) if _cell(val)]
        if pairs:
            sections.append(
                DocumentSection(
                    text=f"Row {row_number}: " + "; ".join(pairs),
                    metadata={**base_metadata, "row": row_number},
                )
            )
    return sections


def load_csv(filename: str, data: bytes) -> list[DocumentSection]:
    text = decode_text(data)
    try:
        df = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise DocumentLoadError("CSV file is empty.") from exc
    except Exception as exc:
        raise DocumentLoadError("CSV file could not be parsed.") from exc

    sections = dataframe_to_sections(df, {"source": filename, "file_type": "csv"})
    if not sections:
        raise DocumentLoadError("CSV file has no data rows.")
    return sections
