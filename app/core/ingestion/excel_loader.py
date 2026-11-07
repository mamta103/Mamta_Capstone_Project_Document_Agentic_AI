from io import BytesIO

import pandas as pd

from app.core.ingestion.csv_loader import dataframe_to_sections
from app.core.ingestion.models import DocumentLoadError, DocumentSection


def load_excel(filename: str, data: bytes) -> list[DocumentSection]:
    try:
        sheets = pd.read_excel(BytesIO(data), sheet_name=None, dtype=str, engine="openpyxl")
    except Exception as exc:
        raise DocumentLoadError("Excel file is corrupt or cannot be read.") from exc

    sections = []
    for sheet_name, df in sheets.items():
        base = {"source": filename, "file_type": "xlsx", "sheet": str(sheet_name)}
        sections.extend(dataframe_to_sections(df, base))
    if not sections:
        raise DocumentLoadError("Excel file has no data rows.")
    return sections
