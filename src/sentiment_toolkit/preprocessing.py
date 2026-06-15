"""Text and file preprocessing utilities."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass


@dataclass(frozen=True)
class TextRecord:
    """A single text row for batch analysis."""

    index: int
    text: str
    source_column: str | None = None


def split_sentences(text: str) -> list[str]:
    """Split multiline input into non-empty sentences."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def read_csv_texts(
    file_content: str | bytes,
    text_column: str = "text",
    encoding: str = "utf-8",
) -> list[TextRecord]:
    """
    Parse CSV content and extract text rows.

    Uses the named column when present; otherwise falls back to the first column.
    """
    if isinstance(file_content, bytes):
        file_content = file_content.decode(encoding)

    reader = csv.DictReader(io.StringIO(file_content))
    if not reader.fieldnames:
        raise ValueError("CSV file has no header row.")

    column = text_column if text_column in reader.fieldnames else reader.fieldnames[0]
    records: list[TextRecord] = []

    for index, row in enumerate(reader):
        value = (row.get(column) or "").strip()
        if value:
            records.append(TextRecord(index=index, text=value, source_column=column))

    if not records:
        raise ValueError(f"No non-empty values found in column '{column}'.")

    return records


def records_to_csv(
    records: list[dict],
    fieldnames: list[str],
) -> str:
    """Serialize analysis results to CSV string."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()
