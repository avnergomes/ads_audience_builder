"""Helpers for loading customer list files with resilient CSV parsing."""

from __future__ import annotations

import csv
from typing import IO, Any

import pandas as pd


CSV_OPTIONS = {
    "sep": ",",
    "quotechar": '"',
    "quoting": csv.QUOTE_MINIMAL,
    "escapechar": "\\",
    "engine": "python",
}


def _reset_stream(stream: Any) -> None:
    """Rewind file-like *stream* if it supports ``seek``."""

    if hasattr(stream, "seek"):
        stream.seek(0)


def read_audience_csv(source: IO[str] | IO[bytes] | str) -> pd.DataFrame:
    """Read a CSV file while respecting quoted fields such as "Surname, Name"."""

    _reset_stream(source)
    try:
        return pd.read_csv(source, encoding="utf-8-sig", **CSV_OPTIONS)
    except UnicodeDecodeError:
        _reset_stream(source)
        return pd.read_csv(source, encoding="utf-8", **CSV_OPTIONS)
    except pd.errors.ParserError:
        # Rewind and retry without the escape character as a best-effort fallback.
        fallback_options = {key: value for key, value in CSV_OPTIONS.items() if key != "escapechar"}
        _reset_stream(source)
        return pd.read_csv(source, encoding="utf-8-sig", **fallback_options)

