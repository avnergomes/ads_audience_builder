"""Helpers for loading customer list files with resilient CSV parsing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import IO, Any, Dict

import pandas as pd


CSV_OPTIONS = {
    "sep": ",",
    "quotechar": '"',
    "quoting": csv.QUOTE_MINIMAL,
    "doublequote": True,
    "escapechar": "\\",
    "dtype": str,
    "keep_default_na": False,
    "na_values": [],
}

# Encodings that commonly show up in CRM exports. We try UTF-8 variants first
# and then fall back to the more permissive Windows-1252/Latin encodings.
ENCODINGS = ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1")


def _reset_stream(stream: Any) -> None:
    """Rewind file-like *stream* if it supports ``seek``."""

    try:
        stream.seek(0)
    except (AttributeError, OSError):
        pass


def _read_sample(source: IO[str] | IO[bytes] | str, size: int = 4096) -> str:
    """Return a text sample from *source* without consuming the stream."""

    if isinstance(source, (str, Path)):
        try:
            with Path(source).open("rb") as handle:
                data = handle.read(size)
        except FileNotFoundError:
            return ""
    else:
        try:
            position = source.tell()
        except (AttributeError, OSError):
            position = None
        try:
            data = source.read(size)
        except Exception:
            return ""
        finally:
            if position is not None:
                try:
                    source.seek(position)
                except (AttributeError, OSError):
                    pass
            else:
                _reset_stream(source)

    if isinstance(data, bytes):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    return str(data)


def _sniff_dialect(sample: str) -> Dict[str, Any]:
    """Best-effort detection of CSV delimiters and quoting."""

    if not sample:
        return {}

    sniff = csv.Sniffer()
    try:
        dialect = sniff.sniff(sample, delimiters=[",", ";", "|", "\t"])
    except csv.Error:
        return {}

    options: Dict[str, Any] = {
        "sep": dialect.delimiter or CSV_OPTIONS["sep"],
        "quotechar": dialect.quotechar or CSV_OPTIONS["quotechar"],
        "doublequote": dialect.doublequote,
    }

    if dialect.escapechar:
        options["escapechar"] = dialect.escapechar

    return options


def _read_with_engine(
    source: IO[str] | IO[bytes] | str,
    encoding: str,
    options: Dict[str, Any],
    engine: str,
) -> pd.DataFrame:
    """Attempt to read CSV with a specific pandas engine."""

    read_options = {**options, "engine": engine}
    return pd.read_csv(source, encoding=encoding, **read_options)


def _read_without_escapechar(
    source: IO[str] | IO[bytes] | str,
    encoding: str,
    options: Dict[str, Any],
    engine: str,
) -> pd.DataFrame:
    """Retry CSV parsing without the ``escapechar`` hint."""

    fallback = {key: value for key, value in options.items() if key != "escapechar"}
    return _read_with_engine(source, encoding, fallback, engine)


def read_audience_csv(source: IO[str] | IO[bytes] | str) -> pd.DataFrame:
    """Read a CSV file while respecting quoted fields such as "Surname, Name"."""

    sample = _read_sample(source)
    sniffed_options = _sniff_dialect(sample)
    read_options = {**CSV_OPTIONS, **sniffed_options}

    for encoding in ENCODINGS:
        # First try the high-performance C engine with our hints.
        _reset_stream(source)
        try:
            return _read_with_engine(source, encoding, read_options, "c")
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, ValueError):
            # ParserError covers malformed CSVs, ValueError happens when options
            # are incompatible with the C engine (e.g. unsupported sep/quote).
            pass

        # Fall back to the more permissive Python engine.
        _reset_stream(source)
        try:
            return _read_with_engine(source, encoding, read_options, "python")
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            # Retry without escape char if parser chokes on malformed CSVs.
            _reset_stream(source)
            try:
                return _read_without_escapechar(source, encoding, read_options, "python")
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

    # Final attempt with pandas defaults for exotic files.
    _reset_stream(source)
    return pd.read_csv(source)

