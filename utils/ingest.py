"""Helpers for loading customer list files with resilient CSV parsing."""

from __future__ import annotations

import csv
import re
from typing import IO, Any, Dict, Iterable, List

import pandas as pd


CSV_OPTIONS = {
    "sep": ",",
    "quotechar": '"',
    "quoting": csv.QUOTE_MINIMAL,
    "escapechar": "\\",
    "engine": "python",
}

# Broad heuristics used to detect and infer headers when a file does not ship
# with an explicit header row. They intentionally favour recall over
# precision – we only promote a column when we are confident it contains a
# particular type of data.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^[+\d][\d\s().-]{5,}$")
DATE_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}$"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}$"),
    re.compile(r"[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}$"),
]
META_LEAD_RE = re.compile(r"^(?:l:)?\d{6,}$", re.IGNORECASE)
KNOWN_HEADER_TOKENS = {
    "email",
    "mail",
    "phone",
    "mobile",
    "first name",
    "last name",
    "name",
    "full name",
    "city",
    "state",
    "zip",
    "postal",
    "country",
    "lead",
    "created",
    "campaign",
    "source",
    "ad",
    "form",
}


def _reset_stream(stream: Any) -> None:
    """Rewind file-like *stream* if it supports ``seek``."""

    if hasattr(stream, "seek"):
        stream.seek(0)


def _read_csv(source: IO[str] | IO[bytes] | str, *, header: int | None | str) -> pd.DataFrame:
    """Read *source* with the standard parsing options and helpful fallbacks."""

    _reset_stream(source)
    try:
        return pd.read_csv(source, encoding="utf-8-sig", dtype=str, header=header, **CSV_OPTIONS)
    except UnicodeDecodeError:
        _reset_stream(source)
        return pd.read_csv(source, encoding="utf-8", dtype=str, header=header, **CSV_OPTIONS)
    except pd.errors.ParserError:
        fallback_options = {key: value for key, value in CSV_OPTIONS.items() if key != "escapechar"}
        _reset_stream(source)
        return pd.read_csv(source, encoding="utf-8-sig", dtype=str, header=header, **fallback_options)


def _normalise_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _looks_like_missing_header(df: pd.DataFrame) -> bool:
    """Return ``True`` when the column labels appear to contain data values."""

    if df.empty:
        return False

    column_texts = [_normalise_text(column) for column in df.columns]
    if not column_texts:
        return False

    # If any column header looks like an e-mail address we can safely assume
    # the first row was treated as the header.
    email_like = sum(bool(EMAIL_RE.match(text)) for text in column_texts)
    if email_like:
        return True

    known_headers = sum(text in KNOWN_HEADER_TOKENS for text in column_texts)
    if known_headers:
        return False

    data_like = sum(bool(re.search(r"\d", text)) or "@" in text for text in column_texts)
    return data_like >= max(1, len(column_texts) // 2)


def _sample_values(series: pd.Series, limit: int = 100) -> List[str]:
    values: List[str] = []
    for value in series.dropna():
        text = str(value).strip()
        if text:
            values.append(text)
        if len(values) >= limit:
            break
    return values


def _ratio(predicate: Iterable[bool]) -> float:
    values = list(predicate)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _is_phone_candidate(value: str) -> bool:
    if not PHONE_RE.match(value):
        return False

    normalised = re.sub(r"\D", "", value)
    if len(normalised) < 7 or len(normalised) > 15:
        return False

    stripped = value.strip()
    for pattern in DATE_PATTERNS:
        if pattern.fullmatch(stripped):
            return False

    return True


def _infer_textual_columns(columns: List[int], df: pd.DataFrame) -> Dict[int, str]:
    inferred: Dict[int, str] = {}
    textual_candidates: List[int] = []

    for column in columns:
        values = _sample_values(df[column])
        if not values:
            continue

        alpha_ratio = _ratio(bool(re.fullmatch(r"[A-Za-z\s'.-]+", value)) for value in values)
        if alpha_ratio < 0.6:
            continue

        textual_candidates.append(column)

    if not textual_candidates:
        return inferred

    if len(textual_candidates) == 1:
        inferred[textual_candidates[0]] = "full_name"
        return inferred

    # Preserve the column order to keep the mapping intuitive for the user.
    ordered = sorted(textual_candidates)
    inferred[ordered[0]] = "fn"
    inferred[ordered[1]] = "ln"

    if len(ordered) > 2:
        inferred[ordered[2]] = "full_name"

    return inferred


def _infer_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort header inference for header-less CSV files."""

    rename: Dict[int, str] = {}
    columns = list(df.columns)

    email_scores: Dict[int, float] = {}
    phone_scores: Dict[int, float] = {}
    meta_lead_scores: Dict[int, float] = {}

    for column in columns:
        values = _sample_values(df[column])
        if not values:
            continue

        email_scores[column] = _ratio(bool(EMAIL_RE.match(value.lower())) for value in values)
        phone_scores[column] = _ratio(_is_phone_candidate(value) for value in values)
        meta_lead_scores[column] = _ratio(bool(META_LEAD_RE.match(value)) for value in values)

    if email_scores:
        best_email = max(email_scores, key=email_scores.get)
        if email_scores[best_email] >= 0.5:
            rename[best_email] = "email"

    remaining_for_phone = [column for column in columns if column not in rename]
    if remaining_for_phone and phone_scores:
        best_phone = max(phone_scores, key=phone_scores.get)
        if best_phone in remaining_for_phone and phone_scores[best_phone] >= 0.4:
            rename[best_phone] = "phone"

    remaining_for_lead = [column for column in columns if column not in rename]
    if remaining_for_lead and meta_lead_scores:
        best_lead = max(meta_lead_scores, key=meta_lead_scores.get)
        if best_lead in remaining_for_lead and meta_lead_scores[best_lead] >= 0.5:
            rename[best_lead] = "meta_lead_id"

    remaining_columns = [column for column in columns if column not in rename]
    rename.update(_infer_textual_columns(remaining_columns, df))

    # Provide readable fallbacks for any remaining numeric column headers.
    for column in columns:
        if column not in rename:
            rename[column] = f"column_{int(column) + 1}"

    return df.rename(columns=rename)


def read_audience_csv(source: IO[str] | IO[bytes] | str) -> pd.DataFrame:
    """Read a CSV file while respecting quoted fields such as "Surname, Name"."""

    df = _read_csv(source, header="infer")

    if _looks_like_missing_header(df):
        df = _read_csv(source, header=None)
        df = _infer_headers(df)

    return df

