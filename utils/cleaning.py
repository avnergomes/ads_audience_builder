"""Utilities for normalising and cleaning uploaded audience data."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import pandas as pd


# Canonical column names we support throughout the application. Whenever we
# ingest a file we attempt to map the incoming headers to this set so the rest of
# the pipeline can remain deterministic.
STANDARD_COLUMNS: Dict[str, str] = {
    "e-mail": "email",
    "email address": "email",
    "mail": "email",
    "phone number": "phone",
    "mobile": "phone",
    "first name": "fn",
    "firstname": "fn",
    "given name": "fn",
    "last name": "ln",
    "lastname": "ln",
    "surname": "ln",
    "family name": "ln",
    "name": "full_name",
    "full name": "full_name",
    "fullname": "full_name",
    "zip code": "zip",
    "zipcode": "zip",
    "postal": "zip",
    "state/region": "st",
    "province": "st",
    "city/town": "ct",
    "country code": "country",
    "lead_id": "meta_lead_id",
    "lead id": "meta_lead_id",
    "source": "source",
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with lower-cased, canonicalised headers."""

    def _stringify(value: object) -> str:
        if value is None:
            return ""
        # Using f-strings ensures we always end up with a Python ``str`` even
        # when the original value is a numpy scalar or another exotic type.
        text = value if isinstance(value, str) else f"{value}"
        return text

    renamed = {}
    for column in df.columns:
        text = _stringify(column).strip().lower()
        renamed[column] = STANDARD_COLUMNS.get(text, text)
    return df.rename(columns=renamed)


def trim_strings(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    def _trim(value: object) -> str:
        if pd.isna(value):
            return ""
        return (value if isinstance(value, str) else f"{value}").strip()

    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(_trim)
    return df


def validate_email(email: str) -> bool:
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", str(email)))


def clean_emails(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Normalise e-mails and drop invalid rows.

    Returns the cleaned DataFrame along with the number of records removed.
    """

    removed = 0
    if "email" in df.columns:
        def _normalise_email(value: object) -> str:
            if pd.isna(value):
                return ""
            text = value if isinstance(value, str) else f"{value}"
            return text.strip().lower()

        df["email"] = df["email"].apply(_normalise_email)
        mask = df["email"].apply(validate_email)
        removed = int((~mask).sum())
        df = df[mask]
    return df, removed


def deduplicate(df: pd.DataFrame, subset: Iterable[str]) -> Tuple[pd.DataFrame, int]:
    """Drop duplicate rows using the supplied *subset* of columns."""

    subset = [column for column in subset if column in df.columns]
    if not subset:
        return df, 0

    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    return df, before - len(df)


def column_summary(df: pd.DataFrame, columns: Iterable[str]) -> List[Dict[str, int]]:
    """Return missing-value counts for *columns* present in *df*."""

    summary = []
    for column in columns:
        if column in df.columns:
            summary.append({
                "column": column,
                "missing": int(df[column].isna().sum()),
                "populated": int(df[column].notna().sum()),
            })
    return summary


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], List[Dict[str, int]]]:
    """Execute the baseline cleaning pipeline for the uploaded DataFrame."""

    stats: Dict[str, int] = {"initial_rows": len(df)}

    df = normalise_columns(df)
    df = trim_strings(df, df.select_dtypes(include="object").columns)

    df, invalid_emails = clean_emails(df)
    stats["invalid_emails"] = invalid_emails

    df, duplicates_removed = deduplicate(df, ["email", "phone"])
    stats["duplicates_removed"] = duplicates_removed
    stats["final_rows"] = len(df)

    summary = column_summary(df, ["email", "phone", "fn", "ln", "ct", "st", "zip", "country"])

    return df, stats, summary


# Backwards compatibility exports
normalize_columns = normalise_columns
