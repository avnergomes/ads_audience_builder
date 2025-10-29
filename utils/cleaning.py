"""Utilities for normalising and cleaning uploaded audience data."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import phonenumbers
from pandas.api.types import is_scalar


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

        if is_scalar(value):
            # Using f-strings ensures we always end up with a Python ``str`` even
            # when the original value is a numpy scalar or another exotic type.
            text = value if isinstance(value, str) else f"{value}"
            return text

        # Fallback for lists/Series/etc. – treat them as empty so they do not
        # break normalisation.
        return ""

    renamed = {}
    for column in df.columns:
        text = _stringify(column).strip().lower()
        renamed[column] = STANDARD_COLUMNS.get(text, text)
    return df.rename(columns=renamed)


def trim_strings(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    def _trim(value: object) -> str:
        if value is None:
            return ""

        if is_scalar(value):
            if pd.isna(value):
                return ""
            return (value if isinstance(value, str) else f"{value}").strip()

        # Non scalar objects (lists, Series, dicts, etc.) are not meaningful
        # audience attributes; normalise them to empty strings so downstream
        # steps can safely treat them as missing.
        return ""

    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(_trim)
    return df


def validate_email(email: object) -> bool:
    if email is None or not is_scalar(email):
        return False

    if pd.isna(email):
        return False

    text = str(email).strip()
    if not text:
        return False

    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", text.lower()))


def validate_phone(phone: object, default_country: str = "US") -> bool:
    """Validate and check if a phone number is parseable.
    
    Args:
        phone: The phone number to validate
        default_country: Default country code for parsing (default: US)
    
    Returns:
        True if the phone number is valid, False otherwise
    """
    if phone is None or not is_scalar(phone):
        return False

    if pd.isna(phone):
        return False

    text = str(phone).strip()
    if not text:
        return False

    try:
        parsed = phonenumbers.parse(text, default_country)
        return phonenumbers.is_valid_number(parsed)
    except phonenumbers.NumberParseException:
        return False


def normalize_phone(phone: object, default_country: str = "US") -> str:
    """Normalize a phone number to E164 format.
    
    Args:
        phone: The phone number to normalize
        default_country: Default country code for parsing (default: US)
    
    Returns:
        Normalized phone number in E164 format or empty string if invalid
    """
    if phone is None or not is_scalar(phone):
        return ""

    if pd.isna(phone):
        return ""

    text = str(phone).strip()
    if not text:
        return ""

    try:
        parsed = phonenumbers.parse(text, default_country)
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        pass

    return text


def clean_phones(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Normalize phone numbers to E164 format.
    
    Returns the cleaned DataFrame along with the number of invalid phones found.
    Note: Unlike emails, invalid phones are not removed, just flagged in stats.
    """
    invalid_phones = 0
    if "phone" in df.columns:
        df["phone"] = df["phone"].apply(normalize_phone)
        invalid_mask = df["phone"].apply(lambda x: x == "" or not validate_phone(x))
        invalid_phones = int(invalid_mask.sum())
    return df, invalid_phones


def clean_emails(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Normalise e-mails and drop invalid rows.

    Returns the cleaned DataFrame along with the number of records removed.
    """

    removed = 0
    if "email" in df.columns:
        def _normalise_email(value: object) -> str:
            if value is None or not is_scalar(value):
                return ""
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

    df, invalid_phones = clean_phones(df)
    stats["invalid_phones"] = invalid_phones

    df, duplicates_removed = deduplicate(df, ["email", "phone"])
    stats["duplicates_removed"] = duplicates_removed
    stats["final_rows"] = len(df)

    summary = column_summary(df, ["email", "phone", "fn", "ln", "ct", "st", "zip", "country"])

    return df, stats, summary


# Backwards compatibility exports
normalize_columns = normalise_columns
