"""Utilities for normalising and cleaning uploaded audience data."""
from __future__ import annotations

import math
import numbers
import re
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import phonenumbers
from pandas.api.types import is_scalar


# Common e-mail validation helpers.
EMAIL_CANDIDATE_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+", re.IGNORECASE)
EMAIL_VALID_RE = re.compile(
    r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$",
    re.IGNORECASE,
)

# Top level domain typos we can confidently auto-correct.
EMAIL_TLD_FIXES = {
    "con": "com",
    "c0m": "com",
    "cpm": "com",
    "cim": "com",
}


# Canonical column names we support throughout the application. Whenever we
# ingest a file we attempt to map the incoming headers to this set so the rest of
# the pipeline can remain deterministic.
STANDARD_COLUMNS: Dict[str, str] = {
    "e-mail": "email",
    "email address": "email",
    "mail": "email",
    "email1": "email",
    "email 1": "email",
    "email2": "email",
    "email 2": "email",
    "email-2": "email",
    "phone number": "phone",
    "phone no": "phone",
    "phone #": "phone",
    "phone1": "phone",
    "phone 1": "phone",
    "phone2": "phone",
    "phone 2": "phone",
    "mobile": "phone",
    "mobile phone": "phone",
    "secondary phone": "phone",
    "secondary phone number": "phone",
    "whatsapp": "phone",
    "whatsapp number": "phone",
    "first": "fn",
    "first name": "fn",
    "firstname": "fn",
    "given name": "fn",
    "fname": "fn",
    "last": "ln",
    "last name": "ln",
    "lastname": "ln",
    "surname": "ln",
    "family name": "ln",
    "lname": "ln",
    "name": "full_name",
    "full name": "full_name",
    "fullname": "full_name",
    "zip code": "zip",
    "zipcode": "zip",
    "postal": "zip",
    "postal code": "zip",
    "state/region": "st",
    "province": "st",
    "state province": "st",
    "city/town": "ct",
    "city": "ct",
    "country code": "country",
    "lead_id": "meta_lead_id",
    "lead id": "meta_lead_id",
    "source": "source",
}

CONTACT_COLUMNS = ("email", "phone", "fn", "ln", "full_name")


def _stringify_header(value: object) -> str:
    if value is None or not is_scalar(value):
        return ""

    if isinstance(value, str):
        return value

    return f"{value}"


def _stringify_scalar_value(value: object) -> str:
    if value is None or not is_scalar(value):
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, bool):
        return str(value)

    if pd.isna(value):
        return ""

    if isinstance(value, numbers.Integral):
        return str(int(value))

    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            return ""
        text = format(number, "f").rstrip("0").rstrip(".")
        return text or str(int(round(number)))

    return str(value).strip()


def _normalise_header_text(value: object) -> str:
    text = _stringify_header(value).strip().lower()
    if not text:
        return ""

    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\s._-]*\d+$", "", text)
    return text


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with lower-cased, canonicalised headers."""

    renamed: Dict[object, str] = {}
    seen: Dict[str, int] = {}

    for column in df.columns:
        text = _normalise_header_text(column)
        canonical = STANDARD_COLUMNS.get(text, text)
        count = seen.get(canonical, 0)
        new_name = canonical if count == 0 else f"{canonical}__{count}"
        seen[canonical] = count + 1
        renamed[column] = new_name

    return df.rename(columns=renamed)


def trim_strings(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    def _trim(value: object) -> str:
        if value is None:
            return ""

        if is_scalar(value):
            if pd.isna(value):
                return ""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, bool):
                return str(value)
            if isinstance(value, numbers.Integral):
                return str(int(value))
            if isinstance(value, numbers.Real):
                number = float(value)
                if not math.isfinite(number):
                    return ""
                text = format(number, "f").rstrip("0").rstrip(".")
                return text or str(int(round(number)))
            return str(value).strip()

        return ""

    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(_trim)
    return df


def validate_email(email: object) -> bool:
    text = _stringify_scalar_value(email)
    if not text:
        return False

    return bool(EMAIL_VALID_RE.match(text))


def _sanitise_email(value: object) -> Tuple[str, Dict[str, bool]]:
    """Return a cleaned-up e-mail string and metadata about the transformation."""

    meta = {"missing": False, "corrected": False, "parse_error": False}

    if value is None or not is_scalar(value):
        meta["missing"] = True
        return "", meta

    if pd.isna(value):
        meta["missing"] = True
        return "", meta

    text = value if isinstance(value, str) else f"{value}"
    text = text.strip()
    if not text:
        meta["missing"] = True
        return "", meta

    # Remove zero width characters or BOM markers that frequently sneak into
    # CRM exports.
    cleaned = text.replace("\ufeff", "").replace("\u200b", "")
    if cleaned != text:
        meta["corrected"] = True
        text = cleaned

    # Trim surrounding punctuation such as quotes or brackets.
    stripped = text.strip("\"'<>[]{}()")
    if stripped != text:
        meta["corrected"] = True
        text = stripped

    text = text.lower()

    if text in {"none", "nan", "null"}:
        meta["missing"] = True
        return "", meta

    if text.startswith("mailto:"):
        text = text[len("mailto:") :]
        meta["corrected"] = True

    if re.search(r"\s", text):
        text = re.sub(r"\s+", "", text)
        meta["corrected"] = True

    match = EMAIL_CANDIDATE_RE.search(text)
    if match:
        candidate = match.group(0).lower()
        if match.start() != 0 or match.end() != len(text):
            meta["parse_error"] = True
            meta["corrected"] = True
        text = candidate

    trimmed = text.rstrip(",;:.)]")
    if trimmed != text:
        meta["parse_error"] = True
        meta["corrected"] = True
        text = trimmed

    if ".." in text:
        text = re.sub(r"\.\.+", ".", text)
        meta["corrected"] = True

    if text.endswith("@gmail"):
        text = f"{text}.com"
        meta["corrected"] = True
    else:
        for wrong, right in EMAIL_TLD_FIXES.items():
            suffix = f".{wrong}"
            if text.endswith(suffix):
                text = f"{text[: -len(wrong)]}{right}"
                meta["corrected"] = True
                break

    if text.endswith("@hotmail.co") or text.endswith("@outlook.co"):
        text = f"{text}m"
        meta["corrected"] = True

    if text.count("@") != 1:
        return text, meta

    local, domain = text.split("@", 1)
    if not local or not domain:
        meta["missing"] = True

    return text, meta


def validate_phone(phone: object, default_country: str = "US") -> bool:
    """Validate and check if a phone number is parseable."""

    text = _stringify_scalar_value(phone)
    if not text:
        return False

    try:
        parsed = phonenumbers.parse(text, default_country)
        return phonenumbers.is_valid_number(parsed)
    except phonenumbers.NumberParseException:
        return False


def normalize_phone(phone: object, default_country: str = "US") -> str:
    """Normalize a phone number to E164 format."""

    text = _stringify_scalar_value(phone)
    if not text:
        return ""

    cleaned = text.replace("\ufeff", "").replace("\u200b", "")
    cleaned = cleaned.lower()
    cleaned = re.split(r"(?:ext|extension|x)\s*", cleaned)[0]
    cleaned = cleaned.strip()

    original = cleaned

    if not cleaned:
        return ""

    has_plus = cleaned.startswith("+")
    digits = re.sub(r"\D", "", cleaned)
    if not digits:
        return ""

    candidate = f"+{digits}" if has_plus else digits

    try:
        parsed = phonenumbers.parse(candidate, default_country)
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        pass

    return original


def clean_phones(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Normalize phone numbers to E164 format."""

    invalid_phones = 0
    if "phone" in df.columns:
        df["phone"] = df["phone"].apply(normalize_phone)
        invalid_mask = df["phone"].apply(lambda x: x == "" or not validate_phone(x))
        invalid_phones = int(invalid_mask.sum())
    return df, invalid_phones


def _coalesce_values(values: Iterable[object]) -> str:
    for value in values:
        if isinstance(value, bool):
            continue
        text = _stringify_scalar_value(value)
        if text:
            return text
    return ""


def coalesce_contact_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Combine duplicate contact columns keeping the first populated value."""

    if df.empty:
        return df

    df = df.copy()
    column_labels = list(df.columns)

    for base in CONTACT_COLUMNS:
        indices = [idx for idx, label in enumerate(column_labels) if str(label).startswith(base)]
        if not indices:
            continue

        subset = df.iloc[:, indices]
        if isinstance(subset, pd.Series):
            subset = subset.to_frame()

        combined = subset.apply(lambda row: _coalesce_values(row.tolist()), axis=1)

        primary_label = column_labels[indices[0]]
        df.iloc[:, indices[0]] = combined
        if primary_label != base:
            df.rename(columns={primary_label: base}, inplace=True)

        for position in reversed(indices[1:]):
            df.drop(columns=column_labels[position], inplace=True)

        column_labels = list(df.columns)

    return df


def clean_emails(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Normalise e-mail values and auto-correct common issues."""

    stats = {"invalid": 0, "missing": 0, "corrected": 0, "parsing_errors": 0}

    if "email" not in df.columns:
        return df, stats

    cleaned_emails: List[str] = []
    valid_mask: List[bool] = []

    for value in df["email"]:
        email, meta = _sanitise_email(value)
        cleaned_emails.append(email)

        if meta["missing"] and not email:
            stats["missing"] += 1
            valid_mask.append(True)
            continue

        if meta["parse_error"]:
            stats["parsing_errors"] += 1

        if meta["corrected"]:
            stats["corrected"] += 1

        is_valid = bool(EMAIL_VALID_RE.match(email))
        if not is_valid:
            stats["invalid"] += 1
        valid_mask.append(is_valid)

    df["email"] = cleaned_emails
    df = df[valid_mask]

    return df, stats


def drop_unreachable_contacts(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove rows where both the e-mail and phone fields are empty."""

    if df.empty:
        return df, 0

    if "email" in df.columns:
        email_empty = df["email"].fillna("").astype(str).str.strip().eq("")
    else:
        email_empty = pd.Series(True, index=df.index)

    if "phone" in df.columns:
        phone_empty = df["phone"].fillna("").astype(str).str.strip().eq("")
    else:
        phone_empty = pd.Series(True, index=df.index)

    unreachable_mask = email_empty & phone_empty
    removed = int(unreachable_mask.sum())
    if not removed:
        return df, 0

    return df[~unreachable_mask], removed


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
            series = df[column]
            if series.dtype == object:
                values = series.fillna("").astype(str).str.strip()
                missing_mask = values.eq("")
            else:
                missing_mask = series.isna()

            missing = int(missing_mask.sum())
            populated = len(series) - missing
            summary.append({"column": column, "missing": missing, "populated": populated})
    return summary


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], List[Dict[str, int]]]:
    """Execute the baseline cleaning pipeline for the uploaded DataFrame."""

    stats: Dict[str, int] = {"initial_rows": len(df)}

    df = normalise_columns(df)
    df = trim_strings(df, df.select_dtypes(include="object").columns)

    # Merge duplicate contact fields so that enrichment/validation steps have a
    # single canonical column to work with.
    df = coalesce_contact_columns(df)

    df, email_stats = clean_emails(df)
    stats["invalid_emails"] = email_stats["invalid"]
    stats["missing_emails"] = email_stats["missing"]
    stats["email_corrections"] = email_stats["corrected"]
    stats["email_parsing_errors"] = email_stats["parsing_errors"]

    df, invalid_phones = clean_phones(df)
    stats["invalid_phones"] = invalid_phones

    df, unreachable_removed = drop_unreachable_contacts(df)
    stats["rows_without_contact"] = unreachable_removed

    df, duplicates_removed = deduplicate(df, ["email", "phone"])
    stats["duplicates_removed"] = duplicates_removed
    stats["final_rows"] = len(df)

    summary = column_summary(df, ["email", "phone", "fn", "ln", "ct", "st", "zip", "country"])

    return df, stats, summary


# Backwards compatibility exports
normalize_columns = normalise_columns
