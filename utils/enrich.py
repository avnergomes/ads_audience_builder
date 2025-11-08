"""Data enrichment helpers."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
GENDERIZE_ENDPOINT = "https://api.genderize.io/"
NAME_PARSER_ENDPOINT = "https://parserator.datamade.us/api/v1/parse/"

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=512)
def _query_name_parser_api(name: str) -> Optional[Dict[str, Any]]:
    if not name:
        return None

    try:
        response = requests.get(
            NAME_PARSER_ENDPOINT,
            params={"string": name},
            timeout=5,
        )
    except requests.RequestException as exc:
        LOGGER.debug("Name parser lookup failed for %s: %s", name, exc)
        return None

    if response.status_code != 200:
        LOGGER.debug(
            "Name parser API returned status %s for %s",
            response.status_code,
            name,
        )
        return None

    try:
        payload = response.json()
    except ValueError as exc:
        LOGGER.debug("Name parser API returned invalid JSON for %s: %s", name, exc)
        return None

    return payload if isinstance(payload, dict) else None


def _extract_name_from_payload(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(data, dict):
        return None, None

    candidates: Dict[str, Any] = {}
    if "result" in data and isinstance(data["result"], dict):
        candidates.update(data["result"])
    if "components" in data and isinstance(data["components"], dict):
        candidates.update(data["components"])
    if "parsed" in data and isinstance(data["parsed"], dict):
        candidates.update(data["parsed"])

    for key, value in list(data.items()):
        if isinstance(value, dict):
            candidates.setdefault(key, value)

    first: Optional[str] = None
    last: Optional[str] = None

    for container in candidates.values():
        if not isinstance(container, dict):
            continue
        first = first or container.get("given_name") or container.get("first") or container.get("first_name")
        last = last or container.get("surname") or container.get("last") or container.get("last_name")

    if not first:
        first = candidates.get("given_name") or candidates.get("first") or candidates.get("first_name")
    if not last:
        last = candidates.get("surname") or candidates.get("last") or candidates.get("last_name")

    def _clean(value: Optional[str]) -> Optional[str]:
        if not isinstance(value, str):
            return None
        text = value.strip()
        return text.title() if text else None

    return _clean(first), _clean(last)


@lru_cache(maxsize=1)
def load_zip_database() -> pd.DataFrame:
    path = ROOT / "data" / "zip_database.csv"
    if not path.exists():
        return pd.DataFrame(columns=["zip_code", "city", "state"])
    return pd.read_csv(path, dtype=str)


@lru_cache(maxsize=1)
def load_gender_dictionary() -> Dict[str, str]:
    path = ROOT / "data" / "name_gender_dict.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


EMAIL_NAME_SPLIT_RE = re.compile(r"[._+\-]+")
COMMON_EMAIL_PREFIXES = {
    "info",
    "contact",
    "admin",
    "hello",
    "support",
    "sales",
    "marketing",
    "billing",
}


def _tidy_email_fragment(fragment: str) -> Optional[str]:
    cleaned = re.sub(r"\d+", "", fragment)
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    return cleaned


def derive_name_from_email(email: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(email, str) or "@" not in email:
        return None, None

    local_part = email.split("@", 1)[0]
    if "+" in local_part:
        local_part = local_part.split("+", 1)[0]

    fragments = EMAIL_NAME_SPLIT_RE.split(local_part)
    processed = [
        _tidy_email_fragment(fragment)
        for fragment in fragments
        if fragment and fragment.lower() not in COMMON_EMAIL_PREFIXES
    ]
    processed = [fragment for fragment in processed if fragment]

    if not processed:
        return None, None

    if len(processed) == 1:
        token = processed[0]
        if len(token) <= 3:
            return None, None
        return token.title(), None

    first = processed[0].title()
    last = processed[-1].title()
    return first, last


def _normalise_name(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(part for part in value.strip().split() if part)
    return cleaned or None


@lru_cache(maxsize=4096)
def _query_gender_api(name: str) -> Optional[Dict[str, Any]]:
    if not name:
        return None

    try:
        response = requests.get(GENDERIZE_ENDPOINT, params={"name": name}, timeout=5)
    except requests.RequestException as exc:
        LOGGER.debug("Gender API lookup failed for %s: %s", name, exc)
        return None

    if response.status_code != 200:
        LOGGER.debug(
            "Gender API returned status %s for %s", response.status_code, name
        )
        return None

    try:
        payload = response.json()
    except ValueError as exc:
        LOGGER.debug("Gender API returned invalid JSON for %s: %s", name, exc)
        return None

    return payload if isinstance(payload, dict) else None


def _is_missing_string(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, dict, set, pd.Series, pd.DataFrame)):
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return True

    try:
        return bool(result)
    except Exception:
        return True


def _string_series_missing_mask(series: pd.Series) -> pd.Series:
    return series.apply(_is_missing_string)


def _fill_names_from_full_name_api(df: pd.DataFrame) -> pd.DataFrame:
    if "full_name" not in df.columns:
        return df

    df = df.copy()
    if "fn" not in df.columns:
        df["fn"] = pd.NA
    if "ln" not in df.columns:
        df["ln"] = pd.NA

    fn_missing = _string_series_missing_mask(df["fn"])
    ln_missing = _string_series_missing_mask(df["ln"])
    needs_lookup = fn_missing | ln_missing

    if not needs_lookup.any():
        return df

    names_to_indices: Dict[str, list[int]] = {}
    for idx in df.index[needs_lookup]:
        candidate = _normalise_name(df.at[idx, "full_name"])
        if candidate:
            names_to_indices.setdefault(candidate, []).append(idx)

    for name, indices in names_to_indices.items():
        data = _query_name_parser_api(name)
        first, last = (None, None)
        if data:
            first, last = _extract_name_from_payload(data)

        if not first or (not last and " " in name):
            pieces = [piece for piece in name.split() if piece]
            if pieces:
                first = first or pieces[0].title()
                if len(pieces) > 1:
                    last = last or pieces[-1].title()

        for idx in indices:
            if first and fn_missing.loc[idx]:
                df.at[idx, "fn"] = first
                fn_missing.loc[idx] = False
            if last and ln_missing.loc[idx]:
                df.at[idx, "ln"] = last
                ln_missing.loc[idx] = False

    return df


def autofill_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    had_fn = "fn" in df.columns
    had_ln = "ln" in df.columns

    if not had_fn:
        df["fn"] = pd.NA
    if not had_ln:
        df["ln"] = pd.NA

    df = _fill_names_from_full_name_api(df)

    if not had_fn and not had_ln and "email" in df.columns:
        derived = df["email"].apply(derive_name_from_email)
        derived_df = pd.DataFrame(
            derived.tolist(), columns=["fn_email", "ln_email"], index=df.index
        )

        fn_missing = _string_series_missing_mask(df["fn"])
        ln_missing = _string_series_missing_mask(df["ln"])

        if "fn_email" in derived_df:
            df.loc[fn_missing, "fn"] = derived_df.loc[fn_missing, "fn_email"]
        if "ln_email" in derived_df:
            df.loc[ln_missing, "ln"] = derived_df.loc[ln_missing, "ln_email"]

    for column in ("fn", "ln"):
        if column in df.columns:
            df[column] = df[column].apply(
                lambda value: value.title().strip() if isinstance(value, str) else value
            )

    return df


def infer_gender(df: pd.DataFrame) -> pd.DataFrame:
    if "fn" not in df.columns:
        return df

    dictionary = {
        (key if isinstance(key, str) else f"{key}").lower(): value
        for key, value in load_gender_dictionary().items()
        if key is not None
    }

    df = df.copy()
    names = df["fn"].apply(_normalise_name)
    unique_names = {name for name in names if name}

    lookup: Dict[str, Tuple[str, Optional[float]]] = {}
    for name in unique_names:
        gender = dictionary.get(name.lower()) if dictionary else None
        confidence: Optional[float] = None

        if not isinstance(gender, str) or gender.lower() == "unknown":
            data = _query_gender_api(name)
            if data:
                gender = data.get("gender") or gender
                probability = data.get("probability")
                try:
                    confidence = float(probability) if probability is not None else None
                except (TypeError, ValueError):
                    confidence = None

        lookup[name] = (gender.lower() if isinstance(gender, str) else "unknown", confidence)

    inferred = []
    confidences = []
    for name in names:
        if not name:
            inferred.append("unknown")
            confidences.append(None)
            continue
        gender, confidence = lookup.get(name, ("unknown", None))
        inferred.append(gender or "unknown")
        confidences.append(confidence)

    df["gender_inferred"] = inferred
    if any(confidence is not None for confidence in confidences):
        df["gender_confidence"] = confidences

    return df


def enrich_from_zip(df: pd.DataFrame, overwrite_existing: bool = False) -> pd.DataFrame:
    if "zip" not in df.columns:
        return df

    zip_db = load_zip_database()
    if zip_db.empty:
        return df

    df = df.copy()
    df["zip"] = df["zip"].astype(str).str.zfill(5)

    enriched = df.merge(
        zip_db.rename(columns={"zip_code": "zip"}),
        on="zip",
        how="left",
        suffixes=("", "_lookup"),
    )

    for source, lookup in (("ct", "city"), ("st", "state")):
        if lookup not in enriched.columns:
            continue

        should_replace = overwrite_existing or source not in enriched.columns
        if not should_replace:
            missing_mask = enriched[source].isna() | (enriched[source].astype(str) == "")
        else:
            missing_mask = slice(None)

        enriched.loc[missing_mask, source] = enriched.loc[missing_mask, lookup]
        enriched = enriched.drop(columns=[lookup])

    return enriched
