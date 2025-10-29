"""Data enrichment helpers."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
GENDER_API_ENDPOINT = "https://gender-api.com/get"
GENDER_API_KEY = os.getenv(
    "GENDER_API_KEY",
    "23f1759173b056d4c4ce5bb29f3366d01665ff2789c6f003110ce6a6e4173237",
)

LOGGER = logging.getLogger(__name__)


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


def derive_name_from_email(email: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(email, str) or "@" not in email:
        return None, None
    local_part = email.split("@", 1)[0]
    fragments = [fragment for fragment in local_part.replace(".", " ").split() if fragment]

    first = fragments[0].title() if fragments else None
    last = fragments[-1].title() if len(fragments) > 1 else None
    return first, last


def _normalise_name(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(part for part in value.strip().split() if part)
    return cleaned or None


@lru_cache(maxsize=4096)
def _query_gender_api(name: str) -> Optional[Dict[str, Any]]:
    if not name or not GENDER_API_KEY:
        return None

    params = {"name": name, "key": GENDER_API_KEY, "split": "true"}
    try:
        response = requests.get(GENDER_API_ENDPOINT, params=params, timeout=5)
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


def _extract_name_parts(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    first = data.get("firstname") or data.get("first_name")
    last = data.get("lastname") or data.get("last_name")

    if not first and isinstance(data.get("name"), str):
        pieces = [piece for piece in data["name"].split() if piece]
        if pieces:
            first = pieces[0]
            if len(pieces) > 1:
                last = pieces[-1]

    first = first.title() if isinstance(first, str) and first else None
    last = last.title() if isinstance(last, str) and last else None
    return first, last


def _fill_names_from_gender_api(df: pd.DataFrame) -> pd.DataFrame:
    needs_columns = {"fn", "ln", "full_name"}
    if not any(column in df.columns for column in needs_columns):
        return df

    df = df.copy()
    if "fn" not in df.columns:
        df["fn"] = pd.NA
    if "ln" not in df.columns:
        df["ln"] = pd.NA

    fn_missing = df["fn"].isna() | (df["fn"].astype(str).str.strip() == "")
    ln_missing = df["ln"].isna() | (df["ln"].astype(str).str.strip() == "")
    needs_lookup = fn_missing | ln_missing

    if not needs_lookup.any():
        return df

    name_to_indices: Dict[str, list[int]] = {}
    for idx in df.index[needs_lookup]:
        candidate: Optional[str] = None
        if "full_name" in df.columns:
            candidate = _normalise_name(df.at[idx, "full_name"])

        if not candidate:
            parts = []
            if not fn_missing.loc[idx] and isinstance(df.at[idx, "fn"], str):
                parts.append(str(df.at[idx, "fn"]))
            if not ln_missing.loc[idx] and isinstance(df.at[idx, "ln"], str):
                parts.append(str(df.at[idx, "ln"]))
            candidate = _normalise_name(" ".join(parts)) if parts else None

        if not candidate:
            continue

        name_to_indices.setdefault(candidate, []).append(idx)

    for name, indices in name_to_indices.items():
        data = _query_gender_api(name)
        if not data:
            continue
        first, last = _extract_name_parts(data)
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

    if "fn" not in df.columns:
        df["fn"] = pd.NA
    if "ln" not in df.columns:
        df["ln"] = pd.NA

    if "email" in df.columns:
        derived = df["email"].apply(derive_name_from_email)
        derived_df = pd.DataFrame(derived.tolist(), columns=["fn_email", "ln_email"], index=df.index)

        fn_missing = df["fn"].isna() | (df["fn"].astype(str).str.strip() == "")
        ln_missing = df["ln"].isna() | (df["ln"].astype(str).str.strip() == "")

        if "fn_email" in derived_df:
            df.loc[fn_missing, "fn"] = derived_df.loc[fn_missing, "fn_email"].apply(
                lambda value: value.title() if isinstance(value, str) else value
            )
        if "ln_email" in derived_df:
            df.loc[ln_missing, "ln"] = derived_df.loc[ln_missing, "ln_email"].apply(
                lambda value: value.title() if isinstance(value, str) else value
            )

    df = _fill_names_from_gender_api(df)

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
                accuracy = data.get("accuracy")
                try:
                    confidence = float(accuracy) if accuracy is not None else None
                except (TypeError, ValueError):
                    confidence = None

                # If we were missing names earlier, leverage the API split here too.
                if "ln" in df.columns or "fn" in df.columns:
                    first, last = _extract_name_parts(data)
                    mask = names == name
                    if first and "fn" in df.columns:
                        df.loc[
                            mask
                            & (df["fn"].isna() | (df["fn"].astype(str).str.strip() == "")),
                            "fn",
                        ] = first
                    if last and "ln" in df.columns:
                        df.loc[
                            mask
                            & (df["ln"].isna() | (df["ln"].astype(str).str.strip() == "")),
                            "ln",
                        ] = last

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
