"""Data enrichment helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


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

    first = fragments[0].capitalize() if fragments else None
    last = fragments[-1].capitalize() if len(fragments) > 1 else None
    return first, last


def autofill_names(df: pd.DataFrame) -> pd.DataFrame:
    if "email" not in df.columns:
        return df

    derived = df["email"].apply(derive_name_from_email)
    df[["fn", "ln"]] = pd.DataFrame(derived.tolist(), index=df.index)
    return df


def infer_gender(df: pd.DataFrame) -> pd.DataFrame:
    dictionary = load_gender_dictionary()
    if not dictionary or "fn" not in df.columns:
        return df

    df["gender_inferred"] = (
        df["fn"].astype(str).str.lower().map(dictionary).fillna("unknown")
    )
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
