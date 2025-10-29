"""Meta export helpers."""

from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd

DEFAULT_COLUMNS = [
    "email",
    "phone",
    "fn",
    "ln",
    "ct",
    "st",
    "zip",
    "country",
    "meta_lead_id",
    "source",
]


def hash_field(value: str) -> str:
    if pd.isna(value) or value == "":
        return ""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def export_meta(df: pd.DataFrame, hash_fields: bool = False) -> pd.DataFrame:
    meta_df = ensure_columns(df.copy(), DEFAULT_COLUMNS)

    if hash_fields:
        for column in ("email", "phone", "fn", "ln", "ct", "st", "zip", "country"):
            if column in meta_df.columns:
                meta_df[column] = meta_df[column].apply(hash_field)

    # Meta expects specific headers (snake case is acceptable).
    return meta_df[DEFAULT_COLUMNS]
