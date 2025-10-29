"""Meta export helpers.

Provides functions to export audience data in Meta (Facebook/Instagram) Ads format,
with optional SHA-256 hashing for personally identifiable information.
"""

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
    """Hash a field value using SHA-256.
    
    Args:
        value: The value to hash
    
    Returns:
        SHA-256 hexdigest of the value, or empty string if value is empty/null
    """
    if pd.isna(value) or value == "":
        return ""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Ensure all required columns exist in the DataFrame.
    
    Args:
        df: Input DataFrame
        columns: List of column names to ensure exist
    
    Returns:
        DataFrame with all required columns (missing ones filled with empty strings)
    """
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def export_meta(df: pd.DataFrame, hash_fields: bool = False) -> pd.DataFrame:
    """Export DataFrame in Meta Ads format.
    
    Args:
        df: Cleaned DataFrame with standardized columns
        hash_fields: If True, hash PII fields with SHA-256 (recommended for privacy)
    
    Returns:
        DataFrame formatted for Meta Ads Custom Audiences upload
    
    Note:
        Meta accepts both hashed and unhashed data, but hashing is recommended
        for privacy. The hash should be done client-side (as this function does)
        before uploading to Meta.
    """
    meta_df = ensure_columns(df.copy(), DEFAULT_COLUMNS)

    if hash_fields:
        # Hash all PII fields for privacy
        pii_columns = ("email", "phone", "fn", "ln", "ct", "st", "zip", "country")
        for column in pii_columns:
            if column in meta_df.columns:
                meta_df[column] = meta_df[column].apply(hash_field)

    # Fill NaN values with empty strings
    meta_df = meta_df.fillna("")
    
    # Meta expects specific headers (snake case is acceptable)
    return meta_df[DEFAULT_COLUMNS]
