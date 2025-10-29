"""Google Customer Match export helpers."""

from __future__ import annotations

import pandas as pd

GOOGLE_COLUMNS = [
    "Email",
    "Phone",
    "FirstName",
    "LastName",
    "CountryCode",
    "Zip",
    "State",
    "City",
    "Source",
]


def export_google(df: pd.DataFrame) -> pd.DataFrame:
    """Export DataFrame in Google Customer Match format.
    
    Args:
        df: Cleaned DataFrame with standardized columns
    
    Returns:
        DataFrame formatted for Google Customer Match upload
    """
    export_df = pd.DataFrame()
    export_df["Email"] = df["email"] if "email" in df.columns else ""
    export_df["Phone"] = df["phone"] if "phone" in df.columns else ""
    export_df["FirstName"] = df["fn"] if "fn" in df.columns else ""
    export_df["LastName"] = df["ln"] if "ln" in df.columns else ""
    export_df["CountryCode"] = df["country"] if "country" in df.columns else ""
    export_df["Zip"] = df["zip"] if "zip" in df.columns else ""
    export_df["State"] = df["st"] if "st" in df.columns else ""
    export_df["City"] = df["ct"] if "ct" in df.columns else ""
    export_df["Source"] = df["source"] if "source" in df.columns else ""
    
    # Fill NaN values with empty strings
    export_df = export_df.fillna("")
    return export_df[GOOGLE_COLUMNS]
