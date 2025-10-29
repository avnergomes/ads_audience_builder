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
    export_df = pd.DataFrame(columns=GOOGLE_COLUMNS)
    export_df["Email"] = df.get("email")
    export_df["Phone"] = df.get("phone")
    export_df["FirstName"] = df.get("fn")
    export_df["LastName"] = df.get("ln")
    export_df["CountryCode"] = df.get("country")
    export_df["Zip"] = df.get("zip")
    export_df["State"] = df.get("st")
    export_df["City"] = df.get("ct")
    export_df["Source"] = df.get("source")
    return export_df
