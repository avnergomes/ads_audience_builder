"""TikTok audience export helpers."""

from __future__ import annotations

import pandas as pd

TIKTOK_COLUMNS = ["Email", "Phone", "FirstName", "LastName", "State", "City", "Zip", "Country", "Source"]


def export_tiktok(df: pd.DataFrame) -> pd.DataFrame:
    export_df = pd.DataFrame(columns=TIKTOK_COLUMNS)
    export_df["Email"] = df.get("email")
    export_df["Phone"] = df.get("phone")
    export_df["FirstName"] = df.get("fn")
    export_df["LastName"] = df.get("ln")
    export_df["State"] = df.get("st")
    export_df["City"] = df.get("ct")
    export_df["Zip"] = df.get("zip")
    export_df["Country"] = df.get("country")
    export_df["Source"] = df.get("source")
    return export_df
