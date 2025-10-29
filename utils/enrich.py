import pandas as pd

def derive_name_from_email(email):
    if pd.isna(email): return None, None
    parts = email.split('@')[0].replace('.', ' ').split()
    return (parts[0].capitalize() if parts else None,
            parts[-1].capitalize() if len(parts) > 1 else None)

def autofill_names(df: pd.DataFrame) -> pd.DataFrame:
    if "email" in df.columns:
        df[["fn", "ln"]] = df["email"].apply(
            lambda e: pd.Series(derive_name_from_email(e))
        )
    return df
