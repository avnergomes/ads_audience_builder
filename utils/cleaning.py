import pandas as pd
import re

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'e-mail': 'email', 'mail': 'email', 'phone number': 'phone',
        'first name': 'fn', 'last name': 'ln'
    }
    df.rename(columns={c: rename_map.get(c.lower(), c.lower()) for c in df.columns}, inplace=True)
    return df

def validate_email(email: str) -> bool:
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", str(email)))

def clean_emails(df: pd.DataFrame) -> pd.DataFrame:
    if "email" in df.columns:
        df["email"] = df["email"].astype(str).str.lower().str.strip()
        df = df[df["email"].apply(validate_email)]
    return df
