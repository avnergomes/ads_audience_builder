import pandas as pd, hashlib

def hash_field(value: str) -> str:
    if pd.isna(value): return None
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()

def export_meta(df: pd.DataFrame, hash_fields=False) -> pd.DataFrame:
    meta_df = df.copy()
    if hash_fields:
        for col in ['email', 'phone']:
            if col in meta_df.columns:
                meta_df[col] = meta_df[col].apply(hash_field)
    # Meta format compliance (from guidelines)
    required_cols = ['email', 'phone', 'fn', 'ln', 'ct', 'st', 'country']
    for c in required_cols:
        if c not in meta_df.columns:
            meta_df[c] = None
    return meta_df[required_cols]
