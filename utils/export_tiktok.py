import pandas as pd

def export_tiktok(df: pd.DataFrame) -> pd.DataFrame:
    tiktok_cols = ['Email', 'Phone']
    export_df = pd.DataFrame(columns=tiktok_cols)
    export_df['Email'] = df.get('email')
    export_df['Phone'] = df.get('phone')
    return export_df
