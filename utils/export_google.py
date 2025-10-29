import pandas as pd

def export_google(df: pd.DataFrame) -> pd.DataFrame:
    google_cols = ['Email', 'Phone', 'CountryCode', 'Zip']
    export_df = pd.DataFrame(columns=google_cols)
    export_df['Email'] = df.get('email')
    export_df['Phone'] = df.get('phone')
    export_df['CountryCode'] = df.get('country')
    export_df['Zip'] = df.get('zip')
    return export_df
