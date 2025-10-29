import pandas as pd

def split_audience(df: pd.DataFrame, fraction=0.5):
    half = int(len(df) * fraction)
    return df.iloc[:half], df.iloc[half:]
