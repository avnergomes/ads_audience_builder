"""Audience segmentation helpers."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


def tag_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if not source:
        return df
    df = df.copy()
    df["source"] = source
    return df


def create_ab_segments(df: pd.DataFrame, fraction: float = 0.5) -> Dict[str, pd.DataFrame]:
    fraction = max(0.05, min(0.95, fraction))
    pivot = int(len(df) * fraction)
    return {
        "Segment_A": df.iloc[:pivot].copy(),
        "Segment_B": df.iloc[pivot:].copy(),
    }


def available_segments(df: pd.DataFrame, ab_fraction: float, enable_ab: bool) -> Dict[str, pd.DataFrame]:
    if not enable_ab or len(df) == 0:
        return {"Audience": df}
    return create_ab_segments(df, ab_fraction)


def tag_segments(segments: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    tagged: Dict[str, pd.DataFrame] = {}
    for name, segment in segments.items():
        tagged_segment = segment.copy()
        tagged_segment["segment"] = name
        tagged[name] = tagged_segment
    return tagged


def merge_segments(segments: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not segments:
        return pd.DataFrame()
    return pd.concat(list(segments.values()), ignore_index=True)


# Backwards compatibility with initial helper
def split_audience(df: pd.DataFrame, fraction: float = 0.5) -> List[pd.DataFrame]:
    segments = create_ab_segments(df, fraction)
    return [segments["Segment_A"], segments["Segment_B"]]
