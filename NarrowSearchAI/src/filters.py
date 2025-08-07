from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    categories: Optional[Iterable[str]] = None,
    tags: Optional[Iterable[str]] = None,
    date_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
    score_range: Optional[tuple[float, float]] = None,
    meta_key: Optional[str] = None,
    meta_val: Optional[str] = None,
) -> pd.DataFrame:
    res = df
    if categories:
        res = res[res["category"].isin(categories)]
    if tags:
        res = res[res["tags"].apply(lambda ts: set(tags).issubset(ts))]
    if date_range:
        start, end = date_range
        res = res[(res["date"] >= start) & (res["date"] <= end)]
    if score_range:
        low, high = score_range
        res = res[(res["score"] >= low) & (res["score"] <= high)]
    if meta_key and meta_val:
        res = res[res["meta"].apply(lambda m: meta_key in m and meta_val in str(m[meta_key]))]
    return res
