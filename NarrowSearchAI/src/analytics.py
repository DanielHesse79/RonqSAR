from __future__ import annotations

import pandas as pd


def by_category(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("category").size().reset_index(name="count")


def tag_frequency(df: pd.DataFrame) -> pd.DataFrame:
    explode = df.explode("tags")
    return explode.groupby("tags").size().reset_index(name="count")


def by_time(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    series = df.set_index("date").resample(freq).size()
    return series.reset_index(name="count")


def score_vs_date(df: pd.DataFrame) -> pd.DataFrame:
    return df[["date", "score", "category"]]
