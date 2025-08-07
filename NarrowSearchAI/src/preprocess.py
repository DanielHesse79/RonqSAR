from __future__ import annotations

import json
from datetime import datetime
from typing import List

import pandas as pd


def normalize_tags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tags"] = df["tags"].fillna("").apply(lambda x: [t.strip() for t in x.split(";") if t])
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def parse_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["meta"] = df["meta"].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_tags(df)
    df = parse_dates(df)
    df = parse_meta(df)
    return df
