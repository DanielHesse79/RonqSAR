import pandas as pd
from src.filters import apply_filters


def make_df():
    data = {
        "id": [1, 2],
        "category": ["a", "b"],
        "tags": [["x"], ["y"]],
        "date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "score": [10, 20],
        "meta": [{"m": 1}, {"m": 2}],
    }
    return pd.DataFrame(data)


def test_category_filter():
    df = make_df()
    res = apply_filters(df, categories=["a"])
    assert len(res) == 1 and res.iloc[0]["id"] == 1


def test_score_range():
    df = make_df()
    res = apply_filters(df, score_range=(15, 25))
    assert len(res) == 1 and res.iloc[0]["id"] == 2
