import numpy as np
import pandas as pd
from src.search import rank_results


def test_rank_results():
    df = pd.DataFrame({"id": [1, 2, 3]})
    indices = [2, 0]
    dists = [0.1, 0.5]
    res = rank_results(df, indices, dists)
    assert res.iloc[0]["id"] == 3
    assert res.iloc[1]["id"] == 1
