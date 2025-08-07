from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .config import load_config
from .embedder import load_embedder
from .filters import apply_filters
from .load_data import load_dataframe
from .preprocess import preprocess
from .vector_store import build_index, load_index


_DEF_TOPK = 10


def semantic_search(query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    cfg = load_config()
    vs = load_index()
    if vs is None:
        vs = build_index(cfg.search.embedder)
    embedder = load_embedder(cfg.search.embedder)
    if embedder.name == "tfidf" and hasattr(embedder.model, "fit"):
        df = preprocess(load_dataframe())
        texts = df[cfg.data.text_fields].fillna("").agg(" ".join, axis=1).tolist()
        embedder.model.fit(texts)
    qvec = embedder.embed([query])
    if vs.index.__class__.__name__ == "IndexFlatIP":
        import faiss

        faiss.normalize_L2(qvec)
    distances, indices = vs.search(qvec, top_k)
    return distances, indices


def rank_results(df: pd.DataFrame, indices: Iterable[int], distances: Iterable[float]) -> pd.DataFrame:
    res = df.iloc[list(indices)].copy()
    res["similarity"] = 1 - np.array(distances)
    res.sort_values("similarity", ascending=False, inplace=True)
    return res


def search(
    query: str | None = None,
    categories: Iterable[str] | None = None,
    tags: Iterable[str] | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    score_range: tuple[float, float] | None = None,
    meta_key: str | None = None,
    meta_val: str | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    cfg = load_config()
    df = preprocess(load_dataframe())
    filtered = apply_filters(df, categories, tags, date_range, score_range, meta_key, meta_val)
    if query and cfg.search.use_semantic:
        distances, idx = semantic_search(query, top_k or cfg.search.top_k)
        sub = rank_results(filtered, idx, distances)
        return sub
    return filtered.head(top_k or _DEF_TOPK)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--top_k", type=int, default=_DEF_TOPK)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    res = search(args.query or None, top_k=args.top_k)
    if args.export:
        out = Path("export.csv")
        res.to_csv(out, index=False)
        logger.info(f"Exported to {out}")
    else:
        print(res[["id", "title", "score"]].head())
