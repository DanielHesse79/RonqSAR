from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

from loguru import logger
from sklearn.neighbors import NearestNeighbors

from .config import load_config
from .embedder import load_embedder
from .load_data import load_dataframe
from .preprocess import preprocess

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


class VectorStore:
    def __init__(self, index, embeddings: np.ndarray):
        self.index = index
        self.embeddings = embeddings

    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if faiss and isinstance(self.index, faiss.Index):
            D, I = self.index.search(query_vec, k)
            return D[0], I[0]
        distances, indices = self.index.kneighbors(query_vec, n_neighbors=k)
        return distances[0], indices[0]


def build_index(embedder_name: str | None = None) -> VectorStore:
    cfg = load_config()
    df = preprocess(load_dataframe())
    texts = df[cfg.data.text_fields].fillna("").agg(" ".join, axis=1).tolist()
    embedder = load_embedder(embedder_name or cfg.search.embedder)
    if embedder is None:
        raise RuntimeError("Embedder required to build index")
    if embedder.name == "tfidf":
        embedder.model.fit(texts)
    embeddings = embedder.embed(texts)
    if faiss:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
    else:
        index = NearestNeighbors(metric="cosine")
        index.fit(embeddings)
    np.save(CACHE_DIR / "embeddings.npy", embeddings)
    return VectorStore(index, embeddings)


def load_index() -> Optional[VectorStore]:
    emb_path = CACHE_DIR / "embeddings.npy"
    if not emb_path.exists():
        return None
    embeddings = np.load(emb_path)
    if faiss:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
    else:
        index = NearestNeighbors(metric="cosine")
        index.fit(embeddings)
    return VectorStore(index, embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        build_index()
        logger.info("Index built")
