from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


@dataclass
class Embedder:
    name: str
    model: object

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.name == "tfidf":
            return self.model.transform(texts).toarray().astype("float32")
        return self.model.encode(texts, show_progress_bar=False).astype("float32")


def load_embedder(name: str) -> Embedder | None:
    if name == "none":
        return None
    if name == "tfidf":
        vectorizer = TfidfVectorizer(stop_words="english")
        return Embedder("tfidf", vectorizer)
    if SentenceTransformer is None:
        logger.warning("sentence-transformers not installed; falling back to tfidf")
        return load_embedder("tfidf")
    model = SentenceTransformer(name)
    return Embedder(name, model)
