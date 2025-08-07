"""Utilities for SHAP explanations."""
from __future__ import annotations

import numpy as np

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


def compute_shap_values(model, X: np.ndarray) -> np.ndarray:
    if shap is None:
        raise ImportError("shap is required for explanations")
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)
