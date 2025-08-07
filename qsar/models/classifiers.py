"""Baseline classification models."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


CLASSIFIERS: Dict[str, Any] = {
    "random_forest": RandomForestClassifier,
}
if LGBMClassifier is not None:
    CLASSIFIERS["lightgbm"] = LGBMClassifier
if XGBClassifier is not None:
    CLASSIFIERS["xgboost"] = XGBClassifier


METRICS = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "auroc": roc_auc_score,
    "auprc": average_precision_score,
}
