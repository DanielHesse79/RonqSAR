"""Baseline regression models."""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


REGRESSORS: Dict[str, Any] = {
    "random_forest": RandomForestRegressor,
}
if LGBMRegressor is not None:
    REGRESSORS["lightgbm"] = LGBMRegressor
if XGBRegressor is not None:
    REGRESSORS["xgboost"] = XGBRegressor


METRICS = {
    "rmse": lambda y, p: mean_squared_error(y, p, squared=False),
    "mae": mean_absolute_error,
    "r2": r2_score,
}
