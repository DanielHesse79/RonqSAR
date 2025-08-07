"""Inductive Conformal Prediction for regression."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ICPRegressor:
    alpha: float = 0.1

    def fit(self, y_calib: np.ndarray, y_pred_calib: np.ndarray) -> None:
        self.residuals_ = np.abs(y_calib - y_pred_calib)
        self.quantile_ = np.quantile(self.residuals_, 1 - self.alpha)

    def predict(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = y_pred - self.quantile_
        upper = y_pred + self.quantile_
        return lower, upper
