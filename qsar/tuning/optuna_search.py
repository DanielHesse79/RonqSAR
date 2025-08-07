"""Optuna hyperparameter search utilities."""
from __future__ import annotations

from typing import Callable, Dict, Any

import optuna


def tune(objective: Callable[[optuna.Trial], float], n_trials: int, seed: int = 0) -> optuna.Study:
    """Run an Optuna study with fixed seed."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study
