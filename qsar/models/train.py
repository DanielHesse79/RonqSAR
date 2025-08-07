"""Training pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import yaml

from ..dataio import read_input, standardize_frame
from ..chem.featurize import morgan_fingerprint, maccs_keys
from ..chem.scaffold import scaffold_split, random_split
from ..report.report import generate_report
from .regressors import REGRESSORS, METRICS
from ..tuning.optuna_search import tune


def featurize(smiles: list[str], cfg: Dict[str, Any]) -> np.ndarray:
    feats = []
    if "ecfp4" in cfg:
        feats.append(morgan_fingerprint(smiles, radius=2, n_bits=cfg.get("ecfp4", 2048)))
    if "maccs" in cfg:
        feats.append(maccs_keys(smiles))
    if not feats:
        raise ValueError("No features specified")
    return np.hstack(feats)


def train(config_path: str) -> Dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text())
    df = read_input(config["data"])
    df = standardize_frame(df)
    smiles = df["smiles"].tolist()
    target_col = config.get("target", "y")
    y = df[target_col].to_numpy()
    X = featurize(smiles, config.get("features", {}))

    split_type = config.get("split", "scaffold")
    test_size = config.get("test_size", 0.2)
    seed = config.get("seed", 0)
    if split_type == "scaffold":
        split = scaffold_split(smiles, test_size=test_size, seed=seed)
    else:
        split = random_split(len(smiles), test_size=test_size, seed=seed)
    X_train_full, y_train_full = X[split.train], y[split.train]
    X_test, y_test = X[split.test], y[split.test]

    # split calibration set
    n_calib = max(1, int(0.2 * len(X_train_full)))
    X_calib, y_calib = X_train_full[:n_calib], y_train_full[:n_calib]
    X_train, y_train = X_train_full[n_calib:], y_train_full[n_calib:]

    model_name = config.get("model", "lightgbm")
    Model = REGRESSORS[model_name]

    def objective(trial):
        params = {}
        if model_name == "lightgbm":
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        model = Model(random_state=seed, **params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        from sklearn.metrics import mean_squared_error

        return mean_squared_error(y_test, pred, squared=False)

    n_trials = config.get("n_trials", 5)
    study = tune(objective, n_trials=n_trials, seed=seed)
    best_params = study.best_params
    model = Model(random_state=seed, **best_params)
    model.fit(X_train, y_train)
    preds_calib = model.predict(X_calib)
    resid = np.abs(y_calib - preds_calib)
    quantile = float(np.quantile(resid, 1 - config.get("alpha", 0.1)))
    model.fit(X_train_full, y_train_full)
    preds = model.predict(X_test)

    metrics = {name: fn(y_test, preds) for name, fn in METRICS.items()}
    out_dir = config.get("output", "artifacts")
    Path(out_dir).mkdir(exist_ok=True)
    model_path = Path(out_dir) / "best_model.joblib"
    joblib.dump({"model": model, "features": config.get("features", {}), "icp_quantile": quantile}, model_path)
    run_dir = generate_report(out_dir, config, metrics)
    card = Path(out_dir) / "model_card.md"
    with card.open("w") as f:
        f.write("# Model Card\n\n")
        f.write(f"Model: {model_name}\n\n")
        f.write(f"Training data: {config['data']}\n")
    return {"model_path": str(model_path), "metrics": metrics}
