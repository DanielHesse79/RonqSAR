"""FastAPI service for QSAR predictions."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from ..chem.standardize import standardize_smiles
from ..chem.featurize import morgan_fingerprint, maccs_keys

app = FastAPI(title="QSAR Service")


class PredictRequest(BaseModel):
    smiles: List[str]


class PredictRecord(BaseModel):
    mean: float
    pi_low: float
    pi_high: float
    in_domain: bool
    notes: str | None = None


MODEL_PATH = Path(os.getenv("QSAR_MODEL", "artifacts/best_model.joblib"))
ARTIFACT = joblib.load(MODEL_PATH)
MODEL = ARTIFACT["model"]
FEATURE_CFG = ARTIFACT.get("features", {})
ICP_Q = ARTIFACT.get("icp_quantile", 0.0)


def _featurize(smiles: list[str]) -> np.ndarray:
    feats = []
    if "ecfp4" in FEATURE_CFG:
        feats.append(morgan_fingerprint(smiles, radius=2, n_bits=FEATURE_CFG.get("ecfp4", 2048)))
    if "maccs" in FEATURE_CFG:
        feats.append(maccs_keys(smiles))
    return np.hstack(feats)


@app.post("/predict", response_model=List[PredictRecord])
def predict(req: PredictRequest) -> List[PredictRecord]:
    smi_std = [standardize_smiles(s) or "" for s in req.smiles]
    X = _featurize(smi_std)
    preds = MODEL.predict(X)
    results = []
    for p in preds:
        low = p - ICP_Q
        high = p + ICP_Q
        results.append(PredictRecord(mean=float(p), pi_low=float(low), pi_high=float(high), in_domain=True))
    return results
