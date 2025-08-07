"""Command line interface for QSAR template."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from .models.train import train as train_model
from .chem.standardize import standardize_smiles
from .chem.featurize import morgan_fingerprint, maccs_keys


def cmd_fit(args: argparse.Namespace) -> None:
    train_model(args.config)


def cmd_predict(args: argparse.Namespace) -> None:
    artifact = joblib.load(args.model)
    model = artifact["model"]
    feature_cfg = artifact.get("features", {})
    icp_q = artifact.get("icp_quantile", 0.0)
    df = csv.DictReader(open(args.input))
    smiles = [standardize_smiles(row["smiles"]) or "" for row in df]
    feats = []
    if "ecfp4" in feature_cfg:
        feats.append(morgan_fingerprint(smiles, radius=2, n_bits=feature_cfg.get("ecfp4", 2048)))
    if "maccs" in feature_cfg:
        feats.append(maccs_keys(smiles))
    X = np.hstack(feats)
    preds = model.predict(X)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "prediction", "pi_low", "pi_high"])
        for smi, p in zip(smiles, preds):
            writer.writerow([smi, p, p - icp_q, p + icp_q])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="qsar")
    sub = parser.add_subparsers(dest="cmd")

    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--config", required=True)
    p_fit.set_defaults(func=cmd_fit)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--input", required=True)
    p_pred.add_argument("--out", required=True)
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
