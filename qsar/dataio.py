"""Data loading and preprocessing utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from rdkit import Chem

from .chem.standardize import standardize_dataset
from .chem.featurize import morgan_fingerprint, maccs_keys, mordred_descriptors
from advisor.sessions import propose_smarts_filters


def read_input(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    if "smiles" not in df.columns:
        raise KeyError("Input must contain a 'smiles' column")
    return df


def standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = standardize_dataset(df["smiles"].tolist())
    df = df.iloc[result.valid_idx].reset_index(drop=True)
    df["smiles"] = result.smiles
    if result.failed:
        fail_pct = len(result.failed) / (len(result.smiles) + len(result.failed))
        examples = "\n".join(result.failed[:5])
        Path("data_lint.md").write_text(
            f"Failed to parse {fail_pct:.2%} SMILES\n\nExamples:\n{examples}\n"
        )
        if fail_pct > 0.02:
            raise ValueError("Too many SMILES failed standardization")
    # Query advisor for SMARTS filters suggestions
    suggestions = propose_smarts_filters(df["smiles"].head(20).tolist())
    if suggestions:
        import yaml

        filters_path = Path("filters/smarts.yaml")
        filters_path.parent.mkdir(exist_ok=True)
        with filters_path.open("w") as f:
            yaml.safe_dump({"suggestions": suggestions}, f)
    return df
