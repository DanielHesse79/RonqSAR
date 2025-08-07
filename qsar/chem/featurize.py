"""Molecular featurization utilities."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.preprocessing import RobustScaler

try:
    from mordred import Calculator, descriptors
except Exception:  # pragma: no cover - mordred optional
    Calculator = None
    descriptors = None


def morgan_fingerprint(smiles: List[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan/ECFP fingerprints."""
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=int))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    return np.asarray(fps)


def maccs_keys(smiles: List[str]) -> np.ndarray:
    """Compute MACCS keys."""
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(167, dtype=int))
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr[1:])  # drop first bit per RDKit docs
    return np.asarray(fps)


def mordred_descriptors(smiles: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Compute Mordred descriptors with robust scaling."""
    if Calculator is None:
        raise ImportError("mordred package is required for descriptors")
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    calc = Calculator(descriptors, ignore_3D=True)
    df = calc.pandas(mols)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    scaler = RobustScaler().fit(df.values)
    arr = scaler.transform(df.values)
    return arr.astype(float), list(df.columns)
