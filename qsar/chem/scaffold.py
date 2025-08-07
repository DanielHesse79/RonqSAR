"""Scaffold utilities including Bemis-Murcko scaffold split."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


@dataclass
class SplitIndices:
    train: np.ndarray
    test: np.ndarray


def bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core)


def scaffold_split(smiles: Sequence[str], test_size: float = 0.2, seed: int = 0) -> SplitIndices:
    """Split dataset by scaffolds."""
    scaffolds = {}
    for idx, smi in enumerate(smiles):
        scaf = bemis_murcko_scaffold(smi)
        scaffolds.setdefault(scaf, []).append(idx)
    rng = np.random.default_rng(seed)
    scaffold_keys = list(scaffolds.keys())
    rng.shuffle(scaffold_keys)
    n_total = len(smiles)
    n_test = int(n_total * test_size)
    test_indices = []
    for scaf in scaffold_keys:
        if len(test_indices) >= n_test:
            break
        test_indices.extend(scaffolds[scaf])
    train_indices = [i for i in range(n_total) if i not in test_indices]
    return SplitIndices(train=np.array(train_indices), test=np.array(test_indices))


def random_split(n: int, test_size: float = 0.2, seed: int = 0) -> SplitIndices:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_size)
    return SplitIndices(train=idx[n_test:], test=idx[:n_test])
