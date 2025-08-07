"""Molecule standardization utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolStandardize


@dataclass
class StandardizationResult:
    """Holds standardized molecules and statistics."""
    smiles: List[str]
    valid_idx: List[int]
    failed: List[str]


def standardize_smiles(smiles: str) -> str | None:
    """Standardize a SMILES string.

    Steps:
    - Parse SMILES and sanitize
    - Remove salts and small fragments
    - Uncharge / reionize
    - Return canonical SMILES
    """

    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        # Remove salts / fragments
        lfrag = rdMolStandardize.LargestFragmentChooser()
        mol = lfrag.choose(mol)
        # Reionize / neutralize
        reionizer = rdMolStandardize.Reionizer()
        mol = reionizer.reionize(mol)
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def standardize_dataset(smiles_list: Iterable[str]) -> StandardizationResult:
    """Standardize an iterable of SMILES.

    Returns standardized SMILES, their original indices, and list of failed
    SMILES strings.
    """

    good: List[str] = []
    valid_idx: List[int] = []
    bad: List[str] = []
    for idx, smi in enumerate(smiles_list):
        std = standardize_smiles(smi)
        if std is None:
            bad.append(smi)
        else:
            good.append(std)
            valid_idx.append(idx)
    return StandardizationResult(smiles=good, valid_idx=valid_idx, failed=bad)
