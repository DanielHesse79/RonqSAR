"""Map ECFP bits back to atom environments."""
from __future__ import annotations

from typing import Dict, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem


def bit_substructure(smiles: str, bit_id: int, radius: int = 2) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    info: Dict[int, Tuple[int, int]] = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048, bitInfo=info)
    if bit_id not in info:
        return None
    atoms, rad = info[bit_id][0]
    env = AllChem.FindAtomEnvironmentOfRadiusN(mol, rad, atoms)
    amap = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
    return submol
