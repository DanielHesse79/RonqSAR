"""Domain of applicability via Tanimoto kNN."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from rdkit import Chem, DataStructs


@dataclass
class KNNDomain:
    k: int = 5
    percentile: float = 95.0

    def fit(self, train_fps: List[Chem.rdchem.Mol]) -> None:
        self.train_fps = train_fps
        dists = []
        for fp in train_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            sims = sorted(sims, reverse=True)[1 : self.k + 1]
            dists.append(1 - np.mean(sims))
        self.threshold_ = np.percentile(dists, self.percentile)

    def predict(self, fps: List[Chem.rdchem.Mol]) -> np.ndarray:
        flags = []
        for fp in fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, self.train_fps)
            sims = sorted(sims, reverse=True)[: self.k]
            dist = 1 - np.mean(sims)
            flags.append(dist <= self.threshold_)
        return np.asarray(flags)
