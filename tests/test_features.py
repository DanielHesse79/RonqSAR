import numpy as np
from qsar.chem.featurize import morgan_fingerprint, maccs_keys


def test_feature_shapes():
    smiles = ["CCO", "c1ccccc1"]
    X1 = morgan_fingerprint(smiles)
    X2 = maccs_keys(smiles)
    assert X1.shape == (2, 2048)
    assert X2.shape == (2, 166)
    assert X1.dtype == int
    assert X2.dtype == int
