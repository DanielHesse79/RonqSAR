from qsar.chem.scaffold import scaffold_split, bemis_murcko_scaffold


def test_scaffold_split_no_leakage():
    smiles = [
        "CCO",
        "CCN",
        "CCC",
        "c1ccccc1",
        "c1ccncc1",
        "CC(=O)O",
        "CCOC(=O)C",
        "CCSC",
    ]
    split = scaffold_split(smiles, test_size=0.25, seed=0)
    sc_train = {bemis_murcko_scaffold(smiles[i]) for i in split.train}
    sc_test = {bemis_murcko_scaffold(smiles[i]) for i in split.test}
    assert sc_train.isdisjoint(sc_test)
