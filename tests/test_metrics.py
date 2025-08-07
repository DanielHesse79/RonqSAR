import yaml
from pathlib import Path

from qsar.models.train import train


def test_metric_reproducibility(tmp_path):
    cfg = {
        "data": str(Path("data/esol.csv")),
        "target": "y",
        "model": "lightgbm",
        "features": {"ecfp4": 2048},
        "split": "scaffold",
        "seed": 1,
        "n_trials": 1,
        "output": str(tmp_path),
    }
    cfg_path = tmp_path / "cfg.yaml"
    yaml.safe_dump(cfg, cfg_path.open("w"))
    res1 = train(str(cfg_path))
    res2 = train(str(cfg_path))
    assert res1["metrics"]["rmse"] == res2["metrics"]["rmse"]
