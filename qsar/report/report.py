"""Reporting utilities for QSAR runs."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import yaml


def generate_report(output: str | Path, config: Dict[str, Any], metrics: Dict[str, float]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = run_dir / "summary.md"
    with summary.open("w") as f:
        f.write("# QSAR Summary\n\n")
        f.write("## Configuration\n")
        yaml.safe_dump(config, f)
        f.write("\n## Metrics\n")
        for k, v in metrics.items():
            f.write(f"- {k}: {v:.4f}\n")
    return run_dir
