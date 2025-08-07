from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from .config import load_config
from .utils import timed


@timed
@pd.api.extensions.register_dataframe_accessor("ns")
class Loader:
    pass  # placeholder to demonstrate accessor, not used


@timed
def load_dataframe(path: Optional[Path] = None) -> pd.DataFrame:
    cfg = load_config()
    path = path or Path(cfg.data.path)
    logger.info(f"Loading data from {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df
