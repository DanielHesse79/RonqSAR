from __future__ import annotations

from pathlib import Path
from typing import List

import toml
from pydantic import BaseModel, Field

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"


class AppConfig(BaseModel):
    title: str = "NarrowSearchAI"
    theme: str = "auto"
    results_per_page: int = 25


class SearchConfig(BaseModel):
    use_semantic: bool = True
    embedder: str = "all-MiniLM-L6-v2"
    top_k: int = 50
    w_sim: float = 0.6
    w_attr: float = 0.3
    w_recency: float = 0.1


class DataConfig(BaseModel):
    path: str
    text_fields: List[str]


class Config(BaseModel):
    app: AppConfig
    search: SearchConfig
    data: DataConfig


_DEF = Config(
    app=AppConfig(),
    search=SearchConfig(),
    data=DataConfig(path="data/sample.csv", text_fields=["title", "text"]),
)


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_PATH
    if path.exists():
        data = toml.load(path)
        return Config(**data)
    return _DEF


def save_config(cfg: Config, path: Path | None = None) -> None:
    path = path or CONFIG_PATH
    with open(path, "w") as f:
        toml.dump(cfg.dict(), f)
