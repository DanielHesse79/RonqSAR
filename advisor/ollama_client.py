"""Client for local Ollama server."""
from __future__ import annotations

import os
from typing import Dict

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "biomistral:7b-instruct")


def generate(prompt: str, system: str = "") -> str:
    payload = {"model": MODEL_NAME, "prompt": prompt, "system": system}
    r = requests.post(OLLAMA_URL, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")
