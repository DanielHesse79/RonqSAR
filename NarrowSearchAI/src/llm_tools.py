"""Optional helper to summarize records via local LLM endpoint.
Currently only a placeholder.
"""
from __future__ import annotations

from typing import List

import requests


def summarize(texts: List[str], endpoint: str = "http://localhost:11434/summarize") -> List[str]:
    summaries = []
    for t in texts:
        try:
            resp = requests.post(endpoint, json={"text": t}, timeout=10)
            summaries.append(resp.json().get("summary", ""))
        except Exception:
            summaries.append("")
    return summaries
