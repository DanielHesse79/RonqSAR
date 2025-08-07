"""Advisor sessions for querying Ollama LLM."""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from .ollama_client import generate

SYS_PROMPT = Path(__file__).with_name("system_prompt.txt").read_text()


def propose_smarts_filters(sample_smiles: List[str]) -> List[str]:
    prompt = (
        "Given the following SMILES list, suggest a few SMARTS patterns to filter common bad actors. "
        "Return a bullet list of SMARTS strings.\n" + "\n".join(sample_smiles)
    )
    text = generate(prompt, system=SYS_PROMPT)
    lines = [line.strip("-* \n") for line in text.splitlines() if line.strip()]
    return [l for l in lines if l]


def suggest_hparam_space(model_name: str) -> Dict[str, str]:
    prompt = f"Suggest Optuna hyperparameter search space for {model_name}. Return YAML mapping."
    text = generate(prompt, system=SYS_PROMPT)
    try:
        import yaml

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return {k: str(v) for k, v in data.items()}
    except Exception:
        pass
    return {"raw": text}


def explain_outliers(smiles_list: List[str], y_true: List[float], y_pred: List[float]) -> str:
    prompt = (
        "Explain possible reasons for the following prediction outliers in QSAR."\
        "\nSMILES, true, pred\n" + "\n".join(f"{s},{t},{p}" for s, t, p in zip(smiles_list, y_true, y_pred))
    )
    return generate(prompt, system=SYS_PROMPT)
