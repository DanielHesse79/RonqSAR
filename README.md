# RonqSAR QSAR Template

A minimal, local-first QSAR template integrating RDKit and a local Ollama LLM advisor.

## Quickstart

```bash
pip install -r requirements.txt  # optional
python -m qsar.cli fit --config configs/esol.yaml
uvicorn qsar.serve.api:app --reload
```

The demo trains a LightGBM regressor on a small ESOL-like dataset using a scaffold split and generates a report under `artifacts/`.

## CLI

- `qsar fit --config configs/esol.yaml`
- `qsar predict --model artifacts/best_model.joblib --input new.csv --out preds.csv`

## Tests

Run unit tests with `pytest -q`.
