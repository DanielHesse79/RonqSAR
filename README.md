# RonqSAR

Local-first QSAR template with RDKit featurization, classic ML models (LightGBM/XGBoost/RandomForest), a FastAPI prediction service, and an optional UI demo (NarrowSearchAI). Includes a lightweight LLM advisor via local Ollama.

### What’s inside
- QSAR CLI pipeline: standardization, featurization (ECFP4/MACCS), scaffold split, Optuna tuning, conformal intervals, reporting
- FastAPI service for batch predictions
- Optional Streamlit UI (separate app under `NarrowSearchAI/`)
- Local LLM advisor hooks for SMARTS suggestions

## Prerequisites
- Python 3.10–3.13 (Windows: use `py` launcher)
- RDKit (installed via `pip install rdkit-pypi`)
- Optional: `lightgbm`, `xgboost`, `optuna`, `uvicorn[standard]`

## Quickstart (CLI)
```bash
pip install -r requirements.txt
py -m qsar.cli fit --config configs/esol.yaml   # Windows
# or: python -m qsar.cli fit --config configs/esol.yaml
```

Artifacts:
- `artifacts/best_model.joblib`
- `artifacts/run_YYYYMMDD_HHMMSS/summary.md`

## Prediction (CLI)
```bash
py -m qsar.cli predict --model artifacts/best_model.joblib \
  --input data/esol.csv --out preds.csv
```

## FastAPI service
```bash
py -m pip install uvicorn fastapi
py -m uvicorn qsar.serve.api:app --reload
```
Environment:
- `QSAR_MODEL` to point at a model artifact (default `artifacts/best_model.joblib`)

## UI (optional)
The demo UI lives in `NarrowSearchAI/` and runs on port 8502 by default.

Quick start on Windows:
```bash
./start_ui.bat
# or manually
py -m pip install -r NarrowSearchAI/requirements.txt
cd NarrowSearchAI
py -m streamlit run ui/app.py --server.port 8502
```

See `UI_STARTUP_GUIDE.md` for detailed steps and troubleshooting.

## Configuration
Example: `configs/esol.yaml`
```yaml
data: data/esol.csv
target: y
model: lightgbm
features:
  ecfp4: 2048
split: scaffold
test_size: 0.2
seed: 0
n_trials: 1
output: artifacts
alpha: 0.1
```

## Tests
```bash
py -m pip install pytest
pytest -q
```

## Troubleshooting (Windows)
- If `python` is not found, use `py` instead
- RDKit install issues: `py -m pip install rdkit-pypi`
- Port busy: change `--server.port` for Streamlit; set `UVICORN_PORT` (or change CLI arg) for API

## License
MIT
