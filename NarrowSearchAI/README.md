# NarrowSearchAI

Streamlit UI demo bundled with RonqSAR. Provides faceted filtering, optional semantic search, analytics, and export over a small sample dataset.

## Features
- Filter by category, tags, date range, score, and meta fields
- Optional semantic search via sentence-transformers or TF‑IDF
- Interactive analytics (Plotly)
- CSV export
- Molecular tools panel integrates with `qsar/` utilities when RDKit is available

## Quick start (Windows)
```bash
py -m pip install -r requirements.txt
py -m streamlit run ui/app.py --server.port 8502
```

Alternatively, from repo root use:
```bash
./start_ui.bat
```

## Config
`config.toml` controls app behavior:
- `search.use_semantic`: enable semantic search
- `search.embedder`: `all-MiniLM-L6-v2`, `paraphrase-MiniLM-L12-v2`, or `tfidf`
- `data.path`: CSV path; default `data/sample.csv`

## Notes
- RDKit is optional for core UI but required for molecular tools
- Default port is 8502; change with `--server.port`

## License
MIT
