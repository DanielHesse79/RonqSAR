# NarrowSearchAI

NarrowSearchAI is a demo application for combining deterministic filters and optional semantic search over structured datasets. It features a Streamlit frontend and pluggable embedding backends.

## Features
- Filter by category, tags, date range, score and meta fields
- Optional semantic search using sentence-transformers or TF-IDF
- Interactive analytics dashboard via Plotly
- Export current results to CSV

## Usage
```bash
pip install -r requirements.txt
make dev
```

## Dataset
A small sample dataset is provided in `data/sample.csv` for quick experimentation.

## License
MIT
