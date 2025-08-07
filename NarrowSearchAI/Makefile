.PHONY: dev build_index export

dev:
STREAMLIT_BROWSER.gatherUsageStats=false streamlit run ui/app.py

build_index:
python -m src.vector_store --build

export:
python -m src.search --export
