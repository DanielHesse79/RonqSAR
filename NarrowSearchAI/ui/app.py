from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from loguru import logger

from src.analytics import by_category, by_time, score_vs_date, tag_frequency
from src.config import Config, load_config, save_config
from src.filters import apply_filters
from src.load_data import load_dataframe
from src.preprocess import preprocess
from src.search import search
from src.vector_store import build_index

st.set_page_config(page_title="NarrowSearchAI", layout="wide")

cfg = load_config()

# header
col1, col2 = st.columns([1, 10])
with col1:
    st.image(str(Path(__file__).resolve().parents[1] / "assets/logo.svg"), width=80)
with col2:
    st.title(cfg.app.title)

# sidebar filters
with st.sidebar:
    st.header("Filters")
    df = preprocess(load_dataframe())
    categories = st.multiselect("Category", sorted(df["category"].unique()))
    all_tags = sorted({t for tags in df["tags"] for t in tags})
    tags = st.multiselect("Tags", all_tags)
    date_min, date_max = df["date"].min(), df["date"].max()
    dr = st.slider("Date range", min_value=date_min, max_value=date_max, value=(date_min, date_max))
    score_min, score_max = float(df["score"].min()), float(df["score"].max())
    sr = st.slider("Score range", min_value=score_min, max_value=score_max, value=(score_min, score_max))
    meta_keys = list(df["meta"].iloc[0].keys()) if not df.empty else []
    meta_key = st.selectbox("Meta key", [""] + meta_keys)
    meta_val = st.text_input("Meta value") if meta_key else None
    clear = st.button("Clear filters")
    if clear:
        st.experimental_rerun()

    st.header("Semantic search")
    use_sem = st.checkbox("Use semantic search", value=cfg.search.use_semantic)
    query = st.text_input("Query") if use_sem else ""
    topk = st.slider("Top K", 5, 100, cfg.search.top_k)
    model = st.selectbox("Model", ["none", "all-MiniLM-L6-v2", "paraphrase-MiniLM-L12-v2", "tfidf"], index=1)

# main tabs
results_tab, analytics_tab, details_tab, settings_tab = st.tabs(["Results", "Analytics", "Details", "Settings"])

filtered = apply_filters(df, categories, tags, dr, sr, meta_key or None, meta_val or None)

with results_tab:
    if use_sem and query:
        res = search(query, categories, tags, dr, sr, meta_key or None, meta_val or None, topk)
    else:
        res = filtered.head(topk)
    if res.empty:
        st.info("No results")
    else:
        for idx, row in res.iterrows():
            st.write(f"### {row['title']}")
            st.caption(", ".join(row["tags"]))
            st.write(row["text"][:200] + "...")
            if st.button("View", key=f"view_{row['id']}"):
                st.session_state["detail"] = row.to_dict()
                st.experimental_rerun()

with analytics_tab:
    if filtered.empty:
        st.info("No data")
    else:
        st.plotly_chart(px.bar(by_category(filtered), x="category", y="count"), use_container_width=True)
        st.plotly_chart(px.bar(tag_frequency(filtered), x="tags", y="count"), use_container_width=True)
        st.plotly_chart(px.line(by_time(filtered), x="date", y="count"), use_container_width=True)
        st.plotly_chart(px.scatter(score_vs_date(filtered), x="date", y="score", color="category"), use_container_width=True)

with details_tab:
    detail = st.session_state.get("detail")
    if not detail:
        st.info("Select a record from Results")
    else:
        st.header(detail["title"])
        st.write(detail["text"])
        st.write("Tags:", ", ".join(detail["tags"]))
        st.json(detail["meta"])

with settings_tab:
    st.write("Current model:", cfg.search.embedder)
    if st.button("Rebuild index"):
        build_index(model)
        st.success("Index built")
    if st.button("Export filtered CSV"):
        filtered.to_csv("export.csv", index=False)
        st.success("export.csv saved")
