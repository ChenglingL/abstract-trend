#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Explorer — BERTopic Physics/Cond‑Mat Trends (2005–2025)
----------------------------------------------------------------
Features
- Loads the trained BERTopic model + artifacts (assignments, topic info, yearly trends)
- Browsable topic table with human‑readable labels (PrettyName if present)
- Per‑topic details: top terms, representative docs, yearly prevalence plot
- Concept search (e.g., "glass" → find best topic id) with robust fallback
- Optional new‑text classification (if the model includes an embedding model)

Run from repo root:
    streamlit run app/streamlit_app.py

Expected files (created by your training script):
- models/bertopic_physics          (BERTopic.save(..., save_embedding_model=True) if possible)
- outputs/topic_info.csv          (includes Name/Representation; PrettyName if added)
- outputs/topic_trends_by_year.csv (columns: year, topic, count, n, share)
- data/processed/bertopic_assignments.parquet (or .csv fallback)

If Parquet reading fails, this app will fall back to CSV when possible.
"""

from __future__ import annotations
import os
import ast
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------- Paths & helpers ---------------------
ROOT = Path(__file__).resolve().parents[1]  # repo root
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = MODELS_DIR / "bertopic_physics"
TOPIC_INFO_CSV = OUT_DIR / "topic_info.csv"
TRENDS_CSV = OUT_DIR / "topic_trends_by_year.csv"
ASSIGN_PARQUET = DATA_DIR / "bertopic_assignments.parquet"
ASSIGN_CSV = DATA_DIR / "bertopic_assignments.csv"

st.set_page_config(page_title="Physics Topics Explorer", layout="wide")

@st.cache_data(show_spinner=False)
def safe_read(path_parquet: Path, path_csv: Path) -> pd.DataFrame:
    if path_parquet.exists():
        try:
            return pd.read_parquet(path_parquet, engine="pyarrow")
        except Exception as e:
            st.warning(f"read_parquet failed on {path_parquet.name}: {e}\nFalling back to CSV…")
    if path_csv.exists():
        return pd.read_csv(path_csv)
    raise FileNotFoundError(f"Neither {path_parquet} nor {path_csv} found.")

@st.cache_resource(show_spinner=True)
def load_model(model_path: Path):
    from bertopic import BERTopic
    try:
        model = BERTopic.load(str(model_path))
        return model
    except Exception as e:
        st.error(f"Failed to load BERTopic model at {model_path}: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_tables():
    try:
        info = pd.read_csv(TOPIC_INFO_CSV)
    except Exception as e:
        st.error(f"Failed to read topic info CSV at {TOPIC_INFO_CSV}: {e}")
        info = None
    try:
        trend = pd.read_csv(TRENDS_CSV)
    except Exception as e:
        st.error(f"Failed to read trends CSV at {TRENDS_CSV}: {e}")
        trend = None
    try:
        assign = safe_read(ASSIGN_PARQUET, ASSIGN_CSV)
    except Exception as e:
        st.warning(f"Assignments table missing/invalid: {e}")
        assign = pd.DataFrame()

    # Normalize dtypes if loaded
    if isinstance(info, pd.DataFrame) and "Topic" in info.columns:
        try:
            info["Topic"] = info["Topic"].astype(int)
        except Exception:
            pass
    if isinstance(trend, pd.DataFrame):
        for c in ("year", "topic"):
            if c in trend.columns:
                try:
                    trend[c] = trend[c].astype(int)
                except Exception:
                    pass
    return info, trend, assign

# Robust concept → topic id utility
def find_concept_topic(model, concept: str, info_df: pd.DataFrame) -> Optional[int]:
    # Valid topics (exclude -1)
    valid = {int(t) for t in info_df.get("Topic", pd.Series(dtype=int)).tolist() if int(t) != -1}
    # Try search APIs
    try:
        res = model.search_topics(concept)
        cand = [int(r[0]) for r in res if int(r[0]) in valid]
        if cand:
            return cand[0]
    except Exception:
        try:
            ids, _ = model.find_topics(concept, top_n=min(10, max(1, len(valid))))
            cand = [int(i) for i in ids if int(i) in valid]
            if cand:
                return cand[0]
        except Exception:
            pass
    # Fallback: scan topic words for keyword hits
    import re
    pat = re.compile(r"(glass|glassy|glass[- ]transition|spin[- ]glass|amorphous|vitreous)", re.I)
    best, score = None, -1.0
    for t in sorted(valid):
        reps = model.get_topic(int(t)) or []
        s = sum(float(w) for term, w in reps if pat.search(term))
        if s > score:
            best, score = int(t), s
    return best

# Pretty label utility
def pretty_label(model, topic_id: int) -> str:
    reps = model.get_topic(int(topic_id)) or []
    phrases = [t for t, w in reps if " " in t]
    chosen = phrases[:3] if phrases else [t for t, w in reps[:3]]
    return " / ".join(chosen) if chosen else str(topic_id)

# --------------------- UI ---------------------
st.title("Physics/Cond‑Mat Topics Explorer (2005–2025)")
col1, col2, col3 = st.columns([1.2, 1, 1])
col1.caption("Browse topics learned with BERTopic and visualize 20‑year trends. Use the sidebar to filter.")

with st.sidebar:
    st.header("Controls")
    include_outliers = st.checkbox("Include outliers (Topic = -1)", value=False)
    concept = st.text_input("Concept search", value="glass")
    yr_min, yr_max = st.slider("Year range", 2005, 2025, (2005, 2025), step=1)
    st.divider()
    st.caption("Data locations (read‑only)")
    st.code(f"{MODEL_PATH}\n{TOPIC_INFO_CSV}\n{TRENDS_CSV}")

# Load artifacts
st.subheader("Health Check")
st.code(f"""ROOT={ROOT}
MODEL_PATH={MODEL_PATH}
TOPIC_INFO_CSV={TOPIC_INFO_CSV}
TRENDS_CSV={TRENDS_CSV}
ASSIGN_PARQUET={ASSIGN_PARQUET}""")

model = load_model(MODEL_PATH)
info, trend, assign = load_tables()

missing = []
if model is None:
    missing.append("model")
if not isinstance(info, pd.DataFrame):
    missing.append("topic_info.csv")
if not isinstance(trend, pd.DataFrame):
    missing.append("topic_trends_by_year.csv")

if missing:
    st.error("Missing/failed artifacts: " + ", ".join(missing))
    st.info("Ensure your training script wrote these files to the expected paths above. Adjust paths at the top of this app if your layout differs.")
    st.stop()

# KPIs
n_docs = int(assign.shape[0]) if not assign.empty else int(trend["n"].max())
outlier_row = info[info["Topic"] == -1]
outliers = int(outlier_row["Count"].iloc[0]) if not outlier_row.empty else 0
n_topics = int((info["Topic"] != -1).sum())

with col1:
    st.metric("Documents", f"{n_docs:,}")
with col2:
    st.metric("Topics (excl. -1)", f"{n_topics:,}")
with col3:
    share = (outliers / max(n_docs, 1)) if n_docs else 0.0
    st.metric("Outlier share", f"{share:.1%}")

# Prepare table
name_col = "PrettyName" if "PrettyName" in info.columns else "Name"
show_df = info.copy()
if not include_outliers:
    show_df = show_df[show_df["Topic"] != -1]
show_df = show_df[["Topic", "Count", name_col]].rename(columns={name_col: "Label"}).sort_values("Count", ascending=False)

st.subheader("Topics")
st.dataframe(show_df.reset_index(drop=True), use_container_width=True, hide_index=True)

# Topic selector
topic_choices = show_df["Topic"].tolist()
def_label_map = {int(t): pretty_label(model, int(t)) for t in topic_choices[:200]}  # cache first 200 lazily
sel_topic = st.selectbox("Select a topic", options=topic_choices, format_func=lambda x: def_label_map.get(int(x), f"Topic {x}"))

# Filter trend by year
trend_filt = trend[(trend["year"] >= yr_min) & (trend["year"] <= yr_max)].copy()

# Topic details block
st.markdown("---")
left, right = st.columns([1.2, 1])

with left:
    st.markdown(f"### Topic {sel_topic} — details")
    reps = model.get_topic(int(sel_topic)) or []
    if reps:
        terms = pd.DataFrame(reps, columns=["term", "weight"])[:20]
        st.write("**Top terms**")
        st.table(terms)
    # Representative docs
    st.write("**Representative documents**")
    rep_docs: List[str] = []
    # Try model API first
    try:
        rep_docs = model.get_representative_docs(int(sel_topic))  # available in newer versions
    except Exception:
        # Fallback: parse from CSV if present
        if "Representative_Docs" in info.columns:
            try:
                rep_docs = ast.literal_eval(info.loc[info["Topic"] == int(sel_topic), "Representative_Docs"].iloc[0])
            except Exception:
                rep_docs = []
    for i, doc in enumerate(rep_docs[:3], 1):
        st.markdown(f"**Doc {i}.** {doc}")

with right:
    st.write("**Yearly prevalence (share of papers)**")
    sub = trend_filt[trend_filt["topic"] == int(sel_topic)].sort_values("year")
    if sub.empty:
        st.info("No trend data for this topic in the selected year range.")
    else:
        # Smooth with 3-year rolling average (display both raw and smooth)
        sub = sub.copy()
        sub["share_smooth"] = sub["share"].rolling(3, center=True).mean()
        st.line_chart(sub.set_index("year")[ ["share", "share_smooth"] ])

st.markdown("---")

# Concept search & classification
st.subheader("Concept tools")
cc1, cc2 = st.columns([1,1])
with cc1:
    if concept.strip():
        t_id = find_concept_topic(model, concept.strip(), info)
        if t_id is None:
            st.warning(f"No matching topic found for '{concept}'.")
        else:
            st.success(f"Best topic for '{concept}': {t_id} — {pretty_label(model, t_id)}")
            sub = trend_filt[trend_filt["topic"] == int(t_id)].sort_values("year")
            if not sub.empty:
                sub = sub.copy(); sub["share_smooth"] = sub["share"].rolling(3, center=True).mean()
                st.line_chart(sub.set_index("year")[ ["share", "share_smooth"] ])

with cc2:
    st.write("**Classify a new abstract**")
    txt = st.text_area("Paste an abstract to predict its topic", height=180)
    if st.button("Predict"):
        if not txt.strip():
            st.warning("Please paste an abstract first.")
        else:
            # Only works if the model has an embedding model bundled
            try:
                topics, probs = model.transform([txt])
                t = int(topics[0])
                st.info(f"Predicted topic: {t} — {pretty_label(model, t)}")
            except Exception as e:
                st.error(f"Prediction unavailable (model may lack an embedding model): {e}")

st.caption("Tip: toggle outliers, narrow the year range, or change the concept term to explore different trends (e.g., 'spin glass', 'amorphous solid', 'superconducting qubits').")
