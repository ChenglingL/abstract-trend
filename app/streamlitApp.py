#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Explorer — BERTopic Physics/Cond‑Mat Trends (Lite Mode)
----------------------------------------------------------------
Runs **without** a BERTopic model. Uses saved CSV artifacts only:
- outputs/topic_info.csv               (Topic, Count, PrettyName/Name optional)
- outputs/topic_terms.csv              (topic, term, weight)
- outputs/topic_trends_by_year.csv     (year, topic, count, n, share)
- outputs/topic_rep_docs.csv           (topic, rank, doc_full|abstract|doc|doc_id, title?) [optional]

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------- Paths ---------------------
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data" / "processed"

TOPIC_INFO_CSV   = OUT_DIR / "topic_info.csv"
TOPIC_TERMS_CSV  = OUT_DIR / "topic_terms.csv"
TOPIC_TRENDS_CSV = OUT_DIR / "topic_trends_by_year.csv"
TOPIC_REPDOCS_CSV= OUT_DIR / "topic_rep_docs.csv"   # optional
ASSIGN_CSV       = DATA_DIR / "bertopic_assignments.csv"  # optional

st.set_page_config(page_title="Physics Topics Explorer (Lite)", layout="wide")

# --------------------- IO helpers ---------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
        # allow parquet fallbacks for known tables
        pqp = path.with_suffix(".parquet")
        if pqp.exists():
            return pd.read_parquet(pqp)
    except Exception as e:
        st.warning(f"Failed to load {path.name}: {e}")
    return None


# --------------------- Load artifacts & Health Check ---------------------
SHOW_HEALTHCHECK = bool(int(os.getenv("SHOW_HEALTHCHECK", "0"))) ## debugging mode
if SHOW_HEALTHCHECK:
    st.subheader("Health Check")
    st.code("\n".join([
        str(TOPIC_INFO_CSV),
        str(TOPIC_TERMS_CSV),
        str(TOPIC_TRENDS_CSV),
        f"{str(TOPIC_REPDOCS_CSV)} (optional)",
        f"{str(ASSIGN_CSV)} (optional)",
    ]))
info   = load_csv(TOPIC_INFO_CSV)
terms  = load_csv(TOPIC_TERMS_CSV)
trends = load_csv(TOPIC_TRENDS_CSV)
repdocs= load_csv(TOPIC_REPDOCS_CSV)
assign = load_csv(ASSIGN_CSV)

missing = []
if not isinstance(info, pd.DataFrame):
    missing.append("outputs/topic_info.csv")
if not isinstance(terms, pd.DataFrame):
    missing.append("outputs/topic_terms.csv")
if not isinstance(trends, pd.DataFrame):
    missing.append("outputs/topic_trends_by_year.csv")
if missing:
    st.error("Missing/failed artifacts: " + ", ".join(missing))
    st.stop()

# Normalize dtypes
with np.errstate(all='ignore'):
    if "Topic" in info.columns:
        info["Topic"] = pd.to_numeric(info["Topic"], errors="coerce").astype("Int64")
    if "Count" in info.columns:
        info["Count"] = pd.to_numeric(info["Count"], errors="coerce")
    terms["topic"] = pd.to_numeric(terms["topic"], errors="coerce").astype("Int64")
    if "weight" in terms.columns:
        terms["weight"] = pd.to_numeric(terms["weight"], errors="coerce")
    trends["topic"] = pd.to_numeric(trends["topic"], errors="coerce").astype("Int64")
    trends["year"] = pd.to_numeric(trends["year"], errors="coerce").astype("Int64")
    for c in ("share","count","n"):
        if c in trends.columns:
            trends[c] = pd.to_numeric(trends[c], errors="coerce")
    if isinstance(repdocs, pd.DataFrame):
        if "topic" in repdocs.columns:
            repdocs["topic"] = pd.to_numeric(repdocs["topic"], errors="coerce").astype("Int64")
        if "rank" in repdocs.columns:
            repdocs["rank"] = pd.to_numeric(repdocs["rank"], errors="coerce").astype("Int64")
        for idcol in ("doc_id","id","paper_id"):
            if idcol in repdocs.columns:
                repdocs[idcol] = repdocs[idcol].astype(str)

# --------------------- UI ---------------------
st.title("Physics/Cond‑Mat Topics Explorer (2005–2025)")
col1, col2, col3 = st.columns([1.2, 1, 1])
col1.caption("Browse topics learned offline and visualize 20‑year trends.")

with st.sidebar:
    st.header("Controls")
    include_outliers = st.checkbox("Include outliers (Topic = -1)", value=False)
    yr_min, yr_max = st.slider("Year range", 2005, 2025, (2005, 2025), step=1)
    st.divider()
    st.caption("Data locations (read‑only)")
    st.code("".join([str(TOPIC_INFO_CSV), str(TOPIC_TERMS_CSV), str(TOPIC_TRENDS_CSV)]))

# KPIs
n_docs = int(assign.shape[0]) if isinstance(assign, pd.DataFrame) and not assign.empty else int(info["Count"].sum())
outliers = int(info.loc[info["Topic"] == -1, "Count"].iloc[0]) if (-1 in info["Topic"].values) else 0
n_topics = int((info["Topic"] != -1).sum())
with col1:
    st.metric("Documents", f"{n_docs:,}")
with col2:
    st.metric("Topics (excl. -1)", f"{n_topics:,}")
with col3:
    share = (outliers / max(n_docs, 1)) if n_docs else 0.0
    st.metric("Outlier share", f"{share:.1%}")

# Prepare table
name_col = "PrettyName" if "PrettyName" in info.columns else ("Name" if "Name" in info.columns else None)
show_df = info.copy()
if not include_outliers:
    show_df = show_df[show_df["Topic"] != -1]
if name_col:
    show_df = show_df[["Topic", "Count", name_col]].rename(columns={name_col: "Label"})
else:
    show_df = show_df[["Topic", "Count"]].assign(Label=lambda d: d["Topic"].astype(str))
show_df = show_df.sort_values("Count", ascending=False)

st.subheader("Topics")
st.dataframe(show_df.reset_index(drop=True), use_container_width=True, hide_index=True)

# Helper: pretty label from terms
def pretty_label_topic(topic_id: int) -> str:
    tid = int(topic_id)
    try:
        terms_list = (terms[terms["topic"] == tid]
                        .sort_values("weight", ascending=False)["term"].tolist())
        phrases = [t for t in terms_list if (" " in t) or ("-" in t)]
        chosen = (phrases[:3] or terms_list[:3])
        if chosen:
            return " / ".join(chosen)
    except Exception:
        pass
    try:
        return str(show_df.set_index("Topic").loc[tid, "Label"])  # fallback
    except Exception:
        return f"Topic {tid}"

# Topic selector & details
topic_choices = show_df["Topic"].tolist()
def_label_map = {int(t): pretty_label_topic(int(t)) for t in topic_choices}


trend_filt = trends[(trends["year"] >= yr_min) & (trends["year"] <= yr_max)].copy()

st.markdown("---")
sel_topic = st.selectbox("Select a topic", options=topic_choices,
                         format_func=lambda x: def_label_map.get(int(x), f"Topic {x}"))
left, right = st.columns([1.2, 1])
with left:
    st.markdown(f"### Topic {sel_topic} — details")
    # Top terms
    t_terms = terms[terms["topic"] == int(sel_topic)].sort_values("weight", ascending=False).head(20)
    if not t_terms.empty:
        st.write("**Top terms**")
        st.table(t_terms)
    # Representative docs (prefer full abstract)
    st.write("**Representative documents**")
    display_docs: List[str] = []
    if isinstance(repdocs, pd.DataFrame):
        subrep = repdocs[repdocs["topic"] == int(sel_topic)].sort_values("rank" if "rank" in repdocs.columns else repdocs.columns[0]).copy()
        if "doc_full" in subrep.columns:
            display_docs = subrep["doc_full"].dropna().astype(str).tolist()
        elif "abstract" in subrep.columns:
            display_docs = subrep["abstract"].dropna().astype(str).tolist()
        elif "doc" in subrep.columns:
            display_docs = subrep["doc"].dropna().astype(str).tolist()
        elif "doc_id" in subrep.columns and isinstance(FULL_CORPUS, pd.DataFrame) and {"id","abstract"}.issubset(FULL_CORPUS.columns):
            merged = subrep.merge(FULL_CORPUS.rename(columns={"id":"doc_id"}), on="doc_id", how="left")
            display_docs = merged["abstract"].dropna().astype(str).tolist()
    if not display_docs:
        st.info("No representative documents available for this topic.")
    else:
        for i, doc in enumerate(display_docs[:3], 1):
            with st.expander(f"Doc {i}"):
                st.write(doc)

with right:
    st.write("**Yearly prevalence (share of papers)**")
    sub = trend_filt[trend_filt["topic"] == int(sel_topic)].sort_values("year")
    if sub.empty:
        st.info("No trend data for this topic in the selected year range.")
    else:
        sub = sub.copy(); sub["share_smooth"] = sub["share"].rolling(3, center=True).mean()
        st.line_chart(sub.set_index("year")[ ["share", "share_smooth"] ])

st.markdown("---")

# Concept search (unified: search box + docs + trend)
st.subheader("Concept search")
concept_query = st.text_input("Search topic", value="glass", help="Try phrases like 'glass transition', 'spin glass', 'amorphous solid', 'superconducting qubits'.")
if concept_query.strip():
    import re
    pat = re.compile(concept_query.strip(), re.I)
    hits = terms.assign(hit=terms["term"].str.contains(pat))
    scores = (hits.groupby("topic")["hit"].sum()
                   .reset_index()
                   .sort_values(["hit", "topic"], ascending=[False, True]))
    if scores.empty or scores.iloc[0]["hit"] <= 0:
        st.warning(f"No matching topic found for '{concept_query}'.")
    else:
        t_id = int(scores.iloc[0]["topic"])
        st.success(f"Best topic for '{concept_query}': {t_id} — {pretty_label_topic(t_id)}")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.write("**Representative documents**")
            display_docs: List[str] = []
            if isinstance(repdocs, pd.DataFrame):
                subrep = repdocs[repdocs["topic"] == int(t_id)].sort_values("rank" if "rank" in repdocs.columns else repdocs.columns[0]).copy()
                if "doc_full" in subrep.columns:
                    display_docs = subrep["doc_full"].dropna().astype(str).tolist()
                elif "abstract" in subrep.columns:
                    display_docs = subrep["abstract"].dropna().astype(str).tolist()
                elif "doc" in subrep.columns:
                    display_docs = subrep["doc"].dropna().astype(str).tolist()
                elif "doc_id" in subrep.columns and isinstance(FULL_CORPUS, pd.DataFrame) and {"id","abstract"}.issubset(FULL_CORPUS.columns):
                    merged = subrep.merge(FULL_CORPUS.rename(columns={"id":"doc_id"}), on="doc_id", how="left")
                    display_docs = merged["abstract"].dropna().astype(str).tolist()
            if not display_docs:
                st.info("No representative documents available.")
            else:
                for i, doc in enumerate(display_docs[:3], 1):
                    with st.expander(f"Doc {i}"):
                        st.write(doc)
        with c2:
            st.write("**Yearly prevalence (share of papers)**")
            sub = trend_filt[trend_filt["topic"] == int(t_id)].sort_values("year")
            if sub.empty:
                st.info("No trend data for this topic in the selected year range.")
            else:
                sub = sub.copy(); sub["share_smooth"] = sub["share"].rolling(3, center=True).mean()
                st.line_chart(sub.set_index("year")[ ["share", "share_smooth"] ])

st.caption("Lite mode uses precomputed topic terms and trends — no model required. Representative docs prefer full abstracts when available.")
