#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Tuple


import numpy as np
import pandas as pd, re

# ---- Config ----
DATA_DIR = Path("../data/processed")
RAW_DIR = Path("../data/raw")
MODELS_DIR = Path("../models")
FIGS_DIR = Path("../figs")
OUT_DIR = Path("../outputs")

## exchange with you own field
INPUT_PARQUET = DATA_DIR / "physics_clean.parquet" # expected columns: id (optional), title, abstract, abstract_clean, categories, year
INPUT_CSV = DATA_DIR / "physics_clean.csv"

PRIMARY_PARQUET = DATA_DIR / "physics_clean_primary.parquet"
ASSIGN_PARQUET = DATA_DIR / "bertopic_assignments.parquet"
MODEL_PATH = MODELS_DIR / "bertopic_physics"
ASSIGN_CSV = DATA_DIR / "bertopic_assignments.csv"
TOPIC_INFO_CSV = OUT_DIR / "topic_info.csv"
TRENDS_CSV = OUT_DIR / "topic_trends_by_year.csv"
GLASS_TREND_CSV = OUT_DIR / "glass_trend.csv"


MAX_DOCS = 500000 
RANDOM_STATE = 42

CONCEPT_SEED = "glass"

# keep in your script near the top
WHITELIST_PREFIXES = ("physics.", "cond-mat.")
EXCLUDE_SUBFIELDS = {
    "physics.soc-ph", "physics.pop-ph", "physics.hist-ph", "physics.ed-ph", "physics.med-ph",
    # optional: exclude noisy umbrella buckets
    # "cond-mat.other",
}

def safe_read(path_parquet: Path, path_csv: Path) -> pd.DataFrame:
    if path_parquet.exists():
        try:
            return pd.read_parquet(path_parquet, engine="pyarrow")
        except Exception as e:
            print(f"[warn] read_parquet failed ({e}); falling back to CSV if available…")
    if path_csv.exists():
        print("[info] reading CSV fallback")
        return pd.read_csv(path_csv)
    raise FileNotFoundError(f"Neither {path_parquet} nor {path_csv} exists.")


def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, engine="pyarrow", index=False)
        print(f"[ok] wrote Parquet: {path}")
    except Exception as e:
        print(f"[warn] to_parquet failed ({e}); writing CSV fallback…")
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"[ok] wrote CSV fallback: {csv_path}")
        
# ---- 1) Load cleaned dataset ----

def load_cleaned() -> pd.DataFrame:
    df = safe_read(INPUT_PARQUET, INPUT_CSV)
    # If abstract_clean missing, create a quick one
    if "abstract_clean" not in df.columns and "abstract" in df.columns:
        import re
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        def clean(t: str) -> str:
            t = str(t).lower()
            t = re.sub(r"http\S+|www\.\S+", " ", t)
            toks = re.findall(r"[a-z]+", t)
            toks = [w for w in toks if w not in ENGLISH_STOP_WORDS and len(w) > 2]
            return " ".join(toks)
        df["abstract_clean"] = df["abstract"].map(clean)
    # Ensure year column is numeric
    if "year" not in df.columns and "published" in df.columns:
        df["year"] = pd.to_datetime(df["published"], errors="coerce").dt.year
    df = df.dropna(subset=["abstract", "abstract_clean", "year"]).copy()
    df["year"] = df["year"].astype(int)
    return df


# ---- 2) Filter to primary physics ----




def keep_row(cats: str) -> bool:
    toks = str(cats).split()
    if not toks:
        return False
    # include if ANY category matches the whitelist prefixes
    ok = any(any(t.startswith(pref) for pref in WHITELIST_PREFIXES) for t in toks)
    # exclude if ANY excluded subfield shows up
    bad = any(t in EXCLUDE_SUBFIELDS for t in toks)
    return ok and not bad

def filter_domain(df: pd.DataFrame) -> pd.DataFrame:
    if "categories" not in df.columns:
        print("[warn] 'categories' missing; skipping domain filter.")
        return df
    df2 = df[df["categories"].apply(keep_row)].copy()
    safe_write_parquet(df2, PRIMARY_PARQUET)  # reuse your existing save helper
    return df2

# ---- 3) Build stopword-aware vectorizer ----

def build_vectorizer():
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    domain_fillers = {
        "result", "results", "show", "shows", "paper", "study", "work",
        "approach", "method", "methods", "model", "models", "based", "using",
        "novel", "experimental", "theoretical", "numerical", "provide", "propose",
        "investigate", "analysis", "properties", "system", "systems"
    }
    stopwords_list = sorted(set(ENGLISH_STOP_WORDS).union(domain_fillers))
    vectorizer_model = CountVectorizer(
        stop_words=stopwords_list,
        ngram_range=(2, 3),
        min_df=1,
        max_df=0.4,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",  # keep hyphenated scientific terms
    )
    return vectorizer_model


# ---- 4) Initialize models (embedding, UMAP, HDBSCAN, BERTopic) ----

def init_models():
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic

    # Embedding model (science-focused is better for abstracts)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # alt: "sentence-transformers/all-mpnet-base-v2"

    # Reducer + clustering (seeded for stability)
    umap_model = UMAP(n_neighbors=30, n_components=5, min_dist=0.0, metric="cosine", random_state=RANDOM_STATE)
    hdbscan_model = HDBSCAN(min_cluster_size=60, min_samples=5, metric="euclidean",
                            cluster_selection_method="eom", prediction_data=True)

    # Vectorizer for topic representations
    vectorizer_model = build_vectorizer()

    # BERTopic tying it together
    topic_model = BERTopic(
        embedding_model=encoder,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=None,
        verbose=True,
    )
    return topic_model, encoder

# ---- 5) Fit, reduce outliers, update labels ----

def fit_bertopic(df: pd.DataFrame, topic_model, encoder) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # Text for clustering vs representation
    docs_raw = df["abstract"].astype(str).tolist()
    docs_clean = df["abstract_clean"].astype(str).tolist()

    # Precompute embeddings from raw text (better semantics & reproducibility)
    print("[info] encoding documents …")
    embs = encoder.encode(docs_raw, batch_size=256, show_progress_bar=True, convert_to_numpy=True)

    # Fit
    print("[info] fitting BERTopic …")
    topics, probs = topic_model.fit_transform(docs_clean, embeddings=embs)

    # Shrink outliers by reassigning with c-TF-IDF similarities
    print("[info] reducing outliers …")
    topics = topic_model.reduce_outliers(docs_clean, topics)

    # Improve labels with KeyBERTInspired (uses the same encoder)
    from bertopic.representation import KeyBERTInspired
    rep = KeyBERTInspired()
    vec = topic_model.vectorizer_model
    print("[info] updating topics/labels …")
    topic_model.update_topics(docs_clean, topics=topics,vectorizer_model=vec, representation_model=rep)

    return topics, probs


# ---- 6) Save artifacts and tables ----

def save_artifacts(df: pd.DataFrame, topics: np.ndarray, probs: Optional[np.ndarray], topic_model) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    print("[info] saving BERTopic model …")
    topic_model.save(str(MODEL_PATH), save_embedding_model=True)

    # Assignments (keep id if present)
    cols = [c for c in ["arxiv_id", "id", "published", "year"] if c in df.columns]
    assign = df[cols].copy()
    assign["topic"] = topics
    if probs is not None:
        assign["prob"] = probs
    safe_write_parquet(assign, ASSIGN_PARQUET)
    assign.to_csv(ASSIGN_CSV, index=False)

    # Topic info table
    info = topic_model.get_topic_info()
    info.to_csv(TOPIC_INFO_CSV, index=False)
    print(f"[ok] wrote topic info: {TOPIC_INFO_CSV}")
    
# ---- 7) Trends by year (counts & share) ----

def compute_trends(assign_path: Path) -> pd.DataFrame:
    df = safe_read(assign_path, assign_path.with_suffix(".csv"))
    if "year" not in df.columns:
        raise ValueError("Assignments missing 'year' column.")
    trend = df.groupby(["year", "topic"]).size().reset_index(name="count")
    total = df.groupby("year").size().rename("n").reset_index()
    trend = trend.merge(total, on="year")
    trend["share"] = trend["count"] / trend["n"]
    trend = trend.sort_values(["year", "topic"]) # tidy
    trend.to_csv(TRENDS_CSV, index=False)
    print(f"[ok] wrote trends: {TRENDS_CSV}")
    return trend



# ---- 8) Concept trend (e.g., glass) ----

def concept_trend(topic_model, trend: pd.DataFrame, concept: str = CONCEPT_SEED) -> pd.DataFrame:
    """
    Robust concept trend extractor that avoids BERTopic.find_topics index errors
    after topic reduction. It tries search/find with bounds checks and falls
    back to scanning topic representations for matching n-grams.
    """
    # Valid, non-outlier topics
    info = topic_model.get_topic_info()
    valid_topics = {int(t) for t in info["Topic"].tolist() if int(t) != -1}
    ordered_topics = [int(t) for t in info["Topic"].tolist() if int(t) != -1]

    # Try BERTopic's search APIs first, but guard against bad indices
    cand_ids = []
    try:
        # Newer BERTopic
        res = topic_model.search_topics(concept)
        cand_ids = [int(r[0]) for r in res if int(r[0]) in valid_topics]
    except Exception:
        # Older BERTopic
        try:
            topn = max(1, min(10, len(ordered_topics)))
            ids, _ = topic_model.find_topics(concept, top_n=topn)
            cand_ids = [int(i) for i in list(ids) if int(i) in valid_topics]
        except Exception:
            cand_ids = []

    # Fallback: match by topic representation keywords
    t_id = None
    if not cand_ids:
        import re
        pat = re.compile(r"(glass|glassy|glass[- ]transition|spin[- ]glass|amorphous|vitreous)", re.I)
        best, best_score = None, -1.0
        for t in ordered_topics:
            words = topic_model.get_topic(int(t)) or []   # list of (term, weight)
            score = sum(float(w) for term, w in words if pat.search(term))
            if score > best_score:
                best, best_score = int(t), score
        t_id = best
    else:
        t_id = cand_ids[0]

    if t_id is None or t_id not in valid_topics:
        print(f"[warn] no suitable topic matched concept '{concept}' — returning empty result")
        return pd.DataFrame()

    glass = trend[trend["topic"] == int(t_id)].sort_values("year").copy()
    if glass.empty:
        print(f"[warn] no trend rows for topic {t_id}")
        return glass

    glass.to_csv(GLASS_TREND_CSV, index=False)

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(glass["year"], glass["share"])
        plt.title(f"Topic {t_id} (‘{concept}’) share over time")
        plt.xlabel("Year"); plt.ylabel("Share of papers")
        plt.tight_layout()
        fig_path = FIGS_DIR / f"trend_{concept.replace(' ', '_')}.png"
        plt.savefig(fig_path, dpi=160)
        print(f"[ok] wrote figure: {fig_path}")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

    return glass


def pretty_label(topic_model, topic_id, topk=6):
    reps = topic_model.get_topic(topic_id) or []  # list[(term, weight)]
    # prefer multi-word phrases
    phrases = [t for t, w in reps if " " in t]
    chosen = phrases[:3] if phrases else [t for t, w in reps[:3]]
    return " / ".join(chosen)
# ---- Main ----


def main():
    np.random.seed(RANDOM_STATE)

    df_all = load_cleaned()
    df = filter_domain(df_all) 
    # df = pd.read_parquet("../data/processed/physics_clean_primary.parquet")

    # Sample for memory/run-time (optional)
    if len(df) > MAX_DOCS:
        df = df.sample(n=MAX_DOCS, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[info] sampled to {len(df)} docs for training")
    print("share with cond-mat anywhere:", df_all["categories"].str.contains("cond-mat", na=False).mean())
    print("after filter, rows:", len(df))
    print(df["categories"].head(3).tolist())
    print("[step] init models …")
    topic_model, encoder = init_models()

    print("[step] fit BERTopic …")
    topics, probs = fit_bertopic(df, topic_model, encoder)

    print("[step] save artifacts …")
    save_artifacts(df, topics, probs, topic_model)

    print("[step] compute trends …")
    trend = compute_trends(ASSIGN_PARQUET)

    print(f"[step] concept trend: {CONCEPT_SEED} …")
    _ = concept_trend(topic_model, trend, concept=CONCEPT_SEED)
    
    info = topic_model.get_topic_info()
    if "Name" in info.columns:
        pretty = []
        for t in info["Topic"]:
            if int(t) == -1:
                pretty.append("outliers")
            else:
                pretty.append(pretty_label(topic_model, int(t)))
        info["PrettyName"] = pretty
        info.to_csv(TOPIC_INFO_CSV, index=False)

    print("[done] Training + exports complete.")
    
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[exit] interrupted by user")