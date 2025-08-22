# export_rep_docs.py
import pandas as pd
from pathlib import Path

ASSIGN = Path("../data/processed/bertopic_assignments.parquet")  # or .csv
FULL   = Path("../data/processed/physics_clean.parquet")         # or .csv
OUT    = Path("../outputs/topic_rep_docs.csv")
K = 5  # top-k reps/topic

ID_CAND_ASSIGN   = ["doc_id", "id", "paper_id", "arxiv_id", "uid"]
ID_CAND_FULL     = ["id", "paper_id", "arxiv_id", "uid"]
TOPIC_CAND       = ["topic", "Topic"]
PROB_CAND        = ["prob", "probability", "score"]
TITLE_CAND       = ["title", "paper_title", "name"]
ABSTRACT_CAND    = ["abstract", "summary", "abstract_text", "doc_full", "full_abstract"]

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path.with_suffix(".csv"))

def pick(df: pd.DataFrame, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of {candidates} found. Available: {list(df.columns)}")
    return None

def main():
    assign = read_any(ASSIGN)
    full   = read_any(FULL)

    # --- detect & normalize column names ---
    a_id    = pick(assign, ID_CAND_ASSIGN)
    a_topic = pick(assign, TOPIC_CAND)
    a_prob  = pick(assign, PROB_CAND, required=False)

    f_id     = pick(full, ID_CAND_FULL)
    f_title  = pick(full, TITLE_CAND, required=False)
    f_abs    = pick(full, ABSTRACT_CAND)

    # standardize schemas
    assign = assign.rename(columns={a_id: "id", a_topic: "topic"})
    full   = full.rename(columns={f_id: "id", f_abs: "abstract"})
    if f_title:
        full = full.rename(columns={f_title: "title"})

    # ensure types
    assign["id"] = assign["id"].astype(str)
    full["id"]   = full["id"].astype(str)

    # sort to get “representative” docs first
    if a_prob and a_prob in assign.columns:
        assign = assign.sort_values(["topic", a_prob], ascending=[True, False])
    else:
        assign = assign.sort_values(["topic"])

    # join to get FULL abstracts (and titles)
    keep_cols = ["id", "abstract"] + (["title"] if "title" in full.columns else [])
    rep = assign.merge(full[keep_cols], on="id", how="left")

    # top-K per topic
    rep["rank"] = rep.groupby("topic").cumcount() + 1
    rep_topk = rep[rep["rank"] <= K].copy()

    # final output (full abstracts go into 'doc_full')
    cols = ["topic", "rank", "id", "abstract"] + (["title"] if "title" in rep_topk.columns else [])
    rep_out = rep_topk[cols].rename(columns={"id": "doc_id", "abstract": "doc_full"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rep_out.to_csv(OUT, index=False)
    print(f"[ok] wrote {OUT} with {len(rep_out)} rows")
    print("column map:",
          {"assign_id": a_id, "assign_topic": a_topic, "assign_prob": a_prob,
           "full_id": f_id, "full_title": f_title, "full_abstract": f_abs})

if __name__ == "__main__":
    main()
