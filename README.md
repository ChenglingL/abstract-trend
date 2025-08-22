# abstract-trend — Topic Trends from Abstracts (BERTopic)

Train your own topic model on research **abstracts** and generate multi-year **trend plots** — starting with arXiv **physics + cond-mat**. You can swap in **any field** (bio, NLP, economics, etc.) by changing the data source and category filter. The pipeline also exports **Lite CSV artifacts** for a tiny Streamlit app (no large model required).

> **App demo :** [Physics](https://physicstrend-gyc5hefqjfhjtymkheyvbb.streamlit.app/)



## What it does

- **Preprocess** abstracts (clean text, normalize metadata, filter by categories/years)
- **Train BERTopic** (Sentence-Transformers → UMAP → HDBSCAN → c-TF-IDF)
- **Label topics** with (2–3)-gram phrases + domain stopwords
- **Assign** each abstract to a topic (with outlier handling)
- **Compute yearly trends** (counts + normalized share)
- **Export Lite CSVs** (topics/terms/trends/rep-docs) for a model-free Streamlit app


## Repository layout
```bash
abstract-trend/
├─ scripts/
│ ├─ preprocess.ipynb # clean + filter → physics_clean.parquet
│ ├─ train_bertopic.py # fit BERTopic, save assignments + topic_info
│ ├─ export_rep_docs.py # write topic_terms.csv, topic_trends_by_year.csv topic_rep_docs.csv
├─ app/
│ └─ streamlit_app.py # Lite Streamlit app (no model needed)
├─ data/(ignored by git)
│ └─ processed/ # local outputs
├─ models/ # saved BERTopic model (ignored by git)
├─ outputs/ # small CSV artifacts used by the app
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Quickstart

> Requires **Python 3.11** (recommended) and ~8–16 GB RAM for training on ~50–100k abstracts.

```bash
# 0) clone & create env
git clone https://github.com/<you>/abstract-trend.git
cd abstract-trend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) put your raw data somewhere (CSV or Parquet)
#    run fetchData.ipynb (switch to your own field)

# 2) preprocess → clean + filter (physics + cond-mat by default)
#    run preprocess

# 3) train BERTopic (tunes vectorizer/UMAP/HDBSCAN)
python scripts/train_bertopic.py  ## change the white list for the abstracts

# 4) export Lite artifacts (small CSVs for the app)
python scripts/export_rep_docs.py

# 5) (optional) run the Lite Streamlit app locally
streamlit run app/streamlit_app.py