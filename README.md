# Hybrid Movie Recommender

A content-based movie recommender that takes a movie title and returns the
top-N most relevant movies, each annotated with a **TF-IDF similarity score**,
an **XGBoost recommendation score**, and a **blended final score**. It ships
with a small Gradio web app.

## Project overview

Given a movie you like, the system retrieves textually similar movies and
reranks them using metadata signals, then blends the two into a single ranked
list. It exists as a compact, end-to-end portfolio example of a two-stage
**retrieve-then-rerank** recommendation pipeline built only with classical ML.

## Dataset

Built on the **TMDB 5000** dataset:

- `data/raw/tmdb_5000_movies.csv` — titles, overviews, genres, keywords, votes, popularity
- `data/raw/tmdb_5000_credits.csv` — cast and crew


## Methodology

1. **Preprocessing** — the two raw files are merged and parsed into clean
   columns (genres, keywords, top-5 cast, director, release year) and a
   `combined_text` field used for vectorisation.
2. **Retrieval (TF-IDF)** — `combined_text` is vectorised with a
   `TfidfVectorizer`, and candidate movies are retrieved by **cosine
   similarity** against the query movie.
3. **Reranking (XGBoost)** — the candidates are scored by an `XGBClassifier`
   trained over pairwise metadata features: genre overlap, shared cast,
   director match, popularity, release-year difference, and same-decade match.

## How scoring works

Each recommendation carries three scores, all in `[0, 1]`:

- **`similarity_score`** — TF-IDF cosine similarity between the input movie and the candidate.
- **`recommendation_score`** — the XGBoost reranker's `predict_proba` output for the candidate pair (a metadata-quality signal).
- **`final_score`** — an equal blend of the two:

  ```text
  final_score = 0.5 * recommendation_score + 0.5 * similarity_score
  ```

Results are sorted by `final_score` descending.


## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

Run every command from the project root:

```bash
# 1. Build data/processed/movies_clean.csv from the raw TMDB files
python src/preprocess.py

# 2. Train the TF-IDF + XGBoost pipeline and save model artifacts to models/
python src/train_model.py

# 3. Launch the Gradio web app
python app.py
```

Then open the printed local URL (default `http://127.0.0.1:7860`), enter a movie
title, and choose how many recommendations to return!

