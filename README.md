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

These files are expected to already exist locally in `data/raw/`. They are not
downloaded at runtime, and the scripts raise a clear error if they are missing.

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

## Limitations

- **The XGBoost target is a proxy heuristic, not real user feedback.** The model
  is trained on a synthetic "good recommendation" label derived from similarity
  and movie-quality thresholds (vote average and vote count). To avoid target
  leakage, the three target-defining fields (`tfidf_similarity_score`,
  `candidate_vote_average`, `candidate_vote_count`) are deliberately **excluded**
  from the model's inputs; XGBoost predicts the proxy target from metadata only.
- **It is not personalised.** There is no user model — the same query always
  returns the same recommendations.
- **It may favour popular, highly rated movies**, because the proxy target
  rewards exactly those qualities.

This is honestly a TF-IDF content recommender plus a heuristic XGBoost reranker,
not a true personalised recommendation system.

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
title, and choose how many recommendations to return.

## Example output

`recommend_movies("The Dark Knight", top_n=5)`:

| rank | title | genres | release_year | similarity_score | recommendation_score | final_score |
|---|---|---|---|---|---|---|
| 1 | Batman Begins | Action, Crime, Drama | 2005 | 0.3248 | 0.7562 | 0.5405 |
| 2 | The Dark Knight Rises | Action, Crime, Drama, Thriller | 2012 | 0.4022 | 0.6535 | 0.5278 |
| 3 | Batman v Superman: Dawn of Justice | Action, Adventure, Fantasy | 2016 | 0.2158 | 0.2289 | 0.2224 |
| 4 | Batman: The Dark Knight Returns, Part 2 | Action, Animation | 2013 | 0.3203 | 0.0784 | 0.1994 |
| 5 | Batman Returns | Action, Fantasy | 1992 | 0.3300 | 0.0408 | 0.1854 |

## Future improvements

- Train on **real user feedback** instead of a proxy target.
- Incorporate **implicit feedback** (watches, clicks, ratings).
- Replace the heuristic reranker with a proper **learning-to-rank** model.
- Use **text/embedding** representations if the classical-ML constraint is lifted.
- Add **autocomplete / dropdown** title search in the app.
