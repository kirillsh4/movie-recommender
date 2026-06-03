# PROJECT.md — hybrid-movie-recommender

> **Execution document for Claude Code.**
> This file is the single source of truth for building the project. Work through it **phase by phase, top to bottom**. Do not skip ahead. After each phase, run the stated **Verification** steps and confirm the **Exit Criteria** are met before moving on. If a verification fails, fix it before proceeding.

---

## 0. Role, Mission & Operating Rules

### Your role

You are **Claude Code** operating inside a local VS Code workspace. You have a terminal, a filesystem, and the ability to create, edit, and run files. You build incrementally and verify as you go.

### Mission

Build a complete, runnable **1–2 day portfolio project** named **`hybrid-movie-recommender`**: a Gradio web app that takes a movie title and returns top-N recommendations, each annotated with a **TF-IDF similarity score**, an **XGBoost recommendation score**, and a **blended final score**.

### Minimum viable version

The project is complete when:

- A user enters a movie title.
- The system finds similar movies using TF-IDF cosine similarity.
- XGBoost reranks the TF-IDF candidate movies using metadata-based pairwise features.
- The app returns title, genres, release year, similarity score, recommendation score, and final blended score.

Do not add extra features until this core version works end-to-end.

### Hard constraints

- **Allowed libraries only:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `gradio`, `joblib`.
  - Standard library modules such as `ast`, `os`, `re`, `pickle`, `difflib`, and `pathlib` are fine.
- **Do NOT** use `matplotlib`, `seaborn`, or create an EDA notebook. This project should stay focused on the recommender pipeline and app.
- **Processed-data format:** persist processed data as **CSV**:
  ```text
  data/processed/movies_clean.csv
  ```
- **Do NOT** use parquet because it would require an extra engine dependency such as `pyarrow` or `fastparquet`.
- Because CSV does not preserve Python lists, list-valued columns must be stored as **`|`-delimited strings** and reconstructed with helper functions wherever set operations are needed.
- **Do NOT** use external APIs, network calls, or live data fetching.
- **Do NOT** use deep learning.
- **Do NOT** use collaborative filtering.
- **Do NOT** use matrix factorisation.
- **Do NOT** invent dataset columns. Work only with what TMDB 5000 provides.

### Important modelling note

The XGBoost model is not trained on real user preference data. It is trained on a **proxy target** based on similarity and movie-quality thresholds.

Use honest wording throughout the code comments and README:

> XGBoost is used as a heuristic reranker over TF-IDF candidates, using metadata-based features such as genre overlap, shared cast, director match, popularity, release-year difference, and same-decade match. Movie-quality fields (vote average and vote count) and the TF-IDF similarity score are deliberately excluded from the model's inputs because they define the proxy target; feeding them in would let the model trivially reconstruct the label.

Do not claim this is a true personalised recommendation system.

### Engineering principles

- **Working end-to-end beats clever.** Ship a coherent pipeline before optimising anything.
- **Keep the scope tight.** Build only what is needed for a 1–2 day portfolio project.
- **Run from project root.** Every script must be runnable as `python src/<file>.py` or `python -m src.<module>` from the repo root.
- **Comment with intent.** Explain why decisions are made, not what every line does.
- **Fail loudly and helpfully.** Validate inputs and raise clear errors.
- **Deterministic where possible.** Set `random_state=42` for sampling, splitting, and modelling.
- **Use phase-level verification.** Do not over-test every tiny helper individually unless something breaks.

### Dataset assumption

These files already exist locally and must not be downloaded:

```text
data/raw/tmdb_5000_movies.csv
data/raw/tmdb_5000_credits.csv
```

If they are missing at runtime, scripts should raise a clear, actionable error.

---

## 1. Target Structure

Build exactly this tree:

```text
hybrid-movie-recommender/
│
├── data/
│   ├── raw/
│   │   ├── tmdb_5000_movies.csv
│   │   └── tmdb_5000_credits.csv
│   └── processed/
│       └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train_model.py
│   ├── recommender.py
│   └── utils.py
│
├── models/
│   └── .gitkeep
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

Recommended `.gitignore`:

```text
.venv/
__pycache__/
*.pyc
models/*.pkl
data/processed/*
!data/processed/.gitkeep
.DS_Store
```

---

## 2. Phase Plan Overview

The project should be executed in **4 main phases**, not many small stages.

| Phase | Goal | Main deliverables |
|---|---|---|
| 1 | Setup + data pipeline | Environment, folder tree, raw data loading, preprocessing, clean CSV |
| 2 | Recommendation logic + model | TF-IDF retrieval, pairwise features, XGBoost reranker, saved artifacts |
| 3 | Gradio app | User-facing app using `recommend_movies()` |
| 4 | README + final smoke test | Recruiter-readable documentation and full end-to-end verification |

---

# PHASE 1 — Setup + Data Pipeline

## Goal

Create the project structure, install dependencies, load and merge the TMDB files, clean the data, and write the processed CSV.

This phase combines the old scaffold, utility, loader, and preprocessing stages into one coherent data-preparation phase.

---

## 1.1 Scaffold & environment

### Tasks

Create the folder tree from section 1.

Create `requirements.txt` containing exactly these pinned dependencies:

```text
pandas>=2.2,<3
numpy>=1.26,<3
scikit-learn>=1.4,<2
xgboost>=2.0,<3
gradio>=5,<6
joblib>=1.3,<2
```

These pins lock each library to a known-compatible major version so the documented commands keep working on a fresh install. After the environment is built and verified, optionally capture the exact resolved versions (including transitive dependencies) in a **separate lock file** for full reproducibility:

```bash
pip freeze > requirements-lock.txt
```

Do **not** freeze into `requirements.txt` itself — it must stay limited to the six top-level packages above. Commit `requirements-lock.txt` alongside it if you want byte-exact reproducibility.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Verification

```bash
python -c "import pandas, numpy, sklearn, xgboost, gradio, joblib; print('deps OK')"
ls data/raw
```

Expected raw files:

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

---

## 1.2 Implement `src/utils.py`

### Goal

Centralise paths, IO, and small reusable helpers.

### Implement

Path constants resolved relative to the project root:

- `PROJECT_ROOT`
- `RAW_DIR`
- `PROCESSED_DIR`
- `MODELS_DIR`

Functions:

```python
ensure_dirs()
save_pickle(obj, path)
load_pickle(path)
normalise_title(title: str) -> str
safe_year_to_decade(year) -> int | None
join_list_field(items: list[str]) -> str
split_list_field(value) -> list[str]
```

### Behaviour requirements

`save_pickle()` / `load_pickle()` should wrap `joblib.dump` / `joblib.load` so all model artifacts use a single IO path. `train_model.py` and `recommender.py` must read and write the four `.pkl` artifacts through these helpers (not a mix of `joblib` and `pickle` calls).

`normalise_title()` should:

- lowercase
- strip leading/trailing whitespace
- collapse repeated whitespace

`join_list_field()` should:

- drop empty values
- strip each item
- join using `|`

`split_list_field()` should:

- split `|`-delimited strings back into lists
- return `[]` for empty, missing, NaN, or non-string values

### Verification

```bash
python -c "from src.utils import PROJECT_ROOT, normalise_title; print(PROJECT_ROOT, normalise_title('  The   Matrix '))"
```

Expected title output:

```text
the matrix
```

---

## 1.3 Implement `src/data_loader.py`

### Goal

Load the two raw CSVs and merge them into one dataframe.

### Implement

```python
load_movies(path=RAW_DIR / "tmdb_5000_movies.csv") -> pd.DataFrame
load_credits(path=RAW_DIR / "tmdb_5000_credits.csv") -> pd.DataFrame
merge_data(movies, credits) -> pd.DataFrame
get_merged_data() -> pd.DataFrame
```

### Merge requirements

- Merge on `movies.id == credits.movie_id`.
- The credits file usually contains a duplicate `title` column.
- Drop the redundant credits title before merging so the final dataframe has a single clean `title` column.
- Raise `FileNotFoundError` with a clear message if either CSV is missing.

Add:

```python
if __name__ == "__main__":
    ...
```

The script should print the merged dataframe shape and columns.

### Verification

```bash
python src/data_loader.py
```

Expected:

- Around 4,803 rows.
- A single `title` column.
- No `title_x` or `title_y`.

---

## 1.4 Implement `src/preprocess.py`

### Goal

Turn the raw merged dataframe into clean, model-ready columns and persist them as CSV.

### Validate raw columns first

Before any parsing, check that the merged dataframe contains every column this stage depends on. Define the required set and raise a clear, actionable `ValueError` (or `KeyError` with a readable message) listing any that are missing, so a wrong or schema-shifted CSV fails loudly here instead of deep inside parsing.

Required merged columns:

```text
id, title, overview, genres, keywords, cast, crew,
release_date, vote_average, vote_count, popularity
```

(`genres`, `keywords`, `overview`, `release_date`, `vote_*`, `popularity`, `id`, `title` come from the movies file; `cast` and `crew` come from the credits file.)

### Parsing helpers

Use `ast.literal_eval`, wrapped in `try/except` returning `[]` on failure.

Implement:

```python
parse_json_column(value) -> list
extract_names(parsed_list, key="name", top_k=None) -> list[str]
extract_director(crew_value) -> str
```

### Clean columns to build

Build these columns:

- `id`
- `title`
- `overview`
- `genres_clean`
- `keywords_clean`
- `cast_clean`
- `director`
- `vote_average`
- `vote_count`
- `popularity`
- `release_year`
- `combined_text`

### Column logic

`genres_clean`:

- parsed genre names from `genres`

`keywords_clean`:

- parsed keyword names from `keywords`

`cast_clean`:

- top 5 cast names by listed order

`director`:

- crew member where `job == "Director"`
- empty string if none found

`release_year`:

- parsed from `release_date`
- handle missing or invalid dates

Missing-value handling:

- `overview`, `title` → `""`
- `vote_average`, `vote_count`, `popularity` → `0`

### `combined_text`

Build `combined_text` by concatenating:

```text
title + overview + genres_clean + keywords_clean + cast_clean + director
```

Requirements:

- lowercase the result
- use the in-memory list versions of genres, keywords, and cast before they are joined for CSV storage
- optionally remove spaces inside multi-word names so `"Tom Hanks"` becomes `"tomhanks"` to keep entities atomic
- document the chosen approach in a short comment

### Persisted list format

Before writing CSV:

- `genres_clean`
- `keywords_clean`
- `cast_clean`

must be stored as `|`-delimited strings using `utils.join_list_field()`.

### Public functions

```python
preprocess(df: pd.DataFrame) -> pd.DataFrame
run() -> pd.DataFrame
```

`run()` should:

1. load via `data_loader.get_merged_data()`
2. preprocess
3. write:

```text
data/processed/movies_clean.csv
```

4. return the processed dataframe

Add:

```python
if __name__ == "__main__":
    run()
```

### Verification

```bash
python src/preprocess.py
python -c "import pandas as pd; d=pd.read_csv('data/processed/movies_clean.csv'); print(d.columns.tolist()); print(d[['title','combined_text']].head(2))"
```

Expected:

- `data/processed/movies_clean.csv` exists
- all clean columns are present
- `combined_text` is non-empty
- list columns are `|`-delimited strings

---

## Phase 1 Exit Criteria

All must be true:

- [ ] Folder tree matches section 1.
- [ ] Dependencies import cleanly.
- [ ] Raw TMDB CSV files are present or missing-file errors are clear.
- [ ] Missing required raw columns raise a clear, listing error before parsing.
- [ ] `python src/data_loader.py` runs from the project root.
- [ ] `python src/preprocess.py` writes `data/processed/movies_clean.csv`.
- [ ] Processed CSV contains the exact clean schema.
- [ ] List columns are stored as `|`-delimited strings.
- [ ] No EDA notebook is created.

---

# PHASE 2 — Recommendation Logic + Model

## Goal

Build TF-IDF retrieval, pairwise candidate features, an XGBoost reranker, and the final `recommend_movies()` function.

This phase combines the old feature, training, and recommender stages into the core modelling phase.

---

## 2.1 Implement `src/features.py`

### Goal

Create TF-IDF vectorisation, cosine similarity retrieval, and pairwise candidate features.

### Implement

```python
FEATURE_COLUMNS = [...]
build_tfidf(corpus: list[str]) -> tuple
cosine_similarity_for_index(tfidf_matrix, idx) -> np.ndarray
top_candidates(sim_vector, idx, candidate_n) -> list[int]
pairwise_features(input_row, candidate_rows: pd.DataFrame, sim_scores) -> pd.DataFrame
```

### TF-IDF requirements

Use:

```python
TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 1)
)
```

Fit on `combined_text`.

### Candidate requirements

`top_candidates()` should:

- exclude the input movie itself
- return the indices of the top `candidate_n` most similar movies

### Pairwise feature requirements

Before set operations, reconstruct list columns with `utils.split_list_field()`.

Do **not** call `set()` directly on raw strings like `"Action|Adventure|Sci-Fi"`.

`pairwise_features()` should create one row per candidate containing **all** of the following columns:

- `tfidf_similarity_score`  *(used for the proxy target and the final blend — NOT a model input)*
- `genre_overlap_count`
- `genre_overlap_ratio`
- `same_director`
- `shared_cast_count`
- `candidate_vote_average`  *(used for the proxy target — NOT a model input)*
- `candidate_vote_count`  *(used for the proxy target — NOT a model input)*
- `candidate_popularity`
- `release_year_difference`
- `same_decade`

All ten columns are produced so that training can build the target and the recommender can build the blend. Only a subset of them is fed to the model.

### Avoiding target leakage

The proxy target is a deterministic function of `tfidf_similarity_score`, `candidate_vote_average`, and `candidate_vote_count` (see section 2.2). If those three columns were also given to XGBoost as inputs, the model could reconstruct the label almost perfectly and validation ROC-AUC would sit near 1.0 by construction — a meaningless number that hides the fact that the other features are being ignored.

To prevent this, `FEATURE_COLUMNS` deliberately **excludes** the three target-defining columns. XGBoost must instead predict the proxy target from metadata only (genre, cast, director, popularity, recency). This makes the learning task non-trivial, makes the AUC informative, and gives a clean separation of signals: TF-IDF supplies similarity, XGBoost supplies a metadata-quality signal, and they are combined later in the blend.

Define the exact model-input order in:

```python
FEATURE_COLUMNS = [
    "genre_overlap_count",
    "genre_overlap_ratio",
    "same_director",
    "shared_cast_count",
    "candidate_popularity",
    "release_year_difference",
    "same_decade",
]
```

Training and inference must both use this constant. Note that `candidate_popularity` is intentionally kept: it is correlated with movie quality but is not part of the target definition, so it acts as a legitimate (soft) signal rather than leakage.

### Verification

```bash
python -c "
import pandas as pd
from src import features as F

d = pd.read_csv('data/processed/movies_clean.csv').reset_index(drop=True)
vec, m = F.build_tfidf(d['combined_text'].tolist())
sims = F.cosine_similarity_for_index(m, 0)
cand = F.top_candidates(sims, 0, 5)
print('candidates', cand)
pf = F.pairwise_features(d.iloc[0], d.iloc[cand], sims[cand])
print(pf[F.FEATURE_COLUMNS].head())
"
```

Expected:

- 5 candidate rows
- all 7 columns in `FEATURE_COLUMNS` populated, no NaNs
- the full `pairwise_features()` output still also contains `tfidf_similarity_score`, `candidate_vote_average`, and `candidate_vote_count` (needed for the target and blend), even though they are absent from `FEATURE_COLUMNS`

---

## 2.2 Implement `src/train_model.py`

### Goal

Create training pairs, define a proxy target, train an XGBoost classifier, and save artifacts.

### Module-level constants

```python
RANDOM_STATE = 42
CANDIDATE_N = 50
SIM_THRESHOLD = 0.20
TFIDF_MAX_FEATURES = 20000
```

### Training pipeline

1. Load `data/processed/movies_clean.csv`.
   - If missing, run preprocessing first or raise a clear error telling the user to run:
     ```bash
     python src/preprocess.py
     ```

2. Reset index and preserve row order.

3. Build TF-IDF using `features.build_tfidf()` over `combined_text`.

4. Create training pairs:
   - For each movie, or for a documented random sample of movies if speed is an issue, take its top `CANDIDATE_N` TF-IDF neighbours.
   - Generate `pairwise_features()` for candidate rows.

5. Define proxy target:

```python
good_recommendation = 1 if (
    tfidf_similarity_score >= SIM_THRESHOLD
    and candidate_vote_average >= 7.0
    and candidate_vote_count >= 100
) else 0
```

Build this label from the full `pairwise_features()` output columns. The training design matrix `X`, however, must be selected as `X = pairs[FEATURE_COLUMNS]` so the three target-defining columns never reach the model (see "Avoiding target leakage" in section 2.1).

6. Print the positive-class rate.
   - If it is around 0% or 100%, adjust `SIM_THRESHOLD` and rerun.
   - Document the final threshold in a code comment.

7. Train `xgboost.XGBClassifier`:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.9,
    eval_metric="logloss",
    random_state=42
)
```

8. Use `train_test_split` with `random_state=42`.

9. Print validation:
   - ROC-AUC
   - logloss

These are only diagnostics for the proxy task, not proof of real recommendation quality. Because the target-defining columns are excluded from `FEATURE_COLUMNS`, the ROC-AUC should land meaningfully below 1.0 (a believable range is roughly 0.75–0.92). An AUC at or near 1.0 means a target-defining column has leaked into the feature matrix — re-check `X = pairs[FEATURE_COLUMNS]`.

10. Save artifacts with `utils.save_pickle()` (which wraps `joblib`) into `models/`:

```text
models/tfidf_vectorizer.pkl
models/xgboost_reranker.pkl
models/movie_data.pkl
models/tfidf_matrix.pkl
```

### Critical artifact contract

`movie_data.pkl` row order must match `tfidf_matrix.pkl` row order.

Reset the dataframe index before saving and do not reorder it afterwards.

### Verification

```bash
python src/train_model.py
ls -la models
```

Expected:

- positive-class rate prints
- validation ROC-AUC and logloss print
- four `.pkl` files exist

---

## 2.3 Implement `src/recommender.py`

### Goal

Create the inference function that powers the app.

### Implement

```python
def resolve_movie_title(movie_title: str) -> tuple[str, int]:
    """Resolve a user query to a canonical (title, row_index) in movie_data.
    Raises ValueError with a clear message if no match is found."""
    ...

def recommend_movies(movie_title: str, top_n: int = 10, candidate_n: int = 50) -> pd.DataFrame:
    ...
```

### Behaviour

1. Lazy-load artifacts once using a module-level cache, reading each with `utils.load_pickle()`:
   - `tfidf_vectorizer`
   - `xgboost_reranker`
   - `movie_data`
   - `tfidf_matrix`

2. If artifacts are missing, raise a clear error telling the user to run:

```bash
python src/train_model.py
```

3. Resolve the title with `resolve_movie_title(movie_title)`, which performs:
   - case-insensitive exact match via `utils.normalise_title`
   - contains or startswith fallback
   - optional fuzzy match using `difflib.get_close_matches`
   - if more than one exact match exists, pick deterministically (e.g. the row with the highest `vote_count`)

   It returns the canonical matched title and its row index in `movie_data`. `recommend_movies()` calls this helper rather than duplicating the matching logic.

4. If no match is found, `resolve_movie_title()` raises `ValueError` with a clear movie-not-found message.

5. Retrieve candidates:
   - cosine similarity of the matched movie's TF-IDF row against all rows
   - top `candidate_n`
   - exclude the input movie itself

6. Engineer features using `features.pairwise_features()` and `FEATURE_COLUMNS`.

7. Score candidates:

```python
recommendation_score = xgboost_reranker.predict_proba(X)[:, 1]
```

8. Blend scores:

```python
similarity_score = tfidf_similarity_score
final_score = 0.5 * recommendation_score + 0.5 * similarity_score
```

9. Sort by `final_score` descending.

10. Return exactly these columns:

```text
rank, title, genres, release_year, similarity_score, recommendation_score, final_score
```

### Output requirements

- `rank` starts at 1.
- `genres` should be readable, comma-joined text.
- score columns should be rounded to 4 decimal places.
- scores should be in `[0, 1]`.

### Verification

```bash
python -c "from src.recommender import resolve_movie_title; print(resolve_movie_title('dark knight'))"
python -c "from src.recommender import recommend_movies; print(recommend_movies('The Dark Knight', top_n=5).to_string(index=False))"
python -c "from src.recommender import recommend_movies; recommend_movies('zzz-not-a-movie')" || echo 'handled-not-found OK'
```

Expected:

- `resolve_movie_title('dark knight')` returns the canonical title and its index
- a clean 5-row table for a real title
- a clear error for a fake title

---

## Phase 2 Exit Criteria

All must be true:

- [ ] `features.py` builds TF-IDF and pairwise features successfully.
- [ ] `FEATURE_COLUMNS` is defined and used in both training and inference.
- [ ] `FEATURE_COLUMNS` excludes `tfidf_similarity_score`, `candidate_vote_average`, and `candidate_vote_count` (the target-defining columns).
- [ ] `train_model.py` saves all four artifacts.
- [ ] Positive-class rate is non-degenerate.
- [ ] Validation ROC-AUC and logloss print finite values, with AUC below 1.0 (no leakage).
- [ ] `recommend_movies()` returns the exact 7-column schema.
- [ ] `resolve_movie_title()` returns the canonical matched title and index, and `recommend_movies()` uses it for matching.
- [ ] Exact, partial, and missing title inputs are handled gracefully.
- [ ] Score blending is implemented as `0.5 * recommendation_score + 0.5 * similarity_score`.

---

# PHASE 3 — Gradio App

## Goal

Build a minimal Gradio web app that wraps `recommend_movies()`.

---

## Implement `app.py`

### Requirements

Import:

```python
from src.recommender import recommend_movies, resolve_movie_title
```

Inputs:

- movie title: `gr.Textbox`
- number of recommendations: `gr.Slider(minimum=1, maximum=20, value=10, step=1)`

Outputs:

- `gr.Dataframe` for recommendations
- simple status message for errors or successful match

Wrapper behaviour:

- resolve the query with `resolve_movie_title(title)` so the matched movie can be shown
- on success, display a status line such as `Showing recommendations for: The Dark Knight` and call `recommend_movies(title, top_n)`
- if the title is missing or not found (the `ValueError` from `resolve_movie_title`/`recommend_movies`), return an empty dataframe and a helpful message
- do not crash the app for normal user input errors

App text:

- title
- 2–3 sentence description explaining:
  - TF-IDF retrieves similar movies
  - XGBoost reranks candidate movies
  - final score blends both signals

Launch:

```python
if __name__ == "__main__":
    demo.launch()
```

---

## Verification

```bash
python app.py
```

Manual checks:

- open the printed local URL
- search `"Inception"`
- confirm a table renders and the status shows which title was matched
- search a nonsense title
- confirm a friendly message appears

---

## Phase 3 Exit Criteria

All must be true:

- [ ] App launches from the project root.
- [ ] User can enter a movie title.
- [ ] App returns the recommendation table.
- [ ] App displays which movie title was matched for the query.
- [ ] App handles unknown titles without crashing.
- [ ] App uses the existing `recommend_movies()` function rather than duplicating recommendation logic.

---

# PHASE 4 — README + Final Smoke Test

## Goal

Write concise documentation and verify the full project from preprocessing to app launch.

---

## 4.1 Write `README.md`

The README should be recruiter-readable and not overly long.

Include these sections:

1. **Project overview**
   - What the project does and why it exists.

2. **Dataset**
   - TMDB 5000 movies and credits files.
   - Mention that files are expected locally in `data/raw/`.

3. **Methodology**
   - TF-IDF over `combined_text`.
   - Cosine similarity for candidate retrieval.
   - XGBoost heuristic reranker over candidate features.

4. **How scoring works**
   - Explain:
     - `similarity_score`
     - `recommendation_score`
     - `final_score = 0.5 * recommendation_score + 0.5 * similarity_score`

5. **Limitations**
   - Be explicit that the XGBoost target is a proxy heuristic, not real user feedback.
   - Mention that the recommender is not personalised.
   - Mention that it may favour popular, highly rated movies.

6. **Installation**
   - venv setup
   - `pip install -r requirements.txt`

7. **How to run**
   - preprocessing
   - training
   - app launch

8. **Example output**
   - include a small markdown table from a real recommendation result

9. **Future improvements**
   - real user feedback
   - implicit feedback
   - proper learning-to-rank
   - embeddings if constraints are lifted
   - autocomplete/dropdown title search

### Required commands in README

```bash
python src/preprocess.py
python src/train_model.py
python app.py
```

Make sure every command works from the project root.

---

## 4.2 Final end-to-end smoke test

Run this sequence from a clean state:

```bash
source .venv/bin/activate
python src/preprocess.py
python src/train_model.py
python -c "from src.recommender import recommend_movies; print(recommend_movies('Toy Story', top_n=10).to_string(index=False))"
python app.py
```

Expected:

- processed CSV is created
- four model artifacts are created
- `recommend_movies()` returns a ranked table
- Gradio launches successfully

---

## Phase 4 Exit Criteria

All must be true:

- [ ] README is concise and recruiter-readable.
- [ ] README commands work exactly as written.
- [ ] Full smoke test passes from preprocessing to app launch.
- [ ] Project can be explained honestly as a TF-IDF recommender plus heuristic XGBoost reranker.

---

# Definition of Done

The project is complete only when every item below is true:

- [ ] Folder tree matches section 1.
- [ ] `requirements.txt` lists only these packages, each with a version pin:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `gradio`
  - `joblib`
- [ ] No EDA notebook is created.
- [ ] No `matplotlib` dependency is used.
- [ ] No banned techniques are used:
  - no APIs
  - no deep learning
  - no collaborative filtering
  - no matrix factorisation
- [ ] `python src/preprocess.py` writes `data/processed/movies_clean.csv`.
- [ ] Processed CSV contains the exact clean schema.
- [ ] List columns are `|`-delimited and round-trip through `utils.split_list_field()`.
- [ ] `python src/train_model.py` saves all four artifacts.
- [ ] `movie_data.pkl` and `tfidf_matrix.pkl` are index-aligned.
- [ ] Positive class rate is non-degenerate.
- [ ] `recommend_movies()` returns:
  ```text
  rank, title, genres, release_year, similarity_score, recommendation_score, final_score
  ```
- [ ] Scores are in `[0, 1]`.
- [ ] Title matching handles exact, partial, fuzzy, and missing titles gracefully.
- [ ] Gradio app launches from root and renders recommendations.
- [ ] README clearly explains the proxy-target limitation.
- [ ] Every script runs from the project root.

---

# Appendix — Quick Reference

## Final score formula

```text
similarity_score      = TF-IDF cosine similarity between input movie and candidate
recommendation_score  = XGBoost predict_proba output for candidate pair
final_score           = 0.5 * recommendation_score + 0.5 * similarity_score
```

## Proxy target

```text
good_recommendation = 1 if (
    tfidf_similarity_score >= SIM_THRESHOLD
    and candidate_vote_average >= 7.0
    and candidate_vote_count >= 100
) else 0
```

## Artifact contract

```text
models/tfidf_vectorizer.pkl   # fitted TfidfVectorizer
models/xgboost_reranker.pkl   # fitted XGBClassifier
models/movie_data.pkl         # processed dataframe, index 0..n-1
models/tfidf_matrix.pkl       # sparse matrix, rows aligned to movie_data
```

## Suggested config defaults

```text
RANDOM_STATE       = 42
CANDIDATE_N        = 50
SIM_THRESHOLD      = 0.20
TOP_N              = 10
TFIDF_MAX_FEATURES = 20000
```

Build the minimum viable version first. Verify each phase. Do not declare the project complete until the full end-to-end smoke test passes.
