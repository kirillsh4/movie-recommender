"""TF-IDF retrieval and pairwise candidate features.

This module supplies the two halves of the recommender's signal:

* TF-IDF + cosine similarity to retrieve candidate movies similar to a query.
* Metadata-based pairwise features describing each (input, candidate) pair,
  which the XGBoost reranker scores.

`pairwise_features` produces ten columns. Only the seven listed in
`FEATURE_COLUMNS` are fed to the model. The remaining three
(`tfidf_similarity_score`, `candidate_vote_average`, `candidate_vote_count`)
define the proxy training target (see train_model.py); feeding them to the
model would let it trivially reconstruct the label, so they are deliberately
withheld. `candidate_popularity` is kept: it correlates with quality but is not
part of the target definition, so it is a legitimate soft signal, not leakage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Support both `python src/features.py` and `python -m src.features`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import safe_year_to_decade, split_list_field

# Exact model-input order. Used identically in training and inference so the
# design matrix columns always line up with what the model was fit on. The
# three target-defining columns are intentionally absent (see module docstring).
FEATURE_COLUMNS = [
    "genre_overlap_count",
    "genre_overlap_ratio",
    "same_director",
    "shared_cast_count",
    "candidate_popularity",
    "release_year_difference",
    "same_decade",
]

# Sentinel year-difference used when either movie has no parseable release year.
# A large value marks "very different / unknown era" without introducing NaNs
# into the feature matrix.
_UNKNOWN_YEAR_DIFF = 100

TFIDF_MAX_FEATURES = 20000


def build_tfidf(corpus: list[str]) -> tuple[TfidfVectorizer, "object"]:
    """Fit a TF-IDF vectoriser on ``corpus`` and return (vectoriser, matrix).

    The vectoriser uses English stop words, unigrams only, and a capped
    vocabulary. The returned matrix is L2-normalised (sklearn default), which
    lets cosine similarity be computed as a plain dot product downstream.
    """
    vectoriser = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 1),
    )
    # Guard against NaN/None slipping into the corpus.
    clean_corpus = [text if isinstance(text, str) else "" for text in corpus]
    matrix = vectoriser.fit_transform(clean_corpus)
    return vectoriser, matrix


def cosine_similarity_for_index(tfidf_matrix, idx: int) -> np.ndarray:
    """Cosine similarity of row ``idx`` against every row in ``tfidf_matrix``.

    The matrix is L2-normalised, so the linear (dot-product) kernel already
    equals cosine similarity and avoids re-normalising. Returns a 1-D array of
    length ``n_rows``.
    """
    sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix)
    return sims.ravel()


def top_candidates(sim_vector: np.ndarray, idx: int, candidate_n: int) -> list[int]:
    """Return indices of the ``candidate_n`` most similar movies to ``idx``.

    The input movie itself is excluded. Results are ordered by descending
    similarity.
    """
    sims = np.asarray(sim_vector, dtype=float).copy()
    sims[idx] = -np.inf  # never recommend the movie to itself
    n_available = np.count_nonzero(np.isfinite(sims))
    k = int(min(candidate_n, n_available))
    if k <= 0:
        return []
    # argpartition for the top-k, then sort just those by similarity desc.
    top_unsorted = np.argpartition(sims, -k)[-k:]
    top_sorted = top_unsorted[np.argsort(sims[top_unsorted])[::-1]]
    return top_sorted.tolist()


def _as_set(value) -> set[str]:
    """Reconstruct a |-delimited list column into a lowercased set."""
    return {item.lower() for item in split_list_field(value)}


def pairwise_features(
    input_row: pd.Series,
    candidate_rows: pd.DataFrame,
    sim_scores,
) -> pd.DataFrame:
    """Build one feature row per candidate for a single input movie.

    Returns a dataframe (indexed like ``candidate_rows``) with all ten columns:
    the seven model inputs in ``FEATURE_COLUMNS`` plus the three target-defining
    columns (``tfidf_similarity_score``, ``candidate_vote_average``,
    ``candidate_vote_count``) needed to build the proxy target and the blend.
    """
    sim_scores = np.asarray(sim_scores, dtype=float)

    # Input-side attributes, computed once.
    input_genres = _as_set(input_row.get("genres_clean"))
    input_cast = _as_set(input_row.get("cast_clean"))
    input_director = str(input_row.get("director") or "").strip().lower()
    input_year = input_row.get("release_year")
    input_decade = safe_year_to_decade(input_year)
    input_year_val = None
    try:
        if input_year is not None and input_year == input_year:  # not NaN
            input_year_val = int(input_year)
    except (TypeError, ValueError):
        input_year_val = None

    records: list[dict] = []
    for pos, (_, cand) in enumerate(candidate_rows.iterrows()):
        cand_genres = _as_set(cand.get("genres_clean"))
        cand_cast = _as_set(cand.get("cast_clean"))
        cand_director = str(cand.get("director") or "").strip().lower()

        genre_overlap = input_genres & cand_genres
        genre_union = input_genres | cand_genres
        overlap_count = len(genre_overlap)
        overlap_ratio = overlap_count / len(genre_union) if genre_union else 0.0

        same_director = int(bool(input_director) and input_director == cand_director)
        shared_cast_count = len(input_cast & cand_cast)

        # Release-year difference and same-decade, robust to missing years.
        cand_year = cand.get("release_year")
        cand_year_val = None
        try:
            if cand_year is not None and cand_year == cand_year:
                cand_year_val = int(cand_year)
        except (TypeError, ValueError):
            cand_year_val = None

        if input_year_val is not None and cand_year_val is not None:
            year_diff = abs(input_year_val - cand_year_val)
        else:
            year_diff = _UNKNOWN_YEAR_DIFF

        cand_decade = safe_year_to_decade(cand_year)
        same_decade = int(
            input_decade is not None
            and cand_decade is not None
            and input_decade == cand_decade
        )

        records.append(
            {
                # Target-defining columns (NOT model inputs).
                "tfidf_similarity_score": float(sim_scores[pos]),
                "candidate_vote_average": float(cand.get("vote_average") or 0.0),
                "candidate_vote_count": float(cand.get("vote_count") or 0.0),
                # Model-input columns.
                "genre_overlap_count": overlap_count,
                "genre_overlap_ratio": overlap_ratio,
                "same_director": same_director,
                "shared_cast_count": shared_cast_count,
                "candidate_popularity": float(cand.get("popularity") or 0.0),
                "release_year_difference": year_diff,
                "same_decade": same_decade,
            }
        )

    return pd.DataFrame(records, index=candidate_rows.index)


if __name__ == "__main__":
    # Smoke test mirroring the Phase 2.1 verification.
    data = (
        pd.read_csv("data/processed/movies_clean.csv").reset_index(drop=True)
    )
    vec, mat = build_tfidf(data["combined_text"].tolist())
    sims = cosine_similarity_for_index(mat, 0)
    cand = top_candidates(sims, 0, 5)
    print("candidates", cand)
    pf = pairwise_features(data.iloc[0], data.iloc[cand], sims[cand])
    print(pf[FEATURE_COLUMNS].head())
