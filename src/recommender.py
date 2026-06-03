"""Inference: resolve a query title and return blended recommendations.

This is the single entry point the Gradio app calls. It lazy-loads the four
artifacts once, retrieves TF-IDF candidates for the matched movie, reranks them
with the XGBoost model over metadata features, and blends the two signals:

    final_score = 0.5 * recommendation_score + 0.5 * similarity_score

The XGBoost reranker is a heuristic over metadata (genre overlap, shared cast,
director match, popularity, recency); it is not trained on real user feedback.
Treat its output as a metadata-quality signal, not personalised relevance.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

import pandas as pd

# Support both `python src/recommender.py` and `python -m src.recommender`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import features as F
from src.utils import MODELS_DIR, load_pickle, normalise_title, split_list_field

CANDIDATE_N = 50
TOP_N = 10

# Module-level artifact cache so the app loads the models only once.
_ARTIFACTS: dict | None = None

_ARTIFACT_FILES = {
    "tfidf_vectorizer": "tfidf_vectorizer.pkl",
    "xgboost_reranker": "xgboost_reranker.pkl",
    "movie_data": "movie_data.pkl",
    "tfidf_matrix": "tfidf_matrix.pkl",
}


def _load_artifacts() -> dict:
    """Lazy-load and cache the four pipeline artifacts."""
    global _ARTIFACTS
    if _ARTIFACTS is not None:
        return _ARTIFACTS

    missing = [
        name for name, fname in _ARTIFACT_FILES.items()
        if not (MODELS_DIR / fname).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {missing}.\n"
            "Train the model first:\n    python src/train_model.py"
        )

    artifacts = {
        name: load_pickle(MODELS_DIR / fname)
        for name, fname in _ARTIFACT_FILES.items()
    }
    # Ensure a known, position-based index so row positions line up with the
    # TF-IDF matrix rows regardless of how the dataframe was saved.
    artifacts["movie_data"] = artifacts["movie_data"].reset_index(drop=True)
    _ARTIFACTS = artifacts
    return _ARTIFACTS


def resolve_movie_title(movie_title: str) -> tuple[str, int]:
    """Resolve a user query to a canonical (title, row_index) in movie_data.

    Matching cascade: case-insensitive exact -> startswith -> contains ->
    fuzzy (difflib). When several rows share the matched title, the most-voted
    one is chosen so the result is deterministic. Raises ValueError if nothing
    matches.
    """
    if not isinstance(movie_title, str) or not movie_title.strip():
        raise ValueError("Please enter a movie title.")

    data = _load_artifacts()["movie_data"]
    query = normalise_title(movie_title)
    norm_titles = data["title"].apply(normalise_title)

    def _pick(mask) -> int | None:
        """Among matching rows, pick the highest vote_count deterministically."""
        matches = data.index[mask]
        if len(matches) == 0:
            return None
        if len(matches) == 1:
            return int(matches[0])
        subset = data.loc[matches]
        return int(subset["vote_count"].astype(float).idxmax())

    # 1) exact, 2) startswith, 3) contains
    for mask in (
        norm_titles == query,
        norm_titles.str.startswith(query),
        norm_titles.str.contains(query, regex=False),
    ):
        idx = _pick(mask)
        if idx is not None:
            return str(data.at[idx, "title"]), idx

    # 4) fuzzy fallback on normalised titles
    close = difflib.get_close_matches(query, norm_titles.tolist(), n=1, cutoff=0.6)
    if close:
        idx = _pick(norm_titles == close[0])
        if idx is not None:
            return str(data.at[idx, "title"]), idx

    raise ValueError(
        f"Movie not found: {movie_title!r}. "
        "Try a different or more complete title."
    )


def _format_genres(value) -> str:
    """Render the |-delimited genres column as readable, comma-joined text."""
    return ", ".join(split_list_field(value))


def recommend_movies(
    movie_title: str,
    top_n: int = TOP_N,
    candidate_n: int = CANDIDATE_N,
) -> pd.DataFrame:
    """Return the top-``top_n`` blended recommendations for ``movie_title``.

    Output columns (exact order):
        rank, title, genres, release_year,
        similarity_score, recommendation_score, final_score
    """
    artifacts = _load_artifacts()
    vectoriser = artifacts["tfidf_vectorizer"]  # noqa: F841  (loaded for contract)
    model = artifacts["xgboost_reranker"]
    data = artifacts["movie_data"]
    tfidf_matrix = artifacts["tfidf_matrix"]

    matched_title, idx = resolve_movie_title(movie_title)

    # Retrieve candidates via cosine similarity, excluding the input movie.
    sims = F.cosine_similarity_for_index(tfidf_matrix, idx)
    cand = F.top_candidates(sims, idx, candidate_n)
    if not cand:
        # Degenerate corner case (e.g. a single-row dataset).
        return pd.DataFrame(
            columns=[
                "rank", "title", "genres", "release_year",
                "similarity_score", "recommendation_score", "final_score",
            ]
        )

    candidate_rows = data.iloc[cand]
    pairs = F.pairwise_features(data.iloc[idx], candidate_rows, sims[cand])

    # Rerank with XGBoost over the metadata features only, then blend.
    recommendation_score = model.predict_proba(pairs[F.FEATURE_COLUMNS])[:, 1]
    similarity_score = pairs["tfidf_similarity_score"].to_numpy()
    final_score = 0.5 * recommendation_score + 0.5 * similarity_score

    out = pd.DataFrame(
        {
            "title": candidate_rows["title"].to_numpy(),
            "genres": [_format_genres(g) for g in candidate_rows["genres_clean"]],
            "release_year": candidate_rows["release_year"].to_numpy(),
            "similarity_score": similarity_score,
            "recommendation_score": recommendation_score,
            "final_score": final_score,
        }
    )

    out = out.sort_values("final_score", ascending=False).head(int(top_n))
    out = out.reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)

    # release_year is nicer as a plain integer where present.
    out["release_year"] = out["release_year"].astype("Int64")
    for col in ("similarity_score", "recommendation_score", "final_score"):
        out[col] = out[col].round(4)

    return out[
        [
            "rank", "title", "genres", "release_year",
            "similarity_score", "recommendation_score", "final_score",
        ]
    ]


if __name__ == "__main__":
    print(resolve_movie_title("dark knight"))
    print(recommend_movies("The Dark Knight", top_n=5).to_string(index=False))
