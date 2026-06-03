"""Train the XGBoost heuristic reranker and save the four pipeline artifacts.

There is no real user-feedback data, so the model is trained on a *proxy*
target: a (input, candidate) pair is labelled a "good recommendation" when the
candidate is TF-IDF-similar to the input AND is itself a well-rated, well-known
movie. The three columns that define this target are deliberately excluded from
the feature matrix (see features.FEATURE_COLUMNS); the model must instead learn
to predict the proxy from metadata only (genre/cast/director/popularity/recency).
This keeps the task non-trivial and the validation AUC informative.

Artifacts written to models/ (all via utils.save_pickle / joblib):
    tfidf_vectorizer.pkl, xgboost_reranker.pkl, movie_data.pkl, tfidf_matrix.pkl

Critical contract: movie_data.pkl and tfidf_matrix.pkl share row order. The
dataframe index is reset before TF-IDF is built and never reordered afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Support both `python src/train_model.py` and `python -m src.train_model`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import features as F
from src.utils import MODELS_DIR, PROCESSED_DIR, ensure_dirs, save_pickle

RANDOM_STATE = 42
CANDIDATE_N = 50
# Final tuned threshold. TF-IDF similarities are heavily skewed toward zero, so
# the suggested default of 0.20 left only ~0.2% positives (effectively
# degenerate). Lowering to 0.10 yields a non-degenerate ~2.2% positive rate
# while still requiring genuine textual similarity; the vote thresholds below
# remain fixed per the proxy-target definition.
SIM_THRESHOLD = 0.10
TFIDF_MAX_FEATURES = 20000

DATA_PATH = PROCESSED_DIR / "movies_clean.csv"


def _load_clean_data() -> pd.DataFrame:
    """Load the processed CSV, with a clear error if preprocessing hasn't run."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing processed data: {DATA_PATH}\n"
            "Run preprocessing first:\n    python src/preprocess.py"
        )
    # Reset index and preserve this row order for the rest of the pipeline so
    # the saved dataframe stays aligned with the TF-IDF matrix.
    return pd.read_csv(DATA_PATH).reset_index(drop=True)


def build_training_pairs(df: pd.DataFrame, tfidf_matrix) -> pd.DataFrame:
    """Generate pairwise features for every movie's top-CANDIDATE_N neighbours.

    All movies are used as anchors (the dataset is small enough). Returns a
    single dataframe holding the full ten-column pairwise output.
    """
    frames: list[pd.DataFrame] = []
    n = len(df)
    for idx in range(n):
        sims = F.cosine_similarity_for_index(tfidf_matrix, idx)
        cand = F.top_candidates(sims, idx, CANDIDATE_N)
        if not cand:
            continue
        pf = F.pairwise_features(df.iloc[idx], df.iloc[cand], sims[cand])
        frames.append(pf)
        if (idx + 1) % 1000 == 0:
            print(f"  ...built pairs for {idx + 1}/{n} movies")
    return pd.concat(frames, ignore_index=True)


def make_proxy_target(pairs: pd.DataFrame) -> pd.Series:
    """Deterministic proxy label from the three target-defining columns.

    good_recommendation = 1 iff the candidate is sufficiently similar AND is a
    well-rated, well-known movie.
    """
    good = (
        (pairs["tfidf_similarity_score"] >= SIM_THRESHOLD)
        & (pairs["candidate_vote_average"] >= 7.0)
        & (pairs["candidate_vote_count"] >= 100)
    )
    return good.astype(int)


def run() -> None:
    ensure_dirs()

    df = _load_clean_data()
    print(f"Loaded {len(df)} movies from {DATA_PATH}")

    vectoriser, tfidf_matrix = F.build_tfidf(df["combined_text"].tolist())
    print(f"Built TF-IDF matrix: {tfidf_matrix.shape}")

    print("Building training pairs (top "
          f"{CANDIDATE_N} neighbours per movie)...")
    pairs = build_training_pairs(df, tfidf_matrix)
    print(f"Generated {len(pairs)} training pairs")

    y = make_proxy_target(pairs)
    pos_rate = float(y.mean())
    print(f"Positive-class rate: {pos_rate:.4f}")
    if pos_rate < 0.01 or pos_rate > 0.99:
        print("WARNING: positive-class rate is near-degenerate; "
              "consider adjusting SIM_THRESHOLD.")

    # Select ONLY the model-input columns; the target-defining columns never
    # reach the model (prevents trivial label reconstruction / leakage).
    X = pairs[F.FEATURE_COLUMNS]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    ll = log_loss(y_val, val_proba)
    print(f"Validation ROC-AUC: {auc:.4f}")
    print(f"Validation logloss: {ll:.4f}")
    if auc > 0.995:
        print("WARNING: ROC-AUC ~1.0 suggests a target-defining column leaked "
              "into X. Re-check X = pairs[FEATURE_COLUMNS].")

    # Persist artifacts. movie_data and tfidf_matrix share row order (df was
    # reset and never reordered).
    save_pickle(vectoriser, MODELS_DIR / "tfidf_vectorizer.pkl")
    save_pickle(model, MODELS_DIR / "xgboost_reranker.pkl")
    save_pickle(df, MODELS_DIR / "movie_data.pkl")
    save_pickle(tfidf_matrix, MODELS_DIR / "tfidf_matrix.pkl")
    print(f"Saved 4 artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    run()
