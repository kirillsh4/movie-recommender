"""Load the two raw TMDB CSVs and merge them into a single dataframe.

The movies file and the credits file share the same set of films, keyed by
``movies.id == credits.movie_id``. The credits file also ships a redundant
``title`` column, which we drop so the merged frame keeps one clean title.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running both as `python src/data_loader.py` (which puts src/ on the
# path) and as `python -m src.data_loader` from the repo root, by ensuring the
# project root is importable so the `src` package resolves either way.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import RAW_DIR


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    """Read a raw CSV, raising a clear, actionable error if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label} file: {path}\n"
            "Expected the TMDB 5000 dataset under data/raw/. Place "
            "tmdb_5000_movies.csv and tmdb_5000_credits.csv there and retry."
        )
    return pd.read_csv(path)


def load_movies(path: Path = RAW_DIR / "tmdb_5000_movies.csv") -> pd.DataFrame:
    """Load the TMDB movies CSV."""
    return _read_csv(path, "movies")


def load_credits(path: Path = RAW_DIR / "tmdb_5000_credits.csv") -> pd.DataFrame:
    """Load the TMDB credits CSV."""
    return _read_csv(path, "credits")


def merge_data(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    """Merge movies and credits on the shared film id.

    The credits ``title`` duplicates the movies ``title``; we drop it before
    merging so there is no ``title_x`` / ``title_y`` split afterwards.
    """
    credits = credits.rename(columns={"movie_id": "id"})
    if "title" in credits.columns:
        credits = credits.drop(columns=["title"])
    return movies.merge(credits, on="id", how="inner")


def get_merged_data() -> pd.DataFrame:
    """Convenience entry point: load both raw files and return the merge."""
    return merge_data(load_movies(), load_credits())


if __name__ == "__main__":
    df = get_merged_data()
    print(f"Merged shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
