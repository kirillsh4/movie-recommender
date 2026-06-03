"""Shared paths, IO, and small reusable helpers.

Centralising these here keeps a single, consistent path for model IO (all
artifacts go through joblib via save_pickle/load_pickle) and a single
definition of how list-valued columns are serialised to / from CSV.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib

# --- Path constants -------------------------------------------------------
# Resolve everything relative to the project root (the parent of src/) so the
# scripts work no matter which directory Python is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_dirs() -> None:
    """Create the data/model output directories if they do not yet exist."""
    for directory in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_pickle(obj, path) -> None:
    """Persist any object to ``path`` via joblib.

    All model artifacts go through this single IO path so we never mix joblib
    and pickle calls across the codebase.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path):
    """Load an object previously saved with :func:`save_pickle`."""
    return joblib.load(Path(path))


def normalise_title(title: str) -> str:
    """Lowercase, strip, and collapse repeated whitespace in a title."""
    if not isinstance(title, str):
        return ""
    collapsed = re.sub(r"\s+", " ", title)
    return collapsed.strip().lower()


def safe_year_to_decade(year) -> int | None:
    """Map a release year to its decade (e.g. 1997 -> 1990).

    Returns None for missing / unparseable input so downstream code can treat
    "no decade" explicitly rather than guessing.
    """
    try:
        if year is None:
            return None
        year_int = int(year)
    except (TypeError, ValueError):
        return None
    if year_int <= 0:
        return None
    return (year_int // 10) * 10


def join_list_field(items: list[str]) -> str:
    """Serialise a list of strings to a ``|``-delimited string for CSV storage.

    Empty / whitespace-only items are dropped and each kept item is stripped.
    """
    if items is None:
        return ""
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    return "|".join(cleaned)


def split_list_field(value) -> list[str]:
    """Reconstruct a list from a ``|``-delimited string written by
    :func:`join_list_field`.

    Returns ``[]`` for empty, missing, NaN, or non-string values so set
    operations downstream never choke on bad input.
    """
    # NaN is the canonical "missing" value pandas produces; it is a float and
    # not equal to itself, which is the cheapest reliable way to detect it
    # without importing pandas here.
    if value is None or value != value:
        return []
    if not isinstance(value, str):
        return []
    return [item.strip() for item in value.split("|") if item.strip()]
