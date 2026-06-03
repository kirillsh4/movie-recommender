"""Turn the raw merged TMDB dataframe into clean, model-ready columns.

The raw ``genres``/``keywords``/``cast``/``crew`` columns are JSON-ish strings
(Python list-of-dict literals). We parse them, extract the useful names, build a
single ``combined_text`` field for TF-IDF, and persist everything as CSV. Since
CSV cannot hold Python lists, list columns are stored as ``|``-delimited
strings via utils.join_list_field and reconstructed with utils.split_list_field.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

# Support both `python src/preprocess.py` and `python -m src.preprocess`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import get_merged_data
from src.utils import PROCESSED_DIR, ensure_dirs, join_list_field

# Columns the merged raw frame must contain before we attempt any parsing.
# Failing here gives a clear error instead of a cryptic one deep in parsing.
REQUIRED_COLUMNS = [
    "id",
    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "crew",
    "release_date",
    "vote_average",
    "vote_count",
    "popularity",
]

# Final clean schema written to CSV.
CLEAN_COLUMNS = [
    "id",
    "title",
    "overview",
    "genres_clean",
    "keywords_clean",
    "cast_clean",
    "director",
    "vote_average",
    "vote_count",
    "popularity",
    "release_year",
    "combined_text",
]

OUTPUT_PATH = PROCESSED_DIR / "movies_clean.csv"


# --- Parsing helpers ------------------------------------------------------
def parse_json_column(value) -> list:
    """Parse a TMDB list-of-dict string into a Python list.

    Returns ``[]`` on any failure (NaN, malformed literal, etc.) so a single
    bad row never aborts the whole preprocessing run.
    """
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def extract_names(parsed_list, key: str = "name", top_k=None) -> list[str]:
    """Pull the ``key`` field out of each dict, preserving listed order.

    ``top_k`` optionally caps how many names are returned (used for top-N cast).
    """
    if not isinstance(parsed_list, list):
        return []
    names = [
        item[key]
        for item in parsed_list
        if isinstance(item, dict) and item.get(key)
    ]
    return names[:top_k] if top_k is not None else names


def extract_director(crew_value) -> str:
    """Return the first crew member whose ``job`` is ``Director``, else ""."""
    parsed = parse_json_column(crew_value)
    for member in parsed:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "") or ""
    return ""


def _parse_release_year(release_date) -> int | None:
    """Extract a 4-digit year from a TMDB release_date string."""
    if not isinstance(release_date, str) or not release_date.strip():
        return None
    try:
        # release_date is YYYY-MM-DD; pandas handles odd / partial values.
        year = pd.to_datetime(release_date, errors="coerce").year
    except (ValueError, TypeError):
        return None
    return int(year) if pd.notna(year) else None


def _atomise(name: str) -> str:
    """Collapse a multi-word entity into one token, e.g. 'Tom Hanks' -> 'tomhanks'.

    This keeps people / genres atomic for TF-IDF so the model does not match on
    a shared first name like 'Tom' across unrelated actors.
    """
    return "".join(name.split()).lower()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Build the clean, model-ready dataframe from the merged raw frame."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Merged dataframe is missing required columns: "
            f"{missing}. Got columns: {df.columns.tolist()}"
        )

    out = pd.DataFrame()
    out["id"] = df["id"]

    # Simple text/number columns with documented missing-value handling.
    out["title"] = df["title"].fillna("").astype(str)
    out["overview"] = df["overview"].fillna("").astype(str)
    out["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0)
    out["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    out["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    out["release_year"] = df["release_date"].apply(_parse_release_year)

    # Parsed list columns (kept as in-memory lists for now).
    genres_list = df["genres"].apply(lambda v: extract_names(parse_json_column(v)))
    keywords_list = df["keywords"].apply(lambda v: extract_names(parse_json_column(v)))
    cast_list = df["cast"].apply(lambda v: extract_names(parse_json_column(v), top_k=5))
    out["director"] = df["crew"].apply(extract_director)

    # Build combined_text from the in-memory list versions BEFORE they are
    # joined for CSV storage. Multi-word names are atomised (spaces removed) so
    # each person/genre is a single TF-IDF token.
    def _build_combined(row_idx) -> str:
        parts: list[str] = []
        parts.append(out["title"].iat[row_idx])
        parts.append(out["overview"].iat[row_idx])
        parts.extend(_atomise(g) for g in genres_list.iat[row_idx])
        parts.extend(_atomise(k) for k in keywords_list.iat[row_idx])
        parts.extend(_atomise(c) for c in cast_list.iat[row_idx])
        director = out["director"].iat[row_idx]
        if director:
            parts.append(_atomise(director))
        return " ".join(p for p in parts if p).lower()

    out["combined_text"] = [_build_combined(i) for i in range(len(df))]

    # Serialise list columns to |-delimited strings for CSV persistence.
    out["genres_clean"] = genres_list.apply(join_list_field)
    out["keywords_clean"] = keywords_list.apply(join_list_field)
    out["cast_clean"] = cast_list.apply(join_list_field)

    return out[CLEAN_COLUMNS].reset_index(drop=True)


def run() -> pd.DataFrame:
    """Load -> preprocess -> write CSV -> return the processed dataframe."""
    ensure_dirs()
    merged = get_merged_data()
    processed = preprocess(merged)
    processed.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(processed)} rows to {OUTPUT_PATH}")
    print(f"Columns: {processed.columns.tolist()}")
    return processed


if __name__ == "__main__":
    run()
