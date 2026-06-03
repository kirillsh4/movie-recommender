"""Gradio web app for the hybrid movie recommender.

Thin UI layer over ``recommend_movies()``: it resolves the user's query to a
canonical title, shows which movie was matched, and renders the blended
recommendation table. All recommendation logic lives in ``src.recommender`` —
this module only handles input/output and friendly error messages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import pandas as pd

# Allow `python app.py` from the project root to import the src package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.recommender import recommend_movies, resolve_movie_title

# Empty table shape returned on any input error so the Dataframe stays valid.
_EMPTY = pd.DataFrame(
    columns=[
        "rank", "title", "genres", "release_year",
        "similarity_score", "recommendation_score", "final_score",
    ]
)


def recommend(movie_title: str, top_n: int):
    """Resolve the query, then return (recommendations_df, status_message).

    Normal user errors (blank or unknown titles) surface as a friendly status
    line with an empty table rather than crashing the app.
    """
    try:
        matched_title, _ = resolve_movie_title(movie_title)
        results = recommend_movies(matched_title, top_n=int(top_n))
        return results, f"Showing recommendations for: {matched_title}"
    except ValueError as exc:
        # Blank input or movie-not-found — raised by resolve_movie_title.
        return _EMPTY, str(exc)


with gr.Blocks(title="Hybrid Movie Recommender") as demo:
    gr.Markdown(
        """
        # Hybrid Movie Recommender

        Enter a movie title to get similar recommendations. **TF-IDF** retrieves
        textually similar candidate movies, an **XGBoost** reranker scores those
        candidates on metadata features (genre overlap, shared cast, director,
        popularity, recency), and the **final score** blends both signals equally:
        `final_score = 0.5 * recommendation_score + 0.5 * similarity_score`.
        """
    )

    with gr.Row():
        title_input = gr.Textbox(
            label="Movie title",
            placeholder="e.g. The Dark Knight",
        )
        top_n_input = gr.Slider(
            minimum=1, maximum=20, value=10, step=1,
            label="Number of recommendations",
        )

    search_button = gr.Button("Recommend", variant="primary")
    status = gr.Markdown()
    table = gr.Dataframe(label="Recommendations")

    # Trigger on button click and on Enter inside the textbox.
    search_button.click(
        recommend, inputs=[title_input, top_n_input], outputs=[table, status]
    )
    title_input.submit(
        recommend, inputs=[title_input, top_n_input], outputs=[table, status]
    )


if __name__ == "__main__":
    demo.launch()
