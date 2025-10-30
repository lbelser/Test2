"""Application factory for the Dash dashboard."""
from __future__ import annotations

from dash import Dash

from . import callbacks, data, layout

EXTERNAL_STYLESHEETS = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
]


def create_dashboard() -> Dash:
    """Create and configure the Dash application."""
    dataframe = data.load_data()
    filter_options = data.get_filter_options(dataframe)

    app = Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS, suppress_callback_exceptions=True)
    app.title = "Executive Performance Dashboard"
    app.layout = layout.create_layout(dataframe, filter_options)
    callbacks.register_callbacks(app)
    return app
