from __future__ import annotations

from dash import Dash

from callbacks import register_callbacks
from data import PREPARED_DATA
from layout import create_layout


EXTERNAL_STYLESHEETS = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
]


def create_app() -> Dash:
    app = Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)
    app.title = "Loyalty Marketing Intelligence"
    app.layout = create_layout(PREPARED_DATA)
    register_callbacks(app)
    return app


app = create_app()
server = app.server


if __name__ == "__main__":
    app.run(debug=True)

