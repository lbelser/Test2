"""Entry point for running the Plotly Dash dashboard locally."""
from __future__ import annotations

from dashboard import create_dashboard


dash_app = create_dashboard()
server = dash_app.server


def main() -> None:
    dash_app.run_server(debug=True)


if __name__ == "__main__":
    main()
