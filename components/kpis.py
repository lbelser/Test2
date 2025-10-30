from __future__ import annotations

from typing import Optional

from dash import html


def build_kpi_card(card_id: str, title: str, value: str = "--", subtitle: Optional[str] = None) -> html.Div:
    """Return a styled KPI card container."""

    return html.Div(
        [
            html.Div(title, className="kpi-title"),
            html.Div(value, id=card_id, className="kpi-value"),
            html.Div(subtitle, className="kpi-subtitle") if subtitle else None,
        ],
        className="kpi-card",
    )

