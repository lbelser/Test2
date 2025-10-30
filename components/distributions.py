from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
from dash import dcc


def build_distribution_graph(
    graph_id: str,
    data: pd.DataFrame,
    column: str,
    chart_type: str = "histogram",
    bins: Optional[int] = None,
    category: Optional[str] = None,
) -> dcc.Graph:
    """Return a histogram or box plot graph component."""

    if column not in data.columns:
        figure = px.scatter(title="No data available")
    else:
        filtered = data.dropna(subset=[column])
        if chart_type == "box":
            figure = px.box(
                filtered,
                x=category,
                y=column,
                color=category if category else None,
                points="suspectedoutliers",
            )
        else:
            figure = px.histogram(
                filtered,
                x=column,
                nbins=bins,
                color=category if category else None,
                marginal="box" if category is None else None,
            )

        figure.update_layout(margin=dict(l=10, r=10, t=40, b=10))

    return dcc.Graph(id=graph_id, figure=figure)

