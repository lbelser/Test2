from __future__ import annotations

import pandas as pd
import plotly.express as px
from dash import dcc


def build_correlation_graph(graph_id: str, matrix: pd.DataFrame) -> dcc.Graph:
    """Render a heatmap for the supplied correlation matrix."""

    if matrix.empty:
        figure = px.imshow([[0]], text_auto=True, title="No data available")
    else:
        figure = px.imshow(
            matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlation heatmap",
        )
        figure.update_layout(margin=dict(l=40, r=20, t=60, b=40))

    return dcc.Graph(id=graph_id, figure=figure)

