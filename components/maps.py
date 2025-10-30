from __future__ import annotations

import pandas as pd
import plotly.express as px
from dash import dcc


def build_map_graph(graph_id: str, data: pd.DataFrame) -> dcc.Graph:
    """Generate a scatter map of customer metrics by province."""

    if data.empty:
        figure = px.scatter_mapbox(title="No geographic data")
    else:
        figure = px.scatter_mapbox(
            data,
            lat="latitude",
            lon="longitude",
            size="customers",
            color="avg_ltv",
            hover_name="province",
            hover_data={
                "customers": True,
                "avg_ltv": ":.2f",
                "avg_tenure_months": ":.1f",
            },
            color_continuous_scale="Plasma",
            size_max=45,
            zoom=2,
        )
        figure.update_layout(
            mapbox_style="carto-positron",
            margin=dict(l=10, r=10, t=40, b=10),
        )

    return dcc.Graph(id=graph_id, figure=figure)

