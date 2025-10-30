from __future__ import annotations

from dash import dcc, html
import plotly.express as px

from components import (
    build_correlation_graph,
    build_distribution_graph,
    build_kpi_card,
    build_map_graph,
)
from data import PreparedData


def _missing_data_figure(prepared: PreparedData):
    customers = prepared.customers
    missing = customers.isna().mean().reset_index()
    missing.columns = ["field", "missing_pct"]
    missing = missing[missing["missing_pct"] > 0]
    if missing.empty:
        missing_fig = px.bar(title="No missing values detected")
    else:
        missing_fig = px.bar(
            missing,
            x="field",
            y="missing_pct",
            title="Missing data ratio",
            labels={"missing_pct": "Missing %"},
        )
        missing_fig.update_layout(xaxis_tickangle=-45)
    return missing_fig


def _initial_distribution(prepared: PreparedData):
    if "customer_lifetime_value" in prepared.customer_metrics.columns:
        column = "customer_lifetime_value"
    else:
        column = prepared.customer_metrics.select_dtypes("number").columns[0]
    return build_distribution_graph(
        "overview-distribution",
        prepared.customer_metrics,
        column=column,
        chart_type="histogram",
        bins=30,
    )


def _initial_composition(prepared: PreparedData):
    segment = prepared.segment_breakdowns.get("loyalty_status")
    if segment is None or segment.empty:
        return dcc.Graph(id="marketing-composition")
    fig = px.bar(
        segment,
        x="loyalty_status",
        y="customers",
        color="loyalty_status",
        title="Customer composition by loyalty status",
        text_auto=True,
    )
    fig.update_layout(showlegend=False)
    return dcc.Graph(id="marketing-composition", figure=fig)


def _initial_cohort(prepared: PreparedData):
    cohort = prepared.segment_breakdowns.get("cohort")
    if cohort is None or cohort.empty:
        return dcc.Graph(id="marketing-cohort")
    fig = px.bar(
        cohort,
        x="enrollment_year",
        y="customers",
        color="loyalty_status",
        barmode="stack",
        title="Cohort comparison",
    )
    return dcc.Graph(id="marketing-cohort", figure=fig)


def _initial_trend(prepared: PreparedData):
    enrollments = prepared.time_series.get("enrollments")
    if enrollments is None or enrollments.empty:
        return dcc.Graph(id="time-trend-enrollments")
    fig = px.line(
        enrollments,
        x="month",
        y="customers",
        title="Enrollments over time",
    )
    fig.update_traces(mode="markers+lines")
    return dcc.Graph(id="time-trend-enrollments", figure=fig)


def _initial_engagement(prepared: PreparedData):
    engagement = prepared.time_series.get("engagement")
    if engagement is None or engagement.empty:
        return dcc.Graph(id="time-trend-engagement")
    fig = px.line(
        engagement,
        x="year_month",
        y="total_flights",
        title="Flight activity over time",
    )
    fig.update_traces(mode="markers+lines")
    return dcc.Graph(id="time-trend-engagement", figure=fig)


def create_layout(prepared: PreparedData) -> html.Div:
    stores = prepared.to_store_payload()
    customers = prepared.customers

    enrollment_min = customers["enrollment_date"].min()
    enrollment_max = customers["enrollment_date"].max()

    loyalty_options = [
        {"label": status, "value": status}
        for status in sorted(customers["loyalty_status"].dropna().unique())
    ]
    province_options = [
        {"label": province, "value": province}
        for province in sorted(customers["province"].dropna().unique())
    ]

    tenure_buckets = customers["tenure_bucket"].cat.categories if "tenure_bucket" in customers else []
    tenure_options = [
        {"label": str(bucket), "value": str(bucket)} for bucket in tenure_buckets
    ]

    numeric_columns = [
        {"label": col.replace("_", " ").title(), "value": col}
        for col in prepared.customer_metrics.select_dtypes("number").columns
    ]

    category_columns = [
        {"label": col.replace("_", " ").title(), "value": col}
        for col in prepared.customer_metrics.select_dtypes(exclude="number").columns
        if col not in {"customer_name", "first_name", "last_name"}
    ]

    layout = html.Div(
        [
            dcc.Store(id="store-customers", data=stores["customers"]),
            dcc.Store(id="store-flights", data=stores["flights"]),
            dcc.Store(id="store-metrics", data=stores["customer_metrics"]),
            dcc.Store(id="store-time-series", data=stores["time_series"]),
            dcc.Store(id="store-segments", data=stores["segments"]),
            dcc.Store(id="store-correlations", data=stores["correlations"]),
            dcc.Store(id="store-map", data=stores["map_data"]),

            html.H1("Loyalty Marketing Intelligence"),

            html.Div(
                [
                    build_kpi_card("kpi-customer-count", "Customers"),
                    build_kpi_card("kpi-avg-ltv", "Avg Lifetime Value"),
                    build_kpi_card("kpi-redemption", "Redemption Ratio"),
                    build_kpi_card("kpi-flights", "Avg Flights / Month"),
                ],
                className="kpi-strip",
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Date range"),
                            dcc.DatePickerRange(
                                id="filter-date-range",
                                min_date_allowed=enrollment_min,
                                max_date_allowed=enrollment_max,
                                start_date=enrollment_min,
                                end_date=enrollment_max,
                            ),
                        ],
                        className="filter-item",
                    ),
                    html.Div(
                        [
                            html.Label("Loyalty status"),
                            dcc.Dropdown(
                                id="filter-loyalty-status",
                                options=loyalty_options,
                                multi=True,
                                placeholder="Select status",
                            ),
                        ],
                        className="filter-item",
                    ),
                    html.Div(
                        [
                            html.Label("Province"),
                            dcc.Dropdown(
                                id="filter-province",
                                options=province_options,
                                multi=True,
                                placeholder="Select province",
                            ),
                        ],
                        className="filter-item",
                    ),
                    html.Div(
                        [
                            html.Label("Tenure bucket"),
                            dcc.Dropdown(
                                id="filter-tenure",
                                options=tenure_options,
                                multi=True,
                                placeholder="Select tenure",
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filter-row",
            ),

            html.Section(
                [
                    html.H2("Data Quality & Overview"),
                    dcc.Graph(id="data-quality-missing", figure=_missing_data_figure(prepared)),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Measure"),
                                    dcc.Dropdown(
                                        id="distribution-field",
                                        options=numeric_columns,
                                        value=(numeric_columns[0]["value"] if numeric_columns else None),
                                    ),
                                ],
                                className="control-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Chart type"),
                                    dcc.RadioItems(
                                        id="distribution-type",
                                        options=[
                                            {"label": "Histogram", "value": "histogram"},
                                            {"label": "Box plot", "value": "box"},
                                        ],
                                        value="histogram",
                                        inline=True,
                                    ),
                                ],
                                className="control-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Bins"),
                                    dcc.Slider(
                                        id="distribution-bins",
                                        min=10,
                                        max=80,
                                        step=5,
                                        value=30,
                                    ),
                                ],
                                className="control-item",
                            ),
                        ],
                        className="control-row",
                    ),
                    _initial_distribution(prepared),
                ]
            ),

            html.Section(
                [
                    html.H2("Relationships"),
                    html.Div(
                        [
                            html.Label("Correlation metric"),
                            dcc.RadioItems(
                                id="correlation-method",
                                options=[
                                    {"label": "Pearson", "value": "pearson"},
                                    {"label": "Spearman", "value": "spearman"},
                                ],
                                value="pearson",
                                inline=True,
                            ),
                        ]
                    ),
                    build_correlation_graph(
                        "relationships-correlation",
                        prepared.correlations.get("pearson", prepared.customer_metrics.corr()),
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Category"),
                                    dcc.Dropdown(
                                        id="relationship-category",
                                        options=category_columns,
                                        value=(
                                            category_columns[0]["value"]
                                            if category_columns
                                            else None
                                        ),
                                    ),
                                ],
                                className="control-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Measure"),
                                    dcc.Dropdown(
                                        id="relationship-measure",
                                        options=numeric_columns,
                                        value=(numeric_columns[0]["value"] if numeric_columns else None),
                                    ),
                                ],
                                className="control-item",
                            ),
                        ],
                        className="control-row",
                    ),
                    dcc.Graph(id="relationships-boxplot"),
                ]
            ),

            html.Section(
                [
                    html.H2("Marketing Insights"),
                    _initial_composition(prepared),
                    _initial_cohort(prepared),
                ]
            ),

            html.Section(
                [
                    html.H2("Time & Geography"),
                    _initial_trend(prepared),
                    _initial_engagement(prepared),
                    build_map_graph("geo-map", prepared.map_data),
                ]
            ),
        ],
        className="app-container",
    )

    return layout

