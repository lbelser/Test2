"""Layout components for the management dashboard."""
from __future__ import annotations

import calendar
from typing import Any

from dash import dcc, html
from dash.dash_table import DataTable


def create_layout(dataframe, filter_options: dict[str, list[Any]]):
    """Return the Dash layout for the application."""
    data_records = dataframe.to_dict("records")

    year_options = [{"label": str(year), "value": year} for year in filter_options.get("years", [])]
    department_options = [
        {"label": department, "value": department} for department in filter_options.get("departments", [])
    ]
    region_options = [{"label": region, "value": region} for region in filter_options.get("regions", [])]
    quarter_options = [{"label": quarter, "value": quarter} for quarter in filter_options.get("quarters", [])]
    month_values = filter_options.get("months", [])
    if month_values:
        min_month, max_month = month_values[0], month_values[-1]
    else:
        min_month, max_month = 1, 12
    month_marks = {month: calendar.month_abbr[month] for month in month_values} if month_values else {}

    return html.Div(
        className="dashboard-container",
        children=[
            dcc.Store(id="store-data", data=data_records),
            html.Header(
                className="dashboard-header",
                children=[
                    html.Div(
                        [
                            html.H1("Executive Performance Dashboard"),
                            html.P(
                                "Interactive analytics for revenue, customer health, and operational efficiency. "
                                "Use the controls below to drill into the segments your leadership team cares about."
                            ),
                        ]
                    ),
                ],
            ),
            html.Section(
                className="control-panel",
                children=[
                    html.Div(
                        className="control-group",
                        children=[
                            html.Label("Year"),
                            dcc.Dropdown(
                                id="year-filter",
                                options=year_options,
                                value=[option["value"] for option in year_options],
                                multi=True,
                                placeholder="Select year(s)",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="control-group",
                        children=[
                            html.Label("Department"),
                            dcc.Dropdown(
                                id="department-filter",
                                options=department_options,
                                value=[option["value"] for option in department_options],
                                multi=True,
                                placeholder="Select departments",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="control-group",
                        children=[
                            html.Label("Region"),
                            dcc.Dropdown(
                                id="region-filter",
                                options=region_options,
                                value=[option["value"] for option in region_options],
                                multi=True,
                                placeholder="Select regions",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="control-group",
                        children=[
                            html.Label("Quarter"),
                            dcc.Dropdown(
                                id="quarter-filter",
                                options=quarter_options,
                                value=[option["value"] for option in quarter_options],
                                multi=True,
                                placeholder="Select quarters",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        className="control-group",
                        children=[
                            html.Label("Month Range"),
                            dcc.RangeSlider(
                                id="month-range",
                                min=min_month,
                                max=max_month,
                                step=1,
                                value=[min_month, max_month],
                                allowCross=False,
                                marks=month_marks,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                    ),
                    html.Div(
                        className="control-group",
                        children=[
                            html.Label("Export"),
                            html.Button(
                                "Download filtered CSV",
                                id="download-button",
                                className="download-button",
                            ),
                            dcc.Download(id="download-dataframe"),
                        ],
                    ),
                ],
            ),
            html.Section(id="kpi-cards", className="kpi-grid"),
            dcc.Tabs(
                className="dashboard-tabs",
                value="overview",
                children=[
                    dcc.Tab(
                        label="Revenue & Profit",
                        value="overview",
                        children=[
                            html.Div(
                                className="chart-grid",
                                children=[
                                    html.Div(
                                        className="chart-card span-2",
                                        children=[
                                            html.H3("Financial Trajectory"),
                                            dcc.Graph(id="revenue-trend"),
                                        ],
                                    ),
                                    html.Div(
                                        className="chart-card",
                                        children=[
                                            html.H3("Year-over-Year Growth"),
                                            dcc.Graph(id="yoy-revenue"),
                                        ],
                                    ),
                                    html.Div(
                                        className="chart-card",
                                        children=[
                                            html.H3("Department Contribution"),
                                            dcc.Graph(id="department-performance"),
                                        ],
                                    ),
                                    html.Div(
                                        className="chart-card",
                                        children=[
                                            html.H3("Regional Mix"),
                                            dcc.Graph(id="regional-performance"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Customer & Workforce",
                        value="people",
                        children=[
                            html.Div(
                                className="chart-grid",
                                children=[
                                    html.Div(
                                        className="chart-card",
                                        children=[
                                            html.H3("Satisfaction vs. Engagement"),
                                            dcc.Graph(id="satisfaction-engagement"),
                                        ],
                                    ),
                                    html.Div(
                                        className="chart-card",
                                        children=[
                                            html.H3("Retention & Growth"),
                                            dcc.Graph(id="churn-customers"),
                                        ],
                                    ),
                                    html.Div(
                                        className="chart-card",
                                        children=[
                                            html.H3("Revenue per Customer"),
                                            dcc.Graph(id="revenue-per-customer"),
                                        ],
                                    ),
                                    html.Div(
                                        className="chart-card span-2",
                                        children=[
                                            html.H3("Recent Activity"),
                                            DataTable(
                                                id="data-table",
                                                page_size=12,
                                                style_table={"overflowX": "auto"},
                                                style_cell={
                                                    "textAlign": "center",
                                                    "fontFamily": "'Inter', 'Segoe UI', sans-serif",
                                                },
                                                style_header={
                                                    "fontWeight": "600",
                                                    "backgroundColor": "#f3f4f6",
                                                },
                                                columns=[
                                                    {"name": "Date", "id": "Date"},
                                                    {"name": "Department", "id": "Department"},
                                                    {"name": "Region", "id": "Region"},
                                                    {"name": "Revenue", "id": "Revenue", "type": "numeric"},
                                                    {"name": "Expenses", "id": "Expenses", "type": "numeric"},
                                                    {"name": "Profit", "id": "Profit", "type": "numeric"},
                                                    {"name": "New Customers", "id": "NewCustomers"},
                                                    {"name": "Churn %", "id": "ChurnRate"},
                                                    {"name": "CSAT", "id": "CustomerSatisfaction"},
                                                    {"name": "Engagement", "id": "EmployeeEngagement"},
                                                    {"name": "Tickets", "id": "SupportTickets"},
                                                    {
                                                        "name": "Profit Margin %",
                                                        "id": "ProfitMargin",
                                                        "type": "numeric",
                                                    },
                                                    {
                                                        "name": "Revenue per Customer",
                                                        "id": "RevenuePerCustomer",
                                                        "type": "numeric",
                                                    },
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Operational Insights",
                        value="operations",
                        children=[
                            html.Div(
                                className="chart-grid",
                                children=[
                                    html.Div(
                                        className="chart-card",
                                        children=[
                                            html.H3("Support Load Over Time"),
                                            dcc.Graph(id="support-trend"),
                                        ],
                                    ),
                                    html.Div(
                                        className="chart-card",
                                        children=[
                                            html.H3("Training vs. Initiatives"),
                                            dcc.Graph(id="training-initiatives"),
                                        ],
                                    ),
                                    html.Div(
                                        className="chart-card span-2",
                                        children=[
                                            html.H3("KPI Correlations"),
                                            dcc.Graph(id="correlation-heatmap"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
