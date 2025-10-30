"""Dash callbacks powering the interactive dashboard."""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from .data import build_correlation_frame, compute_summary, filter_data, records_to_dataframe


KPI_CARD_ORDER: Sequence[tuple[str, str]] = (
    ("revenue", "Total Revenue"),
    ("revenue_growth", "YoY Revenue Growth"),
    ("profit", "Net Profit"),
    ("profit_margin", "Profit Margin"),
    ("avg_satisfaction", "Avg. Customer Satisfaction"),
    ("avg_engagement", "Avg. Employee Engagement"),
    ("new_customers", "New Customers"),
    ("avg_churn", "Mean Churn Rate"),
    ("tickets", "Support Tickets"),
)


def register_callbacks(app) -> None:
    """Register all Dash callbacks."""

    @app.callback(
        Output("kpi-cards", "children"),
        Output("revenue-trend", "figure"),
        Output("yoy-revenue", "figure"),
        Output("department-performance", "figure"),
        Output("regional-performance", "figure"),
        Output("satisfaction-engagement", "figure"),
        Output("churn-customers", "figure"),
        Output("support-trend", "figure"),
        Output("revenue-per-customer", "figure"),
        Output("training-initiatives", "figure"),
        Output("correlation-heatmap", "figure"),
        Output("data-table", "data"),
        Input("store-data", "data"),
        Input("year-filter", "value"),
        Input("department-filter", "value"),
        Input("region-filter", "value"),
        Input("quarter-filter", "value"),
        Input("month-range", "value"),
    )
    def update_dashboard(
        records: list[dict],
        years: Iterable[int] | None,
        departments: Iterable[str] | None,
        regions: Iterable[str] | None,
        quarters: Iterable[str] | None,
        months: Sequence[int] | None,
    ):
        """Refresh all dashboard widgets based on the filter state."""

        df = records_to_dataframe(records)
        filtered = filter_data(
            df,
            _coerce_iterable(years),
            _coerce_iterable(departments),
            _coerce_iterable(regions),
            _coerce_iterable(quarters),
            _coerce_range(months),
        )

        summary = compute_summary(filtered)
        kpi_cards = _render_kpi_cards(summary)
        revenue_fig = _build_revenue_trend(filtered)
        yoy_fig = _build_yoy_revenue_chart(filtered)
        department_fig = _build_department_chart(filtered)
        regional_fig = _build_region_chart(filtered)
        satisfaction_fig = _build_satisfaction_scatter(filtered)
        churn_fig = _build_churn_growth_chart(filtered)
        support_fig = _build_support_trend(filtered)
        revenue_per_customer_fig = _build_revenue_per_customer(filtered)
        training_fig = _build_training_initiatives(filtered)
        correlation_fig = _build_correlation_heatmap(filtered)
        table_data = _build_table_data(filtered)

        return (
            kpi_cards,
            revenue_fig,
            yoy_fig,
            department_fig,
            regional_fig,
            satisfaction_fig,
            churn_fig,
            support_fig,
            revenue_per_customer_fig,
            training_fig,
            correlation_fig,
            table_data,
        )

    @app.callback(
        Output("download-dataframe", "data"),
        Input("download-button", "n_clicks"),
        State("store-data", "data"),
        State("year-filter", "value"),
        State("department-filter", "value"),
        State("region-filter", "value"),
        State("quarter-filter", "value"),
        State("month-range", "value"),
        prevent_initial_call=True,
    )
    def download_filtered_dataset(
        n_clicks: int | None,
        records: list[dict] | None,
        years: Iterable[int] | None,
        departments: Iterable[str] | None,
        regions: Iterable[str] | None,
        quarters: Iterable[str] | None,
        months: Sequence[int] | None,
    ):
        if not n_clicks or not records:
            raise PreventUpdate

        df = records_to_dataframe(records)
        filtered = filter_data(
            df,
            _coerce_iterable(years),
            _coerce_iterable(departments),
            _coerce_iterable(regions),
            _coerce_iterable(quarters),
            _coerce_range(months),
        )

        if filtered.empty:
            empty = df.head(0).copy()
            if "Date" in empty.columns:
                empty["Date"] = pd.to_datetime(empty["Date"])
            return dcc.send_data_frame(empty.to_csv, "dashboard-export.csv", index=False)

        export_df = filtered.sort_values("Date").copy()
        export_df["Date"] = export_df["Date"].dt.strftime("%Y-%m-%d")
        export_df["ProfitMargin"] = (export_df.get("ProfitMargin", 0) * 100).round(1)
        export_df["ChurnRate"] = (export_df.get("ChurnRate", 0) * 100).round(2)
        export_df["RevenuePerCustomer"] = export_df.get("RevenuePerCustomer", 0).round(2)

        return dcc.send_data_frame(export_df.to_csv, "dashboard-export.csv", index=False)


def _coerce_iterable(values) -> list:
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        return list(values)
    return [values]


def _coerce_range(values: Sequence[int] | None) -> tuple[int, int] | None:
    if not values:
        return None
    if len(values) != 2:
        return None
    start, end = values
    if start is None or end is None:
        return None
    if start > end:
        start, end = end, start
    return int(start), int(end)


def _render_kpi_cards(summary: dict[str, float]) -> list:
    cards = []
    for key, label in KPI_CARD_ORDER:
        value = summary.get(key, 0)
        formatted = _format_metric(key, value)
        delta = _metric_hint(key, value)
        cards.append(
            html.Div(
                className="kpi-card",
                children=[
                    html.Span(label, className="kpi-label"),
                    html.Span(formatted, className="kpi-value"),
                    html.Span(delta, className="kpi-hint"),
                ],
            )
        )
    return cards


def _format_metric(metric: str, value: float | int) -> str:
    if metric in {"revenue", "profit"}:
        return f"${value:,.0f}"
    if metric in {"avg_satisfaction", "avg_engagement"}:
        return f"{value:,.1f}" if value else "0.0"
    if metric in {"avg_churn", "profit_margin", "revenue_growth"}:
        return f"{value * 100:,.1f}%"
    if metric == "tickets":
        return f"{value:,d}" if value else "0"
    if metric == "new_customers":
        return f"{int(value):,d}" if value else "0"
    return str(value)


def _metric_hint(metric: str, value: float | int) -> str:
    if metric in {"revenue", "profit"}:
        return "vs. plan"
    if metric == "profit_margin":
        return "margin"
    if metric in {"avg_satisfaction", "avg_engagement"}:
        return "avg score"
    if metric == "revenue_growth":
        return "vs. prior year"
    if metric == "new_customers":
        return "acquired"
    if metric == "avg_churn":
        return "mean churn"
    if metric == "tickets":
        return "tickets logged"
    return ""


def _build_revenue_trend(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No data for selected filters")

    series = df.groupby("Date")[["Revenue", "Expenses", "Profit"]].sum().sort_index()
    fig = go.Figure()
    for column, color in zip(series.columns, ["#2563eb", "#f97316", "#16a34a"]):
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series[column],
                mode="lines+markers",
                name=column,
                line=dict(color=color, width=3),
            )
        )
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="USD",
        xaxis_title="Month",
        template="plotly_white",
    )
    return fig


def _build_department_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No data for selected filters")

    grouped = (
        df.groupby("Department")
        .agg({"Revenue": "sum", "Profit": "sum"})
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    grouped["ProfitMargin"] = grouped["Profit"] / grouped["Revenue"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["Department"],
            y=grouped["Revenue"],
            name="Revenue",
            marker_color="#2563eb",
        )
    )
    fig.add_trace(
        go.Bar(
            x=grouped["Department"],
            y=grouped["Profit"],
            name="Profit",
            marker_color="#16a34a",
        )
    )
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title="USD",
    )
    return fig


def _build_yoy_revenue_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No data for selected filters")

    yearly = (
        df.groupby("Year")["Revenue"].sum().sort_index()
    )
    if yearly.empty:
        return _empty_figure("No data for selected filters")

    data = yearly.reset_index()
    data["YoYGrowth"] = data["Revenue"].pct_change()

    # Prepare readable labels for the bars
    growth_labels: list[str] = []
    for pct in data["YoYGrowth"]:
        if pd.isna(pct):
            growth_labels.append("Base Year")
        else:
            growth_labels.append(f"{pct * 100:,.1f}% YoY")
    data["YoYGrowth"] = data["YoYGrowth"].fillna(0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=data["Year"].astype(str),
            y=data["Revenue"],
            name="Revenue",
            marker_color="#2563eb",
            text=growth_labels,
            textposition="outside",
        )
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=30),
        yaxis_title="Total Revenue",
        xaxis_title="Fiscal Year",
        uniformtext=dict(mode="hide", minsize=12),
    )
    return fig


def _build_region_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No data for selected filters")

    grouped = df.groupby(["Region", "Department"]).agg({"Revenue": "sum", "Profit": "sum"}).reset_index()
    fig = px.treemap(
        grouped,
        path=["Region", "Department"],
        values="Revenue",
        color="Profit",
        color_continuous_scale="Blues",
        hover_data={"Profit": ":,.0f"},
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    return fig


def _build_satisfaction_scatter(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No data for selected filters")

    fig = px.scatter(
        df,
        x="CustomerSatisfaction",
        y="EmployeeEngagement",
        size="NewCustomers",
        color="Department",
        hover_data={"Region": True, "Revenue": ":,.0f", "Profit": ":,.0f"},
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Customer Satisfaction",
        yaxis_title="Employee Engagement",
    )
    return fig


def _build_churn_growth_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No data for selected filters")

    grouped = (
        df.groupby(["Date", "Department"])
        .agg({"NewCustomers": "sum", "ChurnRate": "mean"})
        .reset_index()
    )
    fig = px.area(
        grouped,
        x="Date",
        y="NewCustomers",
        color="Department",
        line_group="Department",
        hover_data={"ChurnRate": ":.2%"},
    )
    fig.update_traces(mode="lines")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title="New Customers",
    )
    return fig


def _build_support_trend(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No data for selected filters")

    series = df.groupby("Date")["SupportTickets"].sum().reset_index()
    fig = px.bar(series, x="Date", y="SupportTickets", color="SupportTickets", color_continuous_scale="Purples")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title="Tickets",
        coloraxis_showscale=False,
    )
    return fig


def _build_revenue_per_customer(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No data for selected filters")

    grouped = (
        df.groupby(["Department", "Region"])["RevenuePerCustomer"].mean().reset_index()
    )
    fig = px.bar(
        grouped,
        x="Department",
        y="RevenuePerCustomer",
        color="Region",
        barmode="group",
        labels={"RevenuePerCustomer": "Revenue per Customer"},
        hover_data={"RevenuePerCustomer": ":,.0f"},
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title="USD",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_training_initiatives(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("No data for selected filters")

    grouped = (
        df.groupby("Department")
        .agg({"TrainingHours": "mean", "StrategicInitiatives": "sum"})
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["Department"],
            y=grouped["StrategicInitiatives"],
            name="Strategic Initiatives",
            marker_color="#f97316",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["Department"],
            y=grouped["TrainingHours"],
            name="Avg Training Hours",
            mode="lines+markers",
            marker=dict(color="#2563eb", size=10),
            line=dict(width=3),
            yaxis="y2",
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(title="Initiatives"),
        yaxis2=dict(title="Avg Training Hours", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = build_correlation_frame(df)
    if corr.empty:
        return _empty_figure("Not enough data for correlation")

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            reversescale=True,
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=60, r=20, t=30, b=60),
        xaxis=dict(tickangle=-45),
    )
    return fig


def _build_table_data(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    columns = [
        "Date",
        "Department",
        "Region",
        "Revenue",
        "Expenses",
        "Profit",
        "NewCustomers",
        "ChurnRate",
        "CustomerSatisfaction",
        "EmployeeEngagement",
        "SupportTickets",
        "ProfitMargin",
        "RevenuePerCustomer",
    ]
    table_df = df[columns].copy()
    table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m-%d")
    table_df["Revenue"] = table_df["Revenue"].round(2)
    table_df["Expenses"] = table_df["Expenses"].round(2)
    table_df["Profit"] = table_df["Profit"].round(2)
    table_df["ChurnRate"] = (table_df["ChurnRate"] * 100).round(1)
    if "ProfitMargin" in table_df.columns:
        table_df["ProfitMargin"] = (table_df["ProfitMargin"] * 100).round(1)
    if "RevenuePerCustomer" in table_df.columns:
        table_df["RevenuePerCustomer"] = table_df["RevenuePerCustomer"].round(2)
    table_df = table_df.sort_values("Date", ascending=False)
    return table_df.to_dict("records")


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig
