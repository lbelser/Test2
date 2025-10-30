from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output

from components import (
    build_correlation_graph,
    build_distribution_graph,
    build_map_graph,
)
from data import frame_from_json


def _parse_multi(value: Optional[List[str]]) -> List[str]:
    if not value:
        return []
    return value if isinstance(value, list) else [value]


def _filter_frames(
    customers_json: str,
    flights_json: str,
    start_date: Optional[str],
    end_date: Optional[str],
    loyalty: Optional[List[str]],
    provinces: Optional[List[str]],
    tenures: Optional[List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    customers = frame_from_json(customers_json)
    flights = frame_from_json(flights_json)

    if customers.empty:
        return customers, flights

    loyalty = _parse_multi(loyalty)
    provinces = _parse_multi(provinces)
    tenures = _parse_multi(tenures)

    if start_date:
        start = pd.to_datetime(start_date)
        customers = customers[customers["enrollment_date"] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        customers = customers[customers["enrollment_date"] <= end]

    if loyalty:
        customers = customers[customers["loyalty_status"].isin(loyalty)]
    if provinces:
        customers = customers[customers["province"].isin(provinces)]
    if tenures:
        customers = customers[
            customers["tenure_bucket"].astype(str).isin([str(t) for t in tenures])
        ]

    if flights.empty:
        return customers, flights

    flights = flights[flights["loyalty_id"].isin(customers["loyalty_id"])]

    if start_date:
        start = pd.to_datetime(start_date)
        flights = flights[flights["year_month_date"] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        flights = flights[flights["year_month_date"] <= end]

    return customers, flights


def _build_metrics(customers: pd.DataFrame, flights: pd.DataFrame) -> pd.DataFrame:
    if customers.empty:
        return customers.copy()

    if flights.empty:
        metrics = customers.copy()
        metrics = metrics.assign(
            total_flights=0,
            total_companion_flights=0,
            total_distance_km=0,
            total_points_accumulated=0,
            total_points_redeemed=0,
            total_dollar_cost_points=0,
            active_months=0,
            avg_flights_per_month=0,
            redemption_ratio=0,
        )
        return metrics

    grouped = (
        flights.assign(year_month=lambda df: df["year_month_date"].dt.to_period("M"))
        .groupby("loyalty_id")
        .agg(
            total_flights=("num_flights", "sum"),
            total_companion_flights=("num_flights_with_companions", "sum"),
            total_distance_km=("distance_km", "sum"),
            total_points_accumulated=("points_accumulated", "sum"),
            total_points_redeemed=("points_redeemed", "sum"),
            total_dollar_cost_points=("dollar_cost_points_redeemed", "sum"),
            active_months=("year_month", "nunique"),
        )
        .reset_index()
    )

    grouped["avg_flights_per_month"] = np.where(
        grouped["active_months"] > 0,
        grouped["total_flights"] / grouped["active_months"],
        0,
    )
    grouped["redemption_ratio"] = np.where(
        grouped["total_points_accumulated"] > 0,
        grouped["total_points_redeemed"] / grouped["total_points_accumulated"],
        0,
    )

    metrics = customers.merge(grouped, on="loyalty_id", how="left")
    fill_values = {
        "total_flights": 0,
        "total_companion_flights": 0,
        "total_distance_km": 0,
        "total_points_accumulated": 0,
        "total_points_redeemed": 0,
        "total_dollar_cost_points": 0,
        "active_months": 0,
        "avg_flights_per_month": 0,
        "redemption_ratio": 0,
    }
    metrics = metrics.fillna(fill_values)
    return metrics


def register_callbacks(app):
    @app.callback(
        Output("kpi-customer-count", "children"),
        Output("kpi-avg-ltv", "children"),
        Output("kpi-redemption", "children"),
        Output("kpi-flights", "children"),
        Input("store-customers", "data"),
        Input("store-flights", "data"),
        Input("filter-date-range", "start_date"),
        Input("filter-date-range", "end_date"),
        Input("filter-loyalty-status", "value"),
        Input("filter-province", "value"),
        Input("filter-tenure", "value"),
    )
    def update_kpis(customers_json, flights_json, start_date, end_date, loyalty, province, tenure):
        customers, flights = _filter_frames(
            customers_json,
            flights_json,
            start_date,
            end_date,
            loyalty,
            province,
            tenure,
        )

        metrics = _build_metrics(customers, flights)

        customer_count = len(metrics)
        avg_ltv = metrics["customer_lifetime_value"].mean() if not metrics.empty else 0
        total_points_redeemed = flights["points_redeemed"].sum()
        total_points_accumulated = flights["points_accumulated"].sum()
        redemption_ratio = (
            total_points_redeemed / total_points_accumulated
            if total_points_accumulated > 0
            else 0
        )
        avg_flights_per_month = (
            metrics["avg_flights_per_month"].mean() if not metrics.empty else 0
        )

        return (
            f"{customer_count:,}",
            f"${avg_ltv:,.0f}",
            f"{redemption_ratio:.1%}",
            f"{avg_flights_per_month:.2f}",
        )

    @app.callback(
        Output("data-quality-missing", "figure"),
        Input("store-customers", "data"),
        Input("filter-date-range", "start_date"),
        Input("filter-date-range", "end_date"),
        Input("filter-loyalty-status", "value"),
        Input("filter-province", "value"),
        Input("filter-tenure", "value"),
    )
    def update_missing(customers_json, start_date, end_date, loyalty, province, tenure):
        customers, _ = _filter_frames(
            customers_json,
            "[]",
            start_date,
            end_date,
            loyalty,
            province,
            tenure,
        )

        if customers.empty:
            return px.bar(title="No data available")

        missing = customers.isna().mean().reset_index()
        missing.columns = ["field", "missing_pct"]
        missing = missing[missing["missing_pct"] > 0]
        if missing.empty:
            return px.bar(title="No missing values detected")
        fig = px.bar(
            missing,
            x="field",
            y="missing_pct",
            labels={"missing_pct": "Missing %"},
            title="Missing data ratio",
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    @app.callback(
        Output("overview-distribution", "figure"),
        Input("store-customers", "data"),
        Input("store-flights", "data"),
        Input("filter-date-range", "start_date"),
        Input("filter-date-range", "end_date"),
        Input("filter-loyalty-status", "value"),
        Input("filter-province", "value"),
        Input("filter-tenure", "value"),
        Input("distribution-field", "value"),
        Input("distribution-type", "value"),
        Input("distribution-bins", "value"),
    )
    def update_distribution(
        customers_json,
        flights_json,
        start_date,
        end_date,
        loyalty,
        province,
        tenure,
        measure,
        chart_type,
        bins,
    ):
        customers, flights = _filter_frames(
            customers_json,
            flights_json,
            start_date,
            end_date,
            loyalty,
            province,
            tenure,
        )
        metrics = _build_metrics(customers, flights)
        if not measure or metrics.empty or measure not in metrics.columns:
            return px.histogram(title="No data available")

        graph = build_distribution_graph(
            "overview-distribution",
            metrics,
            column=measure,
            chart_type=chart_type,
            bins=bins,
        )
        return graph.figure

    @app.callback(
        Output("relationships-correlation", "figure"),
        Input("store-customers", "data"),
        Input("store-flights", "data"),
        Input("filter-date-range", "start_date"),
        Input("filter-date-range", "end_date"),
        Input("filter-loyalty-status", "value"),
        Input("filter-province", "value"),
        Input("filter-tenure", "value"),
        Input("correlation-method", "value"),
    )
    def update_correlation(
        customers_json,
        flights_json,
        start_date,
        end_date,
        loyalty,
        province,
        tenure,
        method,
    ):
        customers, flights = _filter_frames(
            customers_json,
            flights_json,
            start_date,
            end_date,
            loyalty,
            province,
            tenure,
        )
        metrics = _build_metrics(customers, flights)
        numeric = metrics.select_dtypes(include=[np.number])
        if numeric.empty:
            return build_correlation_graph("relationships-correlation", pd.DataFrame()).figure

        corr = numeric.corr(method=method or "pearson")
        return build_correlation_graph("relationships-correlation", corr).figure

    @app.callback(
        Output("relationships-boxplot", "figure"),
        Input("store-customers", "data"),
        Input("store-flights", "data"),
        Input("filter-date-range", "start_date"),
        Input("filter-date-range", "end_date"),
        Input("filter-loyalty-status", "value"),
        Input("filter-province", "value"),
        Input("filter-tenure", "value"),
        Input("relationship-category", "value"),
        Input("relationship-measure", "value"),
    )
    def update_boxplot(
        customers_json,
        flights_json,
        start_date,
        end_date,
        loyalty,
        province,
        tenure,
        category,
        measure,
    ):
        customers, flights = _filter_frames(
            customers_json,
            flights_json,
            start_date,
            end_date,
            loyalty,
            province,
            tenure,
        )
        metrics = _build_metrics(customers, flights)
        if metrics.empty or not measure or measure not in metrics.columns:
            return px.box(title="No data available")

        if category and category in metrics.columns:
            filtered = metrics.dropna(subset=[category, measure])
            if filtered.empty:
                return px.box(title="No data available")
            fig = px.box(
                filtered,
                x=category,
                y=measure,
                color=category,
                points="suspectedoutliers",
            )
        else:
            fig = px.box(metrics, y=measure, points="suspectedoutliers")
        fig.update_layout(title="Category vs measure distribution")
        return fig

    @app.callback(
        Output("marketing-composition", "figure"),
        Output("marketing-cohort", "figure"),
        Input("store-customers", "data"),
        Input("store-flights", "data"),
        Input("filter-date-range", "start_date"),
        Input("filter-date-range", "end_date"),
        Input("filter-loyalty-status", "value"),
        Input("filter-province", "value"),
        Input("filter-tenure", "value"),
    )
    def update_marketing(
        customers_json,
        flights_json,
        start_date,
        end_date,
        loyalty,
        province,
        tenure,
    ):
        customers, flights = _filter_frames(
            customers_json,
            flights_json,
            start_date,
            end_date,
            loyalty,
            province,
            tenure,
        )
        metrics = _build_metrics(customers, flights)

        if metrics.empty:
            return px.bar(title="No data"), px.bar(title="No data")

        composition = (
            metrics.groupby("loyalty_status")
            .agg(customers=("loyalty_id", "nunique"))
            .reset_index()
        )
        comp_fig = px.bar(
            composition,
            x="loyalty_status",
            y="customers",
            color="loyalty_status",
            text_auto=True,
            title="Customer composition by loyalty status",
        )
        comp_fig.update_layout(showlegend=False)

        if "enrollment_date" in metrics.columns:
            cohort = (
                metrics.dropna(subset=["enrollment_date"])
                .assign(enrollment_year=lambda df: df["enrollment_date"].dt.to_period("Y"))
                .groupby(["enrollment_year", "loyalty_status"])
                .agg(customers=("loyalty_id", "nunique"))
                .reset_index()
            )
            cohort["enrollment_year"] = cohort["enrollment_year"].astype(str)
            cohort_fig = px.bar(
                cohort,
                x="enrollment_year",
                y="customers",
                color="loyalty_status",
                barmode="stack",
                title="Cohort comparison",
            )
        else:
            cohort_fig = px.bar(title="No cohort data")

        return comp_fig, cohort_fig

    @app.callback(
        Output("time-trend-enrollments", "figure"),
        Output("time-trend-engagement", "figure"),
        Output("geo-map", "figure"),
        Input("store-customers", "data"),
        Input("store-flights", "data"),
        Input("filter-date-range", "start_date"),
        Input("filter-date-range", "end_date"),
        Input("filter-loyalty-status", "value"),
        Input("filter-province", "value"),
        Input("filter-tenure", "value"),
    )
    def update_time_and_geo(
        customers_json,
        flights_json,
        start_date,
        end_date,
        loyalty,
        province,
        tenure,
    ):
        customers, flights = _filter_frames(
            customers_json,
            flights_json,
            start_date,
            end_date,
            loyalty,
            province,
            tenure,
        )

        enroll_fig = px.line(title="Enrollments over time")
        if not customers.empty:
            enrollments = (
                customers.assign(month=lambda df: df["enrollment_date"].dt.to_period("M"))
                .dropna(subset=["month"])
                .groupby("month")
                .agg(customers=("loyalty_id", "nunique"))
                .reset_index()
            )
            enrollments["month"] = enrollments["month"].astype(str)
            enroll_fig = px.line(
                enrollments,
                x="month",
                y="customers",
                markers=True,
                title="Enrollments over time",
            )

        engagement_fig = px.line(title="Flight activity over time")
        if not flights.empty:
            engagement = (
                flights.assign(year_month=lambda df: df["year_month_date"].dt.to_period("M"))
                .groupby("year_month")
                .agg(total_flights=("num_flights", "sum"))
                .reset_index()
            )
            engagement["year_month"] = engagement["year_month"].astype(str)
            engagement_fig = px.line(
                engagement,
                x="year_month",
                y="total_flights",
                markers=True,
                title="Flight activity over time",
            )

        map_fig = build_map_graph(
            "geo-map",
            pd.DataFrame(),
        ).figure
        if not customers.empty:
            map_data = (
                customers.dropna(subset=["province"])
                .groupby("province")
                .agg(
                    customers=("loyalty_id", "nunique"),
                    avg_ltv=("customer_lifetime_value", "mean"),
                    avg_tenure_months=("tenure_months", "mean"),
                    latitude=("latitude", "mean"),
                    longitude=("longitude", "mean"),
                )
                .reset_index()
            )
            map_fig = build_map_graph("geo-map", map_data).figure

        return enroll_fig, engagement_fig, map_fig

