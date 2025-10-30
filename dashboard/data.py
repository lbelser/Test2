"""Utilities for loading and transforming dashboard data."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "company_performance.csv"


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """Load and enrich the company performance dataset."""
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["Profit"] = df["Revenue"] - df["Expenses"]
    df["ProfitMargin"] = df["Profit"] / df["Revenue"]
    df["ProfitMargin"] = df["ProfitMargin"].replace([pd.NA, pd.NaT, float("inf"), float("-inf")], 0).fillna(0)
    df["RevenuePerCustomer"] = df["Revenue"] / df["NewCustomers"].replace({0: pd.NA})
    df["RevenuePerCustomer"] = df["RevenuePerCustomer"].fillna(0)
    return df


def get_filter_options(df: pd.DataFrame) -> dict[str, list]:
    """Build filter option lists from the dataset."""
    return {
        "years": sorted(df["Year"].unique().tolist()),
        "months": sorted(df["Month"].unique().tolist()),
        "departments": sorted(df["Department"].unique().tolist()),
        "regions": sorted(df["Region"].unique().tolist()),
        "quarters": sorted(df["Quarter"].unique().tolist()),
    }


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert stored JSON records back to a pandas DataFrame."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def filter_data(
    df: pd.DataFrame,
    years: Iterable[int] | None,
    departments: Iterable[str] | None,
    regions: Iterable[str] | None,
    quarters: Iterable[str] | None,
    month_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Filter the dataframe based on the provided filters."""
    if df.empty:
        return df

    filtered = df.copy()
    if years:
        filtered = filtered[filtered["Year"].isin(years)]
    if month_range:
        start, end = month_range
        filtered = filtered[(filtered["Month"] >= start) & (filtered["Month"] <= end)]
    if departments:
        filtered = filtered[filtered["Department"].isin(departments)]
    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]
    if quarters:
        filtered = filtered[filtered["Quarter"].isin(quarters)]
    return filtered


def compute_summary(df: pd.DataFrame) -> dict[str, float]:
    """Calculate KPI metrics for the filtered dataset."""
    if df.empty:
        return {
            "revenue": 0.0,
            "revenue_growth": 0.0,
            "profit": 0.0,
            "profit_margin": 0.0,
            "avg_satisfaction": 0.0,
            "avg_engagement": 0.0,
            "new_customers": 0,
            "avg_churn": 0.0,
            "tickets": 0,
        }

    revenue = df["Revenue"].sum()
    profit = df["Profit"].sum()
    profit_margin = (profit / revenue) if revenue else 0
    avg_satisfaction = df["CustomerSatisfaction"].mean()
    avg_engagement = df["EmployeeEngagement"].mean()
    new_customers = df["NewCustomers"].sum()
    avg_churn = df["ChurnRate"].mean()
    tickets = df["SupportTickets"].sum()
    revenue_growth = 0.0
    yearly_revenue = df.groupby("Year")["Revenue"].sum().sort_index()
    if len(yearly_revenue) >= 2:
        latest = yearly_revenue.iloc[-1]
        prior = yearly_revenue.iloc[-2]
        if prior:
            revenue_growth = (latest - prior) / prior

    return {
        "revenue": float(revenue),
        "revenue_growth": float(revenue_growth),
        "profit": float(profit),
        "profit_margin": float(profit_margin),
        "avg_satisfaction": float(avg_satisfaction),
        "avg_engagement": float(avg_engagement),
        "new_customers": int(new_customers),
        "avg_churn": float(avg_churn),
        "tickets": int(tickets),
    }


def build_correlation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create a correlation matrix for operational metrics."""
    if df.empty:
        return pd.DataFrame()

    metrics = [
        "Revenue",
        "Expenses",
        "Profit",
        "NewCustomers",
        "ChurnRate",
        "CustomerSatisfaction",
        "EmployeeEngagement",
        "SupportTickets",
        "TrainingHours",
        "StrategicInitiatives",
    ]
    available = [metric for metric in metrics if metric in df.columns]
    corr = df[available].corr()
    return corr
