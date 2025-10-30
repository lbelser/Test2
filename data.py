"""Data loading and preparation utilities for the loyalty dashboard.

This module centralises the ingestion of the static CSV extracts that ship with
the project and exposes a :func:`prepare_data` helper returning tidy, analytics
ready tables.  The goal is to keep all heavy data work in one place so that the
Dash callbacks can focus on presentation logic.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
CUSTOMER_PATH = REPO_ROOT / "DM_AIAI_CustomerDB.csv"
FLIGHTS_PATH = REPO_ROOT / "DM_AIAI_FlightsDB.csv"
METADATA_PATH = REPO_ROOT / "DM_AIAI_Metadata.csv"


COMMON_CSV_KWARGS = {
    "dtype_backend": "numpy_nullable",
    "low_memory": False,
}

DATA_SOURCES = {
    "customers": {
        "path": CUSTOMER_PATH,
        "read_kwargs": {"index_col": 0},
    },
    "flights": {
        "path": FLIGHTS_PATH,
        "read_kwargs": {},
    },
    "metadata": {
        "path": METADATA_PATH,
        "read_kwargs": {"header": None},
    },
}


def _sniff_delimiter(path: Path, sample_size: int = 4096) -> str:
    """Attempt to detect the delimiter used in a CSV file."""

    default = ","
    try:
        with path.open("r", newline="") as handle:
            sample = handle.read(sample_size)
            if not sample:
                return default
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                return default
            delimiter = getattr(dialect, "delimiter", None)
            return delimiter or default
    except OSError:
        return default


def _read_csv_with_autodelimiter(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV file using an auto-detected delimiter."""

    delimiter = kwargs.pop("delimiter", None) or _sniff_delimiter(path)
    read_kwargs = {**COMMON_CSV_KWARGS, **kwargs, "delimiter": delimiter}

    try:
        return pd.read_csv(path, **read_kwargs)
    except TypeError as exc:
        # ``dtype_backend`` is only available in pandas >= 2.0; retry without it
        if "dtype_backend" in read_kwargs:
            read_kwargs.pop("dtype_backend", None)
            return pd.read_csv(path, **read_kwargs)
        raise exc


def _read_dataset(name: str) -> pd.DataFrame:
    """Load a named dataset using shared CSV settings."""

    config = DATA_SOURCES[name]
    read_kwargs = dict(config.get("read_kwargs", {}))
    return _read_csv_with_autodelimiter(config["path"], **read_kwargs)


DATE_NAME_PATTERNS = (
    "date",
    "timestamp",
    "time",
    "year_month",
)


def _coerce_datetime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert columns with date-like names or values to datetimes."""

    result = frame.copy()

    for column in result.columns:
        series = result[column]
        if not isinstance(series, pd.Series):
            continue

        name_hint = str(column).lower()
        has_name_hint = any(token in name_hint for token in DATE_NAME_PATTERNS)

        if not has_name_hint:
            if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
                continue
            sample = series.dropna().astype(str).head(50)
            if sample.empty:
                continue
            parsed_sample = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            success_ratio = parsed_sample.notna().mean()
            if success_ratio < 0.6:
                continue

        result[column] = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

    return result


def _slugify(column: str) -> str:
    """Convert raw column headers to snake_case identifiers."""

    cleaned = column.strip()
    cleaned = cleaned.replace("#", "_id")
    cleaned = cleaned.replace("%", "pct")
    cleaned = cleaned.replace("/", "_")
    cleaned = cleaned.replace("&", "and")
    cleaned = cleaned.replace("(", "_").replace(")", "_")
    cleaned = cleaned.replace("-", "_")
    cleaned = cleaned.replace(" ", "_")
    cleaned = cleaned.lower()
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _read_customers() -> pd.DataFrame:
    customers = _read_dataset("customers")
    customers = customers.rename(columns=_slugify)
    customers = _coerce_datetime_columns(customers)

    customers = customers.rename(
        columns={
            "enrollmentdateopening": "enrollment_date",
            "cancellationdate": "cancellation_date",
            "customer_lifetime_value": "customer_lifetime_value",
            "province_or_state": "province",
            "postal_code": "postal_code",
        }
    )

    latest_activity = customers[["enrollment_date", "cancellation_date"]].max().max()
    if pd.isna(latest_activity):
        latest_activity = pd.Timestamp("today").normalize()

    customers["tenure_days"] = (
        customers.get("cancellation_date", pd.NaT).fillna(latest_activity)
        - customers["enrollment_date"]
    ).dt.days
    customers["tenure_months"] = customers["tenure_days"] / 30.4375

    tenure_bins = [0, 180, 365, 730, 1095, np.inf]
    tenure_labels = [
        "< 6 months",
        "6-12 months",
        "1-2 years",
        "2-3 years",
        "3+ years",
    ]
    customers["tenure_bucket"] = pd.cut(
        customers["tenure_days"], bins=tenure_bins, labels=tenure_labels, right=False
    )

    numeric_columns = [
        "income",
        "customer_lifetime_value",
        "age",
        "latitude",
        "longitude",
    ]

    for column in numeric_columns:
        if column in customers.columns:
            customers[column] = pd.to_numeric(customers[column], errors="coerce")

    return customers


def _read_flights() -> pd.DataFrame:
    flights = _read_dataset("flights")
    flights = flights.rename(columns=_slugify)
    flights = _coerce_datetime_columns(flights)

    if "yearmonthdate" in flights.columns:
        flights["year_month_date"] = pd.to_datetime(
            flights["yearmonthdate"], errors="coerce"
        )
    else:
        flights["year_month_date"] = pd.to_datetime(
            flights[["year", "month"]].assign(day=1)
        )

    numeric_columns = [
        "num_flights",
        "num_flights_with_companions",
        "distance_km",
        "points_accumulated",
        "points_redeemed",
        "dollar_cost_points_redeemed",
    ]

    for column in numeric_columns:
        if column in flights.columns:
            flights[column] = pd.to_numeric(flights[column], errors="coerce").fillna(0)

    for column in ("year", "month"):
        if column in flights.columns:
            flights[column] = pd.to_numeric(flights[column], errors="coerce").astype("Int64")

    flights["year_month"] = flights["year_month_date"].dt.to_period("M")

    return flights


def _read_metadata() -> Optional[pd.DataFrame]:
    if not METADATA_PATH.exists():
        return None
    try:
        metadata = _read_dataset("metadata")
    except pd.errors.EmptyDataError:
        return None
    return _coerce_datetime_columns(metadata)


@dataclass
class PreparedData:
    customers: pd.DataFrame
    flights: pd.DataFrame
    customer_metrics: pd.DataFrame
    kpis: Dict[str, float]
    segment_breakdowns: Dict[str, pd.DataFrame]
    correlations: Dict[str, pd.DataFrame]
    time_series: Dict[str, pd.DataFrame]
    map_data: pd.DataFrame
    metadata: Optional[pd.DataFrame]

    def to_store_payload(self) -> Dict[str, str]:
        """Serialise key frames for storage in Dash dcc.Store components."""

        def to_json(df: Optional[pd.DataFrame]) -> Optional[str]:
            if df is None:
                return None
            return df.to_json(date_format="iso", orient="records")

        payload = {
            "customers": to_json(self.customers),
            "flights": to_json(self.flights),
            "customer_metrics": to_json(self.customer_metrics),
            "map_data": to_json(self.map_data),
        }

        payload["time_series"] = {
            name: to_json(frame) for name, frame in self.time_series.items()
        }
        payload["segments"] = {
            name: to_json(frame)
            for name, frame in self.segment_breakdowns.items()
        }
        payload["correlations"] = {
            name: to_json(frame) for name, frame in self.correlations.items()
        }

        payload["kpis"] = self.kpis
        return payload


def prepare_data() -> PreparedData:
    customers = _read_customers()
    flights = _read_flights()
    metadata = _read_metadata()

    # Aggregate flight activity at the customer level
    flight_agg = (
        flights.groupby("loyalty_id")
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

    flight_agg["avg_flights_per_month"] = np.where(
        flight_agg["active_months"] > 0,
        flight_agg["total_flights"] / flight_agg["active_months"],
        0,
    )
    flight_agg["redemption_ratio"] = np.where(
        flight_agg["total_points_accumulated"] > 0,
        flight_agg["total_points_redeemed"]
        / flight_agg["total_points_accumulated"],
        0,
    )

    customer_metrics = customers.merge(
        flight_agg, how="left", on="loyalty_id"
    ).fillna(
        {
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
    )

    # KPI metrics
    kpis = {
        "customer_count": int(customer_metrics["loyalty_id"].nunique()),
        "avg_lifetime_value": float(
            customer_metrics["customer_lifetime_value"].mean()
        ),
        "redemption_ratio": float(
            customer_metrics["redemption_ratio"].replace([np.inf, -np.inf], np.nan).mean()
        ),
        "avg_flights_per_month": float(
            customer_metrics["avg_flights_per_month"].mean()
        ),
    }

    # Segment breakdowns for marketing insights
    segment_breakdowns: Dict[str, pd.DataFrame] = {}
    for segment in ["loyalty_status", "province", "tenure_bucket", "education"]:
        if segment in customer_metrics.columns:
            segment_breakdowns[segment] = (
                customer_metrics.groupby(segment)
                .agg(
                    customers=("loyalty_id", "nunique"),
                    avg_ltv=("customer_lifetime_value", "mean"),
                    avg_redemption=("redemption_ratio", "mean"),
                )
                .reset_index()
                .sort_values("customers", ascending=False)
            )

    # Cohort table by enrollment year and loyalty status
    if "enrollment_date" in customer_metrics.columns:
        cohort = (
            customer_metrics.assign(
                enrollment_year=lambda df: df["enrollment_date"].dt.to_period("Y")
            )
            .dropna(subset=["enrollment_year"])
            .groupby(["enrollment_year", "loyalty_status"])
            .agg(
                customers=("loyalty_id", "nunique"),
                avg_ltv=("customer_lifetime_value", "mean"),
            )
            .reset_index()
        )
        cohort["enrollment_year"] = cohort["enrollment_year"].astype(str)
        segment_breakdowns["cohort"] = cohort

    # Correlation matrices for numeric attributes
    numeric_cols = customer_metrics.select_dtypes(include=[np.number])
    correlations = {
        method: numeric_cols.corr(method=method)
        for method in ("pearson", "spearman")
    }

    # Time series summaries
    enrollments_ts = (
        customer_metrics.dropna(subset=["enrollment_date"])
        .assign(month=lambda df: df["enrollment_date"].dt.to_period("M"))
        .groupby("month")
        .agg(customers=("loyalty_id", "nunique"), avg_ltv=("customer_lifetime_value", "mean"))
        .reset_index()
    )
    enrollments_ts["month"] = enrollments_ts["month"].astype(str)

    engagement_ts = (
        flights.dropna(subset=["year_month_date"])
        .groupby("year_month")
        .agg(
            total_flights=("num_flights", "sum"),
            points_accumulated=("points_accumulated", "sum"),
            points_redeemed=("points_redeemed", "sum"),
        )
        .reset_index()
    )
    engagement_ts["year_month"] = engagement_ts["year_month"].astype(str)

    time_series = {
        "enrollments": enrollments_ts,
        "engagement": engagement_ts,
    }

    # Map-ready data aggregated at province level
    map_data = (
        customer_metrics.dropna(subset=["province"])
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

    return PreparedData(
        customers=customers,
        flights=flights,
        customer_metrics=customer_metrics,
        kpis=kpis,
        segment_breakdowns=segment_breakdowns,
        correlations=correlations,
        time_series=time_series,
        map_data=map_data,
        metadata=metadata,
    )


PREPARED_DATA = prepare_data()


def frame_from_json(data: Optional[str]) -> pd.DataFrame:
    """Utility to reconstruct a DataFrame from ``dcc.Store`` JSON payloads."""

    if not data:
        return pd.DataFrame()
    return pd.read_json(data, orient="records", convert_dates=True)


