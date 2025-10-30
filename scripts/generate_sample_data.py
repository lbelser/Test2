"""Generate a synthetic company performance dataset for the dashboard.

This script recreates the dataset shipped with the repository.
"""
from __future__ import annotations

import csv
import math
import random
from datetime import datetime
from pathlib import Path


def generate_dataset(output_path: Path) -> None:
    """Create the CSV dataset used by the dashboard."""
    random.seed(42)

    start_year = 2022
    end_year = 2023
    months = [
        datetime(year, month, 1)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]
    departments = ["Sales", "Marketing", "Operations"]
    regions = ["North", "South", "West"]
    base_revenue = {"Sales": 82000, "Marketing": 54000, "Operations": 60000}
    region_multiplier = {"North": 1.05, "South": 0.95, "West": 1.0}
    base_satisfaction = {"Sales": 82, "Marketing": 78, "Operations": 86}
    base_engagement = {"Sales": 79, "Marketing": 75, "Operations": 83}
    base_tickets = {"Sales": 120, "Marketing": 90, "Operations": 160}

    rows = []
    for date in months:
        seasonal = 1 + 0.08 * math.sin((date.month - 1) / 12 * 2 * math.pi)
        trend = 1 + 0.06 * (date.year - start_year)
        for department in departments:
            for region in regions:
                revenue_noise = random.gauss(0, 3500)
                base_rev = base_revenue[department] * region_multiplier[region]
                revenue = base_rev * seasonal * trend + revenue_noise
                revenue = max(revenue, base_rev * 0.7)

                expense_ratio = 0.58 + random.gauss(0, 0.03)
                expense_ratio = min(max(expense_ratio, 0.48), 0.72)
                expenses = revenue * expense_ratio

                new_customers_mean = 140 if department == "Sales" else 90 if department == "Marketing" else 70
                new_customers = max(int(random.gauss(new_customers_mean, 12)), 30)

                churn_mean = 0.06 if department == "Sales" else 0.08 if department == "Marketing" else 0.05
                churn_rate = min(max(random.gauss(churn_mean, 0.012), 0.02), 0.12)

                satisfaction_adjust = 0.8 if region == "North" else -0.5 if region == "South" else 0
                satisfaction_trend = 0.5 * (date.year - start_year)
                customer_satisfaction = min(
                    max(
                        base_satisfaction[department]
                        + random.gauss(0, 3.5)
                        + satisfaction_adjust
                        + satisfaction_trend,
                        70,
                    ),
                    95,
                )
                engagement_trend = 0.4 * (date.year - start_year)
                employee_engagement = min(
                    max(base_engagement[department] + random.gauss(0, 2.8) + engagement_trend, 68),
                    92,
                )

                support_tickets = int(min(max(base_tickets[department] * (1 + random.gauss(0, 0.12)), 50), 250))
                training_hours = round(
                    min(max(16 + random.gauss(0, 3) + (2 if department == "Operations" else 0), 8), 28), 1
                )
                initiatives_mean = 6 if department == "Marketing" else 4 if department == "Sales" else 5
                initiatives = int(min(max(random.gauss(initiatives_mean, 1.4), 1), 12))

                rows.append(
                    [
                        date.strftime("%Y-%m-%d"),
                        department,
                        region,
                        round(revenue, 2),
                        round(expenses, 2),
                        new_customers,
                        round(churn_rate, 3),
                        round(customer_satisfaction, 1),
                        round(employee_engagement, 1),
                        support_tickets,
                        training_hours,
                        initiatives,
                    ]
                )

    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Date",
                "Department",
                "Region",
                "Revenue",
                "Expenses",
                "NewCustomers",
                "ChurnRate",
                "CustomerSatisfaction",
                "EmployeeEngagement",
                "SupportTickets",
                "TrainingHours",
                "StrategicInitiatives",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "data" / "company_performance.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(output)
