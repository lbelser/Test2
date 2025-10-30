# Executive Performance Dashboard

This project packages a Plotly Dash dashboard that leadership teams can use to explore a
synthetic company performance dataset. The experience highlights the revenue trajectory,
department and regional contributions, customer health, employee engagement and
operational workload — all without needing to touch code. Recent updates add year-over-
year growth diagnostics and allow exporting any filtered view for offline analysis.

## Getting started

1. **Install dependencies** (preferably inside a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard locally**:

   ```bash
   python app.py
   ```

3. Open the provided URL (by default `http://127.0.0.1:8050/`) in your browser.

## Dataset

A synthetic dataset that spans 2022–2023 is included in `data/company_performance.csv`.
The columns cover revenue, expenses, new customer acquisition, churn, satisfaction,
employee engagement and operational workload by department and region. If you want to
refresh the data you can rerun:

```bash
python scripts/generate_sample_data.py
```

## Dashboard highlights

- **Dynamic filters** for year, department, region, quarter and month range update every chart in real time.
- **KPI cards** summarise total revenue, year-over-year revenue growth, profit, margin,
  customer sentiment, employee engagement, churn and operational load.
- **Financial visualisations** track revenue vs. expenses vs. profit over time, year-over-
  year growth, department contribution and regional mix.
- **Customer and workforce insights** display satisfaction vs. engagement, retention vs.
  growth, revenue per customer and a sortable activity table for the most recent months.
- **Operational analytics** combine support load trends, training vs. initiative trade-offs
  and a correlation heatmap to surface relationships between metrics.
- **Data export tools** let you download the current filter context (with profit margin and
  revenue-per-customer fields) for presentations or deeper spreadsheet modeling.

Replace the CSV with your own dataset (keeping the same column names) to reuse the
interactive analysis for a different organisation.
