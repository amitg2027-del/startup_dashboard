# Failed Indian Startups Analytics Dashboard

Interactive Streamlit dashboard for high-level business analysis of failed Indian startups/MSMEs using `startups.xlsx`.

## Features

- Executive KPI overview
- Shutdown and funding charts
- Sector and failure-category analysis
- Correlation heatmap
- Regression modeling and feature importance
- What-if prediction form
- Filterable data explorer with CSV export

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-dashboard.txt
```

## Run

```powershell
streamlit run startup_analytics_dashboard.py
```

By default, the app looks for:

```text
C:\Users\hp\Desktop\startups.xlsx
```

You can change the file path from the dashboard sidebar.

## Notes

The workbook contains failed startups only, so the predictive section is not a survivor-vs-failure classifier. It predicts outcomes within the failed-company dataset, such as funding, losses, or other numeric business metrics.
