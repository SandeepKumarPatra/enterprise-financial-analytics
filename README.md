# Enterprise Financial Analytics & Forecasting System

An end-to-end analytics portfolio project built around the **Sample Superstore** dataset. It demonstrates data ingestion, SQL storage, cleaning, EDA, statistical KPIs, time-series feature engineering, revenue forecasting, anomaly detection, and automated business insights.

> **Data honesty:** Sample Superstore is retail order data, not a complete corporate general ledger. This project forecasts revenue and analyses recorded profit. It does not claim to have actual operating expenses, departmental budgets, or audited financial statements.

## Architecture

```text
Sample Superstore CSV
        |
        v
Column standardization + validation
        |
        v
Pandas cleaning and feature engineering
        |
        +--------------------------+
        |                          |
        v                          v
SQLite database              EDA + KPI analysis
(sales_clean,                      |
 monthly_summary)                  |
        |                          |
        +------------+-------------+
                     |
           +---------+---------+
           |                   |
           v                   v
 Monthly revenue forecast   Anomaly detection
           |                   |
           +---------+---------+
                     |
                     v
        Business insights + CSV exports
                     |
                     v
             Streamlit dashboard
```

## Features

- Load a CSV with common Sample Superstore column-name variants.
- Validate required fields and parse dates/numeric columns.
- Report invalid dates and exact duplicate rows removed.
- Derive year, month, quarter, monthly period, and profit margin.
- Persist cleaned transactions and monthly summaries in SQLite.
- KPIs: sales, profit, margin, order count.
- EDA: monthly trends, regional performance, category profitability.
- Forecast monthly revenue using lag and rolling features with a chronological holdout.
- Evaluate forecasts with MAE and RMSE when enough history is available.
- Flag unusual months with Isolation Forest.
- Generate rule-based business insights.
- Export cleaned data, monthly summaries, forecasts, and anomaly reports.

## 1. Get the dataset

Download the classic Sample Superstore CSV:
https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

Unzip the download. Place the CSV at:

```text
data/Sample - Superstore.csv
```

If the downloaded file has another name, either rename it or pass its path to the CLI. The dashboard also allows a CSV upload.

## 2. Setup on Windows

Open Command Prompt or the VS Code terminal inside this project folder.

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run the dashboard

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal. Upload the CSV in the sidebar if it is not in the default `data` path.

## 4. Run the pipeline without the dashboard

```powershell
python run_pipeline.py
```

Or specify a different CSV:

```powershell
python run_pipeline.py --input "C:\Users\YourName\Downloads\SampleSuperstore.csv" --horizon 6
```

Outputs are written to `outputs/`:
- `superstore.db` — SQLite database
- `superstore_cleaned.csv`
- `monthly_summary.csv`
- `revenue_forecast.csv`
- `monthly_anomalies.csv`
- `quality_report.json`

## SQL examples

```sql
-- Monthly revenue and profit
SELECT strftime('%Y-%m', order_month) AS month,
       SUM(revenue) AS revenue,
       SUM(profit) AS profit
FROM monthly_summary
GROUP BY month
ORDER BY month;

-- Regional profitability
SELECT region, SUM(sales) AS sales, SUM(profit) AS profit
FROM sales_clean
GROUP BY region
ORDER BY profit DESC;

-- Lowest-profit product category
SELECT category, SUM(profit) AS profit
FROM sales_clean
GROUP BY category
ORDER BY profit ASC;
```

## Methodology and limitations

### Cleaning
- Required fields: order date, sales, and profit.
- Unparseable dates and rows without required financial values are removed.
- Exact duplicate rows are removed; similar-looking rows are retained because repeated line items may be legitimate.
- Missing category/dimension values are labelled `Unknown`; missing quantity and discount values are set to zero where those columns exist.
- No currency conversion is performed. Values remain in the dataset's original units/currency.

### Forecasting
- The target is monthly revenue, aggregated from order-line sales.
- Lag and rolling features are created using only prior observations.
- Evaluation uses a chronological 80/20 split rather than a random split.
- Small datasets use a seasonal-naive/last-value fallback. A very short history should not be presented as a reliable forecast.
- Recursive multi-step forecasts can drift. Compare against a simple baseline before making claims about model quality.

### Anomaly detection
Isolation Forest flags unusual monthly combinations of revenue, profit, and margin. An anomaly is only a review signal; it is not proof of fraud, error, or a business cause.

### Business insights
The app produces descriptive insights from the uploaded data. Correlation does not establish causation. Validate any finding before using it for a business decision.

