from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.analytics import (
    build_forecast, detect_monthly_anomalies, load_and_clean_csv,
    load_to_sqlite, make_business_insights, monthly_summary
)

BASE = Path(__file__).parent
DB_PATH = BASE / "outputs" / "superstore.db"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

st.set_page_config(page_title="Enterprise Financial Analytics", page_icon="📊", layout="wide")
st.title("📊 Enterprise Financial Analytics & Forecasting")
st.caption("Sample Superstore • SQL • Data Quality • EDA • Forecasting • Anomaly Detection")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload Sample Superstore CSV", type=["csv"])
    horizon = st.slider("Forecast horizon (months)", min_value=3, max_value=12, value=6)
    st.markdown("---")
    st.caption("Forecasts are exploratory estimates, not financial advice or audited projections.")

data_path = BASE / "data" / "Sample - Superstore.csv"
source = uploaded if uploaded is not None else (str(data_path) if data_path.exists() else None)

if source is None:
    st.warning("Add the Sample Superstore CSV to `data/Sample - Superstore.csv`, or upload it using the sidebar.")
    st.markdown("""
    **Dataset:** [Download Sample Superstore on Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

    Then unzip the download and place the CSV in the project's `data` folder. The app accepts common column-name variants.
    """)
    st.stop()

try:
    df, quality = load_and_clean_csv(source)
    load_to_sqlite(df, DB_PATH)
except Exception as exc:
    st.error(f"Could not load this file: {exc}")
    st.stop()

st.success(f"Loaded {len(df):,} cleaned transaction rows into SQLite.")
st.subheader("1. Data quality & KPI overview")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Input rows", f"{quality['input_rows']:,}")
q2.metric("Clean rows", f"{quality['clean_rows']:,}")
q3.metric("Duplicate rows removed", f"{quality['exact_duplicate_rows_removed']:,}")
q4.metric("Date range", f"{quality['date_start']} → {quality['date_end']}")

sales = float(df["sales"].sum())
profit = float(df["profit"].sum())
margin = profit / sales if sales else 0
orders = int(df["order_id"].nunique()) if "order_id" in df.columns else len(df)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sales", f"{sales:,.0f}")
k2.metric("Total Profit", f"{profit:,.0f}")
k3.metric("Profit Margin", f"{margin:.1%}")
k4.metric("Orders / rows", f"{orders:,}")

with st.expander("Cleaning report"):
    st.json(quality)
    st.write("Cleaned table: `sales_clean`; monthly aggregate: `monthly_summary`; SQLite file: `outputs/superstore.db`.")

st.subheader("2. EDA & business breakdowns")
left, right = st.columns(2)
monthly = monthly_summary(df)
with left:
    st.markdown("**Monthly revenue and profit**")
    fig, ax = plt.subplots()
    ax.plot(monthly["order_month"], monthly["revenue"], label="Revenue")
    ax.plot(monthly["order_month"], monthly["profit"], label="Profit")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount")
    ax.legend()
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("**Sales and profit by region**")
    if "region" in df.columns:
        by_region = df.groupby("region", as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"))
        fig, ax = plt.subplots()
        ax.bar(by_region["region"], by_region["sales"], label="Sales")
        ax.set_ylabel("Sales")
        ax.tick_params(axis="x", rotation=25)
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(by_region.sort_values("profit", ascending=False), use_container_width=True)
    else:
        st.info("Region column is missing in this CSV.")

if "category" in df.columns:
    st.markdown("**Category profitability**")
    category = df.groupby("category", as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"))
    st.dataframe(category.sort_values("profit", ascending=False), use_container_width=True)

st.subheader("3. SQL layer")
with st.expander("Example SQL queries (run against outputs/superstore.db)"):
    st.code("""
-- Monthly revenue and profit
SELECT strftime('%Y-%m', order_month) AS month,
       SUM(revenue) AS revenue, SUM(profit) AS profit
FROM monthly_summary
GROUP BY month
ORDER BY month;

-- Profit by region
SELECT region, SUM(sales) AS sales, SUM(profit) AS profit
FROM sales_clean
GROUP BY region
ORDER BY profit DESC;

-- Category with lowest total profit
SELECT category, SUM(profit) AS profit
FROM sales_clean
GROUP BY category
ORDER BY profit ASC;
""", language="sql")
    with sqlite3.connect(DB_PATH) as conn:
        st.write("Monthly SQL query result:")
        st.dataframe(pd.read_sql_query("""
            SELECT strftime('%Y-%m', order_month) AS month,
                   ROUND(SUM(revenue), 2) AS revenue,
                   ROUND(SUM(profit), 2) AS profit
            FROM monthly_summary GROUP BY month ORDER BY month
        """, conn), use_container_width=True)

st.subheader("4. Monthly revenue forecasting")
forecast, metrics = build_forecast(df, horizon=horizon)
m1, m2, m3 = st.columns(3)
m1.metric("Holdout MAE", "N/A" if metrics["mae"] is None else f"{metrics['mae']:,.0f}")
m2.metric("Holdout RMSE", "N/A" if metrics["rmse"] is None else f"{metrics['rmse']:,.0f}")
m3.metric("Forecast horizon", f"{horizon} months")
st.caption(metrics["note"])
fig, ax = plt.subplots()
ax.plot(monthly["order_month"], monthly["revenue"], label="Historical revenue")
ax.plot(forecast["order_month"], forecast["forecast_revenue"], marker="o", linestyle="--", label="Forecast")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")
ax.legend()
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.dataframe(forecast, use_container_width=True)

st.subheader("5. Anomaly detection")
anomalies = detect_monthly_anomalies(df)
flagged = anomalies[anomalies["is_anomaly"]]
if flagged.empty:
    st.info("No monthly anomalies were flagged, or there is not enough monthly data.")
else:
    st.warning(f"{len(flagged)} unusual month(s) flagged. These are investigation signals, not proof of fraud or error.")
    st.dataframe(flagged[["order_month", "revenue", "profit", "profit_margin", "anomaly_score", "anomaly_reason"]], use_container_width=True)
fig, ax = plt.subplots()
ax.plot(anomalies["order_month"], anomalies["revenue"], label="Revenue")
ax.scatter(
    anomalies.loc[anomalies["is_anomaly"], "order_month"],
    anomalies.loc[anomalies["is_anomaly"], "revenue"],
    label="Flagged anomaly", marker="x"
)
ax.legend()
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("6. Automated business insights")
for insight in make_business_insights(df):
    st.markdown(f"- {insight}")

st.subheader("7. Export cleaned data and analysis")
st.download_button("Download cleaned transactions CSV", df.to_csv(index=False).encode("utf-8"), "superstore_cleaned.csv", "text/csv")
st.download_button("Download monthly summary CSV", monthly.to_csv(index=False).encode("utf-8"), "monthly_summary.csv", "text/csv")
st.download_button("Download forecast CSV", forecast.to_csv(index=False).encode("utf-8"), "revenue_forecast.csv", "text/csv")
st.download_button("Download anomaly report CSV", anomalies.to_csv(index=False).encode("utf-8"), "monthly_anomalies.csv", "text/csv")
