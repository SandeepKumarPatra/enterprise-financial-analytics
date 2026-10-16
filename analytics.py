from __future__ import annotations

from pathlib import Path
import sqlite3
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


COLUMN_ALIASES = {
    "order_date": ["Order Date", "order_date", "OrderDate"],
    "sales": ["Sales", "sales", "Revenue", "revenue"],
    "profit": ["Profit", "profit"],
    "quantity": ["Quantity", "quantity", "Qty"],
    "discount": ["Discount", "discount"],
    "region": ["Region", "region"],
    "category": ["Category", "category"],
    "sub_category": ["Sub-Category", "Sub Category", "sub_category"],
    "segment": ["Segment", "segment"],
    "state": ["State", "state"],
    "city": ["City", "city"],
    "order_id": ["Order ID", "OrderID", "order_id"],
    "customer_id": ["Customer ID", "CustomerID", "customer_id"],
    "ship_date": ["Ship Date", "ShipDate", "ship_date"],
    "postal_code": ["Postal Code", "PostalCode", "postal_code"],
    "ship_mode": ["Ship Mode", "ShipMode", "ship_mode"],
}


def _find_column(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def standardize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Map common Sample Superstore column names to consistent internal names."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for target, aliases in COLUMN_ALIASES.items():
        found = _find_column(df.columns, aliases)
        if found is not None:
            rename[found] = target
    df = df.rename(columns=rename)

    required = ["order_date", "sales", "profit"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Required columns not found: " + ", ".join(missing) +
            ". Use the Sample Superstore CSV containing Order Date, Sales, and Profit."
        )
    return df


def load_and_clean_csv(path_or_buffer) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path_or_buffer, encoding_errors="replace")
    original_rows = len(raw)
    df = standardize_columns(raw)

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for col in ["sales", "profit", "quantity", "discount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that cannot be used for dated financial analysis.
    invalid_date_rows = int(df["order_date"].isna().sum())
    df = df.dropna(subset=["order_date", "sales", "profit"]).copy()

    # Missing dimensions are labelled rather than silently dropping transactions.
    for col in ["region", "category", "sub_category", "segment", "state", "city"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            df.loc[df[col].eq(""), col] = "Unknown"

    if "quantity" in df.columns:
        df["quantity"] = df["quantity"].fillna(0)
    if "discount" in df.columns:
        df["discount"] = df["discount"].fillna(0)

    # Remove exact duplicate rows only; repeated product lines can be legitimate.
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["quarter"] = df["order_date"].dt.to_period("Q").astype(str)
    df["profit_margin"] = np.where(
        df["sales"].ne(0), df["profit"] / df["sales"], np.nan
    )

    report = {
        "input_rows": original_rows,
        "clean_rows": len(df),
        "invalid_date_rows_removed": invalid_date_rows,
        "exact_duplicate_rows_removed": duplicate_rows,
        "date_start": str(df["order_date"].min().date()) if len(df) else "N/A",
        "date_end": str(df["order_date"].max().date()) if len(df) else "N/A",
    }
    return df, report


def load_to_sqlite(df: pd.DataFrame, db_path: str | Path) -> None:
    """Persist cleaned transaction-level data and monthly aggregate into SQLite."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("sales_clean", conn, if_exists="replace", index=False)
        monthly = monthly_summary(df)
        monthly.to_sql("monthly_summary", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_order_date ON sales_clean(order_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_region ON sales_clean(region)")
        conn.commit()


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["order_month"] = pd.to_datetime(work["order_month"])
    group_cols = ["order_month"]
    monthly = work.groupby(group_cols, as_index=False).agg(
        revenue=("sales", "sum"),
        profit=("profit", "sum"),
        orders=("order_id", "nunique") if "order_id" in work.columns else ("sales", "size"),
        units=("quantity", "sum") if "quantity" in work.columns else ("sales", "size"),
    )
    monthly["profit_margin"] = np.where(
        monthly["revenue"].ne(0), monthly["profit"] / monthly["revenue"], np.nan
    )
    return monthly.sort_values("order_month").reset_index(drop=True)


def build_forecast(df: pd.DataFrame, horizon: int = 6) -> tuple[pd.DataFrame, dict]:
    """Monthly revenue forecast using lag/rolling features and a chronological holdout."""
    monthly = monthly_summary(df)[["order_month", "revenue"]].copy()
    monthly = monthly.sort_values("order_month").reset_index(drop=True)

    # Include missing calendar months as zero-sales months, preserving a regular series.
    if monthly.empty:
        raise ValueError("No usable rows available for forecasting.")
    full_idx = pd.date_range(monthly["order_month"].min(), monthly["order_month"].max(), freq="MS")
    monthly = monthly.set_index("order_month").reindex(full_idx, fill_value=0)
    monthly.index.name = "order_month"
    monthly = monthly.reset_index()

    for lag in [1, 2, 3, 6, 12]:
        monthly[f"lag_{lag}"] = monthly["revenue"].shift(lag)
    monthly["rolling_mean_3"] = monthly["revenue"].shift(1).rolling(3).mean()
    monthly["rolling_std_3"] = monthly["revenue"].shift(1).rolling(3).std()
    monthly["month_num"] = monthly["order_month"].dt.month
    monthly["time_idx"] = np.arange(len(monthly))
    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_std_3", "month_num", "time_idx"
    ]
    model_data = monthly.dropna(subset=feature_cols + ["revenue"]).copy()
    metrics = {"mae": None, "rmse": None, "train_months": 0, "test_months": 0,
               "note": "Need at least 18 months for a meaningful lag-based evaluation."}

    if len(model_data) < 18:
        # Transparent fallback: seasonal-naive forecast where possible, otherwise last value.
        last_date = monthly["order_month"].max()
        values = monthly["revenue"].tolist()
        forecasts = []
        for i in range(1, horizon + 1):
            if len(values) >= 12:
                prediction = float(values[-12])
            else:
                prediction = float(values[-1]) if values else 0.0
            prediction = max(0.0, prediction)
            forecasts.append({"order_month": last_date + pd.offsets.MonthBegin(i),
                              "forecast_revenue": prediction})
            values.append(prediction)
        return pd.DataFrame(forecasts), metrics

    split = max(1, int(len(model_data) * 0.8))
    train, test = model_data.iloc[:split], model_data.iloc[split:]
    model = RandomForestRegressor(n_estimators=250, max_depth=6, random_state=42, min_samples_leaf=2)
    model.fit(train[feature_cols], train["revenue"])
    if len(test):
        preds = np.maximum(0, model.predict(test[feature_cols]))
        metrics = {
            "mae": float(mean_absolute_error(test["revenue"], preds)),
            "rmse": float(np.sqrt(mean_squared_error(test["revenue"], preds))),
            "train_months": int(len(train)),
            "test_months": int(len(test)),
            "note": "Chronological 80/20 holdout; metrics are in the dataset's currency units."
        }

    # Refit on all known feature-ready observations before recursive forecasting.
    model.fit(model_data[feature_cols], model_data["revenue"])
    history = monthly[["order_month", "revenue"]].copy()
    forecasts = []
    for _ in range(horizon):
        next_date = history["order_month"].max() + pd.offsets.MonthBegin(1)
        rev = history["revenue"].astype(float).tolist()
        row = {
            "lag_1": rev[-1],
            "lag_2": rev[-2] if len(rev) >= 2 else rev[-1],
            "lag_3": rev[-3] if len(rev) >= 3 else rev[-1],
            "lag_6": rev[-6] if len(rev) >= 6 else rev[-1],
            "lag_12": rev[-12] if len(rev) >= 12 else rev[-1],
            "rolling_mean_3": float(np.mean(rev[-3:])),
            "rolling_std_3": float(np.std(rev[-3:], ddof=1)) if len(rev[-3:]) > 1 else 0.0,
            "month_num": next_date.month,
            "time_idx": len(history),
        }
        pred = max(0.0, float(model.predict(pd.DataFrame([row])[feature_cols])[0]))
        forecasts.append({"order_month": next_date, "forecast_revenue": pred})
        history = pd.concat([history, pd.DataFrame([{"order_month": next_date, "revenue": pred}])], ignore_index=True)
    return pd.DataFrame(forecasts), metrics


def detect_monthly_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag unusual monthly revenue/profit using Isolation Forest."""
    monthly = monthly_summary(df).copy()
    if len(monthly) < 6:
        monthly["anomaly_score"] = np.nan
        monthly["is_anomaly"] = False
        monthly["anomaly_reason"] = "Need at least 6 monthly observations."
        return monthly

    features = monthly[["revenue", "profit", "profit_margin"]].replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median()).fillna(0)
    model = IsolationForest(contamination=0.10, random_state=42)
    labels = model.fit_predict(features)
    scores = model.decision_function(features)
    monthly["anomaly_score"] = scores
    monthly["is_anomaly"] = labels == -1

    med_rev = monthly["revenue"].median()
    med_profit = monthly["profit"].median()
    reasons = []
    for _, row in monthly.iterrows():
        if not row["is_anomaly"]:
            reasons.append("Typical pattern")
            continue
        flags = []
        if med_rev and abs(row["revenue"] - med_rev) / abs(med_rev) > 0.35:
            flags.append("unusual revenue")
        if med_profit and abs(row["profit"] - med_profit) / max(abs(med_profit), 1) > 0.5:
            flags.append("unusual profit")
        reasons.append(", ".join(flags) if flags else "unusual combination of financial metrics")
    monthly["anomaly_reason"] = reasons
    return monthly


def make_business_insights(df: pd.DataFrame) -> list[str]:
    insights = []
    total_sales = df["sales"].sum()
    total_profit = df["profit"].sum()
    margin = total_profit / total_sales if total_sales else np.nan
    insights.append(f"Total sales are {total_sales:,.2f}; total profit is {total_profit:,.2f}.")
    insights.append(
        f"Overall profit margin is {margin:.1%}." if pd.notna(margin)
        else "Overall profit margin could not be calculated because total sales are zero."
    )

    if "region" in df.columns:
        reg = df.groupby("region", dropna=False).agg(sales=("sales", "sum"), profit=("profit", "sum"))
        if len(reg):
            best = reg["profit"].idxmax()
            worst = reg["profit"].idxmin()
            insights.append(f"Highest-profit region: {best} ({reg.loc[best, 'profit']:,.2f} profit).")
            insights.append(f"Lowest-profit region: {worst} ({reg.loc[worst, 'profit']:,.2f} profit); investigate mix, discounts, and costs.")

    if "category" in df.columns:
        cat = df.groupby("category").agg(sales=("sales", "sum"), profit=("profit", "sum"))
        if len(cat):
            low = cat["profit"].idxmin()
            insights.append(f"Category needing attention: {low} ({cat.loc[low, 'profit']:,.2f} profit).")

    if "discount" in df.columns and df["discount"].notna().any():
        corr = df[["discount", "profit"]].corr(numeric_only=True).iloc[0, 1]
        if pd.notna(corr):
            insights.append(
                f"Discount-profit correlation is {corr:.2f}. This is an association, not proof that discounts cause lower profit."
            )
    return insights
