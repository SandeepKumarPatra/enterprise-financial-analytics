"""Run the analytics pipeline without launching Streamlit."""
from pathlib import Path
import argparse
import json

from src.analytics import (
    build_forecast, detect_monthly_anomalies, load_and_clean_csv,
    load_to_sqlite, make_business_insights, monthly_summary
)

ROOT = Path(__file__).parent

def main():
    parser = argparse.ArgumentParser(description="Sample Superstore analytics pipeline")
    parser.add_argument("--input", default=str(ROOT / "data" / "Sample - Superstore.csv"),
                        help="Path to Sample Superstore CSV")
    parser.add_argument("--horizon", type=int, default=6, help="Forecast months (3–12 recommended)")
    args = parser.parse_args()

    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    df, quality = load_and_clean_csv(args.input)
    load_to_sqlite(df, out / "superstore.db")
    monthly = monthly_summary(df)
    forecast, metrics = build_forecast(df, args.horizon)
    anomalies = detect_monthly_anomalies(df)

    df.to_csv(out / "superstore_cleaned.csv", index=False)
    monthly.to_csv(out / "monthly_summary.csv", index=False)
    forecast.to_csv(out / "revenue_forecast.csv", index=False)
    anomalies.to_csv(out / "monthly_anomalies.csv", index=False)
    (out / "quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    print("\\n=== DATA QUALITY ===")
    print(json.dumps(quality, indent=2))
    print("\\n=== FORECAST METRICS ===")
    print(json.dumps(metrics, indent=2))
    print("\\n=== BUSINESS INSIGHTS ===")
    for item in make_business_insights(df):
        print("-", item)
    print(f"\\nDone. Outputs saved to: {out}")

if __name__ == "__main__":
    main()
