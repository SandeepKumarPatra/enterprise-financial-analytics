# Enterprise Financial Analytics & Forecasting System

An end-to-end business analytics and machine learning project built using Python, Pandas, NumPy, SQL, Scikit-learn, and statistical techniques. The system processes retail transaction data, analyzes financial performance, forecasts monthly revenue, detects unusual financial patterns, and generates data-driven business insights.

## Key Features

- **Data Cleaning:** Handle missing values, invalid dates, duplicate records, and inconsistent data types.
- **SQL Database:** Store cleaned transactions and monthly aggregates using SQLite.
- **Exploratory Data Analysis:** Analyze revenue, profit, regional performance, and product category profitability.
- **Financial KPIs:** Calculate total sales, total profit, profit margin, and order counts.
- **Feature Engineering:** Create time-based features, lag variables, and rolling statistical features.
- **Revenue Forecasting:** Apply a Random Forest regression model to predict monthly revenue and evaluate performance using MAE and RMSE.
- **Anomaly Detection:** Use Isolation Forest to identify unusual monthly revenue and profit patterns.
- **Business Insights:** Generate automated insights to highlight regional and category performance.
- **Data Export:** Export cleaned transactions, monthly summaries, revenue forecasts, and anomaly reports to CSV files.

## Technology Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Database:** SQLite, SQL
- **Machine Learning:** Scikit-learn
- **Statistical Analysis:** Descriptive statistics, correlation analysis
- **Visualization:** Matplotlib
- **Dashboard (Optional):** Streamlit

## Project Architecture

```text
Sample Superstore CSV
        |
        v
Data Ingestion & Validation
        |
        v
Data Cleaning (Pandas)
        |
        v
SQLite Database
        |
        v
EDA & Statistical Analysis
        |
        v
Feature Engineering
        |
        +----------------------+
        |                      |
        v                      v
Revenue Forecasting     Anomaly Detection
(Random Forest)         (Isolation Forest)
        |                      |
        +-----------+----------+
                    |
                    v
          Business Insights
                    |
                    v
       CSV Reports & Outputs
```

## Dataset

This project uses the Sample Superstore retail dataset, which contains order-level information such as order dates, sales, profit, product categories, customer segments, and regions.

**Dataset source:** [Sample Superstore Dataset on Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

Download the dataset and place the CSV file in the project's `data/` directory.

```text
data/
└── Sample - Superstore.csv
```

Note: The dataset provides retail sales and recorded profit rather than complete corporate financial statements. The project focuses on retail financial analysis and revenue forecasting; it does not claim to use actual departmental budgets or operating expense ledgers.

## Project Structure

```text
enterprise-financial-analytics/
├── data/
│   └── Sample - Superstore.csv
├── outputs/
├── src/
│   ├── __init__.py
│   └── analytics.py
├── app.py
├── run_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-financial-analytics.git
cd enterprise-financial-analytics
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the Sample Superstore CSV and place it in the `data/` directory using the filename specified above.

### 5. Run the analytics pipeline

```bash
python run_pipeline.py
```

The pipeline processes the dataset, stores results in SQLite, generates forecasts and anomaly reports, and prints business insights in the terminal.

### 6. Optional: Launch the dashboard

```bash
streamlit run app.py
```

The dashboard displays KPIs, financial trends, forecasts, anomaly detection results, and downloadable reports.

## Generated Outputs

The pipeline generates the following files in the `outputs/` directory:

- `superstore.db` — SQLite database containing cleaned transactions and monthly summaries.
- `superstore_cleaned.csv` — Cleaned transaction data.
- `monthly_summary.csv` — Monthly revenue, profit, order counts, and profit margins.
- `revenue_forecast.csv` — Predicted revenue for future months.
- `monthly_anomalies.csv` — Monthly financial patterns flagged as unusual.
- `quality_report.json` — Data quality summary.

## Forecast Evaluation

Forecasting performance is evaluated using:

- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.
- **Root Mean Squared Error (RMSE):** Penalizes larger prediction errors more heavily.

The model uses a chronological train/test split to reduce information leakage from future observations into the training data. Forecast accuracy depends on the amount and quality of available historical data.

## Limitations

- The dataset is retail transaction data, not a complete enterprise accounting ledger.
- Anomaly detection flags unusual patterns for investigation; it does not establish fraud or explain the cause.
- Forecasts are estimates and should be evaluated against simple baseline models.
- Correlations between discounts, sales, and profit do not establish causation.

## Future Improvements

- Add budget-versus-actual analysis using a clearly defined budget dataset.
- Compare Random Forest forecasts with statistical forecasting models.
- Add forecast backtesting and confidence intervals.
- Develop more advanced anomaly detection and root-cause analysis.
- Integrate a full financial database with departmental expenses and budget allocations.

## Author

**Sandeep Kumar Patra**

B.Tech in Electronics and Telecommunication Engineering | Aspiring AI/ML and Data Analytics Professional

## License

This project is available under the MIT License.
