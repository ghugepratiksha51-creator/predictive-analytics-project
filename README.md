# Predictive Analytics Using Historical Data

A ready-to-run project that forecasts a time series (synthetic "daily sales"
by default) using both **regression-on-features** models and a **classical
time-series** model, then evaluates and visualizes accuracy.

## What's inside

```
predictive_analytics_project/
├── data/
│   ├── generate_data.py      # creates a realistic synthetic historical dataset
│   └── historical_data.csv   # the generated data (3 years of daily "sales")
├── src/
│   ├── preprocessing.py      # cleaning + feature engineering
│   ├── models.py             # Linear Regression, Random Forest, Holt-Winters
│   └── evaluate.py           # MAE, RMSE, MAPE, R² metrics
├── outputs/                  # created after you run main.py
│   ├── model_metrics.csv
│   ├── predictions.csv
│   ├── forecast_vs_actual.png
│   ├── full_history_with_forecast.png
│   ├── model_comparison.png
│   └── residuals_best_model.png
├── main.py                   # runs the entire pipeline end-to-end
└── README.md
```

## Quick start

```bash
pip install pandas numpy scikit-learn matplotlib
python3 main.py
```

That's it — it will print progress at every stage and drop all charts/tables
into `outputs/`.

## What the pipeline actually does

1. **Clean the data** (`preprocessing.clean_data`)
   - drops duplicate rows
   - fills missing values with time-aware linear interpolation
   - caps outliers using the IQR rule (winsorizing) so a single bad data
     point can't wreck the models

2. **Engineer features** (`preprocessing.engineer_features`)
   - calendar features: month, day-of-week, weekend flag
   - cyclical encoding (sin/cos) of day-of-year and day-of-week, so the
     model understands "December 31 is close to January 1"
   - lag features: yesterday's value, same-day-last-week's value
   - rolling averages: trailing 7-day and 30-day means

3. **Time-aware train/test split**
   - the last 90 days are held out as the test set — **never shuffle a
     time series**, or you'll leak the future into the past

4. **Train three models**
   - **Linear Regression** — fast, interpretable baseline
   - **Random Forest** — captures non-linear relationships and feature
     interactions
   - **Holt-Winters exponential smoothing** — a classical time-series
     method (level + trend + weekly seasonality), implemented from
     scratch here because this environment has no internet access to
     install `statsmodels`. If you have internet access, swap in:
     ```python
     from statsmodels.tsa.holtwinters import ExponentialSmoothing
     ```
     for a more feature-complete version (it supports multiplicative
     seasonality, confidence intervals, etc.) — the rest of the pipeline
     doesn't need to change.

5. **Evaluate** with four standard forecast-accuracy metrics:
   - **MAE** (Mean Absolute Error) — average size of the miss, in the
     original units
   - **RMSE** (Root Mean Squared Error) — like MAE but penalizes big
     misses more
   - **MAPE** (Mean Absolute Percentage Error) — error as a % of actual
     value, easy to communicate to non-technical stakeholders
   - **R²** — how much of the variance in the target the model explains
     (1.0 = perfect, 0 = no better than predicting the average)

6. **Visualize**
   - actual vs. predicted lines for all 3 models on the held-out period
   - full history + forecast overlay for context
   - bar charts comparing RMSE and R² across models
   - residual plot for the winning model (look for patterns — a random
     scatter around 0 means the model isn't missing anything systematic)

## Using your own data instead of the synthetic dataset

Replace `data/historical_data.csv` with your own file. It needs:
- a `date` column (any parseable date format)
- a numeric target column (rename the `TARGET_COL` variable at the top
  of `main.py` if it isn't called `sales`)
- any other numeric/categorical predictor columns you have (like
  `marketing_spend` and `promo_flag` here) — add their names to
  `FEATURE_COLS` in `src/models.py`

Everything else in the pipeline — cleaning, feature engineering, splitting,
training, evaluating, plotting — will work unchanged.

## Extending this project

- Swap Random Forest for `GradientBoostingRegressor` or `XGBoost`
- Add confidence intervals to the Holt-Winters forecast
- Add cross-validation with `sklearn.model_selection.TimeSeriesSplit`
  instead of a single train/test split
- Forecast further into the *future* (beyond the last known date) by
  extending the date range and recursively generating lag/rolling
  features for the unseen period
