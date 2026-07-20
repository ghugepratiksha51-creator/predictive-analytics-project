"""
main.py
-------
End-to-end predictive analytics pipeline:

  1. Load historical data
  2. Clean & preprocess it
  3. Engineer features
  4. Time-aware train/test split (never shuffle time series!)
  5. Train 3 models: Linear Regression, Random Forest, Holt-Winters
  6. Evaluate accuracy (MAE, RMSE, MAPE, R2)
  7. Visualize actual vs. predicted, save charts + metrics + predictions

Run with:  python3 main.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import run_preprocessing
from models import train_linear_regression, train_random_forest, HoltWintersForecaster, FEATURE_COLS
from evaluate import evaluate_model, build_metrics_table

TARGET_COL = "sales"
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "historical_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
TEST_SIZE = 90  # forecast the last 90 days

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 1-3: Load, clean & engineer features")
    print("=" * 70)
    df = run_preprocessing(RAW_DATA_PATH, target_col=TARGET_COL)
    print(f"\nFinal dataset shape after cleaning/feature engineering: {df.shape}")
    print(df[["date", TARGET_COL] + FEATURE_COLS[:4]].tail(3))

    print("\n" + "=" * 70)
    print("STEP 4: Time-aware train/test split (last 90 days held out)")
    print("=" * 70)
    train_df = df.iloc[:-TEST_SIZE].reset_index(drop=True)
    test_df = df.iloc[-TEST_SIZE:].reset_index(drop=True)
    print(f"Train: {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"Test:  {len(test_df)} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    print("\n" + "=" * 70)
    print("STEP 5: Train models")
    print("=" * 70)

    print("Training Linear Regression...")
    lr = train_linear_regression(X_train, y_train)
    lr_preds = lr.predict(X_test)

    print("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)
    rf_preds = rf.predict(X_test)

    print("Training Holt-Winters exponential smoothing (time-series model)...")
    hw = HoltWintersForecaster(season_length=7).fit(train_df[TARGET_COL].values)
    hw_preds = hw.forecast(steps=TEST_SIZE)
    print(f"  -> chosen smoothing params: alpha={hw.alpha}, beta={hw.beta}, gamma={hw.gamma}")

    print("\n" + "=" * 70)
    print("STEP 6: Evaluate accuracy")
    print("=" * 70)
    results = [
        evaluate_model("Linear Regression", y_test, lr_preds),
        evaluate_model("Random Forest", y_test, rf_preds),
        evaluate_model("Holt-Winters (time series)", y_test, hw_preds),
    ]
    metrics_table = build_metrics_table(results)
    print("\n" + metrics_table.to_string(index=False))
    metrics_table.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)

    print("\n" + "=" * 70)
    print("STEP 7: Visualize predictions")
    print("=" * 70)

    # --- Chart 1: Actual vs predicted for all 3 models over the test period
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(test_df["date"], y_test, label="Actual", color="black", linewidth=2)
    ax.plot(test_df["date"], lr_preds, label="Linear Regression", linestyle="--")
    ax.plot(test_df["date"], rf_preds, label="Random Forest", linestyle="--")
    ax.plot(test_df["date"], hw_preds, label="Holt-Winters", linestyle="--")
    ax.set_title("Forecast vs Actual - Last 90 Days (Held-Out Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel(TARGET_COL)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "forecast_vs_actual.png"), dpi=150)
    plt.close(fig)

    # --- Chart 2: Full history + forecast overlay (context view) ----------
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(train_df["date"], train_df[TARGET_COL], label="Historical (train)", color="steelblue", alpha=0.6)
    ax.plot(test_df["date"], y_test, label="Actual (test)", color="black", linewidth=2)
    ax.plot(test_df["date"], rf_preds, label="Random Forest forecast", color="orange", linestyle="--")
    ax.axvline(train_df["date"].iloc[-1], color="gray", linestyle=":", label="Train/Test split")
    ax.set_title("Full History with Forecast (Best Model: Random Forest)")
    ax.set_xlabel("Date")
    ax.set_ylabel(TARGET_COL)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "full_history_with_forecast.png"), dpi=150)
    plt.close(fig)

    # --- Chart 3: Model accuracy comparison bar chart ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(metrics_table["Model"], metrics_table["RMSE"], color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_title("RMSE by Model (lower = better)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(metrics_table["Model"], metrics_table["R2"], color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].set_title("R² by Model (higher = better)")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)

    # --- Chart 4: Residuals plot for the best model -------------------------
    best_model_name = metrics_table.iloc[0]["Model"]
    best_preds = {"Linear Regression": lr_preds, "Random Forest": rf_preds,
                  "Holt-Winters (time series)": hw_preds}[best_model_name]
    residuals = y_test.values - best_preds
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(test_df["date"], residuals, alpha=0.6, color="crimson")
    ax.axhline(0, color="black", linestyle="--")
    ax.set_title(f"Residuals - {best_model_name} (best performing model)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual (Actual - Predicted)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "residuals_best_model.png"), dpi=150)
    plt.close(fig)

    # --- Save predictions table ---------------------------------------------
    preds_df = test_df[["date", TARGET_COL]].copy()
    preds_df.columns = ["date", "actual"]
    preds_df["linear_regression_pred"] = lr_preds
    preds_df["random_forest_pred"] = rf_preds
    preds_df["holt_winters_pred"] = hw_preds
    preds_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

    print(f"\nBest model on this dataset: {best_model_name}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
