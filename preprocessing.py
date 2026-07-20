"""
preprocessing.py
-----------------
Cleans the raw historical data and engineers features that predictive
models can use:
  - remove duplicates
  - handle missing values (time-aware interpolation)
  - cap outliers (winsorizing via IQR)
  - engineer calendar features (day of week, month, etc.)
  - engineer lag & rolling-window features (key for time-series ML)
"""

import numpy as np
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def clean_data(df: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
    df = df.copy()

    # 1. Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    # 2. Sort chronologically and set date as index for time-aware operations
    df = df.sort_values("date").reset_index(drop=True)

    # 3. Handle missing values with time-based interpolation
    n_missing = df[target_col].isna().sum()
    df[target_col] = df[target_col].interpolate(method="linear", limit_direction="both")
    print(f"Filled {n_missing} missing values in '{target_col}' via linear interpolation")

    # 4. Outlier capping using IQR (winsorizing) -- keeps the row, tames the value
    q1, q3 = df[target_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((df[target_col] < lower) | (df[target_col] > upper)).sum()
    df[target_col] = df[target_col].clip(lower=lower, upper=upper)
    print(f"Capped {n_outliers} outlier values in '{target_col}' to [{lower:.1f}, {upper:.1f}]")

    return df


def engineer_features(df: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
    df = df.copy()

    # Calendar features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear

    # Cyclical encoding so the model knows Dec 31 is close to Jan 1
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag features: yesterday's value, last week's value
    df["lag_1"] = df[target_col].shift(1)
    df["lag_7"] = df[target_col].shift(7)

    # Rolling window features: trailing 7 and 30 day averages
    df["rolling_mean_7"] = df[target_col].shift(1).rolling(window=7).mean()
    df["rolling_mean_30"] = df[target_col].shift(1).rolling(window=30).mean()

    # Drop the initial rows where lag/rolling features are NaN (no history yet)
    df = df.dropna().reset_index(drop=True)

    return df


def run_preprocessing(raw_path: str, target_col: str = "sales") -> pd.DataFrame:
    df = load_raw_data(raw_path)
    df = clean_data(df, target_col)
    df = engineer_features(df, target_col)
    return df
