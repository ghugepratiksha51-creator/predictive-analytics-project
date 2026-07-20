"""
generate_data.py
-----------------
Creates a realistic synthetic "historical sales" dataset so the whole
pipeline can be run end-to-end without needing to download anything.

If you have your own historical CSV (sales, stock prices, website traffic,
energy usage, etc.) just replace data/historical_data.csv with your file --
as long as it has a date column and a numeric target column, main.py will
work with only a couple of variable-name tweaks at the top of the script.

The generated series includes, on purpose, the messy things real data has:
  - an upward trend
  - yearly + weekly seasonality
  - random noise
  - a handful of missing values
  - a handful of extreme outliers (e.g. a promo day / data glitch)
so the preprocessing step in main.py has real work to do.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ---- 1. Build a daily date range: 3 years of history ----------------------
start_date = "2022-01-01"
n_days = 3 * 365
dates = pd.date_range(start=start_date, periods=n_days, freq="D")

t = np.arange(n_days)

# ---- 2. Compose the signal -------------------------------------------------
trend = 150 + 0.12 * t                                    # slow upward trend
yearly_seasonality = 25 * np.sin(2 * np.pi * t / 365.25)  # yearly cycle
weekly_seasonality = 10 * np.sin(2 * np.pi * t / 7)       # weekly cycle
noise = np.random.normal(0, 8, n_days)                    # random noise

sales = trend + yearly_seasonality + weekly_seasonality + noise
sales = np.clip(sales, a_min=0, a_max=None)

# A slow-growing marketing spend feature, correlated loosely with sales
marketing_spend = 20 + 0.01 * t + np.random.normal(0, 3, n_days)
marketing_spend = np.clip(marketing_spend, a_min=0, a_max=None)

# A promo flag: random promo days that boost sales
promo_flag = np.random.choice([0, 1], size=n_days, p=[0.9, 0.1])
sales = sales + promo_flag * np.random.normal(30, 5, n_days)

df = pd.DataFrame({
    "date": dates,
    "sales": sales.round(2),
    "marketing_spend": marketing_spend.round(2),
    "promo_flag": promo_flag,
})

# ---- 3. Inject realistic messiness -----------------------------------------
# a) missing values (about 1.5% of rows)
missing_idx = np.random.choice(df.index, size=int(0.015 * n_days), replace=False)
df.loc[missing_idx, "sales"] = np.nan

# b) a few extreme outliers (data glitches / stockouts / one-off spikes)
outlier_idx = np.random.choice(df.index, size=8, replace=False)
df.loc[outlier_idx, "sales"] = df.loc[outlier_idx, "sales"] * np.random.choice([3, 0.05], size=8)

# c) a couple of duplicate rows (common in real exports)
dup_rows = df.sample(3, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

df = df.sort_values("date").reset_index(drop=True)

out_path = "/home/claude/predictive_analytics_project/data/historical_data.csv"
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} rows -> {out_path}")
print(df.head())
