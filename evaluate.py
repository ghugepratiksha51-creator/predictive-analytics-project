"""
evaluate.py
-----------
Standard forecast-accuracy metrics.
"""

import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r_squared(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "MAE": round(mae(y_true, y_pred), 3),
        "RMSE": round(rmse(y_true, y_pred), 3),
        "MAPE (%)": round(mape(y_true, y_pred), 2),
        "R2": round(r_squared(y_true, y_pred), 4),
    }


def build_metrics_table(results: list) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
