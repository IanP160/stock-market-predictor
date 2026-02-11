from __future__ import annotations
import joblib
import numpy as np
import pandas as pd

from .data import fetch_prices
from .features import add_target, add_technical_features, make_dataset

DEFAULT_FEATURES = [
    "ret_1d",
    "vol_5d",
    "ma_ratio_5_20",
    "range_pct",
    "volume_chg",
    "sentiment",
]

def predict_next_day(ticker: str, model_path: str, meta_path: str) -> dict:
    model = joblib.load(model_path)
    meta = joblib.load(meta_path)
    feature_cols = meta["feature_cols"]

    df = fetch_prices(ticker, period="1y")
    df["sentiment"] = 0.0
    df = add_target(add_technical_features(df)).dropna()

    X = df[feature_cols].iloc[[-1]]
    p_up = float(model.predict_proba(X)[:, 1][0])
    return {"ticker": ticker, "prob_up": p_up}
