from __future__ import annotations
import pandas as pd
import numpy as np

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    return out

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["Close"].pct_change()
    out["vol_5d"] = out["ret_1d"].rolling(5).std()
    out["ma_5"] = out["Close"].rolling(5).mean()
    out["ma_20"] = out["Close"].rolling(20).mean()
    out["ma_ratio_5_20"] = out["ma_5"] / out["ma_20"]
    out["range_pct"] = (out["High"] - out["Low"]) / out["Close"]
    out["volume_chg"] = out["Volume"].pct_change()
    return out

def merge_sentiment(df_prices: pd.DataFrame, df_sent: pd.DataFrame) -> pd.DataFrame:
    out = df_prices.copy()
    if df_sent is None or df_sent.empty:
        out["sentiment"] = 0.0
        return out
    s = df_sent.copy()
    s["Date"] = pd.to_datetime(s["Date"])
    out = out.merge(s, on="Date", how="left")
    out["sentiment"] = out["sentiment"].fillna(0.0)
    return out

def make_dataset(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    df2 = df.dropna().copy()
    X = df2[feature_cols].copy()
    y = df2["Target"].astype(int).copy()
    return X, y
