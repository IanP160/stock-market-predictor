from __future__ import annotations
import pandas as pd
import yfinance as yf

def fetch_prices(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No price data returned for ticker={ticker}")
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df
