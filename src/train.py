from __future__ import annotations
import argparse
import os
import joblib
import pandas as pd

from .data import fetch_prices
from .sentiment import fetch_news_sentiment
from .features import add_target, add_technical_features, merge_sentiment, make_dataset
from .model import train_random_forest
from .backtest import walk_forward_backtest, save_equity_plot
from .metrics import historical_up_rate, precision_on_up_predictions

DEFAULT_FEATURES = [
    "ret_1d",
    "vol_5d",
    "ma_ratio_5_20",
    "range_pct",
    "volume_chg",
    "sentiment",
]

def main(ticker: str, period: str, out_dir: str, with_sentiment: bool, prob_threshold: float):
    prices = fetch_prices(ticker, period=period)
    prices = add_target(prices)
    prices = add_technical_features(prices)

    if with_sentiment:
        start = prices["Date"].min().date()
        end = prices["Date"].max().date()
        sent = fetch_news_sentiment(query=ticker, start=start, end=end)
    else:
        sent = pd.DataFrame({"Date": [], "sentiment": []})

    df = merge_sentiment(prices, sent).dropna().copy()

    feature_cols = DEFAULT_FEATURES
    X, y = make_dataset(df, feature_cols)

    # Train/test split (time-based)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    res = train_random_forest(X_train, y_train, X_test, y_test, feature_cols)

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{ticker}_rf.joblib")
    meta_path = os.path.join(out_dir, f"{ticker}_meta.joblib")
    joblib.dump(res.model, model_path)
    joblib.dump({"feature_cols": feature_cols, "test_accuracy": res.test_accuracy}, meta_path)

    # Precision benchmark: compare precision on "predicted up" vs baseline up-rate
    p_up = res.model.predict_proba(X_test)[:, 1]
    pred_up = p_up >= prob_threshold
    model_prec = precision_on_up_predictions(y_test, pd.Series(pred_up))
    base_prec = historical_up_rate(y_test)

    print(f"Ticker: {ticker}")
    print(f"Test accuracy: {res.test_accuracy:.4f}")
    print(f"Model precision (only days predicted UP): {model_prec:.4f}")
    print(f"Baseline precision ('always UP' = up-rate): {base_prec:.4f}")

    # Backtest + equity curve
    bt = walk_forward_backtest(df, feature_cols, prob_threshold=prob_threshold)
    print(f"Backtest eval points: {bt['eval_points']}")
    print(f"Backtest model precision: {bt['model_precision']:.4f}")
    print(f"Backtest baseline precision: {bt['baseline_precision']:.4f}")

    plot_path = os.path.join("reports", f"{ticker}_equity_curve.png")
    save_equity_plot(bt["dates"], bt["equity_curve"], bt["buyhold_curve"], plot_path)
    print(f"Saved equity curve: {plot_path}")
    print(f"Saved model: {model_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--period", default="10y")
    ap.add_argument("--out", default="models")
    ap.add_argument("--with-sentiment", action="store_true", help="Fetch NewsAPI sentiment (requires NEWSAPI_KEY)")
    ap.add_argument("--prob-threshold", type=float, default=0.55, help="Probability threshold for UP signal")
    args = ap.parse_args()
    main(args.ticker, args.period, args.out, args.with_sentiment, args.prob_threshold)
