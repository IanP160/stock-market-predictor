from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit

from .model import train_random_forest, predict_proba_up
from .metrics import historical_up_rate, precision_on_up_predictions

def walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    initial_train_size: int = 252*2,
    step: int = 5,
    prob_threshold: float = 0.55,
) -> dict:
    """Walk-forward backtest.

    - Trains on expanding window
    - Predicts next 'step' days
    - Goes long when p(up) >= prob_threshold
    """
    df2 = df.dropna().reset_index(drop=True)
    X = df2[feature_cols]
    y = df2["Target"].astype(int)
    close = df2["Close"].astype(float)
    dates = df2["Date"]

    preds = np.full(len(df2), np.nan)
    signals = np.zeros(len(df2))

    start = initial_train_size
    while start < len(df2) - 1:
        train_end = start
        test_end = min(start + step, len(df2) - 1)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

        # split last 20% of train as validation-ish test for reporting
        split = int(len(X_train) * 0.8)
        trX, teX = X_train.iloc[:split], X_train.iloc[split:]
        trY, teY = y_train.iloc[:split], y_train.iloc[split:]

        res = train_random_forest(trX, trY, teX, teY, feature_cols)
        p_up = predict_proba_up(res.model, X_test)

        preds[train_end:test_end] = p_up
        signals[train_end:test_end] = (p_up >= prob_threshold).astype(int)

        start = test_end

    # Strategy returns: if signal=1 at day t, earn next-day return from t->t+1
    next_ret = close.pct_change().shift(-1).fillna(0.0).values
    strat_ret = next_ret * signals
    buyhold_ret = next_ret

    equity = (1.0 + strat_ret).cumprod()
    buyhold = (1.0 + buyhold_ret).cumprod()

    # Metrics
    mask = ~np.isnan(preds)
    y_eval = y.iloc[mask]
    sig_eval = pd.Series(signals[mask], index=y_eval.index).astype(bool)
    model_precision = precision_on_up_predictions(y_eval, sig_eval)
    baseline_precision = historical_up_rate(y_eval)  # 'always up' precision == up-rate

    return {
        "dates": dates,
        "pred_proba_up": preds,
        "signals": signals,
        "equity_curve": equity,
        "buyhold_curve": buyhold,
        "model_precision": model_precision,
        "baseline_precision": baseline_precision,
        "eval_points": int(mask.sum()),
    }

def save_equity_plot(dates, equity, buyhold, outpath: str):
    plt.figure()
    plt.plot(dates, equity, label="Strategy")
    plt.plot(dates, buyhold, label="Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity (Growth of $1)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
