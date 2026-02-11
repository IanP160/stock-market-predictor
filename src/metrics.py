from __future__ import annotations
import numpy as np
import pandas as pd

def historical_up_rate(y: pd.Series) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.mean(y.values))

def precision_on_up_predictions(y_true: pd.Series, y_pred_up: pd.Series) -> float:
    # y_pred_up is boolean: predicted up
    idx = y_pred_up.values.astype(bool)
    if idx.sum() == 0:
        return 0.0
    return float((y_true.values[idx] == 1).mean())
