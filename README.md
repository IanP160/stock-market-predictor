# Stock Market Predictor (with Sentiment + Backtesting)

Portfolio project that predicts next-day stock direction using historical price features and optional news sentiment (NewsAPI), with a walk-forward backtest and a small Flask inference API.

## What this demonstrates
- Time-series aware modeling (no shuffling)
- Feature engineering (returns, volatility, moving averages, ranges, volume change)
- Optional external signal integration (NewsAPI sentiment)
- Benchmarking vs a simple baseline ("always up" = historical up-rate)
- Walk-forward backtesting + equity curve vs buy & hold
- Deployable inference endpoint (Flask)

## Project structure
- `src/` – data, features, sentiment, model training, backtesting
- `app.py` – Flask API (`/predict`)
- `StockPredictor.ipynb` – original notebook exploration
- `models/` – saved artifacts (gitignored)
- `reports/` – generated plots (gitignored)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional: enable sentiment (NewsAPI)
Create a NewsAPI key and set:
```bash
export NEWSAPI_KEY="YOUR_KEY"
```

## Train + backtest
Without sentiment:
```bash
python -m src.train --ticker SPY
```

With sentiment:
```bash
python -m src.train --ticker SPY --with-sentiment
```

Notes:
- The training script prints test accuracy.
- It also prints **precision on days the model predicts UP** and compares it to the baseline up-rate.
- It saves an equity curve plot to `reports/<TICKER>_equity_curve.png`.

You can adjust the UP threshold:
```bash
python -m src.train --ticker SPY --prob-threshold 0.60
```

## Run the Flask API
Train first, then:
```bash
python app.py
```

Example request:
```bash
curl "http://127.0.0.1:5000/predict?ticker=SPY"
```

Response:
```json
{ "ticker": "SPY", "prob_up": 0.57 }
```

## Disclaimer
This is an educational/portfolio project, not financial advice.
