from __future__ import annotations
import os
from flask import Flask, request, jsonify

from src.predict import predict_next_day

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.get("/predict")
def predict():
    ticker = request.args.get("ticker", "SPY").upper()
    model_path = request.args.get("model", f"models/{ticker}_rf.joblib")
    meta_path = request.args.get("meta", f"models/{ticker}_meta.joblib")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return jsonify({
            "error": "Model not found. Train first.",
            "expected": [model_path, meta_path],
        }), 400

    out = predict_next_day(ticker, model_path, meta_path)
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
