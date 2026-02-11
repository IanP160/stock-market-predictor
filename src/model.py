from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier

@dataclass
class TrainResult:
    model: RandomForestClassifier
    feature_cols: list[str]
    test_accuracy: float

def train_random_forest(X_train, y_train, X_test, y_test, feature_cols: list[str]) -> TrainResult:
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    acc = float(model.score(X_test, y_test))
    return TrainResult(model=model, feature_cols=feature_cols, test_accuracy=acc)

def predict_proba_up(model, X):
    # returns probability of class 1 (up)
    proba = model.predict_proba(X)
    return proba[:, 1]
