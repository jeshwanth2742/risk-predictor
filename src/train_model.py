"""
train_model.py
----------------
Trains an XGBoost classifier to predict probability of default
(SeriousDlqin2yrs), evaluates it on a held-out test set (accuracy + AUC),
and saves the trained model to disk.

Run:
    python src/train_model.py
"""
import os
import json
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Allow running as `python src/train_model.py` from the project root.
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_raw_data, preprocess_and_split, FEATURE_COLUMNS

MODEL_PATH = "models/xgb_model.joblib"
METRICS_PATH = "models/metrics.json"


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """Train an XGBoost classifier. Uses CPU-only, modest tree count/depth
    since this is a small tabular dataset -- training should take seconds."""
    # Handle class imbalance (defaults are typically the minority class).
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return metrics


def main():
    df, is_synthetic = load_raw_data()
    X_train, X_test, y_train, y_test, pipeline = preprocess_and_split(df)

    print("[train] Training XGBoost classifier...")
    model = train_xgboost(X_train, y_train)

    print("[train] Evaluating on held-out test set...")
    metrics = evaluate_model(model, X_test, y_test)
    metrics["is_synthetic_data"] = is_synthetic
    metrics["n_train"] = int(len(y_train))
    metrics["n_test"] = int(len(y_test))

    print(f"[train] Accuracy: {metrics['accuracy']:.4f}")
    print(f"[train] ROC AUC:  {metrics['roc_auc']:.4f}")
    print(f"[train] Confusion matrix: {metrics['confusion_matrix']}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)
    print(f"[train] Saved trained model to {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Saved metrics to {METRICS_PATH}")

    if is_synthetic:
        print(
            "\n[train] REMINDER: this run used SYNTHETIC validation data. "
            "These metrics do not reflect real-world model performance. "
            "Place the real dataset at data/credit_data.csv and re-run to "
            "get meaningful results.\n"
        )


if __name__ == "__main__":
    main()
