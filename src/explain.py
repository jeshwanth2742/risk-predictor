"""
explain.py
-----------
Wraps a SHAP TreeExplainer around the trained XGBoost model to produce
PER-PREDICTION feature attributions (not global feature importance).

For a single applicant's input row, `explain_prediction` returns:
  - the model's predicted probability of default
  - the SHAP base value (expected model output over the training data)
  - every feature's SHAP value + the applicant's raw input value
  - the top N features (by |SHAP value|) driving THIS specific prediction,
    labeled as pushing risk up or down.

This module never fabricates attributions: every number returned comes
directly from shap.TreeExplainer applied to the real trained model.
"""
import os
import joblib
import numpy as np
import pandas as pd
import shap

MODEL_PATH = "models/xgb_model.joblib"


class CreditRiskExplainer:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run "
                "src/train_model.py first."
            )
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_columns = artifact["feature_columns"]
        # TreeExplainer is exact and fast for tree ensembles like XGBoost.
        self.explainer = shap.TreeExplainer(self.model)

    def explain_prediction(self, X_processed_row: np.ndarray, raw_values: dict, top_n: int = 5) -> dict:
        """
        Args:
            X_processed_row: a single row (1D array, already through the
                fitted preprocessing pipeline) in FEATURE_COLUMNS order.
            raw_values: dict of {feature_name: original_input_value} for
                display purposes (pre-scaling), same applicant.
            top_n: how many top contributing features to return (3-5).

        Returns:
            dict with risk_score, base_value, all per-feature shap values,
            and the top_n factors driving this specific prediction.
        """
        X_row = np.asarray(X_processed_row).reshape(1, -1)

        risk_score = float(self.model.predict_proba(X_row)[0, 1])

        shap_values = self.explainer.shap_values(X_row)
        # shap_values shape: (1, n_features) for binary XGBClassifier with TreeExplainer
        shap_row = np.asarray(shap_values).reshape(-1)
        base_value = float(np.asarray(self.explainer.expected_value).reshape(-1)[0])

        all_factors = []
        for feature_name, shap_val in zip(self.feature_columns, shap_row):
            all_factors.append({
                "feature": feature_name,
                "value": raw_values.get(feature_name),
                "shap_value": float(shap_val),
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
            })

        top_factors = sorted(all_factors, key=lambda f: abs(f["shap_value"]), reverse=True)[:top_n]

        return {
            "risk_score": risk_score,
            "base_value": base_value,
            "all_factors": all_factors,
            "top_factors": top_factors,
        }


def _demo():
    """Quick smoke test using one row from the (real or synthetic) dataset."""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_preprocessing import load_raw_data, load_preprocessor, FEATURE_COLUMNS

    df, is_synthetic = load_raw_data()
    pipeline, feature_columns = load_preprocessor()

    sample_raw = df[feature_columns].iloc[[0]]
    sample_processed = pipeline.transform(sample_raw)[0]
    raw_values = sample_raw.iloc[0].to_dict()

    explainer = CreditRiskExplainer()
    result = explainer.explain_prediction(sample_processed, raw_values, top_n=5)

    print(f"Risk score (P[default]): {result['risk_score']:.4f}")
    print(f"Base value: {result['base_value']:.4f}")
    print("Top factors for THIS applicant:")
    for f in result["top_factors"]:
        arrow = "UP  " if f["shap_value"] > 0 else "DOWN"
        print(f"  [{arrow}] {f['feature']} = {f['value']} -> shap={f['shap_value']:+.4f}")

    if is_synthetic:
        print("\nNOTE: ran against SYNTHETIC validation data.")


if __name__ == "__main__":
    _demo()
