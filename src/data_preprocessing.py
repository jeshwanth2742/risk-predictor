"""
data_preprocessing.py
----------------------
Loads the credit dataset, validates its schema, handles missing values,
scales features, and produces a train/test split.

Expects a CSV at data/credit_data.csv matching the Kaggle "Give Me Some
Credit" schema (target column: SeriousDlqin2yrs). If that file is not
present, falls back to data/sample_synthetic_data.csv for pipeline
validation only -- and says so loudly, rather than silently faking a
real-looking result.

The fitted preprocessing pipeline (imputer + scaler) is saved to
models/preprocessor.joblib so it can be reused identically at inference
time without ever being re-fit on new data.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

TARGET_COL = "SeriousDlqin2yrs"

# Canonical "Give Me Some Credit" feature columns (order matters for the
# saved pipeline and for the API/UI contract).
FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

REAL_DATA_PATH = "data/credit_data.csv"
SYNTHETIC_FALLBACK_PATH = "data/sample_synthetic_data.csv"
PREPROCESSOR_PATH = "models/preprocessor.joblib"


def _resolve_data_path():
    """Prefer the real dataset. Fall back to the synthetic validation file
    only if the real one is missing, and warn clearly either way."""
    if os.path.exists(REAL_DATA_PATH):
        print(f"[data] Using real dataset at {REAL_DATA_PATH}")
        return REAL_DATA_PATH, False

    if os.path.exists(SYNTHETIC_FALLBACK_PATH):
        print(
            "\n"
            "==================== WARNING: NO REAL DATASET FOUND ====================\n"
            f"'{REAL_DATA_PATH}' does not exist.\n"
            f"Falling back to '{SYNTHETIC_FALLBACK_PATH}', which is SYNTHETIC DATA\n"
            "generated only to validate that the pipeline runs end-to-end.\n"
            "Any accuracy/AUC/SHAP results produced from it are NOT meaningful and\n"
            "must not be reported as real model performance.\n"
            "Download the real 'Give Me Some Credit' dataset from Kaggle and place\n"
            f"it at {REAL_DATA_PATH} to get real results.\n"
            "==========================================================================\n"
        )
        return SYNTHETIC_FALLBACK_PATH, True

    raise FileNotFoundError(
        f"Neither '{REAL_DATA_PATH}' nor '{SYNTHETIC_FALLBACK_PATH}' exists. "
        "Place the real dataset at data/credit_data.csv (see README) before "
        "running training."
    )


def load_raw_data(path: str = None) -> tuple[pd.DataFrame, bool]:
    """Load and validate the CSV schema. Returns (dataframe, is_synthetic)."""
    is_synthetic = False
    if path is None:
        path, is_synthetic = _resolve_data_path()

    df = pd.read_csv(path)

    # Drop a stray index column some exports of this dataset include.
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    missing_cols = [c for c in [TARGET_COL] + FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset at '{path}' is missing required columns: {missing_cols}. "
            f"Expected schema: {[TARGET_COL] + FEATURE_COLUMNS}"
        )

    if df[TARGET_COL].isna().any():
        raise ValueError("Target column contains missing values; cannot proceed.")

    print(f"[data] Loaded {len(df)} rows from {path}")
    print(f"[data] Default rate (target=1): {df[TARGET_COL].mean():.3f}")
    print(f"[data] Missing values per column:\n{df[FEATURE_COLUMNS].isna().sum()}")

    return df, is_synthetic


def build_preprocessing_pipeline() -> Pipeline:
    """Median-impute missing values, then standard-scale all features.

    A simple Pipeline (rather than a bare imputer+scaler pair) keeps this
    a single reusable artifact and guarantees imputation is always applied
    before scaling, in both training and inference.
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def preprocess_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save_pipeline: bool = True,
):
    """Fit the preprocessing pipeline on the training split only (to avoid
    leakage), transform both splits, and optionally persist the fitted
    pipeline to disk for reuse at inference time."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_preprocessing_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    if save_pipeline:
        os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
        joblib.dump({"pipeline": pipeline, "feature_columns": FEATURE_COLUMNS}, PREPROCESSOR_PATH)
        print(f"[data] Saved fitted preprocessing pipeline to {PREPROCESSOR_PATH}")

    return X_train_processed, X_test_processed, y_train.to_numpy(), y_test.to_numpy(), pipeline


def load_preprocessor():
    """Load the previously fitted preprocessing pipeline (never re-fit at
    inference time)."""
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"No fitted preprocessor found at {PREPROCESSOR_PATH}. Run "
            "train_model.py first."
        )
    artifact = joblib.load(PREPROCESSOR_PATH)
    return artifact["pipeline"], artifact["feature_columns"]


if __name__ == "__main__":
    df, is_synthetic = load_raw_data()
    X_train, X_test, y_train, y_test, pipeline = preprocess_and_split(df)
    print(f"[data] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    if is_synthetic:
        print("[data] NOTE: this run used SYNTHETIC validation data.")
