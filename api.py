"""
api.py
-------
FastAPI backend exposing a single /predict endpoint that:
  1. validates applicant features via a Pydantic model,
  2. runs them through the SAVED (not re-fit) preprocessing pipeline,
  3. gets a risk score from the SAVED trained XGBoost model,
  4. computes SHAP-based per-prediction explanations,
  5. returns {risk_score, base_value, top_factors, all_factors}.

Run:
    uvicorn api:app --reload --port 8000
"""
import os
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_preprocessing import load_preprocessor, FEATURE_COLUMNS
from src.explain import CreditRiskExplainer

ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the saved preprocessing pipeline and trained model + SHAP
    # explainer ONCE at startup, not per-request.
    try:
        pipeline, feature_columns = load_preprocessor()
        explainer = CreditRiskExplainer()
        ml_artifacts["pipeline"] = pipeline
        ml_artifacts["feature_columns"] = feature_columns
        ml_artifacts["explainer"] = explainer
        print("[api] Model, preprocessor, and SHAP explainer loaded successfully.")
    except FileNotFoundError as e:
        # Defer the error to request time with a clear message, rather than
        # crashing silently on import.
        ml_artifacts["load_error"] = str(e)
        print(f"[api] WARNING at startup: {e}")
    yield
    ml_artifacts.clear()


app = FastAPI(
    title="Explainable Credit Risk API",
    description="Predicts probability of loan default and explains each "
                 "prediction with per-applicant SHAP feature attributions.",
    version="1.0.0",
    lifespan=lifespan,
)


class ApplicantFeatures(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0, description="Total balance on credit cards / lines relative to credit limits"
    )
    age: int = Field(..., ge=18, le=110, description="Applicant age in years")
    NumberOfTime30to59DaysPastDueNotWorse: int = Field(
        ..., ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse",
        description="Number of times 30-59 days past due (not worse), last 2 years"
    )
    DebtRatio: float = Field(..., ge=0, description="Monthly debt payments / monthly gross income")
    MonthlyIncome: float | None = Field(None, ge=0, description="Monthly income (may be null/unknown)")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, description="Number of open loans/credit lines")
    NumberOfTimes90DaysLate: int = Field(..., ge=0, description="Number of times 90+ days late")
    NumberRealEstateLoansOrLines: int = Field(..., ge=0, description="Number of mortgage/real estate loans")
    NumberOfTime60to89DaysPastDueNotWorse: int = Field(
        ..., ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse",
        description="Number of times 60-89 days past due (not worse), last 2 years"
    )
    NumberOfDependents: float | None = Field(None, ge=0, description="Number of dependents (may be null/unknown)")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.45,
                "age": 42,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.35,
                "MonthlyIncome": 5500,
                "NumberOfOpenCreditLinesAndLoans": 7,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2,
            }
        }


class FeatureFactor(BaseModel):
    feature: str
    value: float | None
    shap_value: float
    direction: str


class PredictResponse(BaseModel):
    risk_score: float
    risk_band: str
    base_value: float
    top_factors: list[FeatureFactor]
    all_factors: list[FeatureFactor]


def _risk_band(score: float) -> str:
    if score < 0.10:
        return "Low"
    elif score < 0.30:
        return "Medium"
    else:
        return "High"


@app.get("/")
def root():
    return {
        "service": "Explainable Credit Risk API",
        "endpoints": {"/predict": "POST applicant features -> risk score + SHAP explanation"},
        "docs": "/docs",
    }


@app.get("/health")
def health():
    ready = "explainer" in ml_artifacts
    return {"status": "ok" if ready else "not_ready", "model_loaded": ready}


@app.post("/predict", response_model=PredictResponse)
def predict(applicant: ApplicantFeatures):
    if "explainer" not in ml_artifacts:
        raise HTTPException(
            status_code=503,
            detail=(
                ml_artifacts.get("load_error")
                or "Model not loaded. Run src/train_model.py first, then restart the API."
            ),
        )

    pipeline = ml_artifacts["pipeline"]
    feature_columns = ml_artifacts["feature_columns"]
    explainer: CreditRiskExplainer = ml_artifacts["explainer"]

    raw_dict = applicant.model_dump(by_alias=True)
    raw_row = pd.DataFrame([raw_dict])[feature_columns]

    try:
        processed_row = pipeline.transform(raw_row)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to preprocess input: {e}")

    result = explainer.explain_prediction(processed_row, raw_row.iloc[0].to_dict(), top_n=5)

    return PredictResponse(
        risk_score=result["risk_score"],
        risk_band=_risk_band(result["risk_score"]),
        base_value=result["base_value"],
        top_factors=result["top_factors"],
        all_factors=result["all_factors"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
