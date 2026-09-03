"""
generate_synthetic_sample.py
-----------------------------
THIS FILE IS FOR LOCAL PIPELINE VALIDATION ONLY.

No real dataset (Kaggle "Give Me Some Credit" or UCI German Credit) was
available in this build environment (no internet/Kaggle access). Rather than
silently faking results against a real-looking file, this script generates
a small, CLEARLY LABELED synthetic dataset that matches the "Give Me Some
Credit" column schema, purely so the pipeline (preprocessing -> training ->
SHAP -> API -> UI) can be exercised end-to-end during development.

It writes to data/sample_synthetic_data.csv -- NOT data/credit_data.csv.

TO GET REAL RESULTS: download the actual "Give Me Some Credit" dataset from
Kaggle (https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it at
data/credit_data.csv. The rest of the pipeline requires no code changes.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 4000

age = np.random.randint(21, 75, N)
monthly_income = np.random.lognormal(mean=8.3, sigma=0.6, size=N).round(2)
util = np.clip(np.random.beta(2, 5, N) * 1.3, 0, 2)
debt_ratio = np.clip(np.random.lognormal(mean=-1.0, sigma=1.0, size=N), 0, 5)
open_credit_lines = np.random.poisson(8, N)
real_estate_loans = np.random.poisson(1, N)
dependents = np.random.poisson(0.8, N)
late_30_59 = np.random.poisson(0.3, N)
late_60_89 = np.random.poisson(0.1, N)
late_90 = np.random.poisson(0.1, N)

# introduce some missingness (mirrors real dataset's known NaNs)
income_missing_idx = np.random.choice(N, size=int(N * 0.08), replace=False)
monthly_income = monthly_income.astype(float)
monthly_income[income_missing_idx] = np.nan

dep_missing_idx = np.random.choice(N, size=int(N * 0.03), replace=False)
dependents = dependents.astype(float)
dependents[dep_missing_idx] = np.nan

# synthetic default probability driven by a plausible signal (for demo only)
logit = (
    -3.0
    + 2.5 * util
    + 0.8 * debt_ratio
    + 0.9 * late_30_59
    + 1.4 * late_60_89
    + 1.8 * late_90
    - 0.02 * (age - 40)
    - 0.15 * np.nan_to_num(np.log1p(monthly_income), nan=8.0) + 1.2
)
prob_default = 1 / (1 + np.exp(-logit))
target = (np.random.rand(N) < prob_default).astype(int)

df = pd.DataFrame({
    "SeriousDlqin2yrs": target,
    "RevolvingUtilizationOfUnsecuredLines": util,
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
    "DebtRatio": debt_ratio,
    "MonthlyIncome": monthly_income,
    "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
    "NumberOfTimes90DaysLate": late_90,
    "NumberRealEstateLoansOrLines": real_estate_loans,
    "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
    "NumberOfDependents": dependents,
})

df.to_csv("data/sample_synthetic_data.csv", index=False)
print(f"Wrote {len(df)} synthetic rows to data/sample_synthetic_data.csv")
print(f"Default rate: {target.mean():.3f}")
