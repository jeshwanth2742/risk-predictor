

"""
import os
import requests
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

API_URL = os.environ.get("CREDIT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Explainable Credit Risk Predictor", layout="centered")

st.title("📊 Explainable Credit Risk / Loan Default Predictor")
st.caption(
    "Enter applicant details below. The risk score and every explanation "
    "factor come directly from a trained XGBoost model and real SHAP "
    "values computed for this specific applicant — nothing here is "
    "hardcoded."
)

with st.form("applicant_form"):
    st.subheader("Applicant Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=110, value=35)
        monthly_income = st.number_input(
            "Monthly Income ($)", min_value=0.0, value=4000.0, step=100.0,
            help="Leave at 0 if unknown; the model will impute a typical value."
        )
        util = st.slider(
            "Revolving Credit Utilization (balance / limit)", min_value=0.0, max_value=2.0,
            value=0.3, step=0.01,
            help="Total balance on credit cards/lines divided by credit limits."
        )
        debt_ratio = st.number_input(
            "Debt Ratio (monthly debt payments / monthly income)", min_value=0.0,
            value=0.3, step=0.01
        )
        dependents = st.number_input("Number of Dependents", min_value=0, value=0)

    with col2:
        open_credit_lines = st.number_input("Open Credit Lines & Loans", min_value=0, value=6)
        real_estate_loans = st.number_input("Real Estate Loans / Lines", min_value=0, value=1)
        late_30_59 = st.number_input("Times 30-59 Days Past Due (last 2y)", min_value=0, value=0)
        late_60_89 = st.number_input("Times 60-89 Days Past Due (last 2y)", min_value=0, value=0)
        late_90 = st.number_input("Times 90+ Days Late (last 2y)", min_value=0, value=0)

    submitted = st.form_submit_button("Assess Risk", use_container_width=True)

if submitted:
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": util,
        "age": int(age),
        "NumberOfTime30-59DaysPastDueNotWorse": int(late_30_59),
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income if monthly_income > 0 else None,
        "NumberOfOpenCreditLinesAndLoans": int(open_credit_lines),
        "NumberOfTimes90DaysLate": int(late_90),
        "NumberRealEstateLoansOrLines": int(real_estate_loans),
        "NumberOfTime60-89DaysPastDueNotWorse": int(late_60_89),
        "NumberOfDependents": dependents,
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not reach the API at {API_URL}. Make sure it's running:\n\n"
            "`uvicorn api:app --reload --port 8000`"
        )
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e}\n\n{resp.text}")
        st.stop()

    st.divider()
    st.subheader("Result")

    risk_score = result["risk_score"]
    risk_band = result["risk_band"]

    band_color = {"Low": "green", "Medium": "orange", "High": "red"}.get(risk_band, "gray")

    score_col, band_col = st.columns(2)
    with score_col:
        st.metric("Predicted Probability of Default", f"{risk_score:.1%}")
    with band_col:
        st.markdown(f"### Risk Band: :{band_color}[{risk_band}]")

    st.progress(min(risk_score, 1.0))

    st.subheader("Why this score? (SHAP explanation for this applicant)")
    st.caption(
        "Each bar shows how much that specific feature value pushed THIS "
        "applicant's predicted risk up (red, toward default) or down "
        "(blue, toward non-default), relative to the model's average "
        f"baseline prediction of {result['base_value']:.1%}."
    )

    top_factors = result["top_factors"]
    features = [f["feature"] for f in top_factors][::-1]
    shap_vals = [f["shap_value"] for f in top_factors][::-1]
    raw_vals = [f["value"] for f in top_factors][::-1]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in shap_vals]

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(features) + 1))
    bars = ax.barh(features, shap_vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on predicted default probability, log-odds space)")
    ax.set_title("Top factors driving this prediction")

    for bar, raw_val in zip(bars, raw_vals):
        width = bar.get_width()
        label = f"{raw_val}"
        ax.text(
            width + (0.02 if width >= 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"value={label}",
            va="center",
            ha="left" if width >= 0 else "right",
            fontsize=9,
        )

    st.pyplot(fig)

    with st.expander("See all feature contributions"):
        all_factors = sorted(result["all_factors"], key=lambda f: abs(f["shap_value"]), reverse=True)
        for f in all_factors:
            arrow = "🔺 increases risk" if f["shap_value"] > 0 else "🔻 decreases risk"
            st.write(f"**{f['feature']}** = {f['value']}  →  {arrow} (SHAP = {f['shap_value']:+.4f})")

    st.caption(
        "This explanation is generated fresh for every prediction using "
        "shap.TreeExplainer on the actual trained XGBoost model — it is "
        "not a static or precomputed importance ranking."
    )

st.divider()
st.caption(
    "⚠️ If the backend is currently running against the bundled synthetic "
    "validation dataset (no data/credit_data.csv present), scores and "
    "explanations reflect that synthetic data only and are for pipeline "
    "demonstration purposes, not real credit risk assessment."
)
