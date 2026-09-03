"""
app.py
-------
Streamlit frontend for the Explainable Credit Risk Predictor.

Presents a form for applicant details, calls the FastAPI backend's
/predict endpoint, and displays the risk score plus a SHAP-based
waterfall/bar chart showing exactly which factors pushed THIS
applicant's score up or down.

Run (with the API already running separately):
    streamlit run app.py

NOTE: This file only redesigns the presentation layer. All ML/API logic,
payload construction, response handling, and SHAP computation calls are
functionally identical to the original implementation.
"""
import os
import requests
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

API_URL = os.environ.get("CREDIT_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Explainable Credit Risk Predictor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Global styling — fintech / SaaS dashboard aesthetic
# --------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 1200px;
    }

    :root {
        --ink: #0f172a;
        --sub: #64748b;
        --line: #e2e8f0;
        --card: #ffffff;
        --accent: #4338ca;
        --accent-soft: #eef2ff;
        --danger: #dc2626;
        --danger-soft: #fef2f2;
        --warn: #d97706;
        --warn-soft: #fffbeb;
        --good: #059669;
        --good-soft: #ecfdf5;
    }

    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 22px;
        background: var(--ink);
        border-radius: 14px;
        margin-bottom: 16px;
    }
    .app-header .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-header .mark {
        width: 34px; height: 34px;
        border-radius: 9px;
        background: linear-gradient(135deg, #6366f1, #4338ca);
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; color: white; font-size: 15px;
    }
    .app-header h1 {
        color: white; font-size: 16px; font-weight: 700; margin: 0; letter-spacing: 0.2px;
    }
    .app-header .sub {
        color: #94a3b8; font-size: 11.5px; margin: 0; font-weight: 500;
    }
    .app-header .tag {
        background: rgba(99,102,241,0.18);
        color: #a5b4fc;
        font-size: 10.5px;
        font-weight: 600;
        padding: 5px 10px;
        border-radius: 20px;
        border: 1px solid rgba(99,102,241,0.3);
        letter-spacing: 0.3px;
    }

    .panel {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .panel-title {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: var(--sub);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }

    .stNumberInput input, .stSlider {
        font-size: 13.5px;
    }
    label[data-testid="stWidgetLabel"] p {
        font-size: 12.5px !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: var(--ink) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 0 !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        letter-spacing: 0.3px;
        transition: all 0.15s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--accent) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(67,56,202,0.28);
    }

    .score-card {
        border-radius: 16px;
        padding: 22px 24px;
        text-align: center;
        border: 1px solid var(--line);
    }
    .score-card .score-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 42px;
        font-weight: 700;
        line-height: 1;
        margin: 6px 0;
    }
    .score-card .score-lbl {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: var(--sub);
    }
    .band-pill {
        display: inline-block;
        margin-top: 8px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    .factor-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 10px;
        border-radius: 8px;
        margin-bottom: 4px;
        font-size: 13px;
        background: #f8fafc;
    }
    .factor-row .fname { font-weight: 600; color: var(--ink); }
    .factor-row .fval { color: var(--sub); font-size: 11.5px; font-family: 'JetBrains Mono', monospace; }
    .factor-row .fshap { font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 12.5px; }

    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: var(--sub);
    }
    .empty-state .icon { font-size: 34px; margin-bottom: 10px; opacity: 0.5; }
    .empty-state h3 { color: var(--ink); font-size: 15px; margin-bottom: 4px; }
    .empty-state p { font-size: 12.5px; max-width: 340px; margin: 0 auto; }

    .footnote {
        font-size: 11px;
        color: var(--sub);
        text-align: center;
        margin-top: 10px;
        padding: 8px;
        background: var(--warn-soft);
        border-radius: 8px;
        border: 1px solid #fde68a;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="brand">
        <div class="mark">◈</div>
        <div>
            <h1>Explainable Credit Risk Predictor</h1>
            <p class="sub">XGBoost model &nbsp;·&nbsp; live SHAP explanations</p>
        </div>
    </div>
    <div class="tag">MODEL-DRIVEN · NOT HARDCODED</div>
</div>
""", unsafe_allow_html=True)

input_col, result_col = st.columns([0.42, 0.58], gap="medium")

# --------------------------------------------------------------------------
# LEFT: Applicant input form
# --------------------------------------------------------------------------
with input_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">① Applicant Profile</div>', unsafe_allow_html=True)

    with st.form("applicant_form"):
        tab1, tab2 = st.tabs(["Personal & Income", "Credit History"])

        with tab1:
            age = st.number_input("Age", min_value=18, max_value=110, value=35)
            monthly_income = st.number_input(
                "Monthly Income ($)", min_value=0.0, value=4000.0, step=100.0,
                help="Leave at 0 if unknown; the model will impute a typical value."
            )
            dependents = st.number_input("Number of Dependents", min_value=0, value=0)
            debt_ratio = st.number_input(
                "Debt Ratio (monthly debt / monthly income)", min_value=0.0,
                value=0.3, step=0.01
            )

        with tab2:
            util = st.slider(
                "Revolving Credit Utilization (balance / limit)", min_value=0.0, max_value=2.0,
                value=0.3, step=0.01,
                help="Total balance on credit cards/lines divided by credit limits."
            )
            open_credit_lines = st.number_input("Open Credit Lines & Loans", min_value=0, value=6)
            real_estate_loans = st.number_input("Real Estate Loans / Lines", min_value=0, value=1)

            c1, c2, c3 = st.columns(3)
            with c1:
                late_30_59 = st.number_input("30-59d Late", min_value=0, value=0)
            with c2:
                late_60_89 = st.number_input("60-89d Late", min_value=0, value=0)
            with c3:
                late_90 = st.number_input("90d+ Late", min_value=0, value=0)

        submitted = st.form_submit_button("Assess Risk →", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Call API (unchanged logic)
# --------------------------------------------------------------------------
result = None
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
        with result_col:
            with st.spinner("Scoring applicant..."):
                resp = requests.post(
    f"{API_URL}/predict",
    json=payload,
    timeout=60
)
                resp.raise_for_status()
                result = resp.json()
    except requests.exceptions.ConnectionError:
        with result_col:
            st.error(
                f"Could not reach the API at {API_URL}. Make sure it's running:\n\n"
                "`uvicorn api:app --reload --port 8000`"
            )
        st.stop()
    except requests.exceptions.HTTPError as e:
        with result_col:
            st.error(f"API returned an error: {e}\n\n{resp.text}")
        st.stop()

# --------------------------------------------------------------------------
# RIGHT: Results
# --------------------------------------------------------------------------
with result_col:
    if result is None:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">② Risk Assessment</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="empty-state">
            <div class="icon">◈</div>
            <h3>No assessment yet</h3>
            <p>Fill in the applicant profile on the left and click
            "Assess Risk" to generate a live prediction with a full
            SHAP-based explanation.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        risk_score = result["risk_score"]
        risk_band = result["risk_band"]

        band_style = {
            "Low":    {"bg": "var(--good-soft)",   "fg": "var(--good)",   "border": "#a7f3d0"},
            "Medium": {"bg": "var(--warn-soft)",   "fg": "var(--warn)",   "border": "#fde68a"},
            "High":   {"bg": "var(--danger-soft)", "fg": "var(--danger)", "border": "#fecaca"},
        }.get(risk_band, {"bg": "#f1f5f9", "fg": "#475569", "border": "#e2e8f0"})

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">② Risk Assessment</div>', unsafe_allow_html=True)

        m1, m2 = st.columns([0.5, 0.5])
        with m1:
            st.markdown(f"""
            <div class="score-card" style="background:{band_style['bg']}; border-color:{band_style['border']};">
                <div class="score-lbl">Default Probability</div>
                <div class="score-num" style="color:{band_style['fg']};">{risk_score:.1%}</div>
                <span class="band-pill" style="background:{band_style['fg']}; color:white;">{risk_band} Risk</span>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown('<div style="padding-top:6px;">', unsafe_allow_html=True)
            st.caption("Model baseline (average applicant)")
            st.progress(min(risk_score, 1.0))
            st.markdown(
                f"<div style='font-size:11.5px;color:var(--sub);'>"
                f"Baseline: {result['base_value']:.1%} &nbsp;→&nbsp; This applicant: "
                f"<b style='color:{band_style['fg']}'>{risk_score:.1%}</b></div>",
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Explanation panel
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">③ Why This Score — SHAP Explanation</div>', unsafe_allow_html=True)
        st.caption(
            "Each bar shows how much that specific feature value pushed THIS "
            "applicant's predicted risk up (red) or down (blue), relative to "
            f"the model baseline of {result['base_value']:.1%}."
        )

        top_factors = result["top_factors"]
        features = [f["feature"] for f in top_factors][::-1]
        shap_vals = [f["shap_value"] for f in top_factors][::-1]
        raw_vals = [f["value"] for f in top_factors][::-1]
        colors = ["#dc2626" if v > 0 else "#4338ca" for v in shap_vals]

        fig, ax = plt.subplots(figsize=(6.2, 0.5 * len(features) + 0.8))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        bars = ax.barh(features, shap_vals, color=colors, height=0.6)
        ax.axvline(0, color="#94a3b8", linewidth=0.8)
        ax.set_xlabel("SHAP value (log-odds impact)", fontsize=9, color="#64748b")
        ax.tick_params(labelsize=9, colors="#334155")
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#cbd5e1")

        for bar, raw_val in zip(bars, raw_vals):
            width = bar.get_width()
            ax.text(
                width + (0.02 if width >= 0 else -0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{raw_val}",
                va="center",
                ha="left" if width >= 0 else "right",
                fontsize=8.5,
                color="#475569",
            )

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        with st.expander("See all feature contributions"):
            all_factors = sorted(result["all_factors"], key=lambda f: abs(f["shap_value"]), reverse=True)
            for f in all_factors:
                is_up = f["shap_value"] > 0
                arrow = "▲" if is_up else "▼"
                color = "#dc2626" if is_up else "#4338ca"
                st.markdown(f"""
                <div class="factor-row">
                    <div>
                        <span class="fname">{f['feature']}</span>
                        <span class="fval">&nbsp;= {f['value']}</span>
                    </div>
                    <div class="fshap" style="color:{color};">{arrow} {f['shap_value']:+.4f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:11px;color:var(--sub);margin-top:8px;'>"
            "Generated fresh per prediction via <code>shap.TreeExplainer</code> "
            "on the trained XGBoost model — not a static importance ranking."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footnote">
⚠️ If the backend is running against the bundled synthetic validation dataset
(no <code>data/credit_data.csv</code> present), scores and explanations reflect
that synthetic data only — for pipeline demonstration purposes, not real
credit risk assessment.
</div>
""", unsafe_allow_html=True)
