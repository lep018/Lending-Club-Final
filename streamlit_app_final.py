import pickle

# Define paths for the models and scaler (adjust if needed)
LOGREG_MODEL_PATH = 'logistic_regression_model.pkl'
LASSO_MODEL_PATH = 'lassocv_regressor_model.pkl'
SCALER_PATH = 'scaler_reduced.pkl'

import streamlit as st
import pandas as pd
import numpy as np

# --- Bucknell Themed Styling ---
bucknell_orange = "#E25822"
bucknell_navy = "#041E42"

st.set_page_config(
    page_title="Loan Approval & Return Forecasting Tool",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.bucknell.edu/academics/academic-departments-programs/analytics-operations-management/faculty',
        'Report a bug': "https://www.bucknell.edu/",
        'About': "# Loan Default and Return Rate Prediction App for Bucknell Lending Club.\nDeveloped by Laura Posh and Scarlet Kashuba for ANOP330."
    }
)

# Custom CSS
st.markdown(f"""
<style>
.reportview-container .main .block-container{{
    max-width: 1200px;
    padding-top: 2rem;
    padding-right: 2rem;
    padding-left: 2rem;
    padding-bottom: 2rem;
}}
h1, h2, h3, h4, h5, h6 {{
    color: {bucknell_navy};
}}
.stButton>button {{
    background-color: {bucknell_orange};
    color: white;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    border: 1px solid {bucknell_orange};
}}
.stButton>button:hover {{
    background-color: {bucknell_navy};
    color: white;
    border: 1px solid {bucknell_navy};
}}
.stAlert {{
    border-radius: 5px;
}}
.stAlert.info {{
    background-color: {bucknell_navy};
    color: white;
}}
.stAlert.success {{
    background-color: {bucknell_orange};
    color: white;
}}
.stAlert.error {{
    background-color: #A30000;
    color: white;
}}
</style>
""", unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        with open(LOGREG_MODEL_PATH, 'rb') as f:
            logreg_model = pickle.load(f)
        with open(LASSO_MODEL_PATH, 'rb') as f:
            lasso_model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return logreg_model, lasso_model, scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

logreg_model, lasso_model, scaler = load_models()

# --- Constants ---
avg_return_fully_paid = 4.612149501658192
avg_return_not_fully_paid = -12.537793299762876
custom_cutoff_logreg = 0.55

combined_15_features = [
    'credit_age', 'int_rate', 'verification_status_Source Verified', 'term_num',
    'fico_avg', 'loan_amnt', 'home_ownership_RENT', 'home_ownership_MORTGAGE',
    'revol_util', 'grade_B', 'log_annual_inc', 'grade_D', 'grade_C', 'grade_E', 'dti'
]

# =========================================================
# 🔷 HERO HEADER
# =========================================================

st.markdown(f"""
<div style="background-color: {bucknell_navy}; padding: 30px; border-radius: 10px; margin-bottom: 20px;">
    <h1 style="color: white; margin-bottom: 5px;">
        Loan Decision & Return Forecasting Dashboard
    </h1>
    <h4 style="color: #d3d3d3; margin-top: 0;">
        Built for Bucknell Lending Club
    </h4>
    <p style="color: white; font-size: 16px; margin-top: 15px;">
        A data-driven decision support tool that evaluates loan applications using machine learning to estimate risk, return, and repayment probability.
    </p>
</div>
""", unsafe_allow_html=True)

# Feature highlight cards (UPDATED - removed Expected Return)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center;">
        <h5 style="color:{bucknell_navy}; margin-bottom:5px;">Repayment Probability</h5>
        <p style="font-size: 13px; margin:0;">Likelihood loan is fully paid</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center;">
        <h5 style="color:{bucknell_navy}; margin-bottom:5px;">Predicted Return</h5>
        <p style="font-size: 13px; margin:0;">Pessimistic annualized return</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center;">
        <h5 style="color:{bucknell_navy}; margin-bottom:5px;">Decision Output</h5>
        <p style="font-size: 13px; margin:0;">Approve or reject recommendation</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Inputs ---
st.header("Borrower Information")

col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Loan Amount ($)", 500, 40000, 10000, 500)
    int_rate = st.number_input("Interest Rate (%)", 0.1, 30.0, 9.9, 0.1)
    fico_avg = st.number_input("FICO Score", 300, 850, 675)
    dti = st.number_input("DTI", 0.0, 50.0, 15.0, 0.1)
    revol_util = st.number_input("Revolving Utilization (%)", 0.0, 100.0, 50.0, 0.1)
    annual_inc = st.number_input("Annual Income ($)", 10000, 1000000, 70000, 1000)

with col2:
    credit_age = st.number_input("Credit Age (Years)", 0.1, 60.0, 10.0, 0.1)
    term_num_option = st.selectbox("Loan Term", [36, 60])
    home_ownership_option = st.selectbox("Home Ownership", ['MORTGAGE', 'RENT', 'OWN'])
    grade_option = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E'])
    verification_status_option = st.selectbox("Verification", ['Verified', 'Source Verified', 'Not Verified'])

st.markdown("---")

# --- Predict ---
if st.button("Analyze Loan Application"):

    input_data = {feature: 0 for feature in combined_15_features}

    input_data.update({
        'loan_amnt': loan_amnt,
        'int_rate': int_rate,
        'fico_avg': fico_avg,
        'dti': dti,
        'revol_util': revol_util,
        'credit_age': credit_age,
        'log_annual_inc': np.log1p(annual_inc),
        'term_num': term_num_option
    })

    if home_ownership_option == 'RENT':
        input_data['home_ownership_RENT'] = 1
    elif home_ownership_option == 'MORTGAGE':
        input_data['home_ownership_MORTGAGE'] = 1

    if grade_option == 'B':
        input_data['grade_B'] = 1
    elif grade_option == 'C':
        input_data['grade_C'] = 1
    elif grade_option == 'D':
        input_data['grade_D'] = 1
    elif grade_option == 'E':
        input_data['grade_E'] = 1

    if verification_status_option == 'Source Verified':
        input_data['verification_status_Source Verified'] = 1

    input_df = pd.DataFrame([input_data], columns=combined_15_features)
    scaled_input_df = scaler.transform(input_df)

    # Predictions
    p_fully_paid = logreg_model.predict_proba(scaled_input_df)[:, 1][0]
    predicted_ret = lasso_model.predict(scaled_input_df)[0]

    # --- Results Display ---
    st.header("Loan Decision Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style='text-align:center;'>
            <h4>Probability Loan is Fully Paid</h4>
            <h1 style='color:{bucknell_orange}; font-size:48px;'>{p_fully_paid:.0%}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='text-align:center;'>
            <h4>Predicted Pessimistic Return</h4>
            <h1 style='color:{bucknell_orange}; font-size:48px;'>{predicted_ret:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Recommendation ---
    st.header("Recommended Action")

    if p_fully_paid >= custom_cutoff_logreg:
        st.success("**APPROVE LOAN**")
        st.info("High probability of repayment.")
    else:
        st.error("**REJECT LOAN**")
        st.info("Lower repayment likelihood.")

st.markdown("---")
st.markdown("Developed by Laura Posh and Scarlet Kashuba")
