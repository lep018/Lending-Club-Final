import pickle

# Define paths for the models and scaler (adjust if needed)
LOGREG_MODEL_PATH = 'logistic_regression_model.pkl'
LASSO_MODEL_PATH = 'lassocv_regressor_model.pkl'
SCALER_PATH = 'scaler_reduced.pkl' # Assuming you've saved the scaler as well

import streamlit as st
import pandas as pd
import numpy as np

# --- Bucknell Themed Styling ---
# Official Bucknell University Colors
bucknell_orange = "#E25822"
bucknell_navy = "#041E42"

st.set_page_config(
    page_title="Bucknell Lending Club Loan Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.bucknell.edu/academics/academic-departments-programs/analytics-operations-management/faculty',
        'Report a bug': "https://www.bucknell.edu/",
        'About': "# Loan Default and Return Rate Prediction App for Bucknell Lending Club.\nDeveloped by Laura Posh and Scarlet Kashuba for ANOP330."
    }
)

# Custom CSS for Bucknell theme
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
    color: {bucknell_navy}; /* Bucknell Navy for headers */
}}
.stButton>button {{
    background-color: {bucknell_orange}; /* Bucknell Orange for buttons */
    color: white;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    border: 1px solid {bucknell_orange};
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
    background-color: #A30000; /* A darker red for rejection */
    color: white;
}}
</style>
""", unsafe_allow_html=True)

# --- Load Models and Scaler ---
@st.cache_resource # Cache the models to avoid reloading on each rerun
def load_models():
    """Loads the pre-trained models and scaler."""
    try:
        with open(LOGREG_MODEL_PATH, 'rb') as f:
            logreg_model = pickle.load(f)
        with open(LASSO_MODEL_PATH, 'rb') as f:
            lasso_model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return logreg_model, lasso_model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model or scaler file not found. Please ensure {LOGREG_MODEL_PATH}, {LASSO_MODEL_PATH}, and {SCALER_PATH} are in the same directory as the app.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        st.stop()

logreg_model, lasso_model, scaler = load_models()

# --- Hardcoded Averages and Cutoff from Training ---
# These values were derived from the training data in the notebook
avg_return_fully_paid = 4.612149501658192
avg_return_not_fully_paid = -12.537793299762876
custom_cutoff_logreg = 0.55 # from the notebook analysis for best classifier

# The 15 features used for retraining (maintain specific order)
combined_15_features = [
    'credit_age', 'int_rate', 'verification_status_Source Verified', 'term_num',
    'fico_avg', 'loan_amnt', 'home_ownership_RENT', 'home_ownership_MORTGAGE',
    'revol_util', 'grade_B', 'log_annual_inc', 'grade_D', 'grade_C', 'grade_E', 'dti'
]

# --- Streamlit UI: Title and Introduction ---
st.title("💰 Bucknell Lending Club: Loan Risk Assessment")
st.markdown("---")
st.markdown("""
Welcome to the Bucknell Lending Club Loan Risk Assessment tool.
This application leverages machine learning models to help evaluate potential loan applications by predicting:
1.  **Propensity that the borrower's loan will be 'Fully Paid'**: The likelihood that a borrower will repay their loan in full.
2.  **Predicted Pessimistic Annualized Return**: An estimate of the annualized return under a pessimistic scenario, given all features.
3.  **Expected Pessimistic Annualized Return**: A probability-weighted estimate of the return, combining repayment probability with historical average returns for paid and defaulted loans.
4.  **Recommended Action**: A clear 'Approve' or 'Reject' decision based on the model's insights.
""")
st.markdown("---")

# --- User Input Section ---
st.header("Borrower Information")

col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000, step=500)
    int_rate = st.number_input("Interest Rate (%)", min_value=0.1, max_value=30.0, value=9.9, step=0.1, format="%.1f")
    fico_avg = st.number_input("FICO Score (Average)", min_value=300, max_value=850, value=675, step=1)
    dti = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=50.0, value=15.0, step=0.1, format="%.1f")
    revol_util = st.number_input("Revolving Line Utilization (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f")
    annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=70000, step=1000)

with col2:
    credit_age = st.number_input("Credit History Length (Years)", min_value=0.1, max_value=60.0, value=10.0, step=0.1, format="%.1f")
    term_num_option = st.selectbox("Loan Term (Months)", options=[36, 60], index=0)

    home_ownership_option = st.selectbox(
        "Home Ownership Status",
        options=['MORTGAGE', 'RENT', 'OWN', 'ANY', 'OTHER', 'NONE'],
        index=0 # Default to MORTGAGE
    )

    grade_option = st.selectbox(
        "Loan Grade",
        options=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        index=2 # Default to C
    )

    verification_status_option = st.selectbox(
        "Income Verification Status",
        options=['Verified', 'Source Verified', 'Not Verified'],
        index=1 # Default to Source Verified
    )


st.markdown("---")

# --- Prediction Button ---
if st.button("Analyze Loan Application"):
    # 1. Initialize a dictionary for the input features with all required columns set to 0
    input_data = {feature: 0 for feature in combined_15_features}

    # 2. Populate numerical features
    input_data['loan_amnt'] = loan_amnt
    input_data['int_rate'] = int_rate
    input_data['fico_avg'] = fico_avg
    input_data['dti'] = dti
    input_data['revol_util'] = revol_util
    input_data['credit_age'] = credit_age

    # 3. Apply transformations
    # Handle case where annual_inc might be 0, though min_value is 10000.
    # np.log1p(x) is equivalent to log(1+x), handling 0 gracefully if needed
    input_data['log_annual_inc'] = np.log1p(annual_inc)

    # 4. Handle categorical features (dummy variables)
    input_data['term_num'] = term_num_option # Directly use numerical term_num

    if home_ownership_option == 'RENT':
        input_data['home_ownership_RENT'] = 1
    elif home_ownership_option == 'MORTGAGE':
        input_data['home_ownership_MORTGAGE'] = 1
    # Other home_ownership options (OWN, ANY, OTHER, NONE) result in both dummies being 0

    if grade_option == 'B':
        input_data['grade_B'] = 1
    elif grade_option == 'C':
        input_data['grade_C'] = 1
    elif grade_option == 'D':
        input_data['grade_D'] = 1
    elif grade_option == 'E':
        input_data['grade_E'] = 1
    # Grades A, F, G result in all grade dummies being 0

    if verification_status_option == 'Source Verified':
        input_data['verification_status_Source Verified'] = 1
    # Other verification_status options (Verified, Not Verified) result in dummy being 0

    # Create a DataFrame from the processed input data, ensuring column order
    input_df = pd.DataFrame([input_data], columns=combined_15_features)

    # 5. Scale the input data using the pre-fitted scaler
    scaled_input_df = scaler.transform(input_df)

    # --- Make Predictions ---
    # Logistic Regression (Classifier)
    # P(Fully Paid) is the probability of the positive class (index 1)
    p_fully_paid = logreg_model.predict_proba(scaled_input_df)[:, 1][0]
    p_not_fully_paid = 1 - p_fully_paid

    # LassoCV (Regressor) - Predicted Pessimistic Return
    predicted_ret_PESS = lasso_model.predict(scaled_input_df)[0]

    # Calculate Expected Pessimistic Annualized Return
    expected_ret_PESS = (p_fully_paid * avg_return_fully_paid) + \
                        (p_not_fully_paid * avg_return_not_fully_paid)

    # --- Display Results ---
    st.header("Prediction Results")

    st.markdown(
        f"**Propensity that the borrower's loan will be fully paid:** <span style='color:{bucknell_navy}'>{p_fully_paid:.2%}</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"**Predicted Pessimistic Annualized Return:** <span style='color:{bucknell_navy}'>{predicted_ret_PESS:.2f}%</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"**Expected Pessimistic Annualized Return:** <span style='color:{bucknell_navy}'>{expected_ret_PESS:.2f}%</span>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.header("Recommended Action")

    if p_fully_paid >= custom_cutoff_logreg and expected_ret_PESS > 0:
        st.success(f"**Recommendation: APPROVE LOAN**")
        st.info("Based on the models, this loan has a high likelihood of being fully paid and a positive expected return. Proceed with caution and further due diligence.")
    else:
        st.error(f"**Recommendation: REJECT LOAN**")
        st.info("Based on the models, this loan has a lower likelihood of being fully paid or a negative expected return, indicating higher risk. Consider rejecting or requiring more stringent conditions.")

    st.markdown("---")
    st.subheader("Prediction Breakdown (Probabilities)")
    # Reconstruct DataFrame to explicitly include category and color for st.bar_chart
    probabilities_data = {
        'Category': ['Fully Paid', 'Not Fully Paid'],
        'Probability': [p_fully_paid, p_not_fully_paid],
        'Color': [bucknell_orange, bucknell_navy]
    }
    probabilities_df = pd.DataFrame(probabilities_data)

    # Plot using x, y, and color arguments referencing DataFrame columns
    st.bar_chart(probabilities_df, x='Category', y='Probability', color='Color')

st.markdown("---")
st.markdown("**Developed by Laura Posh and Scarlet Kashuba** | Powered by Streamlit")
