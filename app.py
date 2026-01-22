import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==========================================
# 1. SETUP & LOAD RESOURCES
# ==========================================
st.set_page_config(page_title="Smart Loan Approval", page_icon="🏦")

# Load the saved "Brain" and "Translators"
try:
    model = joblib.load('models/svm_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le_self_emp = joblib.load('models/le_self_emp.pkl')
except FileNotFoundError:
    st.error("🚨 Models not found! Please run 'python train_model.py' first.")
    st.stop()

# ==========================================
# 2. UI DESIGN
# ==========================================
st.title("Smart Loan Approval System")
st.markdown("""
This system uses a **Support Vector Machine (SVM)** to predict loan eligibility.
""")
st.divider()

# Sidebar Inputs
st.sidebar.header("📝 Applicant Details")

# Capture Inputs
income = st.sidebar.number_input("Applicant Income ($)", min_value=0, value=4000, step=100)
loan_amt = st.sidebar.number_input("Loan Amount ($K)", min_value=0, value=100, step=10)
credit = st.sidebar.radio("Credit History", ["No Debt (1.0)", "Past Default (0.0)"])
self_emp = st.sidebar.selectbox("Employment Status", ["No (Salaried)", "Yes (Self-Employed)"])

# Process Inputs
credit_val = 1.0 if "No Debt" in credit else 0.0
# We use the loaded encoder to transform "Yes/No" exactly how the model learned it
self_emp_val = le_self_emp.transform(["Yes" if "Yes" in self_emp else "No"])[0]

# ==========================================
# 3. PREDICTION LOGIC
# ==========================================
if st.button("Check Loan Eligibility", type="primary"):
    
    # Create a DataFrame for the input (must match training feature order exactly)
    # Features: ['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Self_Employed']
    input_data = pd.DataFrame([[income, loan_amt, credit_val, self_emp_val]], 
                              columns=['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Self_Employed'])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Results")
    
    if prediction == 1:
        st.success(f"✅ **Loan Approved**")
        st.caption(f"Confidence: {probability[1]:.2%}")
        st.info("ℹ️ **Reason:** Applicant has a healthy income-to-loan ratio and good credit standing.")
    else:
        st.error(f"❌ **Loan Rejected**")
        st.caption(f"Confidence: {probability[0]:.2%}")
        st.warning("⚠️ **Reason:** High risk detected. This is often due to poor credit history or low income relative to the loan size.")