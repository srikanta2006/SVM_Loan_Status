import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt
import time

# ==========================================
# 1. SETUP & UI CONFIG
# ==========================================
st.set_page_config(
    page_title="FinTrust | AI Loan System",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for that "Glassmorphism" look
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                    url("https://images.unsplash.com/photo-1639322537228-f710d846310a?q=80&w=2832&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: white;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.6) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    h1, h2, h3, h4, p, label { color: #ffffff !important; }
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        font-weight: bold;
        border-radius: 8px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Load Models
try:
    model = joblib.load('models/svm_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le_self_emp = joblib.load('models/le_self_emp.pkl')
except FileNotFoundError:
    st.error("🚨 Logic files missing! Run 'train_model.py' first.")
    st.stop()

# ==========================================
# 2. SIDEBAR INPUTS
# ==========================================
with st.sidebar:
    st.header("🎛️ Application Controls")
    
    # Using Number Inputs for precision
    income = st.number_input("Annual Income ($)", min_value=1000, value=5000, step=500)
    loan_amt = st.number_input("Loan Amount ($K)", min_value=10, value=120, step=10)
    
    st.markdown("---")
    
    credit = st.radio("Credit History", ["No Debt (1.0)", "Past Default (0.0)"])
    self_emp = st.selectbox("Employment", ["No (Salaried)", "Yes (Self-Employed)"])
    
    st.markdown("---")
    run_btn = st.button("✨ RUN RISK ANALYSIS")

# ==========================================
# 3. HYBRID DECISION ENGINE (LOGIC FIX)
# ==========================================

# Metrics Display
ratio = (loan_amt * 1000) / income
c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="glass-card"><h4>Income</h4><h2>${income:,}</h2></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="glass-card"><h4>Loan Request</h4><h2>${loan_amt}k</h2></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="glass-card"><h4>Debt Ratio</h4><h2 style="color:{"#ff4444" if ratio > 15 else "#00ff00"}">{ratio:.1f}x</h2></div>', unsafe_allow_html=True)

if run_btn:
    st.write("---")
    
    # 1. PREPARE INPUTS
    credit_val = 1.0 if "No Debt" in credit else 0.0
    emp_val = le_self_emp.transform(["Yes" if "Yes" in self_emp else "No"])[0]
    
    # 2. LAYER 1: HARD KNOCKOUT RULES (Business Logic)
    # If the user asks for 20x their income, we reject immediately.
    # If credit is 0 (Default), we reject immediately.
    
    rejection_reason = ""
    is_auto_reject = False
    
    if credit_val == 0.0:
        is_auto_reject = True
        rejection_reason = "Applicant has a history of Default (Credit Score Check Failed)."
    elif ratio > 12.0: # If Loan is > 12x Income
        is_auto_reject = True
        rejection_reason = f"Loan amount is {ratio:.1f}x the annual income. Limit is 12x."

    # 3. LAYER 2: AI MODEL (Only if Layer 1 passes)
    if not is_auto_reject:
        input_data = pd.DataFrame([[income, loan_amt, credit_val, emp_val]], 
                                  columns=['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Self_Employed'])
        
        # Scale & Predict
        input_scaled = scaler.transform(input_data)
        
        with st.spinner("🤖 AI Analyzing patterns..."):
            time.sleep(0.5)
            prediction = model.predict(input_scaled)[0]
            probs = model.predict_proba(input_scaled)[0]
            
        # Decision Logic
        final_decision = "Approved" if prediction == 1 else "Rejected"
        confidence = probs[1] if prediction == 1 else probs[0]
    else:
        # If auto-rejected, set values manually
        final_decision = "Rejected"
        confidence = 1.0
        prediction = 0

    # 4. DISPLAY RESULTS
    col_res, col_chart = st.columns([1, 1.5])
    
    with col_res:
        if final_decision == "Approved":
            st.balloons()
            st.markdown(f"""
                <div class="glass-card" style="border: 2px solid #00ff00; background: rgba(0, 50, 0, 0.4);">
                    <h1 style="color: #4dfc4d !important;">ACCEPTED</h1>
                    <p>Confidence: {confidence:.1%}</p>
                    <hr style="border-color: rgba(255,255,255,0.2)">
                    <p>Income and Credit Profile meet approval criteria.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.snow()
            st.markdown(f"""
                <div class="glass-card" style="border: 2px solid #ff4444; background: rgba(50, 0, 0, 0.4);">
                    <h1 style="color: #ff6b6b !important;">REJECTED</h1>
                    <p>Risk Certainty: {confidence:.1%}</p>
                    <hr style="border-color: rgba(255,255,255,0.2)">
                    <p>{rejection_reason if is_auto_reject else "AI detected high risk based on historical default patterns."}</p>
                </div>
            """, unsafe_allow_html=True)

    with col_chart:
        # Chart: Monthly Income vs EMI
        monthly_income = income / 12
        est_emi = (loan_amt * 1000) / 60 # 5 Year Term
        
        data = pd.DataFrame({
            'Component': ['Monthly Income', 'Est. Monthly EMI'],
            'Value': [monthly_income, est_emi],
            'Color': ['#00d4ff', '#ff4444']
        })
        
        c = alt.Chart(data).mark_bar(cornerRadius=10).encode(
            x='Component', y='Value', color=alt.Color('Color', scale=None), tooltip=['Value']
        ).properties(height=300, title="Affordability Check")
        st.altair_chart(c, use_container_width=True)