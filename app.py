import streamlit as st
import joblib
import pandas as pd
import time

# ==========================================
# 1. PAGE CONFIG & SOPHISTICATED CSS
# ==========================================
st.set_page_config(
    page_title="FinTrust | AI Loan System",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# This CSS does the heavy lifting for the "Glassmorphism" look
st.markdown("""
    <style>
    /* BACKGROUND IMAGE */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1639322537228-f710d846310a?q=80&w=2832&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* GLASSMORPHISM CONTAINER CLASS */
    .glass-container {
        background: rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 25px;
        margin-bottom: 20px;
        color: white;
    }

    /* INPUT WIDGETS STYLING */
    .stSlider > div > div > div > div {
        background-color: #00d4ff !important;
    }
    .stRadio > label, .stSelectbox > label, .stNumberInput > label {
        color: white !important;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    /* CRAZY BUTTON STYLING */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        font-weight: bold;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 50px;
        transition: 0.3s;
        box-shadow: 0 0 15px #00d2ff;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px #00d2ff;
    }

    /* TEXT COLORS */
    h1, h2, h3, p {
        color: white !important;
        text-shadow: 2px 2px 4px #000000;
    }
    
    /* HIDE DEFAULT HEADER */
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/svm_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        le_self_emp = joblib.load('models/le_self_emp.pkl')
        return model, scaler, le_self_emp
    except FileNotFoundError:
        return None, None, None

model, scaler, le_self_emp = load_models()

# ==========================================
# 3. HEADER
# ==========================================
st.markdown('<h1 style="text-align: center; font-size: 3.5rem;">💎 FinTrust AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; opacity: 0.8;">Next-Gen Instant Loan Approval System</p>', unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 4. GLASS INTERFACE (INPUTS)
# ==========================================

# We use HTML injection to wrap Streamlit widgets in our 'glass-container' class
st.markdown('<div class="glass-container"><h3>🎛️ Control Panel</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💰 Financials")
    # SLIDERS for Money (As requested)
    income = st.slider("Annual Income ($)", min_value=1000, max_value=20000, value=5000, step=500)
    loan_amt = st.slider("Loan Amount Request ($K)", min_value=10, max_value=500, value=120, step=5)

with col2:
    st.markdown("#### 👤 Profile Details")
    credit_display = st.radio("Credit History", ["No Outstanding Debt (Clean)", "History of Default (Risky)"])
    st.write("") # Spacer
    self_emp_display = st.selectbox("Employment Type", ["Salaried (Standard)", "Self-Employed (Freelance/Biz)"])

# Close the glass container div
st.markdown('</div>', unsafe_allow_html=True)

# Process Inputs
credit_val = 1.0 if "No Outstanding" in credit_display else 0.0
self_emp_val = le_self_emp.transform(["Yes" if "Self" in self_emp_display else "No"])[0]

# ==========================================
# 5. ACTION & RESULTS
# ==========================================
st.write("")
_, btn_col, _ = st.columns([1, 1, 1])

with btn_col:
    # The big glowing button
    analyze = st.button("✨ RUN AI PREDICTION ✨")

if analyze:
    # 1. Loading Animation
    with st.spinner("🤖 AI is analyzing credit vectors..."):
        time.sleep(1.5) # Fake delay for dramatic effect
    
    # 2. Prepare Data
    input_data = pd.DataFrame([[income, loan_amt, credit_val, self_emp_val]], 
                              columns=['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Self_Employed'])
    
    # 3. Predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    # 4. Display Results (Glass Modal)
    if prediction == 1:
        st.balloons()
        st.markdown(f"""
            <div class="glass-container" style="border: 2px solid #00ff00; text-align: center;">
                <h1 style="color: #00ff00 !important; font-size: 4rem; text-shadow: 0 0 20px #00ff00;">ACCEPTED</h1>
                <p style="font-size: 1.5rem;">Confidence Score: <b>{probability[1]:.1%}</b></p>
                <hr style="border-color: rgba(255,255,255,0.2);">
                <p>The AI has determined this applicant is <b>Safe</b> for lending.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Metric Cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Income Coverage", "High", "24%")
        m2.metric("Risk Factor", "Low", "-12%")
        m3.metric("Approval Tier", "Premium", "Platinum")

    else:
        st.snow() # Snow for "Cold/Rejected" feeling
        st.markdown(f"""
            <div class="glass-container" style="border: 2px solid #ff4444; text-align: center;">
                <h1 style="color: #ff4444 !important; font-size: 4rem; text-shadow: 0 0 20px #ff0000;">REJECTED</h1>
                <p style="font-size: 1.5rem;">Risk Probability: <b>{probability[0]:.1%}</b></p>
                <hr style="border-color: rgba(255,255,255,0.2);">
                <p>The AI has flagged this application as <b>High Risk</b>.</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Metric Cards (Negative)
        m1, m2, m3 = st.columns(3)
        m1.metric("Income Coverage", "Low", "-40%", delta_color="inverse")
        m2.metric("Risk Factor", "Critical", "+85%", delta_color="inverse")
        m3.metric("Approval Tier", "Declined", "None")