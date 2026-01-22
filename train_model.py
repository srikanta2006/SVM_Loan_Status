import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train():
    print("⏳ Loading and Balancing Data...")
    
    # 1. Load Data
    try:
        df = pd.read_csv('data/loan_approval.csv')
    except FileNotFoundError:
        print("❌ Error: 'loan_approval.csv' not found in 'data/' folder.")
        return

    # 2. Critical Data Cleaning
    # Fill numbers with Median (Standard practice)
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['ApplicantIncome'] = df['ApplicantIncome'].fillna(df['ApplicantIncome'].median())
    df['Credit_History'] = df['Credit_History'].fillna(0.0) # Assume BAD credit if missing (Safety first)
    df['Self_Employed'] = df['Self_Employed'].fillna('No')

    # 3. Target Encoding
    target_col = 'Loan_Status' if 'Loan_Status' in df.columns else 'Loan_Status (Approved)'
    df['Target'] = df[target_col].map({'Y': 1, 'N': 0})

    # 4. Feature Selection
    # We strictly use the features available in the UI
    X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History']]
    
    # Encode Self_Employed
    le_emp = LabelEncoder()
    df['Self_Employed_Code'] = le_emp.fit_transform(df['Self_Employed'])
    X['Self_Employed'] = df['Self_Employed_Code']
    
    y = df['Target']

    # 5. Scaling (The most important step for Income vs Loan comparison)
    print("⚖️  Scaling Features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Training with STRICTER Rules
    print("🧠 Training SVM (Balanced Mode)...")
    
    # class_weight='balanced': Forces model to respect "Rejections"
    # C=0.8: Reduces overfitting to the "Always Yes" majority
    model = SVC(kernel='rbf', C=0.8, probability=True, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)

    # 7. Validation Report
    print("\n📊 Model Performance:")
    print(classification_report(y, model.predict(X_scaled)))

    # 8. Save
    joblib.dump(model, 'models/svm_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le_emp, 'models/le_self_emp.pkl')
    print("✅ Logic Fixed & Saved!")

if __name__ == "__main__":
    train()