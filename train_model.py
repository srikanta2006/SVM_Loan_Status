import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

def train():
    print("⏳ Loading data...")
    # 1. Load Data
    df = pd.read_csv('data/loan_approval.csv')
    
    # 2. Data Cleaning (Handling missing values based on your dataset)
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    
    # 3. Feature Selection
    # We will use 4 main features for the app
    # (Note: In a real app, you might use more, but we keep it simple for the UI)
    features = ['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Self_Employed']
    
    # Encoding 'Self_Employed' (Yes/No -> 1/0)
    le_self_emp = LabelEncoder()
    df['Self_Employed'] = le_self_emp.fit_transform(df['Self_Employed'])
    
    X = df[features]
    
    # Target encoding (Y/N -> 1/0)
    # Check for the specific column name in your CSV
    target_col = 'Loan_Status' if 'Loan_Status' in df.columns else 'Loan_Status (Approved)'
    df[target_col] = df[target_col].map({'Y': 1, 'N': 0})
    y = df[target_col]

    # 4. Scaling (Crucial for SVM)
    print("⚖️  Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Training
    print("🧠 Training SVM Model (RBF Kernel)...")
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_scaled, y)

    # 6. Save the Model and Scaler
    print("💾 Saving artifacts to 'models/' folder...")
    joblib.dump(model, 'models/svm_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le_self_emp, 'models/le_self_emp.pkl') # Save encoder too!
    
    print("✅ Done! You can now run 'app.py'.")

if __name__ == "__main__":
    train()