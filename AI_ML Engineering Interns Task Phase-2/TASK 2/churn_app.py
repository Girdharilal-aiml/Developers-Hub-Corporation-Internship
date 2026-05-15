"""
Customer Churn Prediction Pipeline
===================================
Run this file directly:  python churn_app.py

What it does:
- Trains the full ML pipeline on the Telco dataset
- Saves the model to churn_pipeline.joblib
- Lets you test it interactively on new customer data

Requirements:
    pip install scikit-learn pandas numpy joblib
"""

import sys
import io
# Windows terminal encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')



import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ----------------------------──────────────────
# CONFIG
# ----------------------------──────────────────
MODEL_PATH   = "churn_pipeline.joblib"
DATA_PATH    = "telco_churn.csv"

NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]


# ----------------------------──────────────────
# HELPERS
# ----------------------------──────────────────
def print_banner(text):
    print()
    print("=" * 55)
    print(f"  {text}")
    print("=" * 55)


def risk_label(prob):
    if prob >= 0.65:
        return "[HIGH RISK]"
    elif prob >= 0.35:
        return "[MEDIUM RISK]"
    else:
        return "[LOW RISK]"


# ----------------------------──────────────────
# STEP 1 - BUILD AND TRAIN THE PIPELINE
# ----------------------------──────────────────
def build_pipeline():
    """Builds the full sklearn Pipeline (preprocessor + RandomForest)."""

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer,     NUMERIC_COLS),
        ('cat', categorical_transformer, CATEGORICAL_COLS)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    return pipeline


def train_and_save():
    """Loads data, trains the pipeline, evaluates it, saves to disk."""

    print_banner("TRAINING THE PIPELINE")

    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"  ❌ Dataset not found: {DATA_PATH}")
        print("  Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        return None

    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} customer records")

    # Clean
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn']        = (df['Churn'] == 'Yes').astype(int)

    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Churn rate: {y.mean()*100:.1f}%")

    # Train
    print("\n  Training Random Forest pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\n  >> Training complete!")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  ROC-AUC  : {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred,
                                 target_names=['Stayed', 'Churned']))

    # Save
    joblib.dump(pipeline, MODEL_PATH)
    size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"  >> Pipeline saved -> {MODEL_PATH} ({size:.1f} MB)")

    return pipeline


# ----------------------------──────────────────
# STEP 2 - PREDICT ON A SINGLE CUSTOMER
# ----------------------------──────────────────
def predict_single(pipeline, customer_dict):
    """Takes a dict of customer features, returns churn probability."""
    df_input = pd.DataFrame([customer_dict])
    prob  = pipeline.predict_proba(df_input)[0][1]
    label = pipeline.predict(df_input)[0]
    return prob, label


def demo_predictions(pipeline):
    """Runs 4 example customers through the model and prints results."""

    print_banner("DEMO: PREDICTING 4 CUSTOMERS")

    customers = [
        {
            "name": "Sarah (new, fiber, month-to-month)",
            "data": {
                'gender': 'Female', 'SeniorCitizen': 0,
                'Partner': 'No',    'Dependents': 'No',
                'tenure': 2,        'PhoneService': 'Yes',
                'MultipleLines': 'No',
                'InternetService': 'Fiber optic',
                'OnlineSecurity': 'No',  'OnlineBackup': 'No',
                'DeviceProtection': 'No','TechSupport': 'No',
                'StreamingTV': 'Yes',    'StreamingMovies': 'Yes',
                'Contract': 'Month-to-month',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 94.5,  'TotalCharges': 189.0,
            }
        },
        {
            "name": "Ahmed (5yr, two-year contract, DSL)",
            "data": {
                'gender': 'Male',   'SeniorCitizen': 0,
                'Partner': 'Yes',   'Dependents': 'Yes',
                'tenure': 60,       'PhoneService': 'Yes',
                'MultipleLines': 'Yes',
                'InternetService': 'DSL',
                'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
                'DeviceProtection': 'Yes','TechSupport': 'Yes',
                'StreamingTV': 'No',     'StreamingMovies': 'No',
                'Contract': 'Two year',
                'PaperlessBilling': 'No',
                'PaymentMethod': 'Bank transfer (automatic)',
                'MonthlyCharges': 55.0,  'TotalCharges': 3300.0,
            }
        },
        {
            "name": "Zara (senior, month-to-month, high bill)",
            "data": {
                'gender': 'Female', 'SeniorCitizen': 1,
                'Partner': 'No',    'Dependents': 'No',
                'tenure': 8,        'PhoneService': 'Yes',
                'MultipleLines': 'Yes',
                'InternetService': 'Fiber optic',
                'OnlineSecurity': 'No',  'OnlineBackup': 'No',
                'DeviceProtection': 'No','TechSupport': 'No',
                'StreamingTV': 'Yes',    'StreamingMovies': 'Yes',
                'Contract': 'Month-to-month',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 108.0, 'TotalCharges': 864.0,
            }
        },
        {
            "name": "Ali (loyal, one-year contract, no internet)",
            "data": {
                'gender': 'Male',   'SeniorCitizen': 0,
                'Partner': 'Yes',   'Dependents': 'Yes',
                'tenure': 48,       'PhoneService': 'Yes',
                'MultipleLines': 'No',
                'InternetService': 'No',
                'OnlineSecurity': 'No internet service',
                'OnlineBackup': 'No internet service',
                'DeviceProtection': 'No internet service',
                'TechSupport': 'No internet service',
                'StreamingTV': 'No internet service',
                'StreamingMovies': 'No internet service',
                'Contract': 'One year',
                'PaperlessBilling': 'No',
                'PaymentMethod': 'Mailed check',
                'MonthlyCharges': 24.0,  'TotalCharges': 1152.0,
            }
        },
    ]

    for c in customers:
        prob, label = predict_single(pipeline, c['data'])
        risk = risk_label(prob)
        action = (
            "-> Send retention offer NOW"   if prob >= 0.65 else
            "-> Monitor and check in"       if prob >= 0.35 else
            "-> No action needed"
        )
        print(f"\n  Customer : {c['name']}")
        print(f"  Prediction : {'CHURN' if label == 1 else 'STAY'}")
        print(f"  Probability: {prob:.1%}  {risk}")
        print(f"  Action     : {action}")


# ----------------------------──────────────────
# STEP 3 - INTERACTIVE MODE
# ----------------------------──────────────────
def interactive_mode(pipeline):
    """Lets the user type in customer details and get a live prediction."""

    print_banner("INTERACTIVE MODE - Enter Customer Details")
    print("  Type the values when prompted. Press Ctrl+C to quit.\n")

    def ask(prompt, default):
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else str(default)

    try:
        while True:
            print()
            customer = {
                'gender':            ask("Gender (Male/Female)", "Male"),
                'SeniorCitizen':     int(ask("Senior Citizen (0=No, 1=Yes)", 0)),
                'Partner':           ask("Partner (Yes/No)", "No"),
                'Dependents':        ask("Dependents (Yes/No)", "No"),
                'tenure':            int(ask("Tenure in months", 12)),
                'PhoneService':      ask("Phone Service (Yes/No)", "Yes"),
                'MultipleLines':     ask("Multiple Lines (Yes/No/No phone service)", "No"),
                'InternetService':   ask("Internet (DSL/Fiber optic/No)", "DSL"),
                'OnlineSecurity':    ask("Online Security (Yes/No/No internet service)", "No"),
                'OnlineBackup':      ask("Online Backup (Yes/No/No internet service)", "No"),
                'DeviceProtection':  ask("Device Protection (Yes/No/No internet service)", "No"),
                'TechSupport':       ask("Tech Support (Yes/No/No internet service)", "No"),
                'StreamingTV':       ask("Streaming TV (Yes/No/No internet service)", "No"),
                'StreamingMovies':   ask("Streaming Movies (Yes/No/No internet service)", "No"),
                'Contract':          ask("Contract (Month-to-month/One year/Two year)", "Month-to-month"),
                'PaperlessBilling':  ask("Paperless Billing (Yes/No)", "Yes"),
                'PaymentMethod':     ask("Payment (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))", "Electronic check"),
                'MonthlyCharges':    float(ask("Monthly Charges ($)", 65.0)),
                'TotalCharges':      float(ask("Total Charges ($)", 780.0)),
            }

            prob, label = predict_single(pipeline, customer)
            risk   = risk_label(prob)
            action = (
                "Send a retention offer immediately!"  if prob >= 0.65 else
                "Monitor this customer closely."       if prob >= 0.35 else
                "Customer appears stable."
            )

            print()
            print("  ─" * 28)
            print(f"  RESULT    : {'!! Will CHURN' if label == 1 else '>> Will STAY'}")
            print(f"  PROBABILITY : {prob:.1%}")
            print(f"  RISK LEVEL  : {risk}")
            print(f"  SUGGESTION  : {action}")
            print("  ─" * 28)

            again = input("\n  Predict another customer? (y/n): ").strip().lower()
            if again != 'y':
                break

    except KeyboardInterrupt:
        print("\n\n  Exiting interactive mode.")


# ----------------------------──────────────────
# MAIN
# ----------------------------──────────────────
def main():
    print()
    print("  +--------------------------------------------------+")
    print("  |   Customer Churn Prediction - ML Pipeline        |")
    print("  |   Telco Dataset - Random Forest - sklearn        |")
    print("  +--------------------------------------------------+")

    # Load existing model or train a new one
    if os.path.exists(MODEL_PATH):
        print(f"\n  Found saved model -> {MODEL_PATH}")
        choice = input("  Use saved model? (y to load, n to retrain): ").strip().lower()
        if choice == 'y':
            pipeline = joblib.load(MODEL_PATH)
            print("  >> Model loaded successfully")
        else:
            pipeline = train_and_save()
    else:
        print(f"\n  No saved model found. Training now...")
        pipeline = train_and_save()

    if pipeline is None:
        return

    # Menu
    while True:
        print_banner("WHAT WOULD YOU LIKE TO DO?")
        print("  1 -> Run demo predictions (4 example customers)")
        print("  2 -> Enter your own customer data interactively")
        print("  3 -> Retrain the model")
        print("  4 -> Exit")
        print()

        choice = input("  Your choice (1/2/3/4): ").strip()

        if choice == '1':
            demo_predictions(pipeline)
        elif choice == '2':
            interactive_mode(pipeline)
        elif choice == '3':
            pipeline = train_and_save()
        elif choice == '4':
            print("\n  Goodbye!\n")
            break
        else:
            print("  Please enter 1, 2, 3 or 4.")


if __name__ == '__main__':
    main()