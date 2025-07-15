import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set the Streamlit page title
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📉 Customer Churn Prediction App")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")

model = load_model()

# Define features used during training
model_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Raw Input Data")
    st.write(data.head())

    # Clean 'TotalCharges' and convert numeric columns
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Encode categorical features
    categorical_cols = [col for col in model_features if col not in numeric_cols]
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Scale numeric features
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # Keep only model features
    X = data[model_features]

    # Predict churn probability and class
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    # Add results to dataframe
    data['Churn Probability'] = probs
    data['Predicted Churn'] = np.where(preds == 1, 'Yes', 'No')

    st.subheader("🔍 Prediction Results")
    st.write(data[['Churn Probability', 'Predicted Churn']].head())

    # Download predictions
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("👆 Please upload a CSV file to get started.")
