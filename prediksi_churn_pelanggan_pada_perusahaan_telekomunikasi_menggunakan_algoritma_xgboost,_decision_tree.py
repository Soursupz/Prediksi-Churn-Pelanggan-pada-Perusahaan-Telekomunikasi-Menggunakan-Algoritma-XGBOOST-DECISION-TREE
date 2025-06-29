import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, RobustScaler

@st.cache_resource
def load_models():
    return {
        'Decision Tree': joblib.load('dt_model.pkl'),
        'XGBoost': joblib.load('xgb_model.pkl'),
        'Random Forest': joblib.load('rf_model.pkl')
    }

@st.cache_data
def preprocess(df):
    df = df.copy()
    df.columns = df.columns.str.lower()
    df['totalcharges'] = df['totalcharges'].replace(r'\s+', np.nan, regex=True).astype(float)
    df['totalcharges'] = df['totalcharges'].fillna(df['totalcharges'].median())
    df['seniorcitizen'] = df['seniorcitizen'].astype('object')
    df = df.replace({'No internet service': 'No', 'No phone service': 'No'})
    df['tenure_range'] = df['tenure'].apply(lambda x: (
        '0-12 bulan' if x <= 12 else
        '13-24 bulan' if x <= 24 else
        '25-36 bulan' if x <= 36 else
        '37-48 bulan' if x <= 48 else
        '49 bulan ke atas'))
    df['tenure_monthly_charge_interaction'] = df['tenure'] * df['monthlycharges']
    df = df.drop(columns=['customerid'])
    df['partner'] = df['partner'].map({'Yes': 0, 'No': 1})
    df['internetservice'] = df['internetservice'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    return df

@st.cache_data
def scale(df):
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)

def main():
    st.title("Telecom Customer Churn Prediction")
    st.write("Upload dataset untuk prediksi batch atau input satu baris.")
    
    models = load_models()
    
    mode = st.sidebar.radio("Mode", ['Single', 'Batch'])
    model_name = st.sidebar.selectbox("Model", list(models.keys()))
    
    df = None
    
    if mode == 'Single':
        st.header("Predict churn (single record)")
        data = {}
        for feature in ['gender', 'seniorcitizen', 'partner', 'dependents',
                        'tenure', 'phoneservice', 'multiplelines', 'internetservice',
                        'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
                        'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod',
                        'monthlycharges', 'totalcharges']:
            data[feature] = st.text_input(feature, "")
        
        if st.button("Predict"):
            df = pd.DataFrame([data])
    else:
        st.header("Predict churn (batch CSV)")
        uploaded = st.file_uploader("Upload CSV", type='csv')
        if uploaded:
            df = pd.read_csv(uploaded)
    
    if df is not None:
        st.write("Data input:")
        st.write(df.head())
        
        df_proc = preprocess(df)
        df_scaled = scale(df_proc.drop(columns=['churn'], errors='ignore'))
        
        model = models[model_name]
        preds = model.predict(df_scaled)
        probs = model.predict_proba(df_scaled)[:, 1]
        
        df['churn_pred'] = preds
        df['churn_proba'] = probs
        
        st.write(df[['churn_pred', 'churn_proba']])

if __name__ == "__main__":
    main()
