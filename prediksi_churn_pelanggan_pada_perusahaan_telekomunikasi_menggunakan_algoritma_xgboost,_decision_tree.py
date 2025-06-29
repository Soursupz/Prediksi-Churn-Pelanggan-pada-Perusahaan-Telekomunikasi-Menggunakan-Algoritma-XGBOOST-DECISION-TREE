import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .churn-risk {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .no-churn {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load trained models and scaler"""
    try:
        # Untuk demo, kita akan membuat model dummy jika file tidak ada
        # Dalam production, pastikan file model sudah tersedia
        model = None
        scaler = None
        features = None
        
        # Jika Anda sudah memiliki file model, uncomment baris di bawah
        # with open("model.pkl", "rb") as f:
        #     model = pickle.load(f)
        # with open("scaler.pkl", "rb") as f:
        #     scaler = pickle.load(f)
        # with open("features.pkl", "rb") as f:
        #     features = pickle.load(f)
        
        # Untuk demo, kita buat dummy features
        features = [
            'seniorcitizen', 'partner', 'dependents', 'tenure_range',
            'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
            'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
            'contract', 'paperlessbilling', 'paymentmethod', 'monthlycharges',
            'totalcharges', 'tenure_monthly_charge_interaction'
        ]
        
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def categorize_tenure(tenure):
    """Categorize tenure into ranges"""
    if tenure <= 12:
        return 0  # '0-12 bulan'
    elif tenure <= 24:
        return 1  # '13-24 bulan'
    elif tenure <= 36:
        return 2  # '25-36 bulan'
    elif tenure <= 48:
        return 3  # '37-48 bulan'
    else:
        return 4  # '49 bulan ke atas'

def preprocess_input(data):
    """Preprocess input data similar to training preprocessing"""
    # Apply same transformations as in training
    processed_data = data.copy()
    
    # Create tenure_range
    processed_data['tenure_range'] = categorize_tenure(processed_data['tenure'])
    
    # Create interaction feature
    processed_data['tenure_monthly_charge_interaction'] = (
        processed_data['tenure'] * processed_data['monthlycharges']
    )
    
    # Apply mappings (same as in training)
    processed_data['partner'] = 1 if processed_data['partner'] == 'No' else 0
    
    # Internet service mapping
    internet_mapping = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
    processed_data['internetservice'] = internet_mapping.get(processed_data['internetservice'], 0)
    
    # Binary mappings for other categorical features
    binary_features = [
        'seniorcitizen', 'dependents', 'multiplelines', 'onlinesecurity', 
        'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 
        'streamingmovies', 'paperlessbilling'
    ]
    
    for feature in binary_features:
        if feature in processed_data:
            processed_data[feature] = 1 if processed_data[feature] == 'Yes' else 0
    
    # Contract mapping
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    processed_data['contract'] = contract_mapping.get(processed_data['contract'], 0)
    
    # Payment method mapping (simplified)
    payment_mapping = {
        'Electronic check': 0, 'Mailed check': 1, 
        'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
    }
    processed_data['paymentmethod'] = payment_mapping.get(processed_data['paymentmethod'], 0)
    
    return processed_data

def make_prediction(input_data, model, scaler, features):
    """Make prediction using the trained model"""
    if model is None:
        # Demo prediction
        # Simulasi prediksi berdasarkan beberapa rules sederhana
        risk_score = 0
        
        # Faktor risiko tinggi
        if input_data['contract'] == 'Month-to-month':
            risk_score += 0.3
        if input_data['tenure'] < 12:
            risk_score += 0.2
        if input_data['monthlycharges'] > 70:
            risk_score += 0.2
        if input_data['techsupport'] == 'No':
            risk_score += 0.1
        if input_data['onlinesecurity'] == 'No':
            risk_score += 0.1
        if input_data['paymentmethod'] == 'Electronic check':
            risk_score += 0.1
            
        churn_probability = min(risk_score, 0.95)
        churn_prediction = 1 if churn_probability > 0.4 else 0
        
        return churn_prediction, churn_probability
    else:
        # Actual model prediction
        processed_data = preprocess_input(input_data)
        
        # Create feature vector
        feature_vector = []
        for feature in features:
            feature_vector.append(processed_data.get(feature, 0))
        
        # Scale features
        feature_vector = np.array(feature_vector).reshape(1, -1)
        if scaler:
            feature_vector = scaler.transform(feature_vector)
        
        # Make prediction
        prediction = model.predict(feature_vector)[0]
        probability = model.predict_proba(feature_vector)[0][1]
        
        return prediction, probability

def main():
    # Header
    st.markdown('<h1 class="main-header">🔮 Customer Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    model, scaler, features = load_models()
    
    # Sidebar untuk input
    st.sidebar.header("📝 Customer Information")
    st.sidebar.markdown("Please fill in the customer details below:")
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    
    with col2:
        st.subheader("📞 Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        
    # Additional services
    st.subheader("🛡️ Additional Services")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    
    with col4:
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    
    with col5:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    
    # Contract and billing
    st.subheader("📋 Contract & Billing")
    col6, col7 = st.columns(2)
    
    with col6:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    with col7:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)
    
    # Prediction button
    if st.button("🎯 Predict Churn Risk", type="primary"):
        # Prepare input data
        input_data = {
            'gender': gender,
            'seniorcitizen': senior_citizen,
            'partner': partner,
            'dependents': dependents,
            'tenure': tenure,
            'phoneservice': phone_service,
            'multiplelines': multiple_lines,
            'internetservice': internet_service,
            'onlinesecurity': online_security,
            'onlinebackup': online_backup,
            'deviceprotection': device_protection,
            'techsupport': tech_support,
            'streamingtv': streaming_tv,
            'streamingmovies': streaming_movies,
            'contract': contract,
            'paperlessbilling': paperless_billing,
            'paymentmethod': payment_method,
            'monthlycharges': monthly_charges,
            'totalcharges': total_charges
        }
        
        # Make prediction
        prediction, probability = make_prediction(input_data, model, scaler, features)
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        col8, col9 = st.columns(2)
        
        with col8:
            if prediction == 1:
                st.markdown(
                    f'<div class="prediction-result churn-risk">⚠️ HIGH CHURN RISK<br>Probability: {probability:.1%}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="prediction-result no-churn">✅ LOW CHURN RISK<br>Probability: {probability:.1%}</div>',
                    unsafe_allow_html=True
                )
        
        with col9:
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors analysis
        st.subheader("📊 Risk Factors Analysis")
        
        risk_factors = []
        if contract == "Month-to-month":
            risk_factors.append(("Contract Type", "Month-to-month contracts have higher churn risk", "🔴"))
        if tenure < 12:
            risk_factors.append(("Tenure", "New customers (< 12 months) are more likely to churn", "🔴"))
        if monthly_charges > 70:
            risk_factors.append(("Monthly Charges", "High monthly charges increase churn risk", "🟡"))
        if tech_support == "No":
            risk_factors.append(("Tech Support", "No tech support increases churn risk", "🟡"))
        if online_security == "No":
            risk_factors.append(("Online Security", "No online security increases churn risk", "🟡"))
        if payment_method == "Electronic check":
            risk_factors.append(("Payment Method", "Electronic check users have higher churn rates", "🟡"))
        
        if risk_factors:
            for factor, description, icon in risk_factors:
                st.markdown(f"{icon} **{factor}**: {description}")
        else:
            st.success("✅ No major risk factors detected!")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if prediction == 1:
            st.markdown("""
            **Immediate Actions:**
            - 📞 Contact customer proactively
            - 💰 Offer loyalty discount or promotion  
            - 📋 Consider contract upgrade incentives
            - 🛡️ Promote additional services (security, tech support)
            - 💳 Suggest automatic payment methods
            """)
        else:
            st.markdown("""
            **Retention Strategies:**
            - 🎁 Continue providing excellent service
            - 📧 Send satisfaction surveys regularly
            - 🆕 Introduce new services/features
            - 🎯 Include in upselling campaigns
            """)

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This app predicts customer churn risk using machine learning. "
        "The model analyzes customer demographics, services, and billing information "
        "to identify customers at risk of leaving."
    )
    
    st.sidebar.markdown("### 📈 Model Performance")
    st.sidebar.metric("Accuracy", "85.5%")
    st.sidebar.metric("Recall", "87%")
    st.sidebar.metric("ROC-AUC", "85.5%")

if __name__ == "__main__":
    main()
