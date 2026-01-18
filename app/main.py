"""
Streamlit App for Revenue & Churn Optimization Engine
Interactive What-If analysis for churn prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Revenue & Churn Optimization Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("📊 Revenue & Churn Optimization Engine")
st.markdown("### Predictive Modeling for Customer Retention & Revenue Optimization")
st.markdown("---")

@st.cache_resource
def load_model():
    """Load the trained model and label encoders"""
    try:
        # Load model
        model_path = Path(__file__).parent.parent / 'models' / 'churn_model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load label encoders
        encoders_path = Path(__file__).parent.parent / 'models' / 'churn_model_encoders.pkl'
        if encoders_path.exists():
            with open(encoders_path, 'rb') as f:
                label_encoders = pickle.load(f)
        else:
            label_encoders = None
        
        return model, label_encoders
    except FileNotFoundError as e:
        st.error(f"Model file not found. Please train the model first using: python src/train.py")
        st.stop()
        return None, None

def encode_categorical_features(df, label_encoders):
    """Encode categorical features using label encoders"""
    if label_encoders:
        for col, encoder in label_encoders.items():
            if col in df.columns:
                # Handle unseen labels
                df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
                df[col] = encoder.transform(df[col])
    return df

def get_feature_importance(model, feature_names):
    """Extract feature importance from the model"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df

def predict_churn(model, input_data, label_encoders=None):
    """Predict churn probability for given input data"""
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Encode categorical features
    if label_encoders:
        df = encode_categorical_features(df, label_encoders)
    
    # Ensure correct column order
    expected_features = [
        'avg_transaction_val', 'fee_percentage', 'tenure_months', 
        'complaints_count', 'transaction_frequency', 'customer_segment',
        'payment_method', 'monthly_volume', 'support_tickets', 'loyalty_points'
    ]
    
    # Reorder columns to match training data
    df = df.reindex(columns=expected_features, fill_value=0)
    
    # Predict
    churn_probability = model.predict_proba(df)[0][1]
    
    return churn_probability

def main():
    """Main Streamlit app function"""
    
    # Load model
    model, label_encoders = load_model()
    
    if model is None:
        return
    
    # Sidebar for What-If analysis
    st.sidebar.header("🔍 What-If Analysis")
    st.sidebar.markdown("Adjust customer characteristics to see churn probability")
    
    # Customer input fields
    avg_transaction_val = st.sidebar.number_input(
        "Average Transaction Value ($)",
        min_value=50.0,
        max_value=5000.0,
        value=500.0,
        step=50.0,
        help="Average value of each transaction in USD"
    )
    
    fee_percentage = st.sidebar.slider(
        "Fee Percentage (%)",
        min_value=2.0,
        max_value=8.0,
        value=4.5,
        step=0.1,
        help="Transaction fee as percentage of transaction value"
    )
    
    tenure_months = st.sidebar.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=60,
        value=12,
        step=1,
        help="Number of months since customer joined"
    )
    
    complaints_count = st.sidebar.number_input(
        "Complaints Count",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
        help="Total number of customer complaints"
    )
    
    transaction_frequency = st.sidebar.number_input(
        "Transaction Frequency (per month)",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help="Number of transactions per month"
    )
    
    customer_segment = st.sidebar.selectbox(
        "Customer Segment",
        options=["Standard", "Silver", "Gold", "Premium"],
        help="Customer tier based on volume"
    )
    
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        options=["Bank Transfer", "Credit Card", "Debit Card", "Digital Wallet", "Cash"],
        help="Preferred payment method"
    )
    
    # Calculate derived features
    monthly_volume = avg_transaction_val * transaction_frequency
    support_tickets = max(0, int(np.random.poisson(0.3) if complaints_count < 2 else complaints_count + 1))
    loyalty_points = max(0, int(tenure_months * 10 * np.random.uniform(0.8, 1.2)))
    
    # Prepare input data
    input_data = {
        'avg_transaction_val': avg_transaction_val,
        'fee_percentage': fee_percentage,
        'tenure_months': tenure_months,
        'complaints_count': complaints_count,
        'transaction_frequency': transaction_frequency,
        'customer_segment': customer_segment,
        'payment_method': payment_method,
        'monthly_volume': monthly_volume,
        'support_tickets': support_tickets,
        'loyalty_points': loyalty_points
    }
    
    # Make prediction
    churn_probability = predict_churn(model, input_data, label_encoders)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Churn Prediction Results")
        
        # Display churn probability with visual indicator
        prob_percentage = churn_probability * 100
        
        # Color coding for churn probability
        if prob_percentage < 30:
            color = "🟢"
            risk_level = "Low Risk"
        elif prob_percentage < 60:
            color = "🟡"
            risk_level = "Medium Risk"
        else:
            color = "🔴"
            risk_level = "High Risk"
        
        # Large display of churn probability
        st.metric(
            label=f"Churn Probability {color}",
            value=f"{prob_percentage:.1f}%",
            delta=risk_level,
            delta_color="inverse"
        )
        
        # Probability bar chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Risk Level"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        st.header("📊 Customer Profile")
        st.markdown("### Current Input Values")
        
        profile_df = pd.DataFrame({
            'Metric': ['Monthly Volume', 'Loyalty Points', 'Support Tickets'],
            'Value': [
                f"${monthly_volume:,.2f}",
                f"{loyalty_points:,}",
                f"{support_tickets}"
            ]
        })
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
        
        st.markdown("### Economic Impact")
        # Estimate monthly revenue
        monthly_revenue = monthly_volume * (fee_percentage / 100)
        annual_revenue = monthly_revenue * 12
        
        st.metric("Monthly Revenue", f"${monthly_revenue:,.2f}")
        st.metric("Annual Revenue (Projected)", f"${annual_revenue:,.2f}")
        
        if churn_probability > 0.5:
            potential_loss = annual_revenue
            st.warning(f"⚠️ At-Risk Revenue: ${potential_loss:,.2f}/year")
    
    st.markdown("---")
    
    # Feature Importance Section
    st.header("🔍 Top 5 Features Driving Prediction")
    
    # Get feature importance
    feature_names = [
        'avg_transaction_val', 'fee_percentage', 'tenure_months',
        'complaints_count', 'transaction_frequency', 'customer_segment',
        'payment_method', 'monthly_volume', 'support_tickets', 'loyalty_points'
    ]
    
    importance_df = get_feature_importance(model, feature_names)
    top_5_features = importance_df.head(5)
    
    # Create bar chart for top 5 features
    fig = px.bar(
        top_5_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 5 Most Important Features for Churn Prediction',
        labels={'importance': 'Feature Importance', 'feature': 'Feature Name'},
        color='importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display feature importance table
    with st.expander("View All Feature Importance Rankings"):
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
    
    # Business Insights Section
    st.markdown("---")
    st.header("💡 Business Insights & Recommendations")
    
    insights = []
    
    if fee_percentage > 6.0:
        insights.append("⚠️ **High Fee Alert**: Fee percentage is above 6%, which significantly increases churn risk. Consider fee optimization strategies.")
    
    if complaints_count > 2:
        insights.append("⚠️ **Service Quality Issue**: Multiple complaints detected. Prioritize customer service intervention to reduce churn.")
    
    if tenure_months < 3:
        insights.append("💡 **New Customer**: Low tenure increases churn risk. Implement onboarding programs to improve retention.")
    
    if transaction_frequency < 2:
        insights.append("💡 **Low Engagement**: Low transaction frequency indicates reduced engagement. Consider re-engagement campaigns.")
    
    if avg_transaction_val < 200:
        insights.append("💡 **Price Sensitivity**: Low transaction values suggest price-sensitive segment. Fee elasticity may be high.")
    
    if tenure_months > 24:
        insights.append("✅ **Loyal Customer**: Long tenure indicates strong customer loyalty. Focus on maintaining satisfaction.")
    
    if not insights:
        insights.append("✅ **Stable Profile**: Customer profile shows moderate risk factors. Continue standard retention practices.")
    
    for insight in insights:
        st.markdown(insight)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>Revenue & Churn Optimization Engine | Powered by Random Forest</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
