import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="🏡 Real Estate Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("model.pkl")
        return scaler, model
    except FileNotFoundError:
        st.error("⚠ Model files not found! Please ensure 'scaler.pkl' and 'model.pkl' are in the same directory.")
        return None, None

scaler, model = load_models()

# Header
st.markdown('<h1 class="main-header">🏡 Real Estate Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for additional features
with st.sidebar:
    st.header("🎯 Prediction Settings")
    
    # Property type selection
    property_type = st.selectbox(
        "Property Type",
        ["Single Family Home", "Condo", "Townhouse", "Apartment"],
        index=0
    )
    
    # Location factor
    location_factor = st.select_slider(
        "Location Desirability",
        options=["Poor", "Fair", "Good", "Excellent", "Premium"],
        value="Good",
        help="Adjust based on neighborhood quality"
    )
    
    # Property age
    property_age = st.slider(
        "Property Age (years)",
        min_value=0,
        max_value=100,
        value=10,
        help="Age of the property in years"
    )
    
    # Advanced options
    st.header("🔧 Advanced Options")
    show_analysis = st.checkbox("Show Market Analysis", value=True)
    show_comparison = st.checkbox("Show Price Comparison", value=True)
    confidence_interval = st.checkbox("Show Confidence Range", value=False)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🏠 Property Details")
    
    # Input fields with better styling
    col_bed, col_bath = st.columns(2)
    
    with col_bed:
        bed = st.number_input(
            "🛏 Number of Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Enter the number of bedrooms"
        )
    
    with col_bath:
        bath = st.number_input(
            "🛁 Number of Bathrooms",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="Enter the number of bathrooms"
        )
    
    sqft = st.number_input(
        "📐 Living Area (sq.ft)",
        min_value=200,
        max_value=10000,
        value=1500,
        step=50,
        help="Enter the total living area in square feet"
    )
    
    # Additional features
    st.subheader("🏡 Additional Features")
    
    col_garage, col_yard = st.columns(2)
    with col_garage:
        garage = st.checkbox("🚗 Garage", value=False)
    with col_yard:
        yard = st.checkbox("🌳 Yard/Garden", value=False)
    
    col_pool, col_fireplace = st.columns(2)
    with col_pool:
        pool = st.checkbox("🏊 Swimming Pool", value=False)
    with col_fireplace:
        fireplace = st.checkbox("🔥 Fireplace", value=False)

with col2:
    st.header("📊 Property Summary")
    
    # Enhanced property summary with icons and better formatting
    features_list = []
    if garage:
        features_list.append("🚗 Garage")
    if yard:
        features_list.append("🌳 Yard")
    if pool:
        features_list.append("🏊 Pool")
    if fireplace:
        features_list.append("🔥 Fireplace")
    
    features_text = ", ".join(features_list) if features_list else "None selected"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);">
        <h3 style="margin: 0 0 1rem 0; text-align: center;">🏠 Property Overview</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; font-size: 0.9rem;">
            <div><strong>🏘 Type:</strong><br>{property_type}</div>
            <div><strong>📍 Location:</strong><br>{location_factor}</div>
            <div><strong>🛏 Bedrooms:</strong><br>{bed}</div>
            <div><strong>🛁 Bathrooms:</strong><br>{bath}</div>
            <div><strong>📐 Area:</strong><br>{sqft:,} sq.ft</div>
            <div><strong>🕐 Age:</strong><br>{property_age} years</div>
        </div>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.3);">
            <strong>✨ Features:</strong><br>
            <span style="font-size: 0.9rem;">{features_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Price per square foot indicator with better styling
    if sqft > 0:
        estimated_psf = st.slider(
            "💰 Est. Price per sq.ft ($)",
            min_value=50,
            max_value=500,
            value=150,
            help="Adjust based on local market rates"
        )
        rough_estimate = sqft * estimated_psf
        
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; 
                    border-left: 4px solid #28a745; margin: 1rem 0;">
            <h4 style="color: #28a745; margin: 0;">💡 Quick Estimate</h4>
            <p style="font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0 0 0;">
                ${rough_estimate:,.0f}
            </p>
            <small style="color: #6c757d;">Based on ${estimated_psf}/sq.ft</small>
        </div>
        """, unsafe_allow_html=True)
    
    # EMI Calculator Section
    st.header("🏦 EMI Calculator")
    
    # EMI inputs
    loan_amount_percent = st.slider(
        "Down Payment (%)",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        help="Percentage of property value as down payment"
    )
    
    interest_rate = st.slider(
        "Interest Rate (% per annum)",
        min_value=6.0,
        max_value=15.0,
        value=8.5,
        step=0.25,
        help="Current home loan interest rate"
    )
    
    loan_tenure = st.selectbox(
        "Loan Tenure",
        [10, 15, 20, 25, 30],
        index=3,
        help="Loan repayment period in years"
    )
    
    # Calculate EMI based on rough estimate
    if sqft > 0:
        property_value = rough_estimate
        down_payment = property_value * (loan_amount_percent / 100)
        loan_amount = property_value - down_payment
        
        # EMI calculation formula
        monthly_rate = interest_rate / (12 * 100)
        num_payments = loan_tenure * 12
        
        if monthly_rate > 0:
            emi = loan_amount * (monthly_rate * (1 + monthly_rate)*num_payments) / ((1 + monthly_rate)*num_payments - 1)
        else:
            emi = loan_amount / num_payments
        
        total_payment = emi * num_payments
        total_interest = total_payment - loan_amount
        
        # EMI Summary Display
        st.markdown("""
        <style>
        .emi-container {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 8px 25px rgba(40, 167, 69, 0.3);
        }
        .emi-header {
            margin: 0 0 1rem 0;
            text-align: center;
            font-size: 1.2rem;
        }
        .emi-amount {
            background: rgba(255,255,255,0.1);
            padding: 0.8rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .emi-monthly {
            font-size: 1.4rem;
            font-weight: bold;
            text-align: center;
        }
        .emi-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.8rem;
            font-size: 0.9rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="emi-container">
            <h3 class="emi-header">💳 EMI Breakdown</h3>
            <div class="emi-amount">
                <div class="emi-monthly">Monthly EMI: ${emi:,.0f}</div>
            </div>
            <div class="emi-grid">
                <div><strong>🏠 Property Value:</strong><br>${property_value:,.0f}</div>
                <div><strong>💰 Down Payment:</strong><br>${down_payment:,.0f}</div>
                <div><strong>🏦 Loan Amount:</strong><br>${loan_amount:,.0f}</div>
                <div><strong>📊 Interest Rate:</strong><br>{interest_rate}% p.a.</div>
                <div><strong>⏰ Tenure:</strong><br>{loan_tenure} years</div>
                <div><strong>💸 Total Interest:</strong><br>${total_interest:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional EMI insights
        monthly_income_required = emi / 0.4  # Assuming 40% of income can go to EMI
        
        st.markdown(f"""
        <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #ffc107; margin: 1rem 0;">
            <h4 style="color: #856404; margin: 0 0 0.5rem 0;">💡 EMI Insights</h4>
            <ul style="margin: 0; padding-left: 1.2rem; color: #856404;">
                <li><strong>Recommended Monthly Income:</strong> ${monthly_income_required:,.0f}</li>
                <li><strong>Total Amount Payable:</strong> ${total_payment:,.0f}</li>
                <li><strong>Interest vs Principal:</strong> {(total_interest/loan_amount)*100:.1f}% extra</li>
                <li><strong>Monthly Payment Burden:</strong> Consider if EMI is &lt;40% of income</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # EMI Comparison Chart
        st.subheader("📈 EMI Comparison by Tenure")
        
        tenures = [10, 15, 20, 25, 30]
        emis = []
        total_interests = []
        
        for tenure in tenures:
            num_pay = tenure * 12
            if monthly_rate > 0:
                temp_emi = loan_amount * (monthly_rate * (1 + monthly_rate)*num_pay) / ((1 + monthly_rate)*num_pay - 1)
            else:
                temp_emi = loan_amount / num_pay
            
            temp_total = temp_emi * num_pay
            temp_interest = temp_total - loan_amount
            
            emis.append(temp_emi)
            total_interests.append(temp_interest)
        
        comparison_df = pd.DataFrame({
            'Tenure (Years)': tenures,
            'Monthly EMI ($)': emis,
            'Total Interest ($)': total_interests
        })
        
        st.dataframe(
            comparison_df.style.format({
                'Monthly EMI ($)': '${:,.0f}',
                'Total Interest ($)': '${:,.0f}'
            }).highlight_max(subset=['Monthly EMI ($)'], color='lightcoral')
            .highlight_min(subset=['Total Interest ($)'], color='lightgreen'),
            use_container_width=True
        )
        
    else:
        st.info("💡 Enter property details to calculate EMI")

# Prediction section
st.markdown("---")
st.header("🎯 Price Prediction")

# Input validation
def validate_inputs():
    errors = []
    if bed <= 0:
        errors.append("Number of bedrooms must be positive")
    if bath <= 0:
        errors.append("Number of bathrooms must be positive")
    if sqft <= 0:
        errors.append("Square footage must be positive")
    return errors

# Prediction button
col_predict, col_reset = st.columns([3, 1])

with col_predict:
    prediction_button = st.button("🔮 Predict Property Price", type="primary", use_container_width=True)

with col_reset:
    if st.button("🔄 Reset", use_container_width=True):
        st.experimental_rerun()

# Main prediction logic
if prediction_button:
    errors = validate_inputs()
    
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
    elif model is None or scaler is None:
        st.error("❌ Model not loaded. Please check if model files exist.")
    else:
        # Show loading animation
        with st.spinner("🔍 Analyzing property data..."):
            time.sleep(1)  # Simulate processing time
            
            # Prepare input data
            X = [bed, bath, sqft]
            X1 = np.array(X)
            X_array = scaler.transform([X1])
            y_pred = model.predict(X_array)[0]
            
            # Apply location and feature adjustments
            location_multipliers = {
                "Poor": 0.85,
                "Fair": 0.95,
                "Good": 1.0,
                "Excellent": 1.15,
                "Premium": 1.35
            }
            
            location_mult = location_multipliers[location_factor]
            
            # Age depreciation (simplified)
            age_factor = max(0.7, 1 - (property_age * 0.005))
            
            # Feature bonuses
            feature_bonus = 0
            if garage:
                feature_bonus += 0.05
            if yard:
                feature_bonus += 0.03
            if pool:
                feature_bonus += 0.08
            if fireplace:
                feature_bonus += 0.02
            
            # Final adjusted prediction
            adjusted_price = y_pred * location_mult * age_factor * (1 + feature_bonus)
            
            # Display results with animation
            st.balloons()
            
            # Main prediction display
            st.markdown(f"""
            <div class="prediction-box">
                💰 Predicted Price: ${adjusted_price:,.0f}
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed breakdown
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Base Prediction</h4>
                    <h3>${y_pred:,.0f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Price per sq.ft</h4>
                    <h3>${adjusted_price/sqft:.0f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Confidence</h4>
                    <h3>{"High" if abs(adjusted_price - y_pred) < y_pred * 0.2 else "Medium"}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Show confidence interval if enabled
            if confidence_interval:
                lower_bound = adjusted_price * 0.9
                upper_bound = adjusted_price * 1.1
                st.info(f"📊 *Price Range:* ${lower_bound:,.0f} - ${upper_bound:,.0f}")
            
            # Market analysis
            if show_analysis:
                st.subheader("📈 Market Analysis")
                
                # Create a simple market comparison chart
                market_data = {
                    'Property Type': ['Studio', '1 BR', '2 BR', '3 BR', '4+ BR'],
                    'Avg Price': [200000, 300000, 450000, 600000, 800000],
                    'Your Property': [0, 0, 0, 0, 0]
                }
                
                # Highlight user's property type
                if bed == 1:
                    market_data['Your Property'][1] = adjusted_price
                elif bed == 2:
                    market_data['Your Property'][2] = adjusted_price
                elif bed == 3:
                    market_data['Your Property'][3] = adjusted_price
                else:
                    market_data['Your Property'][4] = adjusted_price
                
                df = pd.DataFrame(market_data)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df['Property Type'],
                    y=df['Avg Price'],
                    name='Market Average',
                    marker_color='lightblue'
                ))
                fig.add_trace(go.Bar(
                    x=df['Property Type'],
                    y=df['Your Property'],
                    name='Your Property',
                    marker_color='darkblue'
                ))
                
                fig.update_layout(
                    title='Market Comparison',
                    xaxis_title='Property Type',
                    yaxis_title='Price ($)',
                    barmode='overlay',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Price comparison table
            if show_comparison:
                st.subheader("🏘 Similar Properties Estimate")
                
                comparison_data = []
                for i in range(3):
                    similar_bed = bed + np.random.choice([-1, 0, 1])
                    similar_bath = bath + np.random.choice([-0.5, 0, 0.5])
                    similar_sqft = int(sqft + np.random.normal(0, 200))
                    
                    if similar_bed > 0 and similar_bath > 0 and similar_sqft > 0:
                        similar_X = np.array([similar_bed, similar_bath, similar_sqft])
                        similar_X_array = scaler.transform([similar_X])
                        similar_pred = model.predict(similar_X_array)[0] * location_mult * age_factor
                        
                        comparison_data.append({
                            'Bedrooms': int(similar_bed),
                            'Bathrooms': similar_bath,
                            'Sq.Ft': similar_sqft,
                            'Estimated Price': f"${similar_pred:,.0f}"
                        })
                
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    st.dataframe(df_comparison, use_container_width=True)
            
            # Additional insights
            st.subheader("💡 Insights & Recommendations")
            
            insights = []
            
            if adjusted_price > y_pred:
                insights.append("✅ Location and features add significant value to your property")
            else:
                insights.append("📝 Consider property improvements to increase value")
            
            if sqft < 1000:
                insights.append("🏠 Compact living space - great for first-time buyers")
            elif sqft > 3000:
                insights.append("🏰 Spacious property - ideal for large families")
            
            if property_age < 5:
                insights.append("✨ New construction - minimal maintenance needed")
            elif property_age > 30:
                insights.append("🔧 Older property - consider renovation impact on value")
            
            for insight in insights:
                st.write(insight)

else:
    st.info("👆 Enter your property details above and click 'Predict Property Price' to get started!")
    
    # Show example predictions
    st.subheader("📋 Example Predictions")
    
    examples = [
        {"Bedrooms": 2, "Bathrooms": 1, "Sq.Ft": 800, "Type": "Condo"},
        {"Bedrooms": 3, "Bathrooms": 2, "Sq.Ft": 1500, "Type": "Single Family"},
        {"Bedrooms": 4, "Bathrooms": 3, "Sq.Ft": 2200, "Type": "Townhouse"}
    ]
    
    for i, example in enumerate(examples):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"{example['Type']}:** {example['Bedrooms']} bed, {example['Bathrooms']} bath, {example['Sq.Ft']} sq.ft")
        with col2:
            if st.button(f"Try This", key=f"example_{i}"):
                st.session_state.bed = example['Bedrooms']
                st.session_state.bath = example['Bathrooms']
                st.session_state.sqft = example['Sq.Ft']
                st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🏡 Real Estate Price Predictor | Built with ❤ using Streamlit</p>
    <p><small>Predictions are estimates based on machine learning models. Actual prices may vary based on market conditions.</small></p>
</div>
""", unsafe_allow_html=True)