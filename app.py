import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="AQI PM2.5 Predictor", page_icon="🌫️", layout="centered")

st.title("🌫️ Air Quality Index (PM 2.5) Predictor")
st.write("Enter weather details & past PM 2.5 values to predict tomorrow's air quality.")

# 2. Load Model & Scaler (Caching for fast loading)
@st.cache_resource
def load_assets():
    model = joblib.load('best_pm25_xgb_ultra_0.8671.pkl')
    scaler = joblib.load('scaler_ultra.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("Model & Scaler loaded successfully!")
except Exception as e:
    st.error("Please place 'best_pm25_xgb_ultra_0.8671.pkl' and 'scaler_ultra.pkl' in the same folder.")

st.markdown("---")

# 3. User Input Form
st.subheader("📊 Weather Parameters")
col1, col2 = st.columns(2)

with col1:
    t = st.number_input("Avg Temperature (°C)", value=25.0)
    tm = st.number_input("Max Temperature (°C)", value=32.0)
    tm_min = st.number_input("Min Temperature (°C)", value=18.0)
    slp = st.number_input("Sea Level Pressure (hPa)", value=1012.0)

with col2:
    h = st.number_input("Humidity (%)", value=60.0)
    vv = st.number_input("Visibility (km)", value=3.0)
    v = st.number_input("Wind Speed (m/s)", value=2.0)
    vm = st.number_input("Max Wind Speed (m/s)", value=5.0)

st.subheader("📈 Past Pollution Lags")
pm25_lag1 = st.number_input("Yesterday's PM 2.5 Level (Lag 1)", value=180.0)
pm25_lag2 = st.number_input("2 Days Ago PM 2.5 Level (Lag 2)", value=175.0)
pm25_lag3 = st.number_input("3 Days Ago PM 2.5 Level (Lag 3)", value=170.0)
pm25_roll7 = st.number_input("7-Day Moving Avg PM 2.5", value=165.0)

# 4. Predict Button Logic
if st.button("🔮 Predict PM 2.5 Level", use_container_width=True):
    # Dummy placeholder feature dictionary matching the exact 37 features
    input_dict = {
        'T': t, 'TM': tm, 'Tm': tm_min, 'SLP': slp, 'H': h, 'VV': vv, 'V': v, 'VM': vm,
        'day_index': 2088,
        'sin_day': np.sin(2 * np.pi * 2088 / 365.25),
        'cos_day': np.cos(2 * np.pi * 2088 / 365.25),
        'Temp_Range': tm - tm_min,
        'Ventilation_Index': v * vv,
        'Humid_Temp_Ratio': h / (t + 1e-5)
    }
    
    # Fill remaining lag/rolling features
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]:
        input_dict[f'PM25_lag_{lag}'] = pm25_lag1 if lag == 1 else (pm25_lag2 if lag == 2 else pm25_lag3)
        
    for w in [3, 7, 14]:
        input_dict[f'PM25_roll_mean_{w}'] = pm25_roll7
        input_dict[f'PM25_roll_std_{w}'] = 15.0
        input_dict[f'PM25_roll_max_{w}'] = pm25_lag1 + 20
        input_dict[f'PM25_roll_min_{w}'] = pm25_lag1 - 20

    # DataFrame conversion & scaling
    input_df = pd.DataFrame([input_dict])[scaler.feature_names_in_]
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    
    st.markdown("---")
    st.metric(label="Predicted PM 2.5 Level", value=f"{prediction:.2f} µg/m³")
    
    # AQI Status Indicator
    if prediction <= 50:
        st.success("Air Quality Status: GOOD 😊")
    elif prediction <= 100:
        st.info("Air Quality Status: MODERATE 😐")
    elif prediction <= 200:
        st.warning("Air Quality Status: UNHEALTHY FOR SENSITIVE GROUPS 😷")
    else:
        st.error("Air Quality Status: SEVERE / HAZARDOUS 🚨")