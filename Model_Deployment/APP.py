import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Define log_and_rename as in your pipeline ---
original_log_input_cols = ['Rainfall', 'ALT', 'Wind_Speed']
trained_log_cols = ['Rainfall log', 'ALT log', 'Wind_Speed log']
rename_map = dict(zip(original_log_input_cols, trained_log_cols))

def log_and_rename(df: pd.DataFrame):
    df = df.copy()
    for orig, new in rename_map.items():
        if orig in df.columns:
            df[new] = np.log1p(df[orig])
            df.drop(columns=[orig], inplace=True)
    return df

st.title("🌊 Flood Prediction App")

# --- Load pipeline safely ---
base_dir = os.path.dirname(__file__)
pipeline_path = os.path.join(base_dir, 'flood_pipeline.pkl')
model = joblib.load(pipeline_path)

# --- Define user inputs ---
rainfall_fall = st.number_input("Rainfall (mm)", min_value=0.0)
altitude = st.number_input("Altitude (m)", min_value=0.0)
windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)

# --- Define realistic default values for all features ---
# Replace these values with the mean or typical values from your training data
feature_means = {
    "Rainfall": 100.0,
    "ALT": 15.0,
    "Wind_Speed": 12.0,
    "Temperature": 30.0,
    "Humidity": 80.0,
    "Bright_Sunshine": 5.0,
    "Cloud_Coverage": 6.0,
    "LATITUDE": 23.8,
    "LONGITUDE": 90.4,
    "Max_Temp": 32.0,
    "Min_Temp": 22.0,
    
}

# --- Build input dataframe ---
expected_features = model.named_steps['model'].feature_names_in_
data_dict = {feat: feature_means.get(feat, 0) for feat in expected_features}

# Overwrite with user-provided inputs
data_dict['Rainfall'] = rainfall_fall
data_dict['ALT'] = altitude
data_dict['Wind_Speed'] = windspeed
if 'Max_Temp' in data_dict:
    data_dict['Max_Temp'] = temperature
if 'Humidity' in data_dict:
    data_dict['Humidity'] = humidity

data = pd.DataFrame([data_dict])

# --- Predict ---
prediction = model.predict(data)
st.success(f"Predicted flood risk: {prediction[0]}")
