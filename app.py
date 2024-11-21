import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import os

# Helper function to load files with error handling
def load_file(file_path):
    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Ensure the file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        st.stop()

# Safely load each required file
try:
    mlp = load_file('models/mlp_model.pkl')
    kmeans = load_file('models/kmeans_model.pkl')
    income_encoder = load_file('models/income_encoder.pkl')
    grid_encoder = load_file('models/grid_encoder.pkl')
    scaler = load_file('models/scaler.pkl')
except Exception as e:
    st.error(f"Error initializing models or encoders: {e}")
    st.stop()

# Load metadata file with error handling
try:
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
except FileNotFoundError:
    st.error("Metadata file 'metadata.json' not found.")
    st.stop()
except json.JSONDecodeError:
    st.error("Error decoding 'metadata.json'. Ensure it is a valid JSON file.")
    st.stop()

# Extract metadata keys
features_to_scale = metadata.get('features_to_scale', [])
viability_map = metadata.get('viability_map', {})

# Streamlit App
st.title("Electrification Viability Prediction")
st.write("Determine whether a location is viable for grid electrification, wind microgrid, or neither.")

# User inputs
latitude = st.number_input("Latitude", value=0.0, step=0.1)
longitude = st.number_input("Longitude", value=0.0, step=0.1)
pop_density = st.number_input("Population Density (2020)", value=100, step=1)
wind_speed = st.number_input("Wind Speed (m/s)", value=5.0, step=0.1)
grid_value = st.selectbox("Grid Value", grid_encoder.classes_ if hasattr(grid_encoder, 'classes_') else [])
income_distribution = st.selectbox("Income Distribution", income_encoder.classes_ if hasattr(income_encoder, 'classes_') else [])

# Predict button
if st.button("Predict"):
    try:
        # Encode categorical variables
        grid_value_encoded = grid_encoder.transform([grid_value])[0]
        income_distribution_encoded = income_encoder.transform([income_distribution])[0]

        # Prepare input data
        input_data = pd.DataFrame({
            'Latitude': [latitude],
            'Longitude': [longitude],
            'Pop_Density_2020': [pop_density],
            'Wind_Speed': [wind_speed],
            'Grid_Value': [grid_value_encoded],
            'Income_Distribution': [income_distribution_encoded]
        })

        # Scale features
        scaled_features = scaler.transform(input_data)

        # Apply KMeans clustering
        cluster = kmeans.predict(scaled_features)[0]
        input_data['Cluster'] = cluster

        # Add cluster to scaled features
        final_features = np.hstack([scaled_features, [[cluster]]])

        # Predict with MLP
        viability_prediction = mlp.predict(final_features)[0]
        viability_label = viability_map.get(str(viability_prediction), "Unknown")

        # Display results
        st.subheader("Prediction Result")
        st.write(f"**Viability:** {viability_label}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
