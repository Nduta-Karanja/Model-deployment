import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

# Load models and utilities
mlp = joblib.load('models/mlp_model.pkl')
kmeans = joblib.load('models/kmeans_model.pkl')
income_encoder = joblib.load('models/income_encoder.pkl')
grid_encoder = joblib.load('models/grid_encoder.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load metadata
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

features_to_scale = metadata['features_to_scale']
viability_map = metadata['viability_map']

# Streamlit App
st.title("Electrification Viability Prediction")
st.write("Determine whether a location is viable for grid electrification, wind microgrid, or neither.")

# User inputs
latitude = st.number_input("Latitude", value=0.0, step=0.1)
longitude = st.number_input("Longitude", value=0.0, step=0.1)
pop_density = st.number_input("Population Density (2020)", value=100, step=1)
wind_speed = st.number_input("Wind Speed (m/s)", value=5.0, step=0.1)
grid_value = st.selectbox("Grid Value", grid_encoder.classes_)
income_distribution = st.selectbox("Income Distribution", income_encoder.classes_)

# Predict button
if st.button("Predict"):
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
    viability_label = viability_map[str(viability_prediction)]

    # Display results
    st.subheader("Prediction Result")
    st.write(f"**Viability:** {viability_label}")
