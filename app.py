import streamlit as st
import numpy as np
from PIL import Image
import os

from vqc.vqc_model import load_vqc_model, predict_vqc
from qnn.qnn_model import load_qnn_model, predict_qnn
from qknn.qknn_model import load_qknn_model, predict_qknn


# Cache model loading to avoid reloading on every interaction
def load_models():
    vqc_model, vqc_scaler, vqc_class_names = load_vqc_model()
    qnn_model, qnn_scaler, qnn_label_encoder = load_qnn_model()
    qknn_model, qknn_scaler, qknn_label_encoder = load_qknn_model()
    return vqc_model, vqc_scaler, vqc_class_names, qnn_model, qnn_scaler, qnn_label_encoder, qknn_model, qknn_scaler, qknn_label_encoder

vqc_model, vqc_scaler, vqc_class_names, qnn_model, qnn_scaler, qnn_label_encoder, qknn_model, qknn_scaler, qknn_label_encoder = load_models()

# App title
st.markdown("""
    <div style='text-align: center;'>
        <h1>🌱 <strong>CropQast</strong></h1>
        <h3>Quantum-Powered Crop Predictor 🌾</h3>
        <h4>Let Quantum Models Do the Farming Math – Predict the Best Crop Based on Soil & Weather Conditions!</h4>
    </div>
""", unsafe_allow_html=True)

# Custom CSS to enhance appearance
st.markdown("""
    <style>
        .container {
            max-width: 900px;
            margin: auto;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .section-header {
            color: #3A8D4D;
            font-size: 24px;
            font-weight: bold;
            border-bottom: 2px solid #3A8D4D;
            padding-bottom: 5px;
            margin-bottom: 20px;
        }
        .model-selector, .predict-button {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }
        .model-selector select, .predict-button button {
            padding: 10px;
            font-size: 16px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .input-column {
            padding: 15px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .input-column > div {
            margin-bottom: 10px;
        }
        .crop-image {
            margin-top: 20px;
            text-align: center;
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""<div style='text-align: center;'>
            <h3>Available Models:</h3>
             </div>
""", unsafe_allow_html=True)

# Create 3 columns for the 3 buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Variational Quantum Classification"):
        st.session_state.model_choice = "Variational Quantum Classification"

with col2:
    if st.button("Quantum Neural Network"):
        st.session_state.model_choice = "Quantum Neural Network"

with col3:
    if st.button("Quantum K-Nearest Neighbour"):
        st.session_state.model_choice = "Quantum K-Nearest Neighbour"

# Set a default if nothing is selected yet
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Variational Quantum Classification"

# Display selected model
st.success(f"Selected Model: **{st.session_state.model_choice}**")


st.markdown('<div class="section-header">🌍 Soil Features: </div>', unsafe_allow_html=True)
def soil_input():
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen (N)", value=90.0, min_value=0.0)
    with col2:
        P = st.number_input("Phosphorus (P)", value=42.0, min_value=0.0)
    with col3:
        K = st.number_input("Potassium (K)", value=43.0, min_value=0.0)
    return [N, P, K]

soil_features = soil_input()

st.markdown('<div class="section-header">🌤 Weather Features: </div>', unsafe_allow_html=True)
def weather_input():
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("Temperature (°C)", value=20.0, min_value=-30.0, max_value=60.0)
    with col2:
        humidity = st.number_input("Humidity (%)", value=82.0, min_value=0.0, max_value=100.0)
    col1, col2 = st.columns(2)
    with col1:
        ph = st.number_input("Soil pH Value", value=6.5, min_value=0.0, max_value=14.0)
    with col2:
        rainfall = st.number_input("Rainfall (mm)", value=202.9, min_value=0.0)
    return [temperature, humidity, ph, rainfall]

weather_features = weather_input()

# Combine both soil and weather features
features = soil_features + weather_features

# Function to load crop image
def load_crop_image(crop_name):
    img_path = rf"C:\Users\priya\Desktop\CropPredictor\assests\{crop_name.lower()}.jpg"
    if os.path.exists(img_path):
        return Image.open(img_path)
    else:
        return None

# Prediction button
def main():
    model_choice = st.session_state.model_choice
    st.markdown('<div class="predict-button">', unsafe_allow_html=True)
    if st.button("Predict Crop 🚀"):
        with st.spinner("Quantum models are calculating the best crop for you..."):
            if model_choice == "Variational Quantum Classification":
                prediction = predict_vqc(features, vqc_model, vqc_scaler, vqc_class_names)
            elif model_choice == "Quantum Neural Network":
                arr = np.array(features).reshape(1, -1)
                prediction = predict_qnn(qnn_model, qnn_scaler, qnn_label_encoder, arr)[0]
            elif model_choice == "Quantum K-Nearest Neighbour":
                prediction = predict_qknn(features, qknn_model, qknn_scaler, qknn_label_encoder)
            st.subheader(f"{model_choice}'s Prediction Result")
            st.success(f"The predicted crop according to your input is **{prediction.upper()}**")

            # Display crop image
            crop_image = load_crop_image(prediction)
            if crop_image:
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                st.image(crop_image, caption=f"Predicted Crop: {prediction.upper()}", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning(f"No image available for this crop: {prediction}")

    

st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
