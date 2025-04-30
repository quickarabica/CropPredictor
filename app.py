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
st.title("🌱 QCrop Predictor")

# Model selection
model_choice = st.selectbox(
    "Choose Prediction Model:",
    ("Variational Quantum Classification", "Quantum Neural Network", "Quantum K-Nearest Neighbour")
)

st.header("Enter Soil & Weather Features")

# Input fields for 7 features
def user_input():
    N = st.number_input("Nitrogen (N)", value=90.0)
    P = st.number_input("Phosphorus (P)", value=42.0)
    K = st.number_input("Potassium (K)", value=43.0)
    temperature = st.number_input("Temperature", value=20.0)
    humidity = st.number_input("Humidity", value=82.0)
    ph = st.number_input("pH Value", value=6.5)
    rainfall = st.number_input("Rainfall", value=202.9)
    return [N, P, K, temperature, humidity, ph, rainfall]

features = user_input()

# Function to load crop image
def load_crop_image(crop_name):
    # Assuming image file name format is exactly same as crop name (lowercase) e.g. rice.png, wheat.png
    img_path = f"QCropPredictor/assests/{crop_name.lower()}.png"
    if os.path.exists(img_path):
        return Image.open(img_path)
    else:
        return None

# Prediction button
def main():
    if st.button("Predict Crop 🚀"):
        if model_choice == "Variational Quantum Classification":
            prediction = predict_vqc(features, vqc_model, vqc_scaler, vqc_class_names)
        elif model_choice == "Quantum Neural Network":
            arr = np.array(features).reshape(1, -1)
            prediction = predict_qnn(qnn_model, qnn_scaler, qnn_label_encoder, arr)[0]
        elif model_choice == "Quantum K-Nearest Neighbour":
            prediction = predict_qknn(features, vqc_model, vqc_scaler, vqc_class_names)
        
        st.success(f"**The predicted according to your inputs is: ** {prediction}")

        # Display crop image
        crop_image = load_crop_image(prediction)
        if crop_image:
            st.image(crop_image, caption=prediction, use_column_width=True)
        else:
            st.warning(f"No image available for this crop: {prediction}")

if __name__ == "__main__":
    main()
