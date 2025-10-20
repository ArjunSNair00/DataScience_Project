# app.py
import streamlit as st
import joblib
import numpy as np
import os

# Load model from the folder where app.py is located
current_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_folder, 'diabetes_model.pkl')
model = joblib.load(model_path)

st.title("🩺 Diabetes Prediction App")

st.write("Enter the following details:")

# User inputs
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

if st.button("Predict"):
    # Fill missing features with default values
    features = np.array([[
        2,       # Pregnancies (default)
        glucose, # Glucose
        bp,      # BloodPressure
        30,      # SkinThickness (default)
        100,     # Insulin (default)
        bmi,     # BMI
        0.5,     # DiabetesPedigreeFunction (default)
        age      # Age
    ]])
    
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.error("⚠️ Diabetes predicted!")
    else:
        st.success("✅ No Diabetes predicted")
