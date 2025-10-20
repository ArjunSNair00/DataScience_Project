import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# Set page configuration
st.set_page_config(page_title="House Price Predictor", layout="wide")

# Title and description
st.title("House Price Prediction App")
st.write("Enter the details of the house to get a price prediction")

def load_model_and_scaler():
    try:
        # Load the trained model and scaler
        model = joblib.load('house_price_model.joblib')
        scaler = joblib.load('house_price_scaler.joblib')
        features_list = joblib.load('house_price_features.joblib')
        return model, scaler, features_list
    except:
        st.error("Model files not found. Please run the training notebook first!")
        return None, None, None

def make_prediction(input_data, model, scaler, features_list):
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply one-hot encoding
    input_encoded = pd.get_dummies(input_df, columns=['mainroad', 'guestroom', 'basement', 
                                                     'hotwaterheating', 'airconditioning', 
                                                     'prefarea', 'furnishingstatus'])
    
    # Ensure all columns from training are present
    for col in features_list:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[features_list]
    
    # Scale the features
    input_scaled = scaler.transform(input_encoded)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    return prediction

def main():
    model, scaler, features_list = load_model_and_scaler()
    if model is None:
        return

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Numerical Features")
        area = st.number_input("Area (sq ft)", min_value=1000, max_value=30000, value=7000)
        bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Number of bathrooms", min_value=1, max_value=10, value=2)
        stories = st.number_input("Number of stories", min_value=1, max_value=10, value=2)
        parking = st.number_input("Number of parking spaces", min_value=0, max_value=10, value=1)

    with col2:
        st.subheader("Property Features")
        mainroad = st.selectbox("Is it on the main road?", ["yes", "no"])
        guestroom = st.selectbox("Does it have a guest room?", ["yes", "no"])
        basement = st.selectbox("Does it have a basement?", ["yes", "no"])
        hotwaterheating = st.selectbox("Does it have hot water heating?", ["yes", "no"])
        airconditioning = st.selectbox("Does it have air conditioning?", ["yes", "no"])
        prefarea = st.selectbox("Is it in a preferred area?", ["yes", "no"])
        furnishingstatus = st.selectbox("Furnishing status", 
                                      ["furnished", "semi-furnished", "unfurnished"])

    # Create a dictionary with all inputs
    input_data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus
    }

    # Add a predict button
    if st.button("Predict Price"):
        prediction = make_prediction(input_data, model, scaler, features_list)
        
        # Display the prediction in a nice format
        st.subheader("Prediction Results")
        st.write("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(label="Predicted House Price", value=f"₹{prediction:,.2f}")

if __name__ == "__main__":
    main()