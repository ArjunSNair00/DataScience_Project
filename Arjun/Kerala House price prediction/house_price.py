import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Feature Engineering Function
def prepare_features(df):
    df['bathrooms_ratio'] = df['bathrooms'] / df['area']
    df['total_rooms_est'] = df['bathrooms'] + df['balconies'] + 1  # +1 for living room
    df['households_est'] = 1
    df['household_rooms'] = df['total_rooms_est'] / df['households_est']
    
    # Apply log transformation to numerical features
    for col in ['age', 'amneties', 'area', 'atmDistance', 'balconies', 'bathrooms',
                'hospitalDistance', 'restrauntDistance', 'schoolDistance', 'shoppingDistance', 'status']:
        df[col] = np.log(df[col] + 1)
    
    return df

# Load and train model
@st.cache_resource
def get_model():
    data = pd.read_csv('data.csv')
    data = prepare_features(data)
    X = data.drop('price', axis=1)
    y = data['price']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

model = get_model()

# Streamlit UI
st.title("Kerala House Price Predictor")

# User inputs
age = st.number_input("Age of Property (years)", min_value=0)
amneties = st.number_input("Amneties Score", min_value=0)
area = st.number_input("Area (sqft)", min_value=100)
atmDistance = st.number_input("ATM Distance (km)", min_value=0.0)
balconies = st.number_input("Number of Balconies", min_value=0)
bathrooms = st.number_input("Number of Bathrooms", min_value=0)
hospitalDistance = st.number_input("Hospital Distance (km)", min_value=0.0)
restrauntDistance = st.number_input("Restaurant Distance (km)", min_value=0.0)
schoolDistance = st.number_input("School Distance (km)", min_value=0.0)
shoppingDistance = st.number_input("Shopping Complex Distance (km)", min_value=0.0)
status = st.selectbox("Status", options=[1, 2, 3], format_func=lambda x: {1:'Unfurnished', 2:'Under Construction', 3:'Ready to Move'}[x])

# Prepare input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'amneties': [amneties],
    'area': [area],
    'atmDistance': [atmDistance],
    'balconies': [balconies],
    'bathrooms': [bathrooms],
    'hospitalDistance': [hospitalDistance],
    'restrauntDistance': [restrauntDistance],
    'schoolDistance': [schoolDistance],
    'shoppingDistance': [shoppingDistance],
    'status': [status]
})

input_df = prepare_features(input_df)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated House Price: ₹{prediction:,.0f}")
