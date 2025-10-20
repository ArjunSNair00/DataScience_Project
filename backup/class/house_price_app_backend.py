from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__, template_folder='templates')

# Load the model and scaler
try:
    model = joblib.load('house_price_model.joblib')
    scaler = joblib.load('house_price_scaler.joblib')
    features_list = joblib.load('house_price_features.joblib')
except:
    print("Error: Model files not found! Please run the training notebook first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Create input dictionary
        input_data = {
            'area': float(data['area']),
            'bedrooms': int(data['bedrooms']),
            'bathrooms': int(data['bathrooms']),
            'stories': int(data['stories']),
            'parking': int(data['parking']),
            'mainroad': data['mainroad'].lower(),
            'guestroom': data['guestroom'].lower(),
            'basement': data['basement'].lower(),
            'hotwaterheating': data['hotwaterheating'].lower(),
            'airconditioning': data['airconditioning'].lower(),
            'prefarea': data['prefarea'].lower(),
            'furnishingstatus': data['furnishingstatus'].lower()
        }

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
        
        return jsonify({
            'success': True,
            'prediction': f'{prediction:,.2f}'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)