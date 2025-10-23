import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

# Set page config
st.set_page_config(
    page_title="Job Position Classifier",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("🧠 AI Job Position Classifier")
st.markdown("""
This app predicts the job position category based on skills, company, location, and salary information.
Upload a CSV file or enter job details manually to get predictions.
""")

# Load the saved model
@st.cache_resource
def load_model():
    try:
        model_assets = joblib.load('random_forest_position_classifier.pkl')
        return model_assets
    except FileNotFoundError:
        st.error("❌ Model file 'random_forest_position_classifier.pkl' not found!")
        return None

# Process skills input
def process_skills(skills_text):
    """Convert skills string to processed format"""
    skills_list = [skill.strip() for skill in skills_text.split(',')]
    return ' '.join([str(skill).lower().replace(' ', '_') for skill in skills_list])

# Make predictions
def predict_position(model_assets, input_data):
    """Make prediction using the loaded model"""
    try:
        # Extract components
        model = model_assets['model']
        tfidf_vectorizer = model_assets['tfidf_vectorizer']
        svd = model_assets['svd']
        scaler = model_assets['scaler']
        
        # Process skills text
        skills_text = process_skills(input_data['skills'])
        skills_vectorized = tfidf_vectorizer.transform([skills_text])
        skills_reduced = svd.transform(skills_vectorized)
        
        # Prepare numeric features in correct order
        numeric_features = [
            'company_encoded', 'city_encoded', 'position_encoded',
            'total_skills', 'skills_count_all', 'min_salary', 'max_salary', 'average_salary',
            'salary_range', 'salary_midpoint',
            'has_scientist', 'has_engineer', 'has_analyst', 'has_architect', 'has_research'
        ]
        
        # Create numeric input array
        numeric_input = []
        for feature in numeric_features:
            if feature in input_data:
                numeric_input.append(input_data[feature])
            else:
                numeric_input.append(0)  # Default value for missing features
        
        # Scale numeric features
        numeric_scaled = scaler.transform([numeric_input])
        
        # Combine features
        combined_features = np.hstack([skills_reduced, numeric_scaled])
        
        # Make prediction
        prediction = model.predict(combined_features)[0]
        probabilities = model.predict_proba(combined_features)[0]
        
        # Get confidence score
        confidence = np.max(probabilities)
        
        # Get all class probabilities
        class_probabilities = {
            class_name: prob for class_name, prob in zip(model.classes_, probabilities)
        }
        
        return prediction, confidence, class_probabilities
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None, None, None

# Main app
def main():
    # Load model
    model_assets = load_model()
    if model_assets is None:
        return
    
    st.sidebar.header("🔧 Prediction Options")
    prediction_mode = st.sidebar.radio(
        "Choose input method:",
        ["Single Prediction", "Batch Prediction (CSV)"]
    )
    
    if prediction_mode == "Single Prediction":
        st.header("📝 Single Job Prediction")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Job Details")
                skills = st.text_area(
                    "Skills (comma-separated)",
                    placeholder="python, machine learning, data science, sql...",
                    help="Enter skills separated by commas"
                )
                company_encoded = st.number_input(
                    "Company Encoded",
                    min_value=0,
                    value=174,  # Default: Google
                    help="Encoded company ID"
                )
                city_encoded = st.number_input(
                    "City Encoded", 
                    min_value=0,
                    value=167,  # Default: San Bruno
                    help="Encoded city ID"
                )
                position_encoded = st.number_input(
                    "Position Encoded",
                    min_value=0,
                    value=427,
                    help="Encoded position ID"
                )
                
            with col2:
                st.subheader("Skills & Salary")
                total_skills = st.number_input(
                    "Total Skills Count",
                    min_value=0,
                    value=10
                )
                skills_count_all = st.number_input(
                    "Skills Count All",
                    min_value=0,
                    value=9
                )
                min_salary = st.number_input(
                    "Minimum Salary ($)",
                    min_value=0,
                    value=100000
                )
                max_salary = st.number_input(
                    "Maximum Salary ($)",
                    min_value=0,
                    value=150000
                )
                average_salary = st.number_input(
                    "Average Salary ($)",
                    min_value=0,
                    value=125000
                )
            
            # Title-based features
            st.subheader("Job Title Indicators")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                has_scientist = st.checkbox("Has 'Scientist' in title")
                has_engineer = st.checkbox("Has 'Engineer' in title")
            with col4:
                has_analyst = st.checkbox("Has 'Analyst' in title")
                has_architect = st.checkbox("Has 'Architect' in title")
            with col5:
                has_research = st.checkbox("Has 'Research' in title")
            
            # Calculate derived features
            salary_range = max_salary - min_salary
            salary_midpoint = (min_salary + max_salary) / 2
            
            # Submit button
            submitted = st.form_submit_button("🔮 Predict Position")
        
        if submitted:
            if not skills.strip():
                st.warning("⚠️ Please enter skills!")
                return
            
            # Prepare input data
            input_data = {
                'skills': skills,
                'company_encoded': company_encoded,
                'city_encoded': city_encoded,
                'position_encoded': position_encoded,
                'total_skills': total_skills,
                'skills_count_all': skills_count_all,
                'min_salary': min_salary,
                'max_salary': max_salary,
                'average_salary': average_salary,
                'salary_range': salary_range,
                'salary_midpoint': salary_midpoint,
                'has_scientist': 1 if has_scientist else 0,
                'has_engineer': 1 if has_engineer else 0,
                'has_analyst': 1 if has_analyst else 0,
                'has_architect': 1 if has_architect else 0,
                'has_research': 1 if has_research else 0
            }
            
            # Make prediction
            with st.spinner("🤖 Analyzing job details..."):
                prediction, confidence, probabilities = predict_position(model_assets, input_data)
            
            if prediction:
                # Display results
                st.success("🎯 Prediction Complete!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction Result")
                    st.metric(
                        label="Predicted Position",
                        value=prediction,
                        delta=f"{confidence:.1%} confidence"
                    )
                    
                    # Confidence gauge
                    st.progress(float(confidence))
                    st.caption(f"Model Confidence: {confidence:.1%}")
                
                with col2:
                    st.subheader("Probability Distribution")
                    
                    # Sort probabilities
                    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                    
                    for position, prob in sorted_probs:
                        col_left, col_right = st.columns([3, 1])
                        with col_left:
                            st.write(position)
                        with col_right:
                            st.write(f"{prob:.1%}")
                
                # Show feature importance (if available)
                if 'feature_names' in model_assets:
                    st.subheader("🔝 Key Influencing Features")
                    feature_importance = pd.DataFrame({
                        'feature': model_assets['feature_names'],
                        'importance': model_assets['model'].feature_importances_
                    }).sort_values('importance', ascending=False).head(10)
                    
                    st.dataframe(feature_importance, use_container_width=True)
    
    else:  # Batch Prediction
        st.header("📊 Batch Prediction (CSV)")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with job data",
            type=['csv'],
            help="CSV should contain columns: skills, company_encoded, city_encoded, etc."
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(batch_data)} records")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(batch_data.head(), use_container_width=True)
                
                if st.button("🚀 Predict All Jobs"):
                    predictions = []
                    confidences = []
                    
                    with st.spinner("🤖 Processing batch predictions..."):
                        for idx, row in batch_data.iterrows():
                            input_data = row.to_dict()
                            prediction, confidence, _ = predict_position(model_assets, input_data)
                            
                            predictions.append(prediction)
                            confidences.append(confidence)
                    
                    # Add predictions to dataframe
                    batch_data['predicted_position'] = predictions
                    batch_data['prediction_confidence'] = confidences
                    
                    # Display results
                    st.subheader("📈 Prediction Results")
                    st.dataframe(batch_data[['predicted_position', 'prediction_confidence']], 
                                use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("📊 Prediction Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Jobs", len(batch_data))
                    with col2:
                        avg_confidence = np.mean(confidences)
                        st.metric("Average Confidence", f"{avg_confidence:.1%}")
                    with col3:
                        most_common = pd.Series(predictions).value_counts().index[0]
                        st.metric("Most Common Position", most_common)
                    
                    # Download results
                    csv = batch_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="job_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

# Model info sidebar
def sidebar_info():
    st.sidebar.header("ℹ️ Model Information")
    st.sidebar.info("""
    **Model**: Random Forest Classifier  
    **Accuracy**: 88.4%  
    **Features Used**:
    - Skills text (TF-IDF)
    - Company/Location encoding
    - Salary information
    - Title indicators
    """)
    
    st.sidebar.header("🎯 Example Inputs")
    st.sidebar.code("""
Skills: python, machine learning, sql
Company: 174 (Google)
City: 167 (San Bruno)
Min Salary: $100,000
Max Salary: $150,000
    """)

# Run the app
if __name__ == "__main__":
    sidebar_info()
    main()