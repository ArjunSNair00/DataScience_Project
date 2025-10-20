import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np
import ast

# ------------------------------
# Load pre-trained models & data
# ------------------------------
try:
    # Load the Random Forest model and processed data
    model = joblib.load('model_export/job_recommendation_model.joblib')
    jobs_data = pd.read_csv('model_export/jobs_processed.csv')
    with open('model_export/skill_categories.pkl', 'rb') as f:
        skill_categories = pickle.load(f)
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

# ------------------------------
# Helper Functions
# ------------------------------
def safe_eval(s):
    """Safely evaluate string representation of lists"""
    try:
        if isinstance(s, str):
            s = s.replace('[ ', '[').replace(' ]', ']').replace('  ', ' ')
            return ast.literal_eval(s)
        return s
    except:
        return None

def extract_skills_with_categories(user_input, skill_categories):
    """Extract skills and their categories from user input"""
    user_skills = [skill.strip().lower() for skill in user_input.split(',')]
    user_categories = []
    
    for skill in user_skills:
        # Find category for the skill
        for category, skills in skill_categories.items():
            if skill in [s.lower() for s in skills]:
                user_categories.append(category)
                break
    
    return user_skills, user_categories

def calculate_skills_count(categories):
    """Calculate count of skills per category"""
    counts = np.zeros(11)  # 11 categories (0-10)
    for cat in categories:
        counts[cat] += 1
    return counts.tolist()

def recommend_jobs(user_input, jobs_data, model, top_n=5, location_filter=None, min_salary=None, max_salary=None, sort_by="Relevance (Default)"):
    """Generate job recommendations based on user input"""
    # Extract skills and categories from user input
    user_skills, user_categories = extract_skills_with_categories(user_input, skill_categories)
    
    # Calculate skill counts for user input
    user_counts = calculate_skills_count(user_categories)
    user_vector = np.array(user_counts).reshape(1, -1)
    
    # Ensure jobs_data has proper format
    jobs_data['skills'] = jobs_data['skills'].apply(safe_eval)
    jobs_data['skill_categories'] = jobs_data['skill_categories'].apply(safe_eval)
    jobs_data['skills_count_single'] = jobs_data['skill_categories'].apply(calculate_skills_count)
    
    # Create job vectors
    job_vectors = np.array(jobs_data['skills_count_single'].tolist())
    
    # Get match probabilities
    match_probs = model.predict_proba(job_vectors)[:, 1]
    
    # Calculate skill overlap scores
    skill_overlap_scores = []
    for job_vector in job_vectors:
        overlap = sum((user_vector[0] > 0) & (job_vector > 0))
        total = sum((user_vector[0] > 0) | (job_vector > 0))
        skill_overlap_scores.append(overlap / total if total > 0 else 0)
    
    # Combine scores
    final_scores = 0.7 * match_probs + 0.3 * np.array(skill_overlap_scores)
    
    # Create DataFrame with scores
    results_df = jobs_data.copy()
    results_df['match_score'] = final_scores * 100  # Convert to percentage
    
    # Apply filters
    if location_filter:
        results_df = results_df[results_df['location'].str.contains(location_filter, case=False, na=False)]
    
    if min_salary is not None:
        results_df = results_df[results_df['min_salary'] >= min_salary]
    
    if max_salary is not None:
        results_df = results_df[results_df['max_salary'] <= max_salary]
    
    # Apply sorting
    if sort_by == "Salary (High to Low)":
        results_df = results_df.sort_values(by=['max_salary', 'match_score'], ascending=[False, False])
    elif sort_by == "Salary (Low to High)":
        results_df = results_df.sort_values(by=['min_salary', 'match_score'], ascending=[True, False])
    elif sort_by == "Company Name (A-Z)":
        results_df = results_df.sort_values(by=['company', 'match_score'], ascending=[True, False])
    else:  # Default: Sort by match score
        results_df = results_df.sort_values(by='match_score', ascending=False)
    
    # Get top N results
    top_results = results_df.head(top_n)
    
    # Create recommendations list
    recommendations = []
    for _, job in top_results.iterrows():
        recommendations.append({
            'position': job['positionName'],
            'company': job['company'],
            'location': job['location'],
            'match_score': round(job['match_score'], 2),
            'salary_range': f"${job['min_salary']:,.2f} - ${job['max_salary']:,.2f}",
            'required_skills': job['skills'],
            'matched_skills': [skill for skill in user_skills if skill in job['skills']],
            'missing_skills': [skill for skill in job['skills'] if skill not in user_skills]
        })
    
    return recommendations

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="AI Job Recommender", layout="wide")
st.title("🤖 AI-Powered Job Recommendation System")
st.write("This system uses machine learning to match your skills with job opportunities!")

# Sidebar for filters
st.sidebar.header("Filters")

# Location filter
location_filter = st.sidebar.text_input("Filter by location:", "")

# Salary range filter
salary_range = st.sidebar.slider(
    "Salary Range ($)",
    min_value=int(jobs_data['min_salary'].min()),
    max_value=int(jobs_data['max_salary'].max()),
    value=(int(jobs_data['min_salary'].min()), int(jobs_data['max_salary'].max())),
    step=10000
)

# Main area for skill input
st.subheader("Enter Your Skills")
st.write("Enter your skills separated by commas (e.g., Python, SQL, Machine Learning)")
user_skills = st.text_area("Your skills:", height=100)

# Options row
col1, col2 = st.columns([2, 2])

with col1:
    sort_option = st.selectbox(
        "Sort by:",
        ["Relevance (Default)", "Salary (High to Low)", "Salary (Low to High)", "Company Name (A-Z)"]
    )

with col2:
    num_recommendations = st.slider("Number of recommendations:", min_value=1, max_value=20, value=5)

if st.button("Get Job Recommendations"):
    if user_skills.strip():
        recommendations = recommend_jobs(
            user_skills,
            jobs_data,
            model,
            top_n=num_recommendations,
            location_filter=location_filter,
            min_salary=salary_range[0],
            max_salary=salary_range[1],
            sort_by=sort_option
        )
        
        if recommendations:
            st.subheader("📋 Job Recommendations")
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec['position']} at {rec['company']} - Match Score: {rec['match_score']}%"):
                    st.write(f"**Location:** {rec['location']}")
                    st.write(f"**Salary Range:** {rec['salary_range']}")
                    st.write("**Required Skills:**", ", ".join(rec['required_skills']))
                    if rec['matched_skills']:
                        st.write("**✅ Matched Skills:**", ", ".join(rec['matched_skills']))
                    if rec['missing_skills']:
                        st.write("**📚 Skills to Learn:**", ", ".join(rec['missing_skills']))
        else:
            st.warning("No matching jobs found. Try adjusting your filters or adding more skills.")
    else:
        st.warning("Please enter your skills to get recommendations.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Random Forest machine learning model 🤖")