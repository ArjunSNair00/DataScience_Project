import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np
import ast

skills_dict = {
    0: "Programming Languages",
    1: "Math & Statistics",
    2: "Machine Learning & AI",
    3: "ML Frameworks & Libraries",
    4: "Big Data & Data Engineering",
    5: "Databases",
    6: "Cloud & DevOps",
    7: "Data Analysis & BI",
    8: "MLOps & Deployment",
    9: "Systems & HPC",
    10: "Other / Domain"
}

try:
    model = joblib.load('model_export/job_recommendation_model.joblib')
    jobs_data = pd.read_csv('model_export/jobs_processed.csv')
    jobs_desc_url_data= pd.read_csv('job_desc_url.csv')
    with open('model_export/skill_categories.pkl', 'rb') as f:
        skill_categories = pickle.load(f)
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

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
        for category, skills in skill_categories.items():
            if skill in [s.lower() for s in skills]:
                user_categories.append(category)
                break
    
    return user_skills, user_categories

def calculate_skills_count(categories):
    """Calculate count of skills per category"""
    counts = np.zeros(11)  
    for cat in categories:
        counts[cat] += 1
    return counts.tolist()

def recommend_jobs(user_input, jobs_data, model, top_n=5, location_filter=None, min_salary=None, max_salary=None, sort_by="Relevance (Default)"):
    """Generate job recommendations based on user input"""
    user_skills, user_categories = extract_skills_with_categories(user_input, skill_categories)
    
    user_counts = calculate_skills_count(user_categories)
    user_vector = np.array(user_counts).reshape(1, -1)
    
    jobs_data['skills'] = jobs_data['skills'].apply(safe_eval)
    jobs_data['skill_categories'] = jobs_data['skill_categories'].apply(safe_eval)
    jobs_data['skills_count_single'] = jobs_data['skill_categories'].apply(calculate_skills_count)
    
    job_vectors = np.array(jobs_data['skills_count_single'].tolist())
    
    match_probs = model.predict_proba(job_vectors)[:, 1]
    
    skill_overlap_scores = []
    for job_vector in job_vectors:
        overlap = sum((user_vector[0] > 0) & (job_vector > 0))
        total = sum((user_vector[0] > 0) | (job_vector > 0))
        skill_overlap_scores.append(overlap / total if total > 0 else 0)
    
    final_scores = 0.7 * match_probs + 0.3 * np.array(skill_overlap_scores)
    
    results_df = jobs_data.copy()
    results_df['match_score'] = final_scores * 100 
    
    if location_filter:
        results_df = results_df[results_df['location'].str.contains(location_filter, case=False, na=False)]
    
    if min_salary is not None:
        results_df = results_df[results_df['min_salary'] >= min_salary]
    
    if max_salary is not None:
        results_df = results_df[results_df['max_salary'] <= max_salary]
    
    if sort_by == "Salary (High to Low)":
        results_df = results_df.sort_values(by=['max_salary', 'match_score'], ascending=[False, False])
    elif sort_by == "Salary (Low to High)":
        results_df = results_df.sort_values(by=['min_salary', 'match_score'], ascending=[True, False])
    elif sort_by == "Company Name (A-Z)":
        results_df = results_df.sort_values(by=['company', 'match_score'], ascending=[True, False])
    else:  
        results_df = results_df.sort_values(by='match_score', ascending=False)
    
    top_results = results_df.head(top_n)
    
    recommendations = []
    for _, job in top_results.iterrows():
        job_info = jobs_desc_url_data.iloc[int(job.name)]
        recommendations.append({
            'position': job['positionName'],
            'company': job['company'],
            'location': job['location'],
            'match_score': round(job['match_score'], 2),
            'salary_range': f"${job['min_salary']:,.2f} - ${job['max_salary']:,.2f}",
            'required_skills': job['skills'],
            'matched_skills': [skill for skill in user_skills if skill in job['skills']],
            'missing_skills': [skill for skill in job['skills'] if skill not in user_skills],
            'url': job_info['url'],
            'description': job_info['description']
        })
    
    return recommendations

def get_all_skills():
    """Get all unique skills from skill categories"""
    all_skills = []
    for category_skills in skill_categories.values():
        all_skills.extend(category_skills)
    return sorted(list(set(all_skills)))

def get_all_locations():
    """Get all unique locations from jobs data"""
    return sorted(list(jobs_data['location'].dropna().unique()))

def update_skills_text(skills_list):
    """Convert skills list to comma-separated string"""
    return ", ".join(skills_list) if skills_list else ""

def update_skills_selection(skills_text):
    """Convert comma-separated string to list of skills"""
    return [s.strip() for s in skills_text.split(",")] if skills_text else []

if 'selected_skills' not in st.session_state:
    st.session_state.selected_skills = []
if 'skills_text' not in st.session_state:
    st.session_state.skills_text = ""

# Streamlit UI
st.set_page_config(page_title="AI Job Recommender", layout="wide")
st.title("💼 Job Recommendation System Using ML")
st.write("This system uses random forest classifier trained on 700+ AI or data jobs to match your skills with job opportunities!")

st.sidebar.header("🔎 Skills Filter")

skills_search = st.sidebar.text_input("Search skills:", key="skills_search")
all_skills = get_all_skills()
filtered_skills = [skill for skill in all_skills if skills_search.lower() in skill.lower()] if skills_search else all_skills

st.sidebar.markdown("### Select Skills by Category")
for category, category_name in skills_dict.items():
    with st.sidebar.expander(f"📌 {category_name}"):
        category_skills = skill_categories[category]
        filtered_category_skills = [skill for skill in category_skills if skill in filtered_skills]
        for skill in filtered_category_skills:
            if st.checkbox(skill, value=skill in st.session_state.selected_skills):
                if skill not in st.session_state.selected_skills:
                    st.session_state.selected_skills.append(skill)
            else:
                if skill in st.session_state.selected_skills:
                    st.session_state.selected_skills.remove(skill)

if 'location_filter' not in st.session_state:
    st.session_state.location_filter = ""
if 'location_input' not in st.session_state:
    st.session_state.location_input = ""
if 'selected_locations' not in st.session_state:
    st.session_state.selected_locations = []
if 'sidebar_locations' not in st.session_state:
    st.session_state.sidebar_locations = []

st.sidebar.markdown("### 📍 Location Filter")
all_locations = get_all_locations()
location_search = st.sidebar.text_input("Search locations:", key="sidebar_location_search")
filtered_locations = [loc for loc in all_locations if location_search.lower() in loc.lower()] if location_search else all_locations

with st.sidebar.expander("Select Locations"):
    st.session_state.sidebar_locations = [loc for loc in filtered_locations 
                                        if st.checkbox(loc, key=f"sidebar_loc_{loc}",
                                                     value=loc in st.session_state.selected_locations)]
    
if st.session_state.sidebar_locations:
    st.session_state.location_filter = "|".join(st.session_state.sidebar_locations)
    st.session_state.selected_locations = st.session_state.sidebar_locations.copy()

st.subheader("🎯 Your Skills")

if st.session_state.selected_skills != update_skills_selection(st.session_state.get('skills_text', '')):
    st.session_state.skills_text = update_skills_text(st.session_state.selected_skills)

user_skills = st.text_area(
    "Enter or modify your skills: (or use the sidebar to select skills)",
    value=st.session_state.skills_text,
    height=100,
    help="Enter skills separated by commas, or use the sidebar to select skills by category",
    key="skills_input"
)

if user_skills != st.session_state.skills_text:
    st.session_state.selected_skills = update_skills_selection(user_skills)
    st.session_state.skills_text = user_skills

with st.expander("⚙️ Advanced Settings", expanded=False):
    st.markdown("### 💰 Salary Range")
    salary_range = st.slider(
        "Select your desired salary range:",
        min_value=int(jobs_data['min_salary'].min()),
        max_value=int(jobs_data['max_salary'].max()),
        value=(int(jobs_data['min_salary'].min()), int(jobs_data['max_salary'].max())),
        step=10000,
        format="$%d"
    )
    
    st.markdown("### 📍 Location Preferences")
    location_col1, location_col2 = st.columns([1, 1])
    
    with location_col1:
        st.session_state.location_input = st.text_input(
            "Enter location keywords:",
            value=st.session_state.location_input,
            help="Enter cities or states separated by commas, or use the sidebar to select locations"
        )
        if st.session_state.location_input and not st.session_state.sidebar_locations:
            st.session_state.location_filter = st.session_state.location_input
    
    with location_col2:
        if st.session_state.sidebar_locations:
            st.markdown("**Currently Selected Locations:**")
            st.write(", ".join(st.session_state.sidebar_locations))
        
        matching_locations = [loc for loc in get_all_locations() 
                            if any(term.lower() in loc.lower() 
                                  for term in st.session_state.location_input.split(','))] if st.session_state.location_input else []
        if matching_locations:
            st.markdown("**Matching locations from search:**")
            main_selected = [loc for loc in matching_locations 
                           if st.checkbox(loc, key=f"main_loc_{loc}",
                                        value=loc in st.session_state.sidebar_locations)]
            if main_selected:
                st.session_state.sidebar_locations = main_selected
                st.session_state.location_filter = "|".join(main_selected)

    st.markdown("### 🔄 Results Settings")
    sort_col1, sort_col2 = st.columns([1, 1])
    
    with sort_col1:
        sort_option = st.selectbox(
            "Sort results by:",
            ["Relevance (Default)", "Salary (High to Low)", 
             "Salary (Low to High)", "Company Name (A-Z)"]
        )

    with sort_col2:
        num_recommendations = st.slider(
            "Number of recommendations:", 
            min_value=1, 
            max_value=20, 
            value=5
        )

if st.button("🔍 Get Job Recommendations"):
    if st.session_state.selected_skills or user_skills.strip():
        recommendations = recommend_jobs(
            user_skills or update_skills_text(st.session_state.selected_skills),
            jobs_data,
            model,
            top_n=num_recommendations,
            location_filter=st.session_state.location_filter,
            min_salary=salary_range[0],
            max_salary=salary_range[1],
            sort_by=sort_option
        )
        
        if recommendations:
            st.subheader("📋 Job Recommendations")
            for i, rec in enumerate(recommendations, 1):
                with st.expander(f"{i}. {rec['position']} at {rec['company']} - Match Score: {rec['match_score']}%"):
                    st.markdown("### 📋 Basic Information")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**🌍Location:** {rec['location']}")
                        st.write(f"**💵Salary Range:** {rec['salary_range']}")
                        st.write("**📜Required Skills:**", ", ".join(rec['required_skills']))
                        if rec['matched_skills']:
                            st.write("**✅ Matched Skills:**", ", ".join(rec['matched_skills']))
                        if rec['missing_skills']:
                            st.write("**📚 Skills to Learn:**", ", ".join(rec['missing_skills']))
                    with col2:
                        st.write("**Job Links:**")
                        st.markdown(f"[Apply on Indeed]({rec['url']})")
                    
                    st.markdown("### 📝 Job Description")
                    st.write(rec['description'])
        else:
            st.warning("No matching jobs found. Try adjusting your filters or adding more skills.")
    else:
        st.warning("Please enter your skills to get recommendations.")

st.markdown("---")
st.markdown("Built with Streamlit and Random Forest machine learning model 🤖")

with st.expander("ℹ️ About", expanded=False):
    st.markdown("Project Repository: https://github.com/ArjunSNair00/DataScience_Project")
    st.markdown("Python Notebook: [Google Colab Notebook](https://colab.research.google.com/github/ArjunSNair00/DataScience_Project/blob/main/job_recommendation_system_final.ipynb)")
    team = """
    <div style="background-color: #262a36; padding: 10px; border-radius: 5px;">
        <h1>Team Members</h1>
        <h5>Group - 5</h5>
        <ul>
        <li>Arjun S Nair - KNP24AD017</li>
        <li>Akshay K Sasi - KNP24AD010</li>
        <li>Muhammed Raihan - KNP24AD035</li>
        <li>Abhishek S - KNP24AD002</li>
        </ul>
        <h3>Future Roadmap: </h3>
        <p>Resume parser integration, improved UI, expanded dataset across regions and domains, enhanced NLP tokenization</p>
    </div>
    """
    st.markdown(team, unsafe_allow_html=True)