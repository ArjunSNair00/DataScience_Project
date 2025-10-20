import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import ast

# ------------------------------
# Load pre-trained models & data
# ------------------------------
try:
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    kmeans_model = joblib.load('kmeans_model.pkl')
    jobs = pd.read_csv('clustered_jobs.csv')
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

# ------------------------------
# Helper function to format skills
# ------------------------------
def format_skills(skills):
    if isinstance(skills, str):
        try:
            skills_list = ast.literal_eval(skills)
            if isinstance(skills_list, list):
                return ", ".join(skills_list)
        except:
            return skills  # keep as string if parsing fails
    elif isinstance(skills, list):
        return ", ".join(skills)
    return "N/A"

# ------------------------------
# Helper function to parse skills list
# ------------------------------
def parse_skills(skills):
    if isinstance(skills, str):
        try:
            skills_list = ast.literal_eval(skills)
            if isinstance(skills_list, list):
                return [s.lower().strip() for s in skills_list]
        except:
            return [skills.lower().strip()]
    elif isinstance(skills, list):
        return [s.lower().strip() for s in skills]
    return []

# ------------------------------
# Helper function to find missing skills
# ------------------------------
def find_missing_skills(user_skills, job_skills):
    user_skills_set = set([s.lower().strip() for s in user_skills.split(',')])
    job_skills_list = parse_skills(job_skills)
    job_skills_set = set(job_skills_list)
    
    missing = job_skills_set - user_skills_set
    return list(missing)

# ------------------------------
# Extract unique skills and locations from dataset
# ------------------------------
def extract_unique_skills_and_locations(jobs_df):
    # Extract unique skills
    all_skills = set()
    for skills in jobs_df['skills']:
        try:
            skills_list = ast.literal_eval(skills) if isinstance(skills, str) else skills
            if isinstance(skills_list, list):
                all_skills.update([skill.strip().lower() for skill in skills_list])
        except:
            if isinstance(skills, str):
                all_skills.add(skills.strip().lower())
    
    # Extract unique locations
    locations = set()
    if 'city' in jobs_df.columns:
        locations.update(jobs_df['city'].dropna().unique())
    if 'location' in jobs_df.columns:
        locations.update(jobs_df['location'].dropna().unique())
    
    # Clean locations and skills
    unique_locations = sorted([str(loc) for loc in locations if loc and str(loc).strip() != ''])
    unique_skills = sorted([skill for skill in all_skills if skill and len(skill) > 1])
    
    return unique_skills, unique_locations

# ------------------------------
# Recommend jobs function with filters - FIXED VERSION
# ------------------------------
def recommend_jobs(user_input, top_n=5, location_filter=None, position_filter=None, min_salary=None, max_salary=None, selected_skills=None, selected_locations=None):
    # Transform user input
    user_input_vec = tfidf_vectorizer.transform([user_input])
    
    # Predict cluster
    user_cluster = kmeans_model.predict(user_input_vec)[0]
    
    # Filter jobs in that cluster
    cluster_jobs = jobs[jobs['cluster'] == user_cluster].copy()
    
    if cluster_jobs.empty:
        return pd.DataFrame()  # Return empty DataFrame instead of empty list

    # Apply text-based filters if provided
    if location_filter and location_filter.strip():
        location_filter = location_filter.strip()
        if 'city' in cluster_jobs.columns:
            cluster_jobs = cluster_jobs[cluster_jobs['city'].str.contains(location_filter, case=False, na=False)]
        elif 'location' in cluster_jobs.columns:
            cluster_jobs = cluster_jobs[cluster_jobs['location'].str.contains(location_filter, case=False, na=False)]
    
    if position_filter and position_filter.strip():
        position_filter = position_filter.strip()
        cluster_jobs = cluster_jobs[cluster_jobs['positionName'].str.contains(position_filter, case=False, na=False)]
    
    # Apply checkbox filters for locations
    if selected_locations:
        location_filtered_jobs = pd.DataFrame()
        for location in selected_locations:
            location_lower = location.lower()
            if 'city' in cluster_jobs.columns:
                location_matches = cluster_jobs[cluster_jobs['city'].str.lower().str.contains(location_lower, na=False)]
            elif 'location' in cluster_jobs.columns:
                location_matches = cluster_jobs[cluster_jobs['location'].str.lower().str.contains(location_lower, na=False)]
            else:
                location_matches = pd.DataFrame()
            
            location_filtered_jobs = pd.concat([location_filtered_jobs, location_matches])
        
        if not location_filtered_jobs.empty:
            cluster_jobs = location_filtered_jobs.drop_duplicates()
        else:
            cluster_jobs = pd.DataFrame()  # Return empty DataFrame if no matches
    
    # Apply checkbox filters for skills
    if selected_skills:
        skills_filtered_jobs = pd.DataFrame()
        for skill in selected_skills:
            skill_lower = skill.lower()
            skill_matches = cluster_jobs[cluster_jobs['skills'].str.lower().str.contains(skill_lower, na=False)]
            skills_filtered_jobs = pd.concat([skills_filtered_jobs, skill_matches])
        
        if not skills_filtered_jobs.empty:
            cluster_jobs = skills_filtered_jobs.drop_duplicates()
        else:
            cluster_jobs = pd.DataFrame()  # Return empty DataFrame if no matches
    
    if cluster_jobs.empty:
        return pd.DataFrame()  # Return empty DataFrame instead of empty list

    # Salary filtering
    if min_salary is not None or max_salary is not None:
        # Convert salary to numeric
        cluster_jobs['salary_numeric'] = pd.to_numeric(cluster_jobs.get('average_salary', 0), errors='coerce')
        
        if min_salary is not None:
            cluster_jobs = cluster_jobs[cluster_jobs['salary_numeric'] >= min_salary]
        
        if max_salary is not None:
            cluster_jobs = cluster_jobs[cluster_jobs['salary_numeric'] <= max_salary]
    
    if cluster_jobs.empty:
        return pd.DataFrame()  # Return empty DataFrame instead of empty list

    # Transform job skills using TF-IDF
    job_vecs = tfidf_vectorizer.transform(cluster_jobs['skills'])
    
    # Compute cosine similarity
    similarities = cosine_similarity(user_input_vec, job_vecs).flatten()
    cluster_jobs['similarity'] = similarities
    
    # Sort by similarity and return top N
    top_jobs = cluster_jobs.sort_values(by='similarity', ascending=False).head(top_n)
    return top_jobs

# ------------------------------
# Helper function to parse input text to list
# ------------------------------
def parse_input_text(text):
    """Parse comma-separated text input into a list of cleaned items"""
    if not text or not text.strip():
        return []
    items = [item.strip() for item in text.split(',')]
    return [item for item in items if item]  # Remove empty strings

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Job Recommender", layout="wide")
st.title("💼 Job Recommendation System")

# Extract unique skills and locations
unique_skills, unique_locations = extract_unique_skills_and_locations(jobs)

# ------------------------------
# Initialize session state
# ------------------------------
if 'selected_skills' not in st.session_state:
    st.session_state.selected_skills = []

if 'selected_locations' not in st.session_state:
    st.session_state.selected_locations = []

if 'skills_display_count' not in st.session_state:
    st.session_state.skills_display_count = 20

if 'locations_display_count' not in st.session_state:
    st.session_state.locations_display_count = 20

if 'skills_search_term' not in st.session_state:
    st.session_state.skills_search_term = ""

if 'locations_search_term' not in st.session_state:
    st.session_state.locations_search_term = ""

if 'last_skills_input' not in st.session_state:
    st.session_state.last_skills_input = ""

if 'last_location_input' not in st.session_state:
    st.session_state.last_location_input = ""

# ------------------------------
# Left Sidebar for Filters
# ------------------------------
with st.sidebar:
    st.header("🔍 Filter Options")
    
    # Skills Selection
    st.subheader("Skills")
    
    # Skills search with session state
    skills_search = st.text_input(
        "Search skills:", 
        placeholder="Type to search skills...", 
        key="skills_search_input",
        value=st.session_state.skills_search_term
    )
    
    # Update search term in session state
    if skills_search != st.session_state.skills_search_term:
        st.session_state.skills_search_term = skills_search
        st.session_state.skills_display_count = 20  # Reset display count when search changes
    
    # Filter skills based on search
    filtered_skills = unique_skills
    if st.session_state.skills_search_term:
        search_term = st.session_state.skills_search_term.lower()
        filtered_skills = [skill for skill in unique_skills if search_term in skill.lower()]
    
    # Display skills with checkboxes
    skills_to_display = filtered_skills[:st.session_state.skills_display_count]
    
    for skill in skills_to_display:
        checkbox_key = f"skill_{skill}"
        
        # Check if this skill is currently selected
        is_checked = skill in st.session_state.selected_skills
        
        # Create checkbox and handle changes
        if st.checkbox(skill, value=is_checked, key=checkbox_key):
            if skill not in st.session_state.selected_skills:
                st.session_state.selected_skills.append(skill)
        else:
            if skill in st.session_state.selected_skills:
                st.session_state.selected_skills.remove(skill)
    
    # Load more skills button
    if len(filtered_skills) > st.session_state.skills_display_count:
        if st.button("Load More Skills", key="load_more_skills"):
            st.session_state.skills_display_count += 20
            st.rerun()
    
    # Clear skills button
    if st.button("Clear All Skills", key="clear_skills"):
        st.session_state.selected_skills = []
        st.session_state.skills_display_count = 20
        st.rerun()
    
    st.markdown("---")
    
    # Locations Selection
    st.subheader("Locations")
    
    # Locations search with session state
    location_search = st.text_input(
        "Search locations:", 
        placeholder="Type to search locations...", 
        key="locations_search_input",
        value=st.session_state.locations_search_term
    )
    
    # Update search term in session state
    if location_search != st.session_state.locations_search_term:
        st.session_state.locations_search_term = location_search
        st.session_state.locations_display_count = 20  # Reset display count when search changes
    
    # Filter locations based on search
    filtered_locations = unique_locations
    if st.session_state.locations_search_term:
        search_term = st.session_state.locations_search_term.lower()
        filtered_locations = [loc for loc in unique_locations if search_term in loc.lower()]
    
    # Display locations with checkboxes
    locations_to_display = filtered_locations[:st.session_state.locations_display_count]
    
    for location in locations_to_display:
        checkbox_key = f"loc_{location}"
        
        # Check if this location is currently selected
        is_checked = location in st.session_state.selected_locations
        
        # Create checkbox and handle changes
        if st.checkbox(location, value=is_checked, key=checkbox_key):
            if location not in st.session_state.selected_locations:
                st.session_state.selected_locations.append(location)
        else:
            if location in st.session_state.selected_locations:
                st.session_state.selected_locations.remove(location)
    
    # Load more locations button
    if len(filtered_locations) > st.session_state.locations_display_count:
        if st.button("Load More Locations", key="load_more_locations"):
            st.session_state.locations_display_count += 20
            st.rerun()
    
    # Clear locations button
    if st.button("Clear All Locations", key="clear_locations"):
        st.session_state.selected_locations = []
        st.session_state.locations_display_count = 20
        st.rerun()
    
    st.markdown("---")
    
    # Show selected filters
    if st.session_state.selected_skills:
        st.write("**Selected Skills:**")
        for skill in st.session_state.selected_skills:
            st.write(f"- {skill}")
    
    if st.session_state.selected_locations:
        st.write("**Selected Locations:**")
        for location in st.session_state.selected_locations:
            st.write(f"- {location}")

# ------------------------------
# Main Content Area
# ------------------------------

# Create the skills input with current selections
default_skills = ", ".join(st.session_state.selected_skills)
user_input = st.text_input(
    "Enter your skills (comma-separated, e.g., Python, SQL, Machine Learning):",
    value=default_skills,
    placeholder="Select skills from sidebar or type your own...",
    key="main_skills_input"
)

# Sync skills from main input to sidebar
if user_input != st.session_state.last_skills_input:
    st.session_state.last_skills_input = user_input
    # Parse the input and update selected skills
    input_skills = parse_input_text(user_input)
    
    # Add new skills that aren't in the predefined list but are in the input
    for skill in input_skills:
        skill_lower = skill.lower()
        # Check if this skill exists in our unique skills (case-insensitive)
        matching_skill = next((s for s in unique_skills if s.lower() == skill_lower), None)
        if matching_skill and matching_skill not in st.session_state.selected_skills:
            st.session_state.selected_skills.append(matching_skill)
        elif not matching_skill and skill not in st.session_state.selected_skills:
            # For custom skills not in our dataset, add them to selected skills
            st.session_state.selected_skills.append(skill)
    
    # Remove skills that are no longer in the input
    skills_to_remove = []
    for selected_skill in st.session_state.selected_skills:
        selected_lower = selected_skill.lower()
        if not any(skill.lower() == selected_lower for skill in input_skills):
            skills_to_remove.append(selected_skill)
    
    for skill in skills_to_remove:
        st.session_state.selected_skills.remove(skill)

# Additional filters section
with st.expander("🔧 Additional Filters (Optional)", expanded=False):
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        # Location input that syncs with sidebar selections
        location_filter_value = ", ".join(st.session_state.selected_locations) if st.session_state.selected_locations else ""
        location_filter = st.text_input(
            "Location Keyword:", 
            value=location_filter_value,
            placeholder="e.g., New York, Remote",
            key="main_location_input"
        )
        
        # Sync locations from main input to sidebar
        if location_filter != st.session_state.last_location_input:
            st.session_state.last_location_input = location_filter
            # Parse the input and update selected locations
            input_locations = parse_input_text(location_filter)
            
            # Add new locations that aren't in the predefined list but are in the input
            for location in input_locations:
                location_lower = location.lower()
                # Check if this location exists in our unique locations (case-insensitive)
                matching_location = next((loc for loc in unique_locations if loc.lower() == location_lower), None)
                if matching_location and matching_location not in st.session_state.selected_locations:
                    st.session_state.selected_locations.append(matching_location)
                elif not matching_location and location not in st.session_state.selected_locations:
                    # For custom locations not in our dataset, add them to selected locations
                    st.session_state.selected_locations.append(location)
            
            # Remove locations that are no longer in the input
            locations_to_remove = []
            for selected_location in st.session_state.selected_locations:
                selected_lower = selected_location.lower()
                if not any(location.lower() == selected_lower for location in input_locations):
                    locations_to_remove.append(selected_location)
            
            for location in locations_to_remove:
                st.session_state.selected_locations.remove(location)
        
        position_filter = st.text_input("Job Title Keyword:", placeholder="e.g., Data Scientist, Engineer")
    
    with filter_col2:
        salary_col1, salary_col2 = st.columns(2)
        with salary_col1:
            min_salary = st.number_input("Min Salary:", min_value=0, value=0, step=10000, help="Leave as 0 for no minimum")
        with salary_col2:
            max_salary = st.number_input("Max Salary:", min_value=0, value=0, step=10000, help="Leave as 0 for no maximum")

# Options row
col1, col2 = st.columns([2, 2])

with col1:
    sort_option = st.selectbox(
        "Sort by:",
        ["Relevance (Default)", "Salary (High to Low)", "Salary (Low to High)", "Rating (High to Low)", "Company Name (A-Z)"]
    )

with col2:
    num_jobs = st.slider("Number of jobs to show:", min_value=1, max_value=20, value=5, step=1)

if st.button("Get Recommendations", type="primary"):
    if not user_input.strip():
        st.warning("Please enter your skills.")
    else:
        with st.spinner("Finding best matching jobs..."):
            # Prepare filter parameters
            loc_filter = location_filter.strip() if location_filter and location_filter.strip() else None
            pos_filter = position_filter.strip() if position_filter and position_filter.strip() else None
            min_sal = min_salary if min_salary > 0 else None
            max_sal = max_salary if max_salary > 0 else None
            
            top_jobs = recommend_jobs(
                user_input, 
                top_n=num_jobs,
                location_filter=loc_filter,
                position_filter=pos_filter,
                min_salary=min_sal,
                max_salary=max_sal,
                selected_skills=st.session_state.selected_skills,
                selected_locations=st.session_state.selected_locations
            )
        
        # FIXED: Check if DataFrame is empty using len() instead of .empty
        if len(top_jobs) == 0:
            st.info("No jobs found matching your criteria. Try adjusting your filters.")
        else:
            # Apply sorting based on user selection
            if sort_option == "Salary (High to Low)":
                top_jobs['salary_numeric'] = pd.to_numeric(top_jobs.get('average_salary', 0), errors='coerce')
                top_jobs = top_jobs.sort_values(by='salary_numeric', ascending=False)
            elif sort_option == "Salary (Low to High)":
                top_jobs['salary_numeric'] = pd.to_numeric(top_jobs.get('average_salary', 0), errors='coerce')
                top_jobs = top_jobs.sort_values(by='salary_numeric', ascending=True)
            elif sort_option == "Rating (High to Low)":
                if 'rating' in top_jobs.columns:
                    top_jobs = top_jobs.sort_values(by='rating', ascending=False)
                else:
                    st.warning("Rating data not available in the dataset.")
            elif sort_option == "Company Name (A-Z)":
                top_jobs = top_jobs.sort_values(by='company', ascending=True)
            
            st.subheader(f"Top {len(top_jobs)} Job Recommendations")
            
            for idx, row in top_jobs.iterrows():
                skills = format_skills(row.get('skills', 'N/A'))
                rating = row.get('rating', 'N/A')
                
                # Find missing skills
                missing_skills = find_missing_skills(user_input, row.get('skills', ''))
                missing_skills_str = ", ".join(missing_skills) if missing_skills else "None - You have all required skills! ✅"
                
                with st.container():
                    st.markdown(
                        f"""
                        <div style="
                            border: 1px solid rgba(128, 128, 128, 0.3); 
                            border-radius: 10px; 
                            padding: 20px; 
                            margin-bottom: 15px; 
                            background-color: rgba(128, 128, 128, 0.05);
                        ">
                            <h4 style="margin-top:0; margin-bottom:10px;">{row['positionName']} at {row['company']}</h4>
                            <p style="margin:5px 0;"><strong>Location:</strong> {row.get('city', row.get('location', 'N/A'))}</p>
                            <p style="margin:5px 0;"><strong>Salary:</strong> {row.get('average_salary', 'N/A')}</p>
                            <p style="margin:5px 0;"><strong>Rating:</strong> {rating}</p>
                            <p style="margin:5px 0;"><strong>Skills Required:</strong> {skills}</p>
                            <p style="margin:5px 0; color: #ff6b6b;"><strong>Missing Skills:</strong> {missing_skills_str}</p>
                        </div>
                        """, unsafe_allow_html=True
                    )