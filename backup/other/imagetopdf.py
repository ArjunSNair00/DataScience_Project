import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline

# Load emotion detection pipeline
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

emotion_pipeline = load_model()

st.title("🎭 Emotion Detection App")
st.write("Enter text or speech to detect emotions and see the metrics.")

# User input
user_input = st.text_area("Type something to analyze your emotion:")

if user_input:
    # Get emotion predictions
    results = emotion_pipeline(user_input)[0]
    
    # Convert results to dictionary
    emotions = {res['label']: res['score'] for res in results}
    
    # Display metrics
    st.subheader("Emotion Scores:")
    for label, score in emotions.items():
        st.metric(label, f"{score:.2f}")

    # Plot bar chart
    st.subheader("Emotion Distribution")
    fig, ax = plt.subplots()
    ax.bar(emotions.keys(), emotions.values())
    plt.xticks(rotation=45)
    st.pyplot(fig)