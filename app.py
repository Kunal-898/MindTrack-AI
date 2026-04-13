import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline
import pandas as pd 

# 1. Page Configuration
st.set_page_config(page_title="MindTrack AI", layout="wide")

# 2. CSS for clean UI & Metrics
st.markdown("""
    <style>
        .block-container {
            max-width: 100% !important;
            padding: 1rem !important;
        }
        header {visibility: hidden;}
        footer {visibility: hidden;}
        div[data-testid="stMetricValue"] { font-size: 24px; }
        .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)

# 3. Figma Design Section (Your Visual UI)
st.title("🎨 Project Dashboard & UI Design")
original_link = "https://id-preview--843b7c98-2d96-4676-9e66-c7d4537de142.lovable.app/"
embed_url = f"https://www.figma.com/embed?embed_host=share&url={original_link}"

components.html(
    f'<iframe src="{embed_url}" width="100%" height="600" style="border:1px solid #eee;" allowfullscreen></iframe>',
    height=620,
)

# 4. AI Analysis Section (The Core Logic)
st.markdown("---")
st.header("🧠 Live Emotion & Trigger Analysis")
st.info("This section uses the DistilRoBERTa model trained in Google Colab.")

user_input = st.text_area("How are you feeling today? (Write your journal entry)", height=150)

if st.button("Analyze Mood"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # --- Step 1: Emotion Detection ---
        with st.spinner('AI is processing your emotions...'):
            classifier = pipeline("sentiment-analysis", model="kunal1323/MindTrack-AI1")
            result = classifier(user_input)
            label = result[0]['label']
            score = result[0]['score']

        # --- Step 2: Topic Modeling (Keyword Mapping) ---
        topics = {
            "Academic/Exams 📚": ["exam", "test", "study", "college", "marks", "assignment", "result", "semester", "viva"],
            "Work/Professional 💼": ["office", "boss", "work", "deadline", "project", "meeting", "job", "career", "salary"],
            "Relationships/Family ❤️": ["family", "mom", "dad", "friend", "breakup", "relationship", "marriage", "home"],
            "Health/Physical 🏥": ["sick", "hospital", "gym", "workout", "pain", "tired", "sleep", "health"]
        }
        
        detected_topic = "General Life"
        for topic, keywords in topics.items():
            if any(word in user_input.lower() for word in keywords):
                detected_topic = topic
                break

        # --- Display Results ---
        st.markdown("### Analysis Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Primary Emotion", label.upper())
        with col2:
            st.metric("Likely Trigger", detected_topic)
        with col3:
            st.metric("AI Confidence", f"{round(score * 100, 1)}%")

        # --- Step 3: Temporal Trends (7-Day Visualization) ---
        st.markdown("---")
        st.subheader("📈 7-Day Mental Health Trend")
        # Sample trend data to satisfy the slide objective
        trend_data = pd.DataFrame({
            'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'Wellness Score': [70, 65, 85, 50, 45, 60, 75] 
        })
        st.line_chart(trend_data.set_index('Day'))

        # Early Intervention logic
        if label.lower() in ['sadness', 'anger', 'fear']:
            st.error(f"⚠️ **Early Warning:** We've detected an emotional dip related to **{detected_topic}**. Please check the 'Resources' section in the Figma UI for support.")
        else:
            st.success("Your emotional state appears stable. Keep journaling!") 
