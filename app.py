import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline

# 1. Page Config - Sabse upar hona chahiye
st.set_page_config(page_title="MindTrack AI", layout="wide", initial_sidebar_state="collapsed")

# 2. Strong CSS for Full Screen Experience
st.markdown("""
    <style>
        /* Sidebar aur Header ko hide karne ke liye */
        [data-testid="stHeader"], [data-testid="stSidebarNav"] {visibility: hidden;}
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        /* Dashboard Container */
        .dashboard-container {
            border-radius: 15px;
            overflow: hidden;
            border: 1px solid #e6e6e6;
        }
    </style>
    """, unsafe_allow_html=True)

# 3. Your Working Figma Link
original_link = "https://idea-veggie-98907730.figma.site/"
embed_url = f"https://www.figma.com/embed?embed_host=share&url={original_link}"

# --- APP LAYOUT ---

st.title("🧠 MindTrack AI")

# Figma Dashboard Section
st.write("### Your Emotional Insights")
components.html(
    f'<iframe src="{embed_url}" width="100%" height="600" style="border:1px solid #ddd; border-radius:10px;" allowfullscreen></iframe>',
    height=620,
)

st.markdown("---")

# Prediction Section
st.write("### Analyze Your Current Mood")
user_input = st.text_area("What's on your mind?", "I feel amazing today!", height=100)

if st.button("Run AI Analysis"):
    with st.spinner('AI is thinking...'):
        classifier = pipeline("sentiment-analysis", model="kunal1323/MindTrack-AI1")
        result = classifier(user_input)
        
        mood = result[0]['label']
        confidence = result[0]['score']
        
        # Color coding based on mood
        color = "green" if mood == "POSITIVE" else "red"
        st.markdown(f"**Mood Detected:** :{color}[{mood}]")
        st.info(f"AI Confidence Score: {confidence:.2f}")
