import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline

# 1. Page Configuration (Full Screen Layout)
st.set_page_config(page_title="MindTrack AI", layout="wide")

# 2. CSS to remove margins and white spaces
st.markdown("""
    <style>
        .block-container {
            max-width: 100% !important;
            padding: 0rem !important;
        }
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. Figma Design Link (Embed logic)
# Note: Use your actual design link here
# 1. Direct design link use karein
original_link = "https://www.figma.com/design/1OxmuZEM5r2rdd9iOqHUdU/UX-Designer-Portfolio"

# 2. Embed URL banayein
embed_url = f"https://www.figma.com/embed?embed_host=share&url={original_link}""


# 4. Display Figma Design
components.html(
    f'<iframe src="{embed_url}" width="100%" height="800" style="border:none;" allowfullscreen></iframe>',
    height=820,
)

# 5. AI Model Section (Sentiment Analysis)
st.title("🧠 Try MindTrack AI")
user_input = st.text_area("How are you feeling today?", "I feel very positive and happy!")

if st.button("Analyze Mood"):
    # Yeh aapke folder se model load karega
    classifier = pipeline("sentiment-analysis", model="kunal1323/MindTrack-AI1")
    result = classifier(user_input)
    
    label = result[0]['label']
    score = result[0]['score']
    
    st.write(f"**Mood Detected:** {label} (Confidence: {score:.2f})")
