from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import sqlite3
from datetime import datetime

app = Flask(__name__)

# --- 1. Load YOUR Hugging Face Model ---
# Replace 'your_username/MindTrack-AI-Model' with your actual HF path
MODEL_PATH = "your_username/MindTrack-AI-Model"

try:
    # This loads your model directly from the cloud
    analyzer = pipeline("text-classification", model=MODEL_PATH)
    print(f"✅ Success: {MODEL_PATH} is loaded and ready!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Fallback model in case your upload has issues
    analyzer = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# --- 2. Database Logic ---
def init_db():
    conn = sqlite3.connect('mindtrack.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS journal_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, entry TEXT, emotion TEXT, score REAL)''')
    conn.commit()
    conn.close()

init_db()

# --- 3. Routes ---
@app.route('/')
def index():
    return render_template('index.html') # Your Figma HTML file

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_text = data.get('text', '')

    if not user_text:
        return jsonify({"error": "Khali text mat bhejo bhai!"}), 400

    # AI Prediction using your model
    prediction = analyzer(user_text)
    label = prediction[0]['label']
    score = round(prediction[0]['score'] * 100, 2)

    # Save to SQLite
    conn = sqlite3.connect('mindtrack.db')
    c = conn.cursor()
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT INTO journal_logs (date, entry, emotion, score) VALUES (?, ?, ?, ?)",
              (date_now, user_text, label, score))
    conn.commit()
    conn.close()

    return jsonify({"emotion": label, "score": score})

    from waitress import serve

    if __name__ == '__main__':
         # debug=True hata dein, ye server ke liye theek nahi hai
         print("Server starting on port 7860...")
         serve(app, host='0.0.0.0', port=7860)
