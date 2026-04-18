"""
MindTrack – Model Engine
Loads the DistilRoBERTa emotion classifier (6 labels) and exposes predict().

Label mapping (matches config.json id2label):
  0 → sadness  1 → joy  2 → love  3 → anger  4 → fear  5 → surprise
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.environ.get(
    "MINDTRACK_MODEL_DIR",
    r"C:\Users\Administrator\OneDrive\Desktop\final_model"
)

LABEL_META = {
    0: {"name": "Sadness",  "emoji": "😢", "color": "#93C5FD"},
    1: {"name": "Joy",      "emoji": "😊", "color": "#34D399"},
    2: {"name": "Love",     "emoji": "❤️",  "color": "#F472B6"},
    3: {"name": "Anger",    "emoji": "😠", "color": "#F87171"},
    4: {"name": "Fear",     "emoji": "😨", "color": "#FBBF24"},
    5: {"name": "Surprise", "emoji": "😲", "color": "#A78BFA"},
}

# Positive emotions push wellness up, negative push down
WELLNESS_WEIGHTS = {0: -0.8, 1: 1.0, 2: 0.9, 3: -0.7, 4: -0.6, 5: 0.2}

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.eval()
    return _tokenizer, _model


def predict(text: str) -> dict:
    """
    Returns:
      {
        emotions: [ {name, emoji, color, score}, ... ],   # sorted desc
        dominant: {name, emoji, color, score},
        wellness_index: float,   # 0–100
        alert: bool,
        summary: str,
        suggestions: [str, ...]
      }
    """
    tokenizer, model = _load_model()
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1).squeeze()

    emotions = []
    wellness = 50.0
    for idx, prob in enumerate(probs.tolist()):
        meta = LABEL_META[idx]
        score = round(prob * 100, 1)
        emotions.append({
            "name":  meta["name"],
            "emoji": meta["emoji"],
            "color": meta["color"],
            "score": score,
        })
        wellness += WELLNESS_WEIGHTS[idx] * prob * 50

    emotions.sort(key=lambda e: e["score"], reverse=True)
    dominant = emotions[0]
    wellness = round(max(0.0, min(100.0, wellness)), 1)
    negative_mass = sum(e["score"] for e in emotions
                        if e["name"] in ("Sadness", "Anger", "Fear"))
    alert = negative_mass > 55.0

    return {
        "emotions":       emotions,
        "dominant":       dominant,
        "wellness_index": wellness,
        "alert":          alert,
        "summary":        _make_summary(dominant, wellness, alert),
        "suggestions":    _make_suggestions(emotions, alert),
    }


def _make_summary(dominant, wellness, alert):
    name = dominant["name"]
    if alert:
        return (f"Your entry shows strong {name.lower()}. "
                "Consider taking a mindful break or talking to someone you trust.")
    if wellness >= 70:
        return f"You're radiating {name.lower()} — a great mental health day!"
    if wellness >= 45:
        return f"Your mood is balanced with a touch of {name.lower()}. Keep reflecting!"
    return (f"Today feels heavy with {name.lower()}. "
            "Writing more may help you process these feelings.")


def _make_suggestions(emotions, alert):
    tips = []
    names = {e["name"] for e in emotions[:2]}
    if alert:
        tips.append("Try a 5-minute deep breathing exercise right now.")
        tips.append("Reach out to a friend or mental health professional.")
    if "Joy" in names or "Love" in names:
        tips.append("Your positive energy is great — share it with someone!")
    if "Fear" in names or "Sadness" in names:
        tips.append("Journaling more about your feelings can bring clarity.")
    if "Anger" in names:
        tips.append("A short walk can help release built-up tension.")
    if not tips:
        tips.append("Keep journaling — consistency builds self-awareness.")
    tips.append("Your gratitude levels look balanced today.")
    return tips[:3]