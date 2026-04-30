import os
import sys
import json
import math
import datetime
import re
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)

# ─── Model (loaded once at startup) ─────────────────────────────────────────
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR   = r"C:\Users\Administrator\OneDrive\Desktop\project\final_model"
ADAPTED_DIR =  r"C:\Users\Administrator\OneDrive\Desktop\project\adapted_v5"
LABELS      = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL2ID    = {l: i for i, l in enumerate(LABELS)}
POLARITY    = {"joy": 0.9, "love": 0.8, "surprise": 0.1,
               "fear": -0.7, "sadness": -0.8, "anger": -0.9}

# Global adaptation status — polled by /api/adapt/status
_adapt_status = {"running": False, "done": False, "error": None, "log": []}

def _load_model(path):
    tok   = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tok, model

print("[MindTrack] Loading model...")
_active_dir        =  ADAPTED_DIR if os.path.isdir(ADAPTED_DIR) else MODEL_DIR
_tokenizer, _model = _load_model(_active_dir)
_model_version     = "adapted" if os.path.isdir(ADAPTED_DIR) else "base"
print(f"[MindTrack] Model loaded OK! (version: {_model_version})")

# ─── Classification helpers ──────────────────────────────────────────────────
def _classify_chunk(text):
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out   = _model(**inputs)
        probs = torch.softmax(out.logits, dim=1)[0]
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

def classify_text(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sentences) <= 2:
        chunk_scores = [_classify_chunk(text)]
    else:
        chunks = []
        for i in range(0, len(sentences), 2):
            chunks.append(" ".join(sentences[i:i+2]))
        chunk_scores = [_classify_chunk(c) for c in chunks]

    avg_scores = {
        label: round(sum(c[label] for c in chunk_scores) / len(chunk_scores), 4)
        for label in LABELS
    }
    top      = max(avg_scores, key=avg_scores.get)
    polarity = round(sum(avg_scores[lbl] * POLARITY[lbl] for lbl in LABELS), 4)
    return top, polarity, avg_scores

# ─── EWMA Wellness Index (research contribution) ─────────────────────────────
def compute_wellness(entries, alpha=0.3):
    """
    Exponentially Weighted Moving Average wellness score.

    alpha  : recency weight — higher = more sensitive to recent entries.
    Returns: ewma_score [-1,1], trend direction, anomaly flag, wellness [0,100].

    Research note:
        Simple mean treats a bad entry 30 days ago equally to yesterday.
        EWMA gives exponentially more weight to recent entries, which better
        reflects a user's current mental state for clinical relevance.
    """
    if not entries:
        return 0.0, "stable", False, 50

    polarities = [e.polarity for e in entries]

    # EWMA
    ewma = polarities[0]
    for p in polarities[1:]:
        ewma = alpha * p + (1 - alpha) * ewma

    # Trend: compare last 7 vs prior 7 days
    if len(polarities) >= 14:
        recent = sum(polarities[-7:]) / 7
        prior  = sum(polarities[-14:-7]) / 7
        delta  = recent - prior
        trend  = "improving" if delta > 0.1 else "declining" if delta < -0.1 else "stable"
    else:
        trend = "stable"

    # Anomaly: z-score spike detection
    anomaly = False
    if len(polarities) >= 5:
        mean = sum(polarities) / len(polarities)
        std  = math.sqrt(sum((p - mean) ** 2 for p in polarities) / len(polarities)) + 1e-8
        z    = abs(polarities[-1] - mean) / std
        anomaly = z > 2.0

    wellness_pct = round((ewma + 1) / 2 * 100)
    return round(ewma, 4), trend, anomaly, wellness_pct

# ─── Domain Adaptation ───────────────────────────────────────────────────────
def run_domain_adaptation(samples, epochs=3, lr=2e-5, batch_size=4):
    """
    Fine-tune the loaded model on journal-style entries pulled from the DB.
    Saves adapted model to ADAPTED_DIR and hot-swaps the global model live.

    Research contribution:
        DistilRoBERTa was trained on the Kaggle Emotions Dataset — short,
        Twitter-style text. Personal journal entries differ significantly:
        longer sentences, introspective tone, first-person narration, no
        hashtags or @mentions. This fine-tuning step adapts the model to
        that domain without catastrophic forgetting (low LR, few epochs).
    """
    global _tokenizer, _model, _model_version, _adapt_status

    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from torch.optim import AdamW

    if len(samples) < 6:
        raise ValueError(
            f"Need at least 6 labelled samples (got {len(samples)}). "
            "Add more journal entries or run seed.py first."
        )

    # ── Dataset ──────────────────────────────────────────────────────────────
    class JournalDataset(TorchDataset):
        def __init__(self, items, tokenizer):
            self.items     = items
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            enc  = self.tokenizer(
                item["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(LABEL2ID[item["label"]], dtype=torch.long),
            }

    dataset    = JournalDataset(samples, _tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── Fine-tune a fresh copy so the running model isn't disrupted ──────────
    adapt_tok, adapt_model = _load_model(_active_dir)
    adapt_model.train()
    optimizer = AdamW(adapt_model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = adapt_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg  = epoch_loss / len(dataloader)
        line = f"Epoch {epoch+1}/{epochs}  loss={avg:.4f}"
        _adapt_status["log"].append(line)
        print(f"[DomainAdapt] {line}")

    # ── Save & hot-swap ───────────────────────────────────────────────────────
    os.makedirs(ADAPTED_DIR, exist_ok=True)
    adapt_model.save_pretrained(ADAPTED_DIR)
    adapt_tok.save_pretrained(ADAPTED_DIR)
    adapt_model.eval()

    _tokenizer     = adapt_tok
    _model         = adapt_model
    _model_version = "adapted"
    print(f"[DomainAdapt] Saved → {ADAPTED_DIR}  |  Model hot-swapped.")

# ─── Flask app ───────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SQLALCHEMY_DATABASE_URI"]        = "sqlite:///mindtrack.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"]                 = "mindtrack-super-secret-key-2024-absolutely-secure!!"
app.config["JWT_ACCESS_TOKEN_EXPIRES"]       = datetime.timedelta(days=7)
app.config["JWT_TOKEN_LOCATION"]             = ["headers"]
app.config["JWT_HEADER_NAME"]                = "Authorization"
app.config["JWT_HEADER_TYPE"]                = "Bearer"

db     = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt    = JWTManager(app)

# ─── Database models ─────────────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = "users"
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(80), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    entries    = db.relationship("Entry", backref="user", lazy=True,
                                 cascade="all, delete-orphan")

class Entry(db.Model):
    __tablename__ = "entries"
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    text       = db.Column(db.Text, nullable=False)
    emotion    = db.Column(db.String(40), nullable=False)
    intensity  = db.Column(db.Float, default=5.0)
    polarity   = db.Column(db.Float, default=0.0)
    events     = db.Column(db.Text, default="[]")
    scores     = db.Column(db.Text, default="{}")
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id":        self.id,
            "text":      self.text,
            "emotion":   self.emotion,
            "intensity": self.intensity,
            "polarity":  self.polarity,
            "events":    json.loads(self.events),
            "scores":    json.loads(self.scores),
            "date":      self.created_at.isoformat(),
        }

# ─── Auth routes ─────────────────────────────────────────────────────────────
@app.route("/api/signup", methods=["POST"])
def signup():
    data     = request.get_json()
    username = (data.get("username") or "").strip().lower()
    password = data.get("password", "")
    if not username or not password:
        return jsonify(error="Username and password required"), 400
    if len(password) < 6:
        return jsonify(error="Password must be at least 6 characters"), 400
    if User.query.filter_by(username=username).first():
        return jsonify(error="Username already taken"), 409
    hashed = bcrypt.generate_password_hash(password).decode("utf-8")
    user   = User(username=username, password=hashed)
    db.session.add(user)
    db.session.commit()
    token = create_access_token(identity=str(user.id))
    return jsonify(token=token, username=username), 201

@app.route("/api/login", methods=["POST"])
def login():
    data     = request.get_json()
    username = (data.get("username") or "").strip().lower()
    password = data.get("password", "")
    user     = User.query.filter_by(username=username).first()
    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify(error="Invalid credentials"), 401
    token = create_access_token(identity=str(user.id))
    return jsonify(token=token, username=username), 200

# ─── Classify route ───────────────────────────────────────────────────────────
@app.route("/api/classify", methods=["POST"])
@jwt_required()
def classify():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="text is required"), 400
    try:
        emotion, polarity, scores = classify_text(text)
        return jsonify(emotion=emotion, polarity=polarity,
                       scores=scores, model_version=_model_version)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(error=str(e)), 500

# ─── Entry routes ─────────────────────────────────────────────────────────────
@app.route("/api/entries", methods=["POST"])
@jwt_required()
def create_entry():
    user_id   = int(get_jwt_identity())
    data      = request.get_json()
    text      = (data.get("text") or "").strip()
    intensity = float(data.get("intensity", 5))
    events    = data.get("events", [])

    if not text:
        return jsonify(error="text required"), 400

    if data.get("scores"):
        scores   = data["scores"]
        emotion  = data.get("emotion") or max(scores, key=scores.get)
        polarity = float(data.get("polarity", POLARITY.get(emotion, 0.0)))
    else:
        emotion, polarity, scores = classify_text(text)

    entry = Entry(
        user_id=user_id, text=text, emotion=emotion,
        intensity=intensity, polarity=polarity,
        events=json.dumps(events), scores=json.dumps(scores),
    )
    db.session.add(entry)
    db.session.commit()
    return jsonify(entry.to_dict()), 201

@app.route("/api/entries", methods=["GET"])
@jwt_required()
def get_entries():
    user_id = int(get_jwt_identity())
    limit   = int(request.args.get("limit", 200))
    rows    = (Entry.query.filter_by(user_id=user_id)
               .order_by(Entry.created_at.desc()).limit(limit).all())
    return jsonify([e.to_dict() for e in rows])

@app.route("/api/entries/<int:entry_id>", methods=["DELETE"])
@jwt_required()
def delete_entry(entry_id):
    user_id = int(get_jwt_identity())
    entry   = Entry.query.filter_by(id=entry_id, user_id=user_id).first_or_404()
    db.session.delete(entry)
    db.session.commit()
    return jsonify(deleted=True)

# ─── Stats route (EWMA wellness) ─────────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
@jwt_required()
def stats():
    user_id = int(get_jwt_identity())
    rows    = (Entry.query.filter_by(user_id=user_id)
               .order_by(Entry.created_at.asc()).all())

    if not rows:
        return jsonify(total=0, avg_polarity=0, emotion_counts={},
                       trend=[], wellness=50, ewma=0.0,
                       trend_direction="stable", anomaly=False)

    total        = len(rows)
    avg_polarity = round(sum(e.polarity for e in rows) / total, 3)

    emotion_counts = {}
    for e in rows:
        emotion_counts[e.emotion] = emotion_counts.get(e.emotion, 0) + 1

    ewma, trend_direction, anomaly, wellness_pct = compute_wellness(rows)

    trend = [
        {"date": e.created_at.isoformat(), "polarity": e.polarity,
         "emotion": e.emotion, "intensity": e.intensity}
        for e in rows[-30:]
    ]

    return jsonify(
        total=total,
        avg_polarity=avg_polarity,
        emotion_counts=emotion_counts,
        trend=trend,
        wellness=wellness_pct,          # EWMA-based [0–100]
        ewma=ewma,                      # raw score  [-1, 1]
        trend_direction=trend_direction, # improving / declining / stable
        anomaly=anomaly,                # True = today is a statistical outlier
        model_version=_model_version,
    )

# ─── Domain Adaptation route ──────────────────────────────────────────────────
@app.route("/api/adapt", methods=["POST"])
@jwt_required()
def adapt():
    """
    POST /api/adapt
    Kicks off domain adaptation in a background thread — browser won't hang.
    Poll GET /api/adapt/status every few seconds to track progress.

    Body (all optional):
        min_samples : int   minimum entries needed (default 6)
        epochs      : int   training epochs        (default 3)
        lr          : float learning rate          (default 2e-5)
    """
    global _adapt_status

    if _adapt_status["running"]:
        return jsonify(
            message="Adaptation already running. Poll /api/adapt/status."
        ), 202

    data        = request.get_json() or {}
    min_samples = int(data.get("min_samples", 6))
    epochs      = int(data.get("epochs", 3))
    lr          = float(data.get("lr", 2e-5))

    all_entries = Entry.query.filter(Entry.emotion.in_(LABELS)).all()

    if len(all_entries) < min_samples:
        return jsonify(
            error=f"Not enough entries. Need {min_samples}, have {len(all_entries)}. "
                  "Run seed.py first or add more entries manually."
        ), 400

    samples = [{"text": e.text, "label": e.emotion} for e in all_entries]

    class_dist = {}
    for s in samples:
        class_dist[s["label"]] = class_dist.get(s["label"], 0) + 1

    # Reset status tracker
    _adapt_status = {"running": True, "done": False, "error": None, "log": []}

    def train_thread():
        global _adapt_status
        try:
            run_domain_adaptation(samples, epochs=epochs, lr=lr)
            _adapt_status["done"]    = True
            _adapt_status["running"] = False
        except Exception as e:
            import traceback; traceback.print_exc()
            _adapt_status["error"]   = str(e)
            _adapt_status["running"] = False

    threading.Thread(target=train_thread, daemon=True).start()

    return jsonify(
        message="Adaptation started. Poll /api/adapt/status for progress.",
        samples_used=len(samples),
        class_distribution=class_dist,
        epochs=epochs,
        lr=lr,
    ), 202

@app.route("/api/adapt/status", methods=["GET"])
@jwt_required()
def adapt_status():
    """
    Poll this every 3-5 seconds while adaptation runs.
    When done=True the model is already hot-swapped and ready.
    """
    return jsonify(
        running=_adapt_status["running"],
        done=_adapt_status["done"],
        error=_adapt_status["error"],
        log=_adapt_status["log"],       # list of "Epoch N/M  loss=X.XXXX" strings
        model_version=_model_version,
    )

# ─── Model info route (ablation helper) ──────────────────────────────────────
@app.route("/api/model-info", methods=["GET"])
@jwt_required()
def model_info():
    """
    Shows which model is active. Use this to record before/after
    accuracy in your paper's results table.
    """
    return jsonify(
        model_version=_model_version,
        active_dir=_active_dir,
        adapted_model_exists=os.path.isdir(ADAPTED_DIR),
        labels=LABELS,
        polarity_weights=POLARITY,
    )

# ─── Profile route ────────────────────────────────────────────────────────────
@app.route("/api/me", methods=["GET"])
@jwt_required()
def me():
    user_id = int(get_jwt_identity())
    user    = User.query.get_or_404(user_id)
    return jsonify(username=user.username,
                   created_at=user.created_at.isoformat(),
                   entry_count=len(user.entries))

# ─── Serve UI ────────────────────────────────────────────────────────────────
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_ui(path):
    return send_from_directory("templates", "index.html")

# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username="demo").first():
            hashed = bcrypt.generate_password_hash("demo123").decode("utf-8")
            db.session.add(User(username="demo", password=hashed))
            db.session.commit()
            print("[MindTrack] Demo user created: demo / demo123")
    print("[MindTrack] Server starting → http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True)  # threaded=True required