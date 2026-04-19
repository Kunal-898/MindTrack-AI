import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mindtrack-single-file.html")

with open(HTML_FILE, "r", encoding="utf-8") as f:
    original_html = f.read()

JS_BRIDGE = """
<script>
(function() {
  function patchButton() {
    const btn = [...document.querySelectorAll("button")]
      .find(b => b.textContent.includes("Analyze Emotion"));
    if (!btn) { setTimeout(patchButton, 500); return; }
    btn.addEventListener("click", async function(e) {
      const ta = document.querySelector("textarea");
      if (!ta || ta.value.length < 10) return;
      btn.disabled = true;
      btn.textContent = "⏳ Analysing...";
      try {
        const resp = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: ta.value })
        });
        const data = await resp.json();
        window.__MT_LIVE__ = data;
        window.dispatchEvent(new CustomEvent("mt_result", { detail: data }));
      } catch(err) {
        alert("Model error: " + err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = "✨ Analyze Emotion";
      }
    }, true);
  }
  patchButton();
})();
</script>
"""

HTML_WITH_BRIDGE = original_html.replace("</body>", JS_BRIDGE + "\n</body>")

@app.route("/")
def index():
    return HTML_WITH_BRIDGE, 200, {"Content-Type": "text/html; charset=utf-8"}

@app.route("/predict", methods=["POST"])
def predict_route():
    from model_engine import predict
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text or len(text) < 5:
        return jsonify({"error": "Text too short"}), 400
    try:
        return jsonify(predict(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("=" * 40)
    print("  MindTrack running!")
    print("  Open: http://localhost:8501")
    print("=" * 40)
    app.run(host="0.0.0.0", port=8501, debug=False)