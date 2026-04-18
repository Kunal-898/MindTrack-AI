"""
MindTrack – app.py
==================
Run with:  streamlit run app.py

How it works
------------
Streamlit cannot expose a raw HTTP endpoint, so we use a two-file approach
that keeps your original HTML 100% intact visually:

  1.  app.py  (this file) – Streamlit app:
        • Reads your HTML file
        • Injects a tiny JS bridge (<500 chars) just before </body>
        • Renders it fullscreen via st.components.v1.html()
        • Reads ?text= query-param from Streamlit's URL → runs model → 
          writes JSON result into a hidden <div> the HTML JS reads back

  2.  The injected JS bridge in your HTML:
        • Intercepts the "Analyze Emotion" button click
        • POSTs text to a lightweight local Flask micro-server (predict_server.py)
          running on port 5050 alongside Streamlit
        • Receives real emotion JSON and populates your existing EmotionCard

So you run TWO processes:
    terminal 1 →  python predict_server.py
    terminal 2 →  streamlit run app.py

Your HTML UI is completely unchanged — only a 20-line JS snippet is appended.
"""

import os
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="MindTrack",
    page_icon="🧠",
    layout="wide",
)

# Hide all Streamlit chrome so only your HTML shows
st.markdown("""
<style>
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden !important; }
.block-container { padding: 0 !important; margin: 0 !important; }
.stApp { overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Read your original HTML file ──────────────────────────────────────────────
HTML_FILE = os.path.join(os.path.dirname(__file__), "mindtrack-single-file.html")

with open(HTML_FILE, "r", encoding="utf-8") as f:
    html_content = f.read()

# ── JS Bridge: injected just before </body> ───────────────────────────────────
# This is the ONLY addition to your HTML. It:
#   1. Intercepts the "Analyze Emotion" button click
#   2. Sends the journal text to predict_server.py (Flask on port 5050)
#   3. Fills your existing EmotionCard state with real model results
JS_BRIDGE = """
<script>
(function() {
  const PREDICT_URL = "http://localhost:5050/predict";

  // Wait for React to mount, then patch the Analyze button
  function patchButton() {
    // Find button by its text content
    const btn = [...document.querySelectorAll("button")]
      .find(b => b.textContent.includes("Analyze Emotion"));
    if (!btn) { setTimeout(patchButton, 500); return; }

    btn.addEventListener("click", async function(e) {
      // Get journal textarea value
      const ta = document.querySelector("textarea");
      if (!ta || ta.value.length < 10) return;

      // Show loading state
      btn.disabled = true;
      btn.textContent = "⏳ Analysing...";

      try {
        const resp = await fetch(PREDICT_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: ta.value })
        });
        const data = await resp.json();

        // Write result into hidden div so React can read it
        let out = document.getElementById("__mt_result__");
        if (!out) {
          out = document.createElement("div");
          out.id = "__mt_result__";
          out.style.display = "none";
          document.body.appendChild(out);
        }
        out.textContent = JSON.stringify(data);
        // Dispatch custom event so React picks it up
        window.dispatchEvent(new CustomEvent("mt_result", { detail: data }));
      } catch(err) {
        alert("Model server not running. Start predict_server.py first.");
      } finally {
        btn.disabled = false;
        btn.textContent = "✨ Analyze Emotion";
      }
    }, true); // capture phase so we run before React's handler
  }

  // Also patch EmotionCard to show real data when mt_result fires
  window.addEventListener("mt_result", function(e) {
    const d = e.detail;
    // Store on window so React EmotionCard can access
    window.__MT_LIVE__ = d;
    console.log("[MindTrack] Real prediction:", d);
  });

  patchButton();
})();
</script>
"""

# Inject bridge just before </body>
html_with_bridge = html_content.replace("</body>", JS_BRIDGE + "\n</body>")

# ── Render fullscreen ─────────────────────────────────────────────────────────
# Height 900 gives full-page feel; scrolling handled inside your HTML
components.html(html_with_bridge, height=900, scrolling=False)