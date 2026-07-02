import os

import numpy as np
import streamlit as st
from PIL import Image
from groq import Groq
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EcoSort — AI Plastic Recycling Assistant",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "model_final.h5"  # adjust if your model file has a different name/format

CLASS_NAMES = ["HDPE", "LDPE", "Non_recyclable", "PET", "PP", "PS"]

PLASTIC_INFO = {
    "PET": {"code": 1, "symbol": "♳", "full_name": "Polyethylene Terephthalate", "recyclable": True},
    "HDPE": {"code": 2, "symbol": "♴", "full_name": "High-Density Polyethylene", "recyclable": True},
    "LDPE": {"code": 4, "symbol": "♶", "full_name": "Low-Density Polyethylene", "recyclable": True},
    "PP": {"code": 5, "symbol": "♷", "full_name": "Polypropylene", "recyclable": True},
    "PS": {"code": 6, "symbol": "♸", "full_name": "Polystyrene", "recyclable": False},
    "Non_recyclable": {"code": 7, "symbol": "♹", "full_name": "Non-Recyclable Plastic", "recyclable": False},
}

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def get_llm_guidance(plastic_type, confidence):
    client = get_groq_client()
    if client is None:
        return "LLM guidance unavailable — no GROQ_API_KEY configured."
    try:
        info = PLASTIC_INFO[plastic_type]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"You are a recycling expert. Plastic: {plastic_type} "
                        f"({info['full_name']}, Resin Code {info['code']}), "
                        f"confidence: {confidence:.1f}%.\n\n"
                        "Provide guidance:\n"
                        "## ♻️ Recycling Status\n"
                        "## 📦 Common Products (3-4 items)\n"
                        "## 🔄 How to Recycle (3 steps)\n"
                        "## ⚠️ Important Notes\n"
                        "## 🌍 Environmental Impact\n\n"
                        "Be concise and practical."
                    ),
                }
            ],
            max_tokens=450,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM error: {e}"


def classify_plastic(image: Image.Image):
    model = get_model()
    img = image.convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx]) * 100

    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = [(CLASS_NAMES[i], float(preds[i]) * 100) for i in top3_idx]

    guidance = get_llm_guidance(pred_class, confidence)

    return {
        "pred_class": pred_class,
        "confidence": confidence,
        "top3": top3,
        "guidance": guidance,
    }


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --ink: #16231C;
    --muted: #5B6960;
    --bg: #F5F8F2;
    --card: #FFFFFF;
    --border: #DFE7D8;
    --green: #1F6B4C;
    --green-dark: #14472F;
    --green-light: #E6F0E4;
    --amber: #B4711E;
    --amber-light: #FBEEDD;
    --red: #A83A32;
    --red-light: #FBE9E7;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
h1, h2, h3, .hero-title { font-family: 'Manrope', sans-serif; }

.stApp { background: var(--bg); }
.block-container { max-width: 760px; padding-top: 1.5rem; padding-bottom: 3rem; }
#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero {
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    background: var(--green-light); border: 1px solid var(--border);
    border-radius: 18px; padding: 20px 24px; margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.hero-left { display: flex; align-items: center; gap: 14px; }
.hero-emoji { font-size: 34px; line-height: 1; }
.hero-title { font-size: 22px; font-weight: 800; color: var(--green-dark); margin: 0; }
.hero-sub { font-size: 13px; color: var(--muted); margin: 2px 0 0; }
.hero-meta { text-align: right; font-size: 11px; color: var(--muted); line-height: 1.5; }

/* Section label */
.step {
    display: flex; align-items: center; gap: 10px; margin: 22px 0 10px;
}
.step-badge {
    width: 26px; height: 26px; border-radius: 50%; background: var(--green);
    color: #fff; font-size: 12px; font-weight: 700;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.step-label { font-size: 15px; font-weight: 700; color: var(--green-dark); }

/* Result card */
.result-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 16px;
    padding: 20px; display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
}
.result-symbol { font-size: 52px; line-height: 1; }
.result-name { font-size: 22px; font-weight: 800; color: var(--green-dark); }
.result-fullname { font-size: 13px; color: var(--muted); margin: 2px 0 8px; }
.pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; margin-right: 6px; }
.pill-code { background: var(--green-light); color: var(--green-dark); border: 1px solid var(--border); }
.pill-yes { background: var(--green-light); color: var(--green-dark); border: 1px solid var(--green); }
.pill-no { background: var(--red-light); color: var(--red); border: 1px solid var(--red); }

/* Footer */
.footer-bar {
    text-align: center; padding: 14px; background: var(--green-light);
    border: 1px solid var(--border); border-radius: 14px; margin-top: 2rem;
    font-size: 11px; color: var(--muted);
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important; font-weight: 600 !important; font-size: 15px !important;
}
.stButton > button[kind="primary"] { background: var(--green) !important; border: none !important; }
.stButton > button[kind="primary"]:hover { background: var(--green-dark) !important; }

/* Mobile */
@media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .hero { flex-direction: column; align-items: flex-start; }
    .hero-meta { text-align: left; }
    .result-card { flex-direction: column; align-items: flex-start; }
    .hero-title { font-size: 19px; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None
if "image" not in st.session_state:
    st.session_state.image = None

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-left">
            <div class="hero-emoji">🌱</div>
            <div>
                <p class="hero-title">EcoSort</p>
                <p class="hero-sub">AI-powered plastic recycling assistant</p>
            </div>
        </div>
        <div class="hero-meta">
            MobileNetV2 + Llama 3<br>Final Year Project
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Step 1 — input
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="step"><div class="step-badge">1</div>'
    '<div class="step-label">Add a photo of the plastic item</div></div>',
    unsafe_allow_html=True,
)

tab_upload, tab_camera = st.tabs(["📤 Upload photo", "📷 Use camera"])

uploaded_image = None
with tab_upload:
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if file is not None:
        uploaded_image = Image.open(file)

with tab_camera:
    cam_file = st.camera_input("Take a photo", label_visibility="collapsed")
    if cam_file is not None:
        uploaded_image = Image.open(cam_file)

if uploaded_image is not None:
    st.session_state.image = uploaded_image

if st.session_state.image is not None:
    st.image(st.session_state.image, use_container_width=True)

# ---------------------------------------------------------------------------
# Step 2 — analyze
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="step"><div class="step-badge">2</div>'
    '<div class="step-label">Run AI analysis</div></div>',
    unsafe_allow_html=True,
)

col_a, col_b = st.columns([3, 1])
with col_a:
    analyze_clicked = st.button(
        "🔍 Analyze plastic",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.image is None,
    )
with col_b:
    clear_clicked = st.button("🗑️ Clear", use_container_width=True)

if clear_clicked:
    st.session_state.result = None
    st.session_state.image = None
    st.rerun()

if analyze_clicked and st.session_state.image is not None:
    with st.spinner("Analyzing image…"):
        st.session_state.result = classify_plastic(st.session_state.image)

# ---------------------------------------------------------------------------
# Step 3 — result
# ---------------------------------------------------------------------------

result = st.session_state.result

if result is not None:
    pred_class = result["pred_class"]
    confidence = result["confidence"]
    info = PLASTIC_INFO[pred_class]

    st.markdown(
        '<div class="step"><div class="step-badge">3</div>'
        '<div class="step-label">Classification result</div></div>',
        unsafe_allow_html=True,
    )

    recyclable_pill = (
        '<span class="pill pill-yes">✅ Recyclable</span>'
        if info["recyclable"]
        else '<span class="pill pill-no">❌ Not recyclable</span>'
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-symbol">{info['symbol']}</div>
            <div>
                <div class="result-name">{pred_class}</div>
                <div class="result-fullname">{info['full_name']}</div>
                <span class="pill pill-code">Resin code {info['code']}</span>
                {recyclable_pill}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if confidence < 50:
        st.warning("⚠️ **Low confidence** — try retaking a clearer photo.")
    elif confidence < 70:
        st.info("ℹ️ **Medium confidence** — double-check the resin code printed on the product.")
    else:
        st.success("✅ **High confidence** — result should be reliable.")

    st.markdown("**Prediction scores**")
    for name, pct in result["top3"]:
        st.write(f"{name} — {pct:.1f}%")
        st.progress(min(pct, 100) / 100)

    # -----------------------------------------------------------------
    # Step 4 — guidance
    # -----------------------------------------------------------------
    st.markdown(
        '<div class="step"><div class="step-badge">4</div>'
        '<div class="step-label">AI recycling guidance</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(result["guidance"])

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="footer-bar">
        🎓 Final Year Capstone Project &nbsp;|&nbsp;
        MobileNetV2 Transfer Learning &nbsp;|&nbsp;
        Hybrid Dataset 6,000 images &nbsp;|&nbsp;
        Accuracy: 77%
    </div>
    """,
    unsafe_allow_html=True,
)
