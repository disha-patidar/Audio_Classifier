"""
🎵 Audio Classification Dashboard — Cats vs Dogs vs Birds
Run with:  streamlit run app.py

Required files in the same directory:
  - cnn_model.keras
  - mobilenet_model.keras
  - xgb_audio.json
  - scaler.pkl
  - label_encoder.pkl
  - meta_clf.pkl          (stacked ensemble meta-learner)
"""

import io
import pickle
import tempfile
import warnings
from pathlib import Path

import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SoundID · Audio Classifier",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e4dc;
}

/* Remove default Streamlit padding */
.block-container { padding: 2rem 3rem 3rem 3rem !important; max-width: 1100px; }

/* ── Header ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2.4rem, 5vw, 4rem);
    letter-spacing: -0.03em;
    line-height: 1.05;
    background: linear-gradient(135deg, #f5e642 0%, #ff7c4d 50%, #e84aca 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #6b6878;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* ── Upload zone ── */
.upload-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #f5e642;
    margin-bottom: 0.5rem;
}

/* ── Model selector pills ── */
.pill-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.pill {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    padding: 0.35rem 0.9rem;
    border-radius: 100px;
    border: 1px solid #2a2835;
    background: #13111a;
    color: #9995a8;
    cursor: default;
}
.pill.active {
    background: #f5e642;
    color: #0a0a0f;
    border-color: #f5e642;
    font-weight: 700;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #13111a 0%, #1a1525 100%);
    border: 1px solid #2a2835;
    border-radius: 16px;
    padding: 2rem 2rem 1.5rem 2rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(245,230,66,0.07) 0%, transparent 70%);
}
.result-animal {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.2rem;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.result-conf {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #9995a8;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.conf-value {
    font-weight: 700;
    color: #f5e642;
}

/* ── Confidence bars ── */
.bar-wrap { margin: 1.2rem 0 0.5rem 0; }
.bar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9995a8;
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.3rem;
}
.bar-track {
    height: 6px;
    background: #1e1c28;
    border-radius: 100px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

/* ── Spec image ── */
.spec-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #6b6878;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Divider ── */
.ruled { border: none; border-top: 1px solid #1e1c28; margin: 1.5rem 0; }

/* ── Stray Streamlit elements ── */
.stFileUploader > div { background: #13111a !important; border: 1px dashed #2a2835 !important; border-radius: 12px !important; }
.stFileUploader label { color: #9995a8 !important; font-family: 'Space Mono', monospace !important; font-size: 0.78rem !important; }
.stButton > button {
    background: #f5e642 !important;
    color: #0a0a0f !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.6rem !important;
    font-size: 0.78rem !important;
}
.stButton > button:hover { background: #ffe94f !important; transform: translateY(-1px); }
.stSelectbox > div > div { background: #13111a !important; border: 1px solid #2a2835 !important; color: #e8e4dc !important; font-family: 'Space Mono', monospace !important; font-size: 0.78rem !important; border-radius: 8px !important; }
.stAlert { border-radius: 10px !important; font-family: 'Space Mono', monospace !important; font-size: 0.78rem !important; }
[data-testid="stStatusWidget"] { display: none; }
footer { display: none; }
#MainMenu { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
SR         = 22050
DURATION   = 5.0
N_MFCC     = 40
HOP_LENGTH = 512
N_MELS     = 128
IMG_H      = 128
IMG_W      = 128

ANIMAL_EMOJI = {"cats": "🐱", "dogs": "🐶", "birds": "🐦"}
BAR_COLORS   = ["#f5e642", "#ff7c4d", "#e84aca"]

MODEL_FILES = {
    "Stacked Ensemble": ["cnn_model.keras", "mobilenet_model.keras",
                         "xgb_audio.json", "scaler.pkl", "label_encoder.pkl", "meta_clf.pkl"],
    "CNN":              ["cnn_model.keras", "label_encoder.pkl"],
    "MobileNetV2":      ["mobilenet_model.keras", "label_encoder.pkl"],
    "XGBoost":          ["xgb_audio.json", "scaler.pkl", "label_encoder.pkl"],
}

# ── Model loader (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all available models from disk. Returns dict of loaded objects."""
    import xgboost as xgb
    import tensorflow as tf

    loaded = {}
    errors = []

    def _load_pkl(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Label encoder — required by all paths
    if Path("label_encoder.pkl").exists():
        loaded["le"] = _load_pkl("label_encoder.pkl")
    else:
        errors.append("label_encoder.pkl not found")
        return loaded, errors

    if Path("scaler.pkl").exists():
        loaded["scaler"] = _load_pkl("scaler.pkl")

    if Path("meta_clf.pkl").exists():
        loaded["meta_clf"] = _load_pkl("meta_clf.pkl")

    if Path("cnn_model.keras").exists():
        loaded["cnn"] = tf.keras.models.load_model("cnn_model.keras")

    if Path("mobilenet_model.keras").exists():
        loaded["mobilenet"] = tf.keras.models.load_model("mobilenet_model.keras")

    if Path("xgb_audio.json").exists():
        m = xgb.XGBClassifier()
        m.load_model("xgb_audio.json")
        loaded["xgb"] = m

    return loaded, errors


# ── Audio helpers ──────────────────────────────────────────────────────────────
def load_audio_bytes(audio_bytes):
    import librosa
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    y, _ = librosa.load(tmp_path, sr=SR, duration=DURATION)
    target_len = int(SR * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    Path(tmp_path).unlink(missing_ok=True)
    return y[:target_len]


def extract_features(y):
    import librosa
    feat = []
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    feat.extend(np.mean(mfcc, axis=1)); feat.extend(np.std(mfcc, axis=1))
    for order in [1, 2]:
        d = librosa.feature.delta(mfcc, order=order)
        feat.extend(np.mean(d, axis=1)); feat.extend(np.std(d, axis=1))
    chroma = librosa.feature.chroma_stft(y=y, sr=SR, hop_length=HOP_LENGTH)
    feat.extend(np.mean(chroma, axis=1)); feat.extend(np.std(chroma, axis=1))
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    feat.extend(np.mean(mel, axis=1)); feat.extend(np.std(mel, axis=1))
    for fn in [librosa.feature.spectral_centroid,
               librosa.feature.spectral_bandwidth,
               librosa.feature.spectral_rolloff]:
        v = fn(y=y, sr=SR, hop_length=HOP_LENGTH)
        feat.extend([np.mean(v), np.std(v)])
    ct = librosa.feature.spectral_contrast(y=y, sr=SR, hop_length=HOP_LENGTH)
    feat.extend(np.mean(ct, axis=1)); feat.extend(np.std(ct, axis=1))
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    feat.extend([np.mean(zcr), np.std(zcr)])
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    feat.extend([np.mean(rms), np.std(rms)])
    yh = librosa.effects.harmonic(y)
    tz = librosa.feature.tonnetz(y=yh, sr=SR)
    feat.extend(np.mean(tz, axis=1)); feat.extend(np.std(tz, axis=1))
    return np.array(feat, dtype=np.float32)


def to_melspec_image(y):
    import librosa
    from skimage.transform import resize
    mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=IMG_H, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_r  = resize(mel_db, (IMG_H, IMG_W), anti_aliasing=True)
    mel_n  = (mel_r - mel_r.min()) / (mel_r.max() - mel_r.min() + 1e-8)
    return mel_n[..., np.newaxis].astype(np.float32)


def mel_as_png(y):
    """Return mel spectrogram as PNG bytes for display."""
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 2.5))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")
    mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=SR, hop_length=HOP_LENGTH,
                             x_axis="time", y_axis="mel",
                             cmap="inferno", ax=ax)
    ax.set_xlabel("Time (s)", color="#6b6878", fontsize=8)
    ax.set_ylabel("Hz", color="#6b6878", fontsize=8)
    ax.tick_params(colors="#6b6878", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2835")
    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(y_audio, model_choice, loaded):
    """Returns (label, probs_dict) for the chosen model."""
    le = loaded["le"]
    classes = list(le.classes_)

    img     = to_melspec_image(y_audio)[np.newaxis, ...]
    img_rgb = np.repeat(img, 3, axis=-1) * 255

    if model_choice == "CNN":
        cnn = loaded.get("cnn")
        if cnn is None:
            return None, None, "cnn_model.keras not found"
        prob = cnn.predict(img, verbose=0)[0]

    elif model_choice == "MobileNetV2":
        mn = loaded.get("mobilenet")
        if mn is None:
            return None, None, "mobilenet_model.keras not found"
        prob = mn.predict(img_rgb, verbose=0)[0]

    elif model_choice == "XGBoost":
        xgb_ = loaded.get("xgb")
        scaler = loaded.get("scaler")
        if xgb_ is None or scaler is None:
            return None, None, "xgb_audio.json or scaler.pkl not found"
        feat = scaler.transform(extract_features(y_audio).reshape(1, -1))
        prob = xgb_.predict_proba(feat)[0]

    else:  # Stacked Ensemble
        cnn = loaded.get("cnn")
        mn  = loaded.get("mobilenet")
        xgb_ = loaded.get("xgb")
        scaler = loaded.get("scaler")
        meta_clf = loaded.get("meta_clf")
        if None in (cnn, mn, xgb_, scaler, meta_clf):
            return None, None, "One or more model files missing for ensemble"
        p_cnn = cnn.predict(img, verbose=0)
        p_mn  = mn.predict(img_rgb, verbose=0)
        feat  = scaler.transform(extract_features(y_audio).reshape(1, -1))
        p_xgb = xgb_.predict_proba(feat)
        meta  = np.hstack([p_cnn, p_mn, p_xgb])
        prob  = meta_clf.predict_proba(meta)[0]

    idx   = int(np.argmax(prob))
    label = classes[idx]
    probs = {cls: float(prob[i]) for i, cls in enumerate(classes)}
    return label, probs, None


# ── UI Layout ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">SoundID</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Animal Audio Classifier · Cats · Dogs · Birds</div>',
            unsafe_allow_html=True)

# Load models once
with st.spinner("Loading models…"):
    loaded, load_errors = load_models()

if load_errors:
    for e in load_errors:
        st.error(f"⚠️ {e}")

# Determine which models are actually available
available_models = []
if loaded.get("cnn") and loaded.get("le"):
    available_models.append("CNN")
if loaded.get("mobilenet") and loaded.get("le"):
    available_models.append("MobileNetV2")
if loaded.get("xgb") and loaded.get("scaler") and loaded.get("le"):
    available_models.append("XGBoost")
if all(k in loaded for k in ("cnn", "mobilenet", "xgb", "scaler", "meta_clf", "le")):
    available_models.insert(0, "Stacked Ensemble")

if not available_models:
    st.error("No model files found in the current directory. "
             "Run the training notebook first to generate the .keras / .json / .pkl files.")
    st.stop()

# ── Two-column layout ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown('<div class="upload-label">Upload audio file</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["wav", "mp3", "ogg", "flac"],
                                label_visibility="collapsed")

    st.markdown('<div class="upload-label" style="margin-top:1.2rem">Model</div>',
                unsafe_allow_html=True)
    model_choice = st.selectbox("", available_models, label_visibility="collapsed")

    # Show available model pills
    pills_html = '<div class="pill-row">'
    for m in ["Stacked Ensemble", "CNN", "MobileNetV2", "XGBoost"]:
        active = "active" if m == model_choice else ""
        avail  = m in available_models
        style  = "" if avail else "opacity:0.35;"
        pills_html += f'<span class="pill {active}" style="{style}">{m}</span>'
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)

    run_btn = st.button("▶  Identify Sound", disabled=(uploaded is None))

    if uploaded:
        st.audio(uploaded, format="audio/wav")

with col_right:
    if uploaded and run_btn:
        audio_bytes = uploaded.read()

        with st.spinner("Analysing…"):
            try:
                y_audio = load_audio_bytes(audio_bytes)

                # Mel spectrogram display
                st.markdown('<div class="spec-label">Mel Spectrogram</div>',
                            unsafe_allow_html=True)
                spec_png = mel_as_png(y_audio)
                st.image(spec_png, use_container_width=True)

                label, probs, err = predict(y_audio, model_choice, loaded)

                if err:
                    st.error(f"Inference error: {err}")
                else:
                    conf     = probs[label]
                    emoji    = ANIMAL_EMOJI.get(label, "🔊")
                    classes  = list(loaded["le"].classes_)

                    # ── Result card ──────────────────────────────────────────
                    bars_html = '<div class="bar-wrap">'
                    for i, (cls, color) in enumerate(zip(classes, BAR_COLORS)):
                        pct = probs.get(cls, 0.0) * 100
                        bars_html += f"""
                        <div style="margin-bottom:0.8rem">
                          <div class="bar-label">
                            <span>{cls}</span>
                            <span class="conf-value">{pct:.1f}%</span>
                          </div>
                          <div class="bar-track">
                            <div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
                          </div>
                        </div>"""
                    bars_html += "</div>"

                    card_html = f"""
                    <div class="result-card">
                      <div class="result-animal">{emoji} {label.title()}</div>
                      <div class="result-conf">
                        Confidence &nbsp;
                        <span class="conf-value">{conf:.1%}</span>
                        &nbsp;·&nbsp; via {model_choice}
                      </div>
                      <hr class="ruled">
                      {bars_html}
                    </div>"""
                    st.markdown(card_html, unsafe_allow_html=True)

            except Exception as ex:
                st.error(f"Something went wrong: {ex}")
                st.exception(ex)

    elif not uploaded:
        # Placeholder state
        st.markdown("""
        <div style="height:100%;display:flex;flex-direction:column;
                    justify-content:center;align-items:center;
                    padding:3rem 1rem;text-align:center;">
          <div style="font-size:3rem;margin-bottom:1rem;opacity:0.3">🎙️</div>
          <div style="font-family:'Space Mono',monospace;font-size:0.72rem;
                      color:#3d3a4a;letter-spacing:0.1em;text-transform:uppercase;
                      line-height:1.9">
            Upload a WAV file<br>and hit Identify Sound
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border:none;border-top:1px solid #1e1c28;margin:2rem 0 0.8rem'>",
            unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.65rem;
            color:#3d3a4a;letter-spacing:0.08em;text-align:center">
  SOUNDID · MFCC + MEL SPECTROGRAM · CNN · MOBILENETV2 · XGBOOST · STACKED ENSEMBLE
</div>
""", unsafe_allow_html=True)