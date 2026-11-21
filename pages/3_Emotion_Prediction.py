import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so `utils` and `ser_audio` import from any page
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ser_audio
from utils.evaluation import EMOTIONS, ROOT
from utils.ui import EMOJI, COLORS, show_plot, sidebar_title
from utils.audio_preprocessing import temp_audio_file, audio_info, check_audio, cnn_input, classical_features, waveform_preview
from utils.cnn_prediction import CNN_PATH, try_load_cnn, predict_from_input
from utils.classical_prediction import SVM_PATH, RF_PATH, try_load_classical, predict_classical

st.set_page_config(page_title="Emotion Prediction", page_icon="🎧", layout="wide")
st.title("🎧 Emotion Prediction")
st.caption("Upload a short speech recording. The saved models predict its emotion. Nothing is trained here.")


# ------------------------------------------------------------------ cached loading (keyed by file modification time)
def _mtime(p):
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="Loading CNN...")
def get_cnn(_mtime_key):
    return try_load_cnn()


@st.cache_resource(show_spinner="Loading classical model...")
def get_classical(path_str, _mtime_key, name):
    return try_load_classical(path_str, name)


cnn_bundle, cnn_err = get_cnn(_mtime(CNN_PATH))
svm, svm_warn, svm_err = get_classical(str(SVM_PATH), _mtime(SVM_PATH), "SVM")
rf, rf_warn, rf_err = get_classical(str(RF_PATH), _mtime(RF_PATH), "Random Forest")

# ------------------------------------------------------------------ sidebar
sidebar_title()
st.sidebar.markdown("**Models**")
use_cnn = st.sidebar.checkbox("CNN (log-Mel spectrogram)", value=cnn_bundle is not None, disabled=cnn_bundle is None)
use_svm = st.sidebar.checkbox("SVM (audio features)", value=svm is not None, disabled=svm is None)
use_rf = st.sidebar.checkbox("Random Forest (audio features)", value=False, disabled=rf is None)
show_inputs = st.sidebar.checkbox("Show what the models see", value=True)
if st.sidebar.button("🔄 Reload models"):
    st.cache_resource.clear()      # use after adding or replacing a model file
    st.rerun()

missing = [m for m in (cnn_err, svm_err) if m]
if missing:
    with st.expander("⚠️ Some models are not available yet", expanded=cnn_bundle is None and svm is None):
        for m in missing:
            st.markdown(f"- {m}")
for w in (svm_warn, rf_warn):
    if w:
        st.warning(f"scikit-learn version warning while loading a model: {w}. Use the same scikit-learn version as the "
                   "machine that trained it (see requirements.txt).")

# ------------------------------------------------------------------ upload
uploaded = st.file_uploader("Upload a WAV recording (a few seconds of speech)", type=["wav"])
st.caption("Tip: the training clips are 3-5 second studio recordings of English sentences. Quiet recordings of a single "
           "speaker work best.")

if uploaded is None:
    st.info("👆 Upload a `.wav` file to get predictions.")
    st.stop()

data = uploaded.getvalue()
st.audio(data, format="audio/wav")

wants_cnn = use_cnn and cnn_bundle is not None
wants_classical = (use_svm and svm is not None) or (use_rf and rf is not None)
if not (wants_cnn or wants_classical):
    st.error("No model is available. See the warning above for what to add.")
    st.stop()

nm = (cnn_bundle["ckpt"]["norm_mean"], cnn_bundle["ckpt"]["norm_std"]) if cnn_bundle else (0.0, 1.0)


@st.cache_data(show_spinner="Preprocessing the recording (same steps as training)...")
def preprocess(audio_bytes, mean, std, want_cnn, want_classical):
    with temp_audio_file(audio_bytes) as p:
        out = {"info": audio_info(p), "wave": waveform_preview(p)}
        if want_cnn:
            out["cnn"] = cnn_input(p, mean, std)
        if want_classical:
            out["feats"] = classical_features(p)[0]
    return out


try:
    prep = preprocess(data, nm[0], nm[1], wants_cnn, wants_classical)
except Exception as e:
    st.error(f"This file could not be read as audio ({type(e).__name__}: {e}). Please upload a valid WAV file.")
    st.stop()

info = prep["info"]
i1, i2, i3, i4 = st.columns(4)
i1.metric("Duration", f"{info['duration']:.1f} s")
i2.metric("Sample rate", f"{info['samplerate']:,} Hz")
i3.metric("Channels", info["channels"])
i4.metric("Bit depth / type", info["subtype"])
for w in check_audio(info):
    st.warning(w)

# ------------------------------------------------------------------ predictions
results = {}
try:
    if wants_cnn:
        probs = predict_from_input(cnn_bundle, prep["cnn"]["x"])
        results["CNN"] = {"scores": probs, "pred": EMOTIONS[int(np.argmax(probs))], "kind": "probability (softmax)"}
    if use_svm and svm is not None:
        results["SVM"] = predict_classical(svm, prep["feats"])
    if use_rf and rf is not None:
        results["Random Forest"] = predict_classical(rf, prep["feats"])
except Exception as e:
    st.error(f"Prediction failed: {type(e).__name__}: {e}")
    st.stop()

st.subheader("Predictions")
cols = st.columns(len(results))
for col, (name, r) in zip(cols, results.items()):
    with col:
        with st.container(border=True):
            st.markdown(f"#### {name}")
            st.markdown(f"## {EMOJI[r['pred']]} {r['pred'].title()}")
            top = float(np.max(r["scores"]))
            st.metric("Top model score" if r["kind"].startswith("probability") else "Top relative score", f"{top:.0%}")
            fig = go.Figure(go.Bar(x=r["scores"], y=[e.title() for e in EMOTIONS], orientation="h",
                                   marker_color=[COLORS[e] for e in EMOTIONS],
                                   text=[f"{s:.0%}" for s in r["scores"]], textposition="outside"))
            fig.update_layout(height=340, xaxis=dict(range=[0, 1.1], title="model score"),
                              yaxis=dict(autorange="reversed"), title="Scores per emotion")
            show_plot(fig)
            st.caption(f"Score type: {r['kind']}.")

preds = [r["pred"] for r in results.values()]
if len(results) > 1:
    if len(set(preds)) == 1:
        st.success(f"✅ All selected models agree: **{preds[0].title()}**.")
    else:
        st.info("🔀 The models disagree: " + ", ".join(f"{n} → **{r['pred'].title()}**" for n, r in results.items())
                + ". Different models can make different mistakes.")

st.warning("These scores are **not calibrated probabilities**, and the models were trained on acted studio speech "
           "(two English sentences). On other recordings the prediction can easily be wrong.")

# ------------------------------------------------------------------ what the models see
if show_inputs:
    with st.expander("What the models see", expanded=False):
        t, y = prep["wave"]
        f = go.Figure(go.Scatter(x=t, y=y, line_color="#4C9BE8"))
        f.update_layout(title="Waveform (original recording)", xaxis_title="seconds", yaxis_title="amplitude", height=260)
        show_plot(f)
        if "cnn" in prep:
            raw, nv = prep["cnn"]["raw"], prep["cnn"]["n_valid"]
            secs = np.arange(raw.shape[1]) * ser_audio.HOP / ser_audio.SR
            f = px.imshow(raw, x=secs, origin="lower", aspect="auto", color_continuous_scale="Viridis",
                          labels=dict(x="seconds", y="mel band", color="dB"),
                          title="CNN input: log-Mel spectrogram (128 x 173) after trimming silence")
            f.add_vline(x=nv * ser_audio.HOP / ser_audio.SR, line_dash="dash", line_color="white")
            show_plot(f)
            st.caption(f"{nv} of {raw.shape[1]} frames contain audio; the rest is padding (dashed line = end of audio). "
                       "The model input is this array normalised with the training mean and standard deviation.")
        if "feats" in prep:
            st.caption(f"The SVM / Random Forest receive {prep['feats'].shape[1]} hand-crafted features "
                       "(MFCC, chroma, spectral, pitch...) extracted with Member 1's `ser_pipeline.py`.")

with st.expander("Preprocessing used (identical to training)"):
    st.markdown(f"""
1. Load as **mono**, resample to **{ser_audio.SR} Hz**
2. **Trim silence** (threshold {ser_audio.TOP_DB} dB), **peak-normalise**, keep at most **{ser_audio.MAX_SEC:.0f} s**
3. CNN: {ser_audio.N_MELS}-band log-Mel spectrogram (FFT {ser_audio.N_FFT}, hop {ser_audio.HOP}), padded to {ser_audio.N_FRAMES} frames, normalised with the training mean/std stored in the checkpoint
4. SVM / Random Forest: 376 features from `ser_pipeline.extract_features`, scaled by the scaler inside the saved model
""")
