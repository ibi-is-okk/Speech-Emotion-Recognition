import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so `utils` and `ser_audio` import from any page
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.evaluation import ROOT, EMOTIONS, load_final_summary
from utils.ui import EMOJI, COLORS, show_plot, sidebar_title

st.set_page_config(page_title="Project Overview", page_icon="📘", layout="wide")
sidebar_title()
st.sidebar.info("This page describes the project. Numbers come from `data/labels.csv`.")

st.title("📘 Project Overview")

# ------------------------------------------------------------------ introduction
st.header("What is this project?")
st.markdown("""
**Speech Emotion Recognition (SER)** means predicting the emotion a speaker expresses from their voice alone.
We cannot use the words: in our dataset every actor says the *same two sentences*, so the emotion is
carried only by **how** they are spoken (pitch, energy, speed, voice quality).

We compare two ways of solving it, on exactly the same data and the same unseen test speakers:

| | Traditional machine learning | Deep learning |
|---|---|---|
| **Input** | Hand-crafted audio features (MFCC, chroma, spectral, pitch) | Log-Mel spectrogram (an image of sound) |
| **Models** | SVM and Random Forest | Convolutional neural network (CNN) |
| **Idea** | A person decides which features describe emotion | The network learns which patterns matter |
""")

# ------------------------------------------------------------------ dataset
st.header("Dataset: RAVDESS")
labels_path = ROOT / "data" / "labels.csv"
df = pd.read_csv(labels_path) if labels_path.exists() else None

left, right = st.columns([1, 1])
with left:
    n_clips = f"{len(df):,}" if df is not None else "1,440 (1,439 after cleaning)"
    st.markdown(f"""
| | |
|---|---|
| **Name** | Ryerson Audio-Visual Database of Emotional Speech and Song (Livingstone & Russo, 2018) |
| **Subset used** | Audio-only speech |
| **Clips** | {n_clips} |
| **Speakers** | 24 professional actors (12 male, 12 female) |
| **Audio** | WAV, 48 kHz, 16-bit (converted to mono 22.05 kHz) |
| **Sentences** | 2 fixed sentences, each repeated twice |
| **Intensity** | normal and strong (neutral only normal) |
| **Target** | emotion, 8 classes |
| **Source** | zenodo.org/records/1188976 (CC BY-NC-SA 4.0) |
""")
    st.caption("Emotions are **acted** in a studio, so real-world speech is harder than this benchmark.")

with right:
    if df is not None:
        st.markdown("**Speaker-independent split** (whole actors, never mixed)")
        rows = []
        for s, label in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
            sub = df[df["split"] == s]
            rows.append({"Split": label, "Actors": ", ".join(str(a) for a in sorted(sub["actor"].unique())),
                         "Number of actors": sub["actor"].nunique(), "Clips": len(sub)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Splitting by actor prevents the model from memorising voices. "
                   "The test actors are never used for training or tuning.")
    else:
        st.warning("`data/labels.csv` not found, so the split table cannot be shown.")

if df is not None:
    counts = df.groupby(["emotion", "split"]).size().reset_index(name="clips")
    fig = px.bar(counts, x="emotion", y="clips", color="split", category_orders={"emotion": EMOTIONS,
                 "split": ["train", "val", "test"]}, title="Clips per emotion and split", barmode="stack",
                 color_discrete_map={"train": "#4C9BE8", "val": "#F28E2B", "test": "#59A14F"})
    show_plot(fig)
    st.caption("Neutral has about half as many clips as the other emotions (it has no 'strong' version). "
               "We handle this with class weights and report macro-averaged metrics.")

# ------------------------------------------------------------------ classes
st.header("The eight emotion classes")
cues = {
    "neutral": "Relaxed baseline speech. Only recorded at normal intensity.",
    "calm": "Soft, relaxed and low-arousal.",
    "happy": "Typically brighter, higher pitch and more energy.",
    "sad": "Typically lower pitch, slower and quieter.",
    "angry": "Typically loud and tense, with strong high-frequency energy.",
    "fearful": "Typically higher, unsteady pitch and a tense voice.",
    "disgust": "Typically slow and drawn out with a strained quality.",
    "surprised": "Typically sudden pitch rises and bursts of energy.",
}
n_per = df["emotion"].value_counts().to_dict() if df is not None else {}
cols = st.columns(4)
for i, e in enumerate(EMOTIONS):
    with cols[i % 4]:
        with st.container(border=True):
            st.markdown(f"### {EMOJI[e]} {e.title()}")
            if e in n_per:
                st.caption(f"{n_per[e]} clips")
            st.write(cues[e])
st.caption("Descriptions are typical acoustic cues from the speech-emotion literature, not measurements from this project.")

# ------------------------------------------------------------------ pipeline
st.header("How the models work")
a, b = st.columns(2)
with a:
    st.subheader("Traditional ML (Member 1)")
    st.markdown("""
1. Load audio, convert to mono 22.05 kHz, trim silence, peak-normalise, cap at 4 s
2. Extract **376 features** per clip: MFCCs, delta-MFCCs, chroma, spectral contrast, log-mel bands, pitch (F0) statistics, duration
3. Augment training clips (noise, pitch shift, time stretch)
4. Standardise features (fitted on training data only) and train an **SVM** (RBF kernel) and a **Random Forest**
5. Tune with actor-grouped cross-validation
""")
with b:
    st.subheader("Deep learning (Member 2)")
    st.markdown("""
1. Same audio preprocessing as the classical models
2. Convert to a **128 x 173 log-Mel spectrogram** and normalise with training statistics
3. **CNN**: 4 blocks of Conv, BatchNorm, ReLU, MaxPool and Dropout, then pooling and a linear layer
4. Class-weighted loss, learning-rate scheduling and early stopping
5. Tune on 4 validation actors, then evaluate **once** on the 5 test actors
""")
    summ = load_final_summary()
    if summ:
        cfg = summ.get("config", {})
        st.caption(f"Chosen configuration: **{summ.get('chosen_run', '?')}** "
                   f"(SpecAugment: {'on' if cfg.get('specaug') else 'off'}, dropout {cfg.get('dropout')}, "
                   f"learning rate {cfg.get('lr')}), seed {summ.get('chosen_seed')}.")

st.header("Limitations")
st.markdown("""
- **Small dataset:** about 1,440 clips and only 24 speakers; deep learning is data-hungry.
- **Acted emotions and two sentences:** results may not transfer to natural speech.
- **Five test actors (four male, one female):** results are noisy and mostly reflect male voices.
- **Uploaded recordings:** microphone, noise and language can differ from the studio recordings used for training.
""")
