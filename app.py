
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))   # project root, so `utils` and `ser_audio` import from any page

import streamlit as st
import pandas as pd

from utils.evaluation import ROOT, load_comparison, load_history, find_file, CNN_DIRS
from utils.cnn_prediction import CNN_PATH
from utils.classical_prediction import SVM_PATH, RF_PATH

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="wide")

st.title("🎙️ Speech Emotion Recognition")
st.markdown("##### Can a computer hear *how* something is said? A CNN on spectrograms versus classical machine learning, "
            "tested on actors the models have never heard.")

c1, c2, c3 = st.columns(3)
with c1:
    st.page_link("pages/1_Project_Overview.py", label="Project Overview", icon="📘")
    st.caption("The problem, the RAVDESS dataset and the eight emotion classes.")
with c2:
    st.page_link("pages/2_Model_Comparison.py", label="Model Comparison", icon="📊")
    st.caption("CNN vs SVM vs Random Forest: metrics, confusion matrices, training curves.")
with c3:
    st.page_link("pages/3_Emotion_Prediction.py", label="Emotion Prediction", icon="🎧")
    st.caption("Upload a WAV recording and see what each model predicts.")

st.divider()

labels = ROOT / "data" / "labels.csv"
if labels.exists():
    df = pd.read_csv(labels)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clips", f"{len(df):,}")
    k2.metric("Actors", df["actor"].nunique())
    k3.metric("Emotion classes", df["emotion"].nunique())
    k4.metric("Test clips (unseen actors)", int((df["split"] == "test").sum()))

st.subheader("🔧 System status")
st.caption("Which saved files the website found. Missing items are explained on the page that needs them.")

cmp_df, _ = load_comparison()
rows = [
    ("CNN checkpoint", CNN_PATH.exists(), "models/cnn_best.pt"),
    ("SVM model", SVM_PATH.exists(), "models/svm_tuned.joblib (Member 1)"),
    ("Random Forest model (optional)", RF_PATH.exists(), "models/rf_tuned.joblib (Member 1)"),
    ("CNN test metrics", find_file("model_comparison_row.csv", CNN_DIRS) is not None, "outputs/cnn/eval/model_comparison_row.csv"),
    ("CNN training history", load_history() is not None, "outputs/cnn/history.csv"),
    ("Classical test metrics", cmp_df is not None and any("SVM" in m for m in cmp_df["Model"]),
     "outputs/classical/model_comparison.csv (or Member 1's outputs/results_v2/)"),
]
status = pd.DataFrame([{"Item": n, "Status": "✅ found" if ok else "⚠️ missing", "Expected at": p} for n, ok, p in rows])
st.dataframe(status, hide_index=True)

st.info("Models were trained on **acted RAVDESS speech** (two English sentences). Predictions on other recordings are "
        "illustrative: scores are not calibrated probabilities and the models can be wrong.")
