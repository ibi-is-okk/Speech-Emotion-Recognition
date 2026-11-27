import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEBSITE_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, WEBSITE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.evaluation import (EMOTIONS, METRIC_COLUMNS, load_comparison, load_confusion, find_confusion_image,
                              load_class_report, load_history, load_tuning, load_seed_variability,
                              load_final_summary, normalize_rows, difference_note, ROOT)
from utils.ui import MODEL_COLORS, show_plot, sidebar_title

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")
st.title("📊 Model Comparison")
st.caption("CNN vs SVM vs Random Forest on the **same 300 test clips** from 5 actors never used for training or tuning.")

cmp_df, sources = load_comparison()

# ------------------------------------------------------------------ sidebar
sidebar_title()
KEY_OF = {"CNN": "CNN", "SVM (tuned)": "SVM_tuned", "Random Forest (tuned)": "RF_tuned"}
available = list(cmp_df["Model"]) if cmp_df is not None else []
chosen = st.sidebar.multiselect("Models to compare", available, default=available)
cm_mode = st.sidebar.radio("Confusion matrix values", ["Counts", "Row-normalised (recall)"])
show_notes = st.sidebar.checkbox("Show methodology notes", True)

if cmp_df is None:
    st.warning("No evaluation results found yet.")
    st.markdown("""
Copy these files into the project, then reload this page:
- **CNN:** the `eval/` folder produced by `07_cnn_evaluation_colab.ipynb` goes into `outputs/cnn/eval/`
  (needs `model_comparison_row.csv`), plus `history.csv` into `outputs/cnn/`.
- **SVM / Random Forest:** `model_comparison.csv` into `outputs/classical/` (Member 1). Until then the website
  falls back to the existing `outputs/results_v2/` results if present.
""")
    st.stop()

if not chosen:
    st.info("Select at least one model in the Control Panel.")
    st.stop()
view = cmp_df[cmp_df["Model"].isin(chosen)].reset_index(drop=True)
get = lambda m, col: float(cmp_df.loc[cmp_df["Model"] == m, col].iloc[0]) if m in available else None

# ------------------------------------------------------------------ KPI row
st.subheader("Headline results")
cols = st.columns(len(view))
for c, (_, r) in zip(cols, view.iterrows()):
    with c:
        st.metric(f"{r['Model']} accuracy", f"{r['Accuracy']:.1%}")
        st.metric(f"{r['Model']} macro-F1", f"{r['Macro-F1']:.3f}")

if "CNN" in available and "SVM (tuned)" in available:
    st.info("🔎 " + difference_note(get("CNN", "Accuracy"), get("SVM (tuned)", "Accuracy")))

# ------------------------------------------------------------------ metrics chart + table
st.subheader("Accuracy, precision, recall and macro-F1")
left, right = st.columns([3, 2])
metric_cols = [c for c in METRIC_COLUMNS.values() if c in view.columns]
long = view.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Value")
with left:
    fig = px.bar(long, x="Metric", y="Value", color="Model", barmode="group", text_auto=".3f", range_y=[0, 1],
                 color_discrete_map=MODEL_COLORS, title="Test-set performance")
    show_plot(fig)
with right:
    st.dataframe(view.style.format({c: "{:.3f}" for c in metric_cols}), hide_index=True)
    st.caption("Macro = average over the 8 emotions, so the smaller neutral class counts as much as the others. "
               "Chance level is 0.125.")
    srcs = {m: str(p.relative_to(ROOT)) for m, p in sources.items() if m in chosen}
    with st.expander("Where these numbers come from"):
        st.json(srcs)

# ------------------------------------------------------------------ per-class
reports = {m: load_class_report(KEY_OF[m]) for m in chosen}
reports = {m: r for m, r in reports.items() if r is not None}
if reports:
    st.subheader("Per-emotion F1-score")
    pc = pd.concat([r["f1-score"].rename(m) for m, r in reports.items()], axis=1).reset_index(names="emotion")
    pc = pc.melt(id_vars="emotion", var_name="Model", value_name="F1")
    fig = px.bar(pc, x="emotion", y="F1", color="Model", barmode="group", range_y=[0, 1],
                 category_orders={"emotion": EMOTIONS}, color_discrete_map=MODEL_COLORS,
                 title="Which emotions does each model find easy or hard?")
    show_plot(fig)

# ------------------------------------------------------------------ confusion matrices
st.subheader("Confusion matrices")
st.caption("Rows = true emotion, columns = predicted emotion. The diagonal is correct predictions.")
cm_cols = st.columns(len(chosen))
for col, m in zip(cm_cols, chosen):
    with col:
        st.markdown(f"**{m}**")
        cm = load_confusion(KEY_OF[m])
        if cm is not None:
            shown = normalize_rows(cm) if cm_mode.startswith("Row") else cm
            fig = px.imshow(shown, text_auto=".2f" if cm_mode.startswith("Row") else "d", color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="True", color=""), aspect="auto")
            fig.update_xaxes(side="bottom")
            show_plot(fig)
        else:
            img = find_confusion_image(KEY_OF[m])
            if img is not None:
                st.image(str(img), caption="Image from the classical pipeline (counts left, row-normalised right)")
            else:
                st.info("No confusion matrix file yet. Expected `confusion_matrix_*.csv` in the results folder.")

# ------------------------------------------------------------------ training curves
st.subheader("CNN training curves")
hist = load_history()
if hist is None:
    st.info("Copy `history.csv` from `SER/cnn_outputs/` (Colab) into `outputs/cnn/` to show the training curves.")
else:
    best_epoch = int(hist.loc[hist["val_macro_f1"].idxmax(), "epoch"]) if "val_macro_f1" in hist else None
    c1, c2, c3 = st.columns(3)
    with c1:
        f = go.Figure()
        f.add_scatter(x=hist["epoch"], y=hist["train_loss"], name="train", line_color="#4C9BE8")
        f.add_scatter(x=hist["epoch"], y=hist["val_loss"], name="validation", line_color="#F28E2B")
        f.update_layout(title="Loss", xaxis_title="epoch")
        show_plot(f)
    with c2:
        f = go.Figure()
        f.add_scatter(x=hist["epoch"], y=hist["train_acc"], name="train", line_color="#4C9BE8")
        f.add_scatter(x=hist["epoch"], y=hist["val_acc"], name="validation", line_color="#F28E2B")
        f.update_layout(title="Accuracy", xaxis_title="epoch", yaxis_range=[0, 1])
        show_plot(f)
    with c3:
        f = go.Figure()
        f.add_scatter(x=hist["epoch"], y=hist["val_macro_f1"], name="validation macro-F1", line_color="#59A14F")
        if best_epoch:
            f.add_vline(x=best_epoch, line_dash="dash", line_color="gray", annotation_text=f"best epoch {best_epoch}")
        f.update_layout(title="Validation macro-F1", xaxis_title="epoch", yaxis_range=[0, 1])
        show_plot(f)
    st.caption("Curves of the final model. The validation actors (3, 6, 15, 20) were used to pick the best epoch and "
               "the configuration; the test actors were not. A gap between training and validation accuracy is "
               "expected with only about 900 training clips.")

# ------------------------------------------------------------------ notes
if show_notes:
    with st.expander("Methodology and caveats (read before quoting the numbers)", expanded=False):
        st.markdown("""
- **Same test set:** all models are scored on the same 300 clips from actors 1, 9, 12, 17, 19.
- **Training data differs:** the classical models were trained on the 19 non-test actors with augmentation, the CNN on 15 actors
  (the 4 validation actors were used for tuning). The comparison is therefore not perfectly controlled.
- **Noise:** with 300 clips from 5 actors, differences of about 3 accuracy points are within the expected noise.
- **Speaker normalisation:** a classical variant using per-speaker normalisation scores higher but needs several clips
  of the same speaker, so it cannot be used for a single uploaded recording and is not shown here.
""")
        summ = load_final_summary()
        seeds = load_seed_variability()
        if seeds is not None:
            st.markdown("**CNN: variation between random seeds (test set, for transparency only; the model was chosen on validation data)**")
            st.dataframe(seeds, hide_index=True)
        tuning = load_tuning()
        if tuning is not None:
            st.markdown("**CNN hyperparameter tuning (validation data only)**")
            st.dataframe(tuning, hide_index=True)
