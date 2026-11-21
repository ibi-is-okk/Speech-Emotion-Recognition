"""
Loads the saved evaluation results (metrics, confusion matrices, training history) for the website.

Nothing is computed or invented here: every number shown on the dashboard is read from a file
that an evaluation notebook / script wrote. If a file is missing the function returns None and the
page shows what to copy where.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# Where results are looked for (first match wins).
CNN_DIRS = [ROOT / "outputs" / "cnn" / "eval", ROOT / "outputs" / "cnn"]
CLASSICAL_DIRS = [ROOT / "outputs" / "classical", ROOT / "outputs" / "results_v2"]   # results_v2 = Member 1's existing run
CNN_TRAIN_DIRS = [ROOT / "outputs" / "cnn", ROOT / "outputs" / "cnn" / "eval"]

METRIC_COLUMNS = {"accuracy": "Accuracy", "precision_macro": "Precision (macro)",
                  "recall_macro": "Recall (macro)", "f1_macro": "Macro-F1"}
MODEL_KEYS = {"CNN": "CNN", "SVM_tuned": "SVM (tuned)", "RF_tuned": "Random Forest (tuned)"}


def find_file(name, dirs):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def _read_csv(path, **kw):
    if not path:
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return None


# ---------------------------------------------------------------- metrics
def load_cnn_metrics():
    """(DataFrame with one row, source path) or (None, None)."""
    p = find_file("model_comparison_row.csv", CNN_DIRS)
    df = _read_csv(p)
    if df is None or "accuracy" not in df.columns:
        return None, None
    df = df.copy()
    df["model"] = "CNN"
    return df, p


def load_classical_metrics():
    """Rows for SVM_tuned and RF_tuned. Prefers outputs/classical/, falls back to Member 1's results_v2."""
    p = find_file("model_comparison.csv", CLASSICAL_DIRS)
    df = _read_csv(p)
    if df is None or "model" not in df.columns:
        return None, None
    df = df[df["model"].isin(["SVM_tuned", "RF_tuned"])].copy()
    return (df if len(df) else None), p


def load_comparison():
    """Combined table with display names. Returns (DataFrame or None, {model: source file})."""
    frames, sources = [], {}
    cnn, p = load_cnn_metrics()
    if cnn is not None:
        frames.append(cnn)
        sources["CNN"] = p
    cls, p = load_classical_metrics()
    if cls is not None:
        frames.append(cls)
        for m in cls["model"]:
            sources[MODEL_KEYS[m]] = p
    if not frames:
        return None, sources
    df = pd.concat(frames, ignore_index=True)
    df["Model"] = df["model"].map(MODEL_KEYS)
    cols = ["Model"] + [c for c in METRIC_COLUMNS if c in df.columns]
    df = df[cols].rename(columns=METRIC_COLUMNS)
    return df, sources


# ---------------------------------------------------------------- confusion matrices
_CONF_FILES = {"CNN": ("confusion_matrix_CNN", CNN_DIRS),
               "SVM_tuned": ("confusion_matrix_SVM_tuned", CLASSICAL_DIRS),
               "RF_tuned": ("confusion_matrix_RF_tuned", CLASSICAL_DIRS)}


def load_confusion(model_key):
    """Confusion matrix counts as a DataFrame (rows = true, columns = predicted) or None."""
    stem, dirs = _CONF_FILES[model_key]
    df = _read_csv(find_file(stem + ".csv", dirs), index_col=0)
    if df is None or df.shape != (8, 8):
        return None
    try:
        return df.loc[EMOTIONS, EMOTIONS]
    except KeyError:
        return None


def find_confusion_image(model_key):
    """PNG fallback (Member 1's existing results_v2 images) when no CSV counts exist."""
    stem, dirs = _CONF_FILES[model_key]
    return find_file(stem + ".png", dirs)


# ---------------------------------------------------------------- per-class reports
def load_class_report(model_key):
    name = {"CNN": "classification_report_CNN.csv", "SVM_tuned": "classification_report_SVM_tuned.csv",
            "RF_tuned": "classification_report_RF_tuned.csv"}[model_key]
    dirs = CNN_DIRS if model_key == "CNN" else CLASSICAL_DIRS
    df = _read_csv(find_file(name, dirs), index_col=0)
    if df is None or not set(EMOTIONS) <= set(df.index):
        return None
    return df.loc[EMOTIONS, ["precision", "recall", "f1-score", "support"]]


# ---------------------------------------------------------------- CNN training information
def load_history():
    df = _read_csv(find_file("history.csv", CNN_TRAIN_DIRS))
    return df if df is not None and "epoch" in df.columns else None


def load_tuning():
    for name in ("tuning_results_round2.csv", "tuning_results.csv"):
        df = _read_csv(find_file(name, CNN_TRAIN_DIRS))
        if df is not None:
            return df
    return None


def load_seed_variability():
    return _read_csv(find_file("seed_variability.csv", CNN_DIRS))


def load_final_summary():
    p = find_file("final_summary.json", CNN_TRAIN_DIRS)
    try:
        return json.load(open(p)) if p else None
    except Exception:
        return None


# ---------------------------------------------------------------- helpers
def normalize_rows(cm):
    return cm.div(cm.sum(axis=1).replace(0, 1), axis=0)


def difference_note(a, b, name_a="CNN", name_b="SVM", noise=0.03):
    """Plain-language comment on the difference between two accuracies (computed, not hard-coded)."""
    d = a - b
    if abs(d) < noise:
        return (f"{name_a} and {name_b} differ by {abs(d) * 100:.1f} accuracy points. With only 300 test clips from "
                f"5 actors, differences below about {noise * 100:.0f} points are within the expected noise, so "
                f"the two models should be considered comparable.")
    better, worse = (name_a, name_b) if d > 0 else (name_b, name_a)
    return (f"{better} is ahead of {worse} by {abs(d) * 100:.1f} accuracy points. This is larger than the "
            f"~{noise * 100:.0f}-point noise expected on 300 test clips, but it is still based on only 5 test actors.")
