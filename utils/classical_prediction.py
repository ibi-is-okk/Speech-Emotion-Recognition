"""SVM / Random Forest loading and prediction (scikit-learn models saved by Member 1's pipeline)."""
import warnings
from pathlib import Path
import numpy as np

from utils.audio_preprocessing import classical_features
from utils.evaluation import ROOT, EMOTIONS

SVM_PATH = ROOT / "models" / "svm_tuned.joblib"
RF_PATH = ROOT / "models" / "rf_tuned.joblib"


def try_load_classical(path, name="model"):
    """(model, warning_or_None, error_or_None). Never raises."""
    path = Path(path)
    if not path.exists():
        return None, None, (f"{name} not found: `models/{path.name}`. "
                            "Member 1 must add this file (see the README for the export step).")
    try:
        import joblib
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(path)
        warn = "; ".join(str(w.message) for w in caught if "version" in str(w.message).lower()) or None
        return model, warn, None
    except Exception as e:
        return None, None, f"Could not load {name}: {type(e).__name__}: {e}"


def _align(scores, classes):
    """Reorder scores from the model's class order (alphabetical) to EMOTIONS order."""
    pos = {c: i for i, c in enumerate(classes)}
    return np.array([scores[pos[e]] for e in EMOTIONS], dtype=float)


def predict_classical(model, features):
    """features: (1, n_features) from audio_preprocessing.classical_features.
    Returns dict(scores aligned to EMOTIONS, pred, kind)."""
    n_expected = getattr(model, "n_features_in_", None)
    if n_expected is not None and n_expected != features.shape[1]:
        raise ValueError(f"The model expects {n_expected} features but the audio pipeline produced "
                         f"{features.shape[1]}. Was the model trained on the extended feature set (376 features)?")
    classes = list(model.classes_)
    pred = str(model.predict(features)[0])
    if hasattr(model, "predict_proba"):
        scores = _align(model.predict_proba(features)[0], classes)
        kind = "probability"
    else:                                                     # SVC without probability=True
        d = _align(np.asarray(model.decision_function(features))[0], classes)
        s = d - d.min()
        scores = s / s.sum() if s.sum() > 0 else np.full(len(d), 1 / len(d))
        kind = "relative decision score (the SVM does not output probabilities)"
    return {"scores": scores, "pred": pred, "kind": kind}


def classical_from_path(model, path):
    feats, _ = classical_features(path)
    return predict_classical(model, feats)
