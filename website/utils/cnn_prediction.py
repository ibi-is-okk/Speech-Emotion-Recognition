"""CNN loading and prediction. Needs torch (imported lazily so the rest of the website works without it)."""
from pathlib import Path
import numpy as np

import ser_audio
from utils.audio_preprocessing import cnn_input
from utils.evaluation import ROOT, EMOTIONS

CNN_PATH = ROOT / "models" / "cnn_best.pt"


class ModelUnavailable(Exception):
    """Raised with a message that tells the user exactly what to do."""


def load_cnn(path=CNN_PATH):
    """Load the checkpoint and check that it was trained with the preprocessing used here."""
    path = Path(path)
    if not path.exists():
        raise ModelUnavailable(f"CNN checkpoint not found: `models/{path.name}`. "
                               "Copy `cnn_best.pt` from Google Drive (SER/cnn_outputs/) into the `models/` folder.")
    try:
        from ser_cnn import load_checkpoint
    except ImportError as e:
        raise ModelUnavailable(f"PyTorch is not installed. Run `pip install torch` (details: {e})")
    model, ckpt = load_checkpoint(str(path), device="cpu")
    # consistency checks: the website must use the SAME preprocessing as training
    if list(ckpt["emotions"]) != list(ser_audio.EMOTIONS):
        raise ModelUnavailable("The checkpoint's emotion order differs from ser_audio.EMOTIONS.")
    pre = ckpt.get("preprocessing", {})
    for k, v in ser_audio.PREPROCESSING.items():
        if k in pre and pre[k] != v:
            raise ModelUnavailable(f"Preprocessing mismatch for '{k}': checkpoint={pre[k]}, ser_audio={v}. "
                                   "The model was trained with different audio settings.")
    return {"model": model, "ckpt": ckpt}


def try_load_cnn(path=CNN_PATH):
    """(bundle, None) on success, (None, message) on failure. Never raises."""
    try:
        return load_cnn(path), None
    except ModelUnavailable as e:
        return None, str(e)
    except Exception as e:                                    # corrupt file etc.
        return None, f"Could not load the CNN checkpoint: {type(e).__name__}: {e}"


def predict_from_input(bundle, x):
    """x: normalized (1,1,128,173) array -> softmax scores aligned with EMOTIONS."""
    from ser_cnn import predict_proba
    return predict_proba(bundle["model"], x, device="cpu")[0]


def predict_cnn(bundle, path):
    """Full path: file -> preprocessing (same as training) -> CNN -> scores."""
    ckpt = bundle["ckpt"]
    inp = cnn_input(path, ckpt["norm_mean"], ckpt["norm_std"])
    probs = predict_from_input(bundle, inp["x"])
    k = int(np.argmax(probs))
    return {"scores": probs, "pred": EMOTIONS[k], "kind": "probability (softmax)",
            "raw": inp["raw"], "n_valid": inp["n_valid"]}
