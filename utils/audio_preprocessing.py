"""
Turns an uploaded recording into the inputs of the two model families.

IMPORTANT: no new preprocessing is written here. This module only calls the functions that were
used for training:
  * CNN        -> ser_audio.load_logmel / ser_audio.normalize     (identical to notebook 04)
  * SVM / RF   -> ser_pipeline.load_and_preprocess / extract_features (identical to the 'features' stage)
The uploaded bytes are written to a temporary .wav file so both pipelines read it exactly like a training file.
"""
import contextlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ser_audio  # noqa: E402  (CNN preprocessing: numpy + librosa only)


@contextlib.contextmanager
def temp_audio_file(data: bytes, suffix=".wav"):
    """Write uploaded bytes to a temp file, yield its path, always clean up."""
    fd, name = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        yield Path(name)
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def audio_info(path):
    """Basic facts about the original recording (before any resampling)."""
    import soundfile as sf
    i = sf.info(str(path))
    return {"duration": float(i.duration), "samplerate": int(i.samplerate), "channels": int(i.channels),
            "subtype": i.subtype, "format": i.format}


def check_audio(info):
    """Human-readable warnings about a recording (empty list = looks fine)."""
    w = []
    if info["duration"] < 0.5:
        w.append("The recording is very short (< 0.5 s). Predictions will be unreliable.")
    if info["duration"] > 10:
        w.append("The recording is long. Only the first 4 seconds after trimming silence are used (same as training).")
    if info["channels"] > 1:
        w.append("Stereo recording: the channels are averaged to mono (same as training).")
    return w


def cnn_input(path, norm_mean, norm_std):
    """Everything the CNN needs for one clip.
    Returns dict: x (1,1,128,173) normalized float32, raw (128,173) log-mel in dB, n_valid frames."""
    raw, n_valid = ser_audio.load_logmel(str(path))
    x = ser_audio.normalize(raw, norm_mean, norm_std)[None, None, :, :]
    return {"x": x, "raw": raw, "n_valid": int(n_valid)}


def classical_features(path):
    """Feature vector for the SVM / Random Forest, built with Member 1's own functions.
    Returns (array of shape (1, n_features), list of feature names)."""
    import ser_pipeline as sp
    y, sr = sp.load_and_preprocess(str(path))                 # default settings = the 'features' stage
    feats = sp.extract_features(y, sr, extended=True)         # same key order as the training CSV columns
    names = list(feats.keys())
    return np.array([[feats[k] for k in names]], dtype=np.float64), names


def waveform_preview(path, max_points=4000):
    """Downsampled waveform (for plotting only)."""
    y = ser_audio.load_waveform(str(path))
    step = max(1, len(y) // max_points)
    return np.arange(0, len(y), step) / ser_audio.SR, y[::step]
