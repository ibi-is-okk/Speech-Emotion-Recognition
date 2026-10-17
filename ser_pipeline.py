#!/usr/bin/env python3
"""
Speech Emotion Recognition on RAVDESS with classical ML (SVM + Random Forest).

Stages (run individually or all at once):
    python ser_pipeline.py download     # Phase 1: download + unzip RAVDESS speech
    python ser_pipeline.py eda          # Phase 1: metadata, class balance, waveforms, durations, listen page
    python ser_pipeline.py features     # Phase 2: preprocessing + feature extraction -> CSV
    python ser_pipeline.py train        # Phase 3+4: split, baseline, GridSearchCV, metrics, plots, report
    python ser_pipeline.py all          # everything above, in order

Run `python ser_pipeline.py <stage> -h` for options.
"""
import argparse
import base64
import os
import re
import sys
import time
import urllib.request
import zlib
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless-safe; figures are saved to disk
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed, dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SEED = 42
ZENODO_SPEECH_URL = ("https://zenodo.org/records/1188976/files/"
                     "Audio_Speech_Actors_01-24.zip?download=1")
ZENODO_SONG_URL = ("https://zenodo.org/records/1188976/files/"
                   "Audio_Song_Actors_01-24.zip?download=1")

EMOTION_MAP = {1: "neutral", 2: "calm", 3: "happy", 4: "sad",
               5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
EMOTION_ORDER = list(EMOTION_MAP.values())
INTENSITY_MAP = {1: "normal", 2: "strong"}
VOCAL_MAP = {1: "speech", 2: "song"}
STATEMENT_MAP = {1: "Kids are talking by the door", 2: "Dogs are sitting by the door"}

# Audio / feature settings
SR = 22050          # target sample rate (Hz)
TOP_DB = 25         # silence threshold for librosa.effects.trim
MAX_DURATION = 4.0  # seconds; longer clips are truncated after trimming
N_MFCC = 40
AUG_KINDS = ["noise", "pitch", "stretch"]   # augmentations made per clip when --augment is used


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _lazy_librosa():
    """Import librosa only when needed so the 'train' stage works without it."""
    try:
        import librosa
        return librosa
    except ImportError:
        sys.exit("librosa is not installed. Run: pip install -r requirements.txt")


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------
# PHASE 1a - Download
# ----------------------------------------------------------------------------
def _progress_hook(block, block_size, total):
    done = block * block_size
    if total > 0:
        pct = min(100, done * 100 / total)
        sys.stdout.write(f"\r  downloading... {pct:5.1f}%  ({done/1e6:6.1f}/{total/1e6:.1f} MB)")
        sys.stdout.flush()


def download_ravdess(data_dir, include_song=False):
    data_dir = ensure_dir(data_dir)
    jobs = [("Audio_Speech_Actors_01-24.zip", ZENODO_SPEECH_URL)]
    if include_song:
        jobs.append(("Audio_Song_Actors_01-24.zip", ZENODO_SONG_URL))

    for name, url in jobs:
        marker = data_dir / f".{name}.done"
        if marker.exists():
            print(f"[download] {name} already extracted, skipping.")
            continue
        zip_path = data_dir / name
        if not zip_path.exists():
            print(f"[download] fetching {name} from Zenodo (~200-250 MB)")
            try:
                urllib.request.urlretrieve(url, zip_path, _progress_hook)
                print()
            except Exception as e:  # network blocked, etc.
                if zip_path.exists():
                    zip_path.unlink()
                sys.exit(
                    f"\n[download] FAILED: {e}\n"
                    "Download manually from https://zenodo.org/records/1188976 "
                    f"(or Kaggle: 'uwrfkaggler/ravdess-emotional-speech-audio'),\n"
                    f"then unzip so that Actor_01 ... Actor_24 folders are inside: {data_dir}")
        print(f"[download] extracting {name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
        marker.touch()
        zip_path.unlink()  # save disk space
    n = len(list(Path(data_dir).rglob("*.wav")))
    print(f"[download] done. Found {n} .wav files under {data_dir}")


# ----------------------------------------------------------------------------
# PHASE 1b - Metadata from filenames
# Format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
#   e.g. 03-01-05-01-02-01-01.wav -> audio-only, speech, angry, normal,
#        "Dogs are sitting by the door", repetition 1, actor 1 (male)
# ----------------------------------------------------------------------------
def parse_filename(path):
    parts = Path(path).stem.split("-")
    if len(parts) != 7 or not all(p.isdigit() for p in parts):
        return None
    mod, voc, emo, inten, stmt, rep, actor = map(int, parts)
    if emo not in EMOTION_MAP:
        return None
    return {
        "path": str(path),
        "filename": Path(path).name,
        "modality": {1: "full-AV", 2: "video-only", 3: "audio-only"}.get(mod, str(mod)),
        "vocal_channel": VOCAL_MAP.get(voc, str(voc)),
        "emotion_id": emo,
        "emotion": EMOTION_MAP[emo],
        "intensity": INTENSITY_MAP.get(inten, str(inten)),
        "statement": STATEMENT_MAP.get(stmt, str(stmt)),
        "repetition": rep,
        "actor": actor,
        "gender": "male" if actor % 2 == 1 else "female",  # odd = male, even = female
    }


def build_metadata(data_dir, vocal_channel="speech"):
    rows = [parse_filename(p) for p in sorted(Path(data_dir).rglob("*.wav"))]
    rows = [r for r in rows if r is not None]
    if not rows:
        sys.exit(f"No RAVDESS-style .wav files found under {data_dir}. "
                 f"Run the 'download' stage or check --data-dir.")
    df = pd.DataFrame(rows)
    if vocal_channel != "both":
        df = df[df["vocal_channel"] == vocal_channel]
    df = df.sort_values(["actor", "filename"]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# PHASE 1c - EDA
# ----------------------------------------------------------------------------
def _durations(path):
    librosa = _lazy_librosa()
    y, sr = librosa.load(path, sr=SR)
    yt, _ = librosa.effects.trim(y, top_db=TOP_DB)
    return len(y) / sr, len(yt) / sr


def run_eda(data_dir, out_dir, vocal_channel="speech", n_jobs=-1):
    librosa = _lazy_librosa()
    import librosa.display  # noqa: F401
    out = ensure_dir(Path(out_dir) / "eda")
    df = build_metadata(data_dir, vocal_channel)
    df.to_csv(out / "metadata.csv", index=False)
    print(f"[eda] {len(df)} files, {df['actor'].nunique()} actors, "
          f"{df['emotion'].nunique()} emotions")

    # --- class balance -----------------------------------------------------
    print("\n[eda] Emotion counts:\n", df["emotion"].value_counts().reindex(EMOTION_ORDER))
    print("\n[eda] Emotion x intensity:\n", pd.crosstab(df["emotion"], df["intensity"]).reindex(EMOTION_ORDER))
    print("\n[eda] Emotion x gender:\n", pd.crosstab(df["emotion"], df["gender"]).reindex(EMOTION_ORDER))
    print("\nNOTE: 'neutral' has no 'strong' intensity in RAVDESS, so it has about half "
          "as many clips as the other emotions. We use class_weight='balanced' in training.")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    sns.countplot(data=df, x="emotion", order=EMOTION_ORDER, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Class balance (all clips)")
    sns.countplot(data=df, x="emotion", hue="gender", order=EMOTION_ORDER, ax=axes[1])
    axes[1].set_title("By actor gender")
    sns.countplot(data=df, x="emotion", hue="intensity", order=EMOTION_ORDER, ax=axes[2])
    axes[2].set_title("By emotional intensity")
    for a in axes:
        a.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    plt.savefig(out / "class_balance.png", dpi=140)
    plt.close()

    # --- durations ---------------------------------------------------------
    print("[eda] computing durations (loads every file once)...")
    res = Parallel(n_jobs=n_jobs)(delayed(_durations)(p) for p in df["path"])
    df["duration_raw"] = [r[0] for r in res]
    df["duration_trimmed"] = [r[1] for r in res]
    df[["filename", "emotion", "actor", "duration_raw", "duration_trimmed"]].to_csv(
        out / "durations.csv", index=False)
    print(df[["duration_raw", "duration_trimmed"]].describe().round(2))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    axes[0].hist(df["duration_raw"], bins=30, alpha=.6, label="raw")
    axes[0].hist(df["duration_trimmed"], bins=30, alpha=.6, label="after trim")
    axes[0].set_xlabel("seconds"); axes[0].set_title("Duration distribution"); axes[0].legend()
    sns.boxplot(data=df, x="emotion", y="duration_trimmed", order=EMOTION_ORDER, ax=axes[1])
    axes[1].set_title("Trimmed duration by emotion"); axes[1].tick_params(axis="x", rotation=40)
    sns.boxplot(data=df, x="gender", y="duration_trimmed", ax=axes[2])
    axes[2].set_title("Trimmed duration by gender")
    plt.tight_layout()
    plt.savefig(out / "durations.png", dpi=140)
    plt.close()

    # --- waveforms + mel spectrograms (one fixed sample per emotion) ---------
    rng = np.random.RandomState(SEED)
    samples = {e: df[df["emotion"] == e].sample(1, random_state=rng).iloc[0] for e in EMOTION_ORDER
               if (df["emotion"] == e).any()}
    ncols = 4
    nrows = int(np.ceil(len(samples) / ncols))
    for kind in ("waveform", "melspec"):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, (emo, row) in zip(axes, samples.items()):
            y, sr = librosa.load(row["path"], sr=SR)
            if kind == "waveform":
                librosa.display.waveshow(y, sr=sr, ax=ax, color="#4C72B0")
                ax.set_ylim(-1, 1)
            else:
                S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64),
                                        ref=np.max)
                librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", ax=ax)
            ax.set_title(f"{emo}  (actor {row['actor']}, {row['gender'][0].upper()}, {row['intensity']})",
                         fontsize=9)
        for ax in axes[len(samples):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out / f"{kind}_per_emotion.png", dpi=140)
        plt.close()

    # --- "listen" page: open in a browser and click play --------------------
    html = ["<html><head><meta charset='utf-8'><title>RAVDESS samples</title>",
            "<style>body{font-family:sans-serif;margin:2em}td,th{padding:6px 12px;border-bottom:1px solid #ddd}</style>",
            "</head><body><h1>RAVDESS sample clips</h1>",
            "<table><tr><th>Emotion</th><th>Clip</th><th>Actor</th><th>Intensity</th><th>Play</th></tr>"]
    for emo in EMOTION_ORDER:
        sub = df[df["emotion"] == emo]
        if sub.empty:
            continue
        for _, r in sub.sample(min(3, len(sub)), random_state=SEED).iterrows():
            # Embed the audio inside the HTML (base64) so the page works anywhere:
            # no broken relative paths, no Windows backslash / space-in-folder issues.
            b64 = base64.b64encode(Path(r["path"]).read_bytes()).decode("ascii")
            html.append(f"<tr><td><b>{emo}</b></td><td>{r['filename']}</td>"
                        f"<td>{r['actor']} ({r['gender']})</td><td>{r['intensity']}</td>"
                        f"<td><audio controls preload='none' src='data:audio/wav;base64,{b64}'></audio></td></tr>")
    html.append("</table></body></html>")
    (out / "listen.html").write_text("\n".join(html), encoding="utf-8")
    print(f"\n[eda] saved figures + metadata.csv + listen.html in {out}")
    print("      Open listen.html in a browser to hear the samples (or in a notebook use "
          "IPython.display.Audio(path)).")


# ----------------------------------------------------------------------------
# PHASE 2 - Preprocessing + feature engineering
# ----------------------------------------------------------------------------
def load_and_preprocess(path, sr=SR, top_db=TOP_DB, max_duration=MAX_DURATION, pad=False):
    """Load -> resample -> trim silence -> peak-normalise -> cap (or pad) length."""
    librosa = _lazy_librosa()
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = np.nan_to_num(y)
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    if len(yt) < int(0.3 * sr):          # safety: trimming removed (almost) everything
        yt = y
    peak = np.max(np.abs(yt))
    if peak > 0:
        yt = yt / peak                   # amplitude normalisation (removes loudness offsets
                                         # caused by mic gain; RMS energy is still informative
                                         # *within* a clip, but not across clips)
    target = int(max_duration * sr)
    if len(yt) > target:
        yt = yt[:target]
    elif pad and len(yt) < target:
        yt = np.pad(yt, (0, target - len(yt)))
    return yt, sr


def estimate_f0(y, sr, fmin=65.0, fmax=500.0, frame_length=2048, hop=512, voicing_thresh=0.3):
    """Frame-wise F0 by FFT autocorrelation. Pure numpy, so it cannot crash numba.
    Returns (f0 per frame in Hz, boolean mask of voiced frames)."""
    y = np.nan_to_num(np.asarray(y, dtype=np.float64))
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop]
    energy = np.sqrt((frames ** 2).mean(axis=1))
    frames = frames * np.hanning(frame_length)
    spec = np.fft.rfft(frames, n=2 * frame_length, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :frame_length]
    ac = ac / (ac[:, :1] + 1e-12)                       # normalise: lag 0 == 1
    lag_min, lag_max = int(sr / fmax), min(int(sr / fmin), frame_length - 1)
    seg = ac[:, lag_min:lag_max]
    lag = seg.argmax(axis=1) + lag_min
    strength = seg.max(axis=1)
    f0 = sr / lag
    voiced = (strength > voicing_thresh) & (energy > 0.1 * max(energy.max(), 1e-12))
    return f0, voiced


def extract_features(y, sr, extended=True):
    """Return an ordered dict {feature_name: value}. Mean + std over time for each.

    basic    : MFCC(40), chroma(12), spectral centroid/bandwidth/rolloff, ZCR, RMS
    extended : basic + delta/delta-delta MFCC, log-mel bands, spectral contrast,
               pitch (F0) statistics, voiced ratio and clip duration (speaking-pace cue)
    """
    librosa = _lazy_librosa()
    f = {}

    def add(name, M):
        M = np.atleast_2d(M)
        for i in range(M.shape[0]):
            suffix = f"_{i + 1}" if M.shape[0] > 1 else ""
            f[f"{name}{suffix}_mean"] = float(np.mean(M[i]))
            f[f"{name}{suffix}_std"] = float(np.std(M[i]))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    add("mfcc", mfcc)                                                             # spectral envelope
    add("chroma", librosa.feature.chroma_stft(y=y, sr=sr))                        # 12 pitch classes
    add("spectral_centroid", librosa.feature.spectral_centroid(y=y, sr=sr))
    add("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr))
    add("spectral_rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr))
    add("zcr", librosa.feature.zero_crossing_rate(y))
    rms = librosa.feature.rms(y=y)
    add("rms", rms)                                                               # energy

    if extended:
        add("mfcc_delta", librosa.feature.delta(mfcc))                            # how MFCCs change
        add("mfcc_delta2", librosa.feature.delta(mfcc, order=2))
        mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40))
        add("mel", mel_db)
        add("contrast", librosa.feature.spectral_contrast(y=y, sr=sr))
        # --- pitch (F0): pure-numpy autocorrelation tracker (no numba) -------------------
        f0, voiced = estimate_f0(y, sr)
        f0v = f0[voiced] if voiced.sum() > 5 else f0
        f["f0_mean"] = float(np.mean(f0v))
        f["f0_std"] = float(np.std(f0v))
        f["f0_p10"] = float(np.percentile(f0v, 10))
        f["f0_p50"] = float(np.percentile(f0v, 50))
        f["f0_p90"] = float(np.percentile(f0v, 90))
        f["f0_absdiff"] = float(np.mean(np.abs(np.diff(f0v)))) if len(f0v) > 1 else 0.0
        f["voiced_ratio"] = float(np.mean(voiced))
        f["duration"] = float(len(y) / sr)
    return {k: (v if np.isfinite(v) else 0.0) for k, v in f.items()}


def augment_audio(y, sr, kind, rng):
    """Mild augmentations. Only ever applied to clips that end up in TRAINING data
    (the train stage never evaluates on augmented rows)."""
    librosa = _lazy_librosa()
    if kind == "noise":
        snr_db = rng.uniform(20, 30)
        amp = np.sqrt(np.mean(y ** 2) + 1e-12) / (10 ** (snr_db / 20))
        ya = y + rng.randn(len(y)) * amp
    elif kind == "pitch":
        ya = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(rng.choice([-1.0, 1.0])))
    elif kind == "stretch":
        ya = librosa.effects.time_stretch(y, rate=float(rng.choice([0.95, 1.05])))
    else:
        raise ValueError(kind)
    ya = np.nan_to_num(ya)
    peak = np.max(np.abs(ya))
    ya = ya / peak if peak > 0 else ya
    return ya[:int(MAX_DURATION * sr)]


def _process_one(path, pad, extended, augment):
    warnings.filterwarnings("ignore")        # silence librosa deprecation spam inside workers
    y, sr = load_and_preprocess(path, pad=pad)
    out = [("original", extract_features(y, sr, extended))]
    if augment:
        rng = np.random.RandomState(zlib.crc32(str(path).encode()))   # reproducible per file
        for kind in AUG_KINDS:
            out.append((kind, extract_features(augment_audio(y, sr, kind, rng), sr, extended)))
    return out


def run_features(data_dir, out_dir, vocal_channel="speech", n_jobs=-1, pad=False,
                 extended=True, augment=False, features_name="features.csv"):
    out = ensure_dir(out_dir)
    df = build_metadata(data_dir, vocal_channel)
    print(f"[features] {'extended' if extended else 'basic'} feature set, "
          f"augmentation={'ON' if augment else 'off'}, {len(df)} files, n_jobs={n_jobs} ...")
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_one)(p, pad, extended, augment) for p in df["path"])
    meta_cols = ["filename", "emotion", "emotion_id", "intensity", "statement",
                 "repetition", "actor", "gender", "vocal_channel"]
    rows = []
    for (_, meta), res in zip(df[meta_cols].iterrows(), results):
        for kind, feats in res:
            rows.append({**meta.to_dict(), "augmentation": kind, **feats})
    full = pd.DataFrame(rows)
    csv_path = out / features_name
    full.to_csv(csv_path, index=False)
    n_feat = full.shape[1] - len(meta_cols) - 1
    print(f"[features] {n_feat} features x {len(full)} rows in {time.time() - t0:.0f}s -> {csv_path}")
    print("           (scaling is applied later inside the sklearn Pipeline so the scaler is "
          "fit on training data only - no leakage.)")


# ----------------------------------------------------------------------------
# PHASE 3 + 4 - Training, tuning, evaluation
# ----------------------------------------------------------------------------
META_COLS = ["filename", "emotion", "emotion_id", "intensity", "statement",
             "repetition", "actor", "gender", "vocal_channel", "augmentation"]


def feature_family(name):
    """Group feature names for aggregated importance (mfcc_7_mean -> 'mfcc mean')."""
    m = re.match(r"^(.*)_(mean|std)$", name)
    base, stat = (m.group(1), m.group(2)) if m else (name, "")
    base = re.sub(r"_\d+$", "", base)
    return f"{base} {stat}".strip()


def speaker_normalize(df, feat_cols):
    """Z-score every feature WITHIN each actor (using that actor's original clips).
    Removes voice identity (pitch range, timbre) so what is left is mostly emotion.
    Uses no labels, but needs several clips from the same speaker at prediction time."""
    out = df.copy()
    orig = df[df["augmentation"] == "original"]
    for actor, g in orig.groupby("actor"):
        mu = g[feat_cols].mean()
        sd = g[feat_cols].std().replace(0, 1.0).fillna(1.0)
        m = df["actor"] == actor
        out.loc[m, feat_cols] = (df.loc[m, feat_cols] - mu) / (sd + 1e-8)
    return out


def plot_confusion(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"{title} - counts")
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f"{title} - row-normalised (recall)")
    for a in axes:
        a.set_xlabel("Predicted"); a.set_ylabel("True")
        a.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return cm


def top_confusions(cm, labels, k=6):
    pairs = [(labels[i], labels[j], int(cm[i, j]), cm[i, j] / max(cm[i].sum(), 1))
             for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j] > 0]
    return sorted(pairs, key=lambda t: -t[2])[:k]


def evaluate(name, model, X_te, y_te, labels, out):
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
    _, _, f1w, _ = precision_recall_fscore_support(y_te, y_pred, average="weighted", zero_division=0)
    rep = classification_report(y_te, y_pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.round(3).to_csv(out / f"classification_report_{name}.csv")
    print(f"\n===== {name} =====")
    print(classification_report(y_te, y_pred, labels=labels, zero_division=0, digits=3))
    cm = plot_confusion(y_te, y_pred, labels, name, out / f"confusion_matrix_{name}.png")
    return {"model": name, "accuracy": acc, "precision_macro": p, "recall_macro": r,
            "f1_macro": f1, "f1_weighted": f1w}, cm


def run_train(out_dir, split="speaker", test_size=0.2, cv_folds=5, quick=False, n_jobs=-1,
              features_name="features.csv", run_name="results", speaker_norm=False, no_aug=False):
    out = ensure_dir(Path(out_dir) / run_name)
    csv_path = Path(out_dir) / features_name
    if not csv_path.exists():
        sys.exit(f"{csv_path} not found. Run the 'features' stage first.")
    df = pd.read_csv(csv_path)
    if "augmentation" not in df.columns:      # features.csv from the first version
        df["augmentation"] = "original"
    if no_aug:
        df = df[df["augmentation"] == "original"].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    if speaker_norm:
        print("[train] per-speaker feature normalisation ON")
        df = speaker_normalize(df, feat_cols)
    is_orig = (df["augmentation"] == "original").values
    print(f"[train] rows: {len(df)} ({is_orig.sum()} original, {(~is_orig).sum()} augmented)")
    X = df[feat_cols].values
    y = df["emotion"].values
    groups = df["actor"].values
    labels = [e for e in EMOTION_ORDER if e in set(y)]
    print(f"[train] {X.shape[0]} samples, {X.shape[1]} features, {len(labels)} classes, "
          f"{len(np.unique(groups))} actors")

    # ---- split ----------------------------------------------------------------
    if split == "speaker":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
        tr, te = next(gss.split(X, y, groups))
        te = te[is_orig[te]]          # never test on augmented clips
        test_actors = sorted(np.unique(groups[te]).tolist())
        print(f"[train] SPEAKER-INDEPENDENT split. Test actors: {test_actors} "
              f"(train actors: {len(np.unique(groups[tr]))})")
        cv = GroupKFold(n_splits=min(cv_folds, len(np.unique(groups[tr]))))
    else:  # random stratified split - leaks speaker identity, gives optimistic scores
        base = np.where(is_orig)[0]
        tr_o, te = train_test_split(base, test_size=test_size, stratify=y[base], random_state=SEED)
        # augmented copies of a test clip must not leak into training
        te_files = set(zip(df.iloc[te]["actor"], df.iloc[te]["filename"]))
        keep = np.array([(a, f) not in te_files for a, f in zip(df["actor"], df["filename"])])
        tr = np.where(keep)[0]
        print("[train] RANDOM stratified split (actors overlap -> optimistic accuracy!)")
        cv = None
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    g_tr = groups[tr]
    print(f"[train] train={len(tr)}  test={len(te)}")

    # Scaler lives INSIDE the pipeline -> fit on each training fold only (no leakage).
    def make(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    # max_iter bounds worst-case time of a single libsvm fit (linear kernel on a large,
    # near-duplicate augmented set can otherwise run for a very long time)
    svm = make(SVC(class_weight="balanced", random_state=SEED, cache_size=1000, max_iter=100000))
    rf = make(RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=n_jobs))

    # ---- hyper-parameter grids ------------------------------------------------
    if quick:
        svm_grid = [{"clf__kernel": ["rbf"], "clf__C": [1, 10], "clf__gamma": ["scale"]},
                    {"clf__kernel": ["linear"], "clf__C": [0.1]}]
        rf_grid = {"clf__n_estimators": [100], "clf__max_depth": [None, 20],
                   "clf__min_samples_split": [2, 5]}
    else:
        svm_grid = [{"clf__kernel": ["rbf"], "clf__C": [0.1, 1, 10, 100],
                     "clf__gamma": ["scale", 0.0003, 0.001, 0.003, 0.01]},
                    {"clf__kernel": ["linear"], "clf__C": [0.001, 0.01, 0.1]}]
        rf_grid = {"clf__n_estimators": [500], "clf__max_depth": [None, 30],
                   "clf__min_samples_split": [2, 5], "clf__max_features": ["sqrt", 0.3]}

    # ---- baselines (default hyper-parameters) -----------------------------------
    results, cms = [], {}
    for nm, mdl in (("SVM_baseline", svm), ("RF_baseline", rf)):
        mdl.fit(X_tr, y_tr)
        r, cms[nm] = evaluate(nm, mdl, X_te, y_te, labels, out)
        results.append(r)

    # ---- grid search ------------------------------------------------------------
    tuned = {}
    for nm, mdl, grid in (("SVM_tuned", svm, svm_grid), ("RF_tuned", rf, rf_grid)):
        print(f"\n[train] GridSearchCV for {nm} ...")
        gs = GridSearchCV(mdl, grid, scoring="f1_macro", cv=cv if cv else cv_folds,
                          n_jobs=1 if nm.startswith("RF") else n_jobs, refit=True, verbose=1)
        if cv is not None:
            gs.fit(X_tr, y_tr, groups=g_tr)   # GroupKFold keeps each actor in one fold only
        else:
            gs.fit(X_tr, y_tr)
        print(f"[train] best params: {gs.best_params_}  CV macro-F1={gs.best_score_:.3f}")
        pd.DataFrame(gs.cv_results_).sort_values("rank_test_score").to_csv(
            out / f"gridsearch_{nm}.csv", index=False)
        r, cms[nm] = evaluate(nm, gs.best_estimator_, X_te, y_te, labels, out)
        r["cv_f1_macro"] = gs.best_score_
        r["best_params"] = str(gs.best_params_)
        results.append(r)
        tuned[nm] = gs
        dump(gs.best_estimator_, out / f"{nm}.joblib")

    res_df = pd.DataFrame(results)
    res_df.to_csv(out / "model_comparison.csv", index=False)

    # comparison bar chart
    plot_df = res_df.melt(id_vars="model", value_vars=["accuracy", "f1_macro", "precision_macro", "recall_macro"])
    plt.figure(figsize=(10, 4.5))
    sns.barplot(data=plot_df, x="model", y="value", hue="variable")
    plt.ylim(0, 1); plt.title("Test-set performance"); plt.xticks(rotation=15)
    plt.tight_layout(); plt.savefig(out / "model_comparison.png", dpi=140); plt.close()

    # ---- Random Forest feature importance ------------------------------------------
    rf_best = tuned["RF_tuned"].best_estimator_.named_steps["clf"]
    imp = pd.DataFrame({"feature": feat_cols, "importance": rf_best.feature_importances_})
    imp["family"] = imp["feature"].map(feature_family)
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(out / "rf_feature_importance.csv", index=False)
    fam = imp.groupby("family")["importance"].sum().sort_values(ascending=False)
    fam.to_csv(out / "rf_feature_importance_by_family.csv")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    top = imp.head(25).iloc[::-1]
    axes[0].barh(top["feature"], top["importance"], color="#4C72B0")
    axes[0].set_title("Random Forest - top 25 features (Gini importance)")
    axes[1].barh(fam.index[::-1], fam.values[::-1], color="#55A868")
    axes[1].set_title("Importance summed by feature family")
    plt.tight_layout(); plt.savefig(out / "rf_feature_importance.png", dpi=140); plt.close()

    # ---- auto-generated summary --------------------------------------------------------
    cfg = (f"features file `{features_name}` | speaker normalisation: {speaker_norm} | "
           f"augmented rows used for training: {int((~is_orig[tr]).sum())}")
    write_summary(out, res_df, tuned, cms, labels, imp, fam, split, df, te, cfg)
    print(f"\n[train] all outputs saved in {out}")
    print(res_df.round(3).to_string(index=False))


def write_summary(out, res_df, tuned, cms, labels, imp, fam, split, df, te_idx, cfg=""):
    best = res_df[res_df["model"].isin(["SVM_tuned", "RF_tuned"])].sort_values("f1_macro", ascending=False).iloc[0]
    L = ["# Speech Emotion Recognition - results summary", ""]
    L.append(f"* Split: **{split}** | test clips: {len(te_idx)} | test actors: "
             f"{sorted(df.iloc[te_idx]['actor'].unique().tolist())}")
    L.append(f"* Features: {len(imp)} | {cfg}")
    L.append("")
    L.append("## Model comparison (test set)")
    L.append("")
    L.append("```")
    L.append(res_df.drop(columns=["best_params"], errors="ignore").round(3).to_string(index=False))
    L.append("```")
    L.append("")
    for nm, gs in tuned.items():
        L.append(f"* **{nm}** best params: `{gs.best_params_}` (CV macro-F1 {gs.best_score_:.3f})")
    L.append("")
    L.append(f"**Best tuned model: {best['model']}** (accuracy {best['accuracy']:.3f}, "
             f"macro-F1 {best['f1_macro']:.3f}).")
    L.append("")
    L.append("## Most frequently confused emotions (true -> predicted)")
    for nm in ("SVM_tuned", "RF_tuned"):
        L.append(f"\n**{nm}**\n")
        for t, p, c, frac in top_confusions(cms[nm], labels):
            L.append(f"* {t} -> {p}: {c} clips ({frac:.0%} of true {t})")
    L.append("")
    L.append("## Random Forest feature importance")
    L.append("\nTop 10 individual features:\n")
    for _, r in imp.head(10).iterrows():
        L.append(f"* `{r['feature']}` - {r['importance']:.4f}")
    L.append("\nBy feature family:\n")
    for k, v in fam.head(10).items():
        L.append(f"* {k}: {v:.3f}")
    L.append("")
    L.append("## Notes for your write-up")
    L.append("* Compare per-class F1 in `classification_report_*.csv` to see which emotions are easy/hard.")
    L.append("* Acoustically similar emotions (e.g. neutral/calm, fearful/sad, angry/disgust, "
             "happy/surprised) are the usual confusion pairs - check the confusion matrices.")
    L.append("* Re-run with `--split random` to see how much speaker leakage inflates accuracy.")
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    def common(sp):
        sp.add_argument("--data-dir", default="data/RAVDESS", help="where RAVDESS wavs live / will be extracted")
        sp.add_argument("--out-dir", default="outputs", help="where outputs are written")
        sp.add_argument("--vocal-channel", choices=["speech", "song", "both"], default="speech")
        sp.add_argument("--n-jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")

    def feat_args(sp):
        sp.add_argument("--pad", action="store_true", help="zero-pad short clips to MAX_DURATION (default: no padding)")
        sp.add_argument("--basic", action="store_true", help="use the original 114 basic features instead of the extended set")
        sp.add_argument("--augment", action="store_true", help="add noise / pitch / stretch copies of each clip (training only)")
        sp.add_argument("--features-name", default="features.csv", help="output CSV name inside --out-dir")

    def train_args(sp):
        sp.add_argument("--split", choices=["speaker", "random"], default="speaker")
        sp.add_argument("--test-size", type=float, default=0.2)
        sp.add_argument("--cv-folds", type=int, default=5)
        sp.add_argument("--quick", action="store_true", help="tiny grids for a fast smoke test")
        sp.add_argument("--speaker-norm", action="store_true", help="z-score features within each actor")
        sp.add_argument("--no-aug", action="store_true", help="ignore augmented rows even if present in the CSV")
        sp.add_argument("--features-name", default="features.csv", help="features CSV inside --out-dir")
        sp.add_argument("--run-name", default="results", help="results sub-folder name inside --out-dir")

    s = sub.add_parser("download"); common(s)
    s.add_argument("--include-song", action="store_true", help="also download the RAVDESS song set")
    s = sub.add_parser("eda"); common(s)
    s = sub.add_parser("features"); common(s); feat_args(s)
    s = sub.add_parser("train"); common(s); train_args(s)
    s = sub.add_parser("all"); common(s)
    s.add_argument("--include-song", action="store_true")
    feat_args(s)
    s.add_argument("--split", choices=["speaker", "random"], default="speaker")
    s.add_argument("--test-size", type=float, default=0.2)
    s.add_argument("--cv-folds", type=int, default=5)
    s.add_argument("--quick", action="store_true")
    s.add_argument("--speaker-norm", action="store_true")
    s.add_argument("--no-aug", action="store_true")
    s.add_argument("--run-name", default="results")
    return p


def main():
    a = build_parser().parse_args()
    st = a.stage
    if st in ("download", "all"):
        download_ravdess(a.data_dir, include_song=getattr(a, "include_song", False) or a.vocal_channel in ("song", "both"))
    if st in ("eda", "all"):
        run_eda(a.data_dir, a.out_dir, a.vocal_channel, a.n_jobs)
    if st in ("features", "all"):
        run_features(a.data_dir, a.out_dir, a.vocal_channel, a.n_jobs, a.pad,
                     extended=not a.basic, augment=a.augment, features_name=a.features_name)
    if st in ("train", "all"):
        run_train(a.out_dir, a.split, a.test_size, a.cv_folds, a.quick, a.n_jobs,
                  a.features_name, a.run_name, a.speaker_norm, a.no_aug)


if __name__ == "__main__":
    main()