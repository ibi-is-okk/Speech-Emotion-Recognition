"""
ser_audio.py - the ONE place where audio is turned into the CNN input.

Used by: the Colab notebooks, the evaluation code and the Streamlit website.
It reproduces exactly what notebook 04_logmel_preprocessing.ipynb did, so a clip
uploaded to the website gets the same 128 x 173 log-mel input as the training clips.

This file needs only numpy + librosa (no torch).
"""
import numpy as np

# ---- settings (identical to notebook 04) -------------------------------------
SR = 22050            # sample rate (Hz)
TOP_DB = 25           # silence threshold for trimming
MAX_SEC = 4.0         # clips are capped / padded to this length
N_FFT = 2048
HOP = 512
N_MELS = 128
MAX_SAMPLES = int(MAX_SEC * SR)
N_FRAMES = 1 + MAX_SAMPLES // HOP          # 173

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for e, i in LABEL2ID.items()}

PREPROCESSING = dict(SR=SR, TOP_DB=TOP_DB, MAX_SEC=MAX_SEC, N_FFT=N_FFT, HOP=HOP,
                     N_MELS=N_MELS, N_FRAMES=N_FRAMES)


def waveform_to_logmel(y):
    """22.05 kHz mono float waveform -> (log-mel [128 x 173] float32, n_valid_frames).

    Steps (same as training): nan fix, trim silence, peak-normalize, cap at 4 s,
    mel spectrogram, power->dB (ref=1.0, top_db=80), pad on the right with the clip minimum.
    """
    import librosa
    y = np.nan_to_num(np.asarray(y, dtype=np.float32))
    yt, _ = librosa.effects.trim(y, top_db=TOP_DB)
    if len(yt) < int(0.3 * SR):            # trimming removed (almost) everything
        yt = y
    peak = np.max(np.abs(yt)) if len(yt) else 0.0
    if peak > 0:
        yt = yt / peak
    yt = yt[:MAX_SAMPLES]

    S = librosa.feature.melspectrogram(y=yt, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    S = librosa.power_to_db(S, ref=1.0, top_db=80.0)

    n_valid = min(S.shape[1], N_FRAMES)
    S = S[:, :N_FRAMES]
    if S.shape[1] < N_FRAMES:
        S = np.pad(S, ((0, 0), (0, N_FRAMES - S.shape[1])), constant_values=S.min())
    return S.astype(np.float32), n_valid


def load_waveform(path_or_file):
    """Read any audio file / file-like object the way training did: mono, 22.05 kHz."""
    import librosa
    y, _ = librosa.load(path_or_file, sr=SR, mono=True)
    return y


def load_logmel(path_or_file):
    """File (path or file-like) -> (raw log-mel [128 x 173], n_valid_frames)."""
    return waveform_to_logmel(load_waveform(path_or_file))


def normalize(S_raw, mean, std):
    """Apply the TRAINING mean / std (data/norm_stats.npy) to a raw log-mel."""
    return ((S_raw - float(mean)) / float(std)).astype(np.float32)


def prepare_cnn_input(path_or_file, mean, std):
    """Everything the CNN needs for one clip: normalized array of shape (1, 1, 128, 173)."""
    S_raw, n_valid = load_logmel(path_or_file)
    return normalize(S_raw, mean, std)[None, None, :, :], n_valid


def clip_duration_seconds(path_or_file):
    """Length of the uploaded recording in seconds (before trimming)."""
    return len(load_waveform(path_or_file)) / SR
