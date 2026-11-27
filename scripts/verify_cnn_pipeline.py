"""
Verify that the WEBSITE path for the CNN reproduces what was done in training.

Run from the project root:   python scripts/verify_cnn_pipeline.py [--n 12]

Check 1 (needs only numpy + librosa): a clip run through the website preprocessing equals the stored
        spectrogram in data/X_logmel_norm.npy (made by notebook 04).
Check 2 (needs torch + models/cnn_best.pt + outputs/cnn/eval/predictions_test.csv): the website's CNN scores
        equal the scores written by the evaluation notebook for the same test clips.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))
sys.path.insert(0, str(ROOT))

import ser_audio                                                       # noqa: E402
from utils.audio_preprocessing import cnn_input                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=12, help="number of clips to check")
args = ap.parse_args()

labels = pd.read_csv(ROOT / "data" / "labels.csv")
X_norm = np.load(ROOT / "data" / "X_logmel_norm.npy", mmap_mode="r")
mean, std = np.load(ROOT / "data" / "norm_stats.npy")


def wav_path(row):
    return ROOT / "data" / "RAVDESS" / f"Actor_{int(row.actor):02d}" / row.filename


test_idx = np.where(labels["split"] == "test")[0]
rng = np.random.RandomState(0)
pick = rng.choice(test_idx, size=min(args.n, len(test_idx)), replace=False)
ok = True
skipped = []

print(f"CHECK 1: website preprocessing vs stored spectrograms ({len(pick)} test clips)")
worst = 0.0
for i in pick:
    row = labels.iloc[i]
    p = wav_path(row)
    if not p.exists():
        sys.exit(f"Audio not found: {p}. Download RAVDESS first (python ser_pipeline.py download).")
    inp = cnn_input(p, mean, std)
    worst = max(worst, float(np.abs(inp["x"][0, 0] - X_norm[i]).max()))
    if inp["n_valid"] != int(row.n_valid_frames):
        ok = False
        print("  n_valid_frames mismatch for", row.filename)
print(f"  max |difference| in normalized input = {worst:.2e}   ->  {'PASS' if worst < 1e-3 and ok else 'FAIL'}")
ok = ok and worst < 1e-3

print("\nCHECK 2: website CNN scores vs evaluation-notebook scores")
pred_csv = ROOT / "outputs" / "cnn" / "eval" / "predictions_test.csv"
try:
    from utils.cnn_prediction import try_load_cnn, predict_cnn
except Exception as e:
    sys.exit(f"  cannot import CNN code: {e}")
bundle, err = try_load_cnn()
if bundle is None:
    print("  SKIPPED:", err)
    skipped.append("check 2")
elif not pred_csv.exists():
    print("  SKIPPED: outputs/cnn/eval/predictions_test.csv not found (copy the eval/ folder from Colab).")
    skipped.append("check 2")
else:
    preds = pd.read_csv(pred_csv).set_index("filename")
    worst, mism = 0.0, 0
    for i in pick:
        row = labels.iloc[i]
        r = predict_cnn(bundle, wav_path(row))
        ref = preds.loc[row.filename, [f"prob_{e}" for e in ser_audio.EMOTIONS]].values.astype(float)
        worst = max(worst, float(np.abs(r["scores"] - ref).max()))
        mism += int(r["pred"] != preds.loc[row.filename, "pred"])
    good = worst < 1e-2 and mism == 0
    print(f"  max |score difference| = {worst:.2e}, label mismatches = {mism}   ->  {'PASS' if good else 'FAIL'}")
    ok = ok and good

print("\nRESULT:", ("PASSED" + (f" (skipped: {', '.join(skipped)})" if skipped else "")) if ok else "SOMETHING FAILED (see above)")
sys.exit(0 if ok else 1)
