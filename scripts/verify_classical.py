"""
Verify the classical models (Member 1) and export their results for the website.

Run from the project root:
    python scripts/verify_classical.py --features outputs/features_v2.csv \
        --svm models/svm_tuned.joblib --rf models/rf_tuned.joblib

Check 1: the saved models reproduce the test metrics that were reported (outputs/results_v2/model_comparison.csv).
Check 2: the website's feature extraction gives the same numbers as the rows of the training feature CSV.
Export : writes outputs/classical/{model_comparison.csv, confusion_matrix_*.csv, classification_report_*.csv,
         predictions_test_classical.csv}, which the Model Comparison page reads.
Use the SAME scikit-learn version that trained the models (pip show scikit-learn).
"""
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ser_pipeline as sp                                           # noqa: E402
from utils.audio_preprocessing import classical_features            # noqa: E402

EMOTIONS = sp.EMOTION_ORDER
TEST_ACTORS = [1, 9, 12, 17, 19]

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="outputs/features_v2.csv")
ap.add_argument("--svm", default="models/svm_tuned.joblib")
ap.add_argument("--rf", default="models/rf_tuned.joblib")
ap.add_argument("--reported", default="outputs/results_v2/model_comparison.csv")
ap.add_argument("--n-audio", type=int, default=8, help="clips used for the feature-extraction check")
ap.add_argument("--no-export", action="store_true")
args = ap.parse_args()

df = pd.read_csv(ROOT / args.features)
feat_cols = [c for c in df.columns if c not in sp.META_COLS]
test = df[(df["actor"].isin(TEST_ACTORS)) & (df["augmentation"] == "original")].reset_index(drop=True)
print(f"features: {len(feat_cols)} columns | test clips: {len(test)} (expected 300) | actors {sorted(test.actor.unique())}")
ok = len(test) == 300 and len(feat_cols) == 376
X, y = test[feat_cols].values, test["emotion"].values

reported = None
rp = ROOT / args.reported
if rp.exists():
    reported = pd.read_csv(rp).set_index("model")

rows, out = [], ROOT / "outputs" / "classical"
for key, path in (("SVM_tuned", args.svm), ("RF_tuned", args.rf)):
    p = ROOT / path
    if not p.exists():
        print(f"\n{key}: {path} not found, skipped")
        continue
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(p)
    for w in caught:
        if "version" in str(w.message).lower():
            print("  WARNING:", w.message)
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    _, _, f1w, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    row = {"model": key, "accuracy": acc, "precision_macro": pr, "recall_macro": rc, "f1_macro": f1, "f1_weighted": f1w}
    rows.append(row)
    print(f"\n{key}: accuracy {acc:.3f}  precision {pr:.3f}  recall {rc:.3f}  macro-F1 {f1:.3f}")
    if reported is not None and key in reported.index:
        diffs = {k: abs(row[k] - float(reported.loc[key, k])) for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")}
        match = max(diffs.values()) < 0.0015            # reported values are rounded to 3 decimals
        print(f"  reported (results_v2): max |difference| = {max(diffs.values()):.4f}  ->  {'PASS' if match else 'MISMATCH'}")
        ok = ok and match
    if not args.no_export:
        out.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(y, pred, labels=EMOTIONS)
        pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_csv(out / f"confusion_matrix_{key}.csv")
        rep = classification_report(y, pred, labels=EMOTIONS, output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.round(3).to_csv(out / f"classification_report_{key}.csv")
        pd.DataFrame({"filename": test["filename"], "actor": test["actor"], "gender": test["gender"],
                      "intensity": test["intensity"], "true": y, "pred": pred, "correct": y == pred}) \
            .to_csv(out / f"predictions_test_{key}.csv", index=False)

if rows and not args.no_export:
    pd.DataFrame(rows).to_csv(out / "model_comparison.csv", index=False)
    print(f"\nexported results to {out}")

print(f"\nCHECK 2: website feature extraction vs training CSV ({args.n_audio} test clips)")
rng = np.random.RandomState(0)
worst = 0.0
for i in rng.choice(len(test), size=min(args.n_audio, len(test)), replace=False):
    r = test.iloc[i]
    wav = ROOT / "data" / "RAVDESS" / f"Actor_{int(r.actor):02d}" / r.filename
    if not wav.exists():
        sys.exit(f"Audio not found: {wav}")
    vec, names = classical_features(wav)
    if names != feat_cols:
        ok = False
        print("  feature names / order differ from the training CSV!")
        break
    ref = r[feat_cols].values.astype(float)
    worst = max(worst, float(np.max(np.abs(vec[0] - ref) / (np.abs(ref) + 1e-9))))
good = worst < 1e-6
print(f"  max relative difference = {worst:.2e}  ->  {'PASS' if good else 'FAIL'}")
ok = ok and good
print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOMETHING FAILED (see above)")
sys.exit(0 if ok else 1)
