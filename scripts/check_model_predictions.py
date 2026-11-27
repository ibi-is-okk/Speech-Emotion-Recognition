"""Load all deployable models and predict one held-out RAVDESS clip."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "website"))
sys.path.insert(0, str(ROOT))

from utils.audio_preprocessing import classical_features
from utils.classical_prediction import predict_classical, try_load_classical
from utils.cnn_prediction import predict_cnn, try_load_cnn
from utils.evaluation import EMOTIONS


def main():
    labels = pd.read_csv(ROOT / "data" / "labels.csv")
    test_rows = labels[labels["split"] == "test"]
    clip = next(
        (
            ROOT / "data" / "RAVDESS" / f"Actor_{int(row.actor):02d}" / row.filename
            for row in test_rows.itertuples()
            if (ROOT / "data" / "RAVDESS" / f"Actor_{int(row.actor):02d}" / row.filename).is_file()
        ),
        None,
    )
    if clip is None:
        raise SystemExit("No held-out RAVDESS WAV found under data/RAVDESS.")

    cnn, error = try_load_cnn()
    if error:
        raise SystemExit(error)
    cnn_result = predict_cnn(cnn, clip)
    if len(cnn_result["scores"]) != len(EMOTIONS):
        raise RuntimeError(f"CNN returned {len(cnn_result['scores'])} scores, expected {len(EMOTIONS)}.")
    print(f"Clip: {clip.relative_to(ROOT)}")
    print(f"CNN: {cnn_result['pred']} ({cnn_result['kind']})")
    print(dict(zip(EMOTIONS, cnn_result["scores"].tolist())))

    features, _ = classical_features(clip)
    for name, path in (
        ("SVM", ROOT / "models" / "svm_tuned.joblib"),
        ("Random Forest", ROOT / "models" / "rf_tuned.joblib"),
    ):
        model, warning, error = try_load_classical(path, name)
        if error:
            raise SystemExit(error)
        if warning:
            print(f"WARNING: {warning}")
        result = predict_classical(model, features)
        if len(result["scores"]) != len(EMOTIONS):
            raise RuntimeError(f"{name} returned {len(result['scores'])} scores, expected {len(EMOTIONS)}.")
        print(f"{name}: {result['pred']} ({result['kind']})")
        print(dict(zip(EMOTIONS, result["scores"].tolist())))


if __name__ == "__main__":
    main()
