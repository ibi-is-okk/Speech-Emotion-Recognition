import sys; sys.path.insert(0, "."); sys.path.insert(0, "website")
import numpy as np, pandas as pd, joblib, ser_pipeline as sp
from utils.audio_preprocessing import classical_features

df = pd.read_csv("outputs/features_v2.csv")
cols = [c for c in df.columns if c not in sp.META_COLS]
test = df[df.actor.isin([1, 9, 12, 17, 19]) & (df.augmentation == "original")].reset_index(drop=True)

web = np.array([classical_features(f"data/RAVDESS/Actor_{int(r.actor):02d}/{r.filename}")[0][0]
                for _, r in test.iterrows()])
csv = test[cols].values.astype(float)
y = test.emotion.values

ad = np.abs(web - csv); rd = ad / (np.abs(csv) + 1e-9)
rep = pd.DataFrame({"feature": cols, "max_abs": ad.max(0), "max_rel": rd.max(0),
                    "typical_size": np.median(np.abs(csv), 0)}).sort_values("max_rel", ascending=False)
print(rep.head(10).to_string(index=False))

for name, path in (("SVM", "models/svm_tuned.joblib"), ("RF", "models/rf_tuned.joblib")):
    m = joblib.load(path); pc, pw = m.predict(csv), m.predict(web)
    print(f"{name}: accuracy from CSV rows {np.mean(pc == y):.3f} | from website features "
          f"{np.mean(pw == y):.3f} | same prediction on {np.mean(pc == pw):.1%} of clips")