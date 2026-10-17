# Speech Emotion Recognition on RAVDESS (SVM vs Random Forest)

One script, `ser_pipeline.py`, covers all four phases. Each phase is a command you can run on its own.

## 1. Setup (about 5 minutes)

Requirements: Python 3.9 or newer, about 1 GB of free disk space, and an internet connection for the first run.

```bash
mkdir ravdess_ser && cd ravdess_ser      # put ser_pipeline.py and requirements.txt here
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Google Colab works too. Upload the two files, then run `!pip install -r requirements.txt` and use `!python ser_pipeline.py ...` for each command below.

## 2. Run it

Fast path, which runs everything:

```bash
python ser_pipeline.py all
```

Or one phase at a time, which is better for learning and debugging:

| Step | Command | What it does | Time |
|---|---|---|---|
| Phase 1 | `python ser_pipeline.py download` | Downloads `Audio_Speech_Actors_01-24.zip` (about 200 MB) from Zenodo into `data/RAVDESS/` and unzips it. | 1-5 min |
| Phase 1 | `python ser_pipeline.py eda` | Parses filenames, saves class-balance plots, duration plots, waveforms, mel-spectrograms and a `listen.html` page. | about 1 min |
| Phase 2 | `python ser_pipeline.py features` | Loads each clip at 22.05 kHz, trims silence, normalizes it and extracts 114 features into `outputs/features.csv`. | 2-6 min |
| Phase 3+4 | `python ser_pipeline.py train` | Runs the speaker-independent split, baselines, GridSearchCV, metrics, confusion matrices, feature importances and `summary.md`. | 5-20 min |

Useful options:

* `train --quick` runs tiny grids, so you can check the pipeline in under a minute.
* `train --split random` uses a random split with no speaker separation. Compare it with the default to see how much speaker leakage inflates accuracy.
* `--n-jobs 4` limits CPU use.
* `features --pad` zero-pads short clips to 4 s. The default does not pad.
* `--vocal-channel both` adds the singing set. Run `download --include-song` for it first.

### If the download fails (firewall or no internet)

1. Download manually from https://zenodo.org/records/1188976 (`Audio_Speech_Actors_01-24.zip`). Kaggle's "RAVDESS Emotional speech audio" also works.
2. Unzip it so you have `data/RAVDESS/Actor_01/ ... Actor_24/`. The exact nesting doesn't matter, because the code searches recursively for `.wav` files.
3. Skip `download` and run `eda`, `features` and `train`. Use `--data-dir path/to/folder` if your data is elsewhere.

## 3. Listening to samples

Open `outputs/eda/listen.html` in your browser. It has three random clips per emotion with play buttons. In a Jupyter notebook you can use `IPython.display.Audio("path/to/file.wav")` instead.

## 4. What each phase does

**Filename parsing.** `03-01-05-01-02-01-01.wav` has seven fields:

| Field | Meaning | Value in the example |
|---|---|---|
| Modality | 03 = audio-only | audio-only |
| Vocal channel | 01 = speech, 02 = song | speech |
| Emotion | 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised | angry |
| Intensity | 01 normal, 02 strong | normal |
| Statement | 01 or 02 | "Dogs are sitting by the door" |
| Repetition | 01 or 02 | 1 |
| Actor | Odd = male, even = female | Actor 1, male |

**Class balance.** The speech set has 1,440 clips. Neutral has only 96, because it has no "strong" intensity. Every other emotion has 192. The training code uses `class_weight="balanced"` to compensate.

**Preprocessing.**
* Resample to 22,050 Hz and trim silence at 25 dB.
* Peak-normalize each clip and cap it at 4 s.
* Features are means and standard deviations over time, so they don't depend on clip length.

**Features (114 in total).**

| Feature | Count |
|---|---|
| 40 MFCCs, mean and std | 80 |
| 12 chroma bins, mean and std | 24 |
| Spectral centroid, bandwidth, rolloff, ZCR and RMS (energy), mean and std each | 10 |

**Scaling.** `StandardScaler` sits inside the sklearn `Pipeline`, so it is fit on training data only. That means no test information leaks into the scaler, and during cross-validation it is re-fit on each fold. Scaling the whole dataset first would be a common mistake that inflates scores.

**Speaker-independent split.** `GroupShuffleSplit` groups by actor, so about 5 of the 24 actors are held out entirely for testing. GridSearchCV also uses `GroupKFold`, so tuning never sees the same actor in both the training and validation parts of a fold.

**Models and grids.**

| Model | Hyperparameters searched |
|---|---|
| SVM | Kernel, C, gamma. RBF: C in {0.1, 1, 10, 100}, gamma in {scale, 0.001, 0.003, 0.01}. Linear: C in {0.001 ... 10}. |
| Random Forest | n_estimators in {200, 500}, max_depth in {None, 20, 40}, min_samples_split in {2, 5, 10}, max_features in {sqrt, 0.3}. |

The search optimizes macro-F1, so it doesn't favour the majority classes. Baselines with default settings are reported next to the tuned models.

## 5. Output files

All of these are in `outputs/`:

| File | Contents |
|---|---|
| `eda/metadata.csv` | One row per clip, with the parsed fields. |
| `eda/class_balance.png`, `durations.png` | Class distributions and clip lengths. |
| `eda/waveform_per_emotion.png`, `melspec_per_emotion.png` | One example per emotion. |
| `eda/listen.html` | Audio players for sample clips. |
| `features.csv` | Unscaled feature table plus labels and actor IDs. |
| `results/summary.md` | Auto-written results summary: comparison table, best parameters, top confusions and feature importance. |
| `results/model_comparison.csv`, `.png` | Accuracy, macro precision, recall and F1 for all four models. |
| `results/classification_report_*.csv` | Per-emotion precision, recall and F1. |
| `results/confusion_matrix_*.png` | Counts and row-normalized matrices. |
| `results/gridsearch_*.csv` | Every hyperparameter combination with its CV score. |
| `results/rf_feature_importance*.csv`, `.png` | Per-feature and per-family importance. |
| `results/*_tuned.joblib` | Saved models. They include the scaler. |

Predicting a new clip:

```python
import joblib, pandas as pd
import ser_pipeline as sp
model = joblib.load("outputs/results/SVM_tuned.joblib")
y, sr = sp.load_and_preprocess("my_clip.wav")
x = pd.DataFrame([sp.extract_features(y, sr)])
print(model.predict(x.values))
```

## 6. What results to expect

These are typical figures, not guarantees. I could not run the full pipeline on the real dataset where I built this, so check your own numbers.

* Speaker-independent accuracy with classical features on 8 classes is usually around 45-65%.
* A random split often reads 10-20 points higher, because the model has heard the same actors during training.
* Commonly confused pairs are neutral and calm, fearful and sad, angry and disgust, and happy and surprised.
* In RF importance, MFCC features usually dominate.

For your write-up, use `summary.md` and the per-class reports to explain which emotions are hard and why, and compare SVM against RF. To get higher accuracy, the usual next steps are delta MFCCs, mel-spectrogram CNNs or pretrained models such as wav2vec2.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: librosa` | Activate your venv and run `pip install -r requirements.txt`. |
| `No RAVDESS-style .wav files found` | Check `--data-dir`. File names must keep the original `03-01-..` format. |
| `librosa.load` error or `audioread` warnings | Run `pip install soundfile` and make sure the files are not corrupted or only partly downloaded. |
| Very slow features or training | Use `--n-jobs 4`, and `--quick` for training. |
| Memory errors in the RF grid | Remove 500 from `n_estimators`. |
| `GridSearchCV` or `GroupKFold` errors | Use `--cv-folds 3`. You need at least as many training actors as folds. |

## 8. Improving the scores (v2 options)

These options are flags on the same script. Your first results stay untouched because each run uses its own file and folder name.

| Option | What it does |
|---|---|
| Extended features (default for `features`) | Adds delta and delta-delta MFCCs, 40 log-mel bands, spectral contrast, pitch (F0) statistics, voiced ratio and clip duration. This gives 376 features. Use `--basic` to get the original 114. |
| `features --augment` | Adds noise, pitch-shifted and time-stretched copies of every clip. Only training actors' copies are used for training. The test set always contains original clips only. |
| `train --speaker-norm` | Z-scores each feature within each actor, which removes voice identity. It uses no labels, but needs several clips from the same speaker. It is not usable for a single unseen clip. |
| `train --no-aug` | Ignores augmented rows. Use it for an ablation: how much did augmentation help? |

Commands (run from your project folder with the venv active):

```powershell
python ser_pipeline.py eda                                                          # fixes listen.html
python ser_pipeline.py features --augment --features-name features_v2.csv
python ser_pipeline.py train --features-name features_v2.csv --run-name results_v2
python ser_pipeline.py train --features-name features_v2.csv --run-name results_v2_spknorm --speaker-norm
python ser_pipeline.py train --features-name features_v2.csv --run-name results_v2_spknorm_noaug --speaker-norm --no-aug
```

Compare `outputs/results`, `results_v2`, `results_v2_spknorm` and `results_v2_spknorm_noaug` for your write-up.