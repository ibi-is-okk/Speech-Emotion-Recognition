# RAVDESS Speech Emotion Recognition

A speaker-independent speech emotion recognition project comparing hand-crafted audio features with classical machine learning (SVM and Random Forest) against a log-Mel CNN. The Streamlit site presents the project, compares saved test results, and predicts emotions from uploaded WAV recordings.

## Problem and dataset

The task is to classify acted speech into eight emotion labels from voice acoustics, not sentence meaning. The project uses the audio-only speech subset of RAVDESS, available from [Zenodo record 1188976](https://zenodo.org/records/1188976), under the CC BY-NC-SA 4.0 licence.

The source set contains 1,440 speech clips from 24 actors. Phase 1 found one duplicate audio recording and removed that copy, leaving 1,439 labeled clips in `data/labels.csv`. The cleaned label file contains 8 emotions and the actor-independent split. Neutral is the smallest class (96 clips); the other classes have 191 or 192 clips after cleaning. These counts come from `data/cleaning_report.csv` and `data/labels.csv`.

The classical extended feature table has 376 numeric features per clip, as counted from `outputs/features_v2.csv`: MFCCs, delta and delta-delta MFCCs, log-Mel bands, chroma, spectral contrast, spectral statistics, pitch statistics, voiced ratio, and duration. The CNN consumes a 128 × 173 log-Mel spectrogram using the shared preprocessing in `ser_audio.py`.

## Cleaning, EDA, and split

The cleaning report records no missing labels, unreadable files, inconsistent sample rates, silent clips, or clipped clips. It found five stereo files, which the loaders convert to mono, and 258 IQR outlier flags, which were retained as natural emotional variation. The source audio is 48 kHz and is resampled during preprocessing. These findings are recorded in `data/cleaning_report.csv`.

The analysis set is split by actor to avoid speaker leakage. Validation actors are 3, 6, 15, and 20; test actors are 1, 9, 12, 17, and 19. The test set contains 300 clips from those held-out actors. Split assignments are in `data/labels.csv` and `data/split.csv`.

## Modeling pipelines

### Classical ML

`ser_pipeline.py` performs metadata parsing, EDA, preprocessing, feature extraction, speaker-grouped training/validation, and evaluation. SVM and Random Forest use a `StandardScaler` inside each sklearn pipeline. The website uses the non-speaker-normalized SVM and Random Forest checkpoints from the `results_v2` run, staged under `models/`. `scripts/verify_classical.py` checks their reported metrics and exports website result files to `outputs/classical/`.

### CNN

`ser_audio.py` provides the shared log-Mel preprocessing, and `ser_cnn.py` defines the CNN and checkpoint helpers. Notebooks `04_logmel_preprocessing.ipynb` through `07_cnn_evaluation_colab.ipynb` cover CNN inputs, training, and evaluation. The selected checkpoint and evaluation files are stored under `models/` and `outputs/cnn/`. The CNN run metadata in `outputs/cnn/final_summary.json` records `quick_dry_run: false`.

## Final test results

The table below is read from `outputs/cnn/eval/model_comparison_row.csv` and `outputs/classical/model_comparison.csv` (classical rows exported by `scripts/verify_classical.py`). Values are rounded to three decimal places for display.

| Model | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 |
|---|---:|---:|---:|---:|
| CNN | 0.557 | 0.555 | 0.566 | 0.544 |
| SVM (non-speaker-normalized) | 0.553 | 0.594 | 0.556 | 0.554 |
| Random Forest (non-speaker-normalized) | 0.477 | 0.498 | 0.491 | 0.473 |

These are test results, not guarantees for arbitrary recordings. The per-emotion reports and confusion matrices are saved under `outputs/cnn/eval/` and `outputs/classical/`.

## Key insights and limitations

- The neutral class is smaller than the other emotion classes, so macro-averaged metrics are important.
- Confusion matrices show overlap among calm, neutral, and sad; the actual counts are in the saved confusion-matrix CSVs.
- Speaker-normalized experiments in `outputs/results_v2_spknorm/` score higher than the non-normalized run in some metrics, but they require several clips from the same speaker and are not suitable for predicting one uploaded clip. The website deliberately uses the non-normalized models.
- The test set has 300 clips from only 5 actors. Differences around ±3 percentage points should be interpreted cautiously.
- RAVDESS contains acted studio speech and repeated sentences. Results may not transfer to natural speech, other languages, microphones, or noisy recordings.
- The CNN is trained on a small dataset; class-level performance and actor-specific variation should be considered when interpreting its predictions.

## Repository structure

```text
.
├── website/
│   ├── app.py
│   ├── pages/                     Streamlit overview, comparison, and prediction pages
│   └── utils/                     Result loading, audio preprocessing, model prediction, UI
├── ser_audio.py                   Shared audio preprocessing for CNN input
├── ser_cnn.py                     CNN definition and checkpoint utilities
├── ser_pipeline.py                Classical ML pipeline
├── scripts/                       Verification and real-clip prediction checks
├── notebooks/                     Cleaning, EDA, modeling, and final comparison notebooks
├── data/                          Labels and split metadata; WAV/array data are generated or downloaded
├── models/                        CNN checkpoint and deployable classical model files
├── outputs/                       Evaluation artifacts, reports, and generated figures
├── requirements.txt               Runtime dependencies
└── requirements-dev.txt           Runtime plus notebook dependencies
```

## Setup and run

Use Python 3.11 for the live site. In Streamlit Community Cloud, select Python 3.11 in the app's Advanced settings. Hugging Face Spaces is a fallback if the Community Cloud memory limit is exceeded.

Create and activate a virtual environment, then install runtime dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For notebook work, install the development dependencies as well:

```powershell
python -m pip install -r requirements-dev.txt
```

Download the RAVDESS audio if `data/RAVDESS/` is not present:

```powershell
python ser_pipeline.py download
```

The classical feature CSV is ignored by Git because it is generated locally. Create it before running notebook `08` or classical verification if it is missing:

```powershell
python ser_pipeline.py features --augment --features-name features_v2.csv
```

Run the notebooks in order. Notebooks `06_cnn_training_colab.ipynb` and `07_cnn_evaluation_colab.ipynb` require a Colab GPU: run them in Colab with `QUICK = False`, then download the executed notebooks and `cnn_outputs/`; place the notebooks under `notebooks/` and the outputs under `outputs/cnn/`. Do not substitute fabricated notebook outputs.

Execute the local-capable notebooks from the project root:

```powershell
Get-ChildItem notebooks -File | Where-Object { $_.Name -match '^(0[1-5]|0[89])_' } | Sort-Object Name | ForEach-Object {
    jupyter nbconvert --to notebook --execute --inplace $_.FullName
}
```

Start the Streamlit app from the repository root:

```powershell
python -m streamlit run website/app.py
```

Open the local URL printed by Streamlit. The site reads saved models and results; it does not train models at startup.

Run the verification scripts after their required data and checkpoints are present:

```powershell
python scripts/verify_classical.py --features outputs/features_v2.csv
python scripts/verify_cnn_pipeline.py
python scripts/check_model_predictions.py
```

The CNN verification needs the arrays created by notebook `04_logmel_preprocessing.ipynb`, the CNN checkpoint, and its evaluation predictions. Check the `quick_dry_run` field in `outputs/cnn/final_summary.json` before reporting the CNN results. In this workspace, the classical score-reproduction checks pass, but its strict feature-reproduction check against the saved CSV does not; see `HANDOFF.md`. Do not loosen the check tolerance to hide this discrepancy.

## Deployment

The website is launched with `python -m streamlit run website/app.py`. Ensure the repository includes `models/cnn_best.pt`, `models/svm_tuned.joblib`, `models/rf_tuned.joblib`, `data/labels.csv`, and the evaluation files read from `outputs/cnn/` and `outputs/classical/`. Raw WAV files and large training arrays are excluded by `.gitignore`; a WAV can be uploaded directly to the running site.

Live site URL: **[fill in after deployment]**

## Team

- **Member 1 — Classical ML:** feature engineering and SVM / Random Forest pipeline.
- **Member 2 — CNN:** log-Mel preprocessing, CNN training, and evaluation.
- **Website:** Streamlit pages, model loading, preprocessing integration, and prediction interface.
