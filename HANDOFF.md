# Project handoff

## Current state

- The saved CNN run is a full run (`quick_dry_run: false` in `outputs/cnn/final_summary.json`).
- The non-speaker-normalized SVM and Random Forest models are staged in `models/`.
- `scripts/verify_classical.py` checks the saved models against `outputs/results_v2/model_comparison.csv` and exports the site metrics to `outputs/classical/`.
- The website source lives under `website/` and loads saved models/results; it does not train at startup.

## Before deployment

- Keep the non-speaker-normalized classical checkpoints for single-clip prediction.
- Run both verification scripts after the required audio, feature arrays, and evaluation files are present.
- Run notebooks `01`–`05`, `08`, and `09` locally; run CNN training/evaluation notebooks `06` and `07` in Colab with `QUICK = False`.
- Deploy with Python 3.11 using `python -m streamlit run website/app.py`.

## Verification notes from this workspace

- Both classical model metric rows reproduce `outputs/results_v2/model_comparison.csv`. The strict feature-reproduction subcheck in `verify_classical.py` still fails against the saved feature CSV; its tolerance was not changed.
- `verify_cnn_pipeline.py` passed after notebook `04` generated its arrays. The large ignored log-Mel caches were removed afterward; rerun notebook `04` before repeating that check.
- Notebooks `06` and `07` need a Colab GPU run for current execution outputs. Confirm `quick_dry_run` is `false`, then copy the executed notebooks and `cnn_outputs/` back into the repository.

See `README.md` for setup, notebook order, data locations, verification commands, current result-file sources, and deployment notes.
