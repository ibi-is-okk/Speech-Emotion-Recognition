# Speech Emotion Recognition - results summary

* Split: **speaker** | test clips: 300 | test actors: [1, 9, 12, 17, 19]
* Features: 114 (MFCC, chroma, spectral centroid/bandwidth/rolloff, ZCR, RMS)

## Model comparison (test set)

```
       model  accuracy  precision_macro  recall_macro  f1_macro  f1_weighted  cv_f1_macro
SVM_baseline     0.423            0.441         0.428     0.421        0.427          NaN
 RF_baseline     0.417            0.423         0.419     0.407        0.410          NaN
   SVM_tuned     0.430            0.441         0.425     0.423        0.434        0.473
    RF_tuned     0.437            0.443         0.425     0.419        0.433        0.420
```

* **SVM_tuned** best params: `{'clf__C': 10, 'clf__gamma': 0.001, 'clf__kernel': 'rbf'}` (CV macro-F1 0.473)
* **RF_tuned** best params: `{'clf__max_depth': None, 'clf__max_features': 0.3, 'clf__min_samples_split': 5, 'clf__n_estimators': 500}` (CV macro-F1 0.420)

**Best tuned model: SVM_tuned** (accuracy 0.430, macro-F1 0.423).

## Most frequently confused emotions (true -> predicted)

**SVM_tuned**

* calm -> neutral: 10 clips (25% of true calm)
* happy -> sad: 9 clips (22% of true happy)
* surprised -> fearful: 9 clips (22% of true surprised)
* neutral -> calm: 8 clips (40% of true neutral)
* disgust -> sad: 8 clips (20% of true disgust)
* calm -> sad: 6 clips (15% of true calm)

**RF_tuned**

* angry -> disgust: 10 clips (25% of true angry)
* fearful -> happy: 10 clips (25% of true fearful)
* neutral -> calm: 9 clips (45% of true neutral)
* sad -> calm: 8 clips (20% of true sad)
* fearful -> sad: 8 clips (20% of true fearful)
* neutral -> sad: 6 clips (30% of true neutral)

## Random Forest feature importance

Top 10 individual features:

* `mfcc_6_std` - 0.0277
* `mfcc_3_mean` - 0.0257
* `mfcc_1_mean` - 0.0249
* `mfcc_2_std` - 0.0201
* `mfcc_4_std` - 0.0190
* `spectral_centroid_mean` - 0.0166
* `mfcc_20_std` - 0.0153
* `mfcc_21_std` - 0.0148
* `spectral_rolloff_mean` - 0.0144
* `mfcc_8_std` - 0.0143

By feature family:

* mfcc std: 0.394
* mfcc mean: 0.343
* chroma mean: 0.090
* chroma std: 0.067
* spectral_centroid mean: 0.017
* spectral_rolloff mean: 0.014
* spectral_bandwidth std: 0.013
* spectral_rolloff std: 0.013
* spectral_bandwidth mean: 0.011
* zcr mean: 0.011

## Notes for your write-up
* Compare per-class F1 in `classification_report_*.csv` to see which emotions are easy/hard.
* Acoustically similar emotions (e.g. neutral/calm, fearful/sad, angry/disgust, happy/surprised) are the usual confusion pairs - check the confusion matrices.
* Re-run with `--split random` to see how much speaker leakage inflates accuracy.