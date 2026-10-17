# Speech Emotion Recognition - results summary

* Split: **speaker** | test clips: 300 | test actors: [1, 9, 12, 17, 19]
* Features: 376 | features file `features_v2.csv` | speaker normalisation: True | augmented rows used for training: 3420

## Model comparison (test set)

```
       model  accuracy  precision_macro  recall_macro  f1_macro  f1_weighted  cv_f1_macro
SVM_baseline     0.607            0.616         0.612     0.602        0.610          NaN
 RF_baseline     0.550            0.572         0.569     0.543        0.545          NaN
   SVM_tuned     0.613            0.624         0.616     0.606        0.617        0.616
    RF_tuned     0.567            0.601         0.588     0.561        0.568        0.573
```

* **SVM_tuned** best params: `{'clf__C': 10, 'clf__gamma': 0.003, 'clf__kernel': 'rbf'}` (CV macro-F1 0.616)
* **RF_tuned** best params: `{'clf__max_depth': None, 'clf__max_features': 0.3, 'clf__min_samples_split': 5, 'clf__n_estimators': 500}` (CV macro-F1 0.573)

**Best tuned model: SVM_tuned** (accuracy 0.613, macro-F1 0.606).

## Most frequently confused emotions (true -> predicted)

**SVM_tuned**

* calm -> sad: 10 clips (25% of true calm)
* calm -> neutral: 8 clips (20% of true calm)
* disgust -> angry: 6 clips (15% of true disgust)
* surprised -> neutral: 6 clips (15% of true surprised)
* neutral -> sad: 5 clips (25% of true neutral)
* happy -> angry: 5 clips (12% of true happy)

**RF_tuned**

* calm -> neutral: 15 clips (38% of true calm)
* sad -> neutral: 11 clips (28% of true sad)
* angry -> disgust: 9 clips (22% of true angry)
* fearful -> sad: 8 clips (20% of true fearful)
* surprised -> neutral: 6 clips (15% of true surprised)
* surprised -> disgust: 6 clips (15% of true surprised)

## Random Forest feature importance

Top 10 individual features:

* `duration` - 0.0486
* `f0_p50` - 0.0363
* `f0_mean` - 0.0280
* `voiced_ratio` - 0.0221
* `contrast_3_mean` - 0.0210
* `f0_p90` - 0.0162
* `f0_absdiff` - 0.0161
* `f0_p10` - 0.0157
* `mfcc_5_mean` - 0.0133
* `f0_std` - 0.0121

By feature family:

* mel mean: 0.122
* mfcc mean: 0.103
* mfcc_delta mean: 0.086
* mfcc_delta std: 0.080
* mfcc_delta2 std: 0.080
* mfcc std: 0.076
* mfcc_delta2 mean: 0.072
* mel std: 0.065
* duration: 0.049
* contrast mean: 0.037

## Notes for your write-up
* Compare per-class F1 in `classification_report_*.csv` to see which emotions are easy/hard.
* Acoustically similar emotions (e.g. neutral/calm, fearful/sad, angry/disgust, happy/surprised) are the usual confusion pairs - check the confusion matrices.
* Re-run with `--split random` to see how much speaker leakage inflates accuracy.