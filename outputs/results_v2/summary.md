# Speech Emotion Recognition - results summary

* Split: **speaker** | test clips: 300 | test actors: [1, 9, 12, 17, 19]
* Features: 376 | features file `features_v2.csv` | speaker normalisation: False | augmented rows used for training: 3420

## Model comparison (test set)

```
       model  accuracy  precision_macro  recall_macro  f1_macro  f1_weighted  cv_f1_macro
SVM_baseline     0.553            0.594         0.556     0.554        0.555          NaN
 RF_baseline     0.463            0.463         0.462     0.443        0.447          NaN
   SVM_tuned     0.553            0.594         0.556     0.554        0.555        0.571
    RF_tuned     0.477            0.498         0.491     0.473        0.473        0.485
```

* **SVM_tuned** best params: `{'clf__C': 1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}` (CV macro-F1 0.571)
* **RF_tuned** best params: `{'clf__max_depth': None, 'clf__max_features': 0.3, 'clf__min_samples_split': 5, 'clf__n_estimators': 500}` (CV macro-F1 0.485)

**Best tuned model: SVM_tuned** (accuracy 0.553, macro-F1 0.554).

## Most frequently confused emotions (true -> predicted)

**SVM_tuned**

* disgust -> sad: 12 clips (30% of true disgust)
* surprised -> sad: 10 clips (25% of true surprised)
* calm -> sad: 8 clips (20% of true calm)
* happy -> sad: 7 clips (18% of true happy)
* fearful -> happy: 7 clips (18% of true fearful)
* fearful -> sad: 7 clips (18% of true fearful)

**RF_tuned**

* fearful -> sad: 11 clips (28% of true fearful)
* sad -> neutral: 10 clips (25% of true sad)
* disgust -> sad: 9 clips (22% of true disgust)
* surprised -> sad: 7 clips (18% of true surprised)
* surprised -> disgust: 7 clips (18% of true surprised)
* happy -> sad: 6 clips (15% of true happy)

## Random Forest feature importance

Top 10 individual features:

* `duration` - 0.0291
* `f0_absdiff` - 0.0237
* `contrast_3_mean` - 0.0188
* `f0_mean` - 0.0159
* `mel_5_mean` - 0.0137
* `f0_p50` - 0.0130
* `voiced_ratio` - 0.0124
* `mfcc_delta_4_mean` - 0.0115
* `mfcc_delta_6_std` - 0.0105
* `f0_p90` - 0.0088

By feature family:

* mel mean: 0.130
* mfcc_delta mean: 0.103
* mfcc mean: 0.095
* mfcc_delta2 std: 0.091
* mfcc_delta std: 0.088
* mfcc std: 0.086
* mel std: 0.082
* mfcc_delta2 mean: 0.078
* contrast mean: 0.037
* chroma mean: 0.029

## Notes for your write-up
* Compare per-class F1 in `classification_report_*.csv` to see which emotions are easy/hard.
* Acoustically similar emotions (e.g. neutral/calm, fearful/sad, angry/disgust, happy/surprised) are the usual confusion pairs - check the confusion matrices.
* Re-run with `--split random` to see how much speaker leakage inflates accuracy.