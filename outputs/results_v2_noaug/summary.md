# Speech Emotion Recognition - results summary

* Split: **speaker** | test clips: 300 | test actors: [1, 9, 12, 17, 19]
* Features: 376 | features file `features_v2.csv` | speaker normalisation: False | augmented rows used for training: 0

## Model comparison (test set)

```
       model  accuracy  precision_macro  recall_macro  f1_macro  f1_weighted  cv_f1_macro
SVM_baseline     0.520            0.552         0.525     0.518        0.521          NaN
 RF_baseline     0.437            0.444         0.438     0.417        0.418          NaN
   SVM_tuned     0.517            0.563         0.516     0.517        0.524        0.564
    RF_tuned     0.507            0.539         0.503     0.495        0.500        0.508
```

* **SVM_tuned** best params: `{'clf__C': 10, 'clf__gamma': 0.0003, 'clf__kernel': 'rbf'}` (CV macro-F1 0.564)
* **RF_tuned** best params: `{'clf__max_depth': None, 'clf__max_features': 0.3, 'clf__min_samples_split': 5, 'clf__n_estimators': 500}` (CV macro-F1 0.508)

**Best tuned model: SVM_tuned** (accuracy 0.517, macro-F1 0.517).

## Most frequently confused emotions (true -> predicted)

**SVM_tuned**

* disgust -> sad: 14 clips (35% of true disgust)
* sad -> calm: 9 clips (22% of true sad)
* surprised -> sad: 9 clips (22% of true surprised)
* happy -> sad: 8 clips (20% of true happy)
* surprised -> fearful: 8 clips (20% of true surprised)
* fearful -> sad: 7 clips (18% of true fearful)

**RF_tuned**

* surprised -> sad: 11 clips (28% of true surprised)
* sad -> calm: 10 clips (25% of true sad)
* happy -> sad: 9 clips (22% of true happy)
* fearful -> sad: 9 clips (22% of true fearful)
* fearful -> happy: 8 clips (20% of true fearful)
* neutral -> calm: 6 clips (30% of true neutral)

## Random Forest feature importance

Top 10 individual features:

* `f0_absdiff` - 0.0257
* `duration` - 0.0183
* `mfcc_delta2_1_std` - 0.0136
* `mfcc_delta_4_mean` - 0.0129
* `mel_22_std` - 0.0125
* `contrast_3_mean` - 0.0113
* `voiced_ratio` - 0.0100
* `f0_p50` - 0.0092
* `mfcc_delta_6_std` - 0.0086
* `mel_5_mean` - 0.0086

By feature family:

* mel mean: 0.118
* mel std: 0.117
* mfcc_delta2 std: 0.114
* mfcc std: 0.097
* mfcc mean: 0.096
* mfcc_delta mean: 0.090
* mfcc_delta std: 0.088
* mfcc_delta2 mean: 0.067
* contrast mean: 0.036
* f0_absdiff: 0.026

## Notes for your write-up
* Compare per-class F1 in `classification_report_*.csv` to see which emotions are easy/hard.
* Acoustically similar emotions (e.g. neutral/calm, fearful/sad, angry/disgust, happy/surprised) are the usual confusion pairs - check the confusion matrices.
* Re-run with `--split random` to see how much speaker leakage inflates accuracy.