# Speech Emotion Recognition - results summary

* Split: **speaker** | test clips: 300 | test actors: [1, 9, 12, 17, 19]
* Features: 376 | features file `features_v2.csv` | speaker normalisation: True | augmented rows used for training: 0

## Model comparison (test set)

```
       model  accuracy  precision_macro  recall_macro  f1_macro  f1_weighted  cv_f1_macro
SVM_baseline     0.590            0.598         0.603     0.587        0.593          NaN
 RF_baseline     0.527            0.525         0.538     0.518        0.522          NaN
   SVM_tuned     0.590            0.598         0.603     0.587        0.593        0.633
    RF_tuned     0.553            0.560         0.569     0.547        0.552        0.599
```

* **SVM_tuned** best params: `{'clf__C': 1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}` (CV macro-F1 0.633)
* **RF_tuned** best params: `{'clf__max_depth': None, 'clf__max_features': 'sqrt', 'clf__min_samples_split': 5, 'clf__n_estimators': 500}` (CV macro-F1 0.599)

**Best tuned model: SVM_tuned** (accuracy 0.590, macro-F1 0.587).

## Most frequently confused emotions (true -> predicted)

**SVM_tuned**

* calm -> neutral: 8 clips (20% of true calm)
* surprised -> neutral: 8 clips (20% of true surprised)
* sad -> happy: 6 clips (15% of true sad)
* angry -> disgust: 6 clips (15% of true angry)
* happy -> surprised: 5 clips (12% of true happy)
* sad -> neutral: 5 clips (12% of true sad)

**RF_tuned**

* angry -> disgust: 10 clips (25% of true angry)
* disgust -> angry: 10 clips (25% of true disgust)
* calm -> neutral: 9 clips (22% of true calm)
* fearful -> sad: 8 clips (20% of true fearful)
* sad -> neutral: 7 clips (18% of true sad)
* sad -> calm: 7 clips (18% of true sad)

## Random Forest feature importance

Top 10 individual features:

* `duration` - 0.0142
* `f0_p50` - 0.0141
* `f0_mean` - 0.0133
* `mfcc_5_mean` - 0.0118
* `mfcc_3_mean` - 0.0105
* `mfcc_2_mean` - 0.0105
* `voiced_ratio` - 0.0103
* `f0_absdiff` - 0.0098
* `f0_p10` - 0.0095
* `spectral_centroid_mean` - 0.0095

By feature family:

* mel mean: 0.130
* mfcc mean: 0.116
* mfcc_delta2 std: 0.110
* mel std: 0.102
* mfcc_delta std: 0.092
* mfcc std: 0.090
* mfcc_delta mean: 0.076
* mfcc_delta2 mean: 0.066
* contrast mean: 0.031
* chroma mean: 0.024

## Notes for your write-up
* Compare per-class F1 in `classification_report_*.csv` to see which emotions are easy/hard.
* Acoustically similar emotions (e.g. neutral/calm, fearful/sad, angry/disgust, happy/surprised) are the usual confusion pairs - check the confusion matrices.
* Re-run with `--split random` to see how much speaker leakage inflates accuracy.