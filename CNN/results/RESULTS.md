# CNN Holdout Results

Trained on 15 September 2026 against the same grouped holdout split used by the classical models in `parts/`.

## Headline metrics

| Metric | Score |
| :--- | ---: |
| **Accuracy** | 98.67% |
| **Macro F1** | 0.9864 |
| **Precision (macro)** | 0.9870 |
| **Recall (macro)** | 0.9857 |
| **95% bootstrap CI (macro F1)** | [0.9686, 1.0000] |
| **Best validation macro-F1** | 0.9898 (epoch 12) |
| **Test loss** | 0.0489 |
| **Fit time** | 669.4 s (CPU, PyTorch 2.14.0) |

## Per-class performance

| Class | Precision | Recall | F1 | Support |
| :--- | ---: | ---: | ---: | ---: |
| Healthy | 0.9846 | 0.9922 | 0.9884 | 129 |
| Unhealthy | 0.9895 | 0.9792 | 0.9843 | 96 |

## Confusion matrix (holdout, n=225)

|  | Predicted Healthy | Predicted Unhealthy |
| :--- | ---: | ---: |
| **True Healthy** | 128 | 1 |
| **True Unhealthy** | 2 | 94 |

3 errors on 225 holdout images (1 healthy leaf called unhealthy, 2 unhealthy leaves called healthy).

## Split sizes

- Train: 599
- Validation: 300
- Test: 225
- Unique audited images: 1,124

## Comparison with classical models

CNN is the strongest model on this holdout set:

| Model | Stage | Accuracy | Macro F1 |
| :--- | :--- | ---: | ---: |
| Logistic Regression | Data Collection | 94.22% | 0.9410 |
| SVM (RBF) | Preprocessing | 97.78% | 0.9772 |
| KNN (k=1) | Feature Engineering | 96.89% | 0.9681 |
| Decision Tree | Model Selection | 96.89% | 0.9682 |
| Random Forest | Training & Evaluation | 98.22% | 0.9817 |
| Gradient Boosting | Deployment Reporting | 98.22% | 0.9818 |
| **BasilLeafCNN** | **CNN Deep Learning** | **98.67%** | **0.9864** |

Files in this folder: `cnn_metrics.json`, `class_performance.json`, `test_predictions.csv`, `model_comparison_with_cnn.csv`.
