# Six individual end-to-end models

Each existing notebook now performs data audit, EDA, feature extraction, fold-local preprocessing, model-specific tuning, comparison of parameter varieties, final evaluation, export and reload/inference. Run from the project .venv. See the root README for commands.

| Folder | Model | Tuned settings |
|---|---|---|
| logistic_regression | Logistic Regression | C, feature percentage |
| svm | RBF SVM | C, gamma, feature percentage |
| knn | KNN | neighbors, weights, feature percentage |
| decision_tree | Decision Tree | depth, leaf size, feature percentage |
| random_forest | Random Forest | leaf size, feature sampling, feature percentage |
| gradient_boosting | Gradient Boosting | learning rate, depth, feature percentage |

`training.py` defines all searches; `_pipeline.py` provides identical data and features; `evaluation.py` computes metrics and exports the validation-selected deployment artifact. Model bundles contain `estimator`, class order, feature version and dataset fingerprint. Do not load untrusted serialized files.

CNN remains outside these six folders and outside the comparison. The preserved historical progress notebooks are not final-training inputs. A notebook filename's old stage name is historical; its content now covers the entire pipeline.

Every model output folder contains all CV candidate/fold scores, best parameters, preprocessing comparison, test predictions, confusion/tuning plot, audit and split manifest. The main comparison is `outputs/six_model_comparison.csv` at project root.
