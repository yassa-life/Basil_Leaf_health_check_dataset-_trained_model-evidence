# Six-Person Model Assignment

Each of the six group members trains **one machine learning model** end-to-end on the shared labelled dataset under `data/raw/`, and owns one stage of the pipeline narrative.

## Model directories

Every member folder has a notebook and an `outputs/` directory for that model only. There is **no shared `parts/artifacts/` folder** — each notebook loads images from `data/raw`, extracts RGB/HSV/LBP/HOG features, trains its model, and writes metrics/models into its own `outputs/`.

| Member | Folder | Notebook | Assigned ML Model | Pipeline Stage | Primary Outputs |
|---|---|---|---|---|---|
| Person 1 | `parts/logistic_regression/` | `logistic_regression_data_collection.ipynb` | Logistic Regression | Data Collection & Inventory | `outputs/logistic_regression_model.joblib`, `outputs/logistic_regression_metrics.json` |
| Person 2 | `parts/svm/` | `svm_data_preprocessing.ipynb` | SVM | Data Preprocessing & Split | `outputs/svm_model.joblib`, `outputs/svm_metrics.json` |
| Person 3 | `parts/knn/` | `knn_feature_engineering.ipynb` | KNN | Feature Engineering | `outputs/knn_model.joblib`, `outputs/knn_metrics.json` |
| Person 4 | `parts/decision_tree/` | `decision_tree_model_selection.ipynb` | Decision Tree | Model Selection | `outputs/decision_tree_model.joblib`, `outputs/decision_tree_metrics.json` |
| Person 5 | `parts/random_forest/` | `random_forest_training_evaluation.ipynb` | Random Forest | Training & Evaluation | `outputs/random_forest_model.joblib`, `outputs/random_forest_metrics.json` |
| Person 6 | `parts/gradient_boosting/` | `gradient_boosting_deployment_reporting.ipynb` | Gradient Boosting | Deployment Reporting | `outputs/gradient_boosting_model.joblib`, `outputs/gradient_boosting_metrics.json`, `outputs/model_comparison_6_members.csv` |

Shared helper: `parts/_pipeline.py` (dataset discovery, audit, split, handcrafted features).

A seventh **CNN** model lives outside `parts/` at `CNN/` (pixel CNN instead of handcrafted features). Notebook, model weights, analytics, and holdout results are kept under `CNN/outputs/`, `CNN/analytics/`, and `CNN/results/`.

## Rules

- Use the project `.venv` kernel.
- Only the four labelled folders under `data/raw/` are used (`*_Healthy` + `Basil_Plant_Unhealthy`). Legacy short folders (`Amravati`, `Nagpur`, `Pune`, `Bad`) are ignored.
- Each notebook writes only to its own `outputs/` folder.
- The production Flask app still loads the selected model from the root `outputs/model.joblib` produced by `Basil_Leaf_ML_Workflow.ipynb`.
