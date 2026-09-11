# Six-Person Model & Pipeline Assignment

Each of the six group members is assigned **one machine learning model** to train and evaluate end-to-end, while also owning **one stage of the shared ML data pipeline**.

The complete integrated baseline notebook remains at `../Basil_Leaf_ML_Workflow.ipynb`.

## 📁 Model Directories & Notebook Structure

Every team member has a dedicated folder containing their Python notebook (named in `(modelname_which pipeline stage)` format) and their model's output directory (`outputs/`):

| Member | Folder | Notebook File `(modelname_which pipeline stage)` | Assigned ML Model | Pipeline Stage Responsibility | Primary Output Artifacts |
|---|---|---|---|---|---|
| **Person 1** | `parts/logistic_regression/` | [`logistic_regression_data_collection.ipynb`](file:///d:/SLIIT/projectr/Dataset_Train/parts/logistic_regression/logistic_regression_data_collection.ipynb) | **Logistic Regression** | Data Collection & Inventory Audit | `logistic_regression_model.joblib`<br>`logistic_regression_metrics.json` |
| **Person 2** | `parts/svm/` | [`svm_data_preprocessing.ipynb`](file:///d:/SLIIT/projectr/Dataset_Train/parts/svm/svm_data_preprocessing.ipynb) | **Support Vector Machine (SVM)** | Data Preprocessing & Leakage-Safe Split | `svm_model.joblib`<br>`svm_metrics.json` |
| **Person 3** | `parts/knn/` | [`knn_feature_engineering.ipynb`](file:///d:/SLIIT/projectr/Dataset_Train/parts/knn/knn_feature_engineering.ipynb) | **K-Nearest Neighbors (KNN)** | Feature Engineering (RGB/HSV/LBP/HOG) | `knn_model.joblib`<br>`knn_metrics.json` |
| **Person 4** | `parts/decision_tree/` | [`decision_tree_model_selection.ipynb`](file:///d:/SLIIT/projectr/Dataset_Train/parts/decision_tree/decision_tree_model_selection.ipynb) | **Decision Tree Classifier** | Cross-Validation & Model Selection | `decision_tree_model.joblib`<br>`decision_tree_metrics.json` |
| **Person 5** | `parts/random_forest/` | [`random_forest_training_evaluation.ipynb`](file:///d:/SLIIT/projectr/Dataset_Train/parts/random_forest/random_forest_training_evaluation.ipynb) | **Random Forest (Selected Winner)** | Model Fitting & Bootstrap 95% CI Evaluation | `random_forest_model.joblib`<br>`random_forest_metrics.json` |
| **Person 6** | `parts/gradient_boosting/` | [`gradient_boosting_deployment_reporting.ipynb`](file:///d:/SLIIT/projectr/Dataset_Train/parts/gradient_boosting/gradient_boosting_deployment_reporting.ipynb) | **Gradient Boosting Classifier** | Web App Deployment & Benchmarking Report | `gradient_boosting_model.joblib`<br>`gradient_boosting_metrics.json`<br>`model_comparison_6_members.csv` |

## 📐 Shared Guidelines & Rules

- **Shared Virtual Environment:** Use the existing project kernel (`.venv`).
- **Shared Data & Features:** All notebooks read the clean dataset manifest (`parts/artifacts/02_clean_manifest.csv`) and extracted feature matrix (`parts/artifacts/03_features.npz`).
- **Fixed Holdout Test Set:** Do not alter the fixed test set (180 holdout images).
- **Independent Model Evaluation:** Each member evaluates their assigned model and saves metrics (`Accuracy`, `Macro F1`, `Precision`, `Recall`, `Confusion Matrix`) to their folder's `outputs/` directory.
