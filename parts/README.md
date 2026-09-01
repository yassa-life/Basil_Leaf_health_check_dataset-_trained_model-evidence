# Six-person notebook assignment

Run the notebooks in numerical order. The complete integrated notebook remains at `../Basil_Leaf_ML_Workflow.ipynb`; these files divide ownership without deleting it.

| Person | Notebook | Pipeline responsibility | Main handoff |
|---|---|---|---|
| Person 1 | `person_1_data_collection.ipynb` | Source, labels, local download inventory, missing coverage | Inventory and collection report |
| Person 2 | `person_2_data_preprocessing.ipynb` | Decode, orientation, validation, duplicate audit, similarity groups, train/test split | Clean and split manifests |
| Person 3 | `person_3_feature_engineering.ipynb` | RGB/HSV, LBP and HOG feature implementation and extraction | 1,882-feature matrix |
| Person 4 | `person_4_model_selection.ipynb` | Grouped 3-fold comparison, macro-F1 selection, comparison table | Winner and validation evidence |
| Person 5 | `person_5_training_evaluation.ipynb` | Final fit, held-out evaluation, bootstrap interval, error predictions | Saved model and test metrics |
| Person 6 | `person_6_deployment_reporting.ipynb` | Matching inference, deployment verification, final report and limitations | Deployment-ready report |

## Equal-work rule

Each person owns four comparable deliverables: implementation, explanation, validation evidence, and a documented handoff. Model computation differs by stage, so equality is based on responsibility rather than identical line counts. Every person must explain their decisions and verify their output before handing it forward.

## Shared rules

- Use the existing project `.venv` kernel.
- Do not change the fixed test set after Person 2 creates it.
- Do not use the test score to select a model.
- Do not edit another person's artifact silently; rerun from the changed stage onward.
- Files made by these notebooks stay in `parts/artifacts/` so they do not overwrite the integrated workflow.
- The dataset remains a partial download until all listed folders and files are present.
