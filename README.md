# Group abc - Basil leaf final evaluation

The final submission contains exactly six classical models: Logistic Regression, SVM, KNN, Decision Tree, Random Forest and Gradient Boosting. CNN is a separate supplementary experiment. The assignment PDFs in `final/` and all historical `progress/` work are preserved.

## Run from zero

Use Python 3.13 (the measured environment) or a compatible Python version with the pinned dependencies. Put the assigned images into these four folders under `data/raw/`:

- `Amravati_Region_Basil_Plant_Healthy`
- `Nagpur_Region_Basil_Plant_Healthy`
- `Pune_Region_Basil_Plant_Healthy`
- `Basil_Plant_Unhealthy`

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe _run_all_training.py
.venv/Scripts/python.exe CNN/train_cnn.py
.venv/Scripts/python.exe build_report.py --group-id abc
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe app.py
```

Torch in the measured environment is CPU-only. The two CNN trials take longer than the classical search. `--epochs N` controls the maximum per trial (default 16); patience-4 early stopping can finish earlier. Reports read completed training outputs; they do not retrain models.

The report files are `output/pdf/abc.pdf` (six-model group report, <=15 pages) and `output/pdf/abc_CNN.pdf` (separate CNN supplement). The main PDF title uses the supplied group ID, `abc`.

## Each member's complete pipeline

Open the existing notebook in their corresponding `parts/<model>/` folder using the .venv kernel and run all cells. Every notebook now covers the full pipeline, not only one assigned stage:

1. Problem and model suitability.
2. Raw-image audit, duplicate removal, EDA, grouped split and feature extraction.
3. Fold-local scaling and feature selection, model-specific hyperparameters.
4. GridSearchCV over two preprocessing variants for every parameter combination.
5. Validation comparison, refit, holdout evaluation, errors and visualizations.
6. Save, reload and predict a leaf image.
7. Conclusions, limitations and viva discussion.

To run one model from the project root:

```powershell
.venv/Scripts/python.exe _run_all_training.py --model svm
.venv/Scripts/python.exe predict.py "path/to/leaf.jpg" --model svm
```

Model keys: `logistic_regression`, `svm`, `knn`, `decision_tree`, `random_forest`, `gradient_boosting`. Inference also accepts `selected` (default) and `cnn`. Individual runs update only that member's artifacts; rerun all six before regenerating the group comparison/deployment/report.

## Fair comparison and tuning

The shared audit retains 1,124 unique images from 1,131 listed files, removing seven exact pixel duplicates. Similar images are grouped by perceptual hash. Seed 42 creates 899 development / 225 holdout images and three development folds with no shared groups. Actual plant identities are unavailable.

All classical models use 1,882 RGB/HSV/LBP/HOG features. A Pipeline fits StandardScaler and ANOVA SelectPercentile inside each training fold. Both 100% and 25% feature variants are searched. Logistic Regression tunes C; SVM tunes C/gamma; KNN tunes neighbors/weights; Decision Tree tunes depth/leaf size; Random Forest tunes leaf size/feature sampling; Gradient Boosting tunes learning rate/depth. Candidate counts and every fold score are saved. Selection uses mean development macro-F1, with accuracy and balanced accuracy also recorded. Search scores are not unbiased nested-CV estimates.

The holdout was evaluated in earlier project runs. It is excluded from current tuning but is a **reused holdout**, not new external validation. No improvement guarantee is made. Compare current measured scores in `outputs/six_model_comparison.csv`.

CNN uses the same audited holdout but a 599/300 train/validation development split, training-only flips, two Adam configurations, class-weighted loss, learning-rate reduction and validation-based checkpoint selection. Its training budget and validation procedure differ, so it stays outside the six-model ranking.

## Outputs and project scope

- `parts/training.py`: single source of truth for all six model searches.
- `parts/_pipeline.py`: data audit, split and handcrafted features, also used by the web app.
- `parts/evaluation.py`: metrics, group-bootstrap intervals and selected deployment export.
- `parts/<model>/outputs/`: complete tuning table, preprocessing comparison, search space, saved pipeline bundle, metrics, predictions, split manifest, audit and plot.
- `outputs/`: selected web model, six-model comparison, split/audit/environment evidence, figures and cleanup audit.
- `CNN/`: standalone CNN notebook, training implementation, checkpoint and evidence.
- `final/`: the two original assignment PDFs.
- `progress/`: preserved historical preprocessing work; never used as globally fitted final-training input.
- `Basil_Leaf_ML_Workflow.ipynb`: orchestration notebook for the same six-model workflow.
- `app.py`, `templates/`, `static/`, `start_web.bat`: local selected-model demonstration, at http://127.0.0.1:8001.

Old conflicting mixed-model comparison files are removed. Source images with unique content are not discarded just because they are unused; see cleanup notes for duplicate verification.

## Limitations and AI declaration

Whole-image features may capture background, region or camera conditions. Unhealthy region metadata and plant IDs are unavailable. No external validation, calibrated confidence, disease diagnosis or reliable non-leaf rejection is provided. Images uploaded to the web app are handled in memory and not used for training. Dataset attribution: IEEE DataPort DOI `10.21227/a4f6-4413`; follow original dataset terms.

Codex assisted with code, notebooks, tuning orchestration, tests, figures and report drafting. Metrics are computed from real runs. Students must review the work, declare AI assistance and demonstrate their own understanding and contribution; no individual authorship is inferred from a folder assignment.
