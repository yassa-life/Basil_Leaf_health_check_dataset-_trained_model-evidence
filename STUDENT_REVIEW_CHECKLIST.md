# Student review and demonstration checklist

Use this checklist before the model review. Each student must be able to explain and demonstrate at least one complete model experiment.

## Required individual contribution

- [ ] Student name: ____________________
- [ ] Model demonstrated: ____________________
- [ ] The model is one of the supported keys: `logistic_regression`, `svm`, `knn`, `decision_tree`, `random_forest`, or `gradient_boosting`.
- [ ] The student can explain why the selected model is suitable and describe its main hyperparameters.
- [ ] The student can show the relevant training code in `parts/training.py` and the model-specific notebook in `parts/<model>/`.

## Preprocessing and tuning evidence

- [ ] Explain the shared RGB/HSV/LBP/HOG feature extraction and the grouped train/holdout split.
- [ ] Show that preprocessing is fitted inside each cross-validation fold (`StandardScaler` and `SelectPercentile`).
- [ ] Show the comparison between the 25% and 100% feature-selection variants in `parts/<model>/outputs/preprocessing_comparison.csv`.
- [ ] Show the model's hyperparameter search space in `parts/<model>/outputs/search_space.json`.
- [ ] Show the full candidate results and fold scores in `parts/<model>/outputs/tuning_results.csv`.
- [ ] Explain why the selected setting was chosen using grouped validation macro-F1, without using holdout labels.

## Output demonstration

During the review, demonstrate all of the following for the selected model:

1. Run a model-specific training command:

   ```powershell
   .venv/Scripts/python.exe _run_all_training.py --model <model-key>
   ```

2. Open and explain the generated files:

   - `parts/<model-key>/outputs/<model-key>_metrics.json`
   - `parts/<model-key>/outputs/tuning_results.csv`
   - `parts/<model-key>/outputs/preprocessing_comparison.csv`
   - `parts/<model-key>/outputs/evaluation.png`
   - `parts/<model-key>/outputs/test_predictions.csv`

3. Show one prediction using the saved model:

   ```powershell
   .venv/Scripts/python.exe predict.py "path/to/leaf.jpg" --model <model-key>
   ```

4. Point out the validation macro-F1, holdout accuracy, macro-F1, confusion matrix, and at least one limitation of the result.

## Evidence record

Record the files and output shown during the review so individual work is clear:

- Demonstrated code/notebook section: ____________________
- Preprocessing comparison explained: ____________________
- Hyperparameters explained: ____________________
- Output files shown: ____________________
- Prediction command/output shown: ____________________
- Student reflection or limitation: ____________________

The saved artifacts are evidence of the experiment, but every student must understand and present their own selected section. No individual authorship is inferred automatically from a folder assignment.
