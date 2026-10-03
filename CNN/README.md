# Separate CNN experiment

Run `python CNN/train_cnn.py` from the project environment, or run all cells in `CNN_Basil_Leaf_Training.ipynb`. Infer with `python predict.py IMAGE --model cnn`.

Uses 128x128 normalized RGB pixels; four Conv/BatchNorm/ReLU/pooling blocks (32/64/128/256 channels), dense 256 and two output logits, with dropout 0.4/0.3. The same audited classical holdout is retained. Development images are split into training and validation, and only training receives random horizontal/vertical flips.

Two Adam settings are evaluated: learning rate 0.001 with weight decay 0.0001, and learning rate 0.0003 with weight decay 0.001. Each has at most 16 epochs, class-weighted cross entropy, ReduceLROnPlateau and patience-4 early stopping. The best validation macro-F1 checkpoint across trials is restored before holdout evaluation. Use `--epochs N` to change the per-trial cap.

Artifacts: `outputs/cnn_model.pt`; `results/tuning_results.csv`, `cnn_metrics.json`, `class_performance.json`, `test_predictions.csv`, `RESULTS.md`; `analytics/all_trials_history.csv`, `training_history.csv`, `training_curves.png`, `confusion_matrix.png`, `dataset_manifest.csv`.

The 95% F1 interval resamples perceptual groups. This is a reused holdout and not new external validation. CNN has a different training budget and validation procedure from the six classical models. Its separate report is `output/pdf/abc_CNN.pdf`; it is excluded from the six-model comparison.
