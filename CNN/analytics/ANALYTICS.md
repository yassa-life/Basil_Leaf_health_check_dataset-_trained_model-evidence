# CNN analytics

`all_trials_history.csv` records every epoch in both hyperparameter trials. `training_history.csv` and `training_curves.png` show the winning trial (including epochs after its best checkpoint). Holdout predictions and the confusion matrix use the restored best checkpoint, not necessarily the last epoch.

`dataset_manifest.csv` records exact train/val/test roles. Training-only random flips do not apply to validation or test. See `../results/tuning_results.csv` for configuration selection and `../results/cnn_metrics.json` for measured outcomes. The holdout is reused from prior project work; plant-disjoint external testing is still needed.
