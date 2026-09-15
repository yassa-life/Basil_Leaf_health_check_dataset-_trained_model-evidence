# CNN Basil Leaf Classifier

Convolutional Neural Network for basil leaf health classification (`Healthy` vs `Unhealthy`), using the same labelled folders under `data/raw/` as the classical ML notebooks.

## Layout

```
CNN/
├── CNN_Basil_Leaf_Training.ipynb   # Jupyter notebook (run this)
├── train_cnn.py                    # Same pipeline as a CLI script
├── README.md
├── outputs/                        # Trained model
│   └── cnn_model.pt
├── analytics/                      # Training plots, history, dataset roles
│   ├── ANALYTICS.md
│   ├── training_history.csv
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── dataset_manifest.csv
└── results/                        # Holdout metrics, predictions, comparison
    ├── RESULTS.md
    ├── cnn_metrics.json
    ├── class_performance.json
    ├── test_predictions.csv
    └── model_comparison_with_cnn.csv
```

## Holdout results (already trained)

| Metric | Score |
| :--- | ---: |
| Accuracy | **98.67%** |
| Macro F1 | **0.9864** |
| 95% CI | [0.9686, 1.0000] |
| Errors | 3 / 225 (128+94 correct) |

CNN outperforms the six classical models on the same grouped holdout split.

## Run

From the project root with `.venv` activated:

```powershell
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
python CNN/train_cnn.py
```

Or open `CNN/CNN_Basil_Leaf_Training.ipynb` and run all cells. If `outputs/cnn_model.pt` already exists, the notebook loads the saved analytics/results instead of retraining (~11 minutes on CPU).

## Method

- Shared audit / grouped holdout split from `parts/_pipeline.py` (no leakage across near-duplicate groups)
- Images resized to 128×128 RGB, normalized to [-1, 1]
- Custom CNN: 4× (Conv2d + BatchNorm + ReLU + MaxPool) then Dropout + Linear head
- 20 epochs, Adam, class-weighted CrossEntropyLoss
- Best validation macro-F1 checkpoint evaluated on the holdout set
