# CNN Training Analytics

## Architecture

Custom 4-block CNN (`BasilLeafCNN`) on 128×128 RGB images:

1. Conv 3→32 + BatchNorm + ReLU + MaxPool
2. Conv 32→64 + BatchNorm + ReLU + MaxPool
3. Conv 64→128 + BatchNorm + ReLU + MaxPool
4. Conv 128→256 + BatchNorm + ReLU + MaxPool (feature map 8×8)
5. Dropout 0.4 → Linear 16384→256 → ReLU → Dropout 0.3 → Linear 256→2

Optimizer: Adam (lr=1e-3, weight decay=1e-4), class-weighted CrossEntropyLoss, ReduceLROnPlateau on validation macro-F1. Best checkpoint (epoch 12) is restored before the holdout evaluation.

## Training curves

See `training_curves.png`.

- Train loss falls from 1.76 (epoch 1) to ~0.007 by epoch 20.
- Validation macro-F1 peaks at **0.9898** in epoch 12, then plateaus (~0.983).
- Learning rate is halved three times (0.001 → 0.0005 → 0.00025 → 0.000125) when validation F1 stops improving.

## Dataset roles

See `dataset_manifest.csv`.

| Role | Count |
| :--- | ---: |
| train | 599 |
| val | 300 |
| test | 225 |

Grouped holdout from `parts/_pipeline.py` — near-duplicate capture groups stay on one side of the split.

## Files

- `training_history.csv` — per-epoch loss / accuracy / F1 / learning rate
- `training_curves.png` — loss and accuracy plots
- `confusion_matrix.png` — holdout confusion matrix
- `dataset_manifest.csv` — every audited image with split/role labels
