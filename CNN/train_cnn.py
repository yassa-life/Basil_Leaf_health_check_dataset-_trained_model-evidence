"""Train a CNN basil-leaf classifier and write analytics under CNN/outputs/."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parts"))
from _pipeline import CLASSES, SEED, audit_dataset, find_root, make_splits, read_rgb  # noqa: E402

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 16
LR = 1e-3
MODEL_NAME = "BasilLeafCNN"
FEATURE_VERSION = "cnn-rgb-128-v1"


class LeafDataset(Dataset):
    def __init__(self, frame, data_dir, indices, label_to_idx, augment=False):
        self.frame = frame.iloc[indices].reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.label_to_idx = label_to_idx
        self.augment = augment
        self.images = []
        for path in self.frame.path:
            image = read_rgb(self.data_dir / path).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
            arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
            self.images.append(torch.from_numpy(arr).permute(2, 0, 1))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        tensor = self.images[i].clone()
        if self.augment:
            if torch.rand(()) < 0.5: tensor = tensor.flip(2)
            if torch.rand(()) < 0.5: tensor = tensor.flip(1)
        return tensor, self.label_to_idx[row.label]



class BasilLeafCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4), nn.Linear(256 * 8 * 8, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(yb)
        correct += int((logits.argmax(1) == yb).sum().item())
        total += len(yb)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        preds = logits.argmax(1)
        total_loss += float(loss.item()) * len(yb)
        correct += int((preds == yb).sum().item())
        total += len(yb)
        all_preds.extend(preds.cpu().tolist())
        all_true.extend(yb.cpu().tolist())
    return total_loss / max(total, 1), correct / max(total, 1), np.asarray(all_true), np.asarray(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()
    if args.epochs < 1: parser.error('--epochs must be positive')
    torch.set_num_threads(2)
    set_seed(SEED)
    root = find_root(ROOT)
    cnn_dir = root / "CNN"
    out = cnn_dir / "outputs"
    analytics = cnn_dir / "analytics"
    results = cnn_dir / "results"
    for folder in (out, analytics, results):
        folder.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data" / "raw"
    print("Project root:", root)
    print("Outputs:", out)
    print("Analytics:", analytics)
    print("Results:", results)

    frame, audit = audit_dataset(data_dir)
    dev, test, cv = make_splits(frame)
    train_rel, val_rel = cv[0]
    train_idx = np.asarray(dev)[train_rel]
    val_idx = np.asarray(dev)[val_rel]
    test_idx = np.asarray(test)

    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    train_loader = DataLoader(LeafDataset(frame, data_dir, train_idx, label_to_idx, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(LeafDataset(frame, data_dir, val_idx, label_to_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(LeafDataset(frame, data_dir, test_idx, label_to_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    train_labels = frame.iloc[train_idx].label.map(label_to_idx).to_numpy()
    counts = np.bincount(train_labels, minlength=len(CLASSES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    configs = [dict(lr=1e-3, weight_decay=1e-4), dict(lr=3e-4, weight_decay=1e-3)]
    trials, all_history = [], []
    best_val_f1, best_state, history = -1.0, None, []
    start = time.perf_counter()
    for trial, config in enumerate(configs):
        set_seed(SEED)
        model = BasilLeafCNN(n_classes=len(CLASSES)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), **config)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        trial_best, stale, trial_state, trial_history = -1.0, 0, None, []
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, va_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
            va_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            scheduler.step(va_f1)
            row = dict(trial=trial, epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc, val_macro_f1=va_f1, lr=optimizer.param_groups[0]['lr'])
            trial_history.append(row); all_history.append(row)
            print(f'Trial {trial+1}/2 epoch {epoch}/{args.epochs}: validation F1={va_f1:.4f}', flush=True)
            if va_f1 > trial_best:
                trial_best, stale = va_f1, 0
                trial_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else: stale += 1
            if stale >= 4: break
        trials.append(dict(trial=trial, **config, best_val_macro_f1=trial_best, epochs_run=len(trial_history)))
        if trial_best > best_val_f1:
            best_val_f1, best_state, history, best_config = trial_best, trial_state, trial_history, config
    pd.DataFrame(trials).to_csv(results / 'tuning_results.csv', index=False)
    pd.DataFrame(all_history).to_csv(analytics / 'all_trials_history.csv', index=False)
    fit_time = time.perf_counter() - start
    model.load_state_dict(best_state)

    te_loss, te_acc, y_te, y_hat = evaluate(model, test_loader, criterion, device)
    y_te_lbl = np.array([idx_to_label[i] for i in y_te])
    y_hat_lbl = np.array([idx_to_label[i] for i in y_hat])
    p, r, f1, _ = precision_recall_fscore_support(y_te_lbl, y_hat_lbl, average="macro", zero_division=0, labels=CLASSES)
    cm = confusion_matrix(y_te_lbl, y_hat_lbl, labels=CLASSES)
    report = classification_report(y_te_lbl, y_hat_lbl, labels=CLASSES, output_dict=True, zero_division=0)

    rng = np.random.default_rng(SEED)
    groups = frame.iloc[test_idx].split_group.to_numpy()
    unique_groups = np.unique(groups)
    boot = []
    n = len(y_te_lbl)
    for _ in range(1000):
        sampled = rng.choice(unique_groups, len(unique_groups), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        boot.append(f1_score(y_te_lbl[idx], y_hat_lbl[idx], average="macro", zero_division=0))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    torch.save({"model_state_dict": model.state_dict(), "classes": CLASSES, "img_size": IMG_SIZE, "feature_version": FEATURE_VERSION, "architecture": MODEL_NAME, "label_to_idx": label_to_idx}, out / "cnn_model.pt")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(analytics / "training_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hist_df.epoch, hist_df.train_loss, label="train"); axes[0].plot(hist_df.epoch, hist_df.val_loss, label="val"); axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(hist_df.epoch, hist_df.train_acc, label="train acc"); axes[1].plot(hist_df.epoch, hist_df.val_acc, label="val acc"); axes[1].plot(hist_df.epoch, hist_df.val_macro_f1, label="val macro-F1"); axes[1].set_title("Accuracy / F1"); axes[1].legend()
    fig.tight_layout(); fig.savefig(analytics / "training_curves.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("CNN Confusion Matrix (holdout)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout(); fig.savefig(analytics / "confusion_matrix.png", dpi=140); plt.close(fig)

    test_frame = frame.iloc[test_idx].copy().reset_index(drop=True)
    test_frame["y_true"] = y_te_lbl; test_frame["y_pred"] = y_hat_lbl; test_frame["correct"] = test_frame.y_true == test_frame.y_pred
    test_frame.to_csv(results / "test_predictions.csv", index=False)

    split = np.full(len(frame), "development", dtype=object); split[test_idx] = "test"
    role = np.array(["unused"] * len(frame), dtype=object); role[train_idx] = "train"; role[val_idx] = "val"; role[test_idx] = "test"
    manifest = frame.copy(); manifest["split"] = split; manifest["role"] = role
    manifest.to_csv(analytics / "dataset_manifest.csv", index=False)

    per_class = {c: {"precision": float(report[c]["precision"]), "recall": float(report[c]["recall"]), "f1": float(report[c]["f1-score"]), "support": int(report[c]["support"])} for c in CLASSES}
    (results / "class_performance.json").write_text(json.dumps(per_class, indent=2), encoding="utf-8")

    metrics = {
        "model_name": MODEL_NAME, "method": "CNN", "pipeline_stage": "Deep Learning Image Classification",
        "feature_version": FEATURE_VERSION, "img_size": IMG_SIZE, "epochs": len(history), "max_epochs": args.epochs, "early_stopping_patience": 4, "tuning_trials": trials, "augmentation": "training-only horizontal and vertical flips", "batch_size": BATCH_SIZE,
        "learning_rate": best_config["lr"], "weight_decay": best_config["weight_decay"], "optimizer": "Adam", "best_val_macro_f1": best_val_f1,
        "accuracy": float(accuracy_score(y_te_lbl, y_hat_lbl)), "macro_f1": float(f1), "precision": float(p),
        "recall": float(r), "test_loss": float(te_loss), "bootstrap_95_ci": [float(ci_lo), float(ci_hi)],
        "fit_time_seconds": float(fit_time), "confusion_matrix": cm.tolist(),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
        "classes": CLASSES, "class_weights": {CLASSES[i]: float(weights[i]) for i in range(len(CLASSES))},
        "audit_summary": {"valid_unique_count": audit["valid_unique_count"], "class_counts": audit["class_counts"], "split_groups": audit["split_groups"]},
        "created_utc": datetime.now(timezone.utc).isoformat(), "torch_version": torch.__version__, "device": str(device),
    }
    (results / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary = "\n".join(
        [
            "# CNN Holdout Results",
            "",
            f"- **Model**: {MODEL_NAME}",
            f"- **Accuracy**: {metrics['accuracy']:.4f}",
            f"- **Macro F1**: {metrics['macro_f1']:.4f}",
            f"- **Precision**: {metrics['precision']:.4f}",
            f"- **Recall**: {metrics['recall']:.4f}",
            f"- **95% group-bootstrap CI (macro F1)**: [{ci_lo:.4f}, {ci_hi:.4f}]",
            f"- **Best val macro-F1**: {best_val_f1:.4f}",
            f"- **Fit time**: {fit_time:.1f}s on {device}",
            f"- **Split**: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
            f"- **Confusion matrix** (Healthy / Unhealthy): {cm.tolist()}",
            "",
            "See `class_performance.json`, `test_predictions.csv`, and `tuning_results.csv` in this folder.",
        ]
    )
    (results / "RESULTS.md").write_text(summary + "\n", encoding="utf-8")

    print("\n=== Holdout results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
    print(f"95% group-bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print("Confusion matrix:\n", cm)
    print("Saved model to:", out)
    print("Saved analytics to:", analytics)
    print("Saved results to:", results)


if __name__ == "__main__":
    main()
