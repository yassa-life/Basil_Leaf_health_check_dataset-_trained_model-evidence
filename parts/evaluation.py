from pathlib import Path
import time, json, hashlib, sys, importlib.metadata
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
from parts._pipeline import SEED, FEATURE_VERSION

def evaluate_selected(bundle, X, frame, test):
    y = frame.label.to_numpy()[test]
    start = time.perf_counter()
    pred = bundle["estimator"].predict(X[test])
    latency = (time.perf_counter() - start) * 1000 / len(test)
    classes = bundle["classes"]
    rng = np.random.default_rng(SEED)
    groups = frame.split_group.to_numpy()[test]
    unique = np.unique(groups)
    boot = []
    for _ in range(500):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        if set(y[idx]) == set(classes):
            boot.append(f1_score(y[idx], pred[idx], labels=classes, average="macro", zero_division=0))
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_f1_group_bootstrap_95_interval": np.quantile(boot, [0.025, 0.975]).tolist() if boot else None,
        "classifier_ms_per_image": latency,
        "test_images": len(test),
        "classification_report": classification_report(
            y, pred, labels=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
        "classes": classes,
    }
    return metrics, pred


def save_main(data, comparison, estimators):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, audit, dev, test, X, y = (
        data["frame"],
        data["audit"],
        data["dev"],
        data["test"],
        data["X"],
        data["y"],
    )
    inv = data["inventory"]
    coverage = audit["download_coverage"]
    (OUTPUT / "download_inventory.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    frame.to_csv(OUTPUT / "dataset_manifest.csv", index=False)
    (OUTPUT / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    split_table = pd.DataFrame(
        {
            "All unique images": frame.label.value_counts(),
            "Development / final training": frame.iloc[dev].label.value_counts(),
            "Holdout test": frame.iloc[test].label.value_counts(),
        }
    ).fillna(0).astype(int)
    split_table.loc["TOTAL"] = split_table.sum()
    split_table.to_csv(OUTPUT / "split_counts.csv")
    fold_table = pd.DataFrame(
        [{"fold": i + 1, "training_images": len(a), "validation_images": len(b)} for i, (a, b) in enumerate(data["cv"])]
    )
    fold_table.to_csv(OUTPUT / "cv_fold_counts.csv", index=False)

    chosen = comparison.iloc[0]
    estimator = clone(estimators[chosen.model]).fit(X[dev], y[dev])
    classes = sorted(frame.label.unique().tolist())
    bundle = {
        "estimator": estimator,
        "feature": "handcrafted",
        "feature_version": FEATURE_VERSION,
        "name": chosen.model,
        "classes": classes,
        "seed": SEED,
    }
    metrics, predictions = evaluate_selected(bundle, X, frame, test)
    fingerprint = hashlib.sha256(
        frame[["path", "label", "sha256", "split_group"]].to_csv(index=False).encode()
    ).hexdigest()
    bundle["dataset_fingerprint"] = fingerprint
    reason = (
        f"{chosen.model} had the highest mean grouped 3-fold development macro-F1 ({chosen.cv_macro_f1:.4f}). "
        "Macro-F1 weights each class equally. Exact score ties are broken by mean classifier fitting time, then model name. "
        "The holdout was excluded from selection."
    )
    summary = {
        "status": "trained",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": bundle["name"],
        "features": bundle["feature"],
        "classes": bundle["classes"],
        "selection_reason": reason,
        "dataset_fingerprint": fingerprint,
        "data_audit": audit,
        "development_images": len(dev),
        "test_images": len(test),
        "metrics": metrics,
        "comparison": comparison.drop(columns="fold_f1").to_dict("records"),
        "limitations": [
            "Research prototype; not a disease diagnosis or food-safety test.",
            "Whole-image features may learn background, lighting or basil variety rather than health.",
            "No calibrated confidence or reliable non-leaf/out-of-distribution detector is provided.",
            "No external-site or new-plant validation has been performed.",
            "The fixed holdout was evaluated in earlier project runs; it is not a new external test.",
        ],
        "dataset_scope": "partial" if coverage["partial_dataset"] else "complete_listed_counts",
        "transfer_compared": False,
    }
    split = frame.copy()
    split["split"] = "development"
    split.loc[test, "split"] = "test"
    split.to_csv(OUTPUT / "split_manifest.csv", index=False)
    comparison.to_csv(OUTPUT / "model_comparison.csv", index=False)
    test_results = frame.iloc[test][["path", "label", "split_group"]].copy()
    test_results["prediction"] = predictions
    test_results.to_csv(OUTPUT / "test_predictions.csv", index=False)
    joblib.dump(bundle, OUTPUT / "model.tmp")
    (OUTPUT / "model.tmp").replace(OUTPUT / "model.joblib")
    (OUTPUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    packages = ["numpy", "pandas", "pillow", "scipy", "scikit-learn", "scikit-image", "matplotlib", "joblib"]
    versions = {package: importlib.metadata.version(package) for package in packages}
    (OUTPUT / "environment.json").write_text(
        json.dumps({"python": sys.version, "packages": versions}, indent=2), encoding="utf-8"
    )
    print(reason)
    print("Main model saved to", OUTPUT / "model.joblib")
    print(
        f"Test accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"images={len(frame)} train={len(dev)} test={len(test)}"
    )
    return summary


