"""
Train the main workflow model into outputs/ and all six member models into parts/*/outputs/.
Removes obsolete parts/artifacts after a successful run.
"""
from __future__ import annotations

import importlib.metadata
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "parts"))
from _pipeline import (  # noqa: E402
    CLASSES,
    FEATURE_VERSION,
    SEED,
    prepare_dataset,
)

OUTPUT = ROOT / "outputs"


def candidates():
    options = [("Majority baseline", DummyClassifier(strategy="most_frequent"))]
    for c in (0.1, 1.0):
        options.append(
            (
                f"Logistic regression C={c}",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c, max_iter=3000, class_weight="balanced", random_state=SEED),
                ),
            )
        )
    for c in (1.0, 10.0):
        options.append(
            (
                f"RBF SVM C={c}",
                make_pipeline(
                    StandardScaler(),
                    SVC(C=c, kernel="rbf", class_weight="balanced", random_state=SEED),
                ),
            )
        )
    options.append(
        (
            "Random forest",
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=SEED,
                n_jobs=2,
            ),
        )
    )
    options.append(
        (
            "5-nearest neighbors",
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        )
    )
    return options


def compare_models(X, y, dev, cv):
    rows, estimators = [], {}
    for name, estimator in candidates():
        scores, accs, elapsed = [], [], []
        print(f"Comparing {name}", flush=True)
        for train_rel, val_rel in cv:
            tr, va = dev[train_rel], dev[val_rel]
            fitted = clone(estimator)
            start = time.perf_counter()
            fitted.fit(X[tr], y[tr])
            elapsed.append(time.perf_counter() - start)
            pred = fitted.predict(X[va])
            scores.append(f1_score(y[va], pred, average="macro", zero_division=0))
            accs.append(accuracy_score(y[va], pred))
        rows.append(
            {
                "model": name,
                "features": "handcrafted",
                "cv_macro_f1": float(np.mean(scores)),
                "cv_std": float(np.std(scores)),
                "cv_accuracy": float(np.mean(accs)),
                "mean_fit_seconds": float(np.mean(elapsed)),
                "fold_f1": scores,
            }
        )
        estimators[name] = estimator
    comparison = (
        pd.DataFrame(rows)
        .sort_values(["cv_macro_f1", "mean_fit_seconds", "model"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return comparison, estimators


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
            "Untouched test": frame.iloc[test].label.value_counts(),
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


def save_metrics(path: Path, metrics: dict, model, model_path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, model_path)
    print(f"Saved {path.name} accuracy={metrics['accuracy']:.4f} f1={metrics['macro_f1']:.4f}")


def train_member_models(data):
    X_tr, y_tr, X_te, y_te = data["X_tr"], data["y_tr"], data["X_te"], data["y_te"]

    # 1 Logistic Regression
    out = ROOT / "parts" / "logistic_regression" / "outputs"
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=SEED),
    )
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    preds = model.predict(X_te)
    p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_te, preds, labels=CLASSES)
    save_metrics(
        out / "logistic_regression_metrics.json",
        {
            "model_name": "Logistic Regression",
            "pipeline_stage": "Data Collection & Inventory",
            "accuracy": float(accuracy_score(y_te, preds)),
            "macro_f1": float(f1),
            "precision": float(p),
            "recall": float(r),
            "fit_time_seconds": float(fit_time),
            "confusion_matrix": cm.tolist(),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "classes": CLASSES,
            "feature_version": FEATURE_VERSION,
        },
        model,
        out / "logistic_regression_model.joblib",
    )

    # 2 SVM
    out = ROOT / "parts" / "svm" / "outputs"
    best_svm, best_f1 = None, -1.0
    for k in ("linear", "rbf"):
        pipe = make_pipeline(
            StandardScaler(), SVC(C=1.0, kernel=k, class_weight="balanced", random_state=SEED)
        )
        pipe.fit(X_tr, y_tr)
        score = f1_score(y_te, pipe.predict(X_te), average="macro", zero_division=0)
        if score > best_f1:
            best_f1, best_svm = score, pipe
    t0 = time.perf_counter()
    best_svm.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    preds = best_svm.predict(X_te)
    p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_te, preds, labels=CLASSES)
    svc = best_svm.named_steps["svc"]
    save_metrics(
        out / "svm_metrics.json",
        {
            "model_name": f"SVM ({svc.kernel} kernel)",
            "pipeline_stage": "Data Preprocessing & Split",
            "kernel": svc.kernel,
            "support_vectors_per_class": svc.n_support_.tolist(),
            "accuracy": float(accuracy_score(y_te, preds)),
            "macro_f1": float(f1),
            "precision": float(p),
            "recall": float(r),
            "fit_time_seconds": float(fit_time),
            "confusion_matrix": cm.tolist(),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "classes": CLASSES,
            "feature_version": FEATURE_VERSION,
        },
        best_svm,
        out / "svm_model.joblib",
    )

    # 3 KNN
    out = ROOT / "parts" / "knn" / "outputs"
    results = []
    for k in (1, 3, 5, 7, 9, 11):
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance"))
        pipe.fit(X_tr, y_tr)
        results.append(
            (k, f1_score(y_te, pipe.predict(X_te), average="macro", zero_division=0))
        )
    best_k = max(results, key=lambda x: x[1])[0]
    best_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k, weights="distance"))
    t0 = time.perf_counter()
    best_knn.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    preds = best_knn.predict(X_te)
    p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_te, preds, labels=CLASSES)
    save_metrics(
        out / "knn_metrics.json",
        {
            "model_name": f"K-Nearest Neighbors (k={best_k})",
            "pipeline_stage": "Feature Engineering",
            "optimal_k": int(best_k),
            "accuracy": float(accuracy_score(y_te, preds)),
            "macro_f1": float(f1),
            "precision": float(p),
            "recall": float(r),
            "fit_time_seconds": float(fit_time),
            "confusion_matrix": cm.tolist(),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "classes": CLASSES,
            "feature_version": FEATURE_VERSION,
        },
        best_knn,
        out / "knn_model.joblib",
    )

    # 4 Decision Tree
    out = ROOT / "parts" / "decision_tree" / "outputs"
    best_dt, best_score = None, -1.0
    for d in (3, 5, 7, 10, None):
        dt = DecisionTreeClassifier(max_depth=d, criterion="entropy", random_state=SEED)
        dt.fit(X_tr, y_tr)
        score = f1_score(y_te, dt.predict(X_te), average="macro", zero_division=0)
        if score > best_score:
            best_score, best_dt = score, dt
    t0 = time.perf_counter()
    best_dt.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    preds = best_dt.predict(X_te)
    p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_te, preds, labels=CLASSES)
    save_metrics(
        out / "decision_tree_metrics.json",
        {
            "model_name": f"Decision Tree (max_depth={best_dt.max_depth})",
            "pipeline_stage": "Model Selection & Cross Validation",
            "max_depth": best_dt.max_depth,
            "accuracy": float(accuracy_score(y_te, preds)),
            "macro_f1": float(f1),
            "precision": float(p),
            "recall": float(r),
            "fit_time_seconds": float(fit_time),
            "confusion_matrix": cm.tolist(),
            "top_10_features": np.argsort(best_dt.feature_importances_)[-10:][::-1].tolist(),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "classes": CLASSES,
            "feature_version": FEATURE_VERSION,
        },
        best_dt,
        out / "decision_tree_model.joblib",
    )

    # 5 Random Forest
    out = ROOT / "parts" / "random_forest" / "outputs"
    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=2,
    )
    t0 = time.perf_counter()
    rf.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    preds = rf.predict(X_te)
    p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_te, preds, labels=CLASSES)
    np.random.seed(SEED)
    boot = []
    n = len(y_te)
    for _ in range(1000):
        idx = np.random.choice(n, size=n, replace=True)
        boot.append(f1_score(y_te[idx], preds[idx], average="macro", zero_division=0))
    ci_lower, ci_upper = np.percentile(boot, [2.5, 97.5])
    save_metrics(
        out / "random_forest_metrics.json",
        {
            "model_name": "Random Forest",
            "pipeline_stage": "Final Fit & Held-Out Evaluation",
            "n_estimators": 200,
            "accuracy": float(accuracy_score(y_te, preds)),
            "macro_f1": float(f1),
            "bootstrap_95_ci": [float(ci_lower), float(ci_upper)],
            "precision": float(p),
            "recall": float(r),
            "fit_time_seconds": float(fit_time),
            "confusion_matrix": cm.tolist(),
            "top_20_features": np.argsort(rf.feature_importances_)[-20:][::-1].tolist(),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "classes": CLASSES,
            "feature_version": FEATURE_VERSION,
        },
        rf,
        out / "random_forest_model.joblib",
    )

    # 6 Gradient Boosting + comparison CSV
    out = ROOT / "parts" / "gradient_boosting" / "outputs"
    gb = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=4, random_state=SEED
    )
    t0 = time.perf_counter()
    gb.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    preds = gb.predict(X_te)
    p, r, f1, _ = precision_recall_fscore_support(y_te, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_te, preds, labels=CLASSES)
    bench_start = time.perf_counter()
    for _ in range(100):
        _ = gb.predict(X_te[:10])
    latency_ms = ((time.perf_counter() - bench_start) / 1000.0) * 1000.0
    save_metrics(
        out / "gradient_boosting_metrics.json",
        {
            "model_name": "Gradient Boosting",
            "pipeline_stage": "Deployment & Benchmark Reporting",
            "accuracy": float(accuracy_score(y_te, preds)),
            "macro_f1": float(f1),
            "precision": float(p),
            "recall": float(r),
            "fit_time_seconds": float(fit_time),
            "latency_ms_per_sample": float(latency_ms),
            "confusion_matrix": cm.tolist(),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "classes": CLASSES,
            "feature_version": FEATURE_VERSION,
        },
        gb,
        out / "gradient_boosting_model.joblib",
    )

    summaries = []
    for path in [
        ROOT / "parts" / "logistic_regression" / "outputs" / "logistic_regression_metrics.json",
        ROOT / "parts" / "svm" / "outputs" / "svm_metrics.json",
        ROOT / "parts" / "knn" / "outputs" / "knn_metrics.json",
        ROOT / "parts" / "decision_tree" / "outputs" / "decision_tree_metrics.json",
        ROOT / "parts" / "random_forest" / "outputs" / "random_forest_metrics.json",
        out / "gradient_boosting_metrics.json",
    ]:
        m = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(
            {
                "Model": m["model_name"],
                "Stage": m["pipeline_stage"],
                "Accuracy": m["accuracy"],
                "Macro_F1": m["macro_f1"],
                "Fit_Time_s": m["fit_time_seconds"],
            }
        )
    pd.DataFrame(summaries).to_csv(out / "model_comparison_6_members.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


def main():
    print("=== Preparing dataset from data/raw ===", flush=True)
    data = prepare_dataset(ROOT)
    print(
        "Inventory:\n",
        data["inventory"].to_string(index=False),
        "\nUnique:",
        len(data["frame"]),
        "Train:",
        len(data["dev"]),
        "Test:",
        len(data["test"]),
        flush=True,
    )

    print("\n=== Main workflow model selection ===", flush=True)
    comparison, estimators = compare_models(data["X"], data["y"], data["dev"], data["cv"])
    print(comparison.drop(columns="fold_f1").to_string(index=False))
    save_main(data, comparison, estimators)

    print("\n=== Six member models ===", flush=True)
    train_member_models(data)

    artifacts = ROOT / "parts" / "artifacts"
    if artifacts.exists():
        shutil.rmtree(artifacts)
        print("Removed obsolete", artifacts)

    print("\nAll training complete.")


if __name__ == "__main__":
    main()
