"""Generate all Progress Review I EDA / process diagrams as PNG files.

Run from repo root or Group_Deliverable:
  .\\.venv\\Scripts\\python.exe progress\\Group_Deliverable\\_generate_all_visualizations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from skimage.color import rgb2hsv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess_utils import (  # noqa: E402
    CLASSES,
    SEED,
    audit_images,
    discover_images,
    draw_process_flow,
    extract_feature_matrix,
    inventory_table,
    iqr_mask,
    paths,
    read_rgb,
    remove_exact_duplicates,
    stratified_sample,
)

P = paths(ROOT)
RAW, VIZ, OUT, LOGS = P["raw"], P["viz"], P["outputs"], P["logs"]
for d in (VIZ, OUT, LOGS):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
np.random.seed(SEED)

PIPELINE_STEPS = [
    "1 Missing\ndata",
    "2 Categorical\nencoding",
    "3 Outlier /\nduplicate",
    "4 Feature\nengineering",
    "5 Scaling",
    "6 SelectKBest\n+ PCA",
]


def save_flow(name: str, highlight: int | None, title: str) -> None:
    fig, _ = draw_process_flow(PIPELINE_STEPS, title=title, highlight=highlight)
    fig.savefig(VIZ / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", name)


def main() -> None:
    print("Raw data:", RAW)
    save_flow("group_process_flow.png", None, "Group preprocessing pipeline — full process flow")
    for i, name in enumerate(
        [
            "m1_process_flow.png",
            "m2_process_flow.png",
            "m3_process_flow.png",
            "m5_process_flow.png",
            "m4_process_flow.png",
            "m6_process_flow.png",
        ]
    ):
        # highlight index follows PIPELINE_STEPS order; m4 is step 5 (index 4), m5 is step 4 (index 3)
        highlight_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        titles = [
            "Preprocessing pipeline (Member 1 highlighted)",
            "Preprocessing pipeline (Member 2 highlighted)",
            "Preprocessing pipeline (Member 3 highlighted)",
            "Preprocessing pipeline (Member 5 highlighted)",
            "Preprocessing pipeline (Member 4 highlighted)",
            "Preprocessing pipeline (Member 6 highlighted)",
        ]
        save_flow(name, highlight_map[i], titles[i])

    inv = inventory_table(RAW)
    discovered = discover_images(RAW)
    valid, rejected = audit_images(RAW, discovered)
    print(f"Valid={len(valid)} Rejected={len(rejected)}")

    valid.to_csv(OUT / "m1_valid_image_index.csv", index=False)
    rejected.to_csv(OUT / "m1_rejected_images.csv", index=False)
    inv.to_csv(OUT / "m1_folder_inventory.csv", index=False)
    inv.to_csv(OUT / "group_folder_inventory.csv", index=False)

    # --- Member 1 plots ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    class_counts = valid["label"].value_counts().reindex(CLASSES).fillna(0)
    bars = axes[0, 0].bar(class_counts.index, class_counts.values, color=["#2ca02c", "#d62728"])
    axes[0, 0].set_title("Readable images per class")
    axes[0, 0].set_ylabel("Count")
    for b, v in zip(bars, class_counts.values):
        axes[0, 0].text(b.get_x() + b.get_width() / 2, v + 5, str(int(v)), ha="center")
    axes[0, 1].barh(inv["folder"], inv["found"], color="#1f77b4", label="Found")
    axes[0, 1].barh(
        inv["folder"], inv["missing_count"], left=inv["found"], color="#ff7f0e", label="Missing vs expected"
    )
    axes[0, 1].set_title("Folder completeness (found + missing)")
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].set_xlabel("Images")
    x = np.arange(len(inv))
    w = 0.35
    short = [f.replace("_Region_Basil_Plant_Healthy", "").replace("Basil_Plant_", "") for f in inv["folder"]]
    axes[1, 0].bar(x - w / 2, inv["expected"], width=w, label="Expected", color="#7f7f7f")
    axes[1, 0].bar(x + w / 2, inv["found"], width=w, label="Found", color="#1f77b4")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(short, rotation=25, ha="right")
    axes[1, 0].set_title("Expected vs found (bar comparison)")
    axes[1, 0].legend()
    audit_counts = pd.Series({"Valid": len(valid), "Rejected": len(rejected)})
    axes[1, 1].bar(audit_counts.index, audit_counts.values, color=["#2ca02c", "#d62728"])
    axes[1, 1].set_title("Audit outcome: valid vs rejected")
    for i, v in enumerate(audit_counts.values):
        axes[1, 1].text(i, v + max(5, 0.01 * max(audit_counts.values)), str(int(v)), ha="center")
    fig.suptitle("Member 1 — Missing / corrupt data EDA", fontsize=14)
    fig.tight_layout()
    fig.savefig(VIZ / "m1_class_and_completeness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Member 2 ---
    label_map = {c: i for i, c in enumerate(CLASSES)}
    regions = sorted(valid["region"].dropna().unique().tolist())
    region_map = {r: i for i, r in enumerate(regions)}
    encoded = valid.copy()
    encoded["label_encoded"] = encoded["label"].map(label_map).astype(int)
    encoded["region_encoded"] = encoded["region"].map(region_map).astype(int)
    region_dummies = pd.get_dummies(encoded["region"], prefix="region")
    encoded = pd.concat([encoded, region_dummies], axis=1)
    encoded.to_csv(OUT / "m2_encoded_metadata.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    lc = encoded["label_encoded"].value_counts().sort_index()
    axes[0, 0].bar([CLASSES[i] for i in lc.index], lc.values, color=["#2ca02c", "#d62728"])
    axes[0, 0].set_title("Encoded labels (0=Healthy, 1=Unhealthy)")
    rc = encoded.groupby("region")["region_encoded"].count().sort_values(ascending=False)
    axes[0, 1].bar(rc.index, rc.values, color="#9467bd")
    axes[0, 1].set_title("Images per region category")
    axes[0, 1].tick_params(axis="x", rotation=20)
    ct = pd.crosstab(encoded["region"], encoded["label"]).reindex(columns=CLASSES).fillna(0)
    ct.plot(kind="bar", ax=axes[1, 0], color=["#2ca02c", "#d62728"], rot=20)
    axes[1, 0].set_title("Class × region comparison (grouped bars)")
    oh_cols = [c for c in encoded.columns if c.startswith("region_")]
    oh_sums = encoded[oh_cols].sum().sort_values(ascending=False)
    axes[1, 1].barh([c.replace("region_", "") for c in oh_sums.index], oh_sums.values, color="#17becf")
    axes[1, 1].set_title("One-hot region column totals")
    fig.suptitle("Member 2 — Categorical encoding EDA", fontsize=14)
    fig.tight_layout()
    fig.savefig(VIZ / "m2_categorical_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Member 3 ---
    before = len(encoded)
    df_unique, n_dup = remove_exact_duplicates(encoded)
    keep = iqr_mask(df_unique["pixels"]) & iqr_mask(df_unique["mean_g"])
    removed = df_unique.loc[~keep].copy()
    clean = df_unique.loc[keep].reset_index(drop=True)
    clean.to_csv(OUT / "m3_cleaned_no_outliers.csv", index=False)
    clean.to_csv(OUT / "group_cleaned_metadata.csv", index=False)
    removed.to_csv(OUT / "m3_removed_outliers.csv", index=False)
    print(f"Dup={n_dup} IQR={len(removed)} Clean={len(clean)}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.boxplot(
        data=df_unique, x="label", y="pixels", hue="label", order=CLASSES,
        ax=axes[0, 0], palette=["#2ca02c", "#d62728"], legend=False,
    )
    axes[0, 0].set_title("Image resolution (pixels) by class")
    sns.boxplot(
        data=df_unique, x="label", y="mean_g", hue="label", order=CLASSES,
        ax=axes[0, 1], palette=["#2ca02c", "#d62728"], legend=False,
    )
    axes[0, 1].set_title("Mean green intensity by class")
    stages = ["Audited", "After duplicates", "After IQR"]
    stage_counts = [before, len(df_unique), len(clean)]
    axes[1, 0].bar(stages, stage_counts, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1, 0].set_title("Sample count before vs after cleaning")
    for i, v in enumerate(stage_counts):
        axes[1, 0].text(i, v + 5, str(v), ha="center")
    before_cls = encoded["label"].value_counts().reindex(CLASSES).fillna(0)
    after_cls = clean["label"].value_counts().reindex(CLASSES).fillna(0)
    x = np.arange(len(CLASSES))
    axes[1, 1].bar(x - w / 2, before_cls.values, width=w, label="Before", color="#7f7f7f")
    axes[1, 1].bar(x + w / 2, after_cls.values, width=w, label="After clean", color=["#2ca02c", "#d62728"])
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(CLASSES)
    axes[1, 1].set_title("Class counts before vs after (comparison)")
    axes[1, 1].legend()
    fig.suptitle("Member 3 — Outlier / duplicate removal EDA", fontsize=14)
    fig.tight_layout()
    fig.savefig(VIZ / "m3_outlier_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(clean["pixels"], clean["mean_g"], s=12, alpha=0.45, c="#1f77b4", label="Kept")
    if len(removed):
        ax.scatter(removed["pixels"], removed["mean_g"], s=28, alpha=0.85, c="#d62728", marker="x", label="IQR removed")
    ax.set_xlabel("Resolution (pixels)")
    ax.set_ylabel("Mean green")
    ax.set_title("Outlier view: resolution vs mean green")
    ax.legend()
    fig.tight_layout()
    fig.savefig(VIZ / "m3_outlier_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Member 5 features (sample) ---
    sample5 = stratified_sample(clean, 100)
    rows = []
    for row in sample5.to_dict("records"):
        im = read_rgb(RAW / row["path"])
        rgb = np.asarray(im.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        hsv = rgb2hsv(rgb)
        rows.append(
            {
                "path": row["path"],
                "label": row["label"],
                "rgb_r_mean": float(rgb[:, :, 0].mean()),
                "rgb_g_mean": float(rgb[:, :, 1].mean()),
                "rgb_b_mean": float(rgb[:, :, 2].mean()),
                "hsv_h_mean": float(hsv[:, :, 0].mean()),
                "hsv_s_mean": float(hsv[:, :, 1].mean()),
                "hsv_v_mean": float(hsv[:, :, 2].mean()),
                "rgb_g_std": float(rgb[:, :, 1].std()),
                "hsv_s_std": float(hsv[:, :, 1].std()),
            }
        )
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(OUT / "m5_engineered_color_features.csv", index=False)
    num_cols = [c for c in feat_df.columns if c not in {"path", "label"}]
    corr = feat_df[num_cols].corr()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0, 0])
    axes[0, 0].set_title("Correlation of engineered color features")
    sns.boxplot(
        data=feat_df, x="label", y="rgb_g_mean", hue="label", order=CLASSES,
        palette=["#2ca02c", "#d62728"], ax=axes[0, 1], legend=False,
    )
    axes[0, 1].set_title("Mean green channel by class")
    channel_means = feat_df.groupby("label")[["rgb_r_mean", "rgb_g_mean", "rgb_b_mean"]].mean().reindex(CLASSES)
    channel_means.plot(kind="bar", ax=axes[1, 0], color=["#d62728", "#2ca02c", "#1f77b4"], rot=0)
    axes[1, 0].set_title("RGB channel means by class (comparison)")
    hsv_means = feat_df.groupby("label")[["hsv_h_mean", "hsv_s_mean", "hsv_v_mean"]].mean().reindex(CLASSES)
    hsv_means.plot(kind="bar", ax=axes[1, 1], color=["#9467bd", "#e377c2", "#8c564b"], rot=0)
    axes[1, 1].set_title("HSV channel means by class (comparison)")
    fig.suptitle("Member 5 — Feature engineering EDA", fontsize=14)
    fig.tight_layout()
    fig.savefig(VIZ / "m5_feature_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=feat_df, x="label", y="rgb_g_mean", hue="label", order=CLASSES,
        palette=["#2ca02c", "#d62728"], ax=ax, legend=False,
    )
    ax.set_title("Mean green channel by class (engineered feature)")
    fig.tight_layout()
    fig.savefig(VIZ / "m5_greenness_by_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Member 4 + 6 / group features ---
    sample = stratified_sample(clean, 150)
    X, y, kept = extract_feature_matrix(RAW, sample)
    X_std = StandardScaler().fit_transform(X)
    X_mm = MinMaxScaler().fit_transform(X)
    np.savez_compressed(
        OUT / "m4_scaled_features.npz",
        X_raw=X,
        X_standard=X_std,
        X_minmax=X_mm,
        y=y,
        feature_dim=np.array([X.shape[1]]),
    )
    kept.to_csv(OUT / "m4_feature_sample_index.csv", index=False)

    feat_idx = 0
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].hist(X[:, feat_idx], bins=30, color="#1f77b4", alpha=0.85)
    axes[0, 0].set_title(f"Raw feature[{feat_idx}]")
    axes[0, 1].hist(X_std[:, feat_idx], bins=30, color="#ff7f0e", alpha=0.85)
    axes[0, 1].set_title(f"StandardScaler feature[{feat_idx}]")
    axes[1, 0].hist(X_mm[:, feat_idx], bins=30, color="#2ca02c", alpha=0.85)
    axes[1, 0].set_title(f"MinMaxScaler feature[{feat_idx}]")
    methods = ["Raw", "StandardScaler", "MinMaxScaler"]
    mean_abs = [float(np.mean(np.abs(X))), float(np.mean(np.abs(X_std))), float(np.mean(np.abs(X_mm)))]
    stds = [float(np.mean(X.std(axis=0))), float(np.mean(X_std.std(axis=0))), float(np.mean(X_mm.std(axis=0)))]
    x = np.arange(len(methods))
    axes[1, 1].bar(x - w / 2, mean_abs, width=w, label="Mean |value|", color="#1f77b4")
    axes[1, 1].bar(x + w / 2, stds, width=w, label="Mean feature std", color="#ff7f0e")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods, rotation=15)
    axes[1, 1].set_title("Scaler comparison (bar plots)")
    axes[1, 1].legend()
    fig.suptitle("Member 4 — Normalization / scaling EDA", fontsize=14)
    fig.tight_layout()
    fig.savefig(VIZ / "m4_scaling_before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    y_enc = LabelEncoder().fit_transform(y)
    k = min(20, X_std.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_sel = selector.fit_transform(X_std, y_enc)
    pca = PCA(n_components=min(10, X_sel.shape[1]), random_state=SEED)
    X_pca = pca.fit_transform(X_sel)
    np.savez_compressed(
        OUT / "m6_selected_pca_features.npz",
        X_selected=X_sel,
        X_pca=X_pca,
        y=y,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    pd.DataFrame(
        {
            "pc": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(OUT / "m6_pca_variance.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    scores = selector.scores_
    top_idx = np.argsort(scores)[::-1][:15]
    axes[0, 0].barh([f"f{i}" for i in top_idx[::-1]], scores[top_idx][::-1], color="#1f77b4")
    axes[0, 0].set_title("SelectKBest ANOVA F-scores (top 15)")
    pcs = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    axes[0, 1].bar(pcs, pca.explained_variance_ratio_, color="#ff7f0e", label="Per-PC ratio")
    axes[0, 1].plot(pcs, np.cumsum(pca.explained_variance_ratio_), marker="o", color="#d62728", label="Cumulative")
    axes[0, 1].axhline(0.9, color="gray", ls="--", lw=1, label="90%")
    axes[0, 1].set_title("PCA explained variance (bars + cumulative)")
    axes[0, 1].legend(fontsize=8)
    for cls, color in zip(CLASSES, ["#2ca02c", "#d62728"]):
        mask = y == cls
        axes[1, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], s=18, alpha=0.7, label=cls, c=color)
    axes[1, 0].set_xlabel("PC1")
    axes[1, 0].set_ylabel("PC2")
    axes[1, 0].set_title("PCA scatter (SelectKBest → PCA)")
    axes[1, 0].legend()
    dims = ["Raw features", "After SelectKBest", "After PCA"]
    dim_vals = [X_std.shape[1], X_sel.shape[1], X_pca.shape[1]]
    axes[1, 1].bar(dims, dim_vals, color=["#7f7f7f", "#1f77b4", "#2ca02c"])
    axes[1, 1].set_title("Feature dimensionality reduction (comparison)")
    axes[1, 1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(dim_vals):
        axes[1, 1].text(i, v + 0.5, str(v), ha="center")
    fig.suptitle("Member 6 — Feature selection & PCA EDA", fontsize=14)
    fig.tight_layout()
    fig.savefig(VIZ / "m6_pca_variance_and_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Group charts ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    funnel_labels = ["Discovered", "Valid", "Unique", "Clean"]
    funnel_vals = [len(discovered), len(valid), len(df_unique), len(clean)]
    axes[0, 0].bar(funnel_labels, funnel_vals, color=["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[0, 0].set_title("Cleaning funnel (bar comparison)")
    for i, v in enumerate(funnel_vals):
        axes[0, 0].text(i, v + 5, str(v), ha="center")
    x = np.arange(len(inv))
    axes[0, 1].bar(x - w / 2, inv["expected"], width=w, label="Expected", color="#7f7f7f")
    axes[0, 1].bar(x + w / 2, inv["found"], width=w, label="Found", color="#1f77b4")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(short, rotation=25, ha="right")
    axes[0, 1].set_title("Expected vs found by folder")
    axes[0, 1].legend()
    class_counts = clean["label"].value_counts().reindex(CLASSES).fillna(0)
    axes[1, 0].bar(class_counts.index, class_counts.values, color=["#2ca02c", "#d62728"])
    axes[1, 0].set_title("Clean class counts")
    ct = pd.crosstab(clean["region"], clean["label"]).reindex(columns=CLASSES).fillna(0)
    ct.plot(kind="bar", ax=axes[1, 1], color=["#2ca02c", "#d62728"], rot=20)
    axes[1, 1].set_title("Clean class × region comparison")
    fig.suptitle("Group pipeline — inventory & cleaning comparisons", fontsize=14)
    fig.tight_layout()
    fig.savefig(VIZ / "group_cleaning_comparisons.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Group PCA with 5 components (matches notebook)
    pca5 = PCA(n_components=min(5, X_sel.shape[1]), random_state=SEED)
    X_pca5 = pca5.fit_transform(X_sel)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.boxplot(
        data=df_unique, x="label", y="mean_g", hue="label", order=CLASSES,
        ax=axes[0, 0], palette=["#2ca02c", "#d62728"], legend=False,
    )
    axes[0, 0].set_title("Green intensity (outlier view)")
    axes[0, 1].hist(X[:, 0], bins=25, alpha=0.55, label="raw")
    axes[0, 1].hist(X_std[:, 0], bins=25, alpha=0.55, label="scaled")
    axes[0, 1].set_title("Feature[0] before/after scaling")
    axes[0, 1].legend()
    pcs = np.arange(1, len(pca5.explained_variance_ratio_) + 1)
    axes[1, 0].bar(pcs, pca5.explained_variance_ratio_, color="#ff7f0e", label="Per-PC")
    axes[1, 0].plot(pcs, np.cumsum(pca5.explained_variance_ratio_), "o-", color="#d62728", label="Cumulative")
    axes[1, 0].set_title("PCA explained variance (comparison)")
    axes[1, 0].legend(fontsize=8)
    for cls, color in zip(CLASSES, ["#2ca02c", "#d62728"]):
        m = y == cls
        axes[1, 1].scatter(X_pca5[m, 0], X_pca5[m, 1], s=16, alpha=0.75, c=color, label=cls)
    axes[1, 1].set_title("PCA scatter after selection")
    axes[1, 1].legend()
    fig.suptitle("Group pipeline — feature & PCA EDA", fontsize=14)
    fig.tight_layout()
    fig.savefig(VIZ / "group_feature_pca_comparisons.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    dims = ["Raw features", "SelectKBest", "PCA"]
    dim_vals = [X.shape[1], X_sel.shape[1], X_pca5.shape[1]]
    ax.bar(dims, dim_vals, color=["#7f7f7f", "#1f77b4", "#2ca02c"])
    ax.set_title("Dimensionality reduction funnel")
    for i, v in enumerate(dim_vals):
        ax.text(i, v + 0.5, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(VIZ / "group_dimension_funnel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].bar(class_counts.index, class_counts.values, color=["#2ca02c", "#d62728"])
    axes[0, 0].set_title("1) Clean class counts")
    sns.boxplot(
        data=df_unique, x="label", y="mean_g", hue="label", order=CLASSES,
        ax=axes[0, 1], palette=["#2ca02c", "#d62728"], legend=False,
    )
    axes[0, 1].set_title("2) Green intensity (outlier view)")
    axes[1, 0].hist(X[:, 0], bins=25, alpha=0.55, label="raw")
    axes[1, 0].hist(X_std[:, 0], bins=25, alpha=0.55, label="scaled")
    axes[1, 0].set_title("3) Feature[0] before/after scaling")
    axes[1, 0].legend()
    for cls, color in zip(CLASSES, ["#2ca02c", "#d62728"]):
        m = y == cls
        axes[1, 1].scatter(X_pca5[m, 0], X_pca5[m, 1], s=16, alpha=0.75, c=color, label=cls)
    axes[1, 1].set_title("4) PCA scatter after selection")
    axes[1, 1].legend()
    fig.suptitle("Group preprocessing pipeline — EDA summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(VIZ / "group_pipeline_eda_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    kept.assign(label_encoded=[label_map[v] for v in y]).to_csv(OUT / "group_feature_index.csv", index=False)
    pd.DataFrame(X_std).to_csv(OUT / "group_features_scaled.csv", index=False)
    pd.DataFrame(X_pca5, columns=[f"PC{i+1}" for i in range(X_pca5.shape[1])]).assign(label=y).to_csv(
        OUT / "group_features_pca.csv", index=False
    )
    np.savez_compressed(OUT / "group_features.npz", X_scaled=X_std, X_pca=X_pca5, y=y)
    summary = {
        "discovered": int(len(discovered)),
        "valid": int(len(valid)),
        "rejected": int(len(rejected)),
        "duplicates_removed": int(n_dup),
        "clean": int(len(clean)),
        "feature_rows": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "pca_explained_variance_ratio": pca5.explained_variance_ratio_.tolist(),
        "label_map": label_map,
        "region_map": region_map,
    }
    (LOGS / "group_pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (LOGS / "m1_missing_data.json").write_text(
        json.dumps(
            {
                "discovered": int(len(discovered)),
                "valid": int(len(valid)),
                "rejected": int(len(rejected)),
                "missing_vs_expected": int(inv["missing_count"].sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("All visualizations written to", VIZ)
    print("PNG count:", len(list(VIZ.glob("*.png"))))


if __name__ == "__main__":
    main()
