"""Generate Progress Review I notebooks under Group_Deliverable."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(text),
    }


def _src(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    lines = text.splitlines(keepends=True)
    return lines


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


SETUP = r'''
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Resolve Group_Deliverable root whether cwd is notebooks/ or deliverable root
HERE = Path.cwd().resolve()
ROOT = None
for p in (HERE, *HERE.parents):
    if (p / "src" / "preprocess_utils.py").exists():
        ROOT = p
        break
    if (p / "Group_Deliverable" / "src" / "preprocess_utils.py").exists():
        ROOT = p / "Group_Deliverable"
        break
if ROOT is None:
    raise FileNotFoundError("Run from progress/Group_Deliverable or its notebooks/ folder.")

sys.path.insert(0, str(ROOT / "src"))
from preprocess_utils import (
    SEED, SOURCE_URL, DOI, FOLDERS, CLASSES, paths,
    inventory_table, discover_images, audit_images,
    remove_exact_duplicates, iqr_mask, extract_feature_matrix, read_rgb,
    stratified_sample, draw_process_flow,
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
print("Deliverable root:", ROOT)
print("Raw data:", RAW)
print("Dataset:", SOURCE_URL, "| DOI:", DOI)
'''

# ---------------------------------------------------------------------------
# Member notebooks
# ---------------------------------------------------------------------------

member_specs = [
    (
        "ITXXXXXX1_Missing_Data.ipynb",
        "Member 1 — Handling Missing Data",
        "Handling missing / corrupt image files",
        [
            md(
                """# Member 1 — Handling Missing Data

**Technique:** Detect and handle missing or unreadable images before modelling.

## Why this dataset needs it
The basil dataset is organised as image folders, not a tidy CSV. “Missing data” here means:
- expected files listed for a region folder are absent
- files exist but **fail to decode** (corrupt / unsupported)
- an entire labelled folder is empty

Without this step, training would silently drop samples or crash mid-pipeline.
"""
            ),
            code(SETUP),
            md("## 0. Process diagram — where this step sits in the pipeline"),
            code(
                r'''
fig, _ = draw_process_flow(
    PIPELINE_STEPS,
    title="Preprocessing pipeline (Member 1 highlighted)",
    highlight=0,
)
fig.savefig(VIZ / "m1_process_flow.png", dpi=150, bbox_inches="tight")
plt.show()
'''
            ),
            md("## 1. Inventory expected vs found images"),
            code(
                r'''
inv = inventory_table(RAW)
display(inv)

missing_total = int(inv["missing_count"].sum())
print(f"Total missing vs expected file counts: {missing_total}")
print(f"Folders incomplete: {(~inv['complete']).sum()} / {len(inv)}")
'''
            ),
            md("## 2. Discover files and audit readability (treat failed reads as missing)"),
            code(
                r'''
discovered = discover_images(RAW)
print("Discovered image files:", len(discovered))

valid, rejected = audit_images(RAW, discovered)
print("Readable images:", len(valid))
print("Rejected / unreadable:", len(rejected))
if len(rejected):
    display(rejected[["path", "reason"]].head(20))

# Persist cleaned inventory for the group pipeline
valid.to_csv(OUT / "m1_valid_image_index.csv", index=False)
rejected.to_csv(OUT / "m1_rejected_images.csv", index=False)
inv.to_csv(OUT / "m1_folder_inventory.csv", index=False)

log = {
    "discovered": int(len(discovered)),
    "valid": int(len(valid)),
    "rejected": int(len(rejected)),
    "missing_vs_expected": missing_total,
}
(LOGS / "m1_missing_data.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
print("Saved outputs under results/outputs and results/logs")
'''
            ),
            md("## 3. EDA visualizations — class balance, completeness & comparisons"),
            code(
                r'''
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (1) Class counts among readable images
class_counts = valid["label"].value_counts().reindex(CLASSES).fillna(0)
bars = axes[0, 0].bar(class_counts.index, class_counts.values, color=["#2ca02c", "#d62728"])
axes[0, 0].set_title("Readable images per class")
axes[0, 0].set_ylabel("Count")
for b, v in zip(bars, class_counts.values):
    axes[0, 0].text(b.get_x() + b.get_width() / 2, v + 5, str(int(v)), ha="center")

# (2) Folder completeness stacked bars
axes[0, 1].barh(inv["folder"], inv["found"], color="#1f77b4", label="Found")
axes[0, 1].barh(inv["folder"], inv["missing_count"], left=inv["found"], color="#ff7f0e", label="Missing vs expected")
axes[0, 1].set_title("Folder completeness (found + missing)")
axes[0, 1].legend(loc="lower right")
axes[0, 1].set_xlabel("Images")

# (3) Expected vs found comparison (grouped bars)
x = np.arange(len(inv))
w = 0.35
axes[1, 0].bar(x - w / 2, inv["expected"], width=w, label="Expected", color="#7f7f7f")
axes[1, 0].bar(x + w / 2, inv["found"], width=w, label="Found", color="#1f77b4")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels([f.replace("_Region_Basil_Plant_Healthy", "").replace("Basil_Plant_", "") for f in inv["folder"]], rotation=25, ha="right")
axes[1, 0].set_title("Expected vs found (bar comparison)")
axes[1, 0].set_ylabel("Images")
axes[1, 0].legend()

# (4) Audit outcome pie / bar
audit_counts = pd.Series({"Valid": len(valid), "Rejected": len(rejected)})
axes[1, 1].bar(audit_counts.index, audit_counts.values, color=["#2ca02c", "#d62728"])
axes[1, 1].set_title("Audit outcome: valid vs rejected")
axes[1, 1].set_ylabel("Count")
for i, v in enumerate(audit_counts.values):
    axes[1, 1].text(i, v + max(5, 0.01 * max(audit_counts.values)), str(int(v)), ha="center")

fig.suptitle("Member 1 — Missing / corrupt data EDA", fontsize=14)
fig.tight_layout()
fig.savefig(VIZ / "m1_class_and_completeness.png", dpi=150, bbox_inches="tight")
plt.show()
print("Interpretation: bars show whether both classes are present and whether any source folder is under-complete before later preprocessing.")
'''
            ),
            md(
                """## Viva talking points
1. Define missing data for **images** (absent files + failed loads).
2. Show the inventory table and rejected list.
3. Interpret the class-balance / expected-vs-found comparison charts.
"""
            ),
        ],
    ),
    (
        "ITXXXXXX2_Categorical_Encoding.ipynb",
        "Member 2 — Categorical Encoding",
        "Encoding categorical variables",
        [
            md(
                """# Member 2 — Encoding Categorical Variables

**Technique:** Convert categorical fields (`label`, `region`, `folder`) into numeric codes ML models can consume.

## Why this dataset needs it
Folder names carry the **class** (Healthy / Unhealthy) and **region** (Amravati / Nagpur / Pune / Unknown). Models and correlation plots need integers or one-hot columns — not free text.
"""
            ),
            code(SETUP),
            md("## 0. Process diagram — where this step sits in the pipeline"),
            code(
                r'''
fig, _ = draw_process_flow(
    PIPELINE_STEPS,
    title="Preprocessing pipeline (Member 2 highlighted)",
    highlight=1,
)
fig.savefig(VIZ / "m2_process_flow.png", dpi=150, bbox_inches="tight")
plt.show()
'''
            ),
            md("## 1. Load audited index (or rebuild) and encode categories"),
            code(
                r'''
index_path = OUT / "m1_valid_image_index.csv"
if index_path.exists():
    df = pd.read_csv(index_path)
    print("Loaded Member 1 valid index:", len(df))
else:
    df, rejected = audit_images(RAW)
    print("Audited fresh valid images:", len(df), "| rejected:", len(rejected))

# Label encoding (ordered class list for reproducibility)
label_map = {c: i for i, c in enumerate(CLASSES)}
df["label_encoded"] = df["label"].map(label_map).astype(int)

# Region encoding
regions = sorted(df["region"].dropna().unique().tolist())
region_map = {r: i for i, r in enumerate(regions)}
df["region_encoded"] = df["region"].map(region_map).astype(int)

# One-hot for region (useful for linear models / EDA)
region_dummies = pd.get_dummies(df["region"], prefix="region")
encoded = pd.concat([df, region_dummies], axis=1)

display(pd.DataFrame({"category": ["label", "region"], "mapping": [label_map, region_map]}))
display(encoded[["path", "label", "label_encoded", "region", "region_encoded"]].head(10))

encoded.to_csv(OUT / "m2_encoded_metadata.csv", index=False)
(LOGS / "m2_encoding_maps.json").write_text(
    json.dumps({"label_map": label_map, "region_map": region_map}, indent=2),
    encoding="utf-8",
)
'''
            ),
            md("## 2. EDA visualizations — encoded distributions & class × region comparison"),
            code(
                r'''
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Label codes
lc = encoded["label_encoded"].value_counts().sort_index()
axes[0, 0].bar([CLASSES[i] for i in lc.index], lc.values, color=["#2ca02c", "#d62728"])
axes[0, 0].set_title("Encoded labels (0=Healthy, 1=Unhealthy)")
axes[0, 0].set_ylabel("Count")

# Region codes
rc = encoded.groupby("region")["region_encoded"].count().sort_values(ascending=False)
axes[0, 1].bar(rc.index, rc.values, color="#9467bd")
axes[0, 1].set_title("Images per region category")
axes[0, 1].tick_params(axis="x", rotation=20)
axes[0, 1].set_ylabel("Count")

# Class × region stacked / grouped comparison
ct = pd.crosstab(encoded["region"], encoded["label"]).reindex(columns=CLASSES).fillna(0)
ct.plot(kind="bar", ax=axes[1, 0], color=["#2ca02c", "#d62728"], rot=20)
axes[1, 0].set_title("Class × region comparison (grouped bars)")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_xlabel("Region")

# One-hot column totals
oh_cols = [c for c in encoded.columns if c.startswith("region_")]
oh_sums = encoded[oh_cols].sum().sort_values(ascending=False)
axes[1, 1].barh([c.replace("region_", "") for c in oh_sums.index], oh_sums.values, color="#17becf")
axes[1, 1].set_title("One-hot region column totals")
axes[1, 1].set_xlabel("Count")

fig.suptitle("Member 2 — Categorical encoding EDA", fontsize=14)
fig.tight_layout()
fig.savefig(VIZ / "m2_categorical_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Interpretation: encoding preserves class/region structure while making columns numeric for downstream scaling and modelling.")
'''
            ),
            md(
                """## Viva talking points
1. Contrast label encoding vs one-hot encoding.
2. Justify mapping Healthy→0, Unhealthy→1 from folder names.
3. Interpret the region / class×region comparison charts.
"""
            ),
        ],
    ),
    (
        "ITXXXXXX3_Outlier_Removal.ipynb",
        "Member 3 — Outlier Removal",
        "Outlier & duplicate removal",
        [
            md(
                """# Member 3 — Outlier Removal

**Technique:** Remove exact duplicate images and resolution/brightness outliers using the IQR rule.

## Why this dataset needs it
Duplicate leaves leak into both train and test if not removed. Extreme tiny/huge or near-black frames are often capture artefacts and distort color histograms.
"""
            ),
            code(SETUP),
            md("## 0. Process diagram — where this step sits in the pipeline"),
            code(
                r'''
fig, _ = draw_process_flow(
    PIPELINE_STEPS,
    title="Preprocessing pipeline (Member 3 highlighted)",
    highlight=2,
)
fig.savefig(VIZ / "m3_process_flow.png", dpi=150, bbox_inches="tight")
plt.show()
'''
            ),
            md("## 1. Start from encoded metadata / audit, drop duplicates & IQR outliers"),
            code(
                r'''
src = OUT / "m2_encoded_metadata.csv"
if src.exists():
    df = pd.read_csv(src)
else:
    df, _ = audit_images(RAW)
    print("Warning: m2 output missing; using fresh audit.")

before = len(df)
df_unique, n_dup = remove_exact_duplicates(df)
print(f"Exact pixel duplicates removed: {n_dup} (kept {len(df_unique)} / {before})")

# IQR on resolution and green-channel brightness (basil is green-dominant)
mask_px = iqr_mask(df_unique["pixels"], k=1.5)
mask_g = iqr_mask(df_unique["mean_g"], k=1.5)
keep = mask_px & mask_g
removed = df_unique.loc[~keep].copy()
clean = df_unique.loc[keep].reset_index(drop=True)

print(f"IQR outliers removed: {len(removed)}")
print(f"Final cleaned rows: {len(clean)}")
print(clean["label"].value_counts())

clean.to_csv(OUT / "m3_cleaned_no_outliers.csv", index=False)
removed.to_csv(OUT / "m3_removed_outliers.csv", index=False)
(LOGS / "m3_outlier_summary.json").write_text(
    json.dumps({
        "before": before,
        "duplicates_removed": int(n_dup),
        "iqr_removed": int(len(removed)),
        "after": int(len(clean)),
        "class_counts": {str(k): int(v) for k, v in clean["label"].value_counts().items()},
    }, indent=2),
    encoding="utf-8",
)
'''
            ),
            md("## 2. EDA visualizations — boxplots, before/after counts, outlier scatter"),
            code(
                r'''
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

sns.boxplot(data=df_unique, x="label", y="pixels", hue="label", order=CLASSES, ax=axes[0, 0], palette=["#2ca02c", "#d62728"], legend=False)
axes[0, 0].set_title("Image resolution (pixels) by class")
axes[0, 0].set_ylabel("width × height")

sns.boxplot(data=df_unique, x="label", y="mean_g", hue="label", order=CLASSES, ax=axes[0, 1], palette=["#2ca02c", "#d62728"], legend=False)
axes[0, 1].set_title("Mean green intensity by class")
axes[0, 1].set_ylabel("Mean G (0–255)")

# Before / after pipeline counts (comparison bars)
stages = ["Audited", "After duplicates", "After IQR"]
stage_counts = [before, len(df_unique), len(clean)]
axes[1, 0].bar(stages, stage_counts, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
axes[1, 0].set_title("Sample count before vs after cleaning")
axes[1, 0].set_ylabel("Images")
for i, v in enumerate(stage_counts):
    axes[1, 0].text(i, v + 5, str(v), ha="center")

# Class counts before vs after (grouped comparison)
before_cls = df["label"].value_counts().reindex(CLASSES).fillna(0)
after_cls = clean["label"].value_counts().reindex(CLASSES).fillna(0)
x = np.arange(len(CLASSES))
w = 0.35
axes[1, 1].bar(x - w / 2, before_cls.values, width=w, label="Before", color="#7f7f7f")
axes[1, 1].bar(x + w / 2, after_cls.values, width=w, label="After clean", color=["#2ca02c", "#d62728"])
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(CLASSES)
axes[1, 1].set_title("Class counts before vs after (comparison)")
axes[1, 1].set_ylabel("Images")
axes[1, 1].legend()

fig.suptitle("Member 3 — Outlier / duplicate removal EDA", fontsize=14)
fig.tight_layout()
fig.savefig(VIZ / "m3_outlier_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()

# Extra: scatter of resolution vs green, mark removed outliers
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
plt.show()
print("Interpretation: points far outside the whiskers are candidates for IQR removal; both classes should still remain after cleaning.")
'''
            ),
            md(
                """## Viva talking points
1. Explain duplicate leakage risk.
2. Explain IQR fences on `pixels` and `mean_g`.
3. Interpret the before/after comparison bars and boxplots.
"""
            ),
        ],
    ),
    (
        "ITXXXXXX4_Normalization_Scaling.ipynb",
        "Member 4 — Normalization / Scaling",
        "Normalization / scaling",
        [
            md(
                """# Member 4 — Normalization / Scaling

**Technique:** Standardize handcrafted features with `StandardScaler` (zero mean, unit variance) and compare with Min–Max scaling.

## Why this dataset needs it
Color histogram bins and channel statistics sit on different numeric ranges. Distance-based and regularized models (KNN, SVM, logistic regression) are sensitive to unscaled features.
"""
            ),
            code(SETUP),
            md("## 0. Process diagram — where this step sits in the pipeline"),
            code(
                r'''
fig, _ = draw_process_flow(
    PIPELINE_STEPS,
    title="Preprocessing pipeline (Member 4 highlighted)",
    highlight=4,
)
fig.savefig(VIZ / "m4_process_flow.png", dpi=150, bbox_inches="tight")
plt.show()
'''
            ),
            code(
                r'''
from sklearn.preprocessing import StandardScaler, MinMaxScaler

clean_path = OUT / "m3_cleaned_no_outliers.csv"
if clean_path.exists():
    meta = pd.read_csv(clean_path)
else:
    meta, _ = audit_images(RAW)
    meta, _ = remove_exact_duplicates(meta)
    print("Warning: m3 output missing; using audited unique images.")

# Use a stratified sample for interactive demo speed (full run still OK on ~1k images)
sample = stratified_sample(meta, 120)

X, y, kept = extract_feature_matrix(RAW, sample)
print("Feature matrix:", X.shape, "| classes:", dict(zip(*np.unique(y, return_counts=True))))

scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)

# Persist for Members 5–6 / group pipeline
np.savez_compressed(
    OUT / "m4_scaled_features.npz",
    X_raw=X,
    X_standard=X_std,
    X_minmax=X_mm,
    y=y,
    feature_dim=np.array([X.shape[1]]),
)
kept.to_csv(OUT / "m4_feature_sample_index.csv", index=False)
print("Saved scaled feature matrices to results/outputs/m4_scaled_features.npz")
'''
            ),
            md("## EDA visualizations — before/after scaling & StandardScaler vs MinMax comparison"),
            code(
                r'''
feat_idx = 0
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

axes[0, 0].hist(X[:, feat_idx], bins=30, color="#1f77b4", alpha=0.85)
axes[0, 0].set_title(f"Raw feature[{feat_idx}]")
axes[0, 0].set_xlabel("Value")
axes[0, 0].set_ylabel("Frequency")

axes[0, 1].hist(X_std[:, feat_idx], bins=30, color="#ff7f0e", alpha=0.85)
axes[0, 1].set_title(f"StandardScaler feature[{feat_idx}]")
axes[0, 1].set_xlabel("Z-score")

axes[1, 0].hist(X_mm[:, feat_idx], bins=30, color="#2ca02c", alpha=0.85)
axes[1, 0].set_title(f"MinMaxScaler feature[{feat_idx}]")
axes[1, 0].set_xlabel("Scaled [0, 1]")

# Compare mean absolute feature scale across methods (bar comparison)
methods = ["Raw", "StandardScaler", "MinMaxScaler"]
mean_abs = [
    float(np.mean(np.abs(X))),
    float(np.mean(np.abs(X_std))),
    float(np.mean(np.abs(X_mm))),
]
stds = [
    float(np.mean(X.std(axis=0))),
    float(np.mean(X_std.std(axis=0))),
    float(np.mean(X_mm.std(axis=0))),
]
x = np.arange(len(methods))
w = 0.35
axes[1, 1].bar(x - w / 2, mean_abs, width=w, label="Mean |value|", color="#1f77b4")
axes[1, 1].bar(x + w / 2, stds, width=w, label="Mean feature std", color="#ff7f0e")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(methods, rotation=15)
axes[1, 1].set_title("Scaler comparison (bar plots)")
axes[1, 1].legend()

fig.suptitle("Member 4 — Normalization / scaling EDA", fontsize=14)
fig.tight_layout()
fig.savefig(VIZ / "m4_scaling_before_after.png", dpi=150, bbox_inches="tight")
plt.show()

print("Raw mean/std:", float(X[:, feat_idx].mean()), float(X[:, feat_idx].std()))
print("Scaled mean/std:", float(X_std[:, feat_idx].mean()), float(X_std[:, feat_idx].std()))
print("Interpretation: after StandardScaler, features centre near 0 with comparable variance — safer for distance-based learners.")
'''
            ),
            md(
                """## Viva talking points
1. Difference between StandardScaler and MinMaxScaler.
2. Why image-derived histograms need scaling.
3. Interpret the before/after histograms and scaler comparison bars.
"""
            ),
        ],
    ),
    (
        "ITXXXXXX5_Feature_Engineering.ipynb",
        "Member 5 — Feature Engineering",
        "Feature engineering",
        [
            md(
                """# Member 5 — Feature Engineering

**Technique:** Engineer handcrafted visual features from each leaf image (RGB + HSV histograms and channel statistics).

## Why this dataset needs it
Raw pixels are high-dimensional and noisy. Compact color descriptors summarise **health-related appearance** (greenness, discoloration) in a form classical models can learn from.
"""
            ),
            code(SETUP),
            md("## 0. Process diagram — where this step sits in the pipeline"),
            code(
                r'''
fig, _ = draw_process_flow(
    PIPELINE_STEPS,
    title="Preprocessing pipeline (Member 5 highlighted)",
    highlight=3,
)
fig.savefig(VIZ / "m5_process_flow.png", dpi=150, bbox_inches="tight")
plt.show()
'''
            ),
            code(
                r'''
from skimage.color import rgb2hsv
from PIL import Image

meta_path = OUT / "m3_cleaned_no_outliers.csv"
meta = pd.read_csv(meta_path) if meta_path.exists() else audit_images(RAW)[0]
sample = stratified_sample(meta, 100)

# Build a readable feature table (subset of engineered columns)
rows = []
for row in sample.to_dict("records"):
    im = read_rgb(RAW / row["path"])
    rgb = np.asarray(im.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    hsv = rgb2hsv(rgb)
    rows.append({
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
    })

feat_df = pd.DataFrame(rows)
display(feat_df.head())
feat_df.to_csv(OUT / "m5_engineered_color_features.csv", index=False)
print("Engineered feature rows:", len(feat_df))
'''
            ),
            md("## EDA visualizations — correlation, greenness & RGB channel comparisons"),
            code(
                r'''
num_cols = [c for c in feat_df.columns if c not in {"path", "label"}]
corr = feat_df[num_cols].corr()

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0, 0])
axes[0, 0].set_title("Correlation of engineered color features")

sns.boxplot(data=feat_df, x="label", y="rgb_g_mean", hue="label", order=CLASSES, palette=["#2ca02c", "#d62728"], ax=axes[0, 1], legend=False)
axes[0, 1].set_title("Mean green channel by class")

# Grouped bar: mean RGB channels by class
channel_means = feat_df.groupby("label")[["rgb_r_mean", "rgb_g_mean", "rgb_b_mean"]].mean().reindex(CLASSES)
channel_means.plot(kind="bar", ax=axes[1, 0], color=["#d62728", "#2ca02c", "#1f77b4"], rot=0)
axes[1, 0].set_title("RGB channel means by class (comparison)")
axes[1, 0].set_ylabel("Mean intensity (0–1)")
axes[1, 0].legend(title="Channel")

# HSV mean comparison by class
hsv_means = feat_df.groupby("label")[["hsv_h_mean", "hsv_s_mean", "hsv_v_mean"]].mean().reindex(CLASSES)
hsv_means.plot(kind="bar", ax=axes[1, 1], color=["#9467bd", "#e377c2", "#8c564b"], rot=0)
axes[1, 1].set_title("HSV channel means by class (comparison)")
axes[1, 1].set_ylabel("Mean value")
axes[1, 1].legend(title="Channel")

fig.suptitle("Member 5 — Feature engineering EDA", fontsize=14)
fig.tight_layout()
fig.savefig(VIZ / "m5_feature_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=feat_df, x="label", y="rgb_g_mean", hue="label", order=CLASSES, palette=["#2ca02c", "#d62728"], ax=ax, legend=False)
ax.set_title("Mean green channel by class (engineered feature)")
fig.tight_layout()
fig.savefig(VIZ / "m5_greenness_by_class.png", dpi=150, bbox_inches="tight")
plt.show()
print("Interpretation: correlation shows redundancy among features; green-channel stats often separate healthy vs unhealthy leaves.")
'''
            ),
            md(
                """## Viva talking points
1. Why engineer RGB/HSV summaries instead of raw pixels.
2. Walk through one feature (`rgb_g_mean`) and its meaning.
3. Interpret the correlation heatmap and RGB/HSV comparison bars.
"""
            ),
        ],
    ),
    (
        "ITXXXXXX6_Feature_Selection_PCA.ipynb",
        "Member 6 — Feature Selection & PCA",
        "Feature selection and dimension reduction",
        [
            md(
                """# Member 6 — Feature Selection & Dimension Reduction (PCA)

**Technique:** Select informative features (`SelectKBest`) and reduce dimensionality with **PCA**.

## Why this dataset needs it
Engineered histograms are correlated and moderately wide. PCA compresses signal for visualization and can reduce overfitting for small-to-medium image sets.
"""
            ),
            code(SETUP),
            md("## 0. Process diagram — where this step sits in the pipeline"),
            code(
                r'''
fig, _ = draw_process_flow(
    PIPELINE_STEPS,
    title="Preprocessing pipeline (Member 6 highlighted)",
    highlight=5,
)
fig.savefig(VIZ / "m6_process_flow.png", dpi=150, bbox_inches="tight")
plt.show()
'''
            ),
            code(
                r'''
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

npz_path = OUT / "m4_scaled_features.npz"
if npz_path.exists():
    data = np.load(npz_path, allow_pickle=True)
    X_raw, y = data["X_raw"], data["y"]
    print("Loaded Member 4 features:", X_raw.shape)
else:
    meta_path = OUT / "m3_cleaned_no_outliers.csv"
    meta = pd.read_csv(meta_path) if meta_path.exists() else audit_images(RAW)[0]
    sample = stratified_sample(meta, 120)
    X_raw, y, _ = extract_feature_matrix(RAW, sample)
    print("Computed features:", X_raw.shape)

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_std = StandardScaler().fit_transform(X_raw)
k = min(20, X_std.shape[1])
selector = SelectKBest(score_func=f_classif, k=k)
X_sel = selector.fit_transform(X_std, y_enc)
print(f"SelectKBest kept {k} / {X_std.shape[1]} features")

pca = PCA(n_components=min(10, X_sel.shape[1]), random_state=SEED)
X_pca = pca.fit_transform(X_sel)
print("Explained variance ratio (first 5):", np.round(pca.explained_variance_ratio_[:5], 4))
print("Cumulative variance (all kept PCs):", float(pca.explained_variance_ratio_.sum()))

np.savez_compressed(
    OUT / "m6_selected_pca_features.npz",
    X_selected=X_sel,
    X_pca=X_pca,
    y=y,
    explained_variance_ratio=pca.explained_variance_ratio_,
)
pd.DataFrame({
    "pc": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative": np.cumsum(pca.explained_variance_ratio_),
}).to_csv(OUT / "m6_pca_variance.csv", index=False)
'''
            ),
            md("## EDA visualizations — F-scores, PCA variance bars & scatter"),
            code(
                r'''
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# SelectKBest F-scores (top features)
scores = selector.scores_
top_idx = np.argsort(scores)[::-1][:15]
axes[0, 0].barh([f"f{i}" for i in top_idx[::-1]], scores[top_idx][::-1], color="#1f77b4")
axes[0, 0].set_title("SelectKBest ANOVA F-scores (top 15)")
axes[0, 0].set_xlabel("F-score")

# PCA explained variance — bar + cumulative line
pcs = np.arange(1, len(pca.explained_variance_ratio_) + 1)
axes[0, 1].bar(pcs, pca.explained_variance_ratio_, color="#ff7f0e", label="Per-PC ratio")
axes[0, 1].plot(pcs, np.cumsum(pca.explained_variance_ratio_), marker="o", color="#d62728", label="Cumulative")
axes[0, 1].axhline(0.9, color="gray", ls="--", lw=1, label="90%")
axes[0, 1].set_xlabel("Principal component")
axes[0, 1].set_ylabel("Explained variance")
axes[0, 1].set_title("PCA explained variance (bars + cumulative)")
axes[0, 1].legend(fontsize=8)

# 2D scatter
for cls, color in zip(CLASSES, ["#2ca02c", "#d62728"]):
    mask = y == cls
    axes[1, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], s=18, alpha=0.7, label=cls, c=color)
axes[1, 0].set_xlabel("PC1")
axes[1, 0].set_ylabel("PC2")
axes[1, 0].set_title("PCA scatter (SelectKBest → PCA)")
axes[1, 0].legend()

# Dimensionality funnel comparison bars
dims = ["Raw features", "After SelectKBest", "After PCA"]
dim_vals = [X_std.shape[1], X_sel.shape[1], X_pca.shape[1]]
axes[1, 1].bar(dims, dim_vals, color=["#7f7f7f", "#1f77b4", "#2ca02c"])
axes[1, 1].set_title("Feature dimensionality reduction (comparison)")
axes[1, 1].set_ylabel("Dimensions")
axes[1, 1].tick_params(axis="x", rotation=15)
for i, v in enumerate(dim_vals):
    axes[1, 1].text(i, v + 0.5, str(v), ha="center")

fig.suptitle("Member 6 — Feature selection & PCA EDA", fontsize=14)
fig.tight_layout()
fig.savefig(VIZ / "m6_pca_variance_and_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Interpretation: if Healthy/Unhealthy form soft clusters in PC1–PC2, engineered color features carry separable signal after selection/reduction.")
'''
            ),
            md(
                """## Viva talking points
1. Role of SelectKBest vs PCA (filter selection vs rotation/compression).
2. Read the F-score bars and cumulative variance curve.
3. Interpret class separation on the PC1–PC2 scatter.
"""
            ),
        ],
    ),
]


group_cells = [
    md(
        """# Group Pipeline — Integrated Preprocessing & EDA

This notebook **combines Members 1–6** into one logical flow for Progress Review I (group component).

**Order:** Missing data → Categorical encoding → Outlier/duplicate removal → Feature engineering → Scaling → Feature selection / PCA → save artefacts.

All process diagrams, comparison bar plots, boxplots, heatmaps, and PCA charts are generated here and under each member notebook.
"""
    ),
    code(SETUP),
    md("## 0. End-to-end process diagram"),
    code(
        r'''
fig, _ = draw_process_flow(
    PIPELINE_STEPS,
    title="Group preprocessing pipeline — full process flow",
    highlight=None,
)
fig.savefig(VIZ / "group_process_flow.png", dpi=150, bbox_inches="tight")
plt.show()
'''
    ),
    md("## Step 1–3 — Inventory, audit, encode, clean"),
    code(
        r'''
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# 1) Missing / corrupt handling
inv = inventory_table(RAW)
discovered = discover_images(RAW)
valid, rejected = audit_images(RAW, discovered)
print("Inventory:"); display(inv)
print(f"Valid={len(valid)} | Rejected={len(rejected)}")

# 2) Categorical encoding
label_map = {c: i for i, c in enumerate(CLASSES)}
regions = sorted(valid["region"].unique().tolist())
region_map = {r: i for i, r in enumerate(regions)}
valid = valid.copy()
valid["label_encoded"] = valid["label"].map(label_map).astype(int)
valid["region_encoded"] = valid["region"].map(region_map).astype(int)

# 3) Duplicates + IQR outliers
unique, n_dup = remove_exact_duplicates(valid)
mask = iqr_mask(unique["pixels"]) & iqr_mask(unique["mean_g"])
clean = unique.loc[mask].reset_index(drop=True)
print(f"Duplicates removed={n_dup} | IQR removed={(~mask).sum()} | Clean={len(clean)}")
print(clean["label"].value_counts())
'''
    ),
    md("## Cleaning funnel & class / region comparison charts"),
    code(
        r'''
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Funnel of sample counts through early steps
funnel_labels = ["Discovered", "Valid", "Unique", "Clean"]
funnel_vals = [len(discovered), len(valid), len(unique), len(clean)]
axes[0, 0].bar(funnel_labels, funnel_vals, color=["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"])
axes[0, 0].set_title("Cleaning funnel (bar comparison)")
axes[0, 0].set_ylabel("Images")
for i, v in enumerate(funnel_vals):
    axes[0, 0].text(i, v + 5, str(v), ha="center")

# Expected vs found
x = np.arange(len(inv))
w = 0.35
short = [f.replace("_Region_Basil_Plant_Healthy", "").replace("Basil_Plant_", "") for f in inv["folder"]]
axes[0, 1].bar(x - w / 2, inv["expected"], width=w, label="Expected", color="#7f7f7f")
axes[0, 1].bar(x + w / 2, inv["found"], width=w, label="Found", color="#1f77b4")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(short, rotation=25, ha="right")
axes[0, 1].set_title("Expected vs found by folder")
axes[0, 1].legend()

# Clean class counts
class_counts = clean["label"].value_counts().reindex(CLASSES).fillna(0)
axes[1, 0].bar(class_counts.index, class_counts.values, color=["#2ca02c", "#d62728"])
axes[1, 0].set_title("Clean class counts")
axes[1, 0].set_ylabel("Count")

# Class × region after cleaning
ct = pd.crosstab(clean["region"], clean["label"]).reindex(columns=CLASSES).fillna(0)
ct.plot(kind="bar", ax=axes[1, 1], color=["#2ca02c", "#d62728"], rot=20)
axes[1, 1].set_title("Clean class × region comparison")
axes[1, 1].set_ylabel("Count")

fig.suptitle("Group pipeline — inventory & cleaning comparisons", fontsize=14)
fig.tight_layout()
fig.savefig(VIZ / "group_cleaning_comparisons.png", dpi=150, bbox_inches="tight")
plt.show()
'''
    ),
    md("## Step 4–6 — Features, scaling, selection, PCA"),
    code(
        r'''
# Stratified sample keeps the group demo responsive; set max_per_class=None for full data
max_per_class = 150
sample = clean if max_per_class is None else stratified_sample(clean, max_per_class)

X, y, kept = extract_feature_matrix(RAW, sample)
X_scaled = StandardScaler().fit_transform(X)
y_enc = LabelEncoder().fit_transform(y)

selector = SelectKBest(f_classif, k=min(20, X_scaled.shape[1]))
X_sel = selector.fit_transform(X_scaled, y_enc)
pca = PCA(n_components=min(5, X_sel.shape[1]), random_state=SEED)
X_pca = pca.fit_transform(X_sel)

print("Raw features:", X.shape)
print("After SelectKBest:", X_sel.shape)
print("After PCA:", X_pca.shape)
print("PCA variance:", np.round(pca.explained_variance_ratio_, 4))
'''
    ),
    md("## Feature / scaling / PCA comparison visualizations"),
    code(
        r'''
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

sns.boxplot(data=unique, x="label", y="mean_g", hue="label", order=CLASSES, ax=axes[0, 0], palette=["#2ca02c", "#d62728"], legend=False)
axes[0, 0].set_title("Green intensity (outlier view)")

axes[0, 1].hist(X[:, 0], bins=25, alpha=0.55, label="raw")
axes[0, 1].hist(X_scaled[:, 0], bins=25, alpha=0.55, label="scaled")
axes[0, 1].set_title("Feature[0] before/after scaling")
axes[0, 1].legend()

# PCA variance bars
pcs = np.arange(1, len(pca.explained_variance_ratio_) + 1)
axes[1, 0].bar(pcs, pca.explained_variance_ratio_, color="#ff7f0e", label="Per-PC")
axes[1, 0].plot(pcs, np.cumsum(pca.explained_variance_ratio_), "o-", color="#d62728", label="Cumulative")
axes[1, 0].set_title("PCA explained variance (comparison)")
axes[1, 0].set_xlabel("PC")
axes[1, 0].legend(fontsize=8)

for cls, color in zip(CLASSES, ["#2ca02c", "#d62728"]):
    m = y == cls
    axes[1, 1].scatter(X_pca[m, 0], X_pca[m, 1], s=16, alpha=0.75, c=color, label=cls)
axes[1, 1].set_title("PCA scatter after selection")
axes[1, 1].legend()

fig.suptitle("Group pipeline — feature & PCA EDA", fontsize=14)
fig.tight_layout()
fig.savefig(VIZ / "group_feature_pca_comparisons.png", dpi=150, bbox_inches="tight")
plt.show()

# Dimensionality reduction comparison
fig, ax = plt.subplots(figsize=(7, 4))
dims = ["Raw features", "SelectKBest", "PCA"]
dim_vals = [X.shape[1], X_sel.shape[1], X_pca.shape[1]]
ax.bar(dims, dim_vals, color=["#7f7f7f", "#1f77b4", "#2ca02c"])
ax.set_title("Dimensionality reduction funnel")
ax.set_ylabel("Feature dimensions")
for i, v in enumerate(dim_vals):
    ax.text(i, v + 0.5, str(v), ha="center")
fig.tight_layout()
fig.savefig(VIZ / "group_dimension_funnel.png", dpi=150, bbox_inches="tight")
plt.show()
'''
    ),
    md("## Save group artefacts + summary EDA panel"),
    code(
        r'''
# Tables / matrices
inv.to_csv(OUT / "group_folder_inventory.csv", index=False)
clean.to_csv(OUT / "group_cleaned_metadata.csv", index=False)
kept.assign(label_encoded=[label_map[v] for v in y]).to_csv(OUT / "group_feature_index.csv", index=False)
pd.DataFrame(X_scaled).to_csv(OUT / "group_features_scaled.csv", index=False)
pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]).assign(label=y).to_csv(
    OUT / "group_features_pca.csv", index=False
)
np.savez_compressed(OUT / "group_features.npz", X_scaled=X_scaled, X_pca=X_pca, y=y)

summary = {
    "discovered": int(len(discovered)),
    "valid": int(len(valid)),
    "rejected": int(len(rejected)),
    "duplicates_removed": int(n_dup),
    "clean": int(len(clean)),
    "feature_rows": int(X.shape[0]),
    "feature_dim": int(X.shape[1]),
    "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    "label_map": label_map,
    "region_map": region_map,
}
(LOGS / "group_pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# Combined EDA figure for viva walkthrough
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
class_counts = clean["label"].value_counts().reindex(CLASSES).fillna(0)
axes[0, 0].bar(class_counts.index, class_counts.values, color=["#2ca02c", "#d62728"])
axes[0, 0].set_title("1) Clean class counts")

sns.boxplot(data=unique, x="label", y="mean_g", hue="label", order=CLASSES, ax=axes[0, 1], palette=["#2ca02c", "#d62728"], legend=False)
axes[0, 1].set_title("2) Green intensity (outlier view)")

axes[1, 0].hist(X[:, 0], bins=25, alpha=0.55, label="raw")
axes[1, 0].hist(X_scaled[:, 0], bins=25, alpha=0.55, label="scaled")
axes[1, 0].set_title("3) Feature[0] before/after scaling")
axes[1, 0].legend()

for cls, color in zip(CLASSES, ["#2ca02c", "#d62728"]):
    m = y == cls
    axes[1, 1].scatter(X_pca[m, 0], X_pca[m, 1], s=16, alpha=0.75, c=color, label=cls)
axes[1, 1].set_title("4) PCA scatter after selection")
axes[1, 1].legend()

fig.suptitle("Group preprocessing pipeline — EDA summary", fontsize=14)
fig.tight_layout()
fig.savefig(VIZ / "group_pipeline_eda_summary.png", dpi=150, bbox_inches="tight")
plt.show()

print("Group pipeline complete.")
print("Visualizations →", VIZ)
print("Outputs →", OUT)
print("Logs →", LOGS)
'''
    ),
    md(
        """## Collaboration checklist
- [x] Member techniques integrated in dependency order  
- [x] Shared helpers in `src/preprocess_utils.py`  
- [x] Process-flow diagrams for each step  
- [x] Comparison bar plots, boxplots, heatmaps, PCA charts under `results/eda_visualizations/`  
- [x] Processed outputs under `results/outputs/`  
- [x] Run log under `results/logs/`  
"""
    ),
]


def write_nb(path: Path, cells: list[dict]) -> None:
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    print("Wrote", path.relative_to(ROOT))


def main() -> None:
    for filename, _, _, cells in member_specs:
        write_nb(NB_DIR / filename, cells)
    write_nb(ROOT / "group_pipeline.ipynb", group_cells)
    print("Done.")


if __name__ == "__main__":
    main()
