# Progress Review I — Data Preprocessing & EDA

**Module:** IT2011 – Artificial Intelligence and Machine Learning  
**Focus:** Data cleaning, preprocessing, and exploratory data analysis  
**Dataset:** Basil leaf quality (Healthy vs Unhealthy)

> Before Courseweb zip upload, rename this folder from `Group_Deliverable` to your official **Group_ID**, and rename each notebook’s `ITXXXXXXn` prefix to the member’s real **IT number**.

---

## Assigned dataset

| Item | Detail |
|---|---|
| Source | [IEEE DataPort – Basil Plant Leaves Quality Dataset](https://ieee-dataport.org/open-access/leaves-indias-most-famous-basil-plant-leaves-quality-dataset) |
| DOI | `10.21227/a4f6-4413` |
| Classes | `Healthy`, `Unhealthy` |
| Folders used | `Amravati_Region_Basil_Plant_Healthy`, `Nagpur_Region_Basil_Plant_Healthy`, `Pune_Region_Basil_Plant_Healthy`, `Basil_Plant_Unhealthy` |
| Not used for labels | Legacy short folders `Amravati`, `Nagpur`, `Pune`, `Bad` |

Place the four labelled folders under `data/raw/` (this deliverable already junctions to the repo’s `data/raw/` when run from the main project).

---

## Repository layout (as required)

```
Group_ID/
├── README.md
├── data/
│   ├── raw/                 # assigned dataset
│   └── external/            # optional external references
├── notebooks/
│   ├── ITXXXXXX1_Missing_Data.ipynb
│   ├── ITXXXXXX2_Categorical_Encoding.ipynb
│   ├── ITXXXXXX3_Outlier_Removal.ipynb
│   ├── ITXXXXXX4_Normalization_Scaling.ipynb
│   ├── ITXXXXXX5_Feature_Engineering.ipynb
│   └── ITXXXXXX6_Feature_Selection_PCA.ipynb
├── group_pipeline.ipynb     # integrated preprocessing pipeline
├── results/
│   ├── eda_visualizations/  # PNG charts
│   ├── logs/                # optional run logs
│   └── outputs/             # processed tables / feature matrices
└── src/
    └── preprocess_utils.py  # shared helpers
```

---

## Group member roles (individual component)

Each member owns **one preprocessing technique**, explains why it is needed for this image dataset, shows code/output, and presents **at least one EDA visualization**.

| Member | Notebook | Technique | Primary EDA plot |
|---|---|---|---|
| 1 | `ITXXXXXX1_Missing_Data.ipynb` | Handling missing / corrupt data | Class & completeness bar chart |
| 2 | `ITXXXXXX2_Categorical_Encoding.ipynb` | Encoding categorical variables | Encoded label / region counts |
| 3 | `ITXXXXXX3_Outlier_Removal.ipynb` | Outlier & duplicate removal | Boxplot of image size / brightness |
| 4 | `ITXXXXXX4_Normalization_Scaling.ipynb` | Normalization / scaling | Feature distribution before vs after scale |
| 5 | `ITXXXXXX5_Feature_Engineering.ipynb` | Feature engineering (color/texture) | Feature correlation heatmap |
| 6 | `ITXXXXXX6_Feature_Selection_PCA.ipynb` | Feature selection & PCA | PCA scatter / explained variance |

**Group component:** run `group_pipeline.ipynb` — it chains all six steps into one commented pipeline and writes final artefacts under `results/`.

---

## How we applied each preprocessing part (image dataset)

This dataset is **folders of leaf photos**, not a spreadsheet of NaNs. Each member technique is mapped to that reality. Shared helpers live in `src/preprocess_utils.py`. The group notebook runs the same steps in this order:

**Missing data → categorical encoding → outlier / duplicate removal → feature engineering → scaling → SelectKBest + PCA**

### Member 1 — Missing / corrupt data

**What we did:** Count files in the four labelled folders against expected sizes, then try to **decode** every image with Pillow (`read_rgb`). A file that is absent or fails to open is treated as missing (not a CSV `fillna`).

**Why:** Training would crash or silently skip bad JPEGs. Completeness and class counts are plotted before later steps.

**Result (last audit):** 1,131 listed images, **1,131 readable**, **0 rejected**.

### Member 2 — Categorical encoding

**What we did:** Derive categories from folder names:

| Field | Values | Encoding |
|---|---|---|
| `label` | Healthy, Unhealthy | Integer map `Healthy→0`, `Unhealthy→1` |
| `region` | Amravati, Nagpur, Pune, Unknown | Integer codes **and** one-hot columns (`pd.get_dummies`) |

**Why:** Models and correlation plots need numbers. Region is a capture-site category, not a pixel.

**sklearn / pandas:** dictionary maps + `pandas.get_dummies` (no `OneHotEncoder` object; same idea).

### Member 3 — Outlier and duplicate removal

**What we did:**

1. Drop **exact pixel duplicates** using a SHA-256 hash of RGB bytes (`pixel_sha256`).
2. Apply the **IQR rule** (\(k = 1.5\)) on **image size** (`width × height`) and **mean green** brightness (basil is green-dominant). Rows outside the fences are removed.

**Why:** Duplicate leaves leak into train and test. Extreme tiny/huge or near-black frames distort color histograms.

**Result (last run):** **7** duplicates removed, **48** IQR outliers removed, **1,076** images kept.

### Member 5 — Feature engineering (before scaling)

**What we did:** Resize each leaf to **64×64**, convert to RGB and HSV (`skimage.color.rgb2hsv`), then build a compact vector:

- 8-bin histograms per RGB and HSV channel (normalized)
- per-channel mean and standard deviation

That yields **60 numeric features** per image (color appearance: greenness, discoloration), not raw pixels.

**Why:** Pixels are huge and noisy. Handcrafted color stats are what we scale and reduce in Members 4 and 6.

Member notebooks 4 and 6 use a **stratified sample** (about 100–150 images per class) so demos stay interactive; the group pipeline uses the same extractors.

### Member 4 — Normalization / scaling

**What we did:** Fit sklearn scalers on the engineered matrix `X`:

| Method | sklearn class | Role |
|---|---|---|
| **Z-score (primary)** | `StandardScaler` | Mean 0, variance 1 — used by the group pipeline |
| **Min–Max (comparison)** | `MinMaxScaler` | Stretch each feature to \[0, 1\] |

**Why:** Histogram bins and channel stats sit on different ranges. Distance-based models (KNN, SVM) and PCA are sensitive to unscaled features. We scale **features**, not JPEG files.

EDA: histogram of one feature **before vs after** `StandardScaler`.

### Member 6 — Feature selection and PCA

**What we did (sklearn):**

1. `LabelEncoder` on class names (Healthy / Unhealthy).
2. `SelectKBest(score_func=f_classif, k=20)` — keep the 20 features most associated with the class (ANOVA F-test).
3. `PCA` — compress to **10** components in the member notebook, **5** in `group_pipeline.ipynb` (random state 42). Plot cumulative explained variance and a **PC1–PC2 scatter** coloured by class.

**Why:** Histogram features are correlated. Selection drops weak columns; PCA rotates remaining signal for visualization and a smaller matrix.

These are **unsupervised / filter tools**, not a trained classifier. They prepare data for later models.

---

## Methods and models used in this deliverable

Progress Review I is **preprocessing + EDA**. This folder does **not** train Random Forest or CNN (those live in the parent repo). Objects we actually use:

| Step | Library / method |
|---|---|
| Load / validate images | Pillow (`Image.open`, RGB convert, EXIF transpose) |
| Tables and plots | pandas, matplotlib, seaborn |
| Duplicates | SHA-256 of pixel bytes |
| Outliers | Tukey IQR fences on resolution and mean green |
| Color features | NumPy histograms + `skimage.color.rgb2hsv` |
| Scaling | `sklearn.preprocessing.StandardScaler`, `MinMaxScaler` |
| Class codes | integer maps / `LabelEncoder` |
| Feature selection | `sklearn.feature_selection.SelectKBest` + `f_classif` |
| Dimension reduction | `sklearn.decomposition.PCA` |

**Intended next step (parent project, not required for this zip):** the same cleaned images / color features feed classical classifiers (for example Random Forest) and a pixel CNN. This review only needs the preprocessing chain and EDA plots.

---

## How to run

### 1. Environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r progress\Group_Deliverable\requirements.txt
```

### 2. Open notebooks

Start Jupyter from the deliverable folder (or open notebooks in Cursor/VS Code with the `.venv` kernel):

```powershell
cd progress\Group_Deliverable
jupyter notebook
```

Recommended order:

1. Run each member notebook once (Member 1 → 6)  
2. Run `group_pipeline.ipynb` for the integrated deliverable  

Outputs are saved to:

- `results/eda_visualizations/*.png`
- `results/outputs/*.csv` / `*.npz`
- `results/logs/*.txt` (optional)

### 3. Zip for Courseweb

1. Rename `Group_Deliverable` → your `Group_ID`  
2. Rename notebook prefixes to real IT numbers  
3. Zip the folder and upload  

---

## Design notes for viva

- Argue **image-specific** definitions (files vs NaNs; folder labels vs CSV categories).
- Member 4 and 6 operate on **engineered features**, not raw pixels.
- PCA / SelectKBest are **not** the classification model; they are reduction steps after cleaning.

---

## Disclaimer

Academic Progress Review deliverable only. Not a medical or food-safety diagnostic tool.
