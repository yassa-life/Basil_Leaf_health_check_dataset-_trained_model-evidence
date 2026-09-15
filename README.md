# Basil Leaf Health Check & ML Quality Classifier

An end-to-end Machine Learning pipeline and local web application for classifying Basil leaf health (`Healthy` vs `Unhealthy`) using handcrafted visual feature extraction and trained classifiers.

---

## 🌿 Project Overview

This project provides a complete Machine Learning workflow and an interactive local Flask web interface for evaluating the health of basil plant leaves. 

- **Interactive Web Dashboard**: Upload leaf photos to get instant health predictions.
- **Handcrafted Visual Feature Pipeline**: Combines color distribution (RGB & HSV), texture analysis (Local Binary Patterns - LBP), and edge/shape features (Histogram of Oriented Gradients - HOG).
- **Rigorous Model Evaluation**: Evaluated using grouped 3-fold cross-validation and an independent holdout dataset.

---

## 📊 Dataset & Model Performance

### Dataset Information
- **Source**: [IEEE DataPort - Leaves - India's Most Famous Basil Plant Leaves Quality Dataset](https://ieee-dataport.org/open-access/leaves-indias-most-famous-basil-plant-leaves-quality-dataset) (DOI: `10.21227/a4f6-4413`)
- **Classes**: `Healthy` (643 unique) & `Unhealthy` (481 unique) after audit/dedupe
- **Listed images**: 1,131 in the four labelled folders; **1,124 unique** after removing 7 exact duplicates
- Legacy short folders (`Amravati`, `Nagpur`, `Pune`, `Bad`) are ignored for labelling

### Model Evaluation Highlights
The best classical model selected by grouped cross-validation is a **Random Forest Classifier** (98.22% holdout accuracy). A pixel **CNN** in `CNN/` reaches **98.67%** accuracy / **0.9864** macro F1 on the same holdout split.

| Metric | Random Forest | CNN |
| :--- | ---: | ---: |
| **Accuracy** | 98.22% | **98.67%** |
| **Macro F1-Score** | 0.9817 | **0.9864** |
| **95% Bootstrap CI (Macro F1)** | [0.9593, 0.9957] | [0.9686, 1.0000] |
| **Holdout** | 225 test / 899 train | 225 test / 599 train / 300 val |

#### Classification Report (Holdout Test Set)
See `outputs/results.json` for the latest classical-model per-class scores, and `CNN/results/RESULTS.md` for the CNN holdout report.

---

## 📁 Repository Structure

```
├── app.py                         # Flask web application & API backend
├── Basil_Leaf_ML_Workflow.ipynb   # Complete ML training & data audit notebook
├── CNN/                           # CNN deep-learning model (notebook + outputs/analytics/results)
│   ├── CNN_Basil_Leaf_Training.ipynb
│   ├── train_cnn.py
│   ├── outputs/                   # cnn_model.pt
│   ├── analytics/                 # curves, confusion matrix, history
│   └── results/                   # metrics, predictions, comparison
├── parts/                         # Six-person model notebooks (each with outputs/)
│   ├── README.md
│   ├── _pipeline.py               # Shared data/feature helpers
│   ├── logistic_regression/
│   ├── svm/
│   ├── knn/
│   ├── decision_tree/
│   ├── random_forest/
│   └── gradient_boosting/
├── start_web.bat                  # One-click Windows launch script
├── .gitignore
├── templates/ / static/           # Web dashboard
├── outputs/                       # Main notebook model + reports for the web app
└── data/raw/                      # Labelled image folders
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Environment Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/yassa-life/Basil_Leaf_health_check_dataset-_trained_model-evidence.git
cd Basil_Leaf_health_check_dataset-_trained_model-evidence

python -m venv .venv
```

Activate the virtual environment:

- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```

Install the required dependencies:
```bash
pip install flask pillow joblib numpy scikit-image scikit-learn waitress
```

To train the CNN model as well:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python CNN/train_cnn.py
```
Or open `CNN/CNN_Basil_Leaf_Training.ipynb`. Metrics and plots are written to `CNN/analytics/` and `CNN/results/`.

---

## 💻 Running the Web Application

### Option A: Using the Windows Batch File
Double-click `start_web.bat` or run in CMD/PowerShell:
```cmd
start_web.bat
```

### Option B: Running via Python
```bash
python app.py
```

Once started, open your web browser and navigate to:
👉 **[http://127.0.0.1:8001](http://127.0.0.1:8001)**

---

## ⚠️ Disclaimer

This application is an **experimental machine learning quality classification model** created for research and demonstration purposes. It does not replace professional agricultural diagnosis or food-safety evaluations. Unrelated non-leaf images cannot be reliably rejected by the current model.

---

## 📄 License

This repository is shared for academic and research purposes. Dataset rights belong to the original authors on IEEE DataPort.
