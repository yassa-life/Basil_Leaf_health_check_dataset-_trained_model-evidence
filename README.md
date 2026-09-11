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
- **Classes**: `Healthy` (458 samples) & `Unhealthy` (439 samples)
- **Total Valid Unique Images**: 897 images across 759 perceptual groups.

### Model Evaluation Highlights
The best performing model selected by cross-validation is a **Random Forest Classifier**:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 95.56% |
| **Balanced Accuracy** | 95.45% |
| **Macro F1-Score** | 0.9554 |
| **95% Bootstrap CI (Macro F1)** | [0.9221, 0.9833] |
| **Inference Speed** | ~0.39 ms / image |

#### Classification Report (Holdout Test Set - 180 Images)
- **Healthy**: Precision `0.92` | Recall `1.00` | F1-Score `0.96` (92 samples)
- **Unhealthy**: Precision `1.00` | Recall `0.91` | F1-Score `0.95` (88 samples)

---

## 📁 Repository Structure

```
├── app.py                         # Flask web application & API backend
├── Basil_Leaf_ML_Workflow.ipynb   # Complete ML training & data audit notebook
├── parts/                          # Six-person sequential notebook assignment
│   ├── README.md                   # Responsibilities, order, and handoff rules
│   ├── person_1_data_collection.ipynb
│   ├── person_2_data_preprocessing.ipynb
│   ├── person_3_feature_engineering.ipynb
│   ├── person_4_model_selection.ipynb
│   ├── person_5_training_evaluation.ipynb
│   ├── person_6_deployment_reporting.ipynb
│   └── artifacts/                  # Outputs passed between the six notebooks
├── start_web.bat                  # One-click Windows launch script
├── .gitignore                     # Git ignore rules
├── templates/
│   └── index.html                 # Web dashboard UI
├── static/
│   ├── app.js                     # Frontend interaction logic
│   └── style.css                  # UI styles
├── outputs/
│   ├── model.joblib               # Exported trained Random Forest pipeline
│   ├── results.json               # Detailed metrics & metadata
│   ├── dataset_manifest.csv       # Processed dataset file manifest
│   ├── split_counts.csv           # Train/test split summary
│   └── cv_fold_counts.csv         # Grouped CV fold metrics
└── data/
    └── raw/                       # Image dataset folders
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
