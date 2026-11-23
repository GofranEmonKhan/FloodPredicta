<div align="center">

# 🌊 FloodPredicta
### AI-Powered Multi-Station Flood Early Warning System for Bangladesh

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Ensemble-orange.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Framework](https://img.shields.io/badge/Toolkit-Pandas%20%7C%20NumPy%20%7C%20SciPy-blueviolet.svg?style=for-the-badge)](https://pandas.pydata.org/)
[![Evaluation](https://img.shields.io/badge/ROC--AUC-0.9958-brightgreen.svg?style=for-the-badge)](version2/EXPERIMENT_TRACKER.md)
[![Status](https://img.shields.io/badge/Release-Version%202.0%20(Production)-success.svg?style=for-the-badge)](version2/)
[![License: Citation Required](https://img.shields.io/badge/License-Citation%20Required-crimson.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>A 65-year empirical hydrological study (1948–2013) across 33 Bangladesh Meteorological Department (BMD) weather stations, transforming raw climate observations into actionable, high-recall flood catastrophe alerts without data loss or temporal leakage.</b>
</p>

[Key Features](#-key-features) •
[System Evolution (V1 vs V2)](#-system-evolution-version-1-vs-version-20) •
[Methodology](#-hydrological-modeling--methodology) •
[Benchmarks](#-benchmark-results) •
[Threshold Optimization](#-operational-threshold-optimization) •
[Visual Reports](#-visual-analytics--model-diagnostics) •
[Quickstart](#-quickstart-guide) •
[Repository Tree](#-repository-architecture)

</div>

---

## 📌 Overview & Context

Bangladesh sits at the confluence of the **Ganges, Brahmaputra, and Meghna (GBM)** river systems, creating the world's most vulnerable delta ecosystem. Over 80% of the landmass consists of low-lying floodplains with altitudes ranging from 2m to 35m above sea level. Severe monsoon deluges frequently inundate large swathes of the country, causing catastrophic displacement, agricultural destruction, and loss of life.

In this project, I analyzed 65 years of continuous station-level meteorological and spatial measurements (1948–2013; 20,544 monthly records) across **33 BMD weather stations** to predict impending flood disasters in advance.

In **Version 2.0**, I completely re-engineered the analytical pipeline from scratch. I resolved the severe data loss from Version 1, eliminated temporal lookahead bias, introduced catchment hydrology features, and calibrated decision thresholds specifically for real-world disaster management.

---

## 🚀 Key Features

* **100% Disaster Event Preservation:** I did not drop a single flood record. All **4,132 historical flood events** (including the historic 1988, 1998, and 2004 mega-floods) are fully preserved using 99th-percentile winsorization instead of aggressive outlier deletion.
* **Catchment Hydrology Features:** I engineered antecedent precipitation lags, 3-month monsoonal rolling rainfall, and runoff-to-altitude pooling ratios to reflect physical water accumulation.
* **Leakage-Free Temporal Validation:** I implemented walk-forward chronological splits (historical training -> validation tuning -> unseen future testing) to ensure my models genuinely generalize to future seasons.
* **Operational Threshold Calibration:** I tuned decision thresholds on the validation set to maximize disaster F1-Score and recall, overcoming Bangladesh's 80:20 non-flood vs. flood class imbalance.
* **5 Enterprise-Grade Tree Models:** I benchmarked XGBoost, HistGradientBoosting, Random Forest, Extra Trees, and a Stacking Ensemble with a Logistic Regression meta-learner.
* **Standalone Diagnostic Analytics:** I generated independent, publication-ready Confusion Matrices, ROC Curves, Probability Distributions, and Feature Importance bar charts for every model.

---

## 🔬 System Evolution: Version 1 vs. Version 2.0

| Component | Version 1 (Baseline Legacy) | Version 2.0 (My Redesigned Pipeline) | Scientific Impact in Version 2.0 |
| :--- | :--- | :--- | :--- |
| **Outlier Handling** | 5 successive passes of IQR trimming deleting **2,984 records**. | **Non-destructive 99th-percentile winsorization** and log transforms. | **Preserved 1,387 real flood cases** (+33.5% of all disasters) that V1 had discarded as "outliers". |
| **Data Splitting** | Random 80/20 train/test split across all 65 years. | **Walk-Forward Chronological Splitting** (e.g., Train <= 1997, Val 1998–2008, Test >= 2009). | Eliminates temporal lookahead bias; mimics true future disaster deployment. |
| **Feature Set** | 13 isolated single-month meteorological measurements. | **64 engineered features** incorporating catchment saturation and terrain geometry. | Captures multi-month soil saturation, upstream pooling, and seasonal Fourier cycles. |
| **Imbalance Strategy** | SMOTE applied to one-hot spatial dummy vectors. | Native algorithm weighting (`scale_pos_weight`, balanced bootstraps). | Eliminates synthetic decimal station nonsense (e.g. `Dhaka = 0.38`). |
| **Threshold Tuning** | Hardcoded default `0.50` cutoff. | Systematic validation grid sweep (0.10 to 0.85) maximizing disaster F1-Score. | Balances life-saving recall with minimization of costly false alarms. |
| **Deployment Code** | Dimension mismatch (`appRF.py` passed 13 features to 44-feature scaler). | Self-contained serialized pipelines and standalone notebooks. | 100% reproducible and crash-free execution. |

> [!IMPORTANT]
> **Why did Version 1 show a seemingly "higher" 98% accuracy?**  
> In Version 1, 1,387 severe flood deluges were deleted as "outliers", leaving behind only milder, easier-to-classify samples. When I evaluated Version 2 on the exact same shuffled split with all 4,132 extreme events included, **Version 2 significantly outperforms Version 1** across all detection metrics: **Flood Recall: 0.9774 vs 0.9435**, **Flood F1-Score: 0.9374 vs 0.9241**, and **ROC-AUC: 0.9958 vs 0.9946**.

---

## 🌊 Hydrological Modeling & Methodology

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FLOODPREDICTA V2 PIPELINE                              │
├─────────────────────┬───────────────────────┬───────────────────┬──────────────────────┤
│ 1. NON-DESTRUCTIVE  │ 2. CATCHMENT HYDROLOGY│ 3. TEMPORAL SPLIT │ 4. ENSEMBLE TRAINING │
│    PREPROCESSING    │    FEATURE ENGINEERING│    & CALIBRATION  │    & BENCHMARKING    │
├─────────────────────┼───────────────────────┼───────────────────┼──────────────────────┤
│ • 20,544 rows (100%)│ • Antecedent Lags     │ • 70/15/15 Shuff. │ • XGBoost (Champion) │
│ • 4,132 Floods kept │   (Lag-1, Lag-2 rain) │ • Chrono 70/15/15 │ • HistGradBoost      │
│ • 99th Winsorization│ • 3M Rolling Monsoons │ • Chrono 70/20/10 │ • Random Forest      │
│ • Log precipitation │ • Runoff-Pooling Index│ • Val Grid Search │ • Extra Trees        │
│ • Missing target fix│ • Fourier Seasonality │   (Max F1 @ t*)   │ • Stacking Classifier│
└─────────────────────┴───────────────────────┴───────────────────┴──────────────────────┘
```

### 1. Hydrological & Domain Features

Instead of looking at each month in isolation, I engineered features reflecting how real floods occur on the ground:

* **Antecedent Rainfall Lags:** I included rainfall from 1 month ago (Lag-1) and 2 months ago (Lag-2) to account for existing soil moisture before new rain arrives.
* **Cumulative Monsoonal Saturation:** I computed a 3-month rolling rainfall sum to track prolonged catchment water build-up across river basins.
* **Runoff-Pooling Index:** I calculated rainfall relative to station altitude to capture how water rapidly accumulates in low-lying delta and haor areas compared to higher ground.
* **Atmospheric Moisture Index:** I combined relative humidity and cloud coverage to detect persistent monsoon depressions that slow down evaporation.
* **Cyclical Seasonal Signals:** I converted month numbers into continuous cyclical sine and cosine curves, and added an explicit monsoon indicator (June to September).

---

## 📊 Benchmark Results

I implemented and evaluated three distinct partitioning regimes across all 5 models to establish complete scientific rigor:

### Regime 1: Stratified Shuffled Split (70% Train: 14,380 | 15% Val: 3,082 | 15% Test: 3,082)
*Benchmarking against Version 1's split methodology on the full, non-pruned dataset:*

| Model Name | Test Accuracy | Precision (Flood=1) | Recall (Flood=1) | F1-Score (Flood=1) | ROC-AUC | Artifact |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost (Champion)** | **0.9737** | **0.9004** | **0.9774** | **0.9374** | **0.9958** | [`xgboost_model.pkl`](version2/models/split_shuffled_70_15_15/xgboost_model.pkl) |
| **HistGradientBoosting** | 0.9734 | 0.8956 | **0.9823** | 0.9369 | 0.9957 | [`hist_gradient_boosting_model.pkl`](version2/models/split_shuffled_70_15_15/hist_gradient_boosting_model.pkl) |
| **Extra Trees** | **0.9744** | **0.9300** | 0.9435 | 0.9367 | 0.9949 | [`extra_trees_model.pkl`](version2/models/split_shuffled_70_15_15/extra_trees_model.pkl) |
| **Stacking Ensemble** | 0.9731 | 0.8966 | 0.9790 | 0.9360 | **0.9963** | [`stacking_ensemble_model.pkl`](version2/models/split_shuffled_70_15_15/stacking_ensemble_model.pkl) |
| **Random Forest** | 0.9734 | 0.9311 | 0.9371 | 0.9341 | 0.9960 | [`random_forest_model.pkl`](version2/models/split_shuffled_70_15_15/random_forest_model.pkl) |

---

### Regime 2: Chronological Split A (Train <= 2003: 16,656 | Val 2004–2008: 1,920 | Test >= 2009: 1,968)
*Historical 56-year baseline training, mid-2000s validation tuning, and 5-year future test set:*

| Model Name | Test Accuracy | Precision (Flood=1) | Recall (Flood=1) | F1-Score (Flood=1) | ROC-AUC | Artifact |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost (Champion)** | **0.9644** | **0.9127** | **0.9127** | **0.9127** | **0.9871** | [`xgboost_model.pkl`](version2/models/split_chronological_70_15_15/xgboost_model.pkl) |
| **Stacking Ensemble** | **0.9644** | **0.9127** | **0.9127** | **0.9127** | 0.9876 | [`stacking_ensemble_model.pkl`](version2/models/split_chronological_70_15_15/stacking_ensemble_model.pkl) |
| **HistGradientBoosting** | 0.9624 | 0.9057 | 0.9102 | 0.9080 | **0.9879** | [`hist_gradient_boosting_model.pkl`](version2/models/split_chronological_70_15_15/hist_gradient_boosting_model.pkl) |
| **Extra Trees** | 0.9619 | 0.9312 | 0.8778 | 0.9037 | 0.9846 | [`extra_trees_model.pkl`](version2/models/split_chronological_70_15_15/extra_trees_model.pkl) |
| **Random Forest** | 0.9614 | 0.9452 | 0.8603 | 0.9008 | 0.9843 | [`random_forest_model.pkl`](version2/models/split_chronological_70_15_15/random_forest_model.pkl) |

---

### Regime 3: Chronological Split B (70-20-10: Train <= 1997: 14,352 | Val 1998–2008: 4,224 | Test >= 2009: 1,968)
*Strict 70% historical baseline, 20% validation window, and 10% future holdout test set:*

| Model Name | Test Accuracy | Precision (Flood=1) | Recall (Flood=1) | F1-Score (Flood=1) | ROC-AUC | Artifact |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stacking Ensemble (Champion)** | **0.9649** | **0.9150** | **0.9127** | **0.9139** | 0.9859 | [`stacking_ensemble_model.pkl`](version2/models/split_chronological_70_20_10/stacking_ensemble_model.pkl) |
| **XGBoost** | 0.9639 | 0.9084 | **0.9152** | 0.9118 | 0.9866 | [`xgboost_model.pkl`](version2/models/split_chronological_70_20_10/xgboost_model.pkl) |
| **HistGradientBoosting** | 0.9619 | 0.8976 | **0.9177** | 0.9075 | **0.9878** | [`hist_gradient_boosting_model.pkl`](version2/models/split_chronological_70_20_10/hist_gradient_boosting_model.pkl) |
| **Random Forest** | 0.9629 | **0.9409** | 0.8728 | 0.9056 | 0.9814 | [`random_forest_model.pkl`](version2/models/split_chronological_70_20_10/random_forest_model.pkl) |
| **Extra Trees** | 0.9599 | 0.9259 | 0.8728 | 0.8986 | 0.9831 | [`extra_trees_model.pkl`](version2/models/split_chronological_70_20_10/extra_trees_model.pkl) |

---

## 🎯 Operational Threshold Optimization

In flood disaster forecasting, the costs of misclassification are heavily asymmetric:
* **False Negative (Missed Flood):** Extreme catastrophe — unevacuated communities, loss of human life, and severe livestock and crop loss.
* **False Positive (False Alarm):** Evacuation disruption, logistics expense, and public alert fatigue ("crying wolf").

Because non-flood months outnumber flood months roughly 4 to 1, the default **0.50** decision threshold is uncalibrated. 

To solve this, **I swept decision thresholds from 0.10 to 0.85 (in steps of 0.05)** on the independent validation set and selected the optimal cutoff (**t\***) that maximizes the positive flood **F1-Score**:

> **Optimization Objective:** `Optimal Threshold (t*) = argmax F1-Score (Flood = 1)`

```text
Threshold Tuning Operating Points:
├── Balanced Operational Mode (t* ≈ 0.45 - 0.50): Optimal F1-score (~0.93 - 0.94)
└── Life-Safety / Alert Mode  (t  ≈ 0.30 - 0.35): High recall (≥ 0.98) for extreme deluge warnings
```

---

## 📈 Visual Analytics & Model Diagnostics

Every trained model exports 4 publication-quality, standalone diagnostic figures:

<div align="center">

| Confusion Matrix Heatmap | ROC-AUC Discrimination Curve |
| :---: | :---: |
| <img src="version2/reports/split_shuffled_70_15_15/xgboost_model_confusion_matrix.png" width="400" alt="Confusion Matrix"/> | <img src="version2/reports/split_shuffled_70_15_15/xgboost_model_roc_curve.png" width="400" alt="ROC Curve"/> |

| Predicted Probability Distribution | Top 15 Feature Importances |
| :---: | :---: |
| <img src="version2/reports/split_shuffled_70_15_15/xgboost_model_probability_distribution.png" width="400" alt="Probability Distribution"/> | <img src="version2/reports/split_shuffled_70_15_15/xgboost_model_feature_importance.png" width="400" alt="Feature Importance"/> |

</div>

### Key Physical Insights Revealed by Feature Importance:
1. **`Rainfall log` & `Rainfall`:** Primary direct precipitation drivers.
2. **`Rainfall_Rolling3M`:** 3-month antecedent catchment saturation — ground permeability drops significantly after prolonged accumulation.
3. **`Runoff_Ratio` (`Rainfall / (ALT + 1)`):** Low-altitude coastal and depression stations (e.g., Sylhet haor basin) experience severe pooling even under moderate rainfall.
4. **`Cloud_Coverage` & `Relative_Humidity`:** Indicators of persistent low-pressure monsoonal troughs.

---

## 📂 Repository Architecture

```text
FloodPredicta/
├── LICENSE                                    # Attribution & Mandatory Citation License
├── PROJECT_REPORT.md                          # Comprehensive technical audit of Version 1
├── README.md                                  # Global interactive repository documentation
│
├── Dataset/                                   # Legacy Version 1 dataset files
│   ├── flooddata.csv                          # Original raw dataset
│   └── flood_data_preprocessed.csv            # V1 trimmed data (1,387 floods lost)
├── Model Train/                               # Legacy Version 1 model training notebooks
├── Model_Deployment/                          # Legacy Version 1 Flask serving app
│
└── version2/                                  # Production-Grade Version 2.0 System
    ├── README.md                              # V2 Master Redesign Plan & Strategy
    ├── EXPERIMENT_TRACKER.md                  # Detailed Experiment Chronicle & Threshold Audit
    │
    ├── data/
    │   ├── raw/
    │   │   └── flooddata.csv                  # Preserved raw BMD climate dataset (20,544 rows)
    │   └── processed/
    │       ├── flood_data_preprocessed.csv    # 100% flood preserved (20,544 rows × 20 cols)
    │       └── flood_data_featured.csv        # Engineered features (20,544 rows × 64 cols)
    │
    ├── notebooks/
    │   ├── data_preprocessing.ipynb           # Non-destructive preprocessing & winsorization
    │   ├── feature_engineering.ipynb          # Hydrological lags & runoff features
    │   ├── split_shuffled_70_15_15/           # 5 notebooks (XGB, HGB, RF, ET, Stacking)
    │   ├── split_chronological_70_15_15/      # 5 notebooks (Train <=2003, Val 04-08, Test 09-13)
    │   └── split_chronological_70_20_10/      # 5 notebooks (Train <=1997, Val 98-08, Test 09-13)
    │
    ├── models/
    │   ├── split_shuffled_70_15_15/           # 5 serialized .pkl models + feature_columns.pkl
    │   ├── split_chronological_70_15_15/      # 5 serialized .pkl models + feature_columns.pkl
    │   └── split_chronological_70_20_10/      # 5 serialized .pkl models + feature_columns.pkl
    │
    ├── reports/                               # 60 publication-ready standalone diagnostic plots
    │   ├── split_shuffled_70_15_15/           # 20 standalone plots (4 plots × 5 models)
    │   ├── split_chronological_70_15_15/      # 20 standalone plots (4 plots × 5 models)
    │   └── split_chronological_70_20_10/      # 20 standalone plots (4 plots × 5 models)
    │
    └── deployment/                            # Production deployment workspace
```

---

## ⚡ Quickstart Guide

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/GofranEmonKhan/FloodPredicta.git
cd FloodPredicta

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install numpy pandas scikit-learn xgboost scipy matplotlib seaborn joblib jupyter
```

### 2. Launch Jupyter & Run Notebooks
Launch Jupyter to explore or re-run any stage of the pipeline:
```bash
jupyter notebook version2/notebooks/
```

### 3. Execution Pipeline Sequence
All notebooks are self-contained. Execute them in sequential order:
1. **Data Preprocessing:** Run `version2/notebooks/data_preprocessing.ipynb` to generate `flood_data_preprocessed.csv` from raw climate observations with 100% flood event preservation.
2. **Feature Engineering:** Run `version2/notebooks/feature_engineering.ipynb` to calculate antecedent precipitation lags, monsoonal rolling accumulation, and runoff ratios into `flood_data_featured.csv`.
3. **Model Training & Evaluation:** Run any model notebook inside `version2/notebooks/split_.../` (e.g. `train_xgboost.ipynb` or `train_stacking_ensemble.ipynb`) to train the model, tune validation thresholds, and export diagnostic plots.

---

## 📝 Citation & Authorship

If you find this research or code helpful in your academic work or disaster management projects, please cite:

```bibtex
@software{khan2026floodpredicta,
  author       = {Gofran Emon Khan},
  title        = {FloodPredicta: AI-Powered Multi-Station Flood Early Warning System for Bangladesh},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/GofranEmonKhan/FloodPredicta}}
}
```

---

## 📜 License & Citation Policy

This project is licensed under the [FloodPredicta Attribution & Academic Research License](LICENSE).

> [!IMPORTANT]
> **Mandatory Citation Requirement:** You are welcome to use, explore, and build upon this code and methodology, but **you must give proper attribution and cite this repository and author** in any academic publication, thesis, report, software, or project. Using this project without citation is strictly prohibited.

<div align="center">
  <b>Developed with dedication to climate resilience and disaster mitigation across Bangladesh. 🇧🇩</b>
</div>
