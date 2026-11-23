# FloodPredicta (Version 2.0): System Redesign & Strategic Experiment Plan

**Workspace:** `d:\Users\Abdul Gofran Emon\Downloads\FloodPredicta\version2`  
**Status:** Architecture & Planning Phase (Implementation On Hold Pending Review)  
**Target:** Flood Prediction and Early Warning across 33 Bangladesh Meteorological Stations  

---

## 1. Executive Summary & Vision

**Version 2.0** of the **FloodPredicta** project is a comprehensive reimagining of the flood classification pipeline. While Version 1 demonstrated the viability of tree-based models on Bangladesh Meteorological Department (BMD) climate data, it suffered from critical methodological limitations — including extreme-event data loss, temporal lookahead bias, artificial SMOTE noise on spatial features, and deployment runtime crashes.

Version 2 preserves the original raw data (`version2/data/raw/flooddata.csv`) but completely redesigns the data engineering, hydrological modeling, validation protocols, and deployment architecture to build a **realistic, production-grade early warning system**.

---

## 2. Autopsy of Version 1: What Was Done, Flaws & Methodological Limitations

| Component | Version 1 Implementation | Identified Flaw / Limitation | Negative Impact |
| :--- | :--- | :--- | :--- |
| **Outlier Handling** | 5 successive passes of IQR trimming on `Rainfall`, `Max_Temp`, `Wind_Speed`, etc. | **Floods are caused by extreme weather.** Removing statistical rainfall outliers discarded **1,387 real flood cases** (~33% of all positive instances). | The model lost exposure to extreme catastrophe events (e.g. 1988, 1998, 2004 mega-floods) and cannot generalize to severe deluges. |
| **Data Splitting** | Random 80/20 train/test split with `stratify=y`. | **Temporal data leakage.** Weather is sequential. Random splitting places past and future months of the same monsoon season in both splits. | Evaluated test metrics (98% accuracy) were overly optimistic due to temporal memorization (lookahead bias). |
| **Feature Set** | Static single-month weather observations in isolation. | Floods are governed by **soil saturation and antecedent rain**. A single monthly rainfall figure ignores whether previous months saturated the ground. | Misses compound multi-month flood build-ups and river basin pooling. |
| **Class Balancing** | Standard SMOTE applied *after* one-hot encoding the 31 station columns. | SMOTE linearly interpolates dummy columns, creating fractional station vectors (e.g. `Station_Dhaka = 0.38`). | Injects synthetic mathematical nonsense into spatial indicators, confusing tree splits. |
| **Model Regularization** | `RandomForestClassifier` with `max_depth=None` (unbounded). | Unconstrained tree depth overfits specific station quirks and historical noise. | Suboptimal out-of-sample generalization. |
| **Deployment Code** | `scaler.transform(input_df[numeric_cols])` in `appRF.py`. | `scaler.pkl` was fitted on 44 features; `appRF.py` passed only 13 features, crashing with a `ValueError`. In `APP.py`, `'Humidity'` was queried instead of `'Relative_Humidity'`. | Deployment interfaces were dysfunctional and broke upon user submission. |

---

## 3. The Version 2 Strategic Redesign

### A. Non-Destructive Extreme Value Engineering
- **Zero Row Deletion for Extreme Weather:** Keep all historical flood events. Extreme rainfall is not "measurement noise"; it is the exact physical trigger of floods.
- **Robust Feature Scaling:** Replace basic scaling with `RobustScaler` (median & IQR centering) or `QuantileTransformer(output_distribution='normal')` to handle heavy-tailed precipitation without distorting model weights.

### B. Hydrological & Temporal Feature Engineering
We introduce real-world physical and hydrological indicators:
1. **Antecedent Rainfall & Lags:**
   - $\text{Rainfall\_Lag\_1}$: Precipitation in month $t-1$.
   - $\text{Rainfall\_Lag\_2}$: Precipitation in month $t-2$.
2. **Cumulative Monsoonal Saturation:**
   - $\text{Rolling\_3M\_Rainfall}$: 3-month cumulative rainfall ($\sum_{k=0}^{2} \text{Rainfall}_{t-k}$) capturing catchment saturation.
   - $\text{Rainfall\_Spike\_Ratio}$: $\frac{\text{Rainfall}_t}{\text{Mean Rainfall of Station in Month}}$.
3. **Hydraulic Topography & Interaction Ratios:**
   - $\text{Runoff\_Pooling\_Index} = \frac{\text{Rainfall}}{\text{ALT} + 1}$: Quantifies elevated risk for low-lying delta stations compared to hilly areas.
   - $\text{Monsoon\_Moisture\_Index} = \text{Relative\_Humidity} \times \text{Cloud\_Coverage}$: Measures prolonged depressions and low evapotranspiration.
4. **Cyclical Seasonality Encodings:**
   - Rather than arbitrary integer months (1–12), use continuous Fourier cyclical transformations:
     $$\sin\left(\frac{2\pi \cdot \text{Month}}{12}\right), \quad \cos\left(\frac{2\pi \cdot \text{Month}}{12}\right)$$
   - Add a binary domain flag: $\text{Is\_Monsoon} = 1$ for months June through September.

### C. Realistic Chronological & Walk-Forward Validation
- **Train Split:** 1948 – 2003 (55 years of historical baseline).
- **Validation Split:** 2004 – 2008 (5 years for hyperparameter tuning & threshold optimization).
- **Out-of-Time Test Split:** 2009 – 2013 (Final 5 years unseen real future forecasting test).
- **Eliminates temporal data leakage** and guarantees genuine operational reliability.

### D. Tree-Native Class Imbalance Handling
- Drop naive SMOTE on one-hot features.
- Leverage algorithm-native weighting:
  - XGBoost: $\text{scale\_pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}} \approx 5.4$
  - LightGBM: `is_unbalance=True` or `class_weight='balanced'`
  - CatBoost: `auto_class_weights='Balanced'` or Focal Loss.
- Eliminates synthetic noise while protecting minority disaster detection.

### E. Operational Threshold Optimization on Validation Set
- Rather than blindly relying on the default $0.50$ cutoff (which assumes equal class distribution and equal cost of errors), the decision threshold is explicitly calibrated.
- A grid search over thresholds $t \in [0.10, 0.85]$ (step size $0.05$) is evaluated on the **independent validation set** (`X_val`, `y_val`).
- The optimal operational cutoff $t^*$ is chosen by maximizing the disaster-class $F_1$-score:
  $$t^* = \arg\max_{t \in [0.10, 0.85]} F_1(\text{Flood}=1)$$
- This balances disaster detection coverage (Recall) with public credibility and cost of false alarms (Precision), without leaking holdout test data.

---

## 4. The 5 Chosen Machine Learning Models & Selection Justification

Based on the dataset's characteristics (**tabular, high-dimensional mixed categorical-continuous, spatial coordinates, extreme hydrological tails, and 5.4:1 class imbalance**), we select 5 complementary models:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MODEL 1: CATBOOST                               │
│  - Best-in-class categorical handling for the 33 BMD station names    │
│  - Built-in target statistics without one-hot dimensionality bloat     │
│  - Native class weights & ordered boosting to prevent target leakage   │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                        MODEL 2: LIGHTGBM                               │
│  - Fast, leaf-wise tree growth with histogram-based binning            │
│  - Exceptional capture of sharp thresholds in skewed rainfall metrics  │
│  - Built-in positive class weighting & focal loss support              │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                        MODEL 3: OPTIMIZED XGBOOST                      │
│  - Proven top contender from V1 (lowest RMSE: 0.123)                   │
│  - Regularized with exact L1/L2 penalties, colsample_bytree & max_depth│
│  - Calibrated scale_pos_weight for high-recall disaster warning        │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                MODEL 4: BALANCED RANDOM FOREST (REGULARIZED)           │
│  - Robust bagging architecture providing low variance                  │
│  - Constrained tree depth (max_depth: 10–14) to eliminate overfitting  │
│  - Downsampled or balanced bootstrap trees for extreme class fidelity   │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                 MODEL 5: STACKING ENSEMBLE (META-CLASSIFIER)           │
│  - Meta-Learner: Calibrated Logistic Regression or ElasticNet Blender  │
│  - Ingests out-of-fold probabilistic outputs of Models 1–4             │
│  - Blends distinct learning biases (bagging + 3 boosting types) to     │
│    maximize F1-score and minimize localized false alarms               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Notebook Pipeline Architecture (Planned)

In accordance with project requirements, the experimental workflow in `version2/notebooks/` will be structured across dedicated Jupyter Notebooks:

### 📔 `01_eda_and_data_cleaning.ipynb`
- Ingests `data/raw/flooddata.csv`.
- Missing value audit & historical validation of `Flood?` target imputation.
- Non-destructive extreme value analysis (replacing 5-pass IQR row dropping with percentile winsorization / robust scaling).
- Exports `data/processed/flood_cleaned.csv`.

### 📔 `02_hydrological_feature_engineering.ipynb`
- Chronological sorting per meteorological station.
- Computes Lag-1, Lag-2 rainfall, and 3-month rolling monsoonal sums.
- Computes hydrological runoff risk ratios ($\frac{\text{Rain}}{\text{ALT}}$) and moisture interaction indices.
- Cyclical transformation of temporal coordinates.
- Exports `data/processed/flood_featured.csv`.

### 📔 `03_model_training_and_tuning.ipynb`
- Chronological train/validation/test partitioning.
- Training and Optuna hyperparameter optimization for:
  1. CatBoost Classifier
  2. LightGBM Classifier
  3. Optimized XGBoost
  4. Balanced Random Forest
- Construction of Model 5 (Stacking Meta-Ensemble).
- Serializes trained model pipelines into `models/`.

### 📔 `04_evaluation_and_threshold_tuning.ipynb`
- Comprehensive comparative performance matrix (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC).
- Precision-Recall curves and operational early-warning threshold tuning ($t \in [0.10, 0.90]$).
- SHAP (SHapley Additive exPlanations) global and local feature importance analysis.
- Exports publication-ready charts to `reports/`.

---

## 6. Directory Structure Created

```
version2/
├── README.md                                 # Master strategic plan & architecture
├── data/
│   ├── raw/
│   │   └── flooddata.csv                     # Copied pristine raw dataset (20,544 rows)
│   └── processed/                            # (Awaiting Notebook 01/02 outputs)
├── notebooks/                                # (Awaiting creation of structured notebooks)
│   ├── 01_eda_and_data_cleaning.ipynb
│   ├── 02_hydrological_feature_engineering.ipynb
│   ├── 03_model_training_and_tuning.ipynb
│   └── 04_evaluation_and_threshold_tuning.ipynb
├── src/                                      # Reusable Python modules & data loaders
├── models/                                   # Exported model artifacts & scalers
├── reports/                                  # Evaluation charts, confusion matrices, metrics
└── deployment/                               # Production Flask / Streamlit deployment code
```

---

## 7. Current Project State

- [x] Version 1 intact and preserved in parent directory.
- [x] Version 2 directory structure initialized.
- [x] Raw dataset copied to `version2/data/raw/flooddata.csv`.
- [x] Strategic architecture, flaw analysis, and 5-model selection documented.
- [ ] **Implementation Phase:** On hold pending user confirmation.
