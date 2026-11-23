# FloodPredicta (Version 2.0) - Experiment & Implementation Tracker

**File Location:** `version2/EXPERIMENT_TRACKER.md`  
**Purpose:** Living chronicle of all engineering steps, scientific decisions, data transformations, and model evaluations conducted in Version 2.0. This serves as the single source of truth for downstream global documentation.  
**Last Updated:** November 2026  
**Current Milestone:** Step 3 Completed — Triple Partitioning Regimes (Shuffled 70/15/15, Chronological 70/15/15, and Chronological 70/20/10)  

---

## 1. Milestone & Roadmap Tracker

| Phase / Step | Artifact / Notebook | Status | Key Deliverable / Outcome |
| :--- | :--- | :---: | :--- |
| **0. Architecture & Setup** | `version2/README.md` | ✅ **Completed** | Full V1 flaw analysis, V2 strategic redesign, and 5-model selection justification. |
| **1. Data Preprocessing** | `data_preprocessing.ipynb` | ✅ **Completed** | Non-destructive cleaning, missing target resolution, winsorization, log transforms, and chronological sorting. |
| **2. Feature Engineering** | `feature_engineering.ipynb` | ✅ **Completed** | Antecedent rainfall lags (Lag-1, Lag-2), 3-month rolling monsoonal sums, $\frac{\text{Rain}}{\text{ALT}}$ runoff ratios, cyclical seasonality. |
| **3A. Shuffled Split (70/15/15)** | `notebooks/split_shuffled_70_15_15/` | ✅ **Completed** | 5 dedicated notebooks executed with 4 standalone plots each. Higher F1 (0.9374) and ROC-AUC (0.9958) than V1. |
| **3B. Chronological Split (70/15/15)** | `notebooks/split_chronological_70_15_15/` | ✅ **Completed** | 5 dedicated notebooks executed (Train <= 2003, Val 2004–2008, Test >= 2009) with 4 standalone plots each. |
| **3C. Chronological Split (70/20/10)** | `notebooks/split_chronological_70_20_10/` | ✅ **Completed** | 5 dedicated notebooks executed (Train <= 1997, Val 1998–2008, Test >= 2009) with 4 standalone plots each. |
| **4. Visual Reports** | `reports/` (60 standalone plots) | ✅ **Completed** | 20 plots per regime across all 3 partition directories. |
| **5. Production Deployment** | `deployment/` | ⏳ **Next Up** | Production serving pipeline with verified unit tests. |

---

## 2. Deep Dive: Why Did Version 1 Show a Seemingly "Higher" Result?

There are two primary reasons why Version 1 showed 98% accuracy versus initial forward-looking runs:

### Reason A: Version 1 Deleted 33% of Real Flood Disasters as "Outliers"
* In Version 1 (`final_preprocessing.ipynb`), **5 iterative passes of IQR outlier filtering** were executed.
* This deleted 2,984 records, including **1,387 verified flood events** (reducing positive flood samples from 4,132 down to 2,745).
* By dropping the most severe, extreme monsoon deluges, the remaining dataset consisted only of milder, easier-to-classify samples, which artificially inflated the raw accuracy metric.
* In Version 2, **all 4,132 real flood disaster events are 100% preserved**.

### Reason B: The Split Strategy
* In Version 1, data was **randomly shuffled** across all 65 years (`train_test_split(..., shuffle=True)`).
* When we evaluate Version 2 using the **exact same shuffled, stratified split**, **Version 2 actually outperforms Version 1 across all disaster-detection metrics**:

| Metric | Version 1 (Pruned Data, Random Split) | Version 2 (Full Data + Hydrology, Stratified Split) | Improvement in Version 2 |
| :--- | :---: | :---: | :---: |
| **Preserved Flood Events** | 2,745 (1,387 deleted) | **4,132 (100% Preserved)** | **+1,387 real disaster events** |
| **Flood F1-Score** | 0.9241 | **0.9374** | **Higher detection balance** |
| **Flood Recall** | 0.9435 | **0.9774** | **Catches 97.7% of floods** |
| **ROC-AUC Score** | 0.9946 | **0.9958** | **Higher discrimination power** |
| **Accuracy** | 0.9800 | **0.9737** | Genuine on all extreme events |

---

## 3. The 3 Partitioning Regimes Evaluated

### Regime 1: Stratified Shuffled Split (70% Train: 14,380 | 15% Val: 3,082 | 15% Test: 3,082)
*Directory:* `notebooks/split_shuffled_70_15_15/` | `models/split_shuffled_70_15_15/` | `reports/split_shuffled_70_15_15/`  
*Concept:* Standard classification benchmarking where observations across all 33 stations and years are shuffled, ensuring equal geographic and temporal representation in training, validation, and testing.

| Model Name | Notebook File | Accuracy | Precision (Flood=1) | Recall (Flood=1) | F1-Score (Flood=1) | ROC-AUC | Model Artifact |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost (Champion)** | `train_xgboost.ipynb` | **0.9737** | **0.9004** | **0.9774** | **0.9374** | **0.9958** | `models/split_shuffled_70_15_15/xgboost_model.pkl` |
| **HistGradientBoosting** | `train_histgradientboosting.ipynb` | 0.9734 | 0.8956 | **0.9823** | 0.9369 | 0.9957 | `models/split_shuffled_70_15_15/hist_gradient_boosting_model.pkl` |
| **Extra Trees** | `train_extratrees.ipynb` | **0.9744** | **0.9300** | 0.9435 | 0.9367 | 0.9949 | `models/split_shuffled_70_15_15/extra_trees_model.pkl` |
| **Stacking Ensemble** | `train_stacking_ensemble.ipynb` | 0.9731 | 0.8966 | 0.9790 | 0.9360 | **0.9963** | `models/split_shuffled_70_15_15/stacking_ensemble_model.pkl` |
| **Random Forest** | `train_randomforest.ipynb` | 0.9734 | **0.9311** | 0.9371 | 0.9341 | 0.9960 | `models/split_shuffled_70_15_15/random_forest_model.pkl` |

---

### Regime 2: Chronological Split A (Train <= 2003: 16,656 | Val 2004–2008: 1,920 | Test >= 2009: 1,968)
*Directory:* `notebooks/split_chronological_70_15_15/` | `models/split_chronological_70_15_15/` | `reports/split_chronological_70_15_15/`  
*Concept:* Historical 56-year baseline (1948–2003) for training, mid-2000s (2004–2008) for threshold tuning, and the final 5 years (2009–2013) as an out-of-time test set.

| Model Name | Notebook File | Accuracy | Precision (Flood=1) | Recall (Flood=1) | F1-Score (Flood=1) | ROC-AUC | Model Artifact |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost (Champion)** | `train_xgboost.ipynb` | **0.9644** | **0.9127** | **0.9127** | **0.9127** | **0.9871** | `models/split_chronological_70_15_15/xgboost_model.pkl` |
| **Stacking Ensemble** | `train_stacking_ensemble.ipynb` | **0.9644** | **0.9127** | **0.9127** | **0.9127** | 0.9876 | `models/split_chronological_70_15_15/stacking_ensemble_model.pkl` |
| **HistGradientBoosting** | `train_histgradientboosting.ipynb` | 0.9624 | 0.9057 | 0.9102 | 0.9080 | **0.9879** | `models/split_chronological_70_15_15/hist_gradient_boosting_model.pkl` |
| **Extra Trees** | `train_extratrees.ipynb` | 0.9619 | 0.9312 | 0.8778 | 0.9037 | 0.9846 | `models/split_chronological_70_15_15/extra_trees_model.pkl` |
| **Random Forest** | `train_randomforest.ipynb` | 0.9614 | **0.9452** | 0.8603 | 0.9008 | 0.9843 | `models/split_chronological_70_15_15/random_forest_model.pkl` |

---

### Regime 3: Chronological Split B (70-20-10: Train <= 1997: 14,352 | Val 1998–2008: 4,224 | Test >= 2009: 1,968)
*Directory:* `notebooks/split_chronological_70_20_10/` | `models/split_chronological_70_20_10/` | `reports/split_chronological_70_20_10/`  
*Concept:* Strict 70% historical training (1948–1997), 20% validation for threshold tuning (1998–2008), and 10% future holdout test set (2009–2013).

| Model Name | Notebook File | Accuracy | Precision (Flood=1) | Recall (Flood=1) | F1-Score (Flood=1) | ROC-AUC | Model Artifact |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stacking Ensemble (Champion)** | `train_stacking_ensemble.ipynb` | **0.9649** | **0.9150** | **0.9127** | **0.9139** | 0.9859 | `models/split_chronological_70_20_10/stacking_ensemble_model.pkl` |
| **XGBoost** | `train_xgboost.ipynb` | 0.9639 | 0.9084 | **0.9152** | 0.9118 | 0.9866 | `models/split_chronological_70_20_10/xgboost_model.pkl` |
| **HistGradientBoosting** | `train_histgradientboosting.ipynb` | 0.9619 | 0.8976 | **0.9177** | 0.9075 | **0.9878** | `models/split_chronological_70_20_10/hist_gradient_boosting_model.pkl` |
| **Random Forest** | `train_randomforest.ipynb` | 0.9629 | **0.9409** | 0.8728 | 0.9056 | 0.9814 | `models/split_chronological_70_20_10/random_forest_model.pkl` |
| **Extra Trees** | `train_extratrees.ipynb` | 0.9599 | 0.9259 | 0.8728 | 0.8986 | 0.9831 | `models/split_chronological_70_20_10/extra_trees_model.pkl` |

---

## 4. Standalone Visualizations (Matching Model Train Folder)

Every notebook across all three regimes generates the exact 4 evaluation plots from `Model Train/final_trainRF.ipynb`:
1. **Graph 1 — Confusion Matrix Heatmap:** Annotated heatmap (`cmap='Blues'`).
2. **Graph 2 — Receiver Operating Characteristic (ROC) Curve:** Orange curve with dark blue diagonal and printed AUC.
3. **Graph 3 — Predicted Probability Distribution:** Dual KDE + histogram plot (Non-Flood in `skyblue`, Flood in `salmon`).
4. **Graph 4 — Top 15 Feature Importances:** Vertical bar chart matching `Model Train/final_trainRF.ipynb`.

All plots are exported as standalone PNG files:
* `reports/split_shuffled_70_15_15/` (20 plots)
* `reports/split_chronological_70_15_15/` (20 plots)
* `reports/split_chronological_70_20_10/` (20 plots)

---

## 5. Optimal Threshold Selection Methodology & Scientific Basis

### 5.1 Why Threshold Tuning is Mandatory (The Flaw of Default 0.50)
Standard classification algorithms default to a decision threshold of $t = 0.50$ ($\hat{y} = 1 \iff \hat{P}(\text{Flood}=1) \ge 0.50$). This default operates under two strict assumptions:
1. **Balanced Class Priors:** Classes are equally represented ($50\% / 50\%$).
2. **Symmetric Error Costs:** A False Positive (false alarm) has the exact same real-world penalty as a False Negative (missed flood).

In flood early warning across Bangladesh, **both assumptions fail dramatically**:
- **Class Imbalance (~80:20 Skew):** Floods occur in only $\sim 20\%$ of monthly records (4,132 flood months vs. 16,412 normal months). Under extreme class imbalance, uncalibrated models output compressed probability distributions where positive posterior probabilities rarely cross 0.50 during moderate deluges.
- **Asymmetric Disaster Costs:**
  - **False Negative (Missed Flood):** Fatalities, stranded communities, unevacuated livestock, catastrophic loss of life and livelihood.
  - **False Positive (False Alarm):** Evacuation logistical costs, temporary disruption, public alarm fatigue ("crying wolf").
  - While missing a disaster is vastly more dangerous, a model that triggers constant false alarms loses public trust. Hence, an operational early warning system requires a mathematically calibrated decision threshold.

---

### 5.2 How the Optimal Threshold Was Selected in Training

Threshold optimization is executed exclusively on the **independent Validation Set** (`X_val`, `y_val`) to eliminate data leakage and prevent optimistic bias. The holdout test set (`X_test`, `y_test`) remains strictly untouched during this process.

The step-by-step mathematical procedure implemented in each model notebook:
1. **Generate Continuous Validation Probabilities:**
   $$\hat{P}_i = P(y_i = 1 \mid \mathbf{x}_i), \quad \forall i \in \text{Validation Set}$$
2. **Systematic Threshold Grid Sweep:**
   We evaluate an arithmetic grid of candidate thresholds $t \in [0.10, 0.85]$ with step size $\Delta t = 0.05$:
   $$\mathcal{T} = \{0.10, 0.15, 0.20, \dots, 0.80, 0.85\}$$
3. **Binarization & Metric Computation:**
   For each candidate threshold $t \in \mathcal{T}$:
   $$\hat{y}_i(t) = \begin{cases} 1 & \text{if } \hat{P}_i \ge t \\ 0 & \text{if } \hat{P}_i < t \end{cases}$$
   Calculate positive-class metrics specifically on the flood disaster class ($y=1$):
   $$\text{Precision}(t) = \frac{TP(t)}{TP(t) + FP(t)}$$
   $$\text{Recall}(t) = \frac{TP(t)}{TP(t) + FN(t)}$$
   $$F_1(t) = 2 \times \frac{\text{Precision}(t) \times \text{Recall}(t)}{\text{Precision}(t) + \text{Recall}(t)}$$
4. **Objective Function Optimization:**
   The optimal operational threshold $t^*$ is chosen as the point that **maximizes the positive-class $F_1$-score**:
   $$t^* = \arg\max_{t \in \mathcal{T}} F_{1,\text{val}}(t)$$
5. **Operational Sensitivity Table:**
   Every training notebook prints the complete validation grid table. This allows disaster response authorities to choose alternate operational setpoints:
   - **Maximum $F_1$ Mode ($t^* \approx 0.45 - 0.50$):** Optimal mathematical trade-off between false alarms and missed events (validation $F_1 \approx 0.93 - 0.94$).
   - **High-Alert / Life-Safety Mode ($t \approx 0.30 - 0.35$):** Prioritizes disaster recall ($\text{Recall} \ge 98\%$), ensuring virtually zero missed floods at the cost of a modest increase in precautionary alerts.

---

## 6. Clean Version 2 Directory Map

```
version2/
├── README.md                                 # Master strategic architecture & plan
├── EXPERIMENT_TRACKER.md                     # This master tracker file
├── data/
│   ├── raw/
│   │   └── flooddata.csv                     # Original raw dataset (20,544 rows)
│   └── processed/
│       ├── flood_data_preprocessed.csv       # Cleaned & winsorized data (20,544 rows × 20 cols)
│       └── flood_data_featured.csv           # Final engineered data (20,544 rows × 64 cols)
├── notebooks/
│   ├── data_preprocessing.ipynb             # Common data cleaning & winsorization
│   ├── feature_engineering.ipynb            # Common hydrological lags & features
│   ├── split_shuffled_70_15_15/              # Regime 1: Stratified Shuffled Split (70/15/15)
│   │   ├── train_xgboost.ipynb
│   │   ├── train_histgradientboosting.ipynb
│   │   ├── train_randomforest.ipynb
│   │   ├── train_extratrees.ipynb
│   │   └── train_stacking_ensemble.ipynb
│   ├── split_chronological_70_15_15/         # Regime 2: Chronological Split (Train <= 2003, Val 2004-2008, Test >= 2009)
│   │   ├── train_xgboost.ipynb
│   │   ├── train_histgradientboosting.ipynb
│   │   ├── train_randomforest.ipynb
│   │   ├── train_extratrees.ipynb
│   │   └── train_stacking_ensemble.ipynb
│   └── split_chronological_70_20_10/         # Regime 3: Chronological 70-20-10 Split (Train <= 1997, Val 1998-2008, Test >= 2009)
│       ├── train_xgboost.ipynb
│       ├── train_histgradientboosting.ipynb
│       ├── train_randomforest.ipynb
│       ├── train_extratrees.ipynb
│       └── train_stacking_ensemble.ipynb
├── models/
│   ├── split_shuffled_70_15_15/              # 5 trained models from Regime 1 + feature_columns.pkl
│   ├── split_chronological_70_15_15/         # 5 trained models from Regime 2 + feature_columns.pkl
│   └── split_chronological_70_20_10/         # 5 trained models from Regime 3 + feature_columns.pkl
├── reports/
│   ├── split_shuffled_70_15_15/              # 20 standalone plots (4 plots × 5 models)
│   ├── split_chronological_70_15_15/         # 20 standalone plots (4 plots × 5 models)
│   └── split_chronological_70_20_10/         # 20 standalone plots (4 plots × 5 models)
├── src/
│   ├── create_and_run_nb01.py                # Preprocessing runner
│   ├── create_and_run_nb02.py                # Feature engineering runner
│   ├── build_individual_model_notebooks.py   # Shuffled models generator & executor
│   └── rebuild_and_preserve_all_splits.py    # Chronological generator for both regimes
└── deployment/                               # Prepared for production serving
```
