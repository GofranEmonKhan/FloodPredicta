# FloodPredicta (Version 2.0): System Redesign & Strategic Experiment Plan

**Workspace:** `d:\Users\Abdul Gofran Emon\Downloads\FloodPredicta\version2`  
**Status:** Completed & Production Deployed (Pipelines, Benchmarks, Diagnostic Reports & Interactive Web App)  
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

## 5. Notebook Pipeline Architecture & Implemented Notebooks

The experimental workflow in `version2/notebooks/` is structured across dedicated, self-contained Jupyter Notebooks:

### 📔 `data_preprocessing.ipynb`
- Ingests `data/raw/flooddata.csv` (20,544 rows × 19 columns).
- Missing value audit & historical validation of `Flood?` target imputation.
- Non-destructive extreme value analysis: replaces 5-pass IQR row dropping with 99th-percentile winsorization and log transforms.
- **100% disaster preservation:** Retains all 4,132 historical flood events.
- Exports `data/processed/flood_data_preprocessed.csv` (20,544 rows × 20 columns).

### 📔 `feature_engineering.ipynb`
- Chronological sorting per meteorological station.
- Computes Lag-1 and Lag-2 rainfall, and 3-month rolling monsoonal rainfall sums.
- Computes hydrological runoff risk ratios ($\frac{\text{Rainfall}}{\text{ALT} + 1}$) and atmospheric moisture interaction indices.
- Cyclical Fourier transformations of temporal coordinates ($\sin, \cos$).
- One-hot encodes 33 BMD stations with baseline reference column dropped.
- Exports `data/processed/flood_data_featured.csv` (20,544 rows × 64 columns).

### 📔 `split_shuffled_70_15_15/` (Regime 1: Stratified Shuffled 70/15/15)
- 5 self-contained training notebooks: XGBoost, HistGradientBoosting, Random Forest, Extra Trees, and Stacking Meta-Ensemble.
- Stratified validation threshold optimization ($t \in [0.10, 0.85]$).
- Exports serialized models to `models/split_shuffled_70_15_15/` and 20 standalone diagnostic charts to `reports/split_shuffled_70_15_15/`.

### 📔 `split_chronological_70_15_15/` (Regime 2: Chronological Split A)
- 5 self-contained training notebooks: Train $\le 2003$, Val $2004–2008$, Test $\ge 2009$.
- Walk-forward validation threshold calibration preventing lookahead leakage.
- Exports serialized models to `models/split_chronological_70_15_15/` and 20 standalone diagnostic charts to `reports/split_chronological_70_15_15/`.

### 📔 `split_chronological_70_20_10/` (Regime 3: Chronological Split B)
- 5 self-contained training notebooks: Train $\le 1997$, Val $1998–2008$, Test $\ge 2009$.
- Strict historical baseline with 20% validation tuning window.
- Exports serialized models to `models/split_chronological_70_20_10/` and 20 standalone diagnostic charts to `reports/split_chronological_70_20_10/`.

---

## 6. Directory Structure

```text
version2/
├── README.md                                 # Master strategic plan & architecture
├── EXPERIMENT_TRACKER.md                     # Comprehensive experiment chronicle & threshold audit
├── data/
│   ├── raw/
│   │   └── flooddata.csv                     # Preserved raw BMD climate dataset (20,544 rows)
│   └── processed/
│       ├── flood_data_preprocessed.csv       # 100% flood preserved (20,544 rows × 20 cols)
│       └── flood_data_featured.csv           # Engineered features (20,544 rows × 64 cols)
├── notebooks/
│   ├── data_preprocessing.ipynb             # Common data cleaning & winsorization
│   ├── feature_engineering.ipynb            # Common hydrological lags & features
│   ├── split_shuffled_70_15_15/              # Regime 1: Stratified Shuffled Split (5 notebooks)
│   ├── split_chronological_70_15_15/         # Regime 2: Chronological Split A (5 notebooks)
│   └── split_chronological_70_20_10/         # Regime 3: Chronological Split B (5 notebooks)
├── models/
│   ├── split_shuffled_70_15_15/              # 5 trained models + feature_columns.pkl
│   ├── split_chronological_70_15_15/         # 5 trained models + feature_columns.pkl
│   └── split_chronological_70_20_10/         # 5 trained models + feature_columns.pkl
├── reports/
│   ├── split_shuffled_70_15_15/              # 20 standalone diagnostic plots
│   ├── split_chronological_70_15_15/         # 20 standalone diagnostic plots
│   └── split_chronological_70_20_10/         # 20 standalone diagnostic plots
└── deployment/                               # Production Web Application & Serving API
    ├── app.py                                # Flask web application & REST API
    ├── predictor.py                          # 59-feature zero-defect runtime inference engine
    ├── station_metadata.json                 # Precomputed geo-coordinates & precipitation baselines
    ├── templates/
    │   └── index.html                        # River-themed single-page responsive dashboard
    ├── static/
    │   ├── css/style.css                     # Lush greenery & moonlit river night mode styles
    │   └── js/
    │       ├── app.js                        # Dynamic UI, gauge animator & scenario presets
    │       ├── i18n.js                       # Bangladesh-context bilingual engine (EN / বাংলা)
    │       └── audio.js                      # Procedural Web Audio API river soundscape
    └── tests/
        └── test_predictor.py                 # Automated unit tests for inference engine
```

---

## 7. Current Project State

- [x] Version 1 intact and preserved in parent directory.
- [x] Non-destructive data cleaning and hydrological feature engineering completed with 100% disaster event preservation.
- [x] 15 self-contained notebooks executed across 3 rigorous partitioning regimes.
- [x] 15 model pipelines serialized and saved with feature schemas.
- [x] 60 standalone publication-ready diagnostic charts exported to `version2/reports/`.
- [x] Production web application deployed in `version2/deployment/` with Flask, zero-defect predictor, Bangladeshi river greenery UI, day/night mode, procedural delta audio, bilingual engine, and emergency contacts.

---

## 8. Production Web Application & Serving Architecture

The production application is located in `version2/deployment/` and delivers a complete, user-centric disaster monitoring dashboard tailored to Bangladesh's hydrological landscape.

### Key Capabilities
* **Interactive Single-Page Application (SPA):** Clean, modern interface styled with an animated SVG river delta backdrop, smooth glassmorphism cards, and an interactive speedometer risk gauge.
* **Day & Night Riverine Theme:**
  * *Day Mode:* Inspired by lush monsoon riverbanks, fertile silt plains, and emerald paddy fields.
  * *Night Mode:* Inspired by moonlit midnight rivers, deep delta indigo, and luminous emerald accents.
* **Calming River Ambient Audio:** Procedural, synthesized river water flow and gentle water ripples generated natively via the Web Audio API (zero external audio file dependencies). Can be toggled on/off at any time.
* **Authentic Bilingual Localization (English & বাংলা):** Thoughtfully adapted to Bangladeshi meteorological and disaster-management terminology (e.g., *বন্যা পূর্বাভাস ও সতর্কীকরণ ব্যবস্থা*, *অতি প্রবল বন্যা ঝুঁকি*, *হাওর ও নিম্নাঞ্চল প্লাবন*).
* **Bangladesh Emergency Helplines:** Quick-access direct-dial cards for national emergency services:
  * **999** — National Emergency Service
  * **1090** — Disaster Early Warning Helpline
  * **333** — Government Information & Citizen Services
  * **16263** — National Health Call Center
  * **BWDB FFWC** — Flood Forecasting and Warning Centre
* **Historical Scenario Presets:** One-click preloads for historical test scenarios:
  * *1998 Deluge:* Catastrophic mega-flood conditions at Dhaka station.
  * *2022 Sylhet Flash Flood:* Extreme mountain runoff and flash flood pooling in the Surma-Kushiyara basin.
  * *Dry Winter Season:* Typical peaceful winter baseline with zero flood risk.
* **REST API:**
  * `POST /api/predict`: Ingests weather observations, computes all 59 hydrological features, and returns flood probability, binary decision, risk tier, and warning advisory.
  * `GET /api/stations`: Returns metadata and historical monthly rainfall averages for all 33 BMD stations.
  * `GET /api/health`: Health status and loaded model details.

### Launch Instructions
```bash
# From repository root
python version2/deployment/app.py
```
Access the application at `http://127.0.0.1:5000` in any modern web browser.
