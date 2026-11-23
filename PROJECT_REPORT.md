# FloodPredicta: AI-Powered Flood Prediction System
## Comprehensive Technical Project Report

**Project Location:** `d:\Users\Abdul Gofran Emon\Downloads\FloodPredicta`  
**Dataset Domain:** Hydrological & Meteorological Time-Series / Classification (Bangladesh)  
**Target Variable:** `Flood?` (Binary: `1.0` = Flood Event, `0.0` = Non-Flood Event)  

---

## 1. Project Overview & Objectives

**FloodPredicta** is a machine learning initiative aimed at early flood risk prediction across Bangladesh. Bangladesh is one of the most flood-prone delta nations in the world, regularly impacted by seasonal monsoon deluges, flash floods, and cyclonic storm surges. 

The primary objective of this project is to develop and evaluate automated classification models that can ingest local weather indicators, geographic coordinates, and historical temporal cycles to predict whether a specific meteorological station area will experience flood conditions.

---

## 2. End-to-End Workflow & Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          1. DATA INGESTION                             │
│  - Raw dataset: 20,544 records from 33 BMD meteorological stations    │
│  - Time span: 1948 – 2013 across 19 geographical & weather features    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   2. DATA CLEANING & PREPROCESSING                     │
│  - Target Imputation: Missing 'Flood?' values (16,051) imputed to 0.0  │
│  - Outlier Mitigation: 1st/99th quantile clipping + 5-pass IQR trimming│
│  - Dataset pruned from 20,544 rows to 17,560 rows                      │
│  - Log Transforms: log1p(Rainfall, Wind_Speed, ALT) to address skewness│
│  - Dropped redundant 'Station_Number' column                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             3. FEATURE ENGINEERING & IMBALANCE HANDLING                │
│  - One-Hot Encoding: 31 Station Dummies created (44 total features)    │
│  - Stratified Train-Test Split (80% Train: 14,048 / 20% Test: 3,512)   │
│  - Feature Normalization: StandardScaler applied across all features   │
│  - Balancing: SMOTE applied to train split (2,196 -> 11,852 flood rows)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  4. MODEL TRAINING & BENCHMARKING                      │
│  - Random Forest Classifier (200 trees, class_weight='balanced')       │
│  - XGBoost Classifier                                                  │
│  - Logistic Regression (max_iter=1000, class_weight='balanced')        │
│  - Evaluation: Precision, Recall, F1, ROC-AUC, Threshold Tuning        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       5. DEPLOYMENT APPLICATIONS                       │
│  - Flask Web UI (appRF.py + index.html)                                │
│  - Streamlit Dashboard (APP.py + flood_pipeline.pkl)                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset Characteristics & Feature Descriptions

The data originates from the **Bangladesh Meteorological Department (BMD)** recording monthly aggregates across major administrative divisions and river basins.

### Data Dimensions
- **Raw Observations:** 20,544 rows × 19 columns
- **Preprocessed Observations:** 17,560 rows × 21 columns
- **Observation Window:** 1948 to 2013
- **Number of Weather Stations:** 33 stations

### Weather Stations Monitored
*Barisal, Bhola, Bogra, Chandpur, Chittagong (City-Ambagan), Chittagong (IAP-Patenga), Comilla, Cox's Bazar, Dhaka, Dinajpur, Faridpur, Feni, Hatiya, Ishurdi, Jessore, Khepupara, Khulna, Kutubdia, Madaripur, Maijdee Court, Mongla, Mymensingh, Patuakhali, Rajshahi, Rangamati, Rangpur, Sandwip, Satkhira, Sitakunda, Srimangal, Sylhet, Tangail, Teknaf.*

### Feature Taxonomy

| Category | Column Name | Type | Unit / Format | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Identifiers** | `Sl` | Integer | ID | Record sequence counter |
| | `Station_Names` | Categorical | String | Name of the meteorological station |
| | `Station_Number` | Integer | Categorical ID | Station numeric code (dropped in preprocessing) |
| **Temporal** | `Year` | Integer | YYYY | Observation year (1948–2013) |
| | `Month` | Integer | 1–12 | Observation month |
| | `Period` | Float | YYYY.MM | Continuous period representation (e.g. 1949.01) |
| **Meteorological** | `Max_Temp` | Float | °C | Maximum recorded monthly temperature |
| | `Min_Temp` | Float | °C | Minimum recorded monthly temperature |
| | `Rainfall` | Float | mm | Total monthly precipitation |
| | `Relative_Humidity` | Float | % | Average relative humidity percentage |
| | `Wind_Speed` | Float | Knots / m/s | Mean wind velocity |
| | `Cloud_Coverage` | Float | Oktas | Mean cloud density (0–8 scale) |
| | `Bright_Sunshine` | Float | Hours/day | Mean daily hours of direct sunshine |
| **Spatial / Geo** | `X_COR` | Float | Meters | Projected spatial coordinate X |
| | `Y_COR` | Float | Meters | Projected spatial coordinate Y |
| | `LATITUDE` | Float | Decimal Degrees | Latitude coordinates |
| | `LONGITUDE` | Float | Decimal Degrees | Longitude coordinates |
| | `ALT` | Float | Meters | Station altitude above sea level |
| **Target** | `Flood?` | Binary | 0.0 or 1.0 | Historical flood indicator |

---

## 4. Preprocessing & Data Transformation Methodology

1. **Missing Label Resolution:**
   - In the raw file, `Flood?` contained 16,051 `NaN` values, 4,132 occurrences of `1.0`, and 361 occurrences of `0.0`.
   - All `NaN` entries were imputed as `0.0` under the domain assumption that unflagged months had no flooding events.
2. **Winsorization (Percentile Capping):**
   - 1st and 99th percentiles were calculated and applied to `Max_Temp`, `Rainfall`, `Relative_Humidity`, `Wind_Speed`, and `ALT` to clip unreasonable spikes.
3. **Multi-Pass IQR Outlier Trimming:**
   - 5 iterative passes of Interquartile Range ($[Q1 - 1.5 \times \text{IQR},\ Q3 + 1.5 \times \text{IQR}]$) filtering were executed.
   - Total rows reduced from 20,544 down to 17,560 rows (14,815 non-flood [84.4%] and 2,745 flood [15.6%]).
4. **Logarithmic Transforms for Skewed Distributions:**
   - Several weather parameters exhibited extreme right-skewness:
     - $\text{Rainfall log} = \ln(1 + \text{Rainfall})$
     - $\text{Wind\_Speed log} = \ln(1 + \text{Wind\_Speed})$
     - $\text{ALT log} = \ln(1 + \text{ALT})$
5. **Categorical Encoding:**
   - `Station_Names` was converted using `pd.get_dummies(..., drop_first=True)` creating 31 binary station columns, yielding a total feature matrix $X$ of **44 features**.

---

## 5. Model Performance Benchmark & Results

Three distinct classification algorithms were developed and evaluated on an 80/20 stratified test set (3,512 test samples: 2,963 non-flood, 549 flood).

### Comparative Metric Table

| Metric | Random Forest (Champion) | XGBoost Classifier | Logistic Regression |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **0.98 (98%)** | 0.98 (97.5%) | 0.95 (95%) |
| **Precision (Flood = 1)** | **0.91** | 0.92 | 0.78 |
| **Recall (Flood = 1)** | **0.94** | 0.95 | **0.98** |
| **F1-Score (Flood = 1)** | **0.92** | **0.93** | 0.87 |
| **Precision (Non-Flood = 0)** | 0.99 | 0.99 | 1.00 |
| **Recall (Non-Flood = 0)** | 0.98 | 0.99 | 0.95 |
| **F1-Score (Non-Flood = 0)** | 0.99 | 0.99 | 0.97 |
| **ROC-AUC Score** | **0.995** | **0.995** | 0.993 |
| **RMSE** | 0.141 | **0.123** | — |

### Key Takeaways from Evaluation
- **Random Forest** provided the most stable, generalized balance between false alarms (precision 0.91) and disaster detection (recall 0.94).
- **XGBoost** reached comparable and slightly higher F1-score (0.93) with lower RMSE (0.123), making it a top contender.
- **Logistic Regression** achieved the highest raw flood recall (0.98) but at the expense of precision (0.78), resulting in approximately 22% false alarm rate.

### Top Feature Importances (Random Forest)
1. `Rainfall log` — **36.40%** (Primary driver of water volume)
2. `Cloud_Coverage` — **12.92%** (Indicator of monsoon cloud cover & reduced evaporation)
3. `Relative_Humidity` — **10.65%** (Atmospheric saturation)
4. `Min_Temp` — **9.40%** (Seasonal temperature dynamics)
5. `Bright_Sunshine` — **8.48%** (Direct solar radiation / precipitation inversely correlated)
6. `Max_Temp` — **4.46%**
7. `Wind_Speed log` — **2.71%**
8. Geographic coordinates (`X_COR`, `LONGITUDE`, `LATITUDE`, `ALT log`) — **~12% combined**

---

## 6. Deployment Implementations

1. **Flask Application (`Model/appRF.py`):**
   - Integrates `flood_model.pkl`, `scaler.pkl`, and `model_features.pkl`.
   - Uses `templates/index.html` styled with CSS gradients.
   - Provides an alert card ("🚨 Flood Likely" vs "✅ No Flood Risk") with confidence probability.
2. **Streamlit Application (`Model Deployment/APP.py`):**
   - Uses an end-to-end `sklearn.pipeline.Pipeline` (`flood_pipeline.pkl`).
   - Automatically handles `log1p` transformation, scaling, and inference.
   - Provides interactive input widgets with pre-loaded station mean defaults.

---

## 7. Project Directory Map

```
FloodPredicta/
├── PROJECT_REPORT.md                        # Master project documentation
├── Dataset/
│   ├── flooddata.csv                        # Raw dataset (20,544 rows)
│   └── flood_data_preprocessed.csv          # Cleaned dataset (17,560 rows)
├── Model/
│   ├── AI Lab FinalReport.docx              # Written academic final report
│   ├── appRF.py                             # Flask application backend
│   ├── final_preprocessing.ipynb            # Preprocessing & cleaning notebook
│   ├── final_trainLR.ipynb                  # Logistic Regression notebook
│   ├── final_trainRF.ipynb                  # Random Forest training notebook
│   ├── XgBoost.ipynb                        # XGBoost training notebook
│   ├── flood_model.pkl                      # Serialized Random Forest model
│   ├── model_features.pkl                   # 44 feature name list
│   ├── scaler.pkl                           # Serialized StandardScaler
│   └── templates/
│       └── index.html                       # Flask web page UI template
├── Model Deployment/
│   ├── APP.py                               # Streamlit web app
│   └── flood_pipeline.pkl                   # End-to-end scikit-learn Pipeline
├── Model Train/                             # (Redundant copy of Model notebooks)
│   ├── final_preprocessing.ipynb
│   ├── final_trainLR.ipynb
│   ├── final_trainRF.ipynb
│   └── XgBoost.ipynb
└── Model_Deployment/                        # (Duplicate folder containing APP.py and index.html)
```

---

## 8. Critical Code Issues Found & How to Fix Them

1. **Flask Prediction Dimension Crash (`Model/appRF.py`, Line 56):**
   - **Bug:** `scaler.pkl` was fitted on all 44 features. Line 56 executes `input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])`, passing only 13 columns.
   - **Error:** `ValueError: The feature names should match those that were passed during fit.`
   - **Resolution:** Change line 56 to pass the entire encoded DataFrame `scaler.transform(input_df[model_features])`.

2. **Streamlit Variable Name Mismatch (`Model Deployment/APP.py`, Line 41):**
   - **Bug:** The dictionary keys include `"Humidity"`, but the training dataset column is named `"Relative_Humidity"`.
   - **Error:** User humidity inputs are ignored and passed as `0.0`.
   - **Resolution:** Rename `"Humidity"` to `"Relative_Humidity"` in lines 41 and 61–62.

3. **Pipeline Pickle Deserialization Dependency:**
   - **Bug:** `flood_pipeline.pkl` requires `log_and_rename` to be defined in `__main__` prior to unpickling.
   - **Resolution:** Extract custom transformer functions into a shared `pipeline_utils.py` module and import it when serializing.
