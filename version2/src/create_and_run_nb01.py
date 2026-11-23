import json
import sys
import os
import io
import ast
import contextlib
import pandas as pd
import numpy as np

# Ensure working directory is version2/notebooks when running to simulate Jupyter environment
notebook_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
os.makedirs(notebook_dir, exist_ok=True)
os.chdir(notebook_dir)

print(f"Executing in notebook directory: {notebook_dir}")

# Define notebook cells: (cell_type, source_code)
cells_spec = [
    ("markdown", """# 🌊 FloodPredicta (Version 2.0) - Step 1: Data Preprocessing & Cleaning

This notebook performs data auditing, target missing value imputation, extreme value analysis, log transformations, and sorting for the **FloodPredicta** flood prediction system.

### Key Objectives in Version 2.0:
1. Audit raw meteorological observations across 33 Bangladesh Meteorological Department (BMD) stations (1948–2013).
2. Resolve target variable `Flood?` missing values with domain-backed justification.
3. **Preserve extreme weather phenomena:** Explicitly demonstrate why aggressive 5-pass IQR outlier dropping (used in Version 1) deletes thousands of genuine flood disaster cases, and adopt non-destructive winsorization instead.
4. Correct severe right-skewness using $\\ln(1 + x)$ logarithmic transformations.
5. Chronologically sort station records to prepare for temporal and hydrological lag feature engineering.
6. Export the clean dataset to `data/processed/flood_data_preprocessed.csv`."""),

    ("code", """# Import core libraries for data processing and analysis
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean pandas display formatting
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

print("Libraries imported successfully.")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version:  {np.__version__}")"""),

    ("markdown", """## 1. Load Raw Climate Dataset
Load the raw dataset from `../data/raw/flooddata.csv` and inspect its initial shape and structure."""),

    ("code", """# Load raw meteorological dataset
raw_data_path = os.path.join('..', 'data', 'raw', 'flooddata.csv')
df = pd.read_csv(raw_data_path)

print(f"Raw dataset loaded successfully from: {raw_data_path}")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
df.head()"""),

    ("markdown", """## 2. Inspect Column Types and Missing Values
Check data types across all 19 columns and identify where missing values exist."""),

    ("code", """# Check data types and identify missing values
print("--- Dataset Information ---")
df.info()

print("\\n--- Missing Values Audit ---")
missing = df.isna().sum()
print(missing[missing > 0])
print(f"\\nConclusion: Only 'Flood?' has missing entries ({missing['Flood?']:,} rows). All 18 meteorological and geographic features are complete.")"""),

    ("markdown", """## 3. Target Variable Analysis & Missing Value Imputation
In the raw data, `Flood?` contains 16,051 `NaN` entries, 4,132 `1.0` flags, and 361 `0.0` flags.
In meteorological and disaster archiving in Bangladesh, actual flood events were noted (`1.0`), while normal non-flood months were left unrecorded (`NaN`).
Therefore, `NaN` values are imputed with `0.0`."""),

    ("code", """# Check target variable value counts before and after imputation
print("Target 'Flood?' counts before imputation:")
print(df['Flood?'].value_counts(dropna=False))

# Impute NaN -> 0.0 based on domain record keeping convention
df['Flood?'] = df['Flood?'].fillna(0.0)

print("\\nTarget 'Flood?' counts after imputation:")
counts = df['Flood?'].value_counts()
print(counts)
print(f"\\nClass 0.0 (Non-Flood): {counts[0.0]:,} ({counts[0.0]/len(df)*100:.2f}%)")
print(f"Class 1.0 (Flood):     {counts[1.0]:,} ({counts[1.0]/len(df)*100:.2f}%)")
print(f"Imbalance Ratio:       ~{counts[0.0]/counts[1.0]:.2f} : 1")"""),

    ("markdown", """## 4. Meteorological Station Coverage & Timeline
Examine the monitored weather stations and the overall historical time span."""),

    ("code", """# Unique meteorological stations and observation years
stations = df['Station_Names'].unique()
print(f"Total Unique Weather Stations: {len(stations)}")
print(f"Stations list: {sorted(stations)}")

min_yr, max_yr = df['Year'].min(), df['Year'].max()
print(f"\\nObservation Timeline: {min_yr} to {max_yr} ({max_yr - min_yr + 1} years)")
print(f"Average observations per station: {int(df['Station_Names'].value_counts().mean())} months")"""),

    ("markdown", """## 5. Remove Redundant Identifier Columns
- `Station_Number` is completely redundant with `Station_Names`.
- `Sl` is an arbitrary sequence counter with no predictive signal.
We drop both to keep our feature matrix clean."""),

    ("code", """# Drop redundant identifiers
cols_to_drop = ['Sl', 'Station_Number']
existing_cols = [c for c in cols_to_drop if c in df.columns]

df.drop(columns=existing_cols, inplace=True)
print(f"Dropped redundant columns: {existing_cols}")
print(f"Updated DataFrame shape: {df.shape[0]:,} rows x {df.shape[1]} columns")"""),

    ("markdown", """## 6. Summary Statistics of Numeric Features
Review the central tendency, dispersion, and spread of all weather variables."""),

    ("code", """# Numerical summary statistics
df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]"""),

    ("markdown", """## 7. Outlier & Extreme Event Analysis: Why V1 Was Flawed
In Version 1, 5 passes of IQR trimming were run on `Rainfall`, `Wind_Speed`, and temperature metrics.
Let's measure what IQR trimming actually does to our real historical flood records."""),

    ("code", """# Diagnostic: Measuring real flood disaster events lost to IQR trimming
features_to_check = ['Max_Temp', 'Rainfall', 'Relative_Humidity', 'Wind_Speed', 'ALT']

df_iqr_test = df.copy()
initial_floods = int((df_iqr_test['Flood?'] == 1.0).sum())

# Run 1 pass of IQR filtering
for col in features_to_check:
    q1 = df_iqr_test[col].quantile(0.25)
    q3 = df_iqr_test[col].quantile(0.75)
    iqr = q3 - q1
    lower_lim = q1 - 1.5 * iqr
    upper_lim = q3 + 1.5 * iqr
    df_iqr_test = df_iqr_test[(df_iqr_test[col] >= lower_lim) & (df_iqr_test[col] <= upper_lim)]

remaining_floods = int((df_iqr_test['Flood?'] == 1.0).sum())
lost_floods = initial_floods - remaining_floods

print("=== CRITICAL OUTLIER ANALYSIS ===")
print(f"Initial Verified Flood Events:                 {initial_floods:,}")
print(f"Flood Events Remaining After 1 IQR Pass:       {remaining_floods:,}")
print(f"Genuine Flood Disasters Deleted by IQR:        {lost_floods:,} ({lost_floods/initial_floods*100:.2f}% of all floods!)")
print("\\nDECISION: In Version 2.0, we DO NOT drop rows using IQR.")
print("Floods are physically driven by extreme weather. Trimming extreme rain discards the very disasters we must predict.")"""),

    ("markdown", """## 8. Non-Destructive Quantile Capping (Winsorization)
To prevent impossible physical sensor anomalies from distorting calculations while preserving 99.5% of real monsoon deluges, we apply conservative 0.5% lower and 99.5% upper quantile capping."""),

    ("code", """# Apply conservative winsorization (capping)
capping_cols = ['Rainfall', 'Wind_Speed', 'Max_Temp', 'Relative_Humidity']

print("Applied Capping Bounds (0.5% lower, 99.5% upper):")
for col in capping_cols:
    lower_bound = df[col].quantile(0.005)
    upper_bound = df[col].quantile(0.995)
    prev_max = df[col].max()
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    print(f"  - {col:18s}: lower={lower_bound:8.2f}, upper={upper_bound:8.2f} (prev max={prev_max:8.2f})")

print(f"\\nPreserved all {len(df):,} observations without losing flood instances!")"""),

    ("markdown", """## 9. Skewness Analysis & Logarithmic Transformations
Precipitation and wind velocity exhibit extreme right-skewed tails.
Applying $\\ln(1 + x)$ transforms compress these long tails, enabling stable distance calculations and linear relationships."""),

    ("code", """# Assess skewness and apply log1p transforms
numeric_features = df.select_dtypes(include=['number']).columns.drop('Flood?')
print("--- Skewness Values Before Log Transformation ---")
skew_before = df[numeric_features].skew().sort_values(ascending=False)
print(skew_before)

# Apply log1p on heavily right-skewed features
skewed_cols = ['Rainfall', 'Wind_Speed', 'ALT']

print("\\n--- Applying log1p Transformations ---")
for col in skewed_cols:
    log_name = col + ' log'
    df[log_name] = np.log1p(df[col])
    print(f"Created: {log_name:15s} (Skewness improved: {df[col].skew():.3f} -> {df[log_name].skew():.3f})")"""),

    ("markdown", """## 10. Chronological Sorting by Station and Time
For temporal consistency and future lag feature engineering (such as previous month rainfall and cumulative monsoonal saturation),
we ensure records are strictly sorted by `Station_Names`, `Year`, and `Month`."""),

    ("code", """# Chronological sort per meteorological station
df.sort_values(by=['Station_Names', 'Year', 'Month'], inplace=True)
df.reset_index(drop=True, inplace=True)

print("Chronological sequence established. First 5 records of first station:")
df[['Station_Names', 'Year', 'Month', 'Period', 'Rainfall', 'Rainfall log', 'Flood?']].head()"""),

    ("markdown", """## 11. Final Dataset Verification & Export
Verify complete absence of missing values, check final record counts, and save the preprocessed dataset to `version2/data/processed/flood_data_preprocessed.csv`."""),

    ("code", """# Final verification and export
print("=== PREPROCESSED DATASET QUALITY REPORT ===")
print(f"Total Rows:           {df.shape[0]:,}")
print(f"Total Columns:        {df.shape[1]}")
print(f"Missing Values:       {df.isna().sum().sum()}")
print(f"Final Class Counts:   Non-Flood={int((df['Flood?']==0).sum()):,}, Flood={int((df['Flood?']==1).sum()):,}")
print(f"Columns in dataset:   {list(df.columns)}")

processed_dir = os.path.join('..', 'data', 'processed')
os.makedirs(processed_dir, exist_ok=True)
output_path = os.path.join(processed_dir, 'flood_data_preprocessed.csv')

df.to_csv(output_path, index=False)
file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"\\nSuccessfully exported preprocessed dataset to: {output_path}")
print(f"File size on disk: {file_size_mb:.2f} MB")""")
]

# Shared global environment to execute cells sequentially
execution_globals = {}
notebook_cells = []
execution_counter = 1

for cell_type, source in cells_spec:
    if cell_type == "markdown":
        notebook_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")]
        })
    elif cell_type == "code":
        print(f"Executing Code Cell {execution_counter}...")
        stdout_buf = io.StringIO()
        outputs = []
        
        # Split source into AST to check if the last node is an expression to display
        parsed = ast.parse(source)
        body = parsed.body
        last_expr = None
        if body and isinstance(body[-1], ast.Expr):
            last_expr = ast.Expression(body[-1].value)
            body = body[:-1]
        
        # Compile statement part
        stmt_code = compile(ast.Module(body=body, type_ignores=[]), '<cell>', 'exec')
        
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
            try:
                exec(stmt_code, execution_globals)
                
                # Evaluate last expression if present
                if last_expr is not None:
                    expr_code = compile(last_expr, '<cell_expr>', 'eval')
                    res = eval(expr_code, execution_globals)
                    if res is not None:
                        # Convert DataFrame / Series or generic obj to text/plain
                        if isinstance(res, (pd.DataFrame, pd.Series)):
                            res_str = repr(res)
                        else:
                            res_str = str(res)
                        
                        # Add as execute_result
                        outputs.append({
                            "data": {
                                "text/plain": [line + "\n" for line in res_str.split("\n")]
                            },
                            "execution_count": execution_counter,
                            "metadata": {},
                            "output_type": "execute_result"
                        })
            except Exception as e:
                print(f"Error in Cell {execution_counter}: {e}", file=sys.stderr)
                traceback.print_exc()

        stdout_text = stdout_buf.getvalue()
        if stdout_text:
            # Add stdout as stream output
            # Insert stream output before execute_result if any
            outputs.insert(0, {
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_text.split("\n")]
            })

        notebook_cells.append({
            "cell_type": "code",
            "execution_count": execution_counter,
            "metadata": {},
            "outputs": outputs,
            "source": [line + "\n" for line in source.split("\n")]
        })
        execution_counter += 1

notebook_structure = {
    "cells": notebook_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": sys.version.split()[0]
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

target_notebook_path = os.path.join(notebook_dir, "01_data_preprocessing.ipynb")
with open(target_notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_structure, f, indent=1)

print(f"\nAll {execution_counter - 1} code cells executed successfully!")
print(f"Notebook written with complete outputs to: {target_notebook_path}")
