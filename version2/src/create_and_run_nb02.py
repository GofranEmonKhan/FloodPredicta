import json
import sys
import os
import io
import ast
import contextlib
import pandas as pd
import numpy as np

notebook_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
os.chdir(notebook_dir)

# Minimal human-vibed cells
cells_spec = [
    ("markdown", "# Feature Engineering: Lags, Rolling Windows & Hydrology"),

    ("code", """import os
import numpy as np
import pandas as pd

# read preprocessed dataset
data_path = os.path.join('..', 'data', 'processed', 'flood_data_preprocessed.csv')
df = pd.read_csv(data_path)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
df.head(3)"""),

    ("markdown", "### 1. Antecedent rainfall lags & rolling saturation"),

    ("code", """# make sure data is ordered chronologically per station
df = df.sort_values(by=['Station_Names', 'Year', 'Month']).reset_index(drop=True)

# 1-month and 2-month lag of rainfall
df['Rainfall_Lag1'] = df.groupby('Station_Names')['Rainfall'].shift(1).fillna(0)
df['Rainfall_Lag2'] = df.groupby('Station_Names')['Rainfall'].shift(2).fillna(0)

# 3-month rolling rainfall sum for cumulative saturation
df['Rainfall_Rolling3M'] = df.groupby('Station_Names')['Rainfall'].transform(
    lambda s: s.rolling(3, min_periods=1).sum()
)

# log transform the new rainfall metrics as well
df['Rainfall_Lag1_log'] = np.log1p(df['Rainfall_Lag1'])
df['Rainfall_Lag2_log'] = np.log1p(df['Rainfall_Lag2'])
df['Rainfall_Rolling3M_log'] = np.log1p(df['Rainfall_Rolling3M'])

print("Added rainfall lag and rolling features.")
df[['Station_Names', 'Year', 'Month', 'Rainfall', 'Rainfall_Lag1', 'Rainfall_Rolling3M']].head(5)"""),

    ("markdown", "### 2. Runoff risk & moisture interaction"),

    ("code", """# low elevation with high rain pools faster
df['Runoff_Ratio'] = df['Rainfall'] / (df['ALT'] + 1.0)
df['Runoff_Ratio_log'] = np.log1p(df['Runoff_Ratio'])

# moisture saturation proxy from humidity and clouds
df['Moisture_Index'] = (df['Relative_Humidity'] * df['Cloud_Coverage']) / 100.0

print("Added runoff ratio and moisture index.")
df[['Rainfall', 'ALT', 'Runoff_Ratio', 'Relative_Humidity', 'Cloud_Coverage', 'Moisture_Index']].head(3)"""),

    ("markdown", "### 3. Cyclical seasonality & monsoon flag"),

    ("code", """# cyclical encoding so month 12 wraps to month 1
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# monsoon season flag (June to September)
df['Is_Monsoon'] = df['Month'].isin([6, 7, 8, 9]).astype(int)

print("Added seasonal indicators.")
print("Monsoon records:", (df['Is_Monsoon'] == 1).sum(), "out of", len(df))"""),

    ("markdown", "### 4. Station encoding & save dataset"),

    ("code", """# one-hot encode station names for tree models
station_dummies = pd.get_dummies(df['Station_Names'], prefix='Station', drop_first=True, dtype=int)
df_featured = pd.concat([df, station_dummies], axis=1)

print("Final dataset shape:", df_featured.shape)

out_path = os.path.join('..', 'data', 'processed', 'flood_data_featured.csv')
df_featured.to_csv(out_path, index=False)
print(f"Saved to: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")""")
]

# Execute cells and capture output
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
        stdout_buf = io.StringIO()
        outputs = []
        
        parsed = ast.parse(source)
        body = parsed.body
        last_expr = None
        if body and isinstance(body[-1], ast.Expr):
            last_expr = ast.Expression(body[-1].value)
            body = body[:-1]
        
        stmt_code = compile(ast.Module(body=body, type_ignores=[]), '<cell>', 'exec')
        
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
            try:
                exec(stmt_code, execution_globals)
                if last_expr is not None:
                    expr_code = compile(last_expr, '<cell_expr>', 'eval')
                    res = eval(expr_code, execution_globals)
                    if res is not None:
                        res_str = repr(res) if isinstance(res, (pd.DataFrame, pd.Series)) else str(res)
                        outputs.append({
                            "data": {"text/plain": [line + "\n" for line in res_str.split("\n")]},
                            "execution_count": execution_counter,
                            "metadata": {},
                            "output_type": "execute_result"
                        })
            except Exception as e:
                print(f"Error in Cell {execution_counter}: {e}", file=sys.stderr)

        stdout_text = stdout_buf.getvalue()
        if stdout_text:
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
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

target_nb = os.path.join(notebook_dir, "02_feature_engineering.ipynb")
with open(target_nb, "w", encoding="utf-8") as f:
    json.dump(notebook_structure, f, indent=1)

print(f"02_feature_engineering.ipynb created with {execution_counter - 1} executed cells.")
