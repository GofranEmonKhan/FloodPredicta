import os
import sys
import json
import io
import base64
import shutil
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notebooks_root = os.path.join(base_dir, 'notebooks')
models_root = os.path.join(base_dir, 'models')
reports_root = os.path.join(base_dir, 'reports')

# Helper function to generate executed notebook
def generate_notebook(nb_path, title, code_cells_info):
    execution_globals = {'__name__': '__main__'}
    cells = []
    exec_count = 1

    if title:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in title.split("\n")]
        })

    for item in code_cells_info:
        md_text = item.get('md')
        if md_text:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in md_text.split("\n")]
            })

        code_text = item['code']
        stdout_buf = io.StringIO()
        outputs = []
        
        plt.close('all')
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
            try:
                exec(code_text, execution_globals)
            except Exception as e:
                print(f"Execution Error: {e}", file=sys.stderr)

        stdout_val = stdout_buf.getvalue()
        if stdout_val:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_val.split("\n")]
            })

        figs = [plt.figure(n) for n in plt.get_fignums()]
        if figs:
            for fig in figs:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                b64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                outputs.append({
                    "data": {
                        "image/png": b64_img,
                        "text/plain": ["<Figure size ...>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
            plt.close('all')

        cells.append({
            "cell_type": "code",
            "execution_count": exec_count,
            "metadata": {},
            "outputs": outputs,
            "source": [line + "\n" for line in code_text.split("\n")]
        })
        exec_count += 1

    nb_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=1)
    print(f"Generated: {nb_path}")

# Function to build 5 models for a specific chronological split regime
def build_chronological_regime(regime_name, split_code, title_suffix):
    reg_nb_dir = os.path.join(notebooks_root, regime_name)
    reg_m_dir = os.path.join(models_root, regime_name)
    reg_r_dir = os.path.join(reports_root, regime_name)

    os.makedirs(reg_nb_dir, exist_ok=True)
    os.makedirs(reg_m_dir, exist_ok=True)
    os.makedirs(reg_r_dir, exist_ok=True)

    os.chdir(reg_nb_dir)

    models_config = [
        ('train_xgboost.ipynb', 'XGBoost Classifier', """import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.08,
    scale_pos_weight=3.97, random_state=42, eval_metric='logloss'
)""", 'xgboost_model', True),

        ('train_histgradientboosting.ipynb', 'HistGradientBoosting Classifier', """from sklearn.ensemble import HistGradientBoostingClassifier
model = HistGradientBoostingClassifier(
    max_iter=200, max_depth=7, learning_rate=0.08,
    class_weight='balanced', random_state=42
)""", 'hist_gradient_boosting_model', False),

        ('train_randomforest.ipynb', 'Random Forest Classifier', """from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=200, max_depth=None,
    class_weight='balanced', random_state=42, n_jobs=-1
)""", 'random_forest_model', True),

        ('train_extratrees.ipynb', 'Extra Trees Classifier', """from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(
    n_estimators=200, max_depth=None,
    class_weight='balanced', random_state=42, n_jobs=-1
)""", 'extra_trees_model', True),

        ('train_stacking_ensemble.ipynb', 'Stacking Ensemble Classifier', """from sklearn.ensemble import StackingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

base_estimators = [
    ('xgb', xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, scale_pos_weight=3.97, random_state=42, eval_metric='logloss')),
    ('rf', RandomForestClassifier(n_estimators=150, max_depth=None, class_weight='balanced', random_state=42, n_jobs=-1)),
    ('hgb', HistGradientBoostingClassifier(max_iter=150, max_depth=7, learning_rate=0.08, class_weight='balanced', random_state=42))
]
model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
    cv=3, n_jobs=1
)""", 'stacking_ensemble_model', False)
    ]

    for nb_file, model_title, model_init_code, save_name, has_imp in models_config:
        train_eval_code = f"""{model_init_code}

# fit model on chronological training split
model.fit(X_train, y_train)

# validation threshold tuning (0.10 to 0.90)
val_probs = model.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.05)
thresh_results = []
best_thresh, best_val_f1 = 0.50, 0.0

for t in thresholds:
    p = (val_probs >= t).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, p, average='binary', zero_division=0)
    thresh_results.append([round(t, 2), prec, rec, f1])
    if f1 > best_val_f1:
        best_val_f1, best_thresh = f1, t

results_df = pd.DataFrame(thresh_results, columns=['Threshold', 'Precision', 'Recall', 'F1-score'])
print("Validation Threshold Tuning Results:")
print(results_df.to_string(index=False, float_format=lambda x: f"{{x:.4f}}"))
print(f"\\nOptimal Validation Threshold: {{best_thresh:.2f}} (Validation F1: {{best_val_f1:.4f}})")

# evaluate on out-of-time test set using default threshold (0.50)
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.50).astype(int)
roc_auc = roc_auc_score(y_test, test_probs)

print("\\nClassification Report (Test Set - Threshold 0.50):")
print(classification_report(y_test, test_preds, digits=4))
print(f"ROC-AUC Score: {{roc_auc:.4f}}")

# save model artifact
models_out = os.path.join('..', '..', 'models', '{regime_name}')
os.makedirs(models_out, exist_ok=True)
joblib.dump(model, os.path.join(models_out, '{save_name}.pkl'))
print(f"Model saved to models/{regime_name}/{save_name}.pkl")"""

        cm_code = f"""# Plot 1: Confusion Matrix
reports_out = os.path.join('..', '..', 'reports', '{regime_name}')
os.makedirs(reports_out, exist_ok=True)

cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix ({regime_name})')
plt.tight_layout()
plt.savefig(os.path.join(reports_out, '{save_name}_confusion_matrix.png'), dpi=150)
plt.show()"""

        roc_code = f"""# Plot 2: Receiver Operating Characteristic (ROC) Curve
reports_out = os.path.join('..', '..', 'reports', '{regime_name}')
fpr, tpr, _ = roc_curve(y_test, test_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {{roc_auc:.3f}})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(reports_out, '{save_name}_roc_curve.png'), dpi=150)
plt.show()"""

        prob_code = f"""# Plot 3: Predicted Probability Distribution
reports_out = os.path.join('..', '..', 'reports', '{regime_name}')
plt.figure(figsize=(8, 5))
sns.histplot(test_probs[y_test == 0], color='skyblue', label='Non-Flood (0)', bins=30, stat='density', kde=True)
sns.histplot(test_probs[y_test == 1], color='salmon', label='Flood (1)', bins=30, stat='density', kde=True)
plt.xlabel('Predicted Probability of Flood')
plt.ylabel('Density')
plt.title('Predicted Probability Distribution')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(reports_out, '{save_name}_probability_distribution.png'), dpi=150)
plt.show()"""

        if has_imp:
            imp_code = f"""# Plot 4: Top 15 Feature Importances
reports_out = os.path.join('..', '..', 'reports', '{regime_name}')
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
importances[:15].plot(kind='bar', color='steelblue')
plt.title('Top 15 Feature Importances')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(reports_out, '{save_name}_feature_importance.png'), dpi=150)
plt.show()"""
        else:
            imp_code = f"""# Plot 4: Top 15 Feature Importances (Permutation Importances)
reports_out = os.path.join('..', '..', 'reports', '{regime_name}')
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)
perm_imp = pd.Series(result.importances_mean, index=feature_cols).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
perm_imp[:15].plot(kind='bar', color='darkorange')
plt.title('Top 15 Permutation Feature Importances')
plt.ylabel('Mean Importance Decrease')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(reports_out, '{save_name}_feature_importance.png'), dpi=150)
plt.show()"""

        target_nb_path = os.path.join(reg_nb_dir, nb_file)
        generate_notebook(
            target_nb_path,
            f"# {model_title}: {title_suffix}",
            [
                {'md': f"### 1. Data Loading & Chronological Partitioning ({title_suffix})", 'code': split_code},
                {'md': f"### 2. Train {model_title}, Validation Tuning & Test Evaluation", 'code': train_eval_code},
                {'md': "### 3. Graph 1: Confusion Matrix Heatmap", 'code': cm_code},
                {'md': "### 4. Graph 2: Receiver Operating Characteristic (ROC) Curve", 'code': roc_code},
                {'md': "### 5. Graph 3: Predicted Probability Distribution", 'code': prob_code},
                {'md': "### 6. Graph 4: Top 15 Feature Importances", 'code': imp_code}
            ]
        )

    # Save feature_columns.pkl
    df_temp = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'flood_data_featured.csv'), nrows=1)
    feature_cols = [c for c in df_temp.columns if c not in ['Station_Names', 'Year', 'Month', 'Period', 'Flood?']]
    joblib.dump(feature_cols, os.path.join(reg_m_dir, 'feature_columns.pkl'))

# 1. Chronological Split Regime A: 70-15-15 (Train <= 2003, Val 2004-2008, Test >= 2009)
split_code_70_15_15 = """import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    roc_auc_score, precision_recall_fscore_support
)

# load engineered dataset
data_path = os.path.join('..', '..', 'data', 'processed', 'flood_data_featured.csv')
df = pd.read_csv(data_path)

# separate features and target
ignore_cols = ['Station_Names', 'Year', 'Month', 'Period', 'Flood?']
feature_cols = [c for c in df.columns if c not in ignore_cols]

# Chronological Split (Train <= 2003, Val 2004-2008, Test >= 2009):
train_mask = df['Year'] <= 2003
val_mask = (df['Year'] >= 2004) & (df['Year'] <= 2008)
test_mask = df['Year'] >= 2009

X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, 'Flood?']
X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, 'Flood?']
X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, 'Flood?']

print("Chronological Partitioning (Train <= 2003, Val 2004-2008, Test >= 2009):")
print(f"  - Train Set:      {X_train.shape[0]:,} samples (1948-2003) | Floods: {int(y_train.sum()):,}")
print(f"  - Validation Set: {X_val.shape[0]:,} samples (2004-2008) | Floods: {int(y_val.sum()):,}")
print(f"  - Test Set:       {X_test.shape[0]:,} samples (2009-2013) | Floods: {int(y_test.sum()):,}")
print(f"Total Features:     {len(feature_cols)}")"""

# 2. Chronological Split Regime B: 70-20-10 (Train <= 1997, Val 1998-2008, Test >= 2009)
split_code_70_20_10 = """import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    roc_auc_score, precision_recall_fscore_support
)

# load engineered dataset
data_path = os.path.join('..', '..', 'data', 'processed', 'flood_data_featured.csv')
df = pd.read_csv(data_path)

# separate features and target
ignore_cols = ['Station_Names', 'Year', 'Month', 'Period', 'Flood?']
feature_cols = [c for c in df.columns if c not in ignore_cols]

# Chronological 70-20-10 Split (Train <= 1997, Val 1998-2008, Test >= 2009):
train_mask = df['Year'] <= 1997
val_mask = (df['Year'] >= 1998) & (df['Year'] <= 2008)
test_mask = df['Year'] >= 2009

X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, 'Flood?']
X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, 'Flood?']
X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, 'Flood?']

print("Chronological 70-20-10 Partitioning:")
print(f"  - Train Set (70%):      {X_train.shape[0]:,} samples (1948-1997) | Floods: {int(y_train.sum()):,}")
print(f"  - Validation Set (20%): {X_val.shape[0]:,} samples (1998-2008) | Floods: {int(y_val.sum()):,}")
print(f"  - Test Set (10%):       {X_test.shape[0]:,} samples (2009-2013) | Floods: {int(y_test.sum()):,}")
print(f"Total Features:           {len(feature_cols)}")"""

print("\n--- Generating split_chronological_70_15_15 ---")
build_chronological_regime(
    'split_chronological_70_15_15',
    split_code_70_15_15,
    'Chronological Split (Train <= 2003, Val 2004-2008, Test >= 2009)'
)

print("\n--- Generating split_chronological_70_20_10 ---")
build_chronological_regime(
    'split_chronological_70_20_10',
    split_code_70_20_10,
    'Chronological 70-20-10 Split (Train <= 1997, Val 1998-2008, Test >= 2009)'
)

print("\nBoth chronological regimes successfully built and preserved!")
