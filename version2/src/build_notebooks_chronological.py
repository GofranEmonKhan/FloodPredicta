import os
import sys
import json
import io
import base64
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
notebooks_dir = os.path.join(base_dir, 'notebooks', 'split_chronological')
models_dir = os.path.join(base_dir, 'models', 'split_chronological')
reports_dir = os.path.join(base_dir, 'reports', 'split_chronological')

os.makedirs(notebooks_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# Set working directory to chronological notebooks directory
os.chdir(notebooks_dir)

def create_executed_notebook(nb_filename, title, code_cells_info):
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

        # capture each standalone plot generated as display_data
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

    nb_path = os.path.join(notebooks_dir, nb_filename)
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=1)
    print(f"Created and executed: {nb_filename}")
    return nb_path

print("Starting generation of chronological model notebooks (70-20-10 split: Train <= 1997, Val 1998-2008, Test >= 2009)...")

# Chronological 70-20-10 split code snippet
split_data_code = """import os
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

# 70-20-10 Chronological Split:
# - Train Set (70%):      Year <= 1997 (1948-1997: 14,352 samples, 69.9%)
# - Validation Set (20%): 1998 <= Year <= 2008 (4,224 samples, 20.6%)
# - Test Set (10%):       Year >= 2009 (2009-2013: 1,968 samples, 9.6%)
train_mask = df['Year'] <= 1997
val_mask = (df['Year'] >= 1998) & (df['Year'] <= 2008)
test_mask = df['Year'] >= 2009

X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, 'Flood?']
X_val, y_val = df.loc[val_mask, feature_cols], df.loc[val_mask, 'Flood?']
X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, 'Flood?']

print("Chronological 70-20-10 Partitioning Summary:")
print(f"  - Train Set (70%):      {X_train.shape[0]:,} samples (1948-1997) | Floods: {int(y_train.sum()):,}")
print(f"  - Validation Set (20%): {X_val.shape[0]:,} samples (1998-2008) | Floods: {int(y_val.sum()):,}")
print(f"  - Test Set (10%):       {X_test.shape[0]:,} samples (2009-2013) | Floods: {int(y_test.sum()):,}")
print(f"Total Features:           {len(feature_cols)}")"""

# Standard evaluation and 4 plot template generator
def build_model_snippets(model_import_and_init, model_save_name, has_feature_importances=True):
    train_and_eval_code = f"""{model_import_and_init}

# fit model on chronological training split (70%)
model.fit(X_train, y_train)

# validation threshold tuning (0.10 to 0.90) on 20% validation split (1998-2008)
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
print("Validation Threshold Tuning Results (1998-2008):")
print(results_df.to_string(index=False, float_format=lambda x: f"{{x:.4f}}"))
print(f"\\nOptimal Validation Threshold: {{best_thresh:.2f}} (Validation F1: {{best_val_f1:.4f}})")

# evaluate on out-of-time test set (2009-2013, 10%) using default threshold (0.50)
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.50).astype(int)
roc_auc = roc_auc_score(y_test, test_probs)

print("\\nClassification Report (Test Set 2009-2013 - Threshold 0.50):")
print(classification_report(y_test, test_preds, digits=4))
print(f"ROC-AUC Score: {{roc_auc:.4f}}")

# save model artifact
models_out = os.path.join('..', '..', 'models', 'split_chronological')
os.makedirs(models_out, exist_ok=True)
joblib.dump(model, os.path.join(models_out, '{model_save_name}.pkl'))
print(f"Model saved to models/split_chronological/{model_save_name}.pkl")"""

    plot_cm_code = f"""# Plot 1: Confusion Matrix
reports_out = os.path.join('..', '..', 'reports', 'split_chronological')
os.makedirs(reports_out, exist_ok=True)

cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Chronological Test (2009-2013)')
plt.tight_layout()
plt.savefig(os.path.join(reports_out, '{model_save_name}_confusion_matrix.png'), dpi=150)
plt.show()"""

    plot_roc_code = f"""# Plot 2: Receiver Operating Characteristic (ROC) Curve
reports_out = os.path.join('..', '..', 'reports', 'split_chronological')
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
plt.savefig(os.path.join(reports_out, '{model_save_name}_roc_curve.png'), dpi=150)
plt.show()"""

    plot_prob_code = f"""# Plot 3: Predicted Probability Distribution
reports_out = os.path.join('..', '..', 'reports', 'split_chronological')
plt.figure(figsize=(8, 5))
sns.histplot(test_probs[y_test == 0], color='skyblue', label='Non-Flood (0)', bins=30, stat='density', kde=True)
sns.histplot(test_probs[y_test == 1], color='salmon', label='Flood (1)', bins=30, stat='density', kde=True)
plt.xlabel('Predicted Probability of Flood')
plt.ylabel('Density')
plt.title('Predicted Probability Distribution (Chronological Test)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(reports_out, '{model_save_name}_probability_distribution.png'), dpi=150)
plt.show()"""

    if has_feature_importances:
        plot_imp_code = f"""# Plot 4: Top 15 Feature Importances
reports_out = os.path.join('..', '..', 'reports', 'split_chronological')
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
importances[:15].plot(kind='bar', color='steelblue')
plt.title('Top 15 Feature Importances (Chronological Model)')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(reports_out, '{model_save_name}_feature_importance.png'), dpi=150)
plt.show()"""
    else:
        plot_imp_code = f"""# Plot 4: Top 15 Feature Importances (Permutation Importances)
reports_out = os.path.join('..', '..', 'reports', 'split_chronological')
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)
perm_imp = pd.Series(result.importances_mean, index=feature_cols).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
perm_imp[:15].plot(kind='bar', color='darkorange')
plt.title('Top 15 Permutation Feature Importances (Chronological Model)')
plt.ylabel('Mean Importance Decrease')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(reports_out, '{model_save_name}_feature_importance.png'), dpi=150)
plt.show()"""

    return train_and_eval_code, plot_cm_code, plot_roc_code, plot_prob_code, plot_imp_code

# ==========================================
# 1. XGBoost
# ==========================================
xgb_init = """import xgboost as xgb

# XGBoost with scale_pos_weight
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    scale_pos_weight=3.97,
    random_state=42,
    eval_metric='logloss'
)"""

c1, c2, c3, c4, c5 = build_model_snippets(xgb_init, 'xgboost_model', has_feature_importances=True)
create_executed_notebook(
    'train_xgboost.ipynb',
    '# XGBoost Classifier: Chronological 70-20-10 Split (Train <= 1997, Val 1998-2008, Test >= 2009)',
    [
        {'md': '### 1. Chronological 70-20-10 Split', 'code': split_data_code},
        {'md': '### 2. Train XGBoost, Threshold Tuning on Validation, and Test Evaluation', 'code': c1},
        {'md': '### 3. Graph 1: Confusion Matrix Heatmap', 'code': c2},
        {'md': '### 4. Graph 2: Receiver Operating Characteristic (ROC) Curve', 'code': c3},
        {'md': '### 5. Graph 3: Predicted Probability Distribution', 'code': c4},
        {'md': '### 6. Graph 4: Top 15 Feature Importances', 'code': c5}
    ]
)

# ==========================================
# 2. HistGradientBoosting
# ==========================================
hgb_init = """from sklearn.ensemble import HistGradientBoostingClassifier

# HistGradientBoosting with balanced class weights
model = HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=7,
    learning_rate=0.08,
    class_weight='balanced',
    random_state=42
)"""

c1, c2, c3, c4, c5 = build_model_snippets(hgb_init, 'hist_gradient_boosting_model', has_feature_importances=False)
create_executed_notebook(
    'train_histgradientboosting.ipynb',
    '# HistGradientBoosting: Chronological 70-20-10 Split (Train <= 1997, Val 1998-2008, Test >= 2009)',
    [
        {'md': '### 1. Chronological 70-20-10 Split', 'code': split_data_code},
        {'md': '### 2. Train HistGradientBoosting, Threshold Tuning on Validation, and Test Evaluation', 'code': c1},
        {'md': '### 3. Graph 1: Confusion Matrix Heatmap', 'code': c2},
        {'md': '### 4. Graph 2: Receiver Operating Characteristic (ROC) Curve', 'code': c3},
        {'md': '### 5. Graph 3: Predicted Probability Distribution', 'code': c4},
        {'md': '### 6. Graph 4: Top 15 Feature Importances', 'code': c5}
    ]
)

# ==========================================
# 3. Random Forest
# ==========================================
rf_init = """from sklearn.ensemble import RandomForestClassifier

# Random Forest with balanced class weights
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)"""

c1, c2, c3, c4, c5 = build_model_snippets(rf_init, 'random_forest_model', has_feature_importances=True)
create_executed_notebook(
    'train_randomforest.ipynb',
    '# Random Forest Classifier: Chronological 70-20-10 Split (Train <= 1997, Val 1998-2008, Test >= 2009)',
    [
        {'md': '### 1. Chronological 70-20-10 Split', 'code': split_data_code},
        {'md': '### 2. Train Random Forest, Threshold Tuning on Validation, and Test Evaluation', 'code': c1},
        {'md': '### 3. Graph 1: Confusion Matrix Heatmap', 'code': c2},
        {'md': '### 4. Graph 2: Receiver Operating Characteristic (ROC) Curve', 'code': c3},
        {'md': '### 5. Graph 3: Predicted Probability Distribution', 'code': c4},
        {'md': '### 6. Graph 4: Top 15 Feature Importances', 'code': c5}
    ]
)

# ==========================================
# 4. Extra Trees
# ==========================================
et_init = """from sklearn.ensemble import ExtraTreesClassifier

# Extra Trees with balanced class weights
model = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)"""

c1, c2, c3, c4, c5 = build_model_snippets(et_init, 'extra_trees_model', has_feature_importances=True)
create_executed_notebook(
    'train_extratrees.ipynb',
    '# Extra Trees Classifier: Chronological 70-20-10 Split (Train <= 1997, Val 1998-2008, Test >= 2009)',
    [
        {'md': '### 1. Chronological 70-20-10 Split', 'code': split_data_code},
        {'md': '### 2. Train Extra Trees, Threshold Tuning on Validation, and Test Evaluation', 'code': c1},
        {'md': '### 3. Graph 1: Confusion Matrix Heatmap', 'code': c2},
        {'md': '### 4. Graph 2: Receiver Operating Characteristic (ROC) Curve', 'code': c3},
        {'md': '### 5. Graph 3: Predicted Probability Distribution', 'code': c4},
        {'md': '### 6. Graph 4: Top 15 Feature Importances', 'code': c5}
    ]
)

# ==========================================
# 5. Stacking Ensemble
# ==========================================
stack_init = """from sklearn.ensemble import StackingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
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
    cv=3,
    n_jobs=1
)"""

c1, c2, c3, c4, c5 = build_model_snippets(stack_init, 'stacking_ensemble_model', has_feature_importances=False)
create_executed_notebook(
    'train_stacking_ensemble.ipynb',
    '# Stacking Ensemble: Chronological 70-20-10 Split (Train <= 1997, Val 1998-2008, Test >= 2009)',
    [
        {'md': '### 1. Chronological 70-20-10 Split', 'code': split_data_code},
        {'md': '### 2. Train Stacking Ensemble, Threshold Tuning on Validation, and Test Evaluation', 'code': c1},
        {'md': '### 3. Graph 1: Confusion Matrix Heatmap', 'code': c2},
        {'md': '### 4. Graph 2: Receiver Operating Characteristic (ROC) Curve', 'code': c3},
        {'md': '### 5. Graph 3: Predicted Probability Distribution', 'code': c4},
        {'md': '### 6. Graph 4: Top 15 Feature Importances', 'code': c5}
    ]
)

# save feature columns for reference
df_temp = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'flood_data_featured.csv'), nrows=1)
feature_cols = [c for c in df_temp.columns if c not in ['Station_Names', 'Year', 'Month', 'Period', 'Flood?']]
joblib.dump(feature_cols, os.path.join(models_dir, 'feature_columns.pkl'))
print("All 5 chronological 70-20-10 notebooks successfully created and executed!")
