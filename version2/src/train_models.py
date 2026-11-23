import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier,
    ExtraTreesClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def run_training_pipeline(data_path=None, models_dir=None, reports_dir=None, show_plots=True):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(base_dir, 'data', 'processed', 'flood_data_featured.csv')
    if models_dir is None:
        models_dir = os.path.join(base_dir, 'models')
    if reports_dir is None:
        reports_dir = os.path.join(base_dir, 'reports')
        
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print(f"Loading featured dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # non-feature columns
    ignore_cols = ['Station_Names', 'Year', 'Month', 'Period', 'Flood?']
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    # chronological split: 1948-2003 for train, 2004-2013 for test
    train_mask = df['Year'] <= 2003
    test_mask = df['Year'] >= 2004

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'Flood?']
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, 'Flood?']

    print(f"Train split: {X_train.shape[0]} rows (1948-2003) | Test split: {X_test.shape[0]} rows (2004-2013)")
    print(f"Features count: {len(feature_cols)}")

    # define our 5 models
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=180, max_depth=5, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=3.97, random_state=42, eval_metric='logloss'
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=180, max_depth=6, learning_rate=0.08,
            class_weight='balanced', random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=180, max_depth=12, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=180, max_depth=12, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
    }

    # Stacking ensemble combining top 3 models
    stack_estimators = [
        ('xgb', models['XGBoost']),
        ('hgb', models['HistGradientBoosting']),
        ('rf', models['RandomForest'])
    ]
    models['StackingEnsemble'] = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
        cv=3,
        n_jobs=-1
    )

    results = []
    trained_models = {}
    preds_dict = {}
    probs_dict = {}

    print("\n--- Training all 5 models on chronological train set ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        preds_dict[name] = preds
        probs_dict[name] = probs

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision (Flood=1)': prec,
            'Recall (Flood=1)': rec,
            'F1-Score (Flood=1)': f1,
            'ROC-AUC': auc
        })

        # save model artifact
        save_path = os.path.join(models_dir, f"{name.lower()}_model.pkl")
        joblib.dump(model, save_path)

    # save feature columns list
    joblib.dump(feature_cols, os.path.join(models_dir, 'feature_columns.pkl'))

    # summary metrics dataframe
    metrics_df = pd.DataFrame(results).sort_values(by='F1-Score (Flood=1)', ascending=False).reset_index(drop=True)
    print("\n" + "="*80)
    print("           CHRONOLOGICAL OUT-OF-TIME TEST BENCHMARK RESULTS (2004-2013)")
    print("="*80)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*80)

    # best model
    best_model_name = metrics_df.iloc[0]['Model']
    print(f"\nChampion Model: {best_model_name}")
    print("\nClassification Report for Champion Model:")
    print(classification_report(y_test, preds_dict[best_model_name], digits=4))

    # --- Plotting comprehensive evaluation figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. ROC Curves
    ax_roc = axes[0, 0]
    for name in models.keys():
        fpr, tpr, _ = roc_curve(y_test, probs_dict[name])
        auc = roc_auc_score(y_test, probs_dict[name])
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", lw=2)
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_title("ROC Curves (Chronological Test 2004-2013)", fontsize=12)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)

    # 2. Precision-Recall Curves
    ax_pr = axes[0, 1]
    for name in models.keys():
        p, r, _ = precision_recall_curve(y_test, probs_dict[name])
        ax_pr.plot(r, p, label=f"{name}", lw=2)
    ax_pr.set_title("Precision-Recall Curves", fontsize=12)
    ax_pr.set_xlabel("Recall (Flood Detection)")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc='lower left')
    ax_pr.grid(True, alpha=0.3)

    # 3. Confusion Matrix of Best Model
    ax_cm = axes[1, 0]
    cm = confusion_matrix(y_test, preds_dict[best_model_name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['Non-Flood (0)', 'Flood (1)'],
                yticklabels=['Non-Flood (0)', 'Flood (1)'])
    ax_cm.set_title(f"Confusion Matrix: {best_model_name}", fontsize=12)
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("Actual Label")

    # 4. Feature Importances (from XGBoost)
    ax_imp = axes[1, 1]
    xgb_model = trained_models['XGBoost']
    importances = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
    importances.plot(kind='barh', ax=ax_imp, color='steelblue')
    ax_imp.invert_yaxis()
    ax_imp.set_title("Top 15 Feature Importances (XGBoost)", fontsize=12)
    ax_imp.set_xlabel("Importance Score")
    ax_imp.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(reports_dir, 'model_evaluation_metrics.png')
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    print(f"\nEvaluation plots saved to: {chart_path}")

    if show_plots:
        plt.show()

    return metrics_df, trained_models, chart_path

if __name__ == '__main__':
    run_training_pipeline()
