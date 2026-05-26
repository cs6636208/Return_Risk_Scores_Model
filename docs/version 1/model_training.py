import pandas as pd
import numpy as np
import joblib
import os
import warnings
import json
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, roc_curve, 
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

import optuna
from optuna.samplers import TPESampler

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data(path='data/features/train_test_sets.pkl'):
    """Load pre-processed train/test sets."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Data not found at {path}")
    data = joblib.load(path)
    print("=" * 60)
    print("STEP 1: LOAD DATA")
    print("=" * 60)
    print(f"  X_train : {data['X_train'].shape}")
    print(f"  X_test  : {data['X_test'].shape}")
    print(f"  y_train : class 0={int((data['y_train']==0).sum())}, class 1={int((data['y_train']==1).sum())}")
    print(f"  y_test  : class 0={int((data['y_test']==0).sum())}, class 1={int((data['y_test']==1).sum())}")
    return data['X_train'], data['X_test'], data['y_train'], data['y_test'], data['feature_names']


# ============================================================
# 2. EVALUATE MODEL (helper)
# ============================================================
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a model and return a dict of metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {model_name} ---")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  Avg Prec  : {ap:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    return {
        'model_name': model_name,
        'auc_roc': round(auc, 4),
        'f1_score': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'avg_precision': round(ap, 4),
        'confusion_matrix': cm.tolist(),
        'y_proba': y_proba
    }


# ============================================================
# 3. BASELINE: LOGISTIC REGRESSION
# ============================================================
def train_logistic(X_train, y_train, X_test, y_test):
    """Train Logistic Regression as baseline."""
    print("\n" + "=" * 60)
    print("STEP 2: BASELINE - LOGISTIC REGRESSION")
    print("=" * 60)
    model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, "Logistic Regression")
    return model, metrics


# ============================================================
# 4. TREE-BASED MODELS (Default Params)
# ============================================================
def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with default parameters."""
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, "Random Forest")
    return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with default parameters."""
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1
    )
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, "XGBoost (Default)")
    return model, metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM with default parameters."""
    model = LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, "LightGBM (Default)")
    return model, metrics


def train_all_defaults(X_train, y_train, X_test, y_test):
    """Train all tree-based models with default params."""
    print("\n" + "=" * 60)
    print("STEP 3: TREE-BASED MODELS (Default Parameters)")
    print("=" * 60)
    results = {}
    
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results['Random Forest'] = {'model': rf_model, 'metrics': rf_metrics}

    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    results['XGBoost'] = {'model': xgb_model, 'metrics': xgb_metrics}

    lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test)
    results['LightGBM'] = {'model': lgb_model, 'metrics': lgb_metrics}

    return results


# ============================================================
# 5. OPTUNA HYPERPARAMETER TUNING
# ============================================================
def optuna_xgboost_objective(trial, X_train, y_train):
    """Optuna objective for XGBoost with StratifiedKFold CV."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'eval_metric': 'logloss',
        'verbosity': 0,
        'n_jobs': -1,
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    return scores.mean()


def optuna_lightgbm_objective(trial, X_train, y_train):
    """Optuna objective for LightGBM with StratifiedKFold CV."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1,
    }
    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    return scores.mean()


def run_optuna_tuning(X_train, y_train, X_test, y_test, n_trials=50):
    """Run Optuna tuning for both XGBoost and LightGBM, pick the best."""
    print("\n" + "=" * 60)
    print("STEP 4: HYPERPARAMETER TUNING (Optuna)")
    print("=" * 60)

    # --- XGBoost Tuning ---
    print(f"\n[INFO] Tuning XGBoost ({n_trials} trials)...")
    xgb_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    xgb_study.optimize(
        lambda trial: optuna_xgboost_objective(trial, X_train, y_train),
        n_trials=n_trials, show_progress_bar=True
    )
    print(f"  Best XGBoost CV F1: {xgb_study.best_value:.4f}")
    print(f"  Best XGBoost Params: {xgb_study.best_params}")

    # --- LightGBM Tuning ---
    print(f"\n[INFO] Tuning LightGBM ({n_trials} trials)...")
    lgb_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    lgb_study.optimize(
        lambda trial: optuna_lightgbm_objective(trial, X_train, y_train),
        n_trials=n_trials, show_progress_bar=True
    )
    print(f"  Best LightGBM CV F1: {lgb_study.best_value:.4f}")
    print(f"  Best LightGBM Params: {lgb_study.best_params}")

    # --- Pick Winner ---
    if xgb_study.best_value >= lgb_study.best_value:
        best_algo = 'XGBoost'
        best_params = xgb_study.best_params
        best_params.update({'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0, 'n_jobs': -1})
        best_model = XGBClassifier(**best_params)
    else:
        best_algo = 'LightGBM'
        best_params = lgb_study.best_params
        best_params.update({'random_state': 42, 'verbosity': -1, 'n_jobs': -1})
        best_model = LGBMClassifier(**best_params)

    print(f"\n[RESULT] Best Algorithm: {best_algo}")

    # Train on full training set with best params
    best_model.fit(X_train, y_train)
    best_metrics = evaluate_model(best_model, X_test, y_test, f"{best_algo} (Tuned)")

    return best_model, best_metrics, best_algo, best_params, xgb_study, lgb_study


# ============================================================
# 6. SAVE MODEL & REPORT
# ============================================================
def save_model_and_report(best_model, best_metrics, best_algo, best_params, all_metrics):
    """Save the best model and metrics report."""
    print("\n" + "=" * 60)
    print("STEP 5: SAVE BEST MODEL & REPORT")
    print("=" * 60)

    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/model_training', exist_ok=True)

    # Save model
    model_path = 'models/best_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"  [SAVED] Best model -> {model_path}")

    # Save metadata
    metadata = {
        'best_algorithm': best_algo,
        'best_params': {k: (int(v) if isinstance(v, (np.integer,)) else
                            float(v) if isinstance(v, (np.floating,)) else v)
                        for k, v in best_params.items()},
        'test_metrics': {k: v for k, v in best_metrics.items() if k != 'y_proba'},
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    meta_path = 'models/best_model_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] Metadata   -> {meta_path}")

    # Save comparison summary
    summary_rows = []
    for m in all_metrics:
        summary_rows.append({
            'Model': m['model_name'],
            'AUC-ROC': m['auc_roc'],
            'F1-Score': m['f1_score'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'Avg Precision': m['avg_precision'],
        })
    df_summary = pd.DataFrame(summary_rows)
    summary_path = 'reports/model_training/model_comparison.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"  [SAVED] Comparison -> {summary_path}")

    return df_summary


# ============================================================
# 7. PLOT COMPARISON CHARTS
# ============================================================
def plot_comparison(all_metrics, y_test):
    """Plot ROC Curve and PR Curve for all models."""
    os.makedirs('reports/model_training', exist_ok=True)

    # --- ROC Curve ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for m in all_metrics:
        fpr, tpr, _ = roc_curve(y_test, m['y_proba'])
        axes[0].plot(fpr, tpr, label=f"{m['model_name']} (AUC={m['auc_roc']:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve - All Models')
    axes[0].legend(loc='lower right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # --- PR Curve ---
    for m in all_metrics:
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, m['y_proba'])
        axes[1].plot(rec_arr, prec_arr, label=f"{m['model_name']} (AP={m['avg_precision']:.3f})")
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve - All Models')
    axes[1].legend(loc='lower left', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    roc_path = 'reports/model_training/roc_pr_curves.png'
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"  [SAVED] ROC & PR curves -> {roc_path}")

    # --- Bar Chart Comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = [m['model_name'] for m in all_metrics]
    x = np.arange(len(model_names))
    width = 0.18

    auc_vals = [m['auc_roc'] for m in all_metrics]
    f1_vals = [m['f1_score'] for m in all_metrics]
    prec_vals = [m['precision'] for m in all_metrics]
    rec_vals = [m['recall'] for m in all_metrics]

    bars1 = ax.bar(x - 1.5*width, auc_vals, width, label='AUC-ROC', color='#3498db')
    bars2 = ax.bar(x - 0.5*width, f1_vals, width, label='F1-Score', color='#2ecc71')
    bars3 = ax.bar(x + 0.5*width, prec_vals, width, label='Precision', color='#e67e22')
    bars4 = ax.bar(x + 1.5*width, rec_vals, width, label='Recall', color='#e74c3c')

    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    bar_path = 'reports/model_training/model_comparison_bar.png'
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"  [SAVED] Bar chart   -> {bar_path}")


# ============================================================
# MAIN
# ============================================================
def run_model_training():
    """Main entry point for model training pipeline."""
    start_time = datetime.now()
    print(f"\n{'#' * 60}")
    print(f"  MODEL TRAINING PIPELINE")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")

    # 1. Load Data
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # 2. Baseline
    lr_model, lr_metrics = train_logistic(X_train, y_train, X_test, y_test)

    # 3. Tree-based (defaults)
    default_results = train_all_defaults(X_train, y_train, X_test, y_test)

    # 4. Optuna Tuning
    best_model, best_metrics, best_algo, best_params, xgb_study, lgb_study = \
        run_optuna_tuning(X_train, y_train, X_test, y_test, n_trials=50)

    # Collect all metrics for comparison
    all_metrics = [
        lr_metrics,
        default_results['Random Forest']['metrics'],
        default_results['XGBoost']['metrics'],
        default_results['LightGBM']['metrics'],
        best_metrics,
    ]

    # 5. Save
    df_summary = save_model_and_report(best_model, best_metrics, best_algo, best_params, all_metrics)

    # 6. Plot
    plot_comparison(all_metrics, y_test)

    # 7. Final Summary
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print(f"\n  Best Model : {best_algo} (Tuned)")
    print(f"  AUC-ROC    : {best_metrics['auc_roc']}")
    print(f"  F1-Score   : {best_metrics['f1_score']}")
    print(f"  Elapsed    : {elapsed:.1f} seconds")
    print(f"\n{'#' * 60}")
    print(f"  TRAINING COMPLETE!")
    print(f"{'#' * 60}\n")


if __name__ == '__main__':
    run_model_training()
