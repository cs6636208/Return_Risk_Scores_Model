import pandas as pd
import numpy as np
import joblib
import os
import warnings
import json
from datetime import datetime

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, average_precision_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def evaluate_model(model, X_test, y_test, model_name="Model"):
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
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    return {'model_name': model_name, 'auc_roc': auc, 'f1_score': f1}


def optuna_xgboost_objective(trial, X_train, y_train, scale_pos_weight):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'logloss',
        'verbosity': 0,
        'n_jobs': -1,
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    return scores.mean()


def run_model_training_v2():
    print("=" * 60)
    print("MODEL TRAINING V2 (SCALE_POS_WEIGHT)")
    print("=" * 60)
    
    # 1. Load Data
    data = joblib.load('data/features/train_test_sets_v2.pkl')
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    count_0 = (y_train == 0).sum()
    count_1 = (y_train == 1).sum()
    scale_pos = count_0 / count_1
    
    print(f"[INFO] Train shape: {X_train.shape}")
    print(f"[INFO] Target 0: {count_0}, Target 1: {count_1}")
    print(f"[INFO] Calculated scale_pos_weight: {scale_pos:.2f}")

    # 2. XGBoost Default
    xgb_default = XGBClassifier(scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, eval_metric='logloss')
    xgb_default.fit(X_train, y_train)
    evaluate_model(xgb_default, X_test, y_test, "XGBoost (V2 Default, scale_pos_weight)")
    
    # 3. LightGBM Default
    # class_weight='balanced' in LightGBM handles it automatically
    lgb_default = LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbosity=-1)
    lgb_default.fit(X_train, y_train)
    evaluate_model(lgb_default, X_test, y_test, "LightGBM (V2 Default, balanced)")

    # 4. Optuna Tuning (XGBoost only for speed)
    print(f"\n[INFO] Tuning XGBoost (30 trials)...")
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(
        lambda trial: optuna_xgboost_objective(trial, X_train, y_train, scale_pos),
        n_trials=30, show_progress_bar=True
    )
    
    print(f"  Best XGBoost CV F1: {study.best_value:.4f}")
    
    best_params = study.best_params
    best_params.update({'scale_pos_weight': scale_pos, 'random_state': 42, 'eval_metric': 'logloss', 'n_jobs': -1})
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)
    
    evaluate_model(best_model, X_test, y_test, "XGBoost (V2 Tuned)")

    # 5. Save the Model
    joblib.dump(best_model, 'models/best_model_v2.pkl')
    print("\n[SUCCESS] V2 Model Training Completed & Saved as best_model_v2.pkl!")

if __name__ == '__main__':
    run_model_training_v2()
