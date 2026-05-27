import pandas as pd
import numpy as np
import joblib
import os
import warnings
from datetime import datetime

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, average_precision_score
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)

    print(f"\n--- {model_name} ---")
    print(f"  Accuracy  : {accuracy*100:.2f}%")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    return {'model_name': model_name, 'accuracy': accuracy, 'auc_roc': auc, 'f1_score': f1, 'recall': recall, 'precision': precision}

def run_model_training_v3():
    print("=" * 60)
    print("MODEL TRAINING V3 (STACKING ENSEMBLE)")
    print("=" * 60)
    print("[INFO] Checking Data Leakage: We are using strict Train/Test splits from V2.")
    
    # 1. Load Data
    data = joblib.load('data/features/train_test_sets_v2.pkl')
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    count_0 = (y_train == 0).sum()
    count_1 = (y_train == 1).sum()
    scale_pos = count_0 / count_1
    
    print(f"[INFO] Train shape: {X_train.shape}")
    print(f"[INFO] Target 0 (Normal): {count_0}, Target 1 (Return): {count_1}")
    print(f"[INFO] Calculated Imbalance Ratio (scale_pos_weight): {scale_pos:.2f}")

    # 2. Define Base Models
    print("\n[INFO] Initializing Base Models...")
    
    # Model 1: XGBoost (Tuned from V2)
    xgb = XGBClassifier(
        n_estimators=360, max_depth=6, learning_rate=0.038, 
        subsample=0.85, colsample_bytree=0.88,
        scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0
    )
    
    # Model 2: LightGBM (Default but Balanced)
    lgb = LGBMClassifier(
        class_weight='balanced', random_state=42, n_jobs=-1, verbosity=-1
    )
    
    # Model 3: CatBoost (New)
    cat = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        auto_class_weights='Balanced', random_seed=42, 
        verbose=0, thread_count=-1
    )
    
    # 3. Create Stacking Classifier
    print("[INFO] Building Stacking Classifier with Logistic Regression Meta-Learner...")
    estimators = [
        ('xgb', xgb),
        ('lgb', lgb),
        ('cat', cat)
    ]
    
    # Meta-learner to combine predictions
    # We use cv=5 to ensure the meta-learner learns from out-of-fold predictions, avoiding overfitting!
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
        cv=5,
        n_jobs=-1
    )
    
    # 4. Train Model
    print("[INFO] Training Ensemble Model (This may take a few minutes)...")
    stacking_model.fit(X_train, y_train)
    
    # 5. Evaluate Model
    evaluate_model(stacking_model, X_test, y_test, "V3 Stacking (XGB+LGB+CAT)")
    
    # 6. Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(stacking_model, 'models/best_model_v3_stack.pkl')
    print("\n[SUCCESS] V3 Model Training Completed & Saved as best_model_v3_stack.pkl!")

if __name__ == '__main__':
    run_model_training_v3()
