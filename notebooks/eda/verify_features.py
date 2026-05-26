import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

def verify_features():
    print("[INFO] Loading train/test sets...")
    data_path = 'data/features/train_test_sets.pkl'
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found. Please run feature_engineering.py first.")
        return
        
    data = joblib.load(data_path)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    print("[INFO] Training Baseline Random Forest...")
    # ใช้ Random Forest ความลึกปานกลาง (max_depth=10) เพื่อดูพลังของ Feature เบื้องต้น
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    
    # ตรวจสอบชนิดข้อมูลของ X_train เผื่อ SMOTE คืนค่าเป็น numpy array
    if isinstance(X_train, np.ndarray):
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_train_df = X_train
        feature_names = X_train_df.columns
        
    rf.fit(X_train_df, y_train)
    
    print("[INFO] Evaluating on Test Set (Unseen Data)...")
    if isinstance(X_test, np.ndarray):
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
    else:
        X_test_df = X_test
        
    y_pred = rf.predict(X_test_df)
    y_prob = rf.predict_proba(X_test_df)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n======================================")
    print(f"Baseline ROC-AUC Score: {auc:.4f}")
    print(f"======================================\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("[INFO] Plotting Feature Importance...")
    importances = rf.feature_importances_
        
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_imp.head(20), palette='viridis')
    plt.title('Top 20 Feature Importance (Baseline Random Forest)', fontsize=15, pad=20)
    plt.xlabel('Importance Score (Gini)')
    plt.ylabel('Features')
    plt.tight_layout()
    
    os.makedirs('reports', exist_ok=True)
    out_path = 'reports/feature_importance_baseline.png'
    plt.savefig(out_path, dpi=300)
    print(f"[INFO] Saved feature importance plot to {out_path}")
    
    print("\n--- Top 10 Features ---")
    for i, row in feature_imp.head(10).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

if __name__ == "__main__":
    verify_features()
