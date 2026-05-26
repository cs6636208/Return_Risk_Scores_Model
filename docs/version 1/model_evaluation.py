import os
import joblib
import json
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONSTANTS & SETUP
# ---------------------------------------------------------
COST_FN = 150  # Cost of missing a return (shipping cost)
COST_FP = 50   # Cost of false alarm (lost opportunity / confirm call)

def load_resources():
    print("[INFO] Loading model and data...")
    data = joblib.load('data/features/train_test_sets.pkl')
    model = joblib.load('models/best_model.pkl')
    
    # We only need the test set for evaluation
    X_test = data['X_test']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    # Ensure X_test is a DataFrame for SHAP
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=feature_names)
        
    return model, X_test, y_test, feature_names

# ---------------------------------------------------------
# 1. THRESHOLD OPTIMIZATION (COST MATRIX)
# ---------------------------------------------------------
def optimize_threshold(model, X_test, y_test):
    print("\n[INFO] Running Threshold Optimization...")
    y_proba = model.predict_proba(X_test)[:, 1]
    
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        # Handle cases where confusion matrix might not be 2x2
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # If everything is predicted as 0, for instance
            if len(np.unique(y_test)) == 2:
                tn = cm[0,0] if 0 in y_pred else 0
                fp = cm[0,1] if cm.shape[1] > 1 else 0
                fn = cm[1,0] if cm.shape[0] > 1 else 0
                tp = cm[1,1] if cm.shape == (2,2) else 0
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
                
        total_cost = (fn * COST_FN) + (fp * COST_FP)
        costs.append(total_cost)
        
    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    min_cost = costs[best_idx]
    
    print(f"  Best Threshold : {best_threshold:.2f}")
    print(f"  Minimum Cost   : {min_cost} THB")
    
    # Plot Expected Cost Curve
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, lw=2, color='#2c3e50')
    plt.axvline(best_threshold, color='#e74c3c', linestyle='--', label=f'Optimal = {best_threshold:.2f}')
    plt.title(f'Expected Cost vs. Decision Threshold\n(FN Cost={COST_FN} THB, FP Cost={COST_FP} THB)', fontsize=14)
    plt.xlabel('Probability Threshold', fontsize=12)
    plt.ylabel('Total Expected Cost (THB)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('reports/model_evaluation', exist_ok=True)
    cost_plot_path = 'reports/model_evaluation/cost_optimization.png'
    plt.savefig(cost_plot_path, dpi=150)
    plt.close()
    
    print(f"  [SAVED] Cost Curve -> {cost_plot_path}")
    return best_threshold, y_proba

# ---------------------------------------------------------
# 2. SHAP EXPLAINABILITY
# ---------------------------------------------------------
def run_shap_analysis(model, X_test, feature_names, y_proba):
    print("\n[INFO] Running SHAP Explainability Analysis...")
    
    # Use TreeExplainer for XGBoost/LightGBM/RF
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # LightGBM / RF might return a list of arrays for multi-class/binary. 
    # XGBoost usually returns a single array. Let's handle it safely.
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # Take positive class
        
    # --- A) Global Explainability (Summary Plot) ---
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    summary_path = 'reports/model_evaluation/shap_summary.png'
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] SHAP Summary Plot -> {summary_path}")
    
    # --- B) Local Explainability (Waterfall for Top Risk Order) ---
    # Find the order with the highest probability of return
    top_risk_idx = np.argmax(y_proba)
    top_risk_prob = y_proba[top_risk_idx]
    
    print(f"\n  Analyzing Top Risk Order (Index {top_risk_idx}), Prob={top_risk_prob:.4f}")
    
    # We need a SHAP Explanation object for waterfall plot
    # Try using waterfall, fallback to force_plot if error
    try:
        expected_val = explainer.expected_value
        if isinstance(expected_val, np.ndarray) or isinstance(expected_val, list):
            expected_val = expected_val[0]
            
        exp = shap.Explanation(
            values=shap_values[top_risk_idx],
            base_values=expected_val,
            data=X_test.iloc[top_risk_idx],
            feature_names=feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False, max_display=10)
        local_path = 'reports/model_evaluation/shap_waterfall_top_risk.png'
        plt.tight_layout()
        plt.savefig(local_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] SHAP Waterfall Plot -> {local_path}")
        
    except Exception as e:
        print(f"  [WARNING] Could not create waterfall plot: {e}. Falling back to bar plot.")
        plt.figure(figsize=(10, 6))
        # Fallback to a simple bar plot of the top features for this instance
        instance_shap = pd.Series(shap_values[top_risk_idx], index=feature_names)
        instance_shap.abs().sort_values(ascending=False).head(10).plot(kind='barh')
        plt.title('Top SHAP Values (Magnitude) for Top Risk Order')
        local_path = 'reports/model_evaluation/shap_bar_top_risk.png'
        plt.tight_layout()
        plt.savefig(local_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] SHAP Bar Plot -> {local_path}")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("MODEL EVALUATION & EXPLAINABILITY (WEEK 6)")
    print("=" * 60)
    
    model, X_test, y_test, feature_names = load_resources()
    best_threshold, y_proba = optimize_threshold(model, X_test, y_test)
    run_shap_analysis(model, X_test, feature_names, y_proba)
    
    print("\n[SUCCESS] Evaluation complete!")
