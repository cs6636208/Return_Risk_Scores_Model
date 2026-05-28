import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import warnings

warnings.filterwarnings('ignore')

COST_FN = 150
COST_FP = 50

def evaluate_v3():
    print("=" * 60)
    print("MODEL EVALUATION V3 (SCENARIO SIMULATOR)")
    print("=" * 60)
    
    # 1. Load Data & Model
    data = joblib.load('data/features/train_test_sets_v2.pkl')
    X_test = data['X_test']
    y_test = data['y_test']
    
    model = joblib.load('models/best_model_v3_stack.pkl')
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= 0.50).astype(int)

    cm_default = confusion_matrix(y_test, y_pred_default)
    if cm_default.shape == (2, 2):
        tn, fp, fn, tp = cm_default.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    metrics = {
        'model_name': 'V3 Stacking (XGB+LGB+CAT)',
        'threshold': 0.50,
        'auc_roc': roc_auc_score(y_test, y_proba),
        'avg_precision': average_precision_score(y_test, y_proba),
        'f1_score': f1_score(y_test, y_pred_default, zero_division=0),
        'precision': precision_score(y_test, y_pred_default, zero_division=0),
        'recall': recall_score(y_test, y_pred_default, zero_division=0),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }

    print("\n[BASELINE THRESHOLD] 0.50")
    print(f"  AUC-ROC       : {metrics['auc_roc']:.4f}")
    print(f"  Avg Precision : {metrics['avg_precision']:.4f}")
    print(f"  F1-Score      : {metrics['f1_score']:.4f}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  Confusion     : TN={tn} FP={fp} FN={fn} TP={tp}")
    
    # 2. Cost Matrix Simulation
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    accuracies = []
    recalls = []
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            acc = (tn + tp) / np.sum(cm)
            rec = tp / (tp + fn) if (tp+fn) > 0 else 0
        else:
            acc, rec, tn, fp, fn, tp = 0, 0, 0, 0, 0, 0
            
        costs.append((fn * COST_FN) + (fp * COST_FP))
        accuracies.append(acc)
        recalls.append(rec)
        
    best_idx = np.argmin(costs)
    opt_t = thresholds[best_idx]
    min_cost = costs[best_idx]
    metrics['optimal_cost_threshold'] = float(opt_t)
    metrics['optimal_cost'] = int(min_cost)
    metrics['optimal_cost_accuracy'] = float(accuracies[best_idx])
    metrics['optimal_cost_recall'] = float(recalls[best_idx])
    
    print("\n[SCENARIO 1] Optimal Cost Threshold (The Best for Business)")
    print(f"  Threshold : {opt_t:.2f}")
    print(f"  Accuracy  : {accuracies[best_idx]*100:.2f}%")
    print(f"  Recall    : {recalls[best_idx]*100:.2f}%")
    print(f"  Cost      : {min_cost} THB")
    
    # 3. Find Threshold for 80% Accuracy
    acc_target_80 = 0.80
    idx_80 = np.argmin(np.abs(np.array(accuracies) - acc_target_80))
    t_80 = thresholds[idx_80]
    
    print(f"\n[SCENARIO 2] Force 80% Accuracy (The User Request)")
    print(f"  Threshold : {t_80:.2f}")
    print(f"  Accuracy  : {accuracies[idx_80]*100:.2f}%")
    print(f"  Recall    : {recalls[idx_80]*100:.2f}%  <-- Warning: Plummets dramatically!")
    print(f"  Cost      : {costs[idx_80]} THB  <-- Warning: Financial loss increases!")

    # 4. Find Threshold for 85% Accuracy
    acc_target_85 = 0.85
    idx_85 = np.argmin(np.abs(np.array(accuracies) - acc_target_85))
    t_85 = thresholds[idx_85]
    
    print(f"\n[SCENARIO 3] Force 85% Accuracy")
    print(f"  Threshold : {t_85:.2f}")
    print(f"  Accuracy  : {accuracies[idx_85]*100:.2f}%")
    print(f"  Recall    : {recalls[idx_85]*100:.2f}%  <-- Warning: Almost completely blind to returns")
    print(f"  Cost      : {costs[idx_85]} THB")

    # 5. Plotting Accuracy vs Recall Tradeoff
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label='Accuracy', color='blue', lw=2)
    plt.plot(thresholds, recalls, label='Recall (Finding Returns)', color='red', lw=2)
    
    plt.axvline(opt_t, color='green', linestyle='--', label=f'Optimal Cost (T={opt_t:.2f})')
    plt.axvline(t_80, color='orange', linestyle='--', label=f'80% Accuracy (T={t_80:.2f})')
    
    plt.title('V3 Stacking: Trade-off Between Accuracy and Recall')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Score (0.0 to 1.0)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    os.makedirs('reports/model_evaluation_v3', exist_ok=True)
    tradeoff_path = 'reports/model_evaluation_v3/accuracy_recall_tradeoff.png'
    plt.savefig(tradeoff_path, dpi=150)
    plt.close()
    print(f"\n[SAVED] Trade-off Plot -> {tradeoff_path}")

    # 6. Save machine-readable and presentation-friendly metric reports
    metrics_path = 'reports/model_evaluation_v3/metrics_summary_v3.csv'
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"[SAVED] Metrics Summary -> {metrics_path}")

    scenario_rows = pd.DataFrame({
        'threshold': thresholds,
        'accuracy': accuracies,
        'recall': recalls,
        'expected_cost_thb': costs,
    })
    scenario_path = 'reports/model_evaluation_v3/threshold_scenarios_v3.csv'
    scenario_rows.to_csv(scenario_path, index=False)
    print(f"[SAVED] Threshold Scenarios -> {scenario_path}")

if __name__ == '__main__':
    evaluate_v3()
