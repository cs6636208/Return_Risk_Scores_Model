import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
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
    X_test, y_test = data['X_test'], data['X_test']
    y_test = data['y_test']
    
    model = joblib.load('models/best_model_v3_stack.pkl')
    y_proba = model.predict_proba(X_test)[:, 1]
    
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
    
    print("\n[SCENARIO 1] 🎯 Optimal Cost Threshold (The Best for Business)")
    print(f"  Threshold : {opt_t:.2f}")
    print(f"  Accuracy  : {accuracies[best_idx]*100:.2f}%")
    print(f"  Recall    : {recalls[best_idx]*100:.2f}%")
    print(f"  Cost      : {min_cost} THB")
    
    # 3. Find Threshold for 80% Accuracy
    acc_target_80 = 0.80
    idx_80 = np.argmin(np.abs(np.array(accuracies) - acc_target_80))
    t_80 = thresholds[idx_80]
    
    print(f"\n[SCENARIO 2] 📊 Force 80% Accuracy (The User Request)")
    print(f"  Threshold : {t_80:.2f}")
    print(f"  Accuracy  : {accuracies[idx_80]*100:.2f}%")
    print(f"  Recall    : {recalls[idx_80]*100:.2f}%  <-- Warning: Plummets dramatically!")
    print(f"  Cost      : {costs[idx_80]} THB  <-- Warning: Financial loss increases!")

    # 4. Find Threshold for 85% Accuracy
    acc_target_85 = 0.85
    idx_85 = np.argmin(np.abs(np.array(accuracies) - acc_target_85))
    t_85 = thresholds[idx_85]
    
    print(f"\n[SCENARIO 3] ⚠️ Force 85% Accuracy")
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

if __name__ == '__main__':
    evaluate_v3()
