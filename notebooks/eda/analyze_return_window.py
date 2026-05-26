import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Tahoma'

def analyze_windows():
    print("[INFO] Loading data...")
    df = pd.read_csv('data/features/df_engineered.csv')
    
    repeat_returns = df[(df['is_returned'] == 1) & (df['days_since_last_return'] > 0)]
    
    print(f"[INFO] Found {len(repeat_returns)} repeat return events.")
    
    if len(repeat_returns) == 0:
        print("[WARNING] No repeat returns found to analyze.")
        return

    plt.figure(figsize=(12, 6))
    
    sns.histplot(repeat_returns['days_since_last_return'], bins=50, kde=True, color='purple')
    
    p50 = repeat_returns['days_since_last_return'].median()
    p75 = repeat_returns['days_since_last_return'].quantile(0.75)
    p80 = repeat_returns['days_since_last_return'].quantile(0.80)
    p90 = repeat_returns['days_since_last_return'].quantile(0.90)
    
    plt.axvline(p50, color='red', linestyle='--', label=f'50% ภายใน {int(p50)} วัน')
    plt.axvline(p80, color='green', linestyle='--', label=f'80% ภายใน {int(p80)} วัน')
    plt.axvline(p90, color='blue', linestyle='--', label=f'90% ภายใน {int(p90)} วัน')
    
    plt.title('Time Gap Between Successive Returns by the Same Customer', fontsize=14)
    plt.xlabel('Days Since Last Return (ระยะห่างจากวันที่คืนของครั้งล่าสุด)', fontsize=12)
    plt.ylabel('Frequency (จำนวนครั้งที่เกิดการคืนซ้ำ)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs('reports/eda_full', exist_ok=True)
    save_path = 'reports/eda_full/06_time_gap_analysis.png'
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Plot saved to {save_path}")
    
    print(f"\n[SUMMARY STATS]")
    print(f"50% ของการคืนซ้ำ เกิดขึ้นภายใน: {int(p50)} วัน")
    print(f"75% ของการคืนซ้ำ เกิดขึ้นภายใน: {int(p75)} วัน")
    print(f"80% ของการคืนซ้ำ เกิดขึ้นภายใน: {int(p80)} วัน")
    print(f"90% ของการคืนซ้ำ เกิดขึ้นภายใน: {int(p90)} วัน")

if __name__ == "__main__":
    analyze_windows()
