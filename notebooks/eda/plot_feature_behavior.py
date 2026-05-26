import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run_plot_feature_behavior():
    print("=" * 60)
    print("Starting Plotting of Feature Behavior...")
    print("=" * 60)
    
    data_path = 'data/processed/clean_dataset.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    df['is_peak_hour'] = 0
    df.loc[(df['channel_type'] == 'TV_Show') & (df['order_hour'].between(8, 10)), 'is_peak_hour'] = 1
    df.loc[(df['channel_type'] == 'TikTok') & (df['order_hour'].between(21, 23)), 'is_peak_hour'] = 1
    
    df['is_fashion_tv'] = ((df['category'] == 'Fashion') & (df['channel_type'] == 'TV_Show')).astype(int)
    df['is_bracketing'] = ((df['category'] == 'Fashion') & (df['quantity'] > 1)).astype(int)
    df['is_impulse_buy'] = ((df['category'] == 'Fashion') & 
                            (df['channel_type'].isin(['TV_Show', 'TikTok'])) & 
                            (df['is_peak_hour'] == 1)).astype(int)
    
    df['is_remote_area'] = (df['province'] == 'Remote_Area').astype(int)
    df['is_cod'] = (df['payment_method'] == 'COD').astype(int)
    
    df['is_long_distance_cod'] = ((df['province'].isin(['Chiang Mai', 'Phuket', 'Songkhla'])) & 
                                  (df['payment_method'] == 'COD')).astype(int)
    
    df['delivery_gap'] = df['delivery_days'] - df['delivery_time_expected_days']
    df['is_high_friction'] = ((df['delivery_gap'] > 0) & 
                              (df['province'].isin(['Chiang Mai', 'Phuket', 'Songkhla', 'Remote_Area']))).astype(int)
    
    df['is_high_discount'] = (df['total_discount_pct'] > 0.2).astype(int)
    df['is_low_commitment'] = ((df['payment_method'] == 'COD') & 
                               (df['is_high_discount'] == 1)).astype(int)
    
    target_provinces = [
        'Bangkok', 'Nonthaburi', 'Chonburi', 'Khon Kaen',
        'Chiang Mai', 'Phuket', 'Songkhla', 'Remote_Area'
    ]
    df_filtered = df[df['province'].isin(target_provinces) & df['category'].isin(['Fashion', 'Supplement', 'Home_Appliance'])].copy()
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    palette = {"Fashion": "#FF6B6B", "Supplement": "#4D96FF", "Home_Appliance": "#6BCB77"}
    
    ax1 = axes[0, 0]
    agg_ret = df_filtered.groupby(['province', 'category'])['is_returned'].mean().reset_index()
    agg_ret['is_returned'] *= 100
    sns.barplot(data=agg_ret, x='province', y='is_returned', hue='category', palette=palette, ax=ax1, order=target_provinces)
    ax1.set_title("Return Rate (%) by Category and Province", fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel("Province", fontsize=12)
    ax1.set_ylabel("Return Rate (%)", fontsize=12)
    ax1.tick_params(axis='x', rotation=15)

    ax2 = axes[0, 1]
    agg_fric = df_filtered.groupby(['province', 'category'])['is_high_friction'].mean().reset_index()
    agg_fric['is_high_friction'] *= 100
    sns.barplot(data=agg_fric, x='province', y='is_high_friction', hue='category', palette=palette, ax=ax2, order=target_provinces)
    ax2.set_title("High Friction (%) by Category and Province", fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel("Province", fontsize=12)
    ax2.set_ylabel("High Friction (%)", fontsize=12)
    ax2.tick_params(axis='x', rotation=15)

    ax3 = axes[1, 0]
    agg_bracket = df_filtered.groupby(['province', 'category'])['is_bracketing'].mean().reset_index()
    agg_bracket['is_bracketing'] *= 100
    sns.barplot(data=agg_bracket, x='province', y='is_bracketing', hue='category', palette=palette, ax=ax3, order=target_provinces)
    ax3.set_title("Bracketing (%) by Category and Province", fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlabel("Province", fontsize=12)
    ax3.set_ylabel("Bracketing (%)", fontsize=12)
    ax3.tick_params(axis='x', rotation=15)
    
    ax4 = axes[1, 1]
    agg_gap = df_filtered.groupby(['province', 'category'])['delivery_gap'].mean().reset_index()
    sns.barplot(data=agg_gap, x='province', y='delivery_gap', hue='category', palette=palette, ax=ax4, order=target_provinces)
    ax4.set_title("Avg Delivery Gap (Actual - Expected Days)", fontsize=14, fontweight='bold', pad=10)
    ax4.set_xlabel("Province", fontsize=12)
    ax4.set_ylabel("Delivery Gap (Days)", fontsize=12)
    ax4.tick_params(axis='x', rotation=15)
    
    plt.suptitle("Feature Analysis: Product Category x Province Interaction", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_dir = 'reports/Graph Item'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'feature_behavior_category_province.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Feature behavior graph generated at {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_plot_feature_behavior()
