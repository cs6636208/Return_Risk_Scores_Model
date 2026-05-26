import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run_category_profile_analysis():
    print("=" * 60)
    print("Starting Category Profile Analysis...")
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
    target_categories = ['Fashion', 'Supplement', 'Home_Appliance', 'Electronics', 'Cosmetics']
    
    df_filtered = df[
        df['province'].isin(target_provinces) & 
        df['category'].isin(target_categories)
    ].copy()
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten() 
    
    grouped = df_filtered.groupby(['category', 'province']).mean(numeric_only=True).reset_index()
    
    for idx, category in enumerate(target_categories):
        ax = axes[idx]
        cat_data = grouped[grouped['category'] == category].copy()
        cat_data = cat_data.set_index('province').reindex(target_provinces).reset_index()
        
        features_to_melt = ['is_returned', 'is_cod', 'is_high_friction']
        labels = ['Return Rate (%)', 'COD Rate (%)', 'High Friction (%)']
     
        if category == 'Fashion':
            features_to_melt += ['is_bracketing', 'is_impulse_buy']
            labels += ['Bracketing (%)', 'Impulse Buy (%)']
            
        melted = pd.melt(
            cat_data, 
            id_vars=['province'], 
            value_vars=features_to_melt, 
            var_name='Feature', 
            value_name='Percentage'
        )
        melted['Percentage'] *= 100 
        
        feature_map = dict(zip(features_to_melt, labels))
        melted['Feature'] = melted['Feature'].map(feature_map)
        
        feat_palette = {
            'Return Rate (%)': '#FF5733', 
            'COD Rate (%)': '#33FF57', 
            'High Friction (%)': '#3357FF',
            'Bracketing (%)': '#F333FF',
            'Impulse Buy (%)': '#FFD133'
        }
        
        sns.barplot(
            data=melted, 
            x='province', 
            y='Percentage', 
            hue='Feature', 
            palette=feat_palette, 
            ax=ax,
            edgecolor='black',
            linewidth=0.5
        )
        
        ax.set_title(f"Category: {category} Profile", fontsize=15, fontweight='bold', pad=10)
        ax.set_xlabel("Province", fontsize=11)
        ax.set_ylabel("Percentage (%)", fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        ax.tick_params(axis='x', rotation=15)
        
        print(f"\n[INFO] Summary for {category}:")
        summary_table = cat_data.set_index('province')[features_to_melt] * 100
        print(summary_table.round(1).reindex(target_provinces).to_string())

    axes[5].axis('off')
    
    plt.suptitle("Feature Values Profile by Province for Each Specific Category", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_dir = 'reports/Graph Item'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'category_province_profiles.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SUCCESS] Category profile graph generated at {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_category_profile_analysis()
