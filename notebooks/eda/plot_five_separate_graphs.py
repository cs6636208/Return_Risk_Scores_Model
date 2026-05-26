import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run_five_separate_plots():
    print("=" * 60)
    print("Starting Generation of 5 Separate Category Graphs...")
    print("=" * 60)
    
    data_path = 'data/processed/clean_dataset.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    target_provinces = [
        'Bangkok', 'Nonthaburi', 'Chonburi', 'Khon Kaen',
        'Chiang Mai', 'Phuket', 'Songkhla', 'Remote_Area'
    ]
    target_categories = ['Fashion', 'Supplement', 'Home_Appliance', 'Electronics', 'Cosmetics']
    
    df_filtered = df[
        df['province'].isin(target_provinces) & 
        df['category'].isin(target_categories)
    ].copy()
    
    agg_df = df_filtered.groupby(['category', 'province'])['is_returned'].mean().reset_index()
    agg_df['is_returned'] *= 100  
    
    colors = {
        "Fashion": "#FF6B6B",       
        "Supplement": "#4D96FF",    
        "Home_Appliance": "#6BCB77", 
        "Electronics": "#1A5276",   
        "Cosmetics": "#E74C3C"      
    }
    
    output_dir = 'reports/Graph Item'
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    for category in target_categories:
        cat_data = agg_df[agg_df['category'] == category].copy()
        cat_data = cat_data.set_index('province').reindex(target_provinces).reset_index()
        
        plt.figure(figsize=(12, 6.5))
        
        ax = sns.barplot(
            data=cat_data, 
            x='province', 
            y='is_returned', 
            color=colors[category],
            order=target_provinces,
            edgecolor='black',
            linewidth=0.8
        )
        
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f'{height:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    xytext=(0, 8),
                    textcoords='offset points',
                    fontsize=11,
                    fontweight='bold',
                    color='#2C3E50'
                )
                
        plt.title(f"Return Rate (%) by Province for Category: {category}", fontsize=15, fontweight='bold', pad=15)
        plt.xlabel("Province", fontsize=12, labelpad=10)
        plt.ylabel("Return Rate (%)", fontsize=12, labelpad=10)
        plt.ylim(0, 50) 
        plt.xticks(rotation=15, fontsize=10, fontweight='semibold')
        plt.tight_layout()
        
        filename = f"eda_return_rate_{category}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"[SUCCESS] Saved individual plot: {filepath}")
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten()
    
    for idx, category in enumerate(target_categories):
        ax = axes[idx]
        cat_data = agg_df[agg_df['category'] == category].copy()
        cat_data = cat_data.set_index('province').reindex(target_provinces).reset_index()
        
        sns.barplot(
            data=cat_data, 
            x='province', 
            y='is_returned', 
            color=colors[category],
            order=target_provinces,
            edgecolor='black',
            linewidth=0.6,
            ax=ax
        )
       
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f'{height:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center',
                    xytext=(0, 6),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color='#2C3E50'
                )
                
        ax.set_title(f"Category: {category}", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("Province", fontsize=11)
        ax.set_ylabel("Return Rate (%)", fontsize=11)
        ax.set_ylim(0, 50)
        ax.tick_params(axis='x', rotation=15)
        
    axes[5].axis('off')
    
    plt.suptitle("Return Rate (%) by Province across 5 Product Categories", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    grid_path = os.path.join(output_dir, "eda_return_rate_by_category_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Saved combined grid plot: {grid_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_five_separate_plots()
