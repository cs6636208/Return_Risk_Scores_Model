import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run_order_distribution_analysis():
    print("=" * 60)
    print("Starting Order Count Distribution Analysis...")
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
    
    dist_df = df_filtered.groupby(['province', 'category']).size().reset_index(name='order_count')
    
    pivot_dist = dist_df.pivot(index='province', columns='category', values='order_count')
    pivot_dist = pivot_dist.reindex(target_provinces)[target_categories]
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8.5))
    
    palette = {
        "Fashion": "#FF6B6B",
        "Supplement": "#4D96FF",
        "Home_Appliance": "#6BCB77",
        "Electronics": "#1A5276",
        "Cosmetics": "#E74C3C"
    }
    
    ax1 = axes[0]
    sns.heatmap(
        pivot_dist, 
        annot=True, 
        fmt="d", 
        cmap="YlGnBu", 
        linewidths=0.8,
        cbar_kws={'label': 'Number of Orders'},
        ax=ax1,
        annot_kws={"fontsize": 11, "fontweight": "bold"}
    )
    ax1.set_title("Order Count Distribution Heatmap (Sample Size Matrix)", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Product Category", fontsize=12, labelpad=10)
    ax1.set_ylabel("Province", fontsize=12, labelpad=10)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.tick_params(axis='y', labelsize=11)
    
    ax2 = axes[1]
    sns.barplot(
        data=dist_df, 
        x='province', 
        y='order_count', 
        hue='category', 
        palette=palette,
        order=target_provinces,
        hue_order=target_categories,
        edgecolor='black',
        linewidth=0.6,
        ax=ax2
    )
    
    for p in ax2.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax2.annotate(
                f'{int(height)}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center',
                xytext=(0, 6),
                textcoords='offset points',
                fontsize=9,
                fontweight='semibold'
            )
            
    ax2.set_title("Order Volume Distribution by Province and Category", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Province", fontsize=12, labelpad=10)
    ax2.set_ylabel("Number of Orders", fontsize=12, labelpad=10)
    ax2.tick_params(axis='x', rotation=15, labelsize=11)
    ax2.legend(title='Category', title_fontsize='11', fontsize='10', loc='upper right')
    
    plt.suptitle("Data Distribution Analysis: Order Volume (Sample Size) by Category and Province", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_dir = 'reports/Graph Item'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'eda_data_distribution_count.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print("\n[INFO] Complete Order Count Distribution Matrix:")
    print(pivot_dist.to_string())
    
    print(f"\n[SUCCESS] Distribution graph saved to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_order_distribution_analysis()
