import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run_single_return_rate_analysis():
    print("=" * 60)
    print("Starting Single Return Rate Analysis (Category vs Province)...")
    print("=" * 60)
    
    data_path = 'data/processed/clean_dataset.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    # 1. Target provinces and categories
    target_provinces = [
        'Bangkok', 'Nonthaburi', 'Chonburi', 'Khon Kaen',
        'Chiang Mai', 'Phuket', 'Songkhla', 'Remote_Area'
    ]
    target_categories = ['Fashion', 'Supplement', 'Home_Appliance', 'Electronics', 'Cosmetics']
    
    # Filter dataset
    df_filtered = df[
        df['province'].isin(target_provinces) & 
        df['category'].isin(target_categories)
    ].copy()
    
    # 2. Calculate return rates
    agg_df = df_filtered.groupby(['province', 'category'])['is_returned'].mean().reset_index()
    agg_df['is_returned'] *= 100  # Convert to percentage
    
    # 3. Create a beautiful visualization
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 8))
    
    # Curated, harmonious, premium color palette for 5 categories
    palette = {
        "Fashion": "#FF6B6B",        # Vibrant Coral
        "Supplement": "#4D96FF",     # Sky Blue
        "Home_Appliance": "#6BCB77",  # Sage Green
        "Electronics": "#1A5276",     # Deep Royal Blue
        "Cosmetics": "#E74C3C"        # Rose Magenta
    }
    
    ax = sns.barplot(
        data=agg_df, 
        x='province', 
        y='is_returned', 
        hue='category', 
        palette=palette,
        order=target_provinces,
        hue_order=target_categories,
        edgecolor='black',
        linewidth=0.6
    )
    
    # Annotate bars with values
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f'{height:.1f}%',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center',
                xytext=(0, 7),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                color='#2C3E50'
            )
            
    plt.title("Return Rate (%) by Category across Provinces (Detailed Interaction Analysis)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Province", fontsize=13, labelpad=12)
    plt.ylabel("Return Rate (%)", fontsize=13, labelpad=12)
    plt.ylim(0, 50)  # Max return rate is ~42%, so 50 is a great limit to leave space for labels
    
    # Customize legend
    plt.legend(
        title='Product Category', 
        title_fontsize='12', 
        fontsize='11', 
        loc='upper right',
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        framealpha=0.9
    )
    
    plt.xticks(rotation=15, fontsize=11, fontweight='semibold')
    plt.yticks(fontsize=11)
    plt.tight_layout()
    
    # Save image
    output_dir = 'reports/Graph Item'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'eda_province_category_return_rate.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # Console table print
    print("\n[INFO] Complete Return Rate (%) Matrix:")
    pivot_df = agg_df.pivot(index='province', columns='category', values='is_returned')
    pivot_df = pivot_df.reindex(target_provinces)[target_categories]
    print(pivot_df.round(2).to_string())
    
    print(f"\n[SUCCESS] Return Rate comparison graph saved to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_single_return_rate_analysis()
