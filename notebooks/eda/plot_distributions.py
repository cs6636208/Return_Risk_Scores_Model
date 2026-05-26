import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run_distribution_plots():
    print("=" * 60)
    print("Starting Data Distribution Analysis...")
    print("=" * 60)
    
    data_path = 'data/processed/clean_dataset.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    df['customer_age_years'] = df['customer_age_days'] / 365.25
    df['delivery_gap'] = df['delivery_days'] - df['delivery_time_expected_days']
    
    features_to_plot = [
        ('customer_age_years', 'Customer Age (Years)', '#3498DB'),
        ('unit_price', 'Unit Price (THB)', '#2ECC71'),
        ('total_discount_pct', 'Total Discount (%)', '#E74C3C'),
        ('product_rating', 'Product Rating', '#F1C40F'),
        ('hist_return_rate', 'Customer Historical Return Rate', '#9B59B6'),
        ('delivery_gap', 'Delivery Gap (Actual - Expected Days)', '#E67E22')
    ]
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    hue_palette = {0: '#2E86C1', 1: '#CB4335'}
    
    for idx, (col, title, default_color) in enumerate(features_to_plot):
        ax = axes[idx]
       
        plot_df = df[[col, 'is_returned']].dropna()
        plot_df = plot_df[np.isfinite(plot_df[col])]
        
        if col == 'total_discount_pct':
            plot_df[col] = plot_df[col] * 100
            
        sns.kdeplot(
            data=plot_df,
            x=col,
            hue='is_returned',
            palette=hue_palette,
            fill=True,
            alpha=0.4,
            linewidth=2.0,
            common_norm=False, 
            ax=ax
        )
        
        ax.set_title(f"Distribution of {title}", fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        
        legend = ax.get_legend()
        if legend:
            legend.set_title("Order Status")
            for text, label in zip(legend.get_texts(), ["Kept (0)", "Returned (1)"]):
                text.set_text(label)
                
        if col == 'unit_price':
            ax.set_xlim(0, plot_df[col].quantile(0.99)) 
        elif col == 'delivery_gap':
            ax.set_xlim(-5, 10)  

    plt.suptitle("Probability Density Distributions of Key Numerical Features (Kept vs Returned)", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_dir = 'reports/Graph Item'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'eda_feature_distributions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Feature distribution graph saved to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_distribution_plots()
