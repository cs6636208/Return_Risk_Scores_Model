import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

def run_full_eda():
    print("[INFO] Loading engineered features data...")
    data_path = 'data/features/df_engineered.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] Data not found at {data_path}. Please run feature_engineering.py first.")
        return
        
    df = pd.read_csv(data_path)
    os.makedirs('reports/eda_full', exist_ok=True)
    
    print("[INFO] 1. Generating Correlation Heatmap...")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()

    top_corr_features = corr['is_returned'].abs().sort_values(ascending=False).head(16).index
    top_corr = df[top_corr_features].corr()
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(top_corr, dtype=bool))
    sns.heatmap(top_corr, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, 
                vmin=-1, vmax=1, square=True, linewidths=.5)
    plt.title('Top 15 Features Correlated with Return Risk', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('reports/eda_full/01_correlation_heatmap.png', dpi=300)
    plt.close()

    print("[INFO] 2. Analyzing Categorical Impact on Returns...")
    cat_features = ['channel_type', 'payment_method', 'category', 'province', 'membership_tier']
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten()
    
    for i, cat in enumerate(cat_features):
        grouped = df.groupby(cat)['is_returned'].agg(['mean', 'count']).reset_index()
        grouped = grouped.sort_values('mean', ascending=False)
        
        sns.barplot(x='mean', y=cat, data=grouped, ax=axes[i], palette='Reds_r')
        axes[i].set_title(f'Return Rate by {cat.capitalize()}', fontsize=14)
        axes[i].set_xlabel('Average Return Rate')
        axes[i].set_ylabel('')
        
        for j, row in enumerate(grouped.itertuples()):
            axes[i].text(row.mean + 0.01, j, f"(n={row.count})", va='center', fontsize=10, color='gray')
            
    axes[-1].axis('off') 
    plt.tight_layout()
    plt.savefig('reports/eda_full/02_categorical_impact.png', dpi=300)
    plt.close()

    print("[INFO] 3. Analyzing Numerical Distributions...")
    num_features = ['age', 'delivery_time_expected_days', 'total_amount', 'promo_discount_pct']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(num_features):
        sns.kdeplot(data=df, x=col, hue='is_returned', fill=True, common_norm=False, 
                    palette={0: 'blue', 1: 'red'}, alpha=0.5, ax=axes[i])
        axes[i].set_title(f'Distribution of {col} by Return Status', fontsize=12)
        
    plt.tight_layout()
    plt.savefig('reports/eda_full/03_numerical_distributions.png', dpi=300)
    plt.close()

    print("[INFO] 4. Feature Interactions (Cross-tabulations)...")
    cat_channel = pd.crosstab(df['category'], df['channel_type'], 
                              values=df['is_returned'], aggfunc='mean')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cat_channel, annot=True, fmt=".1%", cmap='YlOrRd')
    plt.title('Return Rate: Category vs Channel', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('reports/eda_full/04_interaction_cat_channel.png', dpi=300)
    plt.close()

    print("[INFO] 5. Historical Impact Analysis...")

    df['return_ratio_bin'] = pd.qcut(df['customer_return_ratio'], q=4, duplicates='drop')
    if len(df['return_ratio_bin'].unique()) < 4:
        bins = [-0.1, 0, 0.25, 0.5, 1.0]
        labels = ['0%', '1-25%', '26-50%', '>50%']
        df['return_ratio_bin'] = pd.cut(df['customer_return_ratio'], bins=bins, labels=labels)
        
    plt.figure(figsize=(10, 6))
    sns.barplot(x='return_ratio_bin', y='is_returned', data=df, palette='magma')
    plt.title('How Past Return Ratio Predicts Current Returns', fontsize=14)
    plt.xlabel('Historical Customer Return Ratio')
    plt.ylabel('Current Order Return Rate')
    plt.tight_layout()
    plt.savefig('reports/eda_full/05_historical_impact.png', dpi=300)
    plt.close()
    
    print("[INFO] Full EDA Report Generation Complete! All plots saved in reports/eda_full/")

if __name__ == "__main__":
    run_full_eda()
