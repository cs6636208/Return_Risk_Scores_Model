import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_advanced_eda():

    data_path = 'data/processed/clean_dataset.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] Clean dataset not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)

    sns.set_theme(style="white")
    palette = sns.color_palette("husl", 8)
    
    output_dir = 'reports/Graph Relation Feature'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))
    cat_order = df.groupby('category')['is_returned'].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x='category', y='is_returned', order=cat_order, hue='category', palette="viridis", legend=False)
    plt.axhline(df['is_returned'].mean(), color='red', linestyle='--', label=f'Avg ({df["is_returned"].mean():.2%})')
    plt.title("Return Rate by Category", fontsize=15, pad=20)
    plt.ylabel("Return Probability")
    plt.legend()
    plt.savefig(f'{output_dir}/v2_category_pattern.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    chan_order = df.groupby('channel_type')['is_returned'].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x='channel_type', y='is_returned', order=chan_order, hue='channel_type', palette="magma", legend=False)
    plt.axhline(df['is_returned'].mean(), color='red', linestyle='--', label='Overall Avg')
    plt.title("Return Rate by Channel", fontsize=15, pad=20)
    plt.ylabel("Return Probability")
    plt.savefig(f'{output_dir}/v2_channel_pattern.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(12, 6))
    df['price_bin'] = pd.qcut(df['total_amount'], q=5, labels=['Budget', 'Economy', 'Standard', 'Premium', 'Luxury'])
    price_risk = df.groupby('price_bin', observed=False)['is_returned'].mean()
    sns.lineplot(x=price_risk.index, y=price_risk.values, marker='o', linewidth=3, color='#2ecc71')
    plt.fill_between(price_risk.index, 0, price_risk.values, alpha=0.1, color='#2ecc71')
    plt.title("Return Risk vs. Order Price Segment", fontsize=15, pad=20)
    plt.ylabel("Return Probability")
    plt.savefig(f'{output_dir}/v2_price_pattern.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    bins = [-1, 0, 5, 10, 20, 100]
    labels = ['No Discount', '1-5%', '6-10%', '11-20%', '20%+']
    df['discount_group'] = pd.cut(df['total_discount_pct'], bins=bins, labels=labels)
    sns.barplot(data=df, x='discount_group', y='is_returned', hue='discount_group', palette="flare", legend=False)
    plt.title("Return Rate by Discount Level", fontsize=15, pad=20)
    plt.ylabel("Return Probability")
    plt.savefig(f'{output_dir}/v2_promotion_pattern.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(12, 8))
    pivot_table = df.pivot_table(index='category', columns='channel_type', values='is_returned', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt=".2%", cmap="YlOrRd", cbar_kws={'label': 'Return Rate'})
    plt.title("Interaction: Category vs. Channel Return Rate", fontsize=15, pad=20)
    plt.savefig(f'{output_dir}/v2_interaction_cat_channel.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    pay_risk = df.groupby('payment_method')['is_returned'].mean().sort_values(ascending=False)
    sns.barplot(x=pay_risk.index, y=pay_risk.values, hue=pay_risk.index, palette="Set2", legend=False)
    plt.title("Return Rate by Payment Method (COD Risk Analysis)", fontsize=15, pad=20)
    plt.ylabel("Return Probability")
    plt.savefig(f'{output_dir}/v2_payment_risk.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    df['rating_group'] = pd.cut(df['product_rating'], bins=[0, 3, 4, 4.5, 5], labels=['Low (<3)', 'Mid (3-4)', 'High (4-4.5)', 'Top (4.5-5)'])
    sns.pointplot(data=df, x='rating_group', y='is_returned', color="#e74c3c", markers="D")
    plt.title("Return Rate by Product Rating (Quality Threshold)", fontsize=15, pad=20)
    plt.ylabel("Return Probability")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{output_dir}/v2_rating_threshold.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(12, 6))
    age_bins_bio = [17, 24, 35, 50, 100]
    age_labels_bio = ['Gen Z (18-24)', 'Millennials (25-35)', 'Gen X (36-50)', 'Boomers (>50)']
    df['age_group_bio'] = pd.cut(df['age'], bins=age_bins_bio, labels=age_labels_bio)
    age_risk = df.groupby('age_group_bio', observed=False)['is_returned'].mean()
    sns.barplot(x=age_risk.index, y=age_risk.values, palette="mako")
    plt.axhline(df['is_returned'].mean(), color='red', linestyle='--', label='Average Return Rate')
    plt.title("Return Rate by Biological Age Group", fontsize=15, pad=20)
    plt.ylabel("Return Probability")
    plt.legend()
    plt.savefig(f'{output_dir}/v2_biological_age_pattern.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(14, 10))
    df['delivery_gap'] = df['delivery_days'] - df['delivery_time_expected_days']
    df['is_late'] = (df['delivery_gap'] > 0).astype(int)
    df['is_cod'] = (df['payment_method'] == 'COD').astype(int)
    df['is_long_distance'] = df['province'].isin(['Chiang Mai', 'Phuket', 'Songkhla', 'Remote_Area']).astype(int)
    
    df['is_perfect_storm'] = ((df['category'] == 'Fashion') & 
                              (df['is_late'] == 1) & 
                              (df['is_cod'] == 1)).astype(int)

    core_features = [
        'is_returned', 'is_perfect_storm', 'delivery_gap', 'is_cod', 
        'is_late', 'is_long_distance', 'total_discount_pct', 
        'product_rating', 'hist_return_rate', 'customer_age_days'
    ]
    corr = df[core_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='RdBu_r', center=0, square=True, linewidths=.5)
    plt.title("Core Feature Correlation Matrix", fontsize=15, pad=20)
    plt.savefig(f'{output_dir}/v2_correlation_heatmap.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    run_advanced_eda()
