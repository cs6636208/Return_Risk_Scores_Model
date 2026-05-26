import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

def run_eda():
    
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    df = pd.read_sql("SELECT * FROM order_history_complete_v2", engine)
    
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'Tahoma' 
    
    overall_rate = df['is_returned'].mean() * 100
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    cat_risk = df.groupby('category', observed=False)['is_returned'].mean().sort_values(ascending=False) * 100
    sns.barplot(x=cat_risk.index, y=cat_risk.values, hue=cat_risk.index, palette="viridis", legend=False)
    plt.title("Return Rate by Category (%)")
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    chan_risk = df.groupby('channel_type', observed=False)['is_returned'].mean().sort_values(ascending=False) * 100
    sns.barplot(x=chan_risk.index, y=chan_risk.values, hue=chan_risk.index, palette="magma", legend=False)
    plt.title("Return Rate by Channel (%)")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('reports/Graph Item/eda_category_channel.png')

    plt.figure(figsize=(12, 10))
    cols_to_corr = [
        'is_returned', 'product_price', 'product_rating', 'total_discount_pct',
        'order_hour', 'customer_age_days', 'hist_order_count', 'hist_return_rate',
        'day_since_last_order', 'quantity'
    ]
    existing_cols = [c for c in cols_to_corr if c in df.columns]
    corr = df[existing_cols].corr()
    
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', linewidths=0.5)
    plt.title("Feature Correlation Heatmap (Analyzing Positive & Negative Relationships)")
    plt.savefig('reports/Graph Item/eda_correlation.png')

    plt.figure(figsize=(12, 6))
    hour_risk = df.groupby('order_hour')['is_returned'].mean() * 100
    sns.lineplot(x=hour_risk.index, y=hour_risk.values, marker='o', color='red')
    plt.title("Return Risk Trend by Order Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Return Rate (%)")
    plt.xticks(range(24))
    plt.savefig('reports/Graph Item/eda_hour_trend.png')

    plt.figure(figsize=(10, 6))
    expected_risk = df.groupby('delivery_time_expected_days', observed=False)['is_returned'].mean() * 100
    sns.barplot(x=expected_risk.index, y=expected_risk.values, hue=expected_risk.index, palette="cool", legend=False)
    plt.title("Return Rate by Expected Delivery Days (%)")
    plt.xlabel("Expected Delivery Days")
    plt.ylabel("Return Rate (%)")
    plt.savefig('reports/Graph Item/eda_logistics_expected.png')

    df['delivery_gap'] = df['delivery_days'] - df['delivery_time_expected_days']
    plt.figure(figsize=(10, 6))
    gap_risk = df.groupby('delivery_gap')['is_returned'].mean() * 100
    sns.lineplot(x=gap_risk.index, y=gap_risk.values, marker='s', color='orange')
    plt.title("Return Rate by Delivery Gap (Actual - Expected)")
    plt.xlabel("Days Delayed (Positive = Late, Negative = Early)")
    plt.ylabel("Return Rate (%)")
    plt.savefig('reports/Graph Item/eda_logistics_gap.png')

    brand_counts = df['brand'].value_counts()
    popular_brands = brand_counts[brand_counts > 5].index
    brand_risk = df[df['brand'].isin(popular_brands)].groupby('brand')['is_returned'].mean().sort_values(ascending=False) * 100
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=brand_risk.head(10).values, y=brand_risk.head(10).index, hue=brand_risk.head(10).index, palette="Reds_r", legend=False)
    plt.title("Top 10 High-Risk Brands (Return Rate %)")
    plt.xlabel("Return Rate (%)")
    plt.savefig('reports/Graph Item/eda_brand_high_risk.png')

    df['rating_bin'] = pd.cut(df['product_rating'], bins=[0, 2, 3, 4, 4.5, 5], labels=['1-2', '2-3', '3-4', '4-4.5', '4.5-5'])
    rating_risk = df.groupby('rating_bin', observed=False)['is_returned'].mean() * 100
    
    plt.figure(figsize=(10, 6))
    sns.pointplot(x=rating_risk.index, y=rating_risk.values, color='darkblue')
    plt.title("Return Rate by Product Rating (Insight: Quality Threshold)")
    plt.xlabel("Product Rating Range")
    plt.ylabel("Return Rate (%)")
    plt.savefig('reports/Graph Item/eda_rating_threshold.png')

    province_risk = df.groupby('province', observed=False)['is_returned'].mean().sort_values(ascending=False) * 100
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=province_risk.index, y=province_risk.values, hue=province_risk.index, palette="rocket", legend=False)
    plt.title("Return Rate by Province (%) - Identifying High-Risk Zones")
    plt.xlabel("Province")
    plt.ylabel("Return Rate (%)")
    plt.xticks(rotation=45)
    plt.savefig('reports/Graph Item/eda_province_risk.png')

    plt.figure(figsize=(10, 6))
    df['history_group'] = pd.cut(df['hist_return_rate'], bins=[-0.1, 0, 0.2, 1.1], labels=['Never Returned', 'Low Risk (1-20%)', 'High Risk (>20%)'])
    hist_risk = df.groupby('history_group', observed=False)['is_returned'].mean() * 100
    sns.barplot(x=hist_risk.index, y=hist_risk.values, hue=hist_risk.index, palette="YlOrRd", legend=False)
    plt.title("Return Risk by Customer History (Insight: Loyalty)")
    plt.savefig('reports/Graph Item/eda_customer_history.png')

    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    tier_risk = df.groupby('membership_tier', observed=False)['is_returned'].mean().sort_values() * 100
    sns.barplot(x=tier_risk.index, y=tier_risk.values, hue=tier_risk.index, palette="flare", legend=False)
    plt.title("Return Rate by Membership Tier (%)")

    plt.subplot(1, 2, 2)
    discount_col = 'total_discount_pct' if 'total_discount_pct' in df.columns else 'discount_pct'
    df['discount_group'] = pd.cut(df[discount_col], bins=[-1, 0, 10, 20, 50], labels=['No Discount', '1-10%', '11-20%', '20%+'])
    disc_risk = df.groupby('discount_group', observed=False)['is_returned'].mean() * 100
    sns.lineplot(x=disc_risk.index, y=disc_risk.values, marker='o', color='green')
    plt.title("Return Rate by Discount Percentage (%)")

    plt.tight_layout()
    plt.savefig('reports/Graph Item/eda_tier_discount.png')

if __name__ == "__main__":
    if not os.path.exists('reports'):
        os.makedirs('reports')
    run_eda()
