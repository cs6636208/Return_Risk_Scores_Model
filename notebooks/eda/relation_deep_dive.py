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

def run_relation_analysis():
    
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    df = pd.read_sql("SELECT * FROM order_history_complete_v2", engine)
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'Tahoma'
    
    pivot_cat_chan = df.pivot_table(index='category', columns='channel_type', values='is_returned', aggfunc='mean') * 100
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_cat_chan, annot=True, cmap="YlOrRd", fmt=".1f")
    plt.title("วิเคราะห์อัตราการคืนสินค้า (%) แยกตามหมวดหมู่ และช่องทางการสั่งซื้อ")
    plt.savefig('reports/Graph Relation Feature/relation_cat_channel.png')

    plt.figure(figsize=(10, 6))
    pay_risk = df.groupby('payment_method')['is_returned'].mean().sort_values(ascending=False) * 100
    sns.barplot(x=pay_risk.index, y=pay_risk.values, hue=pay_risk.index, palette="Set2", legend=False)
    plt.title("อัตราการคืนสินค้า (%) ตามช่องทางการชำระเงิน (COD Risk Analysis)")
    plt.ylabel("Return Rate (%)")
    plt.savefig('reports/Graph Relation Feature/relation_payment_risk.png')

    df['delivery_gap'] = df['delivery_days'] - df['delivery_time_expected_days']
    pivot_prov_gap = df.pivot_table(index='province', columns='delivery_gap', values='is_returned', aggfunc='mean') * 100
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_prov_gap, annot=True, cmap="OrRd", fmt=".1f")
    plt.title("อัตราการคืนของ % จังหวัด / ที่อยู่ของลูกค้า เทียบกับ ระยะเวลาการจัดส่ง (Actual-Expected)")
    plt.savefig('reports/Graph Relation Feature/relation_province_gap.png')

    df['rating_group'] = pd.cut(df['product_rating'], bins=[0, 3.5, 4.5, 5.1], labels=['Rating ต่ำ (<3.5)', 'Rating ปานกลาง (3.5-4.5)', 'Rating สูง (>4.5)'])

    df_returns = df[df['is_returned'] == 1]
    reason_rating = pd.crosstab(df_returns['rating_group'], df_returns['return_reason'], normalize='index') * 100
    
    reason_rating.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set3')
    plt.title("อัตราการคืนของ % สาเหตุการคืนสินค้า ตามระดับคะแนนของสินค้า")
    plt.ylabel("Proportion (%)")
    plt.legend(title='Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('reports/Graph Relation Feature/relation_rating_reason.png')

    pivot_hour_chan = df.pivot_table(index='order_hour', columns='channel_type', values='is_returned', aggfunc='mean') * 100
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_hour_chan, cmap="YlGnBu")
    plt.title("Heatmap: Return Risk by Hour & Channel")
    plt.savefig('reports/Graph Relation Feature/relation_hour_channel.png')

    df['tenure_group'] = pd.cut(df['customer_age_days'], bins=[0, 365, 1095, 3650], labels=['ลูกค้ารายใหม่ < 1 ปี', 'ลูกค้าปกติ (1-3 ปี)', 'ลูกค้าประจำ (>3 ปี)'])
    tenure_risk = df.groupby('tenure_group', observed=False)['is_returned'].mean() * 100
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=tenure_risk.index, y=tenure_risk.values, hue=tenure_risk.index, palette="Greens_r", legend=False)
    plt.title("Return Rate % by Customer Tenure (Insight: Loyalty)")
    plt.savefig('reports/Graph Relation Feature/relation_tenure_risk.png')

    plt.figure(figsize=(10, 6))
    discount_col = 'total_discount_pct' if 'total_discount_pct' in df.columns else 'discount_pct'
    sns.boxplot(data=df, x='is_returned', y=discount_col, hue='is_returned', palette="Set3", legend=False)
    plt.title("Discount % Distribution: Returned vs Not Returned")
    plt.savefig('reports/Graph Relation Feature/relation_discount_hunter.png')

    plt.figure(figsize=(8, 6))
    repurchase_risk = df.groupby('is_repurchased_item')['is_returned'].mean() * 100
    sns.barplot(x=repurchase_risk.index, y=repurchase_risk.values, hue=repurchase_risk.index, palette="cool", legend=False)
    plt.title("Return Rate: First Time Buy vs Repurchased Item")
    plt.xticks([0, 1], ['First Time', 'Repurchased'])
    plt.savefig('reports/Graph Relation Feature/relation_repurchase_risk.png')

    pivot_tier_price = df.pivot_table(index='membership_tier', values=['total_amount', 'is_returned'], aggfunc='mean')
    pivot_tier_price['is_returned'] *= 100
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    sns.barplot(x=pivot_tier_price.index, y=pivot_tier_price['total_amount'], ax=ax1, alpha=0.6, color='blue')
    sns.lineplot(x=pivot_tier_price.index, y=pivot_tier_price['is_returned'], ax=ax2, marker='o', color='red')
    ax1.set_ylabel('Avg Order Amount (THB)', color='blue')
    ax2.set_ylabel('Return Rate (%)', color='red')
    plt.title("Tier Analysis: Order Amount vs Return Risk")
    plt.savefig('reports/Graph Relation Feature/relation_tier_spending.png')


if __name__ == "__main__":
    run_relation_analysis()
