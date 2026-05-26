
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style='whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# Load dataset
df = pd.read_csv('data/processed/clean_dataset.csv')
df['order_date'] = pd.to_datetime(df['order_date'])
df['return_date'] = pd.to_datetime(df['return_date'])

# 1. Time from Order to Return (Return Lag)
df_returned = df[df['is_returned'] == 1].copy()
df_returned['days_to_return'] = (df_returned['return_date'] - df_returned['order_date']).dt.days

plt.figure(figsize=(10, 6))
sns.histplot(df_returned['days_to_return'], bins=14, kde=True, color='#1f77b4', edgecolor='black')
plt.title('Distribution of Days from Order to Return', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Days to Return', fontsize=12)
plt.ylabel('Count of Returned Items', fontsize=12)
plt.tight_layout()
os.makedirs('reports', exist_ok=True)
plt.savefig('reports/days_to_return_distribution.png', dpi=300)
plt.close()

# 2. Time between consecutive purchases per customer
df_sorted = df.sort_values(by=['customer_id', 'order_date'])
df_sorted['prev_order_date'] = df_sorted.groupby('customer_id')['order_date'].shift(1)
df_sorted['days_since_prev_order'] = (df_sorted['order_date'] - df_sorted['prev_order_date']).dt.days

plt.figure(figsize=(10, 6))
sns.histplot(df_sorted['days_since_prev_order'].dropna(), bins=30, kde=True, color='#2ca02c', edgecolor='black')
plt.title('Distribution of Days Between Consecutive Purchases per Customer', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Days Between Purchases', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('reports/days_between_purchases.png', dpi=300)
plt.close()

# 3. Density/Sparsity Analysis: Weekly, Monthly, Quarterly
# We will check how many customers have orders/returns in a given week/month/quarter.
# Create a customer-period grid
min_date = df['order_date'].min()
max_date = df['order_date'].max()

# Resample by Customer-Month and Customer-Week
df_customer_order = df.copy()
df_customer_order.set_index('order_date', inplace=True)

# Monthly grouping per customer
monthly_grouped = df_customer_order.groupby('customer_id').resample('ME')['is_returned'].agg(['count', 'sum']).reset_index()
monthly_grouped.columns = ['customer_id', 'month', 'order_count', 'return_count']
monthly_active_pct = (monthly_grouped['order_count'] > 0).mean() * 100
monthly_return_pct = (monthly_grouped['return_count'] > 0).mean() * 100

# Weekly grouping per customer
weekly_grouped = df_customer_order.groupby('customer_id').resample('W')['is_returned'].agg(['count', 'sum']).reset_index()
weekly_grouped.columns = ['customer_id', 'week', 'order_count', 'return_count']
weekly_active_pct = (weekly_grouped['order_count'] > 0).mean() * 100
weekly_return_pct = (weekly_grouped['return_count'] > 0).mean() * 100

print(f'Active Customer-Months (with at least 1 order): {monthly_active_pct:.2f}%')
print(f'Return Customer-Months (with at least 1 return): {monthly_return_pct:.2f}%')
print(f'Active Customer-Weeks (with at least 1 order): {weekly_active_pct:.2f}%')
print(f'Return Customer-Weeks (with at least 1 return): {weekly_return_pct:.2f}%')

# 4. What items they return by category
category_returns = df.groupby('category').agg(
    total_orders=('order_id', 'count'),
    total_returns=('is_returned', 'sum')
).reset_index()
category_returns['return_rate'] = category_returns['total_returns'] / category_returns['total_orders']
category_returns = category_returns.sort_values(by='return_rate', ascending=False)
print('\n=== Category Return Rates ===')
print(category_returns)

plt.figure(figsize=(10, 6))
sns.barplot(data=category_returns, x='return_rate', y='category', palette='viridis')
plt.title('Return Rate by Product Category', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Return Rate', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.tight_layout()
plt.savefig('reports/category_return_rates.png', dpi=300)
plt.close()

# 5. Let\'s analyze return behavior per customer
customer_returns = df.groupby('customer_id').agg(
    total_orders=('order_id', 'count'),
    total_returns=('is_returned', 'sum')
).reset_index()
customer_returns['return_rate'] = customer_returns['total_returns'] / customer_returns['total_orders']

print('\n=== Customer Return Distribution ===')
print(customer_returns['total_returns'].value_counts().sort_index())

# Save results
with open('reports/customer_density_analysis.txt', 'w', encoding='utf-8') as f:
    f.write(f'Active Customer-Months (with at least 1 order): {monthly_active_pct:.2f}%\n')
    f.write(f'Return Customer-Months (with at least 1 return): {monthly_return_pct:.2f}%\n')
    f.write(f'Active Customer-Weeks (with at least 1 order): {weekly_active_pct:.2f}%\n')
    f.write(f'Return Customer-Weeks (with at least 1 return): {weekly_return_pct:.2f}%\n\n')
    f.write('=== Category Return Rates ===\n')
    f.write(category_returns.to_string() + '\n\n')
    f.write('=== Customer Return Distribution ===\n')
    f.write(customer_returns['total_returns'].value_counts().sort_index().to_string() + '\n')
