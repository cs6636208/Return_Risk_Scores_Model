import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

def run_inspection():
    
    load_dotenv(dotenv_path='../../.env')  
    if not os.getenv("DB_USER"):
        load_dotenv(dotenv_path='../../.env')

    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT = os.getenv("DB_PORT", "5433")
    DB_USER = os.getenv("DB_USER", "admin")
    DB_PASS = os.getenv("DB_PASS", "password123")
    DB_NAME = os.getenv("DB_NAME", "gmm_oshopping_db")

    print("[INFO] Connecting to Database...")
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    df = pd.read_sql("SELECT * FROM order_history_complete_v2", engine)
    
    print("[INFO] Data Loaded. Generating Report...")
    
    df['delivery_gap'] = df['delivery_days'] - df['delivery_time_expected_days']
    df['is_late'] = df['delivery_gap'] > 0
    
    report_lines = []
    report_lines.append("# Deep Dive Data Insight Report\n")
    
    report_lines.append("## 1. Geographic & Logistics (Province Analysis)\n")
    prov_stats = df.groupby('province').agg(
        total_orders=('order_id', 'count'),
        avg_delivery_days=('delivery_days', 'mean'),
        avg_expected_days=('delivery_time_expected_days', 'mean'),
        avg_gap=('delivery_gap', 'mean'),
        late_delivery_rate=('is_late', lambda x: x.mean() * 100),
        return_rate=('is_returned', lambda x: x.mean() * 100)
    ).round(2)
    
    late_return = df[df['is_late'] == True].groupby('province')['is_returned'].mean() * 100
    ontime_return = df[df['is_late'] == False].groupby('province')['is_returned'].mean() * 100
    prov_stats['return_rate_when_late'] = late_return.round(2)
    prov_stats['return_rate_when_ontime'] = ontime_return.round(2)
    
    report_lines.append(prov_stats.to_string())
    report_lines.append("\n")
    
    report_lines.append("## 2. COD Risk by Province (Bangkok vs Regional)\n")
    cod_prov = df.pivot_table(index='province', columns='payment_method', values='is_returned', aggfunc=['mean', 'count'])
    cod_prov.columns = ['_'.join(col).strip() for col in cod_prov.columns.values]
    for col in cod_prov.columns:
        if 'mean' in col:
            cod_prov[col] = (cod_prov[col] * 100).round(2)
    report_lines.append(cod_prov.to_string())
    report_lines.append("\n")
 
    report_lines.append("## 3. Channel vs Category (Impulse Buying & Bracketing)\n")
    cat_chan = df.pivot_table(index=['category', 'channel_type'], values=['is_returned', 'order_id'], 
                              aggfunc={'is_returned': 'mean', 'order_id': 'count'})
    cat_chan['is_returned'] = (cat_chan['is_returned'] * 100).round(2)
    cat_chan = cat_chan.rename(columns={'is_returned': 'return_rate_%', 'order_id': 'total_orders'})
    report_lines.append(cat_chan.sort_values(by='return_rate_%', ascending=False).to_string())
    report_lines.append("\n")
    
    report_lines.append("## 4. Return Reasons by Category\n")
    df_returned = df[df['is_returned'] == 1]
    reason_cat = pd.crosstab(df_returned['category'], df_returned['return_reason'], normalize='index') * 100
    report_lines.append(reason_cat.round(2).to_string())
    report_lines.append("\n")

    report_path = 'reports/Graph Relation Feature/insight_data_deepdive.md'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print(f"[SUCCESS] Report saved to {report_path}")

if __name__ == "__main__":
    run_inspection()
