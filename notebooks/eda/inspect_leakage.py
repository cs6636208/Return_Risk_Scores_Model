import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def inspect_point_in_time():
    print("[INFO] Loading data to prove Point-in-Time logic...")
    df = pd.read_csv('data/processed/clean_dataset.csv')
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # เรียงลำดับตามลูกค้าและวันที่สั่งซื้อ
    df_sorted = df.sort_values(['customer_id', 'order_date']).copy()
    
    # 1. จำลองโค้ด total_orders_before (สะสมตลอดชีวิต ยกเว้นครั้งนี้)
    df_sorted['total_orders_before'] = (
        df_sorted.groupby('customer_id')['order_id']
        .expanding()
        .count()
        .groupby(level=0).shift() # เลื่อน 1 step เพื่อป้องกันการนับรอบปัจจุบัน
        .fillna(0)
        .values
    )
    
    # 2. จำลองโค้ด hist_order_count_90d (นับย้อนหลัง 90 วัน ยกเว้นครั้งนี้)
    df_sorted['hist_order_count_90d'] = (
        df_sorted.groupby('customer_id')
        .rolling(window='90D', on='order_date')['order_id']
        .count()
        .groupby(level=0).shift() # เลื่อน 1 step เช่นกัน
        .fillna(0)
        .values
    )
    
    # หาตัวอย่างลูกค้าที่สั่งของบ่อยๆ (มากกว่า 5 ครั้ง) เพื่อให้เห็นภาพชัดเจน
    order_counts = df_sorted['customer_id'].value_counts()
    frequent_shoppers = order_counts[order_counts > 5].index
    
    if len(frequent_shoppers) > 0:
        target_customer = frequent_shoppers[0]
        sample = df_sorted[df_sorted['customer_id'] == target_customer]
        
        # เลือกเฉพาะคอลัมน์ที่อยากโชว์
        cols_to_show = ['customer_id', 'order_id', 'order_date', 'total_orders_before', 'hist_order_count_90d']
        
        print("\n=========================================================")
        print(f" Sample Order History for Customer: {target_customer}")
        print("=========================================================")
        print(sample[cols_to_show].to_string(index=False))
        print("=========================================================")
        
if __name__ == "__main__":
    inspect_point_in_time()
