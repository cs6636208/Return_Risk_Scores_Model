import pandas as pd
import numpy as np
import os

def run_deep_cleansing():
    raw_path = 'data/raw/mock_return_data.csv'
    if not os.path.exists(raw_path):
        print(f"[ERROR] {raw_path} not found.")
        return
        
    df = pd.read_csv(raw_path)

    if 'unit_price_x' in df.columns:
        df = df.rename(columns={'unit_price_x': 'unit_price'})

    df = df[df['unit_price'] > 0]
    
    expected_total = (df['unit_price'] * df['quantity']) - df['discount_applied_amount']
    df['total_amount'] = expected_total
    print("[OK] unit_price & total_amount validated (Price * Qty - Discount).")

    df['product_rating'] = df['product_rating'].clip(1.0, 5.0)
    print("[OK] product_rating clipped to 1.0 - 5.0 range.")

    df['customer_age_days'] = df['customer_age_days'].clip(lower=0)
    print("[OK] customer_age_days validated (min 0).")

    df['hist_order_count'] = df['hist_order_count'].clip(lower=0)
    df['hist_return_rate'] = df['hist_return_rate'].clip(0.0, 1.0).round(4)
    print("[OK] History statistics (Count/Rate) validated & rounded.")

    df['days_since_last_order'] = df['days_since_last_order'].replace(999, -1)
    df.loc[df['days_since_last_order'] < -1, 'days_since_last_order'] = -1
    print("[OK] days_since_last_order formatted (-1 for new customers).")

    df['product_rating'] = df['product_rating'].round(2)
    df['total_amount'] = df['total_amount'].round(2)
    df['unit_price'] = df['unit_price'].round(2)
    df['discount_applied_amount'] = df['discount_applied_amount'].round(2)

    df['return_id'] = df['return_id'].fillna('None')
    df['return_reason'] = df['return_reason'].fillna('Not Returned')
    df['refund_amount'] = df['refund_amount'].fillna(0.00).round(2)
    
    df['return_date'] = pd.to_datetime(df['return_date'])
    df['return_date'] = df['return_date'].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('Not Returned')

    impact_features = [
        'order_hour', 'channel_type', 'payment_method', 'quantity', 'unit_price', 
        'total_discount_pct', 'discount_applied_amount', 'total_amount', 
        'delivery_time_expected_days', 
        'is_repurchased_item', 'hist_order_count',
        'hist_return_rate', 'days_since_last_order', 'membership_tier', 'province',
        'gender', 'age',
        'customer_age_days', 'category', 'is_fragile', 'product_rating', 'brand'
    ]
    analysis_only = ['delivery_days', 'return_id', 'return_reason', 'return_date', 'refund_amount']
    meta_info = ['order_id', 'customer_id', 'customer_name', 'product_name', 'order_date', 'registration_date']
    target = ['is_returned']
    
    df_clean = df[meta_info + impact_features + analysis_only + target].copy()

    print("-" * 60)
    print(f"[SUMMARY] Total Clean Records: {len(df_clean)}")
    print(f"[SUMMARY] Total Features: {len(df_clean.columns)}")
    print(f"[SAMPLE] First 3 rows of key features:\n", df_clean[['total_amount', 'product_rating', 'hist_return_rate']].head(3))
    print("-" * 60)

    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    df_clean.to_csv(f'{output_dir}/clean_dataset.csv', index=False)
    
    try:
        from sqlalchemy import create_engine
        from dotenv import load_dotenv
        load_dotenv()
        engine_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        engine = create_engine(engine_url)
        df_clean.to_sql('order_history_complete_v2', engine, if_exists='replace', index=False)
        print("[SUCCESS] Deep Clean Dataset uploaded to PostgreSQL.")
    except Exception as e:
        print(f"[WARNING] SQL Upload failed: {e}")

    print("=" * 60)

if __name__ == "__main__":
    run_deep_cleansing()
