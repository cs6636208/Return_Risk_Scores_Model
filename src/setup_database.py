import pandas as pd
import numpy as np
import os
import json
from sqlalchemy import create_engine, text, Numeric, Integer, String, Boolean, DateTime
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

def generate_full_erd_data(n_orders=5000):
    print("=" * 60)
    print("[STEP 1] Generating Data based on User ERD v2...")
    print("=" * 60)
    
    np.random.seed(42)
    start_date = datetime(2025, 1, 1)

    suppliers = pd.DataFrame({
        'supplier_id': [f'SUP{str(i).zfill(3)}' for i in range(1, 11)],
        'supplier_name': ['Thai Fashion Co.', 'BangkokTech Ltd.', 'HomeStyle Group', 'Beauty Lab Inc.', 'Wellness Plus', 'SportZone Co.', 'KiddieLand', 'GadgetWorld', 'FreshLook Co.', 'NatureCare'],
        'contact': [f'02-{np.random.randint(100,999)}-{str(i).zfill(4)}' for i in range(1, 11)]
    })

    product_data = [
        ('PRD001', 'Cotton T-Shirt', 'Fashion', 'UniStyle', 'SUP001', 290.00, False, 4.5),
        ('PRD002', 'Denim Jeans', 'Fashion', 'DenimCo', 'SUP001', 890.00, False, 4.2),
        ('PRD003', 'Running Shoes', 'Fashion', 'RunFast', 'SUP006', 1990.00, False, 4.8),
        ('PRD004', 'Silk Dress', 'Fashion', 'SilkTouch', 'SUP001', 2500.00, True, 3.5),
        ('PRD005', 'Wireless Earbuds', 'Electronics', 'SoundMax', 'SUP002', 1200.00, True, 4.0),
        ('PRD006', 'Smart Watch', 'Electronics', 'TimeX', 'SUP002', 3500.0, True, 4.7),
        ('PRD007', 'Bluetooth Speaker', 'Electronics', 'BoomBox', 'SUP008', 990.00, True, 3.8),
        ('PRD008', 'Phone Case', 'Electronics', 'ShieldCase', 'SUP008', 350.00, False, 4.4),
        ('PRD009', 'Microwave', 'Home_Appliance', 'HomePro', 'SUP003', 2200.00, True, 4.1),
        ('PRD010', 'Blender', 'Home_Appliance', 'BlendIt', 'SUP003', 1100.00, True, 3.9),
        ('PRD011', 'Air Purifier', 'Home_Appliance', 'PureAir', 'SUP009', 4500.00, True, 4.6),
        ('PRD012', 'Rice Cooker', 'Home_Appliance', 'CookEasy', 'SUP003', 1500.00, True, 4.3),
        ('PRD013', 'Matte Lipstick', 'Cosmetics', 'GlowUp', 'SUP004', 450.00, True, 4.9),
        ('PRD014', 'Sunscreen SPF50', 'Cosmetics', 'SunSafe', 'SUP004', 650.00, False, 4.5),
        ('PRD015', 'Face Serum', 'Cosmetics', 'SkinLab', 'SUP010', 1200.00, True, 4.2),
        ('PRD016', 'Perfume', 'Cosmetics', 'AromaX', 'SUP004', 2800.00, True, 4.8),
        ('PRD017', 'Whey Protein', 'Supplement', 'MuscleFuel', 'SUP005', 1800.00, False, 4.0),
        ('PRD018', 'Vitamin C 1000mg', 'Supplement', 'VitaBoost', 'SUP005', 890.00, False, 3.7),
        ('PRD019', 'Collagen Drink', 'Supplement', 'GlowDrink', 'SUP010', 1200.00, False, 4.3),
        ('PRD020', 'Fish Oil', 'Supplement', 'OmegaPlus', 'SUP005', 750.00, False, 4.6)
    ]
    products = pd.DataFrame(product_data, columns=['product_id', 'product_name', 'category', 'brand', 'supplier_id', 'unit_price', 'is_fragile', 'product_rating'])

    customer_ids = [f'C{str(i).zfill(4)}' for i in range(1, 501)]
    first_names = ['Somsak', 'Wichai', 'Anong', 'Malee', 'Preecha', 'Siri', 'Kasem', 'Nipa', 'Sunee', 'Thana']
    last_names = ['Rak-Thai', 'Jaidee', 'Sawasdee', 'Mungkorn', 'Wongsuwan', 'Pattana', 'Suksamran', 'Thong-In', 'Srisai', 'Rattanakul']
    
    customers = pd.DataFrame({
        'customer_id': customer_ids,
        'customer_name': [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(500)],
        'gender': np.random.choice(['Male', 'Female', 'Other'], 500, p=[0.45, 0.5, 0.05]),
        'age': np.random.randint(18, 70, size=500),
        'customer_phone': [f'08{np.random.randint(1,9)}-{np.random.randint(100,999)}-{np.random.randint(1000,9999)}' for _ in range(500)],
        'membership_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 500, p=[0.5, 0.3, 0.15, 0.05]),
        'preferred_channel': np.random.choice(['TV', 'App', 'Web', 'LINE'], 500),
        'province': np.random.choice(['Bangkok', 'Nonthaburi', 'Chonburi', 'Chiang Mai', 'Khon Kaen', 'Songkhla', 'Phuket', 'Remote_Area'], 500),
        'registration_date': [start_date - timedelta(days=np.random.randint(30, 2000)) for _ in range(500)]
    })
    customers['customer_age_days'] = (datetime.now() - pd.to_datetime(customers['registration_date'])).dt.days

    couriers = pd.DataFrame({
        'courier_id': ['COUR01', 'COUR02', 'COUR03'],
        'courier_name': ['FastShip', 'SafeLogistics', 'EcoDelivery'],
        'courier_type': ['Express', 'Standard', 'Eco'],
        'avg_delivery_days': [1.5, 3.0, 5.0],
        'damage_rate': [0.01, 0.02, 0.05],
        'coverage_region': ['Nationwide', 'Nationwide', 'Bangkok Only']
    })

    promotions = pd.DataFrame({
        'promo_id': ['PROMO_001', 'PROMO_002', 'PROMO_003', 'PROMO_NONE'],
        'promo_name': ['New Year Mega Sale', 'Summer Fashion Week', 'Tech Gadget Fest', 'No Promotion'],
        'promo_type': ['Campaign', 'Campaign', 'Campaign', 'None'],
        'discount_rate': [0.10, 0.05, 0.15, 0.00],
        'start_date': [datetime(2025, 1, 1), datetime(2025, 4, 1), datetime(2025, 8, 1), datetime(2020, 1, 1)],
        'end_date': [datetime(2025, 1, 31), datetime(2025, 4, 30), datetime(2025, 8, 31), datetime(2030, 1, 1)]
    })

    orders_list = []
    returns_list = []
    risk_scores_list = []
    
    cust_hist = {cid: {'total': 0, 'returns': 0, 'last_date': None, 'purchased_items': set()} for cid in customer_ids}
    
    all_order_dates = []
    end_date_limit = datetime(2026, 5, 8)
    for _ in range(n_orders):
        days_from_start = np.random.randint(0, (end_date_limit - start_date).days + 1)
        order_hour = np.random.choice(range(24), p=[0.03]*8 + [0.16] + [0.04]*15)
        all_order_dates.append(start_date + timedelta(days=days_from_start, hours=int(order_hour)))
    all_order_dates.sort() 

    for i, order_date in enumerate(all_order_dates):
        order_id = f'ORD{str(i+1).zfill(5)}'
        
        eligible_customers = customers[pd.to_datetime(customers['registration_date']) <= order_date]
        if eligible_customers.empty:
            cust_id = np.random.choice(customer_ids)
        else:
            cust_id = np.random.choice(eligible_customers['customer_id'])
            
        prod_row = products.sample(1).iloc[0]
        courier_id = np.random.choice(couriers['courier_id'])
        customer_row = customers[customers['customer_id'] == cust_id].iloc[0]
        
        hist = cust_hist[cust_id]
        hist_count = hist['total']
        hist_rate = hist['returns'] / hist['total'] if hist['total'] > 0 else 0.0
        days_since = (order_date - hist['last_date']).days if hist['last_date'] else -1
        is_repurchased = 1 if prod_row['product_id'] in hist['purchased_items'] else 0
    
        tier_mapping = {'Bronze': 0.05, 'Silver': 0.10, 'Gold': 0.15, 'Platinum': 0.20}
        tier_discount_pct = tier_mapping.get(customer_row['membership_tier'], 0.0)
        
        active_promos = promotions[(promotions['start_date'] <= order_date) & (promotions['end_date'] >= order_date) & (promotions['promo_id'] != 'PROMO_NONE')]
        if not active_promos.empty and np.random.rand() < 0.4: 
            selected_promo = active_promos.sample(1).iloc[0]
            promo_id = selected_promo['promo_id']
            campaign_discount_pct = selected_promo['discount_rate']
        else:
            promo_id = 'PROMO_NONE'
            campaign_discount_pct = 0.0
            
        total_discount_pct = tier_discount_pct + campaign_discount_pct
        
        quantity = np.random.randint(1, 3)
        unit_price = prod_row['unit_price']
        discount_amt = (unit_price * quantity) * total_discount_pct
        total_amount = (unit_price * quantity) - discount_amt
        delivery_time_expected_days = np.random.randint(1, 4)
        expected_delivery_date = order_date + timedelta(days=delivery_time_expected_days)
        actual_delivery_days = np.random.randint(1, 7)
        delivery_date = order_date + timedelta(days=actual_delivery_days)
        
        risk_val = 0.04 
        if prod_row['product_rating'] < 4.0: risk_val += 0.15 
        if hist_rate > 0.15: risk_val += 0.25 
        if is_repurchased: risk_val -= 0.03 
        if total_amount > 3000: risk_val += 0.05 
        if total_discount_pct > 0.20: risk_val += 0.10
        if actual_delivery_days > delivery_time_expected_days: risk_val += 0.15 
        
        is_returned = 1 if np.random.rand() < risk_val else 0
        
        cust_hist[cust_id]['total'] += 1
        cust_hist[cust_id]['returns'] += is_returned
        cust_hist[cust_id]['last_date'] = order_date
        cust_hist[cust_id]['purchased_items'].add(prod_row['product_id'])
        
        orders_list.append({
            'order_id': order_id, 'customer_id': cust_id, 'product_id': prod_row['product_id'], 'courier_id': courier_id,
            'promo_id': promo_id, 'order_date': order_date, 'expected_delivery_date': expected_delivery_date, 
            'delivery_date': delivery_date, 'delivery_time_expected_days': delivery_time_expected_days,
            'channel_type': np.random.choice(['TV_Show', 'Mobile_App', 'TikTok', 'Shopee']),
            'payment_method': np.random.choice(['COD', 'Credit_Card', 'Bank_Transfer']),
            'quantity': quantity, 'unit_price': unit_price, 
            'tier_discount_pct': tier_discount_pct, 'campaign_discount_pct': campaign_discount_pct, 
            'total_discount_pct': total_discount_pct, 'discount_applied_amount': discount_amt, 'total_amount': total_amount,
            'delivery_days': actual_delivery_days, 'is_repurchased_item': is_repurchased,
            'order_hour': order_date.hour, 'days_since_last_order': days_since, 
            'hist_order_count': hist_count, 'hist_return_rate': hist_rate, 'is_returned': is_returned
        })
        
        if is_returned:
            returns_list.append({
                'return_id': f'RET{str(len(returns_list)+1).zfill(5)}',
                'order_id': order_id, 'customer_id': cust_id,
                'return_date': delivery_date + timedelta(days=np.random.randint(1, 10)),
                'return_reason': np.random.choice(['Defective', 'Wrong Item', 'Changed Mind', 'Better Price Elsewhere']),
                'return_scenario': 'Standard Return', 
                'item_condition': np.random.choice(['Unopened', 'Damaged Packaging', 'Used', 'Defective']),
                'return_status': 'Completed', 'refund_amount': total_amount
            })
            
        risk_scores_list.append({
            'score_id': f'SCR{str(i+1).zfill(5)}', 'order_id': order_id,
            'risk_score': min(risk_val, 1.0), 'risk_tier': 'High' if risk_val > 0.4 else ('Medium' if risk_val > 0.2 else 'Low'),
            'scored_at': order_date, 'shap_values': json.dumps({'rating': 0.1, 'history': 0.2} if is_returned else {'rating': -0.05, 'history': -0.1})
        })

    orders = pd.DataFrame(orders_list)
    returns = pd.DataFrame(returns_list)
    risk_scores = pd.DataFrame(risk_scores_list)
    
    print("[STEP 1.1] Merging all 8 tables into 'order_history_rawdata'...")
    
    df_joined = orders.merge(customers, on='customer_id', how='left') \
                      .merge(products, on='product_id', how='left') \
                      .merge(couriers, on='courier_id', how='left') \
                      .merge(suppliers, on='supplier_id', how='left') \
                      .merge(promotions, on='promo_id', how='left') \
                      .merge(risk_scores, on='order_id', how='left') \
                      .merge(returns, on='order_id', how='left', suffixes=('', '_ret'))

    os.makedirs('data/raw', exist_ok=True)
    df_joined.to_csv('data/raw/mock_return_data.csv', index=False)
    print(f"  [OK] Saved consolidated data to 'data/raw/mock_return_data.csv' ({len(df_joined)} rows)")
    
    return suppliers, products, customers, couriers, promotions, orders, returns, risk_scores, df_joined

def upload_to_db(tables):
    try:
        engine_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        engine = create_engine(engine_url)
        print("\n[STEP 2] Uploading tables to PostgreSQL with fixed types (NUMERIC)...")
        
        table_names = ['suppliers', 'products', 'customers', 'couriers', 'promotions', 'orders', 'returns', 'risk_scores', 'order_history_rawdata']
  
        dtypes_map = {
            'products': {'unit_price': Numeric(10, 2), 'product_rating': Numeric(3, 1)},
            'orders': {
                'unit_price': Numeric(10, 2), 
                'tier_discount_pct': Numeric(5, 4), 
                'campaign_discount_pct': Numeric(5, 4), 
                'total_discount_pct': Numeric(5, 4), 
                'discount_applied_amount': Numeric(10, 2), 
                'total_amount': Numeric(10, 2)
            },
            'returns': {'refund_amount': Numeric(10, 2)},
            'risk_scores': {'risk_score': Numeric(5, 4)},
            'order_history_rawdata': {
                'unit_price_x': Numeric(10, 2), 'unit_price_y': Numeric(10, 2),
                'total_amount': Numeric(10, 2), 'refund_amount': Numeric(10, 2),
                'total_discount_pct': Numeric(5, 4), 'discount_applied_amount': Numeric(10, 2)
            }
        }

        for name, df in zip(table_names, tables):
            dtype = dtypes_map.get(name, {})
            try:
                with engine.connect() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {name} CASCADE"))
                    conn.commit()
            except Exception as e:
                print(f"  [WARN] Failed to drop table {name} cascade: {e}")
                
            df.to_sql(name, engine, if_exists='replace', index=False, dtype=dtype)
            print(f"  [OK] Table '{name}' -> {len(df)} rows (Type: {'Fixed-Point' if dtype else 'Default'})")
            
    except Exception as e:
        print(f"[ERROR] Database error: {e}")

if __name__ == "__main__":
    all_tables = generate_full_erd_data()
    upload_to_db(all_tables)
    print("\n" + "=" * 60)
    print("[DONE] Database updated to match your ERD exactly!")
    print("=" * 60)
