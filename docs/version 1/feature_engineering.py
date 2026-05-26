import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def run_feature_engineering():
    data_path = 'data/processed/clean_dataset.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] Clean dataset not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)

    df['is_peak_hour'] = 0
    df.loc[(df['channel_type'] == 'TV_Show') & (df['order_hour'].between(8, 10)), 'is_peak_hour'] = 1
    df.loc[(df['channel_type'] == 'TikTok') & (df['order_hour'].between(21, 23)), 'is_peak_hour'] = 1
    
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['order_dayofweek'] = df['order_date'].dt.dayofweek
    df['is_weekend'] = df['order_dayofweek'].isin([5, 6]).astype(int)
    
    df['promo_discount_pct'] = df['total_discount_pct']
    df['is_high_discount'] = (df['promo_discount_pct'] > 0.2).astype(int)

    df_cat_sorted = df.sort_values(['category', 'order_date'])
    df_cat_sorted['return_rate_by_category_3m'] = (
        df_cat_sorted.groupby('category')
        .rolling(window='90D', on='order_date')['is_returned']
        .mean()
        .groupby(level=0).shift()
        .fillna(df_cat_sorted['is_returned'].mean())
        .values
    )
    df['return_rate_by_category'] = df_cat_sorted['return_rate_by_category_3m'].sort_index()

    df_sorted = df.sort_values(['customer_id', 'order_date'])

    # Point-in-Time Historical Features (Expanding Window)
    df_sorted['total_orders_before'] = (
        df_sorted.groupby('customer_id')['order_id']
        .expanding()
        .count()
        .groupby(level=0).shift()
        .fillna(0)
        .values
    )
    
    df_sorted['total_returns_before'] = (
        df_sorted.groupby('customer_id')['is_returned']
        .expanding()
        .sum()
        .groupby(level=0).shift()
        .fillna(0)
        .values
    )
    
    df_sorted['customer_return_ratio'] = (
        df_sorted.groupby('customer_id')['is_returned']
        .expanding()
        .mean()
        .groupby(level=0).shift()
        .fillna(0.0)
        .values
    )

    df_sorted['return_date_flag'] = np.where(df_sorted['is_returned'] == 1, df_sorted['order_date'], pd.NaT)
    df_sorted['last_return_date'] = df_sorted.groupby('customer_id')['return_date_flag'].shift()
    df_sorted['last_return_date'] = df_sorted.groupby('customer_id')['last_return_date'].ffill()
    df_sorted['last_return_date'] = pd.to_datetime(df_sorted['last_return_date'])
    df_sorted['days_since_last_return'] = (df_sorted['order_date'] - df_sorted['last_return_date']).dt.days
    df_sorted['days_since_last_return'] = df_sorted['days_since_last_return'].fillna(-1)

    df_sorted['log_unit_price'] = np.log1p(df_sorted['unit_price'])
    df_sorted['log_total_amount'] = np.log1p(df_sorted['total_amount'])
    df_sorted['gender_province'] = df_sorted['gender'].astype(str) + '_' + df_sorted['province'].astype(str)
    
    df_sorted['category_payment'] = df_sorted['category'].astype(str) + '_' + df_sorted['payment_method'].astype(str)
    df_sorted['category_channel'] = df_sorted['category'].astype(str) + '_' + df_sorted['channel_type'].astype(str)
    df_sorted['province_payment'] = df_sorted['province'].astype(str) + '_' + df_sorted['payment_method'].astype(str)

    df_sorted['hist_spend_sum_30d'] = (
        df_sorted.groupby('customer_id')
        .rolling(window='30D', on='order_date')['total_amount']
        .sum()
        .groupby(level=0).shift()
        .fillna(0)
        .values
    )
    df_sorted['hist_order_count_30d'] = (
        df_sorted.groupby('customer_id')
        .rolling(window='30D', on='order_date')['order_id']
        .count()
        .groupby(level=0).shift()
        .fillna(0)
        .values
    )
    df_sorted['hist_return_rate_30d'] = (
        df_sorted.groupby('customer_id')
        .rolling(window='30D', on='order_date')['is_returned']
        .mean()
        .groupby(level=0).shift()
        .fillna(0.0)
        .values
    )

    # 60-day (2 months) Rolling Aggregates for customer activity
    df_sorted['hist_spend_sum_60d'] = (
        df_sorted.groupby('customer_id')
        .rolling(window='60D', on='order_date')['total_amount']
        .sum()
        .groupby(level=0).shift()
        .fillna(0)
        .values
    )
    df_sorted['hist_order_count_60d'] = (
        df_sorted.groupby('customer_id')
        .rolling(window='60D', on='order_date')['order_id']
        .count()
        .groupby(level=0).shift()
        .fillna(0)
        .values
    )
    df_sorted['hist_return_rate_60d'] = (
        df_sorted.groupby('customer_id')
        .rolling(window='60D', on='order_date')['is_returned']
        .mean()
        .groupby(level=0).shift()
        .fillna(0.0)
        .values
    )

    # 180-day (6 months) Rolling Aggregates for customer activity
    df_sorted['hist_spend_sum_180d'] = (
        df_sorted.groupby('customer_id')
        .rolling(window='180D', on='order_date')['total_amount']
        .sum()
        .groupby(level=0).shift()
        .fillna(0)
        .values
    )
    df_sorted['hist_order_count_180d'] = (
        df_sorted.groupby('customer_id')
        .rolling(window='180D', on='order_date')['order_id']
        .count()
        .groupby(level=0).shift()
        .fillna(0)
        .values
    )
    df_sorted['hist_return_rate_180d'] = (
        df_sorted.groupby('customer_id')
        .rolling(window='180D', on='order_date')['is_returned']
        .mean()
        .groupby(level=0).shift()
        .fillna(0.0)
        .values
    )
    
    df = df_sorted.sort_index()
    
    df['is_fashion_tv'] = ((df['category'] == 'Fashion') & (df['channel_type'] == 'TV_Show')).astype(int)
    df['is_remote_area'] = (df['province'] == 'Remote_Area').astype(int)
    df['low_rating_alert'] = (df['product_rating'] < 4.0).astype(int)
    df['is_bracketing'] = ((df['category'] == 'Fashion') & (df['quantity'] > 1)).astype(int)
    df['is_cod'] = (df['payment_method'] == 'COD').astype(int)
    df['is_high_risk_customer'] = (df['customer_return_ratio'] > 0.2).astype(int)
    df['is_first_order'] = (df['total_orders_before'] == 0).astype(int)
    
    df['is_long_distance_cod'] = ((df['province'].isin(['Chiang Mai', 'Phuket', 'Songkhla'])) & (df['payment_method'] == 'COD')).astype(int)
    df['is_impulse_buy'] = ((df['category'] == 'Fashion') & 
                            (df['channel_type'].isin(['TV_Show', 'TikTok'])) & 
                            (df['is_peak_hour'] == 1)).astype(int)
    df['is_low_commitment'] = ((df['payment_method'] == 'COD') & 
                               (df['is_high_discount'] == 1)).astype(int)

    features_to_use = [
        'order_hour', 'channel_type', 'payment_method', 'quantity', 
        'unit_price', 'log_unit_price', 'promo_discount_pct', 'total_amount', 'log_total_amount', 
        'is_repurchased_item', 'days_since_last_order',
        'membership_tier', 'province', 'gender_province', 'customer_age_days', 'age', 'category',
        'is_fragile', 'product_rating', 'is_peak_hour', 'is_fashion_tv',
        'is_remote_area', 'low_rating_alert', 'is_cod',
        'is_high_risk_customer', 'is_high_discount', 'is_first_order',
        'order_dayofweek', 'is_weekend', 'return_rate_by_category',
        'delivery_time_expected_days',
        'is_long_distance_cod', 'is_impulse_buy', 'is_low_commitment',
        'category_payment', 'category_channel', 'province_payment',
        'total_orders_before', 'total_returns_before', 'customer_return_ratio',
        'days_since_last_return',
        'hist_spend_sum_30d', 'hist_order_count_30d', 'hist_return_rate_30d',
        'hist_spend_sum_60d', 'hist_order_count_60d', 'hist_return_rate_60d',
        'hist_spend_sum_180d', 'hist_order_count_180d', 'hist_return_rate_180d'
    ]
    
    # Features
    X = df[features_to_use].copy()
    # Target
    y = df['is_returned']
    
    # Save un-encoded features for EDA
    df_for_eda = X.copy()
    df_for_eda['is_returned'] = y
    os.makedirs('data/features', exist_ok=True)
    df_for_eda.to_csv('data/features/df_engineered.csv', index=False)
    
    tier_map = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4}
    X['membership_tier'] = X['membership_tier'].map(tier_map)
    
    # Categorical One-hot encoding
    X = pd.get_dummies(X, columns=['channel_type', 'payment_method', 'province', 'category', 'gender_province', 
                                    'category_payment', 'category_channel', 'province_payment'], drop_first=True)
    
    print(f"[INFO] Features created. Total columns: {len(X.columns)}")

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[INFO] Split Data: Train={len(X_train)}, Test={len(X_test)}")

    # Scaling Numeric columns
    scaler = StandardScaler()
    numeric_cols = ['quantity', 'unit_price', 'log_unit_price', 'promo_discount_pct', 'total_amount', 'log_total_amount',
                    'total_orders_before', 'total_returns_before', 'days_since_last_order', 
                    'customer_age_days', 'age', 'product_rating', 'return_rate_by_category', 'customer_return_ratio',
                    'days_since_last_return', 'order_dayofweek', 'delivery_time_expected_days',
                    'hist_spend_sum_30d', 'hist_order_count_30d', 'hist_return_rate_30d',
                    'hist_spend_sum_60d', 'hist_order_count_60d', 'hist_return_rate_60d',
                    'hist_spend_sum_180d', 'hist_order_count_180d', 'hist_return_rate_180d']
    
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    print(f"[INFO] Class distribution before SMOTE: {np.bincount(y_train)}")
    
    # SMOTE on Training set only to prevent leakage
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"[INFO] Class distribution after SMOTE: {np.bincount(y_train_res)}")

    os.makedirs('data/features', exist_ok=True)
    joblib.dump(scaler, 'data/features/scaler.pkl')
    
    train_test_sets = {
        'X_train': X_train_res,
        'X_test': X_test,
        'y_train': y_train_res,
        'y_test': y_test,
        'feature_names': X.columns.tolist()
    }
    joblib.dump(train_test_sets, 'data/features/train_test_sets.pkl')

if __name__ == "__main__":
    run_feature_engineering()
