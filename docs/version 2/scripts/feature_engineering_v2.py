import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

class SmoothTargetEncoder:
    """Target encoder with smoothing to prevent overfitting on small categories."""
    def __init__(self, m=10):
        self.m = m
        self.global_mean = 0
        self.category_means = {}

    def fit(self, X, y, cols):
        self.cols = cols
        self.global_mean = y.mean()
        for col in cols:
            # Calculate count and mean for each category
            stats = pd.DataFrame({
                'count': X.groupby(col).size(),
                'mean': y.groupby(X[col]).mean()
            })
            # Smoothed mean formula
            stats['smoothed'] = (stats['count'] * stats['mean'] + self.m * self.global_mean) / (stats['count'] + self.m)
            self.category_means[col] = stats['smoothed'].to_dict()

    def transform(self, X):
        X_out = X.copy()
        for col in self.cols:
            # Map values, fill unknown categories with global mean
            X_out[col] = X_out[col].map(self.category_means[col]).fillna(self.global_mean)
        return X_out

def run_feature_engineering_v2():
    print("=" * 60)
    print("FEATURE ENGINEERING V2 (TARGET ENCODING, NO SMOTE)")
    print("=" * 60)
    
    data_path = 'data/processed/clean_dataset.csv'
    if not os.path.exists(data_path):
        print(f"[ERROR] Clean dataset not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"[INFO] Initial data shape: {df.shape}")

    # 1. DROP LEAKAGE AND IRRELEVANT COLUMNS
    drop_cols = [
        'order_id', 'customer_id', 'customer_name', 'customer_phone',
        'product_id', 'product_name', 'supplier_id', 'supplier_name', 'supplier_contact',
        'courier_id', 'return_id', 'return_date', 'return_reason', 'return_scenario', 
        'item_condition', 'return_status', 'refund_amount', 'score_id', 
        'risk_score', 'risk_tier', 'scored_at', 'shap_values',
        'expected_delivery_date', 'delivery_date' # Future leakage
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # 2. TIME-BASED FEATURES
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    
    # Calculate Customer Tenure in Months
    df['customer_tenure_months'] = ((df['order_date'] - df['registration_date']).dt.days / 30).fillna(0)
    
    df['order_month'] = df['order_date'].dt.month
    df['order_dayofweek'] = df['order_date'].dt.dayofweek
    df['is_weekend'] = df['order_dayofweek'].isin([5, 6]).astype(int)
    
    df.drop(columns=['order_date', 'registration_date'], inplace=True)

    # 3. BINNING AGE
    df['age_group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 100], labels=['<20', '20-30', '30-40', '40-50', '>50']).astype(str)
    
    # 4. LOGISTICS & RISK FEATURES
    df['logistics_risk'] = df['damage_rate'] * df['is_fragile']

    # 5. PREPARE FOR TARGET ENCODING
    target_encode_cols = ['province', 'brand', 'category', 'payment_method', 
                          'channel_type', 'courier_name', 'gender', 'membership_tier',
                          'courier_type', 'age_group', 'promo_name', 'promo_type']
    target_encode_cols = [c for c in target_encode_cols if c in df.columns]

    # Split Data BEFORE Target Encoding (to prevent data leakage into Test Set)
    X = df.drop(columns=['is_returned'])
    y = df['is_returned']

    # We use Stratified Split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    # 6. APPLY TARGET ENCODING
    print("[INFO] Applying Target Encoding to categorical features...")
    encoder = SmoothTargetEncoder(m=10)
    encoder.fit(X_train, y_train, target_encode_cols)
    
    X_train_encoded = encoder.transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    
    # 7. HANDLE MISSING VALUES & DROP UNUSED CATEGORICALS
    numeric_cols = X_train_encoded.select_dtypes(include=[np.number]).columns
    X_train_encoded[numeric_cols] = X_train_encoded[numeric_cols].fillna(X_train_encoded[numeric_cols].median())
    X_test_encoded[numeric_cols] = X_test_encoded[numeric_cols].fillna(X_train_encoded[numeric_cols].median())
    
    # Drop any remaining object columns (not encoded)
    X_train_encoded = X_train_encoded.select_dtypes(include=[np.number])
    X_test_encoded = X_test_encoded.select_dtypes(include=[np.number])

    # 8. SCALING
    print("[INFO] Scaling numerical features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    print(f"[INFO] Final Feature Count V2: {X_train_scaled.shape[1]} columns (Down from ~136 in V1)")
    print(f"[INFO] Target Distribution (Train): 0 = {sum(y_train==0)}, 1 = {sum(y_train==1)} (NO SMOTE)")

    # 9. SAVE DATA
    os.makedirs('data/features', exist_ok=True)
    output_data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'feature_names': list(X_train_encoded.columns)
    }
    joblib.dump(output_data, 'data/features/train_test_sets_v2.pkl')
    joblib.dump(scaler, 'data/features/scaler_v2.pkl')
    
    print("[SUCCESS] V2 Feature Engineering Completed & Saved!")

if __name__ == '__main__':
    run_feature_engineering_v2()
