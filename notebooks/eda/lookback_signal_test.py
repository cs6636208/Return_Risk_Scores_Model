import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/clean_dataset.csv")
df["order_date"] = pd.to_datetime(df["order_date"])
df["return_date"] = pd.to_datetime(df["return_date"])

df_sorted = df.sort_values(["customer_id", "order_date"]).copy()
df_sorted = df_sorted.set_index("order_date")

windows = ["7D", "14D", "30D", "60D", "90D", "180D", "365D"]
results = []

for w in windows:
    col = f"return_rate_{w}"
    df_sorted[col] = (
        df_sorted.groupby("customer_id")
        .apply(lambda g: g["is_returned"].shift(1).rolling(window=w).mean(), include_groups=False)
        .droplevel(0)
    )
    valid = df_sorted[col].dropna()
    corr = df_sorted[col].corr(df_sorted["is_returned"])
    non_null = valid.shape[0]
    null_pct = df_sorted[col].isna().sum() / len(df_sorted) * 100
    zero_pct = (valid == 0).sum() / non_null * 100 if non_null > 0 else 0
    std_val = valid.std()
    results.append(dict(window=w, corr=corr, non_null=non_null, null_pct=null_pct, zero_pct=zero_pct, std=std_val))

print("=== Hist Return Rate Correlation with is_returned ===")
print("-" * 65)
for r in results:
    w = r["window"]
    print(f"  {w:>6s} | corr={r['corr']:>8.4f} | n={r['non_null']:>5d} | null={r['null_pct']:>5.1f}% | zero={r['zero_pct']:>5.1f}% | std={r['std']:>.3f}")

results2 = []
for w in windows:
    col = f"order_count_{w}"
    df_sorted[col] = (
        df_sorted.groupby("customer_id")
        .apply(lambda g: g["order_id"].shift(1).rolling(window=w).count(), include_groups=False)
        .droplevel(0)
    )
    valid = df_sorted[col].dropna()
    corr = df_sorted[col].corr(df_sorted["is_returned"])
    non_null = valid.shape[0]
    null_pct = df_sorted[col].isna().sum() / len(df_sorted) * 100
    zero_pct = (valid == 0).sum() / non_null * 100 if non_null > 0 else 0
    std_val = valid.std()
    results2.append(dict(window=w, corr=corr, non_null=non_null, null_pct=null_pct, zero_pct=zero_pct, std=std_val))

print()
print("=== Hist Order Count Correlation with is_returned ===")
print("-" * 65)
for r in results2:
    w = r["window"]
    print(f"  {w:>6s} | corr={r['corr']:>8.4f} | n={r['non_null']:>5d} | null={r['null_pct']:>5.1f}% | zero={r['zero_pct']:>5.1f}% | std={r['std']:>.3f}")
