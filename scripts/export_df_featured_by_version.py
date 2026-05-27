from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA = ROOT / "data" / "processed" / "clean_dataset.csv"
TARGET = "is_returned"

LEAKAGE_OR_POST_EVENT = {
    "return_id",
    "return_date",
    "return_reason",
    "return_scenario",
    "item_condition",
    "return_status",
    "refund_amount",
    "score_id",
    "risk_score",
    "risk_tier",
    "scored_at",
    "shap_values",
    "delivery_date",
    "delivery_days",
    "delay_days",
}

IDENTIFIER_FIELDS = {
    "order_id",
    "customer_id",
    "customer_name",
    "customer_phone",
    "product_id",
    "supplier_id",
    "courier_id",
    "promo_id",
}


def ensure_version_dirs(version: str) -> tuple[Path, Path]:
    data_dir = ROOT / "docs" / version / "data" / "features"
    doc_dir = ROOT / "docs" / version / "feature_documentation"
    data_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, doc_dir


def read_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_DATA, low_memory=False)


def pkl_feature_frame(path: Path, train_prefix: str = "train", test_prefix: str = "test") -> pd.DataFrame:
    data = joblib.load(path)
    feature_names = list(data["feature_names"])
    x_train = pd.DataFrame(data["X_train"], columns=feature_names)
    x_test = pd.DataFrame(data["X_test"], columns=feature_names)
    train = x_train.copy()
    test = x_test.copy()
    train[TARGET] = data["y_train"]
    test[TARGET] = data["y_test"]
    train["dataset_split"] = train_prefix
    test["dataset_split"] = test_prefix
    return pd.concat([train, test], ignore_index=True)


def v4_feature_frame(path: Path) -> pd.DataFrame:
    data = joblib.load(path)
    feature_names = list(data["feature_names"])
    x_train = pd.DataFrame(data["X_train"], columns=feature_names)
    x_test = pd.DataFrame(data["X_test"], columns=feature_names)
    train = x_train.copy()
    test = x_test.copy()
    train[TARGET] = data["y_train"]
    test[TARGET] = data["y_test"]
    train["dataset_split"] = "train"
    test["dataset_split"] = "test"
    return pd.concat([train, test], ignore_index=True)


def load_v2_training_module() -> Any:
    preprocessing_path = ROOT / "docs" / "version 2" / "production_v2" / "preprocessing.py"
    if preprocessing_path.exists() and "src.production_v2.preprocessing" not in sys.modules:
        src_module = types.ModuleType("src")
        production_module = types.ModuleType("src.production_v2")
        sys.modules.setdefault("src", src_module)
        sys.modules.setdefault("src.production_v2", production_module)
        preprocessing_spec = importlib.util.spec_from_file_location("src.production_v2.preprocessing", preprocessing_path)
        if preprocessing_spec is None or preprocessing_spec.loader is None:
            raise RuntimeError(f"Cannot import {preprocessing_path}")
        preprocessing_module = importlib.util.module_from_spec(preprocessing_spec)
        sys.modules["src.production_v2.preprocessing"] = preprocessing_module
        preprocessing_spec.loader.exec_module(preprocessing_module)

    path = ROOT / "docs" / "version 2" / "train_v2_optimized_model.py"
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("v2_optimized_export_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.DATA_PATH = CLEAN_DATA
    return module


def v2_safe_rolling_feature_frame() -> pd.DataFrame:
    module = load_v2_training_module()
    df = module.load_v2_frame()
    feature_columns = module.build_feature_sets(df)["v2_safe_plus_rolling"]
    out = df[feature_columns + [TARGET]].copy()
    out["dataset_split"] = "full_feature_frame"
    return out


def used_features_from_df(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {TARGET, "dataset_split"}]


def read_engineered_columns(path: Path) -> list[str]:
    if not path.exists():
        return []
    return list(pd.read_csv(path, nrows=5, low_memory=False).columns)


def infer_source_used(clean_cols: list[str], used_features: list[str]) -> set[str]:
    used = set(used_features)
    source = set()
    for col in clean_cols:
        if col in used:
            source.add(col)
            continue
        prefix = f"{col}_"
        if any(feature.startswith(prefix) for feature in used):
            source.add(col)
    return source


def feature_group(feature: str) -> str:
    low = feature.lower()
    groups = [
        ("customer_history", ["hist_", "cust_", "orders_before", "returns_before", "return_ratio", "days_since"]),
        ("product_category", ["category", "brand", "product", "rating", "fragile"]),
        ("payment_channel", ["payment", "channel", "cod"]),
        ("promotion_discount", ["promo", "discount", "campaign"]),
        ("logistics", ["courier", "delivery", "delay", "damage", "logistics"]),
        ("customer_profile", ["gender", "age", "membership", "province", "tenure"]),
        ("time", ["hour", "month", "dayofweek", "weekend", "date"]),
        ("price_amount", ["price", "amount", "quantity", "spend"]),
    ]
    for group, tokens in groups:
        if any(token in low for token in tokens):
            return group
    return "other"


def summarize_column(series: pd.Series) -> dict[str, Any]:
    samples = series.dropna().astype(str).head(5).tolist()
    return {
        "dtype": str(series.dtype),
        "non_null_count": int(series.notna().sum()),
        "missing_count": int(series.isna().sum()),
        "unique_count": int(series.nunique(dropna=True)),
        "sample_values": " | ".join(samples),
    }


def schema_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        row = {"column": col}
        row.update(summarize_column(df[col]))
        rows.append(row)
    return pd.DataFrame(rows)


def feature_status(feature: str, used_source: set[str]) -> tuple[str, str]:
    if feature == TARGET:
        return "target", "target label ไม่ใช่ input feature"
    if feature in used_source:
        return "used_source_or_encoded", "ใช้ตรงหรือถูก encode/derive เป็น model feature"
    if feature in LEAKAGE_OR_POST_EVENT:
        return "dropped", "post-event/leakage field ไม่ควรใช้เป็น input model"
    if feature in IDENTIFIER_FIELDS:
        return "query_or_audit_only", "ใช้ join/query/audit ได้ แต่ไม่ควรให้ model จำ identity ตรง ๆ"
    return "not_used_or_replaced", "ไม่ได้ใช้ตรง หรือถูกแทนด้วย engineered/encoded feature"


def make_feature_documentation(
    version_key: str,
    output_df: pd.DataFrame,
    source_before_feature: str,
    engineered_dataset: str,
    engineered_cols: list[str],
    model_name: str,
    note: str,
) -> None:
    data_dir, doc_dir = ensure_version_dirs(version_key)
    df_featured_path = data_dir / "df_featured.csv"
    output_df.to_csv(df_featured_path, index=False, encoding="utf-8-sig")
    output_df.head(30).to_csv(doc_dir / "df_featured_preview.csv", index=False, encoding="utf-8-sig")
    schema_profile(output_df).to_csv(doc_dir / "df_featured_schema.csv", index=False, encoding="utf-8-sig")

    clean_cols = list(read_clean().columns)
    used_features = used_features_from_df(output_df)
    used_source = infer_source_used(clean_cols, used_features)
    used_rows = []
    for feature in used_features:
        used_rows.append(
            {
                "feature": feature,
                "feature_group": feature_group(feature),
                "in_clean_dataset_exact": feature in clean_cols,
                "in_engineered_dataset_exact": feature in engineered_cols,
                "encoded_or_generated": feature not in clean_cols and feature not in engineered_cols,
                "note": "model input feature",
            }
        )
    pd.DataFrame(used_rows).to_csv(doc_dir / "used_features.csv", index=False, encoding="utf-8-sig")

    dropped_rows = []
    for feature in clean_cols:
        status, reason = feature_status(feature, used_source)
        if status not in {"used_source_or_encoded", "target"}:
            dropped_rows.append(
                {
                    "source": "clean_dataset.csv",
                    "feature": feature,
                    "status": status,
                    "reason": reason,
                    "feature_group": feature_group(feature),
                }
            )
    used_set = set(used_features)
    for feature in engineered_cols:
        if feature == TARGET:
            continue
        if feature not in used_set and not any(used.startswith(f"{feature}_") for used in used_features):
            dropped_rows.append(
                {
                    "source": engineered_dataset,
                    "feature": feature,
                    "status": "engineered_not_used_exact",
                    "reason": "ไม่อยู่ใน selected model feature set แบบ exact หรือถูกแทนด้วย feature อื่น",
                    "feature_group": feature_group(feature),
                }
            )
    pd.DataFrame(dropped_rows).to_csv(doc_dir / "dropped_or_not_used_features.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "version": version_key,
                "source_before_feature": source_before_feature,
                "engineered_dataset": engineered_dataset,
                "df_featured_output": str(df_featured_path.relative_to(ROOT)),
                "rows": len(output_df),
                "columns_in_df_featured": len(output_df.columns),
                "model_input_feature_count": len(used_features),
                "target_column": TARGET,
                "model_name": model_name,
                "note": note,
            }
        ]
    )
    summary.to_csv(doc_dir / "source_data_summary.csv", index=False, encoding="utf-8-sig")

    readme = f"""# {version_key} df_featured Export

## Source ก่อนทำ Feature

- `{source_before_feature}`

## Engineered Dataset ที่ใช้อ้างอิง

- `{engineered_dataset}`

## Output หลังทำ Feature

- `data/features/df_featured.csv`
- rows: `{len(output_df)}`
- model input features: `{len(used_features)}`
- target: `{TARGET}`
- split column: `dataset_split`

## Files ในโฟลเดอร์นี้

- `source_data_summary.csv`: อธิบายว่า data มากจากไหนและ output คืออะไร
- `used_features.csv`: feature ที่ใช้เข้า model
- `dropped_or_not_used_features.csv`: feature ที่ตัดทิ้งหรือไม่ใช้ เทียบกับ clean/engineered
- `df_featured_schema.csv`: dtype, missing, unique, sample values ของ df_featured
- `df_featured_preview.csv`: ตัวอย่าง 30 rows แรก สำหรับเปิดดูเร็วใน Excel

## Note

{note}
"""
    (doc_dir / "README_df_featured.md").write_text(readme, encoding="utf-8")


def main() -> None:
    clean = read_clean()

    v1 = pkl_feature_frame(ROOT / "docs" / "version 1" / "data" / "features" / "train_test_sets.pkl")
    make_feature_documentation(
        "version 1",
        v1,
        "data/processed/clean_dataset.csv",
        "docs/version 1/data/features/df_engineered.csv",
        read_engineered_columns(ROOT / "docs" / "version 1" / "data" / "features" / "df_engineered.csv"),
        "XGBClassifier",
        "V1 export เป็น encoded/scaled model feature set 136 features จาก train_test_sets.pkl จึงเป็น feature ที่เข้า model จริง",
    )

    v2 = v2_safe_rolling_feature_frame()
    make_feature_documentation(
        "version 2",
        v2,
        "data/processed/clean_dataset.csv",
        "docs/version 2/data/features/df_engineered_v2_preview.csv + rolling features from train_v2_optimized_model.py",
        read_engineered_columns(ROOT / "docs" / "version 2" / "data" / "features" / "df_engineered_v2_preview.csv"),
        "XGBClassifier: v2_xgboost_safe_plus_rolling",
        "V2 selected export เป็น order-time-safe raw feature frame 60 features ตัด delivery_days/delay_days และเพิ่ม rolling history 30/60/90/180/365d",
    )
    # Mirror selected V2 df_featured into the selected model package too.
    selected_v2_data = ROOT / "docs" / "version 2" / "v2_xgboost_safe_plus_rolling" / "data"
    selected_v2_data.mkdir(parents=True, exist_ok=True)
    v2.to_csv(selected_v2_data / "df_featured.csv", index=False, encoding="utf-8-sig")

    v3 = pkl_feature_frame(ROOT / "docs" / "version 3" / "stacking_model_v3" / "data" / "train_test_sets_v3_from_v2.pkl")
    make_feature_documentation(
        "version 3",
        v3,
        "docs/version 2/data/features/train_test_sets_v2.pkl",
        "docs/version 2/data/features/df_engineered_v2_preview.csv",
        read_engineered_columns(ROOT / "docs" / "version 2" / "data" / "features" / "df_engineered_v2_preview.csv"),
        "StackingClassifier: XGBoost + LightGBM + CatBoost",
        "V3 ไม่ได้สร้าง feature ใหม่เอง แต่ reuse V2 baseline feature set 38 features เพื่อทดลอง model architecture แบบ stacking",
    )
    v3_package_data = ROOT / "docs" / "version 3" / "stacking_model_v3" / "data"
    v3_package_data.mkdir(parents=True, exist_ok=True)
    v3.to_csv(v3_package_data / "df_featured.csv", index=False, encoding="utf-8-sig")

    v4 = v4_feature_frame(ROOT / "docs" / "version 4" / "data" / "features" / "train_test_sets_v4_generated.pkl")
    make_feature_documentation(
        "version 4",
        v4,
        "docs/version 4/data/processed/clean_dataset_v4_generated.csv",
        "docs/version 4/data/features/df_engineered_v4_generated.csv",
        read_engineered_columns(ROOT / "docs" / "version 4" / "data" / "features" / "df_engineered_v4_generated.csv"),
        "XGBoost_SMOTE_Optuna",
        "V4 export เป็น generated/synthetic model feature set 180 features จาก train_test_sets_v4_generated.pkl ไม่ควรเทียบ production ตรงกับข้อมูลจริง",
    )

    print("Exported df_featured by version")
    for version in ["version 1", "version 2", "version 3", "version 4"]:
        data_dir, doc_dir = ensure_version_dirs(version)
        print(data_dir.relative_to(ROOT) / "df_featured.csv")
        print(doc_dir.relative_to(ROOT) / "used_features.csv")
        print(doc_dir.relative_to(ROOT) / "dropped_or_not_used_features.csv")
        print(doc_dir.relative_to(ROOT) / "source_data_summary.csv")


if __name__ == "__main__":
    main()
