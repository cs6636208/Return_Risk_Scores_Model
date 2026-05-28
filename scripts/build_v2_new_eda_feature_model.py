from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "processed" / "clean_dataset_v2.csv"

PACKAGE_DIR = ROOT / "docs" / "version 2" / "v2_xgboost_safe_plus_rolling_NEW"
DATA_DIR = PACKAGE_DIR / "data"
EDA_DIR = PACKAGE_DIR / "eda"
IMAGE_DIR = PACKAGE_DIR / "images"
MODEL_DIR = PACKAGE_DIR / "models"
REPORT_DIR = PACKAGE_DIR / "reports"
DOC_DIR = PACKAGE_DIR / "docs"

ROOT_ENGINEERED_PATH = ROOT / "data" / "processed" / "df_engineered_v2_NEW.csv"
PACKAGE_ENGINEERED_PATH = DATA_DIR / "df_engineered_v2_NEW.csv"
TRAIN_TEST_PATH = DATA_DIR / "train_test_sets_v2_xgboost_safe_plus_rolling.pkl"
TEST_TRAIN_ALIAS_PATH = DATA_DIR / "test_train_sets_v2_xgboost_safe_plus_rolling.pkl"
BEST_MODEL_PATH = MODEL_DIR / "best_model_v2_xgboost_safe_plus_rolling_NEW.pkl"
MODEL_METADATA_PATH = MODEL_DIR / "best_model_v2_xgboost_safe_plus_rolling_NEW_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
COST_FN = 500
COST_FP = 50
ROLLING_WINDOWS = [30, 60, 90, 180, 365]
VERSION_NAME = "v2_xgboost_safe_plus_rolling_NEW"

LEAKAGE_COLUMNS = {
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
    "is_returned",
    "delivery_date",
    "delivery_days",
    "delay_days",
}

IDENTITY_COLUMNS = {
    "order_id",
    "customer_id",
    "customer_name",
    "customer_phone",
    "product_id",
    "product_name",
    "supplier_id",
    "supplier_name",
    "supplier_contact",
    "courier_id",
    "courier_name",
    "promo_id",
    "promo_name",
}


def ensure_dirs() -> None:
    for folder in [DATA_DIR, EDA_DIR, IMAGE_DIR, MODEL_DIR, REPORT_DIR, DOC_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    value_str = str(value).strip()
    return value_str if value_str else "Unknown"


def load_source() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_PATH, low_memory=False)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["return_date_dt"] = pd.to_datetime(
        df["return_date"].replace({"Not Returned": np.nan, "": np.nan}),
        errors="coerce",
    )
    df["is_returned"] = pd.to_numeric(df["is_returned"], errors="coerce").fillna(0).astype(int)
    df["is_fragile"] = df["is_fragile"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    return df.sort_values(["order_date", "order_id"]).reset_index(drop=True)


def rate_table(df: pd.DataFrame, column: str, min_count: int = 50) -> pd.DataFrame:
    table = (
        df.groupby(column, dropna=False)
        .agg(
            rows=("is_returned", "size"),
            return_count=("is_returned", "sum"),
            return_rate=("is_returned", "mean"),
            avg_amount=("total_amount", "mean"),
        )
        .reset_index()
    )
    table[column] = table[column].map(clean_text)
    table = table[table["rows"] >= min_count].copy()
    table["return_rate_pct"] = (table["return_rate"] * 100).round(2)
    table["lift_vs_average"] = (table["return_rate"] / df["is_returned"].mean()).round(3)
    return table.sort_values(["return_rate", "rows"], ascending=[False, False])


def save_rate_chart(table: pd.DataFrame, column: str, title: str, filename: str, top_n: int = 12) -> None:
    plot_df = table.head(top_n).copy()
    if plot_df.empty:
        return
    plt.figure(figsize=(12, 6))
    plt.bar(plot_df[column].astype(str), plot_df["return_rate_pct"], color="#2f6f8f")
    plt.axhline(plot_df["return_rate_pct"].mean(), color="#c74e35", linestyle="--", linewidth=1.5)
    plt.title(title, fontsize=15)
    plt.ylabel("Return rate (%)")
    plt.xticks(rotation=0, ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / filename, dpi=180)
    plt.close()


def build_eda_and_insights(df: pd.DataFrame) -> list[dict[str, str]]:
    profile = pd.DataFrame(
        [
            {"metric": "rows", "value": len(df)},
            {"metric": "columns", "value": len(df.columns)},
            {"metric": "missing_cells", "value": int(df.isna().sum().sum())},
            {"metric": "unique_orders", "value": df["order_id"].nunique()},
            {"metric": "unique_customers", "value": df["customer_id"].nunique()},
            {"metric": "return_rate_pct", "value": round(df["is_returned"].mean() * 100, 2)},
            {"metric": "min_order_date", "value": str(df["order_date"].min())},
            {"metric": "max_order_date", "value": str(df["order_date"].max())},
        ]
    )
    profile.to_csv(EDA_DIR / "eda_profile_v2_NEW.csv", index=False, encoding="utf-8")

    df_eda = df.copy()
    df_eda["discount_band"] = pd.cut(
        pd.to_numeric(df_eda["total_discount_pct"], errors="coerce").fillna(0),
        bins=[-0.001, 0.05, 0.10, 0.15, 0.25, 1.0],
        labels=["<=5%", "5-10%", "10-15%", "15-25%", ">25%"],
    ).astype(str)
    df_eda["rating_band"] = pd.cut(
        pd.to_numeric(df_eda["product_rating"], errors="coerce").fillna(0),
        bins=[0, 3.8, 4.2, 4.6, 5.0],
        labels=["<=3.8", "3.8-4.2", "4.2-4.6", ">4.6"],
        include_lowest=True,
    ).astype(str)
    df_eda["history_band"] = pd.cut(
        pd.to_numeric(df_eda["hist_return_rate"], errors="coerce").fillna(0),
        bins=[-0.001, 0, 0.25, 0.50, 0.75, 1.0],
        labels=["0", "0-25%", "25-50%", "50-75%", "75-100%"],
        include_lowest=True,
    ).astype(str)

    eda_specs = [
        ("category", "Return Rate by Category", "eda_return_rate_by_category.png"),
        ("payment_method", "Return Rate by Payment Method", "eda_return_rate_by_payment.png"),
        ("channel_type", "Return Rate by Sales Channel", "eda_return_rate_by_channel.png"),
        ("province", "Return Rate by Province", "eda_return_rate_by_province.png"),
        ("discount_band", "Return Rate by Discount Band", "eda_return_rate_by_discount.png"),
        ("rating_band", "Return Rate by Product Rating Band", "eda_return_rate_by_rating.png"),
        ("history_band", "Return Rate by Historical Return Rate", "eda_return_rate_by_history.png"),
        ("membership_tier", "Return Rate by Membership Tier", "eda_return_rate_by_tier.png"),
    ]

    insights: list[dict[str, str]] = []
    for column, title, filename in eda_specs:
        table = rate_table(df_eda, column)
        table.to_csv(EDA_DIR / f"eda_{column}_return_rate_v2_NEW.csv", index=False, encoding="utf-8")
        save_rate_chart(table, column, title, filename)
        top = table.iloc[0]
        insights.append(
            {
                "insight_area": column,
                "top_segment": str(top[column]),
                "segment_return_rate_pct": f"{float(top['return_rate_pct']):.2f}",
                "lift_vs_average": f"{float(top['lift_vs_average']):.3f}",
                "feature_action": feature_action_for(column),
            }
        )

    insights_df = pd.DataFrame(insights)
    insights_df.to_csv(EDA_DIR / "eda_insight_to_feature_mapping_v2_NEW.csv", index=False, encoding="utf-8")
    write_insight_report(profile, insights_df)
    return insights


def feature_action_for(column: str) -> str:
    actions = {
        "category": "Keep category/brand and create category_payment/category_channel interactions.",
        "payment_method": "Create is_cod and payment interaction features.",
        "channel_type": "Keep channel_type and create category_channel interaction.",
        "province": "Keep province and create province_payment interaction.",
        "discount_band": "Create is_high_discount and discount_amount_ratio.",
        "rating_band": "Create low_rating_alert and keep product_rating.",
        "history_band": "Create point-in-time rolling history windows 30/60/90/180/365 days.",
        "membership_tier": "Keep membership_tier and customer_tenure_months.",
    }
    return actions.get(column, "Keep as model input.")


def write_insight_report(profile: pd.DataFrame, insights: pd.DataFrame) -> None:
    lines = [
        "# V2 NEW EDA Insight Summary",
        "",
        "Source: `data/processed/clean_dataset_v2.csv`",
        "",
        "## Dataset Profile",
        "",
        dataframe_to_markdown(profile),
        "",
        "## Insight to Feature Mapping",
        "",
        dataframe_to_markdown(insights),
        "",
        "## Modeling Decision",
        "",
        "- Use order-time-safe features only.",
        "- Exclude post-event/leakage fields such as return/refund/risk score fields and actual delivery result fields.",
        "- Use customer historical behavior and rolling windows as the main V2 signal.",
        "- Use XGBoost as the V2 model because it handles non-linear interactions and mixed tabular patterns well.",
    ]
    (DOC_DIR / "eda_insight_summary_v2_NEW.md").write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    engineered = df.copy()
    engineered["promo_type"] = engineered["promo_type"].fillna("No Promotion")

    engineered["customer_tenure_months"] = (
        (engineered["order_date"] - engineered["registration_date"]).dt.days / 30
    ).fillna(0).clip(lower=0)
    engineered["order_month"] = engineered["order_date"].dt.month.fillna(0).astype(int)
    engineered["order_dayofweek"] = engineered["order_date"].dt.dayofweek.fillna(0).astype(int)
    engineered["is_weekend"] = engineered["order_dayofweek"].isin([5, 6]).astype(int)

    engineered["age_group"] = pd.cut(
        pd.to_numeric(engineered["age"], errors="coerce").fillna(0),
        bins=[0, 20, 30, 40, 50, 120],
        labels=["<20", "20-30", "30-40", "40-50", ">50"],
        include_lowest=True,
    ).astype(str)

    engineered["is_cod"] = engineered["payment_method"].eq("COD").astype(int)
    engineered["is_bank_transfer"] = engineered["payment_method"].eq("Bank_Transfer").astype(int)
    engineered["is_credit_card"] = engineered["payment_method"].eq("Credit_Card").astype(int)
    engineered["is_high_discount"] = (
        pd.to_numeric(engineered["total_discount_pct"], errors="coerce").fillna(0) >= 0.15
    ).astype(int)
    engineered["low_rating_alert"] = (
        pd.to_numeric(engineered["product_rating"], errors="coerce").fillna(5) < 4.0
    ).astype(int)
    gross = (
        pd.to_numeric(engineered["unit_price"], errors="coerce").fillna(0)
        * pd.to_numeric(engineered["quantity"], errors="coerce").fillna(1)
    )
    engineered["discount_amount_ratio"] = (
        pd.to_numeric(engineered["discount_applied_amount"], errors="coerce").fillna(0)
        / gross.replace(0, np.nan)
    ).fillna(0)
    engineered["amount_per_item"] = (
        pd.to_numeric(engineered["total_amount"], errors="coerce").fillna(0)
        / pd.to_numeric(engineered["quantity"], errors="coerce").fillna(1).replace(0, 1)
    )
    engineered["log_total_amount"] = np.log1p(pd.to_numeric(engineered["total_amount"], errors="coerce").fillna(0))
    engineered["high_value_order"] = (
        engineered["total_amount"] >= engineered["total_amount"].quantile(0.75)
    ).astype(int)
    engineered["logistics_risk"] = (
        pd.to_numeric(engineered["damage_rate"], errors="coerce").fillna(0) * engineered["is_fragile"].astype(int)
    )

    engineered["category_payment"] = engineered["category"].map(clean_text) + "_" + engineered["payment_method"].map(clean_text)
    engineered["category_channel"] = engineered["category"].map(clean_text) + "_" + engineered["channel_type"].map(clean_text)
    engineered["province_payment"] = engineered["province"].map(clean_text) + "_" + engineered["payment_method"].map(clean_text)
    engineered["tier_payment"] = engineered["membership_tier"].map(clean_text) + "_" + engineered["payment_method"].map(clean_text)

    engineered = add_point_in_time_rolling_features(engineered)
    feature_columns = build_feature_columns(engineered)
    identifier_columns = ["order_id", "customer_id", "order_date"]
    output_columns = identifier_columns + feature_columns + ["is_returned"]
    featured = engineered[output_columns].sort_values(["order_date", "order_id"]).reset_index(drop=True)
    return featured, feature_columns, identifier_columns


def add_point_in_time_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["customer_id", "order_date", "order_id"]).reset_index(drop=True)
    for days in ROLLING_WINDOWS:
        df[f"hist_order_count_{days}d"] = 0
        df[f"hist_return_count_{days}d"] = 0
        df[f"hist_return_rate_{days}d"] = 0.0
        df[f"hist_spend_sum_{days}d"] = 0.0

    df["days_since_last_return"] = -1
    for _, group in df.groupby("customer_id", sort=False):
        idx = group.index.to_numpy()
        dates = group["order_date"].to_numpy(dtype="datetime64[ns]")
        return_dates = group["return_date_dt"].to_numpy(dtype="datetime64[ns]")
        returned = group["is_returned"].to_numpy()
        amounts = pd.to_numeric(group["total_amount"], errors="coerce").fillna(0).to_numpy()

        for pos, current_date in enumerate(dates):
            prior_order_mask = dates < current_date
            known_prior_return_mask = (
                prior_order_mask
                & (returned == 1)
                & ~pd.isna(return_dates)
                & (return_dates < current_date)
            )

            if known_prior_return_mask.any():
                last_return_date = return_dates[known_prior_return_mask].max()
                df.loc[idx[pos], "days_since_last_return"] = int((current_date - last_return_date) / np.timedelta64(1, "D"))

            for days in ROLLING_WINDOWS:
                start_date = current_date - np.timedelta64(days, "D")
                order_mask = prior_order_mask & (dates >= start_date)
                return_mask = known_prior_return_mask & (return_dates >= start_date)
                order_count = int(order_mask.sum())
                return_count = int(return_mask.sum())

                df.loc[idx[pos], f"hist_order_count_{days}d"] = order_count
                df.loc[idx[pos], f"hist_return_count_{days}d"] = return_count
                df.loc[idx[pos], f"hist_return_rate_{days}d"] = return_count / order_count if order_count else 0.0
                df.loc[idx[pos], f"hist_spend_sum_{days}d"] = float(amounts[order_mask].sum()) if order_count else 0.0

    return df


def build_feature_columns(df: pd.DataFrame) -> list[str]:
    base_features = [
        "gender",
        "age",
        "membership_tier",
        "preferred_channel",
        "province",
        "customer_age_days",
        "customer_tenure_months",
        "category",
        "brand",
        "is_fragile",
        "product_rating",
        "courier_type",
        "avg_delivery_days",
        "damage_rate",
        "coverage_region",
        "promo_type",
        "promo_discount_rate",
        "channel_type",
        "payment_method",
        "quantity",
        "unit_price",
        "tier_discount_pct",
        "campaign_discount_pct",
        "total_discount_pct",
        "discount_applied_amount",
        "discount_amount_ratio",
        "total_amount",
        "amount_per_item",
        "log_total_amount",
        "high_value_order",
        "delivery_time_expected_days",
        "is_repurchased_item",
        "order_hour",
        "days_since_last_order",
        "days_since_last_return",
        "hist_order_count",
        "hist_return_rate",
        "order_month",
        "order_dayofweek",
        "is_weekend",
        "age_group",
        "is_cod",
        "is_bank_transfer",
        "is_credit_card",
        "is_high_discount",
        "low_rating_alert",
        "logistics_risk",
        "category_payment",
        "category_channel",
        "province_payment",
        "tier_payment",
    ]
    rolling_features = [
        f"{prefix}_{days}d"
        for days in ROLLING_WINDOWS
        for prefix in ["hist_order_count", "hist_return_count", "hist_return_rate", "hist_spend_sum"]
    ]
    columns = [col for col in base_features + rolling_features if col in df.columns and col not in LEAKAGE_COLUMNS]
    return columns


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", make_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def evaluate_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "cost": int(fn * COST_FN + fp * COST_FP),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.round(np.arange(0.20, 0.81, 0.01), 2):
        row = evaluate_threshold(y_true, y_proba, float(threshold))
        row["selection_score"] = (
            row["accuracy"] * 0.70
            + row["f1"] * 0.12
            + row["precision"] * 0.10
            + row["recall"] * 0.08
            - (row["cost"] / max(len(y_true), 1) / COST_FN) * 0.02
        )
        rows.append(row)
    threshold_df = pd.DataFrame(rows)
    best = threshold_df.sort_values(["accuracy", "f1", "recall", "selection_score"], ascending=False).iloc[0]
    return float(best["threshold"]), threshold_df


def train_xgboost(featured: pd.DataFrame, feature_columns: list[str]) -> dict[str, object]:
    X = featured[feature_columns].copy()
    y = featured["is_returned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train_core, X_val, y_train_core, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    scale_pos_weight = float((y_train_core == 0).sum() / max((y_train_core == 1).sum(), 1))
    model = XGBClassifier(
        n_estimators=520,
        max_depth=4,
        learning_rate=0.045,
        min_child_weight=3,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
    )
    validation_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    validation_pipeline.fit(X_train_core, y_train_core)
    val_proba = validation_pipeline.predict_proba(X_val)[:, 1]
    selected_threshold, threshold_table = choose_threshold(y_val.to_numpy(), val_proba)

    final_preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    final_model = XGBClassifier(
        n_estimators=520,
        max_depth=4,
        learning_rate=0.045,
        min_child_weight=3,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
    )
    final_pipeline = Pipeline(steps=[("preprocessor", final_preprocessor), ("model", final_model)])
    final_pipeline.fit(X_train, y_train)

    test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_threshold(y_test.to_numpy(), test_proba, selected_threshold)
    test_metrics["auc"] = float(roc_auc_score(y_test, test_proba))
    test_metrics["avg_precision"] = float(average_precision_score(y_test, test_proba))
    test_metrics["rows"] = int(len(featured))
    test_metrics["train_rows"] = int(len(X_train))
    test_metrics["test_rows"] = int(len(X_test))
    test_metrics["feature_count"] = int(len(feature_columns))
    test_metrics["model"] = "XGBoost"
    test_metrics["version"] = VERSION_NAME

    threshold_table.to_csv(REPORT_DIR / f"threshold_search_{VERSION_NAME}.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(REPORT_DIR / f"metrics_{VERSION_NAME}.csv", index=False)
    pd.DataFrame(
        confusion_matrix(y_test, (test_proba >= selected_threshold).astype(int), labels=[0, 1]),
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(REPORT_DIR / f"confusion_matrix_{VERSION_NAME}.csv")

    predictions = X_test.copy()
    predictions["actual_is_returned"] = y_test.to_numpy()
    predictions["return_probability"] = test_proba
    predictions["predicted_is_returned"] = (test_proba >= selected_threshold).astype(int)
    predictions.to_csv(REPORT_DIR / f"test_predictions_{VERSION_NAME}.csv", index=False)

    train_test_payload = {
        "version": VERSION_NAME,
        "source_path": str(SOURCE_PATH),
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_column": "is_returned",
        "selected_threshold": selected_threshold,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_probabilities": test_proba,
        "test_predictions": (test_proba >= selected_threshold).astype(int),
    }
    joblib.dump(train_test_payload, TRAIN_TEST_PATH)
    joblib.dump(train_test_payload, TEST_TRAIN_ALIAS_PATH)
    joblib.dump(final_pipeline, BEST_MODEL_PATH)

    metadata = {
        **test_metrics,
        "selected_threshold": selected_threshold,
        "source_path": str(SOURCE_PATH),
        "engineered_path": str(PACKAGE_ENGINEERED_PATH),
        "train_test_path": str(TRAIN_TEST_PATH),
        "test_train_alias_path": str(TEST_TRAIN_ALIAS_PATH),
        "best_model_path": str(BEST_MODEL_PATH),
        "leakage_excluded": sorted(LEAKAGE_COLUMNS),
        "identity_excluded": sorted(IDENTITY_COLUMNS),
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame({"feature": feature_columns}).to_csv(
        DATA_DIR / f"used_features_{VERSION_NAME}.csv",
        index=False,
        encoding="utf-8",
    )
    write_model_report(metadata, feature_columns, numeric_features, categorical_features)
    plot_metrics(metadata)
    return metadata


def write_model_report(
    metadata: dict[str, object],
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> None:
    lines = [
        "# V2 NEW XGBoost Safe Plus Rolling Report",
        "",
        "## Summary",
        "",
        "- Source dataset: `data/processed/clean_dataset_v2.csv`",
        "- Engineered dataset: `df_engineered_v2_NEW.csv`",
        "- Model: XGBoost with order-time-safe insight-driven features",
        "- Post-event/leakage fields are excluded from training.",
        "",
        "## Test Metrics",
        "",
        f"- Accuracy: {float(metadata['accuracy']) * 100:.2f}%",
        f"- Recall: {float(metadata['recall']) * 100:.2f}%",
        f"- Precision: {float(metadata['precision']) * 100:.2f}%",
        f"- F1: {float(metadata['f1']) * 100:.2f}%",
        f"- AUC: {float(metadata['auc']) * 100:.2f}%",
        f"- Cost: {int(metadata['cost']):,}",
        f"- Selected threshold: {float(metadata['selected_threshold']):.2f}",
        "",
        "## Feature Groups",
        "",
        f"- Total features: {len(feature_columns)}",
        f"- Numeric features: {len(numeric_features)}",
        f"- Categorical features: {len(categorical_features)}",
        "- Main insight-driven groups: customer history, rolling return windows, discount, payment, category/channel, province/payment, product rating, logistics risk, repurchase behavior.",
    ]
    (DOC_DIR / f"model_report_{VERSION_NAME}.md").write_text("\n".join(lines), encoding="utf-8")


def plot_metrics(metadata: dict[str, object]) -> None:
    metric_names = ["accuracy", "recall", "precision", "f1", "auc"]
    values = [float(metadata[name]) * 100 for name in metric_names]
    plt.figure(figsize=(10, 5))
    bars = plt.bar([name.upper() for name in metric_names], values, color=["#2f6f8f", "#3d8b6d", "#8f6f2f", "#7a5caa", "#c75a3a"])
    plt.ylim(0, 100)
    plt.ylabel("Percent")
    plt.title("V2 NEW XGBoost Performance")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.2f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"metrics_{VERSION_NAME}.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()
    source = load_source()
    insights = build_eda_and_insights(source)
    featured, feature_columns, _ = engineer_features(source)

    split_marker = pd.Series("train", index=featured.index)
    _, test_idx = train_test_split(
        featured.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=featured["is_returned"],
    )
    split_marker.loc[test_idx] = "test"
    featured["dataset_split"] = split_marker

    featured.to_csv(ROOT_ENGINEERED_PATH, index=False, encoding="utf-8")
    featured.to_csv(PACKAGE_ENGINEERED_PATH, index=False, encoding="utf-8")
    metadata = train_xgboost(featured, feature_columns)

    print(f"source={SOURCE_PATH}")
    print(f"eda_dir={EDA_DIR}")
    print(f"engineered={PACKAGE_ENGINEERED_PATH}")
    print(f"train_test={TRAIN_TEST_PATH}")
    print(f"test_train_alias={TEST_TRAIN_ALIAS_PATH}")
    print(f"best_model={BEST_MODEL_PATH}")
    print(f"features={len(feature_columns)} insights={len(insights)}")
    print(json.dumps({k: metadata[k] for k in ['accuracy', 'recall', 'precision', 'f1', 'auc', 'cost', 'selected_threshold']}, indent=2))


if __name__ == "__main__":
    main()
