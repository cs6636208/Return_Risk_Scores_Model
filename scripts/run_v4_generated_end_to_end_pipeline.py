from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "processed" / "clean_dataset_v4.csv"
if not SOURCE_PATH.exists():
    SOURCE_PATH = ROOT / "data" / "processed" / "clean_dataset.csv"

RAW_OUT = ROOT / "data" / "generated" / "v4_synthetic_orders_returns.csv"
CLEAN_OUT = ROOT / "data" / "processed" / "clean_dataset_v4_generated.csv"
FEATURE_OUT = ROOT / "data" / "features" / "df_engineered_v4_generated.csv"
TRAIN_TEST_OUT = ROOT / "data" / "features" / "train_test_sets_v4_generated.pkl"
REPORT_DIR = ROOT / "reports" / "model_evaluation_v4_generated"
EDA_DIR = ROOT / "reports" / "eda_v4_generated"
DOC_DIR = ROOT / "docs" / "analysis"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42
TARGET_RETURN_RATE = 0.15
COST_FN = 150
COST_FP = 50

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
    "delivery_date",
    "delivery_days",
    "delay_days",
}
ID_COLUMNS = {
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
    "promo_id",
}


def ensure_dirs() -> None:
    for path in [RAW_OUT.parent, CLEAN_OUT.parent, FEATURE_OUT.parent, REPORT_DIR, EDA_DIR, DOC_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_sql_template() -> None:
    sql = """-- V4 Data Collection SQL Template
-- Pull order, return, customer, product, courier, promotion, and risk-scoring data.

SELECT
    o.order_id,
    o.order_date,
    o.expected_delivery_date,
    o.delivery_date,
    c.customer_id,
    c.gender,
    c.age,
    c.membership_tier,
    c.preferred_channel,
    c.province,
    c.registration_date,
    p.product_id,
    p.category,
    p.brand,
    p.is_fragile,
    p.product_rating,
    cr.courier_id,
    cr.courier_name,
    cr.courier_type,
    pr.promo_id,
    pr.promo_name,
    pr.promo_type,
    pr.promo_discount_rate,
    o.channel_type,
    o.payment_method,
    o.quantity,
    o.unit_price,
    o.total_discount_pct,
    o.total_amount,
    r.return_id,
    r.return_date,
    r.return_reason,
    r.refund_amount,
    CASE WHEN r.return_id IS NULL THEN 0 ELSE 1 END AS is_returned
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN products p ON o.product_id = p.product_id
LEFT JOIN couriers cr ON o.courier_id = cr.courier_id
LEFT JOIN promotions pr ON o.promo_id = pr.promo_id
LEFT JOIN returns r ON o.order_id = r.order_id;
"""
    (DOC_DIR / "v4_generated_data_collection_sql.sql").write_text(sql, encoding="utf-8")


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    view = df.fillna("").astype(str)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in view.to_numpy()
    ]
    return "\n".join([header, sep, *rows])


def load_source() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_PATH, low_memory=False)
    for col in ["order_date", "expected_delivery_date", "delivery_date", "registration_date", "promo_start_date", "promo_end_date", "return_date", "scored_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def generate_synthetic_imbalance(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(RANDOM_STATE)
    original = df.copy()
    returns = int(original["is_returned"].sum())
    target_total = int(np.ceil(returns / TARGET_RETURN_RATE))
    rows_to_add = max(target_total - len(original), 0)
    non_returns = original[original["is_returned"].eq(0)].copy()
    synthetic = non_returns.sample(rows_to_add, replace=True, random_state=RANDOM_STATE).reset_index(drop=True)

    if rows_to_add:
        synthetic["order_id"] = [f"ORD_SYN_{i:06d}" for i in range(1, rows_to_add + 1)]
        if "order_date" in synthetic.columns:
            min_date = original["order_date"].min()
            max_date = original["order_date"].max()
            offsets = rng.integers(0, max((max_date - min_date).days, 1) + 1, size=rows_to_add)
            synthetic["order_date"] = min_date + pd.to_timedelta(offsets, unit="D")
            synthetic["expected_delivery_date"] = synthetic["order_date"] + pd.to_timedelta(
                rng.integers(1, 4, size=rows_to_add), unit="D"
            )
            synthetic["delivery_date"] = synthetic["expected_delivery_date"] + pd.to_timedelta(
                rng.integers(0, 3, size=rows_to_add), unit="D"
            )
        for col, sd, lower, upper in [
            ("age", 2.0, 18, 75),
            ("product_rating", 0.10, 1, 5),
            ("unit_price", 0.06, 1, None),
            ("total_discount_pct", 0.03, 0, 0.60),
            ("promo_discount_rate", 0.03, 0, 0.60),
            ("damage_rate", 0.005, 0, 0.30),
        ]:
            if col in synthetic.columns:
                vals = pd.to_numeric(synthetic[col], errors="coerce").fillna(pd.to_numeric(original[col], errors="coerce").median())
                if col == "unit_price":
                    vals = vals * rng.normal(1.0, sd, rows_to_add)
                else:
                    vals = vals + rng.normal(0, sd, rows_to_add)
                vals = np.maximum(vals, lower)
                if upper is not None:
                    vals = np.minimum(vals, upper)
                synthetic[col] = vals
        if "quantity" in synthetic.columns:
            synthetic["quantity"] = rng.choice([1, 2], rows_to_add, p=[0.65, 0.35])
        if "tier_discount_pct" in synthetic.columns and "campaign_discount_pct" in synthetic.columns:
            synthetic["total_discount_pct"] = (
                pd.to_numeric(synthetic["tier_discount_pct"], errors="coerce").fillna(0)
                + pd.to_numeric(synthetic["campaign_discount_pct"], errors="coerce").fillna(0)
            ).clip(0, 0.60)
        if {"unit_price", "quantity", "total_discount_pct"}.issubset(synthetic.columns):
            subtotal = synthetic["unit_price"] * synthetic["quantity"]
            synthetic["discount_applied_amount"] = subtotal * synthetic["total_discount_pct"]
            synthetic["total_amount"] = subtotal - synthetic["discount_applied_amount"]
        for col, value in [
            ("is_returned", 0),
            ("return_id", "NO_RETURN"),
            ("return_reason", "No_Return"),
            ("return_scenario", "No_Return"),
            ("item_condition", "No_Return"),
            ("return_status", "No_Return"),
            ("refund_amount", 0),
            ("risk_score", np.nan),
            ("risk_tier", "Not_Scored"),
            ("shap_values", "[]"),
        ]:
            if col in synthetic.columns:
                synthetic[col] = value
        if "return_date" in synthetic.columns:
            synthetic["return_date"] = pd.NaT
    combined = pd.concat([original, synthetic], ignore_index=True)
    summary = {
        "source_rows": int(len(original)),
        "synthetic_rows_added": int(rows_to_add),
        "output_rows": int(len(combined)),
        "target_return_rate": TARGET_RETURN_RATE,
        "actual_return_rate": float(combined["is_returned"].mean()),
        "target_distribution": {str(k): int(v) for k, v in combined["is_returned"].value_counts().to_dict().items()},
    }
    combined.to_csv(RAW_OUT, index=False, encoding="utf-8-sig")
    return combined, summary


def clean_generated(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    audit: list[dict] = []
    before = len(df)
    df = df.drop_duplicates()
    audit.append({"step": "drop_exact_duplicates", "rows_affected": before - len(df)})
    before = len(df)
    df = df.sort_values(["order_date", "order_id"]).drop_duplicates("order_id", keep="last")
    audit.append({"step": "drop_duplicate_order_id", "rows_affected": before - len(df)})

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    text_fill = {
        "promo_type": "No_Promotion",
        "promo_name": "No_Promotion",
        "return_id": "NO_RETURN",
        "return_reason": "No_Return",
        "return_scenario": "No_Return",
        "item_condition": "No_Return",
        "return_status": "No_Return",
        "risk_tier": "Not_Scored",
        "shap_values": "[]",
    }
    for col, value in text_fill.items():
        if col in df.columns:
            missing = int(df[col].isna().sum())
            df[col] = df[col].fillna(value)
            if missing:
                audit.append({"step": f"fill_{col}", "rows_affected": missing})
    for col in df.select_dtypes(include=[np.number]).columns:
        missing = int(df[col].isna().sum())
        if missing:
            fill_value = 0 if col in {"refund_amount", "risk_score"} else df[col].median()
            df[col] = df[col].fillna(fill_value)
            audit.append({"step": f"fill_numeric_{col}", "rows_affected": missing})
    if {"unit_price", "quantity", "total_discount_pct"}.issubset(df.columns):
        subtotal = df["unit_price"] * df["quantity"]
        expected_discount = subtotal * df["total_discount_pct"]
        mask = (df["discount_applied_amount"] - expected_discount).abs().gt(0.01)
        df.loc[mask, "discount_applied_amount"] = expected_discount.loc[mask]
        audit.append({"step": "recalculate_discount_applied_amount", "rows_affected": int(mask.sum())})
        expected_total = subtotal - df["discount_applied_amount"]
        mask = (df["total_amount"] - expected_total).abs().gt(0.01)
        df.loc[mask, "total_amount"] = expected_total.loc[mask]
        audit.append({"step": "recalculate_total_amount", "rows_affected": int(mask.sum())})
    df = df.reset_index(drop=True)
    df.to_csv(CLEAN_OUT, index=False, encoding="utf-8-sig")
    return df, audit


def write_data_dictionary(df: pd.DataFrame) -> None:
    description = {
        "order_id": "Unique order identifier",
        "customer_id": "Customer identifier",
        "is_returned": "Target: 1 if order was returned, otherwise 0",
        "hist_return_rate": "Historical customer return rate available before the order",
        "total_amount": "Order amount after discount",
        "total_discount_pct": "Total discount percentage",
    }
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique(dropna=False)),
                "description": description.get(col, "Project source field / engineered business context"),
            }
        )
    dictionary = pd.DataFrame(rows)
    dictionary.to_csv(DOC_DIR / "data_dictionary_v4_generated.csv", index=False, encoding="utf-8-sig")
    md = "# Data Dictionary V4 Generated\n\n" + df_to_markdown(dictionary)
    (DOC_DIR / "data_dictionary_v4_generated.md").write_text(md, encoding="utf-8")


def run_eda(df: pd.DataFrame) -> dict:
    sns.set_theme(style="whitegrid")
    target_counts = df["is_returned"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Not Returned", "Returned"], [target_counts.get(0, 0), target_counts.get(1, 0)], color=["#4e79a7", "#e15759"])
    ax.set_title("V4 Generated Target Distribution")
    ax.set_ylabel("Orders")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "01_target_distribution.png", dpi=160)
    plt.close(fig)

    for col, fname in [("category", "02_return_rate_by_category.png"), ("channel_type", "03_return_rate_by_channel.png"), ("payment_method", "04_return_rate_by_payment.png")]:
        if col in df.columns:
            rates = df.groupby(col)["is_returned"].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(9, 5))
            rates.plot(kind="bar", ax=ax, color="#59a14f")
            ax.set_title(f"Return Rate by {col}")
            ax.set_ylabel("Return Rate")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(EDA_DIR / fname, dpi=160)
            plt.close(fig)

    numeric = df.select_dtypes(include=[np.number]).drop(columns=["is_returned"], errors="ignore")
    corr_cols = numeric.columns[:24].tolist() + ["is_returned"]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "05_correlation_heatmap.png", dpi=160)
    plt.close(fig)

    insight = {
        "rows": int(len(df)),
        "return_rate": float(df["is_returned"].mean()),
        "top_category_return_rate": df.groupby("category")["is_returned"].mean().sort_values(ascending=False).head(3).to_dict() if "category" in df.columns else {},
        "top_channel_return_rate": df.groupby("channel_type")["is_returned"].mean().sort_values(ascending=False).head(3).to_dict() if "channel_type" in df.columns else {},
    }
    (DOC_DIR / "eda_insight_v4_generated.md").write_text(
        "# EDA Insight V4 Generated\n\n"
        f"- Rows: `{insight['rows']}`\n"
        f"- Return rate after generated imbalance: `{insight['return_rate']:.4f}`\n"
        f"- Top category return rates: `{insight['top_category_return_rate']}`\n"
        f"- Top channel return rates: `{insight['top_channel_return_rate']}`\n",
        encoding="utf-8",
    )
    return insight


def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    out = df.copy()
    out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce")
    out["registration_date"] = pd.to_datetime(out["registration_date"], errors="coerce")
    out = out.sort_values(["order_date", "order_id"]).reset_index(drop=True)

    out["order_month"] = out["order_date"].dt.month.fillna(0)
    out["order_dayofweek"] = out["order_date"].dt.dayofweek.fillna(0)
    out["is_weekend"] = out["order_dayofweek"].isin([5, 6]).astype(int)
    out["customer_tenure_months"] = ((out["order_date"] - out["registration_date"]).dt.days / 30).fillna(0)
    out["age_group"] = pd.cut(out["age"], [0, 25, 35, 45, 55, 120], labels=["<=25", "26-35", "36-45", "46-55", "56+"]).astype(str)
    out["is_cod"] = out["payment_method"].eq("COD").astype(int)
    out["is_high_discount"] = out["total_discount_pct"].gt(0.20).astype(int)
    out["low_rating_alert"] = out["product_rating"].lt(4.0).astype(int)
    out["log_unit_price"] = np.log1p(out["unit_price"])
    out["log_total_amount"] = np.log1p(out["total_amount"])
    out["discount_amount_ratio"] = (out["discount_applied_amount"] / (out["unit_price"] * out["quantity"]).replace(0, np.nan)).fillna(0)
    out["category_payment"] = out["category"].astype(str) + "_" + out["payment_method"].astype(str)
    out["category_channel"] = out["category"].astype(str) + "_" + out["channel_type"].astype(str)
    out["province_payment"] = out["province"].astype(str) + "_" + out["payment_method"].astype(str)

    for keys, prefix in [
        (["customer_id"], "cust"),
        (["category"], "cat"),
        (["brand"], "brand"),
        (["province"], "province"),
        (["payment_method"], "pay"),
        (["channel_type"], "channel"),
        (["courier_name"], "courier"),
    ]:
        group = out.groupby(keys, sort=False)
        count_before = group.cumcount()
        returns_before = group["is_returned"].cumsum() - out["is_returned"]
        out[f"{prefix}_orders_before"] = count_before
        out[f"{prefix}_returns_before"] = returns_before
        out[f"{prefix}_return_rate_pti"] = (returns_before / count_before.replace(0, np.nan)).fillna(0)

    candidate_cols = [
        c
        for c in out.columns
        if c not in LEAKAGE_COLUMNS
        and c not in ID_COLUMNS
        and c not in {"is_returned", "order_date", "registration_date", "expected_delivery_date", "promo_start_date", "promo_end_date"}
    ]
    x = out[candidate_cols].copy()
    for col in x.select_dtypes(include=[np.number, "bool"]).columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(x[col].median())
    cat_cols = [c for c in x.columns if c not in x.select_dtypes(include=[np.number, "bool"]).columns]
    x[cat_cols] = x[cat_cols].fillna("Unknown").astype(str)
    x = pd.get_dummies(x, columns=cat_cols, dummy_na=False)
    x.columns = (
        pd.Index(x.columns.astype(str))
        .str.replace(r"[^0-9A-Za-z_]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
    )
    y = out["is_returned"].astype(int)
    engineered = pd.concat([out[["order_id", "customer_id", "is_returned"]], x], axis=1)
    engineered.to_csv(FEATURE_OUT, index=False, encoding="utf-8-sig")
    return x, y, x.columns.tolist()


def split_and_smote(x: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
    scaler = StandardScaler()
    x_train_smote_scaled = scaler.fit_transform(x_train_smote)
    x_test_scaled = scaler.transform(x_test)
    joblib.dump(
        {
            "X_train": x_train.to_numpy(),
            "X_train_smote": x_train_smote.to_numpy(),
            "X_train_smote_scaled": x_train_smote_scaled,
            "X_test": x_test.to_numpy(),
            "X_test_scaled": x_test_scaled,
            "y_train": y_train.to_numpy(),
            "y_train_smote": y_train_smote.to_numpy(),
            "y_test": y_test.to_numpy(),
            "feature_names": x.columns.tolist(),
            "smote_summary": {
                "before": {str(k): int(v) for k, v in y_train.value_counts().to_dict().items()},
                "after": {str(k): int(v) for k, v in pd.Series(y_train_smote).value_counts().to_dict().items()},
            },
        },
        TRAIN_TEST_OUT,
    )
    return x_train_smote.to_numpy(), x_train_smote_scaled, x_test.to_numpy(), x_test_scaled, y_train_smote.to_numpy(), y_test.to_numpy(), x.columns.tolist(), scaler


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, proba)),
        "avg_precision": float(average_precision_score(y_true, proba)),
        "cost_thb": int(fn * COST_FN + fp * COST_FP),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_threshold(y_true: np.ndarray, proba: np.ndarray, mode: str) -> tuple[float, dict]:
    best_t = 0.50
    best_stats = evaluate(y_true, proba, best_t)
    best_score = -10**9
    for t in np.linspace(0.20, 0.85, 66):
        stats = evaluate(y_true, proba, float(t))
        if mode == "accuracy_recall_0.40":
            score = stats["accuracy"] + stats["f1"] * 0.05 if stats["recall"] >= 0.40 else stats["recall"] - 2
        elif mode == "cost":
            score = -stats["cost_thb"]
        elif mode == "f1":
            score = stats["f1"]
        else:
            score = stats["accuracy"]
        if score > best_score:
            best_score = score
            best_t = float(t)
            best_stats = stats
    return best_t, best_stats


def tune_models(
    x_train: np.ndarray,
    x_train_scaled: np.ndarray,
    x_test: np.ndarray,
    x_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> tuple[pd.DataFrame, object, str, float]:
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE)
    x_tr_s, x_val_s, _, _ = train_test_split(x_train_scaled, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE)
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    models: list[tuple[str, object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    models.append((
        "LogisticRegression_SMOTE",
        LogisticRegression(max_iter=1000, class_weight=None, random_state=RANDOM_STATE),
        x_tr_s,
        x_val_s,
        x_train_scaled,
        x_test_scaled,
    ))
    models.append((
        "RandomForest_SMOTE",
        RandomForestClassifier(n_estimators=350, max_depth=10, min_samples_leaf=4, random_state=RANDOM_STATE, n_jobs=-1),
        x_tr,
        x_val,
        x_train,
        x_test,
    ))

    def xgb_objective(trial: optuna.Trial) -> float:
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 180, 420),
            max_depth=trial.suggest_int("max_depth", 3, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12),
            subsample=trial.suggest_float("subsample", 0.75, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.70, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 5),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 3.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 0.5),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(x_tr, y_tr)
        proba = model.predict_proba(x_val)[:, 1]
        stats = evaluate(y_val, proba, 0.50)
        return stats["accuracy"] * 0.55 + stats["f1"] * 0.30 + stats["recall"] * 0.15

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    xgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    xgb_study.optimize(xgb_objective, n_trials=12, show_progress_bar=False)
    xgb_params = {
        **xgb_study.best_params,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    models.append(("XGBoost_SMOTE_Optuna", XGBClassifier(**xgb_params), x_tr, x_val, x_train, x_test))

    def lgbm_objective(trial: optuna.Trial) -> float:
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 180, 420),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            subsample=trial.suggest_float("subsample", 0.75, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.70, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 3.0),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_tr, y_tr)
        proba = model.predict_proba(x_val)[:, 1]
        stats = evaluate(y_val, proba, 0.50)
        return stats["accuracy"] * 0.55 + stats["f1"] * 0.30 + stats["recall"] * 0.15

    lgbm_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    lgbm_study.optimize(lgbm_objective, n_trials=12, show_progress_bar=False)
    lgbm_params = {**lgbm_study.best_params, "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1}
    models.append(("LightGBM_SMOTE_Optuna", LGBMClassifier(**lgbm_params), x_tr, x_val, x_train, x_test))

    rows = []
    fitted_models = {}
    thresholds = {}
    for name, model, fit_x, val_x, full_x, test_x in models:
        start = time.perf_counter()
        model.fit(fit_x, y_tr)
        val_proba = model.predict_proba(val_x)[:, 1]
        threshold, _ = find_threshold(y_val, val_proba, "accuracy_recall_0.40")
        final_model = model.__class__(**model.get_params())
        final_model.fit(full_x, y_train)
        test_proba = final_model.predict_proba(test_x)[:, 1]
        stats = evaluate(y_test, test_proba, threshold)
        default_stats = evaluate(y_test, test_proba, 0.50)
        rows.append(
            {
                "model": name,
                "threshold": threshold,
                "accuracy": stats["accuracy"],
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1": stats["f1"],
                "auc": stats["auc"],
                "avg_precision": stats["avg_precision"],
                "cost_thb": stats["cost_thb"],
                "tn": stats["tn"],
                "fp": stats["fp"],
                "fn": stats["fn"],
                "tp": stats["tp"],
                "default_accuracy": default_stats["accuracy"],
                "default_recall": default_stats["recall"],
                "default_f1": default_stats["f1"],
                "default_cost_thb": default_stats["cost_thb"],
                "train_seconds": time.perf_counter() - start,
            }
        )
        fitted_models[name] = final_model
        thresholds[name] = threshold

    results = pd.DataFrame(rows)
    results["performance_score"] = (
        results["accuracy"] * 0.45
        + results["f1"] * 0.30
        + results["recall"] * 0.20
        - results["cost_thb"] / 500000
    )
    results = results.sort_values(["performance_score", "accuracy"], ascending=False)
    best_name = results.iloc[0]["model"]
    best_model = fitted_models[best_name]
    best_threshold = float(thresholds[best_name])
    joblib.dump(best_model, MODEL_DIR / "best_model_v4_generated.pkl")
    (MODEL_DIR / "best_model_v4_generated_metadata.json").write_text(
        json.dumps(
            {
                "best_model": best_name,
                "threshold": best_threshold,
                "feature_count": len(feature_names),
                "metrics": results.iloc[0].to_dict(),
                "target_return_rate": TARGET_RETURN_RATE,
                "uses_smote": True,
                "uses_optuna": True,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    results.to_csv(REPORT_DIR / "v4_generated_model_metrics.csv", index=False, encoding="utf-8-sig")
    return results, best_model, best_name, best_threshold


def run_shap(best_model: object, best_name: str, x_test: np.ndarray, feature_names: list[str]) -> None:
    sample_size = min(350, len(x_test))
    x_sample = pd.DataFrame(x_test[:sample_size], columns=feature_names).astype(float)
    try:
        if "Logistic" in best_name:
            pd.DataFrame({"feature": feature_names, "importance": np.abs(best_model.coef_[0])}).sort_values(
                "importance", ascending=False
            ).to_csv(REPORT_DIR / "v4_generated_shap_like_feature_importance.csv", index=False, encoding="utf-8-sig")
            return
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(x_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap.summary_plot(shap_values, x_sample, show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "v4_generated_shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close()
        mean_abs = np.abs(shap_values).mean(axis=0)
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False
        ).to_csv(REPORT_DIR / "v4_generated_shap_feature_importance.csv", index=False, encoding="utf-8-sig")
    except Exception as exc:
        (REPORT_DIR / "v4_generated_shap_error.txt").write_text(str(exc), encoding="utf-8")


def save_model_plots(results: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results))
    width = 0.22
    ax1.bar(x - width, results["accuracy"], width, label="Accuracy", color="#2f80ed")
    ax1.bar(x, results["recall"], width, label="Recall", color="#eb5757")
    ax1.bar(x + width, results["f1"], width, label="F1", color="#27ae60")
    ax1.set_xticks(x)
    ax1.set_xticklabels(results["model"], rotation=18, ha="right")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Score")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, results["cost_thb"], marker="o", color="#7b61ff", label="Cost THB")
    ax2.set_ylabel("Cost THB")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper center", ncols=4)
    ax1.set_title("V4 Generated Model Metrics")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "v4_generated_model_metrics.png", dpi=160)
    plt.close(fig)


def write_reports(generation_summary: dict, cleaning_audit: list[dict], eda_insight: dict, results: pd.DataFrame) -> None:
    best = results.iloc[0]
    md = f"""# V4 Generated End-to-End Report

## 1.1 Data Collection & Understanding

- SQL template: `docs/analysis/v4_generated_data_collection_sql.sql`
- Data dictionary: `docs/analysis/data_dictionary_v4_generated.csv`
- Generated raw data: `data/generated/v4_synthetic_orders_returns.csv`
- Clean dataset: `data/processed/clean_dataset_v4_generated.csv`
- Rows before generation: `{generation_summary["source_rows"]}`
- Synthetic non-return rows added: `{generation_summary["synthetic_rows_added"]}`
- Rows after generation: `{generation_summary["output_rows"]}`
- Return rate after generation: `{generation_summary["actual_return_rate"]:.4f}`

## 1.2 EDA

- EDA charts folder: `reports/eda_v4_generated`
- Top category return rates: `{eda_insight["top_category_return_rate"]}`
- Top channel return rates: `{eda_insight["top_channel_return_rate"]}`

## 1.3 Feature Engineering & Preprocessing

- Engineered feature CSV: `data/features/df_engineered_v4_generated.csv`
- Train/test/SMOTE artifact: `data/features/train_test_sets_v4_generated.pkl`
- SMOTE is applied only to the training split.
- Leakage fields such as return/refund/risk score and actual delivery outcome fields are excluded from model features.

## 1.4 Model Training & Evaluation

Best model: `{best["model"]}`

- Threshold: `{best["threshold"]:.2f}`
- Accuracy: `{best["accuracy"]:.4f}`
- Precision: `{best["precision"]:.4f}`
- Recall: `{best["recall"]:.4f}`
- F1: `{best["f1"]:.4f}`
- AUC: `{best["auc"]:.4f}`
- Cost: `{int(best["cost_thb"]):,}` THB

## Metrics

{df_to_markdown(results)}

## Cleaning Audit

{df_to_markdown(pd.DataFrame(cleaning_audit))}
"""
    (DOC_DIR / "v4_generated_end_to_end_report.md").write_text(md, encoding="utf-8")

    pdf_path = DOC_DIR / "v4_generated_end_to_end_report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        rightMargin=1.1 * cm,
        leftMargin=1.1 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
    )
    styles = getSampleStyleSheet()
    story: list = []
    story.append(Paragraph("V4 Generated End-to-End Report", styles["Heading1"]))
    story.append(Paragraph("Generated imbalance data, cleansing, EDA, feature engineering, SMOTE, model training, Optuna tuning, and SHAP analysis.", styles["BodyText"]))
    story.append(Spacer(1, 0.35 * cm))
    summary_rows = [
        ["Source rows", str(generation_summary["source_rows"])],
        ["Synthetic rows added", str(generation_summary["synthetic_rows_added"])],
        ["Final rows", str(generation_summary["output_rows"])],
        ["Return rate", f"{generation_summary['actual_return_rate']:.4f}"],
        ["Best model", str(best["model"])],
        ["Accuracy", f"{best['accuracy']:.4f}"],
        ["Recall", f"{best['recall']:.4f}"],
        ["F1", f"{best['f1']:.4f}"],
        ["AUC", f"{best['auc']:.4f}"],
        ["Cost THB", f"{int(best['cost_thb']):,}"],
    ]
    table = Table(summary_rows, colWidths=[5 * cm, 12 * cm])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7dee5")), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f7")), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold")]))
    story.append(table)
    story.append(Spacer(1, 0.4 * cm))
    if (REPORT_DIR / "v4_generated_model_metrics.png").exists():
        story.append(Image(str(REPORT_DIR / "v4_generated_model_metrics.png"), width=23 * cm, height=11 * cm, kind="proportional"))
    story.append(Spacer(1, 0.35 * cm))
    display = results[["model", "threshold", "accuracy", "precision", "recall", "f1", "auc", "cost_thb"]].copy()
    for col in ["threshold", "accuracy", "precision", "recall", "f1", "auc"]:
        display[col] = display[col].map(lambda v: f"{float(v):.4f}")
    display["cost_thb"] = display["cost_thb"].map(lambda v: f"{int(v):,}")
    table = Table([display.columns.tolist()] + display.astype(str).values.tolist(), repeatRows=1)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d7dee5")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story.append(table)
    doc.build(story)


def main() -> None:
    ensure_dirs()
    write_sql_template()
    source = load_source()
    generated, generation_summary = generate_synthetic_imbalance(source)
    clean, cleaning_audit = clean_generated(generated)
    write_data_dictionary(clean)
    eda_insight = run_eda(clean)
    x, y, feature_names = feature_engineering(clean)
    x_train, x_train_scaled, x_test, x_test_scaled, y_train, y_test, feature_names, _ = split_and_smote(x, y)
    results, best_model, best_name, _ = tune_models(x_train, x_train_scaled, x_test, x_test_scaled, y_train, y_test, feature_names)
    save_model_plots(results)
    run_shap(best_model, best_name, x_test_scaled if "Logistic" in best_name else x_test, feature_names)
    write_reports(generation_summary, cleaning_audit, eda_insight, results)

    (REPORT_DIR / "v4_generated_pipeline_summary.json").write_text(
        json.dumps(
            {
                "generation_summary": generation_summary,
                "cleaning_audit": cleaning_audit,
                "eda_insight": eda_insight,
                "best_model": results.iloc[0].to_dict(),
                "outputs": {
                    "raw": str(RAW_OUT),
                    "clean": str(CLEAN_OUT),
                    "features": str(FEATURE_OUT),
                    "train_test": str(TRAIN_TEST_OUT),
                    "model": str(MODEL_DIR / "best_model_v4_generated.pkl"),
                    "report_pdf": str(DOC_DIR / "v4_generated_end_to_end_report.pdf"),
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("[OK] V4 generated end-to-end pipeline complete.")
    print(results[["model", "threshold", "accuracy", "precision", "recall", "f1", "auc", "cost_thb"]].to_string(index=False))


if __name__ == "__main__":
    main()
