from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "test" / "same_xgboost_feature_version_benchmark"
TARGET = "is_returned"
RANDOM_STATE = 42

VERSION_DATASETS = {
    "V1": ROOT / "docs" / "version 1" / "data" / "features" / "df_featured.csv",
    "V2": ROOT / "docs" / "version 2" / "v2_xgboost_safe_plus_rolling" / "data" / "df_featured.csv",
    "V3": ROOT / "docs" / "version 3" / "data" / "features" / "df_featured.csv",
    "V4": ROOT / "docs" / "version 4" / "data" / "features" / "df_featured.csv",
}

DROP_ALWAYS = {
    TARGET,
    "dataset_split",
    "order_id",
    "customer_id",
    "customer_name",
    "customer_phone",
    "return_id",
    "return_date",
    "return_reason",
    "return_scenario",
    "item_condition",
    "return_status",
    "refund_amount",
    "risk_score",
    "risk_tier",
    "score_id",
    "scored_at",
    "shap_values",
}


def load_frame(path: Path, sample_size: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    return df


def build_pipeline(X: pd.DataFrame, scale_pos_weight: float) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=4,
        reg_lambda=2.0,
        reg_alpha=0.2,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def evaluate_version(version: str, path: Path, sample_size: int | None) -> dict:
    df = load_frame(path, sample_size)
    y = df[TARGET]
    feature_cols = [c for c in df.columns if c not in DROP_ALWAYS]
    X = df[feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    positive = max(int(y_train.sum()), 1)
    negative = max(int(len(y_train) - y_train.sum()), 1)
    scale_pos_weight = negative / positive

    pipeline = build_pipeline(X_train, scale_pos_weight)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.20, 0.81, 0.01)
    rows = []
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        fp = int(((pred == 1) & (y_test.to_numpy() == 0)).sum())
        fn = int(((pred == 0) & (y_test.to_numpy() == 1)).sum())
        cost = fp * 50 + fn * 200
        rows.append((threshold, cost, accuracy_score(y_test, pred), recall_score(y_test, pred, zero_division=0), f1_score(y_test, pred, zero_division=0)))
    best_threshold, best_cost, _, _, _ = sorted(rows, key=lambda x: (x[1], -x[3], -x[2]))[0]
    pred = (proba >= best_threshold).astype(int)
    tn = int(((pred == 0) & (y_test.to_numpy() == 0)).sum())
    fp = int(((pred == 1) & (y_test.to_numpy() == 0)).sum())
    fn = int(((pred == 0) & (y_test.to_numpy() == 1)).sum())
    tp = int(((pred == 1) & (y_test.to_numpy() == 1)).sum())
    return {
        "version": version,
        "source_path": str(path.relative_to(ROOT)),
        "sample_size": sample_size or len(df),
        "rows_used": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count_raw": len(feature_cols),
        "model": "XGBoost same-model benchmark",
        "threshold": float(best_threshold),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": roc_auc_score(y_test, proba),
        "avg_precision": average_precision_score(y_test, proba),
        "cost": best_cost,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for sample_size in [5000, 50000]:
        sample_results = []
        for version, path in VERSION_DATASETS.items():
            if not path.exists():
                sample_results.append({"version": version, "sample_size": sample_size, "status": "missing_dataset", "source_path": str(path.relative_to(ROOT))})
                continue
            result = evaluate_version(version, path, sample_size)
            result["status"] = "completed"
            sample_results.append(result)
            all_results.append(result)
        pd.DataFrame(sample_results).to_csv(OUT_DIR / f"same_xgboost_metrics_{sample_size}_rows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_results).to_csv(OUT_DIR / "same_xgboost_metrics_all.csv", index=False, encoding="utf-8-sig")
    print(f"Saved benchmark outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
