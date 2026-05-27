from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.production_v2.preprocessing import PRODUCTION_DROP_COLUMNS, V2ProdPreprocessor  # noqa: E402


RANDOM_STATE = 42
COST_FN = 150
COST_FP = 50

DATA_PATH = ROOT / "data" / "processed" / "clean_dataset.csv"
FEATURE_OUT = ROOT / "data" / "features" / "train_test_sets_v2_prod.pkl"
SCHEMA_OUT = ROOT / "data" / "features" / "v2_prod_feature_schema.json"
MODEL_OUT = ROOT / "models" / "best_model_v2_prod.pkl"
ARTIFACT_OUT = ROOT / "models" / "v2_prod_artifact.pkl"
METADATA_OUT = ROOT / "models" / "best_model_v2_prod_metadata.json"
REPORT_DIR = ROOT / "reports" / "model_evaluation_v2_prod"
DOCS_DIR = ROOT / "docs" / "version 2" / "production"


def cost(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int((fn * COST_FN) + (fp * COST_FP)), int(tn), int(fp), int(fn), int(tp)


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, int]:
    best_threshold = 0.5
    best_cost = None
    for threshold in np.linspace(0.01, 0.99, 99):
        y_pred = (y_proba >= threshold).astype(int)
        total_cost, *_ = cost(y_true, y_pred)
        if best_cost is None or total_cost < best_cost:
            best_threshold = float(threshold)
            best_cost = total_cost
    return best_threshold, int(best_cost)


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float, scenario: str) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    total_cost, tn, fp, fn, tp = cost(y_true, y_pred)
    return {
        "scenario": scenario,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
        "cost": total_cost,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main() -> None:
    for path in [FEATURE_OUT.parent, MODEL_OUT.parent, REPORT_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    y = df["is_returned"].astype(int)
    x = df.drop(columns=["is_returned"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)

    preprocessor = V2ProdPreprocessor()
    x_train_scaled = preprocessor.fit_transform(x_train, y_train)
    x_test_scaled = preprocessor.transform(x_test)

    count_0 = int((y_train == 0).sum())
    count_1 = int((y_train == 1).sum())
    scale_pos_weight = count_0 / count_1

    model = XGBClassifier(
        n_estimators=360,
        max_depth=6,
        learning_rate=0.038,
        subsample=0.85,
        colsample_bytree=0.88,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(x_train_scaled, y_train)

    y_proba = model.predict_proba(x_test_scaled)[:, 1]
    best_threshold, _ = find_best_threshold(y_test.to_numpy(), y_proba)
    metrics = pd.DataFrame(
        [
            evaluate(y_test.to_numpy(), y_proba, 0.50, "default_threshold_0_50"),
            evaluate(y_test.to_numpy(), y_proba, best_threshold, "optimal_cost_threshold"),
        ]
    )

    removed_post_event = sorted(c for c in PRODUCTION_DROP_COLUMNS if c in df.columns)
    metadata = {
        "model_name": "V2 Production XGBoost",
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": len(preprocessor.feature_columns),
        "feature_names": preprocessor.feature_columns,
        "threshold": best_threshold,
        "scale_pos_weight": scale_pos_weight,
        "removed_post_event_or_leakage_columns": removed_post_event,
        "metrics": metrics.to_dict(orient="records"),
    }

    joblib.dump(model, MODEL_OUT)
    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "threshold": best_threshold,
            "metadata": metadata,
        },
        ARTIFACT_OUT,
    )
    joblib.dump(
        {
            "X_train": x_train_scaled,
            "X_test": x_test_scaled,
            "y_train": y_train.to_numpy(),
            "y_test": y_test.to_numpy(),
            "feature_names": preprocessor.feature_columns,
            "raw_train_index": x_train.index.to_numpy(),
            "raw_test_index": x_test.index.to_numpy(),
        },
        FEATURE_OUT,
    )
    SCHEMA_OUT.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    METADATA_OUT.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics.to_csv(REPORT_DIR / "v2_prod_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(DOCS_DIR / "v2_prod_metrics.csv", index=False, encoding="utf-8-sig")
    (DOCS_DIR / "v2_prod_feature_schema.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(MODEL_OUT)
    print(ARTIFACT_OUT)
    print(FEATURE_OUT)
    print(REPORT_DIR / "v2_prod_metrics.csv")


if __name__ == "__main__":
    main()
