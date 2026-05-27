from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model_v2.pkl"
DATA_PATH = ROOT / "data" / "features" / "train_test_sets_v2.pkl"
REPORT_DIR = ROOT / "reports" / "model_evaluation_v2"
DOCS_REPORT_DIR = ROOT / "docs" / "version 2" / "reports" / "model_evaluation"

COST_FN = 150
COST_FP = 50


def cost_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int((fn * COST_FN) + (fp * COST_FP)), int(tn), int(fp), int(fn), int(tp)


def find_best_cost_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, int]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_cost = None
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        total_cost, *_ = cost_from_predictions(y_true, y_pred)
        if best_cost is None or total_cost < best_cost:
            best_threshold = float(threshold)
            best_cost = total_cost
    return best_threshold, int(best_cost)


def summarize(y_true: np.ndarray, y_proba: np.ndarray, threshold: float, scenario: str) -> dict[str, float | int | str]:
    y_pred = (y_proba >= threshold).astype(int)
    total_cost, tn, fp, fn, tp = cost_from_predictions(y_true, y_pred)
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


def result_label(y_true: np.ndarray, y_pred: np.ndarray) -> list[str]:
    labels = []
    for actual, pred in zip(y_true, y_pred):
        if actual == 1 and pred == 1:
            labels.append("TP_return_correct")
        elif actual == 0 and pred == 0:
            labels.append("TN_normal_correct")
        elif actual == 0 and pred == 1:
            labels.append("FP_false_alarm")
        else:
            labels.append("FN_missed_return")
    return labels


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = joblib.load(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    feature_names = list(data["feature_names"])
    X_test = data["X_test"]
    y_test = np.asarray(data["y_test"]).astype(int)

    if not isinstance(X_test, pd.DataFrame):
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
    else:
        X_test_df = X_test.copy()

    y_proba = model.predict_proba(X_test_df)[:, 1]
    best_threshold, _ = find_best_cost_threshold(y_test, y_proba)

    y_pred_050 = (y_proba >= 0.50).astype(int)
    y_pred_cost = (y_proba >= best_threshold).astype(int)

    prediction_df = pd.DataFrame(
        {
            "test_row_id": np.arange(len(y_test)),
            "y_true": y_test,
            "actual_label": np.where(y_test == 1, "return", "no_return"),
            "predict_probability_return": y_proba,
            "y_pred_threshold_0_50": y_pred_050,
            "pred_label_threshold_0_50": np.where(y_pred_050 == 1, "return", "no_return"),
            "result_threshold_0_50": result_label(y_test, y_pred_050),
            "optimal_cost_threshold": best_threshold,
            "y_pred_optimal_cost": y_pred_cost,
            "pred_label_optimal_cost": np.where(y_pred_cost == 1, "return", "no_return"),
            "result_optimal_cost": result_label(y_test, y_pred_cost),
        }
    )
    feature_df = X_test_df.add_prefix("feature_")
    prediction_df = pd.concat([prediction_df, feature_df], axis=1)

    summary_df = pd.DataFrame(
        [
            summarize(y_test, y_proba, 0.50, "default_threshold_0_50"),
            summarize(y_test, y_proba, best_threshold, "optimal_cost_threshold"),
        ]
    )

    prediction_path = REPORT_DIR / "v2_test_predictions.csv"
    summary_path = REPORT_DIR / "v2_test_prediction_summary.csv"
    docs_prediction_path = DOCS_REPORT_DIR / "v2_test_predictions.csv"
    docs_summary_path = DOCS_REPORT_DIR / "v2_test_prediction_summary.csv"

    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(docs_prediction_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(docs_summary_path, index=False, encoding="utf-8-sig")

    print(prediction_path)
    print(summary_path)
    print(docs_prediction_path)
    print(docs_summary_path)


if __name__ == "__main__":
    main()
