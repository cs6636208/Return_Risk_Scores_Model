"""
Verify reference metrics for selected LightGBM V5.

This script does NOT train a new model and does NOT run new predictions.
It reads saved experiment artifacts and prints the exact reference numbers
used in the comparison table:

- Features: 64
- Clean Accuracy: 82.00%
- Real Accuracy: 81.91%
- Gap: -0.09 pp

Use this file when you need to prove that the handoff package matches
the V5 row in the report image.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
HANDOFF_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = SCRIPT_DIR / "outputs"


def find_file(candidates: list[Path], pattern: str) -> Path:
    for base in candidates:
        if not base.exists():
            continue
        matches = sorted(base.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot find required file pattern: {pattern}")


def load_clean_metrics(path: Path) -> dict:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Clean metrics file is empty: {path}")

    row = df.iloc[0].to_dict()
    clean_accuracy = float(row.get("holdout_accuracy"))
    holdout_recall = float(row.get("holdout_recall"))
    holdout_f1 = float(row.get("holdout_f1"))
    holdout_auc = float(row.get("holdout_auc"))
    threshold = float(row.get("threshold"))
    feature_count = int(row.get("feature_count"))
    version = str(row.get("version", "V5"))

    return {
        "version": version,
        "clean_accuracy": clean_accuracy,
        "clean_recall": holdout_recall,
        "clean_f1": holdout_f1,
        "clean_auc": holdout_auc,
        "threshold": threshold,
        "feature_count_from_metrics": feature_count,
        "clean_metrics_path": str(path),
    }


def compute_external_metrics(path: Path) -> dict:
    df = pd.read_csv(path)
    required = ["actual_is_returned", "predicted_is_returned", "predict_probability_return"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"External predictions file missing columns: {missing}")

    y_true = df["actual_is_returned"].astype(int)
    y_pred = df["predicted_is_returned"].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    rows = int(len(df))

    accuracy = (tp + tn) / rows if rows else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    auc = None
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_true, df["predict_probability_return"]))
    except Exception:
        auc = None

    return {
        "real_rows": rows,
        "real_accuracy": accuracy,
        "real_recall": recall,
        "real_precision": precision,
        "real_f1": f1,
        "real_auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "external_predictions_path": str(path),
    }


def load_feature_count(path: Path) -> int:
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError(f"Feature list must contain column 'feature': {path}")
    return int(len(df))


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    search_bases = [
        HANDOFF_DIR,
        PROJECT_ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S1" / "V5",
        PROJECT_ROOT / "docs" / "LightGBM" / "SETD" / "real_dataset" / "S3" / "V5",
        PROJECT_ROOT / "docs" / "LightGBM",
    ]

    clean_metrics_path = find_file(search_bases, "metrics_lgbm_s1_v5.csv")
    external_predictions_path = find_file(search_bases, "external_predictions_lgbm_s1_v5.csv")
    feature_list_path = find_file(search_bases, "used_features_lgbm_s1_v5.csv")

    clean = load_clean_metrics(clean_metrics_path)
    real = compute_external_metrics(external_predictions_path)
    feature_count = load_feature_count(feature_list_path)

    clean_accuracy = clean["clean_accuracy"]
    real_accuracy = real["real_accuracy"]
    gap_pp = (real_accuracy - clean_accuracy) * 100

    verification = {
        "version": clean["version"],
        "features": feature_count,
        "clean_accuracy": clean_accuracy,
        "real_accuracy": real_accuracy,
        "gap_pp": gap_pp,
        "threshold": clean["threshold"],
        "clean_recall": clean["clean_recall"],
        "clean_f1": clean["clean_f1"],
        "clean_auc": clean["clean_auc"],
        "real_recall": real["real_recall"],
        "real_precision": real["real_precision"],
        "real_f1": real["real_f1"],
        "real_auc": real["real_auc"],
        "real_rows": real["real_rows"],
        "tn": real["tn"],
        "fp": real["fp"],
        "fn": real["fn"],
        "tp": real["tp"],
        "clean_metrics_path": clean["clean_metrics_path"],
        "external_predictions_path": real["external_predictions_path"],
        "feature_list_path": str(feature_list_path),
        "source_of_truth": "saved experiment artifacts, no retrain, no new predict",
    }

    csv_path = OUTPUT_DIR / "v5_reference_metrics_verification.csv"
    json_path = OUTPUT_DIR / "v5_reference_metrics_verification.json"

    pd.DataFrame([verification]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("LightGBM V5 Reference Metrics Verification")
    print("=" * 72)
    print(f"Version: {verification['version']}")
    print(f"Features: {verification['features']}")
    print(f"Clean Accuracy: {percent(clean_accuracy)}")
    print(f"Real Accuracy: {percent(real_accuracy)}")
    print(f"Gap: {gap_pp:+.2f} pp")
    print(f"Threshold: {verification['threshold']:.2f}")
    print(f"Real rows: {verification['real_rows']:,}")
    print(f"TN/FP/FN/TP: {real['tn']:,} / {real['fp']:,} / {real['fn']:,} / {real['tp']:,}")
    print()
    print("Saved outputs:")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print()
    print("Note: This verification reads saved artifacts only. It does not train or predict again.")


if __name__ == "__main__":
    main()
