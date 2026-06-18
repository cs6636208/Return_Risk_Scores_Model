"""
Run test for selected LightGBM V5 model.

ใช้ทำอะไร:
1. ทดสอบโมเดล V5 กับไฟล์ที่ผ่าน Feature Engineering แล้ว
2. อ่านผล prediction เดิมของ real_dataset_s1.csv แล้วสรุป metric ให้ดูซ้ำได้

หมายเหตุสำคัญ:
- real_dataset_s1.csv เป็น clean/raw dataset ยังไม่ใช่ feature set พร้อมเข้าโมเดล
- ถ้าจะ predict real_dataset_s1.csv ใหม่ ต้องทำ Feature Engineering ให้ได้ feature 64 ตัวก่อน
- ไฟล์ที่ predict เข้าโมเดลได้ทันทีคือ df_featured_lgbm_s1_v5.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = BASE_DIR / "โมเดล" / "models" / "model_lgbm_s1_v5_lightgbm.pkl"
DEFAULT_METADATA_PATH = BASE_DIR / "โมเดล" / "models" / "model_lgbm_s1_v5_metadata.json"
DEFAULT_FEATURE_LIST_PATH = BASE_DIR / "โมเดล" / "features" / "used_features_lgbm_s1_v5.csv"
DEFAULT_FEATURED_DATA_PATH = BASE_DIR / "โมเดล" / "features" / "df_featured_lgbm_s1_v5.csv"
DEFAULT_SAVED_EXTERNAL_PREDICTIONS = (
    BASE_DIR
    / "รายงานการวัดผล"
    / "external_test_reports"
    / "external_predictions_lgbm_s1_v5.csv"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "ไฟล์รันเทส" / "outputs"


def load_threshold(metadata_path: Path, fallback: float = 0.5) -> float:
    if not metadata_path.exists():
        return fallback
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return float(metadata.get("threshold", fallback))


def load_feature_names(feature_list_path: Path) -> list[str]:
    feature_df = pd.read_csv(feature_list_path)
    if "feature" not in feature_df.columns:
        raise ValueError(f"Feature file must contain a 'feature' column: {feature_list_path}")
    return feature_df["feature"].astype(str).tolist()


def binary_metrics(y_true, y_pred, y_prob=None, fp_cost: float = 50.0, fn_cost: float = 500.0) -> dict:
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    auc = None
    if y_prob is not None:
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None

    return {
        "rows": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cost": fp * fp_cost + fn * fn_cost,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
    }


def print_metrics(metrics: dict, title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Rows      : {metrics['rows']:,}")
    print(f"Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"Recall    : {metrics['recall'] * 100:.2f}%")
    print(f"Precision : {metrics['precision'] * 100:.2f}%")
    print(f"F1        : {metrics['f1'] * 100:.2f}%")
    if metrics.get("auc") is not None:
        print(f"AUC       : {metrics['auc'] * 100:.2f}%")
    else:
        print("AUC       : skipped")
    print(f"TN/FP/FN/TP: {metrics['tn']:,} / {metrics['fp']:,} / {metrics['fn']:,} / {metrics['tp']:,}")
    print(f"Cost      : {metrics['cost']:,.0f}")


def mode_model(args: argparse.Namespace) -> None:
    try:
        import joblib
    except ModuleNotFoundError:
        print("ERROR: ไม่พบ package 'joblib'")
        print("ให้ติดตั้ง package ก่อนด้วยคำสั่ง:")
        print("pip install -r requirements.txt")
        print("หรือถ้าอยู่ในโฟลเดอร์นี้:")
        print("pip install -r \"..\\ทรัพยากรในเครื่องติดตั้งก่อนทำ\\requirements.txt\"")
        sys.exit(1)

    model_path = Path(args.model_path)
    metadata_path = Path(args.metadata_path)
    feature_list_path = Path(args.feature_list_path)
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not feature_list_path.exists():
        raise FileNotFoundError(f"Feature list not found: {feature_list_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    features = load_feature_names(feature_list_path)
    threshold = float(args.threshold) if args.threshold is not None else load_threshold(metadata_path, 0.5)

    df = pd.read_csv(input_path)
    missing = [col for col in features if col not in df.columns]
    if missing:
        print("\nERROR: ไฟล์ input ยังไม่มี feature ครบตามที่ V5 ต้องใช้")
        print(f"Input file: {input_path}")
        print(f"Missing feature count: {len(missing)}")
        print("ตัวอย่าง feature ที่ขาด:")
        for col in missing[:30]:
            print(f"- {col}")
        print("\nสาเหตุที่พบบ่อย:")
        print("- ใช้ real_dataset_s1.csv โดยตรง ซึ่งยังเป็น clean/raw dataset")
        print("- ยังไม่ได้ทำ Feature Engineering ให้กลายเป็น feature 64 ตัวของ V5")
        print("\nวิธีแก้:")
        print("- ใช้ df_featured_lgbm_s1_v5.csv สำหรับทดสอบโมเดลทันที")
        print("- หรือเขียน/รัน Feature Builder ก่อน แล้วค่อยนำไฟล์ผลลัพธ์มาเข้า script นี้")
        sys.exit(2)

    X = df[features].copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    model = joblib.load(model_path)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.predict(X)

    y_pred = (pd.Series(y_prob) >= threshold).astype(int)

    id_cols = [col for col in ["order_id", "customer_id", "order_date"] if col in df.columns]
    result = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
    result["predict_probability_return"] = y_prob
    result["predicted_is_returned"] = y_pred
    result["threshold"] = threshold

    metrics = None
    if args.target_col in df.columns:
        result["actual_is_returned"] = df[args.target_col].astype(int)
        result["correct_prediction"] = (result["actual_is_returned"] == result["predicted_is_returned"]).astype(int)
        metrics = binary_metrics(
            result["actual_is_returned"],
            result["predicted_is_returned"],
            result["predict_probability_return"],
            fp_cost=args.fp_cost,
            fn_cost=args.fn_cost,
        )
        print_metrics(metrics, "LightGBM V5 Model Test Result")
    else:
        print("ไม่พบ target column จึงสร้าง prediction อย่างเดียว:", args.target_col)

    pred_path = output_dir / "v5_model_test_predictions.csv"
    result.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved predictions: {pred_path}")

    if metrics is not None:
        metrics_path = output_dir / "v5_model_test_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
        print(f"Saved metrics    : {metrics_path}")


def mode_saved_predictions(args: argparse.Namespace) -> None:
    predictions_path = Path(args.predictions_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}")

    df = pd.read_csv(predictions_path)
    required = ["actual_is_returned", "predicted_is_returned", "predict_probability_return"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in saved predictions: {missing}")

    metrics = binary_metrics(
        df["actual_is_returned"],
        df["predicted_is_returned"],
        df["predict_probability_return"],
        fp_cost=args.fp_cost,
        fn_cost=args.fn_cost,
    )
    print_metrics(metrics, "LightGBM V5 Saved External Prediction Result")

    metrics_path = output_dir / "v5_saved_external_prediction_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved metrics: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test for selected LightGBM V5 model.")
    parser.add_argument(
        "--mode",
        choices=["model", "saved-predictions"],
        default="saved-predictions",
        help=(
            "model = โหลดโมเดล .pkl แล้ว predict จาก featured CSV, "
            "saved-predictions = อ่านไฟล์ prediction เดิมของ external test แล้วสรุป metric"
        ),
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--feature-list-path", default=str(DEFAULT_FEATURE_LIST_PATH))
    parser.add_argument("--input-csv", default=str(DEFAULT_FEATURED_DATA_PATH))
    parser.add_argument("--predictions-csv", default=str(DEFAULT_SAVED_EXTERNAL_PREDICTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-col", default="is_returned")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--fp-cost", type=float, default=50.0)
    parser.add_argument("--fn-cost", type=float, default=500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "model":
        mode_model(args)
    else:
        mode_saved_predictions(args)


if __name__ == "__main__":
    main()
