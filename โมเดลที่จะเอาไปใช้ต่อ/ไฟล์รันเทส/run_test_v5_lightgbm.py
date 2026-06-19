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

# BASE_DIR คือ path หลักของโฟลเดอร์ "โมเดลที่จะเอาไปใช้ต่อ"
# __file__ = path ของไฟล์ run_test_v5_lightgbm.py
# .resolve() = แปลง path ให้เป็น absolute path แบบเต็ม
# .parents[1] = ถอยขึ้นไป 2 ระดับ:
#   ระดับที่ 0 = โฟลเดอร์ "ไฟล์รันเทส"
#   ระดับที่ 1 = โฟลเดอร์ "โมเดลที่จะเอาไปใช้ต่อ"
BASE_DIR = Path(__file__).resolve().parents[1]

# path ของไฟล์โมเดล LightGBM V5 ที่ train เสร็จแล้ว
# ไฟล์ .pkl นี้คือโมเดลหลักที่ใช้ predict
DEFAULT_MODEL_PATH = BASE_DIR / "โมเดล" / "models" / "model_lgbm_s1_v5_lightgbm.pkl"

# path ของไฟล์ metadata ของโมเดล
# ใช้เก็บข้อมูลประกอบ เช่น threshold, feature count, model version, dataset ที่ใช้ train
DEFAULT_METADATA_PATH = BASE_DIR / "โมเดล" / "models" / "model_lgbm_s1_v5_metadata.json"

# path ของไฟล์รายชื่อ feature 64 ตัวที่โมเดล V5 ต้องใช้
# ใช้เช็คว่า input data มี column ครบตรงกับตอน train หรือไม่
DEFAULT_FEATURE_LIST_PATH = BASE_DIR / "โมเดล" / "features" / "used_features_lgbm_s1_v5.csv"

# path ของไฟล์ feature dataset ที่ผ่าน Feature Engineering แล้ว
# ใช้สำหรับ mode model เพื่อส่งข้อมูลเข้าโมเดล predict ใหม่
DEFAULT_FEATURED_DATA_PATH = BASE_DIR / "โมเดล" / "features" / "df_featured_lgbm_s1_v5.csv"

# path ของไฟล์ผล prediction เดิมจาก real_dataset_s1.csv
# ใช้สำหรับ mode saved-predictions เพื่อคำนวณ metric ซ้ำให้ตรงกับรายงานเดิม
DEFAULT_SAVED_EXTERNAL_PREDICTIONS = (
    BASE_DIR
    / "รายงานการวัดผล"
    / "รายงานผล test"
    / "external_predictions_lgbm_s1_v5.csv"
)

# path ของโฟลเดอร์สำหรับเก็บผลลัพธ์ตอนรัน test
# เช่น metrics CSV หรือ prediction output ที่ script สร้างใหม่
DEFAULT_OUTPUT_DIR = BASE_DIR / "ไฟล์รันเทส" / "outputs"


def resolve_existing_path(path_value: str | Path, filename: str | None = None) -> Path:
    """Return an existing path, or find the file under the handoff folder."""
    # ฟังก์ชันนี้ช่วยแก้ปัญหา path เปลี่ยนหลังย้ายโฟลเดอร์
    # ถ้า path ที่ส่งมาเจอไฟล์จริง จะใช้ path นั้นทันที
    # ถ้าไม่เจอ จะค้นหาไฟล์จากชื่อไฟล์ภายใน BASE_DIR ให้อัตโนมัติ
    path = Path(path_value)
    if path.exists():
        return path

    target_name = filename or path.name
    # rglob คือค้นหาไฟล์แบบ recursive ในโฟลเดอร์ "โมเดลที่จะเอาไปใช้ต่อ"
    matches = sorted(BASE_DIR.rglob(target_name))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"File not found: {path} (also searched for {target_name} under {BASE_DIR})")


def load_threshold(metadata_path: Path, fallback: float = 0.5) -> float:
    # อ่าน threshold จาก metadata ของโมเดล
    # threshold คือจุดตัด probability เช่น 0.67:
    # ถ้า probability >= 0.67 จะทำนายว่าเสี่ยงคืนสินค้า
    # ถ้าไม่มีไฟล์ metadata จะใช้ค่า fallback แทน
    if not metadata_path.exists():
        return fallback
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return float(metadata.get("threshold", fallback))


def load_feature_names(feature_list_path: Path) -> list[str]:
    # อ่านรายชื่อ feature ที่โมเดล V5 ต้องการใช้
    # ไฟล์นี้ต้องมี column ชื่อ "feature"
    # ใช้เพื่อบังคับให้ input data มี column ตรงกับตอน train
    feature_df = pd.read_csv(feature_list_path)
    if "feature" not in feature_df.columns:
        raise ValueError(f"Feature file must contain a 'feature' column: {feature_list_path}")
    return feature_df["feature"].astype(str).tolist()


def binary_metrics(y_true, y_pred, y_prob=None, fp_cost: float = 50.0, fn_cost: float = 500.0) -> dict:
    # คำนวณ metric สำหรับ binary classification:
    # 0 = ไม่คืนสินค้า, 1 = คืนสินค้า
    # reset_index ช่วยป้องกัน index ของ pandas ไม่ตรงกันหลังแบ่ง/โหลดข้อมูล
    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(int).reset_index(drop=True)
    if y_prob is not None:
        y_prob = pd.Series(y_prob).reset_index(drop=True)

    # Confusion matrix:
    # TP = ทายคืน และคืนจริง
    # TN = ทายไม่คืน และไม่คืนจริง
    # FP = ทายคืน แต่จริงๆ ไม่คืน
    # FN = ทายไม่คืน แต่จริงๆ คืน
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # AUC ใช้ probability ไม่ใช่ label
    # ถ้าเครื่องที่รันไม่มี sklearn จะข้าม AUC แต่ metric อื่นยังใช้ได้
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
        # Cost matrix แบบง่าย:
        # FP = แจ้งเตือนเกินจริง ค่าเสียหายน้อยกว่า
        # FN = พลาดเคสคืนสินค้า ค่าเสียหายสูงกว่า
        "cost": fp * fp_cost + fn * fn_cost,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
    }


def print_metrics(metrics: dict, title: str) -> None:
    # แสดงผล metric ใน console ให้อ่านง่าย
    # ใช้ทั้งตอน test จากโมเดลจริง และตอนอ่าน saved predictions
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
    # mode_model ใช้สำหรับ "predict ใหม่จากโมเดล .pkl"
    # เหมาะกับกรณีที่มีข้อมูลบริษัทจริงซึ่งผ่าน Feature Engineering แล้ว
    # input ต้องเป็น feature dataset ที่มี feature ครบ 64 ตัวตาม V5
    try:
        # joblib ใช้โหลดไฟล์โมเดล .pkl ที่ train เก็บไว้
        import joblib
    except ModuleNotFoundError:
        print("ERROR: ไม่พบ package 'joblib'")
        print("ให้ติดตั้ง package ก่อนด้วยคำสั่ง:")
        print("pip install -r requirements.txt")
        print("หรือถ้าอยู่ในโฟลเดอร์นี้:")
        print("pip install -r \"..\\ทรัพยากรในเครื่องติดตั้งก่อนทำ\\requirements.txt\"")
        sys.exit(1)

    # หาไฟล์สำคัญทั้งหมด:
    # - model_path = ไฟล์โมเดล LightGBM
    # - metadata_path = threshold/ข้อมูลประกอบของโมเดล
    # - feature_list_path = รายชื่อ feature ที่โมเดลต้องใช้
    # - input_path = dataset ที่ผ่าน Feature Engineering แล้ว
    model_path = resolve_existing_path(args.model_path, "model_lgbm_s1_v5_lightgbm.pkl")
    metadata_path = resolve_existing_path(args.metadata_path, "model_lgbm_s1_v5_metadata.json")
    feature_list_path = resolve_existing_path(args.feature_list_path, "used_features_lgbm_s1_v5.csv")
    input_path = resolve_existing_path(args.input_csv, "df_featured_lgbm_s1_v5.csv")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # โหลด feature list และ threshold
    # ถ้าผู้ใช้ส่ง --threshold มา จะใช้ค่านั้นแทน metadata
    features = load_feature_names(feature_list_path)
    threshold = float(args.threshold) if args.threshold is not None else load_threshold(metadata_path, 0.5)

    # อ่าน input CSV ที่ต้องเป็น feature-ready dataset
    # ไม่ควรส่ง raw/clean dataset ตรงๆ เข้ามา เพราะยังไม่มี feature 64 ตัว
    df = pd.read_csv(input_path)
    missing = [col for col in features if col not in df.columns]
    if missing:
        # ถ้า feature ไม่ครบ จะหยุดทันที
        # เพื่อป้องกันการ predict ด้วย schema ที่ไม่ตรงกับตอน train
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

    # เตรียม X เฉพาะ feature ที่โมเดลต้องใช้
    # column ที่เป็นข้อความจะเปลี่ยนเป็น category เพื่อให้ LightGBM ใช้งานได้ถูกต้อง
    X = df[features].copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    # โหลดโมเดลแล้วคำนวณ probability ของ class 1 = คืนสินค้า
    model = joblib.load(model_path)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.predict(X)

    # แปลง probability เป็น label ด้วย threshold
    # เช่น probability >= 0.67 จะเป็น predicted_is_returned = 1
    y_pred = (pd.Series(y_prob) >= threshold).astype(int)

    # เก็บ id สำคัญไว้ใน output เพื่อ trace กลับไปยัง order/customer ได้
    id_cols = [col for col in ["order_id", "customer_id", "order_date"] if col in df.columns]
    result = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)
    result["predict_probability_return"] = y_prob
    result["predicted_is_returned"] = y_pred
    result["threshold"] = threshold

    metrics = None
    if args.target_col in df.columns:
        # ถ้า input มีคำตอบจริง is_returned จะคำนวณ Accuracy/Recall/Precision/F1/Cost ได้
        # ใช้กรณี test model กับข้อมูลที่รู้ผลคืนสินค้าแล้ว
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
        # ถ้าเป็น order ใหม่ที่ยังไม่รู้ผลคืนสินค้า จะ predict ได้อย่างเดียว
        # แต่ยังวัด Accuracy ไม่ได้จนกว่าจะรู้ ground truth ภายหลัง
        print("ไม่พบ target column จึงสร้าง prediction อย่างเดียว:", args.target_col)

    # บันทึกผล prediction ทุก row ออกเป็น CSV
    pred_path = output_dir / "v5_model_test_predictions.csv"
    result.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved predictions: {pred_path}")

    if metrics is not None:
        # ถ้ามี ground truth จะบันทึก metric summary เพิ่มอีกไฟล์
        metrics_path = output_dir / "v5_model_test_metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
        print(f"Saved metrics    : {metrics_path}")


def mode_saved_predictions(args: argparse.Namespace) -> None:
    # mode_saved_predictions ใช้สำหรับ "อ่านผล predict เดิม" แล้วคำนวณ metric ซ้ำ
    # mode นี้ไม่โหลดโมเดล ไม่ predict ใหม่ และไม่ train ใหม่
    # ใช้ตรวจค่าให้ตรงรายงานเดิม เช่น Real Accuracy 81.91%
    predictions_path = resolve_existing_path(args.predictions_csv, "external_predictions_lgbm_s1_v5.csv")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ไฟล์ saved predictions ต้องมีคำตอบจริง, label ที่โมเดลทาย, และ probability
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

    # บันทึก metric ที่คำนวณจาก saved predictions
    metrics_path = output_dir / "v5_saved_external_prediction_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved metrics: {metrics_path}")


def parse_args() -> argparse.Namespace:
    # parse_args คือส่วนรับ parameter จาก command line
    # ทำให้ผู้ใช้เปลี่ยนไฟล์ model/input/output ได้โดยไม่ต้องแก้โค้ด
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
    # --model-path ใช้เมื่อต้องการโหลดโมเดล .pkl ไฟล์อื่น
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))

    # --metadata-path ใช้โหลด threshold และข้อมูลประกอบของโมเดล
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))

    # --feature-list-path คือไฟล์รายชื่อ feature ที่ต้องมีใน input
    parser.add_argument("--feature-list-path", default=str(DEFAULT_FEATURE_LIST_PATH))

    # --input-csv ใช้กับ --mode model
    # ต้องเป็นไฟล์ที่ผ่าน Feature Engineering แล้ว ไม่ใช่ raw/clean dataset ธรรมดา
    parser.add_argument("--input-csv", default=str(DEFAULT_FEATURED_DATA_PATH))

    # --predictions-csv ใช้กับ --mode saved-predictions
    # เป็นไฟล์ prediction เดิมที่มี actual/predicted/probability อยู่แล้ว
    parser.add_argument("--predictions-csv", default=str(DEFAULT_SAVED_EXTERNAL_PREDICTIONS))

    # --output-dir คือโฟลเดอร์ปลายทางสำหรับเก็บ CSV ผลลัพธ์ที่ script สร้าง
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    # --target-col คือชื่อ column คำตอบจริง
    # ถ้า input ไม่มี column นี้ script จะ predict อย่างเดียวและไม่คำนวณ Accuracy
    parser.add_argument("--target-col", default="is_returned")

    # --threshold ใช้ override threshold จาก metadata
    # ถ้าไม่ใส่ จะใช้ threshold เดิมของโมเดล
    parser.add_argument("--threshold", type=float, default=None)

    # --fp-cost และ --fn-cost ใช้ปรับ cost matrix ตามมุมมองธุรกิจ
    # FP = แจ้งว่าเสี่ยงคืนแต่จริงๆ ไม่คืน
    # FN = แจ้งว่าไม่เสี่ยงแต่จริงๆ คืน
    parser.add_argument("--fp-cost", type=float, default=50.0)
    parser.add_argument("--fn-cost", type=float, default=500.0)
    return parser.parse_args()


def main() -> None:
    # main คือจุดเริ่มต้นของ script
    # ถ้า mode = model จะโหลดโมเดลแล้ว predict ใหม่
    # ถ้า mode = saved-predictions จะอ่านผล predict เดิมแล้วสรุป metric
    args = parse_args()
    if args.mode == "model":
        mode_model(args)
    else:
        mode_saved_predictions(args)


if __name__ == "__main__":
    main()
