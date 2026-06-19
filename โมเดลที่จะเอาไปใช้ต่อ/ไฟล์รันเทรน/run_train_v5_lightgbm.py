from __future__ import annotations

"""
Train script for selected LightGBM V5 model.

ใช้ทำอะไร:
1. โหลด feature dataset ที่ผ่าน Feature Engineering แล้ว
2. โหลดรายชื่อ feature 64 ตัวของ V5
3. แบ่งข้อมูลเป็น fit / validation / holdout
4. train โมเดล LightGBM
5. เลือก threshold จาก validation set
6. วัดผลสุดท้ายบน holdout test set
7. บันทึก model, metadata, metrics, holdout predictions และ feature importance

หมายเหตุสำคัญ:
- ไฟล์ input ต้องเป็น df_featured_lgbm_s1_v5.csv หรือไฟล์ที่มี feature ครบ 64 ตัวแล้ว
- ไม่ควรส่ง raw/clean dataset เข้า train ตรงๆ เพราะยังไม่ได้สร้าง feature ที่โมเดลต้องใช้
- ค่า Accuracy ที่ใช้รายงานหลัง train ควรดูจาก holdout test ไม่ใช่ validation
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# BASE_DIR คือ path หลักของโฟลเดอร์ "โมเดลที่จะเอาไปใช้ต่อ"
# __file__ = path ของไฟล์ run_train_v5_lightgbm.py
# .resolve() = แปลง path ให้เป็น absolute path แบบเต็ม
# .parents[1] = ถอยขึ้นไป 2 ระดับ:
#   ระดับที่ 0 = โฟลเดอร์ "ไฟล์รันเทรน"
#   ระดับที่ 1 = โฟลเดอร์ "โมเดลที่จะเอาไปใช้ต่อ"
BASE_DIR = Path(__file__).resolve().parents[1]

# path ของไฟล์ feature dataset ที่ผ่าน Feature Engineering แล้ว
# ไฟล์นี้เป็น input หลักสำหรับ train model
DEFAULT_FEATURED_DATA_PATH = BASE_DIR / "โมเดล" / "features" / "df_featured_lgbm_s1_v5.csv"

# path ของไฟล์รายชื่อ feature 64 ตัวที่ V5 ต้องใช้
# ใช้บังคับให้ train ด้วย feature ชุดเดียวกับที่ออกแบบไว้
DEFAULT_FEATURE_LIST_PATH = BASE_DIR / "โมเดล" / "features" / "used_features_lgbm_s1_v5.csv"

# path ของ metadata อ้างอิงจากโมเดลเดิม
# ใช้ดึง LightGBM hyperparameters เดิม เพื่อให้ retrain ด้วย parameter ใกล้เคียงของเดิม
DEFAULT_REFERENCE_METADATA_PATH = BASE_DIR / "โมเดล" / "models" / "model_lgbm_s1_v5_metadata.json"

# path ของโฟลเดอร์สำหรับเก็บ output หลัง train ใหม่
# เช่น model .pkl, metrics, metadata, holdout predictions, feature importance
DEFAULT_OUTPUT_DIR = BASE_DIR / "ไฟล์รันเทรน" / "outputs"


def load_feature_names(feature_list_path: Path) -> list[str]:
    # อ่านรายชื่อ feature ที่โมเดล V5 ต้องใช้จาก CSV
    # ไฟล์นี้ต้องมี column ชื่อ "feature"
    # ถ้าไม่มี column นี้ แปลว่าไฟล์ feature list ไม่ถูก format
    feature_df = pd.read_csv(feature_list_path)
    if "feature" not in feature_df.columns:
        raise ValueError(f"Feature file must contain a 'feature' column: {feature_list_path}")
    return feature_df["feature"].astype(str).tolist()


def load_reference_params(metadata_path: Path) -> dict:
    # โหลด hyperparameters จาก metadata ของโมเดลอ้างอิง
    # จุดประสงค์คือให้ train ใหม่โดยใช้ configuration เดิมของ V5
    # ถ้าไม่มี metadata จะ return dict ว่าง แล้ว script จะใช้ default params ด้านล่างแทน
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return dict(metadata.get("lightgbm_params", {}))


def binary_metrics(y_true, y_pred, y_prob=None, fp_cost: float = 50.0, fn_cost: float = 500.0) -> dict:
    # คำนวณ metric สำหรับปัญหา binary classification:
    # 0 = ไม่คืนสินค้า, 1 = คืนสินค้า
    # reset_index ช่วยป้องกัน index ของ pandas ไม่ตรงกันหลังแบ่ง train/test
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

    # AUC ใช้ probability ในการวัดความสามารถแยก class
    # ถ้าเครื่องไม่มี sklearn จะข้าม AUC แต่ยังคำนวณ metric อื่นได้
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
        # FP = แจ้งว่าเสี่ยงคืน แต่จริงๆ ไม่คืน
        # FN = แจ้งว่าไม่เสี่ยง แต่จริงๆ คืน
        # โดยกำหนดให้ FN แพงกว่า เพราะพลาดเคสคืนสินค้ามีผลเสียทางธุรกิจมากกว่า
        "cost": fp * fp_cost + fn * fn_cost,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
    }


def print_metrics(metrics: dict, title: str) -> None:
    # แสดงผล metric ใน console ให้อ่านง่าย
    # ใช้แสดงผล holdout test หลัง train เสร็จ
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


def find_best_threshold(y_true, y_prob, metric: str = "accuracy") -> tuple[float, dict]:
    # หา threshold ที่ดีที่สุดจาก validation set
    # threshold คือจุดตัด probability:
    # ถ้า probability >= threshold จะทำนายว่า "คืนสินค้า"
    # ถ้าไม่กำหนด --threshold script จะลองตั้งแต่ 0.10 ถึง 0.90 แล้วเลือกค่าที่ metric ดีสุด
    best_threshold = 0.5
    best_metrics = None
    best_score = -1.0

    for threshold in np.round(np.arange(0.10, 0.91, 0.01), 2):
        # แปลง probability เป็น label ด้วย threshold ปัจจุบัน
        y_pred = (pd.Series(y_prob) >= threshold).astype(int)
        metrics = binary_metrics(y_true, y_pred, y_prob)
        score = metrics.get(metric, metrics["accuracy"])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics or {}


def split_data(df: pd.DataFrame, target_col: str, random_state: int):
    # แบ่งข้อมูลออกเป็น 3 ส่วน:
    # 1. fit set       = ใช้ให้โมเดลเรียนรู้จริง
    # 2. validation set = ใช้เลือก threshold / tuning
    # 3. holdout set    = ใช้สอบปลายภาค วัดผลสุดท้ายหลัง train
    from sklearn.model_selection import train_test_split

    # 20% holdout เหมือนแนวทางเดิม
    train_val_df, holdout_df = train_test_split(
        df,
        test_size=0.20,
        random_state=random_state,
        # stratify ทำให้สัดส่วน returned / not returned ใกล้กันในแต่ละ split
        stratify=df[target_col],
    )

    # จาก 80% ที่เหลือ แบ่ง validation ออกมา 20% ของ train_val = 16% ของทั้งหมด
    fit_df, validation_df = train_test_split(
        train_val_df,
        test_size=0.20,
        random_state=random_state,
        # แบ่ง validation โดยรักษาสัดส่วน class เช่นกัน
        stratify=train_val_df[target_col],
    )
    return fit_df, validation_df, holdout_df


def main() -> None:
    # main คือ flow หลักของการ train:
    # parse args -> โหลดไฟล์ -> ตรวจ feature -> split data -> train -> evaluate -> save output
    parser = argparse.ArgumentParser(description="Train LightGBM V5 model from featured dataset.")

    # --input-csv คือ featured dataset ที่จะเอามา train
    # ต้องมี feature ครบ 64 ตัว และมี target column is_returned
    parser.add_argument("--input-csv", default=str(DEFAULT_FEATURED_DATA_PATH))

    # --feature-list-path คือรายชื่อ feature ที่บังคับใช้กับ V5
    parser.add_argument("--feature-list-path", default=str(DEFAULT_FEATURE_LIST_PATH))

    # --reference-metadata-path ใช้ดึง parameter เดิมของ V5
    parser.add_argument("--reference-metadata-path", default=str(DEFAULT_REFERENCE_METADATA_PATH))

    # --output-dir คือโฟลเดอร์เก็บผล train ใหม่
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    # --target-col คือ column คำตอบจริง
    parser.add_argument("--target-col", default="is_returned")

    # --threshold ถ้าใส่ จะใช้ค่าที่กำหนดเลย
    # ถ้าไม่ใส่ จะค้นหา threshold ที่ดีที่สุดจาก validation set
    parser.add_argument("--threshold", type=float, default=None, help="ถ้าไม่ใส่ จะ search threshold จาก validation")

    # --threshold-metric ใช้เลือกว่าจะ optimize threshold จาก metric ไหน
    parser.add_argument("--threshold-metric", default="accuracy", choices=["accuracy", "recall", "precision", "f1"])

    # --random-state ทำให้แบ่งข้อมูลและ train ซ้ำได้ใกล้เคียงเดิม
    parser.add_argument("--random-state", type=int, default=42)

    # --fp-cost และ --fn-cost ใช้กำหนด cost matrix ทางธุรกิจ
    parser.add_argument("--fp-cost", type=float, default=50.0)
    parser.add_argument("--fn-cost", type=float, default=500.0)
    args = parser.parse_args()

    try:
        # joblib ใช้ save/load model .pkl
        # LGBMClassifier คือโมเดล LightGBM สำหรับ classification
        import joblib
        from lightgbm import LGBMClassifier
    except ModuleNotFoundError as exc:
        print(f"ERROR: ไม่พบ package ที่จำเป็น: {exc.name}")
        print("ให้ติดตั้ง package ก่อนด้วยคำสั่ง:")
        print("pip install -r requirements.txt")
        print("หรือ:")
        print("pip install -r \"โมเดลที่จะเอาไปใช้ต่อ\\ทรัพยากรในเครื่องติดตั้งก่อนทำ\\requirements.txt\"")
        sys.exit(1)

    # แปลง path จาก argument ให้เป็น Path object
    input_path = Path(args.input_csv)
    feature_list_path = Path(args.feature_list_path)
    reference_metadata_path = Path(args.reference_metadata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # เช็คว่าไฟล์ input และ feature list มีอยู่จริงก่อนเริ่ม train
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not feature_list_path.exists():
        raise FileNotFoundError(f"Feature list not found: {feature_list_path}")

    # โหลด feature list และ featured dataset
    features = load_feature_names(feature_list_path)
    df = pd.read_csv(input_path)

    # target column จำเป็นสำหรับ train
    # ถ้าไม่มี is_returned จะ train supervised model ไม่ได้
    if args.target_col not in df.columns:
        raise ValueError(f"Target column not found: {args.target_col}")

    # ตรวจว่า dataset มี feature ครบตาม V5
    # ถ้าไม่ครบ แปลว่า input ยังไม่ได้ผ่าน Feature Engineering หรือ schema ไม่ตรง
    missing = [col for col in features if col not in df.columns]
    if missing:
        print("ERROR: input dataset ยังไม่มี feature ครบตาม V5")
        print(f"Missing feature count: {len(missing)}")
        for col in missing[:30]:
            print(f"- {col}")
        print("\nต้องใช้ไฟล์ที่ผ่าน Feature Engineering แล้ว เช่น df_featured_lgbm_s1_v5.csv")
        sys.exit(2)

    # แบ่งข้อมูลเป็น fit / validation / holdout
    fit_df, validation_df, holdout_df = split_data(df, args.target_col, args.random_state)

    # แยก X/y สำหรับแต่ละ split
    # X = feature ที่ใช้เรียนรู้
    # y = target is_returned
    X_fit = fit_df[features].copy()
    y_fit = fit_df[args.target_col].astype(int)
    X_val = validation_df[features].copy()
    y_val = validation_df[args.target_col].astype(int)
    X_holdout = holdout_df[features].copy()
    y_holdout = holdout_df[args.target_col].astype(int)

    # LightGBM รองรับ categorical feature ได้
    # column ที่เป็น object/text จะแปลงเป็น category
    categorical_cols = []
    for frame in [X_fit, X_val, X_holdout]:
        for col in frame.select_dtypes(include=["object"]).columns:
            frame[col] = frame[col].astype("category")
            if col not in categorical_cols:
                categorical_cols.append(col)

    # โหลด parameter เดิมจาก metadata ของ V5
    # ถ้าไม่มี metadata จะสร้าง default parameter ที่ออกแบบไว้ให้เหมาะกับ dataset นี้
    params = load_reference_params(reference_metadata_path)
    if not params:
        neg = int((y_fit == 0).sum())
        pos = int((y_fit == 1).sum())

        # scale_pos_weight ใช้ช่วยจัดการ class imbalance
        # ถ้า class คืนสินค้ามีน้อยกว่า จะให้น้ำหนัก class positive มากขึ้น
        scale_pos_weight = neg / pos if pos else 1.0
        params = {
            "n_estimators": 360,
            "learning_rate": 0.045,
            "num_leaves": 19,
            "max_depth": 4,
            "min_child_samples": 90,
            "subsample": 0.82,
            "colsample_bytree": 0.82,
            "reg_lambda": 8.0,
            "reg_alpha": 0.8,
            "objective": "binary",
            "n_jobs": -1,
            "random_state": args.random_state,
            "verbosity": -1,
            "scale_pos_weight": scale_pos_weight,
        }

    # ให้ random_state มาจาก argument ปัจจุบันเสมอ
    params["random_state"] = args.random_state

    # สร้างและ train โมเดล LightGBM ด้วย fit set เท่านั้น
    # validation/holdout จะไม่ถูกใช้ในการ fit เพื่อป้องกัน leakage
    model = LGBMClassifier(**params)
    model.fit(
        X_fit,
        y_fit,
        categorical_feature=categorical_cols if categorical_cols else "auto",
    )

    # Predict validation set เพื่อหา threshold ที่เหมาะสม
    val_prob = model.predict_proba(X_val)[:, 1]
    if args.threshold is None:
        # ถ้าไม่ได้ระบุ threshold จะเลือก threshold จาก validation ตาม metric ที่กำหนด
        threshold, val_metrics = find_best_threshold(y_val, val_prob, args.threshold_metric)
    else:
        # ถ้าระบุ threshold เอง จะใช้ค่านั้นทันที
        threshold = float(args.threshold)
        val_pred = (pd.Series(val_prob) >= threshold).astype(int)
        val_metrics = binary_metrics(y_val, val_pred, val_prob, args.fp_cost, args.fn_cost)

    # วัดผลสุดท้ายบน holdout set
    # ค่านี้คือผลที่ควรใช้รายงานหลัง train เพราะเป็นข้อมูลที่โมเดลไม่ใช้ตอน fit
    holdout_prob = model.predict_proba(X_holdout)[:, 1]
    holdout_pred = (pd.Series(holdout_prob) >= threshold).astype(int)
    holdout_metrics = binary_metrics(y_holdout, holdout_pred, holdout_prob, args.fp_cost, args.fn_cost)

    print(f"\nSelected threshold: {threshold:.2f}")
    print("Threshold was selected from validation set, but only Holdout Test is shown as the final train result.")
    print_metrics(holdout_metrics, "Holdout Test Result")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"trained_lgbm_v5_{timestamp}.pkl"
    metadata_path = output_dir / f"trained_lgbm_v5_{timestamp}_metadata.json"
    metrics_path = output_dir / f"trained_lgbm_v5_{timestamp}_metrics.csv"
    predictions_path = output_dir / f"trained_lgbm_v5_{timestamp}_holdout_predictions.csv"
    importance_path = output_dir / f"trained_lgbm_v5_{timestamp}_feature_importance.csv"

    joblib.dump(model, model_path)

    id_cols = [col for col in ["order_id", "customer_id", "order_date"] if col in holdout_df.columns]
    pred_df = holdout_df[id_cols].copy() if id_cols else pd.DataFrame(index=holdout_df.index)
    pred_df["actual_is_returned"] = y_holdout.values
    pred_df["predict_probability_return"] = holdout_prob
    pred_df["predicted_is_returned"] = holdout_pred.values
    pred_df["threshold"] = threshold
    pred_df["correct_prediction"] = (pred_df["actual_is_returned"] == pred_df["predicted_is_returned"]).astype(int)
    pred_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(
        [
            {"split": "validation", **val_metrics},
            {"split": "holdout", **holdout_metrics},
        ]
    )
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": features,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")

    metadata = {
        "model": "LightGBM",
        "version": "V5",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(input_path),
        "feature_list": str(feature_list_path),
        "target_col": args.target_col,
        "feature_count": len(features),
        "fit_rows": len(fit_df),
        "validation_rows": len(validation_df),
        "holdout_rows": len(holdout_df),
        "threshold": threshold,
        "threshold_metric": args.threshold_metric,
        "lightgbm_params": params,
        "categorical_features": categorical_cols,
        "model_path": str(model_path),
        "validation_metrics": val_metrics,
        "holdout_metrics": holdout_metrics,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved files:")
    print(f"- Model            : {model_path}")
    print(f"- Metadata         : {metadata_path}")
    print(f"- Metrics          : {metrics_path}")
    print(f"- Holdout predict  : {predictions_path}")
    if importance_path.exists():
        print(f"- Feature importance: {importance_path}")


if __name__ == "__main__":
    main()
