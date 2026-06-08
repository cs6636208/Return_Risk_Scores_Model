from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / ".ml_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


SOURCE = ROOT / "docs" / "XGBoost" / "SETA" / "clean_data" / "clean_dataset_s2.csv"
OUT_ROOT = ROOT / "docs" / "XGBoost" / "SETA" / "clean_data" / "S2"
RANDOM_STATE = 42
TARGET = "is_returned"
THRESHOLD_STRATEGY = "accuracy_balanced_recall_floor_70"
MIN_RECALL_FOR_THRESHOLD = 0.70

XGB_PARAMS = {
    "n_estimators": 460,
    "max_depth": 4,
    "learning_rate": 0.035,
    "min_child_weight": 4,
    "subsample": 0.90,
    "colsample_bytree": 0.88,
    "reg_lambda": 2.25,
    "reg_alpha": 0.12,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

VERSION_DESCRIPTIONS = {
    1: "V1 baseline: ใช้ raw/order-time features จาก clean_dataset_s2 เป็นจุดตั้งต้น",
    2: "V2 history: เอา V1 มาทำต่อ แล้วเพิ่ม customer history และ rolling history แบบ point-in-time",
    3: "V3 interaction: เอา V2 มาทำต่อ แล้วเพิ่ม business interaction และ group return-rate features",
    4: "V4 segment risk: เอา V3 มาทำต่อ แล้วเพิ่ม segment/operation risk features",
    5: "V5 compact: เอา V4 transformation มาทำต่อ แล้วเลือก compact feature set เพื่อลด feature ที่ซ้ำซ้อน/noise",
}


def load_seq_module() -> Any:
    module_path = ROOT / "scripts" / "run_dataset_5000_sequential_version_pipeline.py"
    spec = importlib.util.spec_from_file_location("seq_pipeline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sequential pipeline module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["seq_pipeline"] = module
    spec.loader.exec_module(module)
    return module


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "images").mkdir(parents=True, exist_ok=True)
    for version in range(1, 6):
        for subdir in ["features", "models", "reports"]:
            (OUT_ROOT / f"V{version}" / subdir).mkdir(parents=True, exist_ok=True)


def font(size: int) -> ImageFont.ImageFont:
    font_path = Path("C:/Windows/Fonts/tahoma.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def export_ready(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            values = out[column].dt.strftime("%Y-%m-%d %H:%M:%S")
            if column == "return_date":
                values = values.fillna("Not Returned")
            else:
                values = values.fillna("Unknown")
            out[column] = values
    for column in out.select_dtypes(include=["object", "string"]).columns:
        out[column] = out[column].fillna("Unknown").astype(str)
    for column in out.select_dtypes(include=[np.number]).columns:
        if out[column].isna().any():
            out[column] = out[column].fillna(out[column].median())
    return out


def choose_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, dict[str, float | int]]:
    best_threshold = 0.5
    best_score = -np.inf
    best_metrics: dict[str, float | int] = {}
    fallback_threshold = 0.5
    fallback_score = -np.inf
    fallback_metrics: dict[str, float | int] = {}
    for threshold in np.linspace(0.25, 0.80, 56):
        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        accuracy = accuracy_score(y_true, pred)
        recall = recall_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        cost = int(fn * 500 + fp * 50)
        score = (accuracy * 0.60) + (f1 * 0.25) + (precision * 0.10) + (recall * 0.05) - (cost / 20_000_000)
        metrics = {
            "accuracy": float(accuracy),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "cost": cost,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "threshold_score": float(score),
        }
        if score > fallback_score:
            fallback_score = score
            fallback_threshold = float(threshold)
            fallback_metrics = metrics
        if recall >= MIN_RECALL_FOR_THRESHOLD and score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    if not best_metrics:
        return fallback_threshold, fallback_metrics
    return best_threshold, best_metrics


def apply_version_feature_engineering(seq: Any, version: int, df: pd.DataFrame) -> pd.DataFrame:
    if version == 1:
        return seq.add_v1_features(df)
    if version == 2:
        return seq.add_customer_history(df)
    if version == 3:
        return seq.add_v3_features(df)
    if version == 4:
        return seq.add_v4_features(df)
    if version == 5:
        return df.copy()
    raise ValueError(version)


def train_full_version(seq: Any, version: int, df: pd.DataFrame, previous_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    version_dir = OUT_ROOT / f"V{version}"
    feature_dir = version_dir / "features"
    model_dir = version_dir / "models"
    report_dir = version_dir / "reports"

    features = seq.selected_features(version, df)
    x = df[features].copy()
    y = df[TARGET].astype(int).copy()

    scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    pipeline = Pipeline(
        steps=[
            ("preprocessor", seq.build_preprocessor(x)),
            ("model", XGBClassifier(**params)),
        ]
    )
    pipeline.fit(x, y)

    proba = pipeline.predict_proba(x)[:, 1]
    threshold, _ = choose_threshold(y.to_numpy(), proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    dropped = [feature for feature in previous_features if feature not in features]

    featured_path = feature_dir / f"df_featured_s2_v{version}.csv"
    used_path = feature_dir / f"used_features_s2_v{version}.csv"
    dropped_path = feature_dir / f"dropped_features_s2_v{version}.csv"
    feature_schema_path = feature_dir / f"feature_schema_s2_v{version}.json"
    train_full_path = feature_dir / f"train_full_sets_s2_v{version}.pkl"
    model_path = model_dir / f"model_s2_v{version}_xgboost.pkl"
    metadata_path = model_dir / f"model_s2_v{version}_metadata.json"
    metrics_path = report_dir / f"metrics_s2_v{version}.csv"
    predictions_path = report_dir / f"full_train_predictions_s2_v{version}.csv"

    df_featured = pd.concat(
        [
            df[["order_id", "customer_id", "order_date"]].copy(),
            x,
            y.rename(TARGET),
        ],
        axis=1,
    )
    metrics = {
        "version": f"V{version}",
        "dataset": "S2",
        "model": "XGBoost",
        "evaluation_type": "full_training_in_sample_no_holdout",
        "rows": len(df),
        "train_rows": len(df),
        "test_rows": 0,
        "feature_count": len(features),
        "dropped_from_previous": len(dropped),
        "threshold": threshold,
        "threshold_strategy": THRESHOLD_STRATEGY,
        "minimum_recall_for_threshold": MIN_RECALL_FOR_THRESHOLD,
        "accuracy": float(accuracy_score(y, pred)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, proba)),
        "avg_precision": float(average_precision_score(y, proba)),
        "cost": int(fn * 500 + fp * 50),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "scale_pos_weight": scale_pos_weight,
        "source_clean_dataset": str(SOURCE.relative_to(ROOT)),
        "df_featured_path": str(featured_path.relative_to(ROOT)),
        "model_path": str(model_path.relative_to(ROOT)),
        "train_full_path": str(train_full_path.relative_to(ROOT)),
    }

    export_ready(df_featured).to_csv(featured_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": features}).to_csv(used_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"dropped_feature": dropped}).to_csv(dropped_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "order_id": df["order_id"].to_numpy(),
            "customer_id": df["customer_id"].to_numpy(),
            "actual_is_returned": y.to_numpy(),
            "predict_probability_return": proba,
            "predicted_is_returned": pred,
            "threshold": threshold,
        }
    ).to_csv(predictions_path, index=False, encoding="utf-8-sig")

    feature_schema = {
        "version": f"V{version}",
        "dataset": "S2",
        "feature_names": features,
        "feature_count": len(features),
        "numeric_features": x.select_dtypes(include=[np.number, "bool"]).columns.tolist(),
        "categorical_features": [col for col in x.columns if col not in x.select_dtypes(include=[np.number, "bool"]).columns],
        "target": TARGET,
        "xgboost_params": params,
        "model_update_policy": (
            "This model is a fixed snapshot. It supports new incoming rows only when the same feature schema is built. "
            "It does not learn automatically or jump to a new model; retrain/tune a new version when new data changes pattern."
        ),
    }
    feature_schema_path.write_text(json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "X_train_full": x,
            "y_train_full": y,
            "feature_names": features,
            "threshold": threshold,
            "evaluation_type": "full_training_in_sample_no_holdout",
        },
        train_full_path,
    )
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                **metrics,
                "xgboost_params": params,
                "feature_schema_path": str(feature_schema_path.relative_to(ROOT)),
                "model_update_policy": feature_schema["model_update_policy"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    readme = f"""# S2 V{version} - XGBoost Full Training

{VERSION_DESCRIPTIONS[version]}

- Clean dataset source: `{SOURCE.relative_to(ROOT)}`
- Rows used for training: `{len(df):,}`
- Test split: `0%` / no holdout test
- Evaluation type: `full_training_in_sample_no_holdout`
- Model: `XGBoost`
- Feature count: `{len(features)}`
- Accuracy: `{metrics["accuracy"] * 100:.2f}%`
- Recall: `{metrics["recall"] * 100:.2f}%`
- Precision: `{metrics["precision"] * 100:.2f}%`
- F1: `{metrics["f1"] * 100:.2f}%`
- AUC: `{metrics["auc"] * 100:.2f}%`
- Cost: `{metrics["cost"]:,}`

หมายเหตุ: Accuracy นี้เป็นค่าจากการ train และ predict บนข้อมูลเต็ม 50,000 rows ชุดเดียวกัน ไม่ใช่ holdout test accuracy

## Model Update Policy

Model version นี้รองรับข้อมูลใหม่เมื่อสร้าง feature schema ให้ตรงกับไฟล์ `feature_schema_s2_v{version}.json` แต่ model จะไม่เรียนรู้เองและไม่ทิ้งตัวเก่าแล้วกระโดดเป็นตัวใหม่ ต้อง retrain/tune เป็น version ใหม่ถ้า pattern ของข้อมูลใหม่เปลี่ยน
"""
    (version_dir / "README.md").write_text(readme, encoding="utf-8")
    return pd.DataFrame([metrics]), features


def draw_accuracy_chart(summary: pd.DataFrame) -> None:
    width, height = 1500, 850
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    colors = ["#6D7C85", "#2E7D32", "#F9A825", "#1565C0", "#8E24AA"]

    draw.text((width // 2, 52), "S2 Full-Training Accuracy: V1-V5", font=font(40), fill="#111111", anchor="ma")
    draw.text((width // 2, 98), "No 20% test split: train and predict on full 50,000 rows", font=font(22), fill="#455A64", anchor="ma")
    x0, y0, x1, y1 = 130, 160, 1400, 700
    draw.line((x0, y1, x1, y1), fill="#263238", width=2)
    draw.line((x0, y0, x0, y1), fill="#263238", width=2)
    for tick in [0, 25, 50, 75, 100]:
        y = int(y1 - (tick / 100) * (y1 - y0))
        draw.line((x0 - 6, y, x1, y), fill="#ECEFF1", width=1)
        draw.text((x0 - 14, y), str(tick), font=font(18), fill="#455A64", anchor="rm")

    bar_gap = (x1 - x0 - 160) // len(summary)
    for i, row in summary.iterrows():
        value = float(row["accuracy"]) * 100
        bx = x0 + 80 + i * bar_gap
        bw = 120
        by = int(y1 - (value / 100) * (y1 - y0))
        draw.rounded_rectangle((bx, by, bx + bw, y1), radius=8, fill=colors[i])
        draw.text((bx + bw // 2, by - 18), f"{value:.2f}%", font=font(23), fill="#111111", anchor="mb")
        draw.text((bx + bw // 2, y1 + 34), row["version"], font=font(25), fill="#111111", anchor="ma")
        draw.text((bx + bw // 2, y1 + 66), f"{int(row['feature_count'])} features", font=font(17), fill="#455A64", anchor="ma")

    img.save(OUT_ROOT / "images" / "s2_full_training_accuracy_v1_to_v5.png")


def write_summary_readme(summary: pd.DataFrame) -> None:
    table_rows = []
    for _, row in summary.iterrows():
        table_rows.append(
            f"| {row['version']} | {int(row['feature_count'])} | {row['accuracy'] * 100:.2f}% | "
            f"{row['recall'] * 100:.2f}% | {row['precision'] * 100:.2f}% | "
            f"{row['f1'] * 100:.2f}% | {row['auc'] * 100:.2f}% | {int(row['cost']):,} |"
        )

    content = f"""# S2 Sequential V1-V5 Full Training

Source clean dataset: `{SOURCE.relative_to(ROOT)}`

Flow: `clean_dataset_s2.csv -> V1 -> V2 -> V3 -> V4 -> V5`

ทุก version ใช้ XGBoost และ train ด้วยข้อมูลเต็ม 50,000 rows โดยไม่แบ่ง test 20% ดังนั้น metric ในตารางนี้คือ `full_training_in_sample_no_holdout` ไม่ใช่ holdout test accuracy

| Version | Features | Accuracy | Recall | Precision | F1 | AUC | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## Version Difference

- V1: baseline raw/order-time features
- V2: เอา V1 มาทำต่อ เพิ่ม customer history และ rolling history แบบ point-in-time
- V3: เอา V2 มาทำต่อ เพิ่ม business interaction เช่น category/payment, category/channel, province/payment
- V4: เอา V3 มาทำต่อ เพิ่ม segment/operation risk เช่น price band, discount band, rating band, province-category risk
- V5: เอา V4 transformation มาทำต่อ แต่ใช้ compact feature set ลด feature ที่ซ้ำซ้อน/noise

## Model Update Policy

แต่ละ model เป็น snapshot ของ version นั้น รองรับข้อมูลใหม่ได้เมื่อสร้าง feature schema ให้ตรงกับ version นั้น แต่ model จะไม่เรียนรู้เอง ไม่ทิ้งตัวเก่า และไม่กระโดดไปเป็น model ใหม่โดยอัตโนมัติ หากข้อมูลใหม่เปลี่ยน pattern ต้อง retrain/tune เป็น version ใหม่
"""
    (OUT_ROOT / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    seq = load_seq_module()
    base = pd.read_csv(SOURCE)
    if len(base) != 50_000:
        raise AssertionError(f"Expected 50,000 rows, got {len(base)}")
    if int(base.isna().sum().sum()) != 0:
        raise AssertionError("clean_dataset_s2.csv still has missing/null values")

    current = seq.clean_dataset(base)
    summary_parts: list[pd.DataFrame] = []
    previous_features: list[str] = []
    for version in range(1, 6):
        current = apply_version_feature_engineering(seq, version, current)
        metrics, features = train_full_version(seq, version, current, previous_features)
        summary_parts.append(metrics)
        previous_features = features

    summary = pd.concat(summary_parts, ignore_index=True)
    summary.to_csv(OUT_ROOT / "s2_v1_to_v5_full_train_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_ROOT / "s2_v1_to_v5_full_train_summary.json").write_text(
        summary.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    draw_accuracy_chart(summary)
    write_summary_readme(summary)
    print(summary[["version", "feature_count", "accuracy", "recall", "precision", "f1", "auc", "cost"]].to_string(index=False))
    print(f"Output root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
