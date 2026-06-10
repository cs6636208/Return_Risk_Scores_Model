from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType
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

warnings.filterwarnings("ignore")

TARGET = "is_returned"
SEQ_SCRIPT = ROOT / "scripts" / "run_dataset_5000_sequential_version_pipeline.py"
SEQ_GIT_REF = "HEAD:scripts/run_dataset_5000_sequential_version_pipeline.py"


def load_seq_module() -> ModuleType:
    if SEQ_SCRIPT.exists():
        spec = importlib.util.spec_from_file_location("seq_pipeline", SEQ_SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load feature builder: {SEQ_SCRIPT}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["seq_pipeline"] = module
        spec.loader.exec_module(module)
        return module

    code = subprocess.check_output(["git", "show", SEQ_GIT_REF], cwd=ROOT, text=True, encoding="utf-8")
    module = ModuleType("seq_pipeline_from_git")
    module.__file__ = str(SEQ_SCRIPT)
    module.__dict__["__name__"] = "seq_pipeline_from_git"
    sys.modules[module.__name__] = module
    exec(compile(code, str(SEQ_SCRIPT), "exec"), module.__dict__)
    return module


def apply_version_feature_engineering(seq: Any, version: int, df: pd.DataFrame) -> pd.DataFrame:
    v1 = seq.add_v1_features(df)
    if version == 1:
        return v1
    v2 = seq.add_customer_history(v1)
    if version == 2:
        return v2
    v3 = seq.add_v3_features(v2)
    if version == 3:
        return v3
    v4 = seq.add_v4_features(v3)
    if version == 4:
        return v4
    if version == 5:
        # V5 is the compact feature set selected from all prior feature-engineering steps.
        return v4
    raise ValueError(f"Unsupported version: {version}")


def prepare_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "order_date",
        "expected_delivery_date",
        "delivery_date",
        "registration_date",
        "promo_start_date",
        "promo_end_date",
        "scored_at",
    ]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    if TARGET in out.columns:
        out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int)
    return out


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/tahomabd.ttf" if bold else "C:/Windows/Fonts/tahoma.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def blank_text_count(df: pd.DataFrame) -> int:
    return int(
        sum(
            (df[col].astype(str).str.strip() == "").sum()
            for col in df.select_dtypes(include=["object", "string"]).columns
        )
    )


def validation_row(df: pd.DataFrame, test_data: Path, train_set: str, test_set: str) -> dict[str, object]:
    return {
        "file": str(test_data.relative_to(ROOT)),
        "train_set": train_set,
        "test_set": test_set,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_or_null_cells": int(df.isna().sum().sum()),
        "blank_text_cells": blank_text_count(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_order_id": int(df["order_id"].duplicated().sum()) if "order_id" in df.columns else None,
        "distinct_order_id": int(df["order_id"].nunique()) if "order_id" in df.columns else None,
        "distinct_customer_id": int(df["customer_id"].nunique()) if "customer_id" in df.columns else None,
        "first_order_id": str(df["order_id"].iloc[0]) if "order_id" in df.columns and len(df) else "",
        "last_order_id": str(df["order_id"].iloc[-1]) if "order_id" in df.columns and len(df) else "",
        "min_order_date": str(pd.to_datetime(df["order_date"], errors="coerce").min()) if "order_date" in df.columns else "",
        "max_order_date": str(pd.to_datetime(df["order_date"], errors="coerce").max()) if "order_date" in df.columns else "",
        "returned_count": int((df[TARGET] == 1).sum()),
        "not_returned_count": int((df[TARGET] == 0).sum()),
        "return_rate": float(df[TARGET].mean()),
        "note": "External full-dataset test. SETD data is not split again.",
    }


def load_used_features(model_root: Path, model_prefix: str, version: int) -> list[str]:
    path = model_root / f"V{version}" / "features" / f"used_features_{model_prefix}_v{version}.csv"
    return pd.read_csv(path)["feature"].astype(str).tolist()


def load_holdout_metrics(model_root: Path, model_prefix: str, version: int) -> dict[str, float]:
    path = model_root / f"V{version}" / "reports" / f"metrics_{model_prefix}_v{version}.csv"
    row = pd.read_csv(path).iloc[0].to_dict()
    return {
        "threshold": float(row["threshold"]),
        "holdout_accuracy": float(row["accuracy"]),
        "holdout_recall": float(row["recall"]),
        "holdout_precision": float(row["precision"]),
        "holdout_f1": float(row["f1"]),
        "holdout_auc": float(row["auc"]),
        "holdout_cost": float(row["cost"]),
    }


def evaluate_one(
    *,
    df_featured: pd.DataFrame,
    model_root: Path,
    out_root: Path,
    model_prefix: str,
    version: int,
    train_dataset: str,
    test_dataset: str,
    test_data: Path,
) -> dict[str, object]:
    report_dir = out_root / f"V{version}" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    features = load_used_features(model_root, model_prefix, version)
    missing_features = [feature for feature in features if feature not in df_featured.columns]
    if missing_features:
        raise KeyError(f"{test_dataset} V{version} missing features: {missing_features}")

    model_path = model_root / f"V{version}" / "models" / f"model_{model_prefix}_v{version}_lightgbm.pkl"
    model = joblib.load(model_path)
    x_test = df_featured[features].copy()
    y_true = df_featured[TARGET].astype(int).to_numpy()

    holdout = load_holdout_metrics(model_root, model_prefix, version)
    threshold = holdout["threshold"]
    proba = model.predict_proba(x_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = float(accuracy_score(y_true, y_pred))

    metrics = {
        "version": f"V{version}",
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "evaluation_type": "external_full_dataset_accuracy",
        "rows": int(len(df_featured)),
        "feature_count": int(len(features)),
        "threshold": float(threshold),
        **holdout,
        "external_accuracy": accuracy,
        "external_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "external_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "external_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "external_auc": float(roc_auc_score(y_true, proba)),
        "external_avg_precision": float(average_precision_score(y_true, proba)),
        "external_cost": int(fn * 500 + fp * 50),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy_gap_points": float((accuracy - holdout["holdout_accuracy"]) * 100),
        "model_path": str(model_path.relative_to(ROOT)),
        "test_data_path": str(test_data.relative_to(ROOT)),
        "rerun_source": "scripts/rerun_lightgbm_setd_tests.py",
    }

    suffix = "real_dataset_s1" if "s1" in test_dataset.lower() else "real_dataset_s2"
    predictions = pd.DataFrame(
        {
            "order_id": df_featured["order_id"].astype(str).to_numpy(),
            "customer_id": df_featured["customer_id"].astype(str).to_numpy(),
            "actual_is_returned": y_true,
            "predict_probability_return": proba,
            "predicted_is_returned": y_pred,
            "threshold": threshold,
            "correct_prediction": (y_true == y_pred).astype(int),
        }
    )
    predictions.to_csv(
        report_dir / f"test_predictions_{model_prefix}_v{version}_on_{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([metrics]).to_csv(
        report_dir / f"test_metrics_{model_prefix}_v{version}_on_{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return metrics


def draw_accuracy_chart(summary: pd.DataFrame, out_root: Path, title: str, image_name: str) -> None:
    image_dir = out_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1700, 940
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    colors = ["#607D8B", "#2E7D32", "#F9A825", "#1565C0", "#8E24AA"]

    draw.text((width // 2, 50), title, font=font(38, bold=True), fill="#111111", anchor="ma")
    draw.text((width // 2, 98), "Holdout Accuracy vs SETD External Full-Dataset Accuracy; SETD is not split again", font=font(22), fill="#455A64", anchor="ma")

    x0, y0, x1, y1 = 120, 175, 1575, 720
    draw.line((x0, y1, x1, y1), fill="#263238", width=2)
    draw.line((x0, y0, x0, y1), fill="#263238", width=2)
    for tick in [0, 25, 50, 75, 100]:
        y = int(y1 - (tick / 100) * (y1 - y0))
        draw.line((x0 - 6, y, x1, y), fill="#ECEFF1", width=1)
        draw.text((x0 - 14, y), str(tick), font=font(18), fill="#455A64", anchor="rm")

    group_gap = (x1 - x0 - 120) // len(summary)
    for i, row in summary.reset_index(drop=True).iterrows():
        holdout = float(row["holdout_accuracy"]) * 100
        external = float(row["external_accuracy"]) * 100
        bx = x0 + 60 + i * group_gap
        bw = 58
        holdout_y = int(y1 - (holdout / 100) * (y1 - y0))
        external_y = int(y1 - (external / 100) * (y1 - y0))
        draw.rounded_rectangle((bx, holdout_y, bx + bw, y1), radius=6, fill="#B0BEC5")
        draw.rounded_rectangle((bx + bw + 12, external_y, bx + 2 * bw + 12, y1), radius=6, fill=colors[i % len(colors)])
        draw.text((bx + bw // 2, holdout_y - 8), f"{holdout:.1f}%", font=font(17), fill="#37474F", anchor="ms")
        draw.text((bx + bw + 12 + bw // 2, external_y - 8), f"{external:.1f}%", font=font(17), fill="#111111", anchor="ms")
        draw.text((bx + bw + 6, y1 + 18), str(row["version"]), font=font(22), fill="#111111", anchor="ma")

    legend_y = 790
    draw.rounded_rectangle((600, legend_y, 635, legend_y + 20), radius=4, fill="#B0BEC5")
    draw.text((650, legend_y + 10), "SETC holdout 20%", font=font(20), fill="#37474F", anchor="lm")
    draw.rounded_rectangle((880, legend_y, 915, legend_y + 20), radius=4, fill="#1565C0")
    draw.text((930, legend_y + 10), "SETD full external test", font=font(20), fill="#37474F", anchor="lm")

    image.save(image_dir / image_name)


def write_readme(out_root: Path, summary: pd.DataFrame, train_set: str, test_set: str, test_file: Path) -> None:
    lines = [
        f"# {test_set} -> {train_set} Rerun Result",
        "",
        f"- Train model source: `{train_set}`",
        f"- Test data source: `{test_file.relative_to(ROOT)}`",
        "- Evaluation: full-dataset external inference",
        "- SETD test data is not split into 20%; every row is passed through the matching V1-V5 feature builder and saved model.",
        "",
        "| Version | Rows | Features | Threshold | Holdout Accuracy | External Accuracy | Recall | Precision | F1 | AUC | Cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['version']} | {int(row['rows'])} | {int(row['feature_count'])} | {float(row['threshold']):.2f} | "
            f"{float(row['holdout_accuracy'])*100:.2f}% | {float(row['external_accuracy'])*100:.2f}% | "
            f"{float(row['external_recall'])*100:.2f}% | {float(row['external_precision'])*100:.2f}% | "
            f"{float(row['external_f1'])*100:.2f}% | {float(row['external_auc'])*100:.2f}% | {int(row['external_cost']):,} |"
        )
    lines += [
        "",
        "## Confirmation",
        "",
        f"Mapping confirmed: `{test_set}` was evaluated against `{train_set}` only.",
    ]
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_pair(*, pair: str) -> pd.DataFrame:
    if pair == "s3":
        test_data = ROOT / "docs" / "LightGBM" / "SETD" / "real_dataset" / "real_dataset_s1.csv"
        model_root = ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S1"
        out_root = ROOT / "docs" / "LightGBM" / "SETD" / "real_dataset" / "S3"
        model_prefix = "lgbm_s1"
        train_dataset = "LightGBM_SETC_S1_clean_dataset_s1"
        test_dataset = "LightGBM_SETD_S3_real_dataset_s1"
        summary_name = "lgbm_s1_v1_to_v5_real_dataset_s1_test_summary.csv"
        validation_name = "real_dataset_s1_validation_for_lgbm_s1_model_test.csv"
        comparison_name = "lgbm_s1_holdout_vs_real_dataset_s1_comparison.csv"
        chart_name = "lgbm_s1_models_real_dataset_s1_test_accuracy_v1_to_v5.png"
        title = "LightGBM SETC S1 Models Tested on SETD S3 real_dataset_s1"
    elif pair == "s4":
        test_data = ROOT / "docs" / "LightGBM" / "SETD" / "real_dataset" / "real_dataset_s2.csv"
        model_root = ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S2"
        out_root = ROOT / "docs" / "LightGBM" / "SETD" / "real_dataset" / "S4"
        model_prefix = "lgbm_s2"
        train_dataset = "LightGBM_SETC_S2_clean_dataset_s2"
        test_dataset = "LightGBM_SETD_S4_real_dataset_s2"
        summary_name = "lgbm_s2_v1_to_v5_real_dataset_s2_test_summary.csv"
        validation_name = "real_dataset_s2_validation_for_lgbm_s2_model_test.csv"
        comparison_name = "lgbm_s2_holdout_vs_real_dataset_s2_comparison.csv"
        chart_name = "lgbm_s2_models_real_dataset_s2_test_accuracy_v1_to_v5.png"
        title = "LightGBM SETC S2 Models Tested on SETD S4 real_dataset_s2"
    else:
        raise ValueError(pair)

    seq = load_seq_module()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    for version in range(1, 6):
        (out_root / f"V{version}" / "reports").mkdir(parents=True, exist_ok=True)

    raw = prepare_raw_dataset(pd.read_csv(test_data))
    pd.DataFrame([validation_row(raw, test_data, train_dataset, test_dataset)]).to_csv(
        out_root / validation_name,
        index=False,
        encoding="utf-8-sig",
    )

    v1 = seq.add_v1_features(raw)
    v2 = seq.add_customer_history(v1)
    v3 = seq.add_v3_features(v2)
    v4 = seq.add_v4_features(v3)
    featured_by_version = {1: v1, 2: v2, 3: v3, 4: v4, 5: v4}

    metrics_rows: list[dict[str, object]] = []
    for version, featured in featured_by_version.items():
        metrics_rows.append(
            evaluate_one(
                df_featured=featured,
                model_root=model_root,
                out_root=out_root,
                model_prefix=model_prefix,
                version=version,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                test_data=test_data,
            )
        )

    summary = pd.DataFrame(metrics_rows)
    summary.to_csv(out_root / summary_name, index=False, encoding="utf-8-sig")
    summary.to_json(out_root / summary_name.replace(".csv", ".json"), orient="records", indent=2, force_ascii=False)
    summary.to_csv(out_root / comparison_name, index=False, encoding="utf-8-sig")
    draw_accuracy_chart(summary, out_root, title, chart_name)
    write_readme(out_root, summary, train_dataset, test_dataset, test_data)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun LightGBM SETD S3/S4 external tests against SETC S1/S2 models.")
    parser.add_argument("--pair", choices=["s3", "s4", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = ["s3", "s4"] if args.pair == "both" else [args.pair]
    for pair in pairs:
        summary = run_pair(pair=pair)
        print(f"\n{pair.upper()} rerun complete")
        print(summary[["version", "rows", "feature_count", "holdout_accuracy", "external_accuracy", "external_recall", "external_f1", "external_auc", "external_cost"]].to_string(index=False))


if __name__ == "__main__":
    main()
