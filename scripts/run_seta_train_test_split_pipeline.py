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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET = "is_returned"
TEST_SIZE = 0.20
EVALUATION_TYPE = "train_test_split_80_20_stratified_holdout"

BASE_XGB_PARAMS = {
    "n_estimators": 460,
    "max_depth": 4,
    "learning_rate": 0.035,
    "min_child_weight": 3,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_lambda": 2.0,
    "reg_alpha": 0.10,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

DATASETS = {
    "S1": {
        "source": ROOT / "docs" / "XGBoost" / "SETA" / "clean_data" / "clean_dataset_s1.csv",
        "out_root": ROOT / "docs" / "XGBoost" / "SETA" / "clean_data" / "S1",
        "expected_rows": 5_000,
        "summary_name": "s1_v1_to_v5_train_test_split_summary",
        "model_prefix": "s1",
        "chart_name": "s1_train_test_split_accuracy_v1_to_v5.png",
    },
    "S2": {
        "source": ROOT / "docs" / "XGBoost" / "SETA" / "clean_data" / "clean_dataset_s2.csv",
        "out_root": ROOT / "docs" / "XGBoost" / "SETA" / "clean_data" / "S2",
        "expected_rows": 50_000,
        "summary_name": "s2_v1_to_v5_train_test_split_summary",
        "model_prefix": "s2",
        "chart_name": "s2_train_test_split_accuracy_v1_to_v5.png",
    },
}

VERSION_DESCRIPTIONS = {
    1: "V1 baseline: raw/order-time features from the clean dataset.",
    2: "V2 history: V1 plus customer history and rolling point-in-time history.",
    3: "V3 interaction: V2 plus business interaction and group return-rate features.",
    4: "V4 segment risk: V3 plus price, discount, rating, logistics, and segment-risk features.",
    5: "V5 compact: V4 transformations with a reduced compact feature set.",
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


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    regular = Path("C:/Windows/Fonts/tahoma.ttf")
    bold_path = Path("C:/Windows/Fonts/tahomabd.ttf")
    path = bold_path if bold and bold_path.exists() else regular
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def ensure_dirs(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    for version in range(1, 6):
        for subdir in ["features", "models", "reports"]:
            (out_root / f"V{version}" / subdir).mkdir(parents=True, exist_ok=True)


def export_ready(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            values = out[column].dt.strftime("%Y-%m-%d %H:%M:%S")
            out[column] = values.fillna("Not Returned" if column == "return_date" else "Unknown")
    for column in out.select_dtypes(include=["object", "string"]).columns:
        out[column] = out[column].fillna("Unknown").astype(str)
    for column in out.select_dtypes(include=[np.number]).columns:
        if out[column].isna().any():
            fill = out[column].median()
            out[column] = out[column].fillna(0 if pd.isna(fill) else fill)
    return out


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


def choose_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, dict[str, float | int]]:
    best_threshold = 0.5
    best_score = -np.inf
    best_metrics: dict[str, float | int] = {}
    for threshold in np.linspace(0.20, 0.80, 61):
        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        accuracy = accuracy_score(y_true, pred)
        recall = recall_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        cost = int(fn * 500 + fp * 50)
        score = (accuracy * 0.55) + (f1 * 0.25) + (recall * 0.15) - (cost / 500000)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = {
                "accuracy": float(accuracy),
                "recall": float(recall),
                "precision": float(precision),
                "f1": float(f1),
                "cost": cost,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
    return best_threshold, best_metrics


def safe_auc(y_true: pd.Series | np.ndarray, proba: np.ndarray) -> float:
    y_arr = np.asarray(y_true)
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, proba))


def split_and_train(
    seq: Any,
    dataset_key: str,
    cfg: dict[str, object],
    version: int,
    df: pd.DataFrame,
    previous_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    out_root = Path(cfg["out_root"])
    model_prefix = str(cfg["model_prefix"])
    source = Path(cfg["source"])
    version_dir = out_root / f"V{version}"
    feature_dir = version_dir / "features"
    model_dir = version_dir / "models"
    report_dir = version_dir / "reports"

    features = seq.selected_features(version, df)
    x = df[features].copy()
    y = df[TARGET].astype(int).copy()
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
        shuffle=True,
    )
    x_train = x.loc[train_idx].copy()
    x_test = x.loc[test_idx].copy()
    y_train = y.loc[train_idx].copy()
    y_test = y.loc[test_idx].copy()

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {**BASE_XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    pipeline = Pipeline(
        steps=[
            ("preprocessor", seq.build_preprocessor(x_train)),
            ("model", XGBClassifier(**params)),
        ]
    )
    pipeline.fit(x_train, y_train)

    train_proba = pipeline.predict_proba(x_train)[:, 1]
    threshold, train_threshold_metrics = choose_threshold(y_train.to_numpy(), train_proba)
    test_proba = pipeline.predict_proba(x_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    train_pred = (train_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
    dropped = [feature for feature in previous_features if feature not in features]

    featured_path = feature_dir / f"df_featured_{model_prefix}_v{version}.csv"
    used_path = feature_dir / f"used_features_{model_prefix}_v{version}.csv"
    dropped_path = feature_dir / f"dropped_features_{model_prefix}_v{version}.csv"
    feature_schema_path = feature_dir / f"feature_schema_{model_prefix}_v{version}.json"
    train_test_path = feature_dir / f"train_test_sets_{model_prefix}_v{version}.pkl"
    model_path = model_dir / f"model_{model_prefix}_v{version}_xgboost.pkl"
    metadata_path = model_dir / f"model_{model_prefix}_v{version}_metadata.json"
    metrics_path = report_dir / f"metrics_{model_prefix}_v{version}.csv"
    predictions_path = report_dir / f"test_predictions_{model_prefix}_v{version}.csv"

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
        "dataset": dataset_key,
        "model": "XGBoost",
        "evaluation_type": EVALUATION_TYPE,
        "split_strategy": "train_test_split(test_size=0.20, stratify=is_returned, random_state=42)",
        "rows": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_return_rate": float(y_train.mean()),
        "test_return_rate": float(y_test.mean()),
        "feature_count": int(len(features)),
        "dropped_from_previous": int(len(dropped)),
        "threshold": float(threshold),
        "threshold_source": "chosen_on_train_split_only",
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "train_recall": float(recall_score(y_train, train_pred, zero_division=0)),
        "train_precision": float(precision_score(y_train, train_pred, zero_division=0)),
        "train_f1": float(f1_score(y_train, train_pred, zero_division=0)),
        "train_auc": safe_auc(y_train, train_proba),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "auc": safe_auc(y_test, test_proba),
        "avg_precision": float(average_precision_score(y_test, test_proba)),
        "cost": int(fn * 500 + fp * 50),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "scale_pos_weight": float(scale_pos_weight),
        "train_threshold_accuracy": train_threshold_metrics.get("accuracy"),
        "source_clean_dataset": str(source.relative_to(ROOT)),
        "df_featured_path": str(featured_path.relative_to(ROOT)),
        "model_path": str(model_path.relative_to(ROOT)),
        "train_test_path": str(train_test_path.relative_to(ROOT)),
    }

    export_ready(df_featured).to_csv(featured_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": features}).to_csv(used_path, index=False, encoding="utf-8-sig")
    pd.DataFrame({"dropped_feature": dropped}).to_csv(dropped_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "order_id": df.loc[test_idx, "order_id"].to_numpy(),
            "customer_id": df.loc[test_idx, "customer_id"].to_numpy(),
            "actual_is_returned": y_test.to_numpy(),
            "predict_probability_return": test_proba,
            "predicted_is_returned": test_pred,
            "threshold": threshold,
            "correct_prediction": (y_test.to_numpy() == test_pred).astype(int),
        }
    ).to_csv(predictions_path, index=False, encoding="utf-8-sig")

    numeric_cols = x_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    feature_schema = {
        "version": f"V{version}",
        "dataset": dataset_key,
        "feature_names": features,
        "feature_count": len(features),
        "numeric_features": numeric_cols,
        "categorical_features": [col for col in x_train.columns if col not in numeric_cols],
        "target": TARGET,
        "evaluation_type": EVALUATION_TYPE,
        "split_strategy": metrics["split_strategy"],
        "xgboost_params": params,
        "model_update_policy": (
            "This model is trained on the 80% train split and evaluated on the 20% holdout split. "
            "It supports new incoming rows only when the same feature schema is built. "
            "It does not learn automatically; retrain/tune a new version when data patterns change."
        ),
    }
    feature_schema_path.write_text(json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "X_train": x_train,
            "X_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_indices": train_idx,
            "test_indices": test_idx,
            "feature_names": features,
            "threshold": threshold,
            "evaluation_type": EVALUATION_TYPE,
            "split_strategy": metrics["split_strategy"],
        },
        train_test_path,
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

    readme = f"""# {dataset_key} V{version} - XGBoost 80/20 Train-Test Split

{VERSION_DESCRIPTIONS[version]}

- Source dataset: `{source.relative_to(ROOT)}`
- Evaluation type: `{EVALUATION_TYPE}`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `{len(train_idx):,}`
- Test rows: `{len(test_idx):,}`
- Model: `XGBoost`
- Feature count: `{len(features)}`
- Test Accuracy: `{metrics["accuracy"] * 100:.2f}%`
- Test Recall: `{metrics["recall"] * 100:.2f}%`
- Test Precision: `{metrics["precision"] * 100:.2f}%`
- Test F1: `{metrics["f1"] * 100:.2f}%`
- Test AUC: `{metrics["auc"] * 100:.2f}%`
- Test Cost: `{metrics["cost"]:,}`

## Why This Changed

The previous SETA artifacts used full-training/in-sample evaluation. This version follows train/test split validation: train on 80% of the data and evaluate on the 20% holdout split that the model did not train on.

## Files

- Featured dataset: `{featured_path.relative_to(ROOT)}`
- Train/test artifact: `{train_test_path.relative_to(ROOT)}`
- Model: `{model_path.relative_to(ROOT)}`
- Metrics: `{metrics_path.relative_to(ROOT)}`
- Test predictions: `{predictions_path.relative_to(ROOT)}`
"""
    (version_dir / "README.md").write_text(readme, encoding="utf-8")
    return pd.DataFrame([metrics]), features


def draw_accuracy_chart(dataset_key: str, cfg: dict[str, object], summary: pd.DataFrame) -> None:
    out_root = Path(cfg["out_root"])
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    colors = ["#6D7C85", "#2E7D32", "#F9A825", "#1565C0", "#8E24AA"]

    def text_center(x: int, y: int, text: str, font_obj: ImageFont.ImageFont, fill: str = "#111111") -> None:
        box = draw.textbbox((0, 0), text, font=font_obj)
        draw.text((x - (box[2] - box[0]) // 2, y), text, font=font_obj, fill=fill)

    plot_left, plot_top, plot_right, plot_bottom = 210, 235, 1745, 805
    plot_h = plot_bottom - plot_top
    slot_w = (plot_right - plot_left) / len(summary)
    bar_w = 142

    text_center(width // 2, 48, f"{dataset_key} 80/20 Holdout Test Accuracy: Version 1-5", font(50, True))
    text_center(width // 2, 118, "Train 80% | Test 20% | Stratified by is_returned | Model: XGBoost", font(24), "#455A64")
    draw.text((plot_left, plot_top - 54), "Accuracy (%)", font=font(28, True), fill="#263238")
    for tick in range(0, 101, 10):
        y = plot_bottom - int((tick / 100) * plot_h)
        draw.line((plot_left, y, plot_right, y), fill="#DDE4E8" if tick else "#263238", width=1 if tick else 3)
        label = str(tick)
        box = draw.textbbox((0, 0), label, font=font(23))
        draw.text((plot_left - 24 - (box[2] - box[0]), y - (box[3] - box[1]) // 2), label, font=font(23), fill="#455A64")
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#263238", width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#263238", width=3)

    best_idx = int(summary["accuracy"].idxmax())
    best = summary.loc[best_idx]
    text_center(width // 2, 178, f"Best Test Accuracy: {best['version']} = {best['accuracy'] * 100:.2f}%", font(28, True), "#1B5E20")
    for i, row in summary.reset_index(drop=True).iterrows():
        value = float(row["accuracy"]) * 100
        center_x = int(plot_left + slot_w * i + slot_w / 2)
        x0 = center_x - bar_w // 2
        x1 = center_x + bar_w // 2
        y0 = plot_bottom - int((value / 100) * plot_h)
        outline_w = 4 if row["version"] == best["version"] else 2
        outline = "#1B5E20" if row["version"] == best["version"] else "#263238"
        draw.rectangle((x0, y0, x1, plot_bottom), fill=colors[i], outline=outline, width=outline_w)
        text_center(center_x, y0 - 44, f"{value:.2f}%", font(32, True))
        text_center(center_x, plot_bottom + 30, str(row["version"]), font(34, True))
        text_center(center_x, plot_bottom + 80, f"{int(row['feature_count'])} features", font(21), "#455A64")
    text_center(width // 2, 1008, "Note: This is holdout test accuracy, not full-training/in-sample accuracy.", font(22), "#607D8B")
    image.save(out_root / "images" / str(cfg["chart_name"]))


def write_root_readme(dataset_key: str, cfg: dict[str, object], summary: pd.DataFrame) -> None:
    out_root = Path(cfg["out_root"])
    source = Path(cfg["source"])
    table = []
    for _, row in summary.iterrows():
        table.append(
            f"| {row['version']} | {int(row['feature_count'])} | {int(row['train_rows']):,} | {int(row['test_rows']):,} | "
            f"{row['accuracy'] * 100:.2f}% | {row['recall'] * 100:.2f}% | {row['precision'] * 100:.2f}% | "
            f"{row['f1'] * 100:.2f}% | {row['auc'] * 100:.2f}% | {int(row['cost']):,} |"
        )
    content = f"""# {dataset_key} Sequential V1-V5 - 80/20 Train-Test Split

Source clean dataset: `{source.relative_to(ROOT)}`

Evaluation follows the train/test split idea: train on 80% of the rows and evaluate on the held-out 20% rows that the model did not train on.

- Split method: `train_test_split(test_size=0.20, stratify=is_returned, random_state=42)`
- Model: `XGBoost`
- Evaluation type: `{EVALUATION_TYPE}`
- Threshold: chosen on train split only, then applied to test split

| Version | Features | Train Rows | Test Rows | Test Accuracy | Test Recall | Test Precision | Test F1 | Test AUC | Test Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## Version Flow

- V1: baseline raw/order-time features
- V2: V1 plus customer history and rolling history
- V3: V2 plus business interaction and point-in-time group return-rate features
- V4: V3 plus segment/operation risk features
- V5: compact feature set after V4 transformations

## Important

Previous SETA metrics used `full_training_in_sample_no_holdout`. These artifacts replace that approach with a true 80/20 holdout split, so Accuracy may be lower but is more defensible for explaining model performance.
"""
    (out_root / "README.md").write_text(content, encoding="utf-8")


def run_dataset(dataset_key: str, cfg: dict[str, object], seq: Any) -> pd.DataFrame:
    source = Path(cfg["source"])
    out_root = Path(cfg["out_root"])
    ensure_dirs(out_root)
    if not source.exists():
        raise FileNotFoundError(source)
    raw = pd.read_csv(source)
    expected_rows = int(cfg["expected_rows"])
    if len(raw) != expected_rows:
        raise AssertionError(f"{dataset_key}: expected {expected_rows:,} rows, got {len(raw):,}")
    if int(raw.isna().sum().sum()) != 0:
        raise AssertionError(f"{dataset_key}: source still has missing/null values")

    current = seq.clean_dataset(raw)
    previous_features: list[str] = []
    summary_parts: list[pd.DataFrame] = []
    for version in range(1, 6):
        current = apply_version_feature_engineering(seq, version, current)
        metrics, features = split_and_train(seq, dataset_key, cfg, version, current, previous_features)
        summary_parts.append(metrics)
        previous_features = features

    summary = pd.concat(summary_parts, ignore_index=True)
    summary_name = str(cfg["summary_name"])
    summary.to_csv(out_root / f"{summary_name}.csv", index=False, encoding="utf-8-sig")
    (out_root / f"{summary_name}.json").write_text(
        summary.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    draw_accuracy_chart(dataset_key, cfg, summary)
    write_root_readme(dataset_key, cfg, summary)
    return summary


def main() -> None:
    seq = load_seq_module()
    for dataset_key, cfg in DATASETS.items():
        summary = run_dataset(dataset_key, cfg, seq)
        print(f"\n{dataset_key} output: {cfg['out_root']}")
        print(summary[["version", "feature_count", "train_rows", "test_rows", "accuracy", "recall", "precision", "f1", "auc", "cost"]].to_string(index=False))


if __name__ == "__main__":
    main()
