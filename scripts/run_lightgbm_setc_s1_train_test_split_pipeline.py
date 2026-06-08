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
from lightgbm import LGBMClassifier
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


RANDOM_STATE = 42
TARGET = "is_returned"
TEST_SIZE = 0.20
EVALUATION_TYPE = "train_test_split_80_20_stratified_holdout"

SOURCE_CANDIDATES = [
    ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "clean_dataset_s1.csv",
    ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S1" / "clean_dataset_s1.csv",
]
OUT_ROOT = ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S1"
SOURCE_COPY = OUT_ROOT / "clean_dataset_s1.csv"
SUMMARY_NAME = "lgbm_s1_v1_to_v5_train_test_split_summary"
MODEL_PREFIX = "lgbm_s1"
CHART_NAME = "lgbm_s1_train_test_split_accuracy_v1_to_v5.png"
DATASET_LABEL = "LightGBM_SETC_S1"
DATASET_DISPLAY = "LightGBM SETC S1"
EXPECTED_ROWS = 5_000
SOURCE_FILE_NAME = "clean_dataset_s1.csv"

BASE_LGBM_PARAMS = {
    "n_estimators": 520,
    "learning_rate": 0.035,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 35,
    "subsample": 0.90,
    "subsample_freq": 1,
    "colsample_bytree": 0.90,
    "reg_lambda": 2.0,
    "reg_alpha": 0.10,
    "objective": "binary",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
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


def resolve_source() -> Path:
    for path in SOURCE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {DATASET_DISPLAY} {SOURCE_FILE_NAME}")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    regular = Path("C:/Windows/Fonts/tahoma.ttf")
    bold_path = Path("C:/Windows/Fonts/tahomabd.ttf")
    path = bold_path if bold and bold_path.exists() else regular
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "images").mkdir(parents=True, exist_ok=True)
    for version in range(1, 6):
        for subdir in ["features", "models", "reports"]:
            (OUT_ROOT / f"V{version}" / subdir).mkdir(parents=True, exist_ok=True)


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
    version: int,
    df: pd.DataFrame,
    previous_features: list[str],
    source_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    version_dir = OUT_ROOT / f"V{version}"
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
    params = {**BASE_LGBM_PARAMS, "scale_pos_weight": scale_pos_weight}
    pipeline = Pipeline(
        steps=[
            ("preprocessor", seq.build_preprocessor(x_train)),
            ("model", LGBMClassifier(**params)),
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

    featured_path = feature_dir / f"df_featured_{MODEL_PREFIX}_v{version}.csv"
    used_path = feature_dir / f"used_features_{MODEL_PREFIX}_v{version}.csv"
    dropped_path = feature_dir / f"dropped_features_{MODEL_PREFIX}_v{version}.csv"
    feature_schema_path = feature_dir / f"feature_schema_{MODEL_PREFIX}_v{version}.json"
    train_test_path = feature_dir / f"train_test_sets_{MODEL_PREFIX}_v{version}.pkl"
    model_path = model_dir / f"model_{MODEL_PREFIX}_v{version}_lightgbm.pkl"
    metadata_path = model_dir / f"model_{MODEL_PREFIX}_v{version}_metadata.json"
    metrics_path = report_dir / f"metrics_{MODEL_PREFIX}_v{version}.csv"
    predictions_path = report_dir / f"test_predictions_{MODEL_PREFIX}_v{version}.csv"

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
        "dataset": DATASET_LABEL,
        "model": "LightGBM",
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
        "source_clean_dataset": str(source_path.relative_to(ROOT)),
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
        "dataset": DATASET_LABEL,
        "feature_names": features,
        "feature_count": len(features),
        "numeric_features": numeric_cols,
        "categorical_features": [col for col in x_train.columns if col not in numeric_cols],
        "target": TARGET,
        "evaluation_type": EVALUATION_TYPE,
        "split_strategy": metrics["split_strategy"],
        "lightgbm_params": params,
        "model_update_policy": (
            "This model supports new incoming rows when the exact same feature schema is built. "
            "It does not learn automatically from new data and does not jump to a new model by itself. "
            "Retrain/tune a new version when data patterns drift or new features are introduced."
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
                "lightgbm_params": params,
                "feature_schema_path": str(feature_schema_path.relative_to(ROOT)),
                "model_update_policy": feature_schema["model_update_policy"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    readme = f"""# {DATASET_DISPLAY} V{version} - 80/20 Train-Test Split

{VERSION_DESCRIPTIONS[version]}

- Source dataset: `{source_path.relative_to(ROOT)}`
- Evaluation type: `{EVALUATION_TYPE}`
- Split: `80% train / 20% test`
- Split method: stratified by `is_returned`, `random_state=42`
- Train rows: `{len(train_idx):,}`
- Test rows: `{len(test_idx):,}`
- Model: `LightGBM`
- Feature count: `{len(features)}`
- Test Accuracy: `{metrics["accuracy"] * 100:.2f}%`
- Test Recall: `{metrics["recall"] * 100:.2f}%`
- Test Precision: `{metrics["precision"] * 100:.2f}%`
- Test F1: `{metrics["f1"] * 100:.2f}%`
- Test AUC: `{metrics["auc"] * 100:.2f}%`
- Test Cost: `{metrics["cost"]:,}`

## New Data Policy

The model can predict new rows only when the same feature schema is built. It does not automatically forget old data, learn from new data, or jump to a new version. New patterns require retraining/tuning.

## Files

- Featured dataset: `{featured_path.relative_to(ROOT)}`
- Train/test artifact: `{train_test_path.relative_to(ROOT)}`
- Model: `{model_path.relative_to(ROOT)}`
- Metrics: `{metrics_path.relative_to(ROOT)}`
- Test predictions: `{predictions_path.relative_to(ROOT)}`
"""
    (version_dir / "README.md").write_text(readme, encoding="utf-8")
    return pd.DataFrame([metrics]), features


def draw_accuracy_chart(summary: pd.DataFrame) -> None:
    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    colors = ["#6D7C85", "#2E7D32", "#F9A825", "#1565C0", "#8E24AA"]

    plot_left, plot_top, plot_right, plot_bottom = 210, 235, 1745, 805
    plot_h = plot_bottom - plot_top
    slot_w = (plot_right - plot_left) / len(summary)
    bar_w = 142

    draw.text((width // 2, 52), f"{DATASET_DISPLAY} 80/20 Holdout Accuracy: V1-V5", font=font(48, True), fill="#111111", anchor="ma")
    draw.text((width // 2, 116), f"Train 80% | Test 20% | Stratified by is_returned | Source: {SOURCE_FILE_NAME}", font=font(24), fill="#455A64", anchor="ma")
    draw.text((plot_left, plot_top - 54), "Accuracy (%)", font=font(28, True), fill="#263238")
    for tick in range(0, 101, 10):
        y = plot_bottom - int((tick / 100) * plot_h)
        draw.line((plot_left, y, plot_right, y), fill="#DDE4E8" if tick else "#263238", width=1 if tick else 3)
        draw.text((plot_left - 24, y), str(tick), font=font(23), fill="#455A64", anchor="rm")
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#263238", width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#263238", width=3)

    best_idx = int(summary["accuracy"].idxmax())
    best = summary.loc[best_idx]
    draw.text((width // 2, 176), f"Best Test Accuracy: {best['version']} = {best['accuracy'] * 100:.2f}%", font=font(28, True), fill="#1B5E20", anchor="ma")

    for i, row in summary.reset_index(drop=True).iterrows():
        value = float(row["accuracy"]) * 100
        center_x = int(plot_left + slot_w * i + slot_w / 2)
        x0 = center_x - bar_w // 2
        x1 = center_x + bar_w // 2
        y0 = plot_bottom - int((value / 100) * plot_h)
        outline_w = 4 if row["version"] == best["version"] else 2
        outline = "#1B5E20" if row["version"] == best["version"] else "#263238"
        draw.rounded_rectangle((x0, y0, x1, plot_bottom), radius=8, fill=colors[i], outline=outline, width=outline_w)
        draw.text((center_x, y0 - 18), f"{value:.2f}%", font=font(25, True), fill="#111111", anchor="mb")
        draw.text((center_x, plot_bottom + 34), row["version"], font=font(30), fill="#111111", anchor="ma")
        draw.text((center_x, plot_bottom + 72), f"{int(row['feature_count'])} features", font=font(18), fill="#455A64", anchor="ma")
        draw.text((center_x, plot_bottom + 102), f"Recall {row['recall'] * 100:.1f}%", font=font(18), fill="#455A64", anchor="ma")

    image.save(OUT_ROOT / "images" / CHART_NAME)


def write_dataset_readme(source_path: Path, summary: pd.DataFrame) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"| {row['version']} | {int(row['feature_count'])} | {float(row['accuracy']) * 100:.2f}% | "
            f"{float(row['recall']) * 100:.2f}% | {float(row['precision']) * 100:.2f}% | "
            f"{float(row['f1']) * 100:.2f}% | {float(row['auc']) * 100:.2f}% | {int(row['cost']):,} |"
        )

    best = summary.loc[int(summary["accuracy"].idxmax())]
    content = f"""# {DATASET_DISPLAY} - Sequential V1-V5

Source clean dataset: `{source_path.relative_to(ROOT)}`

This folder contains LightGBM models trained from the {EXPECTED_ROWS:,}-row clean dataset.

## Process

1. Load clean dataset with {EXPECTED_ROWS:,} rows and 65 clean columns.
2. Build features sequentially:
   - V1: baseline clean/order-time features
   - V2: V1 plus customer history/rolling features
   - V3: V2 plus business interaction and group return-rate features
   - V4: V3 plus segment-risk, logistics, price/discount/rating features
   - V5: compact feature set selected from V4
3. Train each version with LightGBM.
4. Use `train_test_split(test_size=0.20, stratify=is_returned, random_state=42)`.
5. Store model, feature set, train/test artifact, metrics, and predictions in each V folder.

## New Data Policy

The model supports new incoming data only when the same feature schema is built before prediction. It does not automatically learn, forget old data, or jump to a new model. If business patterns change or new features are added, retrain/tune a new version.

## Results

| Version | Features | Accuracy | Recall | Precision | F1 | AUC | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Best holdout Accuracy: `{best['version']}` = `{float(best['accuracy']) * 100:.2f}%`
"""
    (OUT_ROOT / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    source_path = resolve_source()
    ensure_dirs()

    raw = pd.read_csv(source_path)
    if len(raw) != EXPECTED_ROWS:
        raise AssertionError(f"Expected {EXPECTED_ROWS:,} rows for {DATASET_DISPLAY}, got {len(raw):,}")
    if len(raw.columns) != 65:
        raise AssertionError(f"Expected 65 columns for {DATASET_DISPLAY}, got {len(raw.columns):,}")
    if int(raw.isna().sum().sum()) != 0:
        raise AssertionError(f"{DATASET_DISPLAY} source still has missing/null values")

    # Keep a source copy inside the dataset model folder for traceability.
    if source_path.resolve() != SOURCE_COPY.resolve():
        raw.to_csv(SOURCE_COPY, index=False, encoding="utf-8-sig")
        source_path = SOURCE_COPY

    seq = load_seq_module()
    current = seq.clean_dataset(raw)
    previous_features: list[str] = []
    summary_rows: list[pd.DataFrame] = []

    for version in range(1, 6):
        current = apply_version_feature_engineering(seq, version, current)
        metrics, previous_features = split_and_train(seq, version, current, previous_features, source_path)
        summary_rows.append(metrics)

    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(OUT_ROOT / f"{SUMMARY_NAME}.csv", index=False, encoding="utf-8-sig")
    (OUT_ROOT / f"{SUMMARY_NAME}.json").write_text(
        summary.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    draw_accuracy_chart(summary)
    write_dataset_readme(source_path, summary)

    print(summary[["version", "feature_count", "accuracy", "recall", "precision", "f1", "auc", "cost"]].to_string(index=False))


if __name__ == "__main__":
    main()
