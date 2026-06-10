from __future__ import annotations

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

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "is_returned"
SEQ_SCRIPT = ROOT / "scripts" / "run_dataset_5000_sequential_version_pipeline.py"
SEQ_GIT_REF = "HEAD:scripts/run_dataset_5000_sequential_version_pipeline.py"

SOURCE_UNSEEN = ROOT / "data" / "processed" / "clean_dataset_v2.csv"
UNSEEN_ROOT = ROOT / "docs" / "LightGBM" / "SETD" / "real_dataset" / "UNSEEN_FROM_V2"
UNSEEN_CSV = UNSEEN_ROOT / "real_dataset_unseen_from_clean_dataset_v2.csv"

TUNED_ROOT = ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S2_TUNED_80_ATTEMPT"
S2_SOURCE = ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "clean_dataset_s2.csv"


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


def prepare_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "order_date",
        "expected_delivery_date",
        "delivery_date",
        "registration_date",
        "promo_start_date",
        "promo_end_date",
        "return_date",
        "scored_at",
    ]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col].replace({"Not Returned": pd.NA}), errors="coerce")
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int)
    return out


def feature_versions(seq: Any, raw: pd.DataFrame) -> dict[int, pd.DataFrame]:
    clean = seq.clean_dataset(raw)
    v1 = seq.add_v1_features(clean)
    v2 = seq.add_customer_history(v1)
    v3 = seq.add_v3_features(v2)
    v4 = seq.add_v4_features(v3)
    return {1: v1, 2: v2, 3: v3, 4: v4, 5: v4}


def safe_auc(y_true: np.ndarray | pd.Series, proba: np.ndarray) -> float:
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": safe_auc(y_true, proba),
        "avg_precision": float(average_precision_score(y_true, proba)),
        "cost": int(fn * 500 + fp * 50),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_search(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, dict[str, float | int]]:
    best_threshold = 0.5
    best_score = -np.inf
    best_metrics: dict[str, float | int] = {}
    for threshold in np.linspace(0.20, 0.80, 61):
        metrics = evaluate_predictions(y_true, proba, float(threshold))
        # Accuracy remains primary, but keep a real return-risk signal through F1/AUC.
        score = (
            float(metrics["accuracy"]) * 0.62
            + float(metrics["f1"]) * 0.24
            + float(metrics["auc"]) * 0.10
            + float(metrics["recall"]) * 0.04
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts/tahomabd.ttf" if bold else "C:/Windows/Fonts/tahoma.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_accuracy_chart(summary: pd.DataFrame, path: Path, title: str, value_col: str = "accuracy") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1500, 850
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    colors = ["#607D8B", "#2E7D32", "#F9A825", "#1565C0", "#8E24AA"]
    draw.text((width // 2, 45), title, font=font(36, True), fill="#111111", anchor="ma")
    x0, y0, x1, y1 = 125, 150, 1390, 650
    draw.line((x0, y1, x1, y1), fill="#263238", width=2)
    draw.line((x0, y0, x0, y1), fill="#263238", width=2)
    for tick in [0, 20, 40, 60, 80, 100]:
        y = int(y1 - tick / 100 * (y1 - y0))
        draw.line((x0 - 6, y, x1, y), fill="#ECEFF1", width=1)
        draw.text((x0 - 15, y), str(tick), font=font(18), fill="#455A64", anchor="rm")
    gap = (x1 - x0) / len(summary)
    for i, row in summary.reset_index(drop=True).iterrows():
        val = float(row[value_col]) * 100
        cx = int(x0 + gap * i + gap / 2)
        bw = 115
        by = int(y1 - val / 100 * (y1 - y0))
        draw.rounded_rectangle((cx - bw // 2, by, cx + bw // 2, y1), radius=8, fill=colors[i % len(colors)])
        draw.text((cx, by - 10), f"{val:.2f}%", font=font(22, True), fill="#111111", anchor="mb")
        draw.text((cx, y1 + 25), str(row["version"]), font=font(24), fill="#111111", anchor="ma")
        if "feature_count" in row:
            draw.text((cx, y1 + 58), f"{int(row['feature_count'])} features", font=font(16), fill="#455A64", anchor="ma")
    img.save(path)


def export_unseen_dataset() -> pd.DataFrame:
    UNSEEN_ROOT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE_UNSEEN)
    raw = raw.drop_duplicates("order_id", keep="last").reset_index(drop=True)
    raw.to_csv(UNSEEN_CSV, index=False, encoding="utf-8-sig")

    validation = {
        "source_file": str(SOURCE_UNSEEN.relative_to(ROOT)),
        "output_file": str(UNSEEN_CSV.relative_to(ROOT)),
        "rows": int(len(raw)),
        "columns": int(len(raw.columns)),
        "missing_or_null_cells": int(raw.isna().sum().sum()),
        "duplicate_order_id": int(raw["order_id"].duplicated().sum()),
        "distinct_order_id": int(raw["order_id"].nunique()),
        "distinct_customer_id": int(raw["customer_id"].nunique()),
        "return_rate": float(raw[TARGET].mean()),
        "note": "Unseen-like external test source selected from clean_dataset_v2.csv, not S1-aligned real_dataset_s1.csv.",
    }
    pd.DataFrame([validation]).to_csv(UNSEEN_ROOT / "real_dataset_unseen_validation.csv", index=False, encoding="utf-8-sig")
    (UNSEEN_ROOT / "real_dataset_unseen_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return raw


def distribution_compare(unseen: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    for label, path in [
        ("SETC_S1_train_source", ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "clean_dataset_s1.csv"),
        ("SETC_S2_train_source", ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "clean_dataset_s2.csv"),
        ("UNSEEN_FROM_V2", UNSEEN_CSV),
    ]:
        df = pd.read_csv(path)
        rows.append(
            {
                "dataset": label,
                "rows": len(df),
                "distinct_customer_id": df["customer_id"].nunique(),
                "return_rate": df[TARGET].mean(),
                "avg_total_amount": pd.to_numeric(df["total_amount"], errors="coerce").mean(),
                "avg_hist_return_rate": pd.to_numeric(df["hist_return_rate"], errors="coerce").mean(),
                "avg_product_rating": pd.to_numeric(df["product_rating"], errors="coerce").mean(),
            }
        )
    pd.DataFrame(rows).to_csv(UNSEEN_ROOT / "distribution_compare_s1_s2_unseen.csv", index=False, encoding="utf-8-sig")

    group_rows: list[dict[str, object]] = []
    for col in ["category", "payment_method", "channel_type", "province", "membership_tier"]:
        base = pd.read_csv(ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "clean_dataset_s1.csv")
        base_rates = base.groupby(col)[TARGET].mean()
        unseen_rates = unseen.groupby(col)[TARGET].mean()
        joined = pd.DataFrame({"s1_rate": base_rates, "unseen_rate": unseen_rates}).dropna()
        joined["abs_diff"] = (joined["unseen_rate"] - joined["s1_rate"]).abs()
        group_rows.append(
            {
                "group_column": col,
                "mean_abs_return_rate_diff_vs_s1": float(joined["abs_diff"].mean()),
                "max_abs_return_rate_diff_vs_s1": float(joined["abs_diff"].max()),
                "groups_compared": int(len(joined)),
            }
        )
    pd.DataFrame(group_rows).to_csv(UNSEEN_ROOT / "group_return_rate_distance_vs_s1.csv", index=False, encoding="utf-8-sig")


def load_used_features(model_root: Path, prefix: str, version: int) -> list[str]:
    return pd.read_csv(model_root / f"V{version}" / "features" / f"used_features_{prefix}_v{version}.csv")["feature"].astype(str).tolist()


def load_holdout_metrics(model_root: Path, prefix: str, version: int) -> dict[str, float]:
    row = pd.read_csv(model_root / f"V{version}" / "reports" / f"metrics_{prefix}_v{version}.csv").iloc[0].to_dict()
    return {
        "threshold": float(row["threshold"]),
        "holdout_accuracy": float(row["accuracy"]),
        "holdout_recall": float(row["recall"]),
        "holdout_precision": float(row["precision"]),
        "holdout_f1": float(row["f1"]),
        "holdout_auc": float(row["auc"]),
        "holdout_cost": float(row["cost"]),
    }


def evaluate_existing_models(seq: Any, raw_unseen: pd.DataFrame, model_set: str) -> pd.DataFrame:
    if model_set == "s1":
        model_root = ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S1"
        prefix = "lgbm_s1"
        out_root = UNSEEN_ROOT / "SETC_S1_MODELS"
        train_label = "LightGBM_SETC_S1_clean_dataset_s1"
    elif model_set == "s2":
        model_root = ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S2"
        prefix = "lgbm_s2"
        out_root = UNSEEN_ROOT / "SETC_S2_MODELS"
        train_label = "LightGBM_SETC_S2_clean_dataset_s2"
    else:
        raise ValueError(model_set)

    out_root.mkdir(parents=True, exist_ok=True)
    versions = feature_versions(seq, prepare_raw_dataset(raw_unseen))
    rows: list[dict[str, object]] = []
    for version, df_feat in versions.items():
        features = load_used_features(model_root, prefix, version)
        model = joblib.load(model_root / f"V{version}" / "models" / f"model_{prefix}_v{version}_lightgbm.pkl")
        holdout = load_holdout_metrics(model_root, prefix, version)
        proba = model.predict_proba(df_feat[features])[:, 1]
        y_true = df_feat[TARGET].astype(int).to_numpy()
        external = evaluate_predictions(y_true, proba, holdout["threshold"])
        row = {
            "version": f"V{version}",
            "train_dataset": train_label,
            "test_dataset": "UNSEEN_FROM_CLEAN_DATASET_V2",
            "rows": int(len(df_feat)),
            "feature_count": int(len(features)),
            **holdout,
            **{f"external_{k}": v for k, v in external.items()},
            "model_path": str((model_root / f"V{version}" / "models" / f"model_{prefix}_v{version}_lightgbm.pkl").relative_to(ROOT)),
            "test_data_path": str(UNSEEN_CSV.relative_to(ROOT)),
        }
        rows.append(row)
        pd.DataFrame([row]).to_csv(out_root / f"metrics_{prefix}_v{version}_on_unseen_from_v2.csv", index=False, encoding="utf-8-sig")
        pred = (proba >= holdout["threshold"]).astype(int)
        pd.DataFrame(
            {
                "order_id": df_feat["order_id"].astype(str),
                "customer_id": df_feat["customer_id"].astype(str),
                "actual_is_returned": y_true,
                "predict_probability_return": proba,
                "predicted_is_returned": pred,
                "threshold": holdout["threshold"],
                "correct_prediction": (pred == y_true).astype(int),
            }
        ).head(10000).to_csv(out_root / f"prediction_sample_10000_{prefix}_v{version}_on_unseen_from_v2.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / f"{prefix}_v1_to_v5_unseen_from_v2_summary.csv", index=False, encoding="utf-8-sig")
    draw_accuracy_chart(
        summary.rename(columns={"external_accuracy": "accuracy"}),
        out_root / "images" / f"{prefix}_unseen_from_v2_accuracy_v1_to_v5.png",
        f"{train_label} tested on UNSEEN_FROM_V2",
        value_col="accuracy",
    )
    return summary


def lgbm_param_grid() -> list[dict[str, Any]]:
    return [
        {
            "n_estimators": 420,
            "learning_rate": 0.035,
            "num_leaves": 23,
            "max_depth": 5,
            "min_child_samples": 90,
            "subsample": 0.82,
            "colsample_bytree": 0.82,
            "reg_lambda": 6.0,
            "reg_alpha": 0.5,
        },
        {
            "n_estimators": 520,
            "learning_rate": 0.030,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 120,
            "subsample": 0.88,
            "colsample_bytree": 0.86,
            "reg_lambda": 8.0,
            "reg_alpha": 0.7,
        },
        {
            "n_estimators": 620,
            "learning_rate": 0.025,
            "num_leaves": 39,
            "max_depth": 7,
            "min_child_samples": 140,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "reg_lambda": 10.0,
            "reg_alpha": 1.0,
        },
        {
            "n_estimators": 360,
            "learning_rate": 0.045,
            "num_leaves": 19,
            "max_depth": 4,
            "min_child_samples": 160,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "reg_lambda": 12.0,
            "reg_alpha": 1.2,
        },
    ]


def train_tuned_s2(seq: Any, raw_unseen: pd.DataFrame) -> pd.DataFrame:
    TUNED_ROOT.mkdir(parents=True, exist_ok=True)
    (TUNED_ROOT / "images").mkdir(parents=True, exist_ok=True)
    s2 = pd.read_csv(S2_SOURCE)
    s2_versions = feature_versions(seq, prepare_raw_dataset(s2))
    unseen_versions = feature_versions(seq, prepare_raw_dataset(raw_unseen))
    rows: list[dict[str, object]] = []

    for version, df_feat in s2_versions.items():
        version_root = TUNED_ROOT / f"V{version}"
        feature_dir = version_root / "features"
        model_dir = version_root / "models"
        report_dir = version_root / "reports"
        feature_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        features = seq.selected_features(version, df_feat)
        x = df_feat[features].copy()
        y = df_feat[TARGET].astype(int).copy()
        train_val_idx, holdout_idx = train_test_split(
            df_feat.index.to_numpy(),
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )
        fit_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.20,
            stratify=y.loc[train_val_idx],
            random_state=RANDOM_STATE,
        )
        x_fit, y_fit = x.loc[fit_idx].copy(), y.loc[fit_idx].copy()
        x_val, y_val = x.loc[val_idx].copy(), y.loc[val_idx].copy()
        x_hold, y_hold = x.loc[holdout_idx].copy(), y.loc[holdout_idx].copy()

        scale_pos_weight = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
        best: dict[str, Any] | None = None
        for grid_id, params in enumerate(lgbm_param_grid(), start=1):
            full_params = {
                **params,
                "objective": "binary",
                "n_jobs": -1,
                "random_state": RANDOM_STATE,
                "verbosity": -1,
                "scale_pos_weight": scale_pos_weight,
            }
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", seq.build_preprocessor(x_fit)),
                    ("model", LGBMClassifier(**full_params)),
                ]
            )
            pipeline.fit(x_fit, y_fit)
            val_proba = pipeline.predict_proba(x_val)[:, 1]
            threshold, val_metrics = threshold_search(y_val.to_numpy(), val_proba)
            score = (
                float(val_metrics["accuracy"]) * 0.62
                + float(val_metrics["f1"]) * 0.24
                + float(val_metrics["auc"]) * 0.10
                + float(val_metrics["recall"]) * 0.04
            )
            candidate = {
                "grid_id": grid_id,
                "params": full_params,
                "pipeline": pipeline,
                "threshold": threshold,
                "val_metrics": val_metrics,
                "score": score,
            }
            if best is None or score > best["score"]:
                best = candidate

        assert best is not None
        # Retrain selected params on train+validation, keep holdout untouched.
        final_pipeline = Pipeline(
            steps=[
                ("preprocessor", seq.build_preprocessor(x.loc[train_val_idx])),
                ("model", LGBMClassifier(**best["params"])),
            ]
        )
        final_pipeline.fit(x.loc[train_val_idx], y.loc[train_val_idx])
        hold_proba = final_pipeline.predict_proba(x_hold)[:, 1]
        hold_metrics = evaluate_predictions(y_hold.to_numpy(), hold_proba, best["threshold"])

        unseen_feat = unseen_versions[version]
        missing = [f for f in features if f not in unseen_feat.columns]
        if missing:
            raise KeyError(f"Tuned V{version} missing unseen features: {missing}")
        unseen_y = unseen_feat[TARGET].astype(int).to_numpy()
        unseen_proba = final_pipeline.predict_proba(unseen_feat[features])[:, 1]
        unseen_metrics = evaluate_predictions(unseen_y, unseen_proba, best["threshold"])

        model_path = model_dir / f"model_lgbm_s2_tuned_v{version}_lightgbm.pkl"
        joblib.dump(final_pipeline, model_path)
        pd.DataFrame({"feature": features}).to_csv(feature_dir / f"used_features_lgbm_s2_tuned_v{version}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "order_id": df_feat["order_id"],
                "customer_id": df_feat["customer_id"],
                **{f: df_feat[f] for f in features},
                TARGET: df_feat[TARGET],
            }
        ).to_csv(feature_dir / f"df_featured_lgbm_s2_tuned_v{version}.csv", index=False, encoding="utf-8-sig")
        joblib.dump(
            {
                "feature_names": features,
                "threshold": best["threshold"],
                "fit_indices": fit_idx,
                "validation_indices": val_idx,
                "holdout_indices": holdout_idx,
                "split_strategy": "64% fit / 16% validation / 20% holdout, stratified by is_returned",
            },
            feature_dir / f"train_validation_holdout_sets_lgbm_s2_tuned_v{version}.pkl",
        )

        row = {
            "version": f"V{version}",
            "dataset": "LightGBM_SETC_S2_TUNED_80_ATTEMPT",
            "model": "LightGBM",
            "evaluation_type": "fit_validation_holdout_tuning",
            "rows": int(len(df_feat)),
            "fit_rows": int(len(fit_idx)),
            "validation_rows": int(len(val_idx)),
            "holdout_rows": int(len(holdout_idx)),
            "feature_count": int(len(features)),
            "selected_grid_id": int(best["grid_id"]),
            "threshold": float(best["threshold"]),
            "validation_accuracy": float(best["val_metrics"]["accuracy"]),
            "validation_recall": float(best["val_metrics"]["recall"]),
            "validation_precision": float(best["val_metrics"]["precision"]),
            "validation_f1": float(best["val_metrics"]["f1"]),
            "validation_auc": float(best["val_metrics"]["auc"]),
            "holdout_accuracy": float(hold_metrics["accuracy"]),
            "holdout_recall": float(hold_metrics["recall"]),
            "holdout_precision": float(hold_metrics["precision"]),
            "holdout_f1": float(hold_metrics["f1"]),
            "holdout_auc": float(hold_metrics["auc"]),
            "holdout_cost": int(hold_metrics["cost"]),
            "unseen_accuracy": float(unseen_metrics["accuracy"]),
            "unseen_recall": float(unseen_metrics["recall"]),
            "unseen_precision": float(unseen_metrics["precision"]),
            "unseen_f1": float(unseen_metrics["f1"]),
            "unseen_auc": float(unseen_metrics["auc"]),
            "unseen_cost": int(unseen_metrics["cost"]),
            "target_80_reliable_achieved": bool(
                float(hold_metrics["accuracy"]) >= 0.80 and float(unseen_metrics["accuracy"]) >= 0.80
            ),
            "model_path": str(model_path.relative_to(ROOT)),
            "test_data_path": str(UNSEEN_CSV.relative_to(ROOT)),
        }
        rows.append(row)
        pd.DataFrame([row]).to_csv(report_dir / f"metrics_lgbm_s2_tuned_v{version}.csv", index=False, encoding="utf-8-sig")
        (model_dir / f"model_lgbm_s2_tuned_v{version}_metadata.json").write_text(
            json.dumps({**row, "lightgbm_params": best["params"]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (version_root / "README.md").write_text(
            f"""# LightGBM S2 Tuned 80% Attempt V{version}

- Feature count: `{len(features)}`
- Selected grid id: `{best['grid_id']}`
- Threshold: `{best['threshold']:.2f}`
- Holdout Accuracy: `{float(hold_metrics['accuracy']) * 100:.2f}%`
- Unseen-from-V2 Accuracy: `{float(unseen_metrics['accuracy']) * 100:.2f}%`
- Target 80% reliable achieved: `{row['target_80_reliable_achieved']}`

This model is tuned on fit/validation data only. The 20% holdout and the external unseen-from-V2 dataset are used for evaluation.
""",
            encoding="utf-8",
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(TUNED_ROOT / "lgbm_s2_tuned_80_attempt_summary.csv", index=False, encoding="utf-8-sig")
    (TUNED_ROOT / "lgbm_s2_tuned_80_attempt_summary.json").write_text(
        summary.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    draw_accuracy_chart(
        summary.rename(columns={"holdout_accuracy": "accuracy"}),
        TUNED_ROOT / "images" / "lgbm_s2_tuned_holdout_accuracy_v1_to_v5.png",
        "LightGBM S2 Tuned Holdout Accuracy V1-V5",
        value_col="accuracy",
    )
    draw_accuracy_chart(
        summary.rename(columns={"unseen_accuracy": "accuracy"}),
        TUNED_ROOT / "images" / "lgbm_s2_tuned_unseen_from_v2_accuracy_v1_to_v5.png",
        "LightGBM S2 Tuned Unseen-from-V2 Accuracy V1-V5",
        value_col="accuracy",
    )

    best_holdout = summary.loc[int(summary["holdout_accuracy"].idxmax())]
    best_unseen = summary.loc[int(summary["unseen_accuracy"].idxmax())]
    achieved = bool(summary["target_80_reliable_achieved"].any())
    (TUNED_ROOT / "README.md").write_text(
        f"""# LightGBM S2 Tuned 80% Reliable Attempt

This folder stores the new tuning/retrain attempt for the 80%+ reliable accuracy goal.

## Evaluation Design

- Train source: `docs/LightGBM/SETC/clean_dataset/clean_dataset_s2.csv`
- External test source: `docs/LightGBM/SETD/real_dataset/UNSEEN_FROM_V2/real_dataset_unseen_from_clean_dataset_v2.csv`
- Split: `64% fit / 16% validation / 20% holdout`
- Hyperparameters are selected using validation only.
- Final reporting uses untouched holdout plus external unseen-from-V2.

## Result

- Best holdout Accuracy: `{best_holdout['version']} = {float(best_holdout['holdout_accuracy']) * 100:.2f}%`
- Best unseen-from-V2 Accuracy: `{best_unseen['version']} = {float(best_unseen['unseen_accuracy']) * 100:.2f}%`
- 80% reliable target achieved on both holdout and unseen: `{achieved}`

If the target is not achieved, the honest interpretation is that current feature signal is still not strong enough for a credible 80%+ claim without using S3-style aligned benchmark data.
""",
        encoding="utf-8",
    )
    return summary


def write_master_readme(existing_s1: pd.DataFrame, existing_s2: pd.DataFrame, tuned: pd.DataFrame) -> None:
    def best_line(df: pd.DataFrame, col: str) -> str:
        row = df.loc[int(df[col].idxmax())]
        return f"{row['version']} = {float(row[col]) * 100:.2f}%"

    content = f"""# Reliable 80% Update - LightGBM

This run implements the corrected workflow after discovering that S3 produced an unrealistic 92% benchmark score.

## 1. New External Test Dataset

- Source: `data/processed/clean_dataset_v2.csv`
- Output: `docs/LightGBM/SETD/real_dataset/UNSEEN_FROM_V2/real_dataset_unseen_from_clean_dataset_v2.csv`
- Purpose: external unseen-like benchmark that is not the S1-aligned `real_dataset_s1.csv`

## 2. Existing Model Re-test

- SETC/S1 models tested on UNSEEN_FROM_V2: best external Accuracy `{best_line(existing_s1, 'external_accuracy')}`
- SETC/S2 models tested on UNSEEN_FROM_V2: best external Accuracy `{best_line(existing_s2, 'external_accuracy')}`

## 3. Tuned/Retrained LightGBM Attempt

- Output folder: `docs/LightGBM/SETC/clean_dataset/S2_TUNED_80_ATTEMPT`
- Best tuned holdout Accuracy: `{best_line(tuned, 'holdout_accuracy')}`
- Best tuned unseen-from-V2 Accuracy: `{best_line(tuned, 'unseen_accuracy')}`
- 80% reliable target achieved on both holdout and unseen: `{bool(tuned['target_80_reliable_achieved'].any())}`

## Interpretation

The new run keeps 80%+ as the target, but does not force the test data to match train. If the tuned models stay below 80% on holdout/unseen, that is a real signal that the dataset needs stronger business features or true company data to support a credible 80%+ model.
"""
    (UNSEEN_ROOT / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    seq = load_seq_module()
    raw_unseen = export_unseen_dataset()
    distribution_compare(raw_unseen)
    existing_s1 = evaluate_existing_models(seq, raw_unseen, "s1")
    existing_s2 = evaluate_existing_models(seq, raw_unseen, "s2")
    tuned = train_tuned_s2(seq, raw_unseen)
    write_master_readme(existing_s1, existing_s2, tuned)

    print("\nExisting SETC/S1 on UNSEEN_FROM_V2")
    print(existing_s1[["version", "feature_count", "holdout_accuracy", "external_accuracy", "external_recall", "external_f1", "external_auc", "external_cost"]].to_string(index=False))
    print("\nExisting SETC/S2 on UNSEEN_FROM_V2")
    print(existing_s2[["version", "feature_count", "holdout_accuracy", "external_accuracy", "external_recall", "external_f1", "external_auc", "external_cost"]].to_string(index=False))
    print("\nTuned S2 80% attempt")
    print(tuned[["version", "feature_count", "holdout_accuracy", "unseen_accuracy", "holdout_recall", "unseen_recall", "holdout_auc", "unseen_auc", "target_80_reliable_achieved"]].to_string(index=False))


if __name__ == "__main__":
    main()
