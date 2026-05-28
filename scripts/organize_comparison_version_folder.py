from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
COMPARISON_DIR = DOCS / "Comparison Version"
METRICS_PATH = COMPARISON_DIR / "version_1_to_4_selected_model_comparison.csv"


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fmt_pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def fmt_num(value: float) -> str:
    return f"{float(value):,.0f}"


def version_configs() -> list[dict]:
    return [
        {
            "key": "V1",
            "folder": "01_Version_1_Baseline_XGBoost",
            "title": "Version 1 - Baseline XGBoost",
            "why": "ใช้เป็น baseline เพื่อวัดผล feature engineering ชุดแรกจาก clean_dataset.csv",
            "result": "Accuracy ใช้ได้ระดับเริ่มต้น แต่ Recall ต่ำ จึงยังพลาด order ที่มีแนวโน้มคืนสินค้าเยอะ",
            "files": [
                (
                    DOCS / "version 1" / "feature_documentation" / "v1_feature_used_unused_audit.pdf",
                    "01_feature_used_unused_audit_v1.pdf",
                ),
                (
                    DOCS / "version 1" / "feature_documentation" / "v1_feature_engineering_process.pdf",
                    "02_feature_engineering_process_v1.pdf",
                ),
                (
                    DOCS / "version 1" / "feature_documentation" / "v1_feature_used_unused_audit.csv",
                    "csv/v1_feature_used_unused_audit.csv",
                ),
                (
                    DOCS / "version 1" / "feature_documentation" / "used_features_current.csv",
                    "csv/used_features_current.csv",
                ),
                (
                    DOCS / "version 1" / "feature_documentation" / "dropped_or_not_used_features_current.csv",
                    "csv/dropped_or_not_used_features_current.csv",
                ),
                (DOCS / "version 1" / "feature_engineering.py", "code/feature_engineering.py"),
                (DOCS / "version 1" / "model_training.py", "code/model_training.py"),
                (DOCS / "version 1" / "model_evaluation.py", "code/model_evaluation.py"),
            ],
        },
        {
            "key": "V2",
            "folder": "02_Version_2_XGBoost_Safe_Rolling_HIGH_ACCURACY",
            "title": "Version 2 - XGBoost Safe Rolling HIGH_ACCURACY",
            "why": "เลือกเป็น candidate หลัก เพราะใช้ rolling customer history, business insight features และตัด leakage/post-event fields",
            "result": "Performance สูงสุดในชุดเปรียบเทียบ: Accuracy, Recall, F1 และ AUC ดีที่สุด",
            "files": [
                (
                    DOCS / "version 2" / "feature_documentation" / "v2_feature_used_unused_audit.pdf",
                    "01_feature_used_unused_audit_v2.pdf",
                ),
                (
                    DOCS / "version 2" / "feature_documentation" / "v2_feature_engineering_process.pdf",
                    "02_feature_engineering_process_v2.pdf",
                ),
                (
                    DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "docs" / "model_report_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.md",
                    "03_model_report_v2_HIGH_ACCURACY.md",
                ),
                (
                    DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "docs" / "eda_insight_summary_v2_NEW.md",
                    "04_eda_insight_summary_v2.md",
                ),
                (
                    DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "README.md",
                    "05_package_README_v2_HIGH_ACCURACY.md",
                ),
                (
                    DOCS / "version 2" / "feature_documentation" / "v2_feature_used_unused_audit.csv",
                    "csv/v2_feature_used_unused_audit.csv",
                ),
                (
                    DOCS / "version 2" / "feature_documentation" / "used_features_current.csv",
                    "csv/used_features_current.csv",
                ),
                (
                    DOCS / "version 2" / "feature_documentation" / "dropped_or_not_used_features_current.csv",
                    "csv/dropped_or_not_used_features_current.csv",
                ),
                (
                    DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "reports" / "metrics_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv",
                    "csv/metrics_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv",
                ),
                (
                    DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "reports" / "feature_importance_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv",
                    "csv/feature_importance_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv",
                ),
                (
                    DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "data" / "used_features_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv",
                    "csv/used_features_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv",
                ),
                (
                    DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "scripts" / "feature_engineered_v2_HIGH_ACCURACY.py",
                    "code/feature_engineered_v2_HIGH_ACCURACY.py",
                ),
                (
                    DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "images" / "metrics_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.png",
                    "images/metrics_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.png",
                ),
                (
                    DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "images" / "eda_return_rate_by_history.png",
                    "images/eda_return_rate_by_history.png",
                ),
                (
                    DOCS / "version 2" / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY" / "images" / "eda_return_rate_by_category.png",
                    "images/eda_return_rate_by_category.png",
                ),
            ],
        },
        {
            "key": "V3",
            "folder": "03_Version_3_Stacking_Model",
            "title": "Version 3 - Stacking Model",
            "why": "ทดลองเพิ่มความซับซ้อนของ model โดยใช้ feature base ใกล้ V2 แล้ว ensemble หลายโมเดล",
            "result": "Recall ดีขึ้นเมื่อเทียบกับ V1 แต่ Accuracy/F1/AUC ยังไม่ชนะ V2 และดูแล production ยากกว่า",
            "files": [
                (
                    DOCS / "version 3" / "feature_documentation" / "v3_feature_used_unused_audit.pdf",
                    "01_feature_used_unused_audit_v3.pdf",
                ),
                (
                    DOCS / "version 3" / "feature_documentation" / "v3_feature_engineering_process.pdf",
                    "02_feature_engineering_process_v3.pdf",
                ),
                (
                    DOCS / "version 3" / "stacking_model_v3" / "docs" / "v3_stacking_model_process_report.pdf",
                    "03_stacking_model_process_report_v3.pdf",
                ),
                (
                    DOCS / "version 3" / "feature_documentation" / "v3_feature_used_unused_audit.csv",
                    "csv/v3_feature_used_unused_audit.csv",
                ),
                (
                    DOCS / "version 3" / "feature_documentation" / "used_features_current.csv",
                    "csv/used_features_current.csv",
                ),
                (
                    DOCS / "version 3" / "feature_documentation" / "dropped_or_not_used_features_current.csv",
                    "csv/dropped_or_not_used_features_current.csv",
                ),
                (
                    DOCS / "version 3" / "stacking_model_v3" / "reports" / "metrics_summary_v3.csv",
                    "csv/metrics_summary_v3.csv",
                ),
                (
                    DOCS / "version 3" / "stacking_model_v3" / "data" / "v3_used_features.csv",
                    "csv/v3_used_features.csv",
                ),
                (
                    DOCS / "version 3" / "stacking_model_v3" / "scripts" / "model_training_v3_stacking.py",
                    "code/model_training_v3_stacking.py",
                ),
                (
                    DOCS / "version 3" / "stacking_model_v3" / "scripts" / "model_evaluation_v3.py",
                    "code/model_evaluation_v3.py",
                ),
                (
                    DOCS / "version 3" / "stacking_model_v3" / "images" / "accuracy_recall_tradeoff.png",
                    "images/accuracy_recall_tradeoff.png",
                ),
            ],
        },
        {
            "key": "V4",
            "folder": "04_Version_4_Generated_SMOTE_Optuna",
            "title": "Version 4 - Generated Data + SMOTE + Optuna",
            "why": "ทดลองเพิ่ม data และจัดการ imbalance ด้วย SMOTE พร้อม tuning ด้วย Optuna",
            "result": "Accuracy สูงกว่า V1/V3 แต่ Recall/F1 ยังไม่ชนะ V2 และ feature volume สูงกว่า",
            "files": [
                (
                    DOCS / "version 4" / "feature_documentation" / "v4_feature_used_unused_audit.pdf",
                    "01_feature_used_unused_audit_v4.pdf",
                ),
                (
                    DOCS / "version 4" / "feature_documentation" / "v4_feature_engineering_process.pdf",
                    "02_feature_engineering_process_v4.pdf",
                ),
                (
                    DOCS / "version 4" / "docs" / "v4_generated_end_to_end_report.pdf",
                    "03_generated_end_to_end_report_v4.pdf",
                ),
                (
                    DOCS / "version 4" / "feature_documentation" / "v4_feature_used_unused_audit.csv",
                    "csv/v4_feature_used_unused_audit.csv",
                ),
                (
                    DOCS / "version 4" / "feature_documentation" / "used_features_current.csv",
                    "csv/used_features_current.csv",
                ),
                (
                    DOCS / "version 4" / "feature_documentation" / "dropped_or_not_used_features_current.csv",
                    "csv/dropped_or_not_used_features_current.csv",
                ),
                (
                    DOCS / "version 4" / "reports" / "model_evaluation" / "v4_generated_model_metrics.csv",
                    "csv/v4_generated_model_metrics.csv",
                ),
                (
                    DOCS / "version 4" / "reports" / "model_evaluation" / "v4_used_features.csv",
                    "csv/v4_used_features.csv",
                ),
                (
                    DOCS / "version 4" / "reports" / "model_evaluation" / "v4_generated_shap_feature_importance.csv",
                    "csv/v4_generated_shap_feature_importance.csv",
                ),
                (
                    DOCS / "version 4" / "scripts" / "run_v4_generated_end_to_end_pipeline.py",
                    "code/run_v4_generated_end_to_end_pipeline.py",
                ),
                (
                    DOCS / "version 4" / "model_evaluation_v4_generated" / "v4_generated_model_metrics.png",
                    "images/v4_generated_model_metrics.png",
                ),
                (
                    DOCS / "version 4" / "model_evaluation_v4_generated" / "v4_generated_shap_summary.png",
                    "images/v4_generated_shap_summary.png",
                ),
            ],
        },
    ]


def organize_overall_folder() -> None:
    overall = COMPARISON_DIR / "00_Overall_Comparison"
    copies = [
        (COMPARISON_DIR / "version_1_to_4_detailed_comparison.pdf", "version_1_to_4_detailed_comparison.pdf"),
        (COMPARISON_DIR / "version_1_to_4_selected_model_comparison.csv", "version_1_to_4_selected_model_comparison.csv"),
        (COMPARISON_DIR / "images" / "version_1_to_4_performance_metrics.png", "images/version_1_to_4_performance_metrics.png"),
        (COMPARISON_DIR / "images" / "version_1_to_4_cost_comparison.png", "images/version_1_to_4_cost_comparison.png"),
        (COMPARISON_DIR / "images" / "version_1_to_4_feature_count.png", "images/version_1_to_4_feature_count.png"),
    ]
    for source, dest in copies:
        copy_if_exists(source, overall / dest)

    write_text(
        overall / "README.md",
        "\n".join(
            [
                "# Overall Comparison",
                "",
                "โฟลเดอร์นี้เป็นภาพรวมสำหรับเทียบ Version 1-4 ทั้งหมด",
                "",
                "## Files",
                "- `version_1_to_4_detailed_comparison.pdf` - เอกสาร PDF เปรียบเทียบรวม",
                "- `version_1_to_4_selected_model_comparison.csv` - ตาราง metric รวม",
                "- `images/version_1_to_4_performance_metrics.png` - Accuracy / Recall / F1 / AUC",
                "- `images/version_1_to_4_cost_comparison.png` - Cost Matrix",
                "- `images/version_1_to_4_feature_count.png` - จำนวน feature ต่อ version",
                "",
                "สรุป: V2 เป็น candidate หลักในสถานะล่าสุด เพราะ Accuracy/Recall/F1/AUC ดีที่สุดและ feature logic ปลอด leakage มากที่สุด",
                "",
                f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
            ]
        ),
    )


def organize_version_folders(metrics: pd.DataFrame) -> None:
    for cfg in version_configs():
        folder = COMPARISON_DIR / cfg["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        metric = metrics[metrics["display_version"].eq(cfg["key"])].iloc[0].to_dict()

        copied_rows = []
        for source, rel_dest in cfg["files"]:
            destination = folder / rel_dest
            copied = copy_if_exists(source, destination)
            copied_rows.append(
                {
                    "source": str(source.relative_to(ROOT)) if source.exists() else str(source.relative_to(ROOT)),
                    "destination": rel_dest,
                    "copied": copied,
                }
            )

        pd.DataFrame([metric]).to_csv(folder / f"{cfg['key'].lower()}_metrics_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(copied_rows).to_csv(folder / "file_manifest.csv", index=False, encoding="utf-8-sig")

        write_text(
            folder / "README.md",
            "\n".join(
                [
                    f"# {cfg['title']}",
                    "",
                    "## Role in Comparison",
                    cfg["why"],
                    "",
                    "## Result Summary",
                    cfg["result"],
                    "",
                    "## Metrics",
                    f"- Model: `{metric['model']}`",
                    f"- Dataset: `{metric['dataset']}`",
                    f"- Feature count: `{int(metric['feature_count'])}`",
                    f"- Accuracy: `{fmt_pct(metric['accuracy'])}`",
                    f"- Recall: `{fmt_pct(metric['recall'])}`",
                    f"- Precision: `{fmt_pct(metric['precision'])}`",
                    f"- F1: `{fmt_pct(metric['f1'])}`",
                    f"- AUC: `{fmt_pct(metric['auc'])}`",
                    f"- Cost: `{fmt_num(metric['cost'])}`",
                    f"- Rating: `{metric['rating']}`",
                    "",
                    "## Folder Layout",
                    "- `01_feature_used_unused_audit_*.pdf` - feature ที่ใช้/ไม่ใช้/ตัดทิ้ง",
                    "- `02_feature_engineering_process_*.pdf` - ขั้นตอน feature engineering",
                    "- `csv/` - metric, used features, dropped features, feature importance",
                    "- `code/` - script หรือ code ที่เกี่ยวข้อง",
                    "- `images/` - กราฟหรือภาพประเมินผลของ version นั้น",
                    "",
                    f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
                ]
            ),
        )


def write_top_readme() -> None:
    write_text(
        COMPARISON_DIR / "README.md",
        "\n".join(
            [
                "# Comparison Version",
                "",
                "โฟลเดอร์นี้ถูกจัดใหม่ให้แยก Version 1-4 ชัดเจน พร้อมส่วนภาพรวมสำหรับเทียบ performance รวม",
                "",
                "## Folder Structure",
                "- `00_Overall_Comparison/` - เอกสารและกราฟเปรียบเทียบรวม V1-V4",
                "- `01_Version_1_Baseline_XGBoost/` - เอกสาร/feature/code ของ V1",
                "- `02_Version_2_XGBoost_Safe_Rolling_HIGH_ACCURACY/` - เอกสาร/feature/code/metrics ของ V2 candidate หลัก",
                "- `03_Version_3_Stacking_Model/` - เอกสาร/feature/code ของ V3 stacking",
                "- `04_Version_4_Generated_SMOTE_Optuna/` - เอกสาร/feature/code ของ V4 generated data",
                "",
                "## Current Decision",
                "เลือก `Version 2 - XGBoost Safe Rolling HIGH_ACCURACY` เป็น candidate หลัก เพราะได้ performance สูงสุดและมี feature logic แบบ order-time safe/rolling history ที่สอดคล้องกับโจทย์ return-risk มากที่สุด",
                "",
                "ไฟล์เดิมระดับบนยังคงอยู่เพื่อไม่ให้ reference เก่าเสีย แต่ไฟล์ที่จัดระเบียบสำหรับอ่านจริงอยู่ใน subfolder ด้านบน",
                "",
                f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
            ]
        ),
    )


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_PATH}")

    metrics = pd.read_csv(METRICS_PATH)
    organize_overall_folder()
    organize_version_folders(metrics)
    write_top_readme()

    print("Organized Comparison Version folder")
    for path in [
        COMPARISON_DIR / "00_Overall_Comparison",
        COMPARISON_DIR / "01_Version_1_Baseline_XGBoost",
        COMPARISON_DIR / "02_Version_2_XGBoost_Safe_Rolling_HIGH_ACCURACY",
        COMPARISON_DIR / "03_Version_3_Stacking_Model",
        COMPARISON_DIR / "04_Version_4_Generated_SMOTE_Optuna",
    ]:
        print(path)


if __name__ == "__main__":
    main()
