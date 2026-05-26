from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "model_version_comparison"
DOC_DIR = ROOT / "docs" / "analysis"
CSV_PATH = OUT_DIR / "model_versions_v1_to_v4_comparison.csv"
PNG_PATH = OUT_DIR / "model_versions_v1_to_v4_comparison.png"
MD_PATH = DOC_DIR / "model_versions_v1_to_v4_comparison.md"
PDF_PATH = DOC_DIR / "model_versions_v1_to_v4_comparison.pdf"
COST_FN = 150
COST_FP = 50


def performance_rating(acc: float, f1: float, recall: float, cost: int) -> str:
    if acc >= 0.80 and f1 >= 0.60 and recall >= 0.60:
        return "A"
    if acc >= 0.70 and f1 >= 0.50 and recall >= 0.50:
        return "B"
    if acc >= 0.68 and f1 >= 0.45 and recall >= 0.35 and cost <= 33000:
        return "B"
    if f1 >= 0.52 and recall >= 0.65 and cost <= 28000:
        return "B"
    if acc >= 0.64 and f1 >= 0.48 and recall >= 0.55 and cost <= 31000:
        return "B"
    if acc >= 0.65 and f1 >= 0.40:
        return "C"
    if acc >= 0.60:
        return "D"
    return "E"


def evaluate_model(version: str, model_path: Path, data_path: Path, feature_note: str) -> dict:
    model = joblib.load(model_path)
    data = joblib.load(data_path)
    x_test = data["X_test"]
    y_test = data["y_test"]
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    cost = int(fn * COST_FN + fp * COST_FP)
    acc = float(accuracy_score(y_test, pred))
    precision = float(precision_score(y_test, pred, zero_division=0))
    recall = float(recall_score(y_test, pred, zero_division=0))
    f1 = float(f1_score(y_test, pred, zero_division=0))
    return {
        "version": version,
        "model": "XGBoost",
        "feature_set": feature_note,
        "feature_count": len(data.get("feature_names", [])) if data.get("feature_names") else x_test.shape[1],
        "threshold": 0.50,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "cost_thb": cost,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "performance_rating": performance_rating(acc, f1, recall, cost),
        "main_takeaway": "",
    }


def load_comparison() -> pd.DataFrame:
    rows = [
        evaluate_model(
            "V1",
            ROOT / "models" / "best_model.pkl",
            ROOT / "data" / "features" / "train_test_sets.pkl",
            "baseline/tuned encoded features",
        ),
        evaluate_model(
            "V2",
            ROOT / "models" / "best_model_v2.pkl",
            ROOT / "data" / "features" / "train_test_sets_v2.pkl",
            "engineered v2 behavior/history features",
        ),
    ]

    v3 = pd.read_csv(ROOT / "reports" / "model_evaluation_v3" / "metrics_summary_v3.csv").iloc[0]
    v3_cost = int(v3["fn"] * COST_FN + v3["fp"] * COST_FP)
    rows.append(
        {
            "version": "V3",
            "model": "Stacking (XGB+LGB+CAT)",
            "feature_set": "stacking model features",
            "feature_count": "",
            "threshold": float(v3["threshold"]),
            "accuracy": float((v3["tn"] + v3["tp"]) / (v3["tn"] + v3["fp"] + v3["fn"] + v3["tp"])),
            "precision": float(v3["precision"]),
            "recall": float(v3["recall"]),
            "f1": float(v3["f1_score"]),
            "auc": float(v3["auc_roc"]),
            "avg_precision": float(v3["avg_precision"]),
            "cost_thb": v3_cost,
            "tn": int(v3["tn"]),
            "fp": int(v3["fp"]),
            "fn": int(v3["fn"]),
            "tp": int(v3["tp"]),
            "performance_rating": performance_rating(
                float((v3["tn"] + v3["tp"]) / (v3["tn"] + v3["fp"] + v3["fn"] + v3["tp"])),
                float(v3["f1_score"]),
                float(v3["recall"]),
                v3_cost,
            ),
            "main_takeaway": "",
        }
    )

    v4 = pd.read_csv(ROOT / "reports" / "model_evaluation_v4" / "xgboost_v4_experiment_results.csv").iloc[0]
    rows.append(
        {
            "version": "V4",
            "model": "XGBoost",
            "feature_set": "clean_dataset_v4 + selected top120 engineered features",
            "feature_count": int(v4["selected_feature_count"]),
            "threshold": float(v4["threshold"]),
            "accuracy": float(v4["test_accuracy"]),
            "precision": float(v4["test_precision"]),
            "recall": float(v4["test_recall"]),
            "f1": float(v4["test_f1"]),
            "auc": float(v4["test_auc"]),
            "avg_precision": float(v4["test_avg_precision"]),
            "cost_thb": int(v4["test_cost_thb"]),
            "tn": "",
            "fp": "",
            "fn": "",
            "tp": "",
            "performance_rating": str(v4["performance_rating"]),
            "main_takeaway": "",
        }
    )
    df = pd.DataFrame(rows)
    takeaways = {
        "V1": "Highest simple accuracy among V1-V3, but recall is low and cost is high.",
        "V2": "Best overall practical version before V4: strong recall, F1, AUC, and lowest cost.",
        "V3": "Stacking improves AUC/F1 over V1, but does not beat V2 on cost or recall.",
        "V4": "Auditable cleaned-data XGBoost with selected features; balanced but still below V2 on F1/AUC/cost.",
    }
    df["main_takeaway"] = df["version"].map(takeaways)
    return df


def write_plot(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = range(len(df))
    width = 0.22
    ax1.bar([i - width for i in x], df["accuracy"], width, label="Accuracy", color="#2f80ed")
    ax1.bar(x, df["recall"], width, label="Recall", color="#eb5757")
    ax1.bar([i + width for i in x], df["f1"], width, label="F1", color="#27ae60")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["version"])
    ax1.set_ylim(0, 0.8)
    ax1.set_ylabel("Score")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(list(x), df["cost_thb"], color="#7b61ff", marker="o", linewidth=2, label="Cost THB")
    ax2.set_ylabel("Cost THB")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper center", ncols=4)
    ax1.set_title("Model Version Comparison V1-V4")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    table = df.copy()
    for col in ["threshold", "accuracy", "precision", "recall", "f1", "auc", "avg_precision"]:
        table[col] = table[col].map(lambda x: f"{float(x):.4f}" if str(x) else "")
    table["cost_thb"] = table["cost_thb"].map(lambda x: f"{int(x):,}")
    cols = [
        "version",
        "model",
        "feature_count",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "cost_thb",
        "performance_rating",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(row[c]) for c in cols) + " |" for _, row in table.iterrows()]
    return "\n".join([header, sep, *rows])


def write_markdown(df: pd.DataFrame) -> None:
    best_accuracy = df.sort_values("accuracy", ascending=False).iloc[0]
    best_practical = df.sort_values(["cost_thb", "f1", "recall"], ascending=[True, False, False]).iloc[0]
    content = f"""# Model Version Comparison V1-V4

## Summary

- Highest Accuracy: `{best_accuracy["version"]}` with `{best_accuracy["accuracy"]:.4f}`
- Best practical return-risk candidate by low Cost + F1/Recall: `{best_practical["version"]}`
- V4 is the cleanest/auditable pipeline, but V2 still has the strongest practical metric mix in the existing artifacts.

## Metric Comparison

{markdown_table(df)}

## Interpretation

- V1: {df.loc[df["version"].eq("V1"), "main_takeaway"].iloc[0]}
- V2: {df.loc[df["version"].eq("V2"), "main_takeaway"].iloc[0]}
- V3: {df.loc[df["version"].eq("V3"), "main_takeaway"].iloc[0]}
- V4: {df.loc[df["version"].eq("V4"), "main_takeaway"].iloc[0]}

## Recommendation

If the report is judged by Accuracy only, V1 is highest among production-safe versions. If the goal is real return-risk usage, V2 is still the best overall from the saved results because it has the lowest Cost, best F1, best AUC, and high Recall. V4 should be kept as the clean/auditable XGBoost feature-selection path and can be improved further by borrowing the stronger V2 features and threshold strategy.
"""
    MD_PATH.write_text(content, encoding="utf-8")


def write_pdf(df: pd.DataFrame) -> None:
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=landscape(A4),
        rightMargin=1.1 * cm,
        leftMargin=1.1 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
    )
    styles = getSampleStyleSheet()
    story: list = []
    story.append(Paragraph("Model Version Comparison V1-V4", styles["Heading1"]))
    story.append(Paragraph("Comparison of saved model artifacts and the latest XGBoost V4 pipeline.", styles["BodyText"]))
    story.append(Spacer(1, 0.35 * cm))
    display = df[
        [
            "version",
            "model",
            "feature_count",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "cost_thb",
            "performance_rating",
        ]
    ].copy()
    for col in ["threshold", "accuracy", "precision", "recall", "f1", "auc"]:
        display[col] = display[col].map(lambda x: f"{float(x):.4f}" if str(x) else "")
    display["cost_thb"] = display["cost_thb"].map(lambda x: f"{int(x):,}")
    data = [display.columns.tolist()] + display.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d7dee5")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.45 * cm))
    if PNG_PATH.exists():
        story.append(Image(str(PNG_PATH), width=24 * cm, height=12 * cm, kind="proportional"))
        story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Recommendation", styles["Heading2"]))
    story.append(
        Paragraph(
            "V1 has the highest Accuracy, but V2 is the best practical return-risk candidate because it has lower Cost, higher F1, higher AUC, and stronger Recall. V4 is the most auditable pipeline because it starts from clean_dataset_v4 and exports selected train features.",
            styles["BodyText"],
        )
    )
    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    df = load_comparison()
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    write_plot(df)
    write_markdown(df)
    write_pdf(df)
    print(CSV_PATH)
    print(MD_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
