from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "docs" / "LightGBM"

PAIRS = [
    {
        "pair": "S1_vs_S3",
        "clean_label": "SETC/S1 clean 5,000 holdout",
        "real_label": "SETD/S3 real 55,000 full test",
        "clean_path": EXP_DIR / "SETC" / "clean_dataset" / "S1" / "lgbm_s1_v1_to_v5_holdout_summary.csv",
        "real_path": EXP_DIR / "SETD" / "real_dataset" / "S3" / "lgbm_s1_v1_to_v5_external_summary.csv",
    },
    {
        "pair": "S2_vs_S4",
        "clean_label": "SETC/S2 clean 50,000 holdout",
        "real_label": "SETD/S4 real 105,000 full test",
        "clean_path": EXP_DIR / "SETC" / "clean_dataset" / "S2" / "lgbm_s2_v1_to_v5_holdout_summary.csv",
        "real_path": EXP_DIR / "SETD" / "real_dataset" / "S4" / "lgbm_s2_v1_to_v5_external_summary.csv",
    },
]

VERSION_EXPLANATIONS = {
    "V1": "Baseline with simpler feature set.",
    "V2": "Adds stronger customer/history and engineered behavior features.",
    "V3": "Adds broader risk features and selected interactions.",
    "V4": "High-signal best candidate with the strongest combined feature set in this benchmark.",
    "V5": "Compact/reduced candidate used to check whether fewer features can keep performance close.",
}


def pct(value: float) -> float:
    return round(float(value) * 100, 2)


def stability_label(abs_gap_pp: float) -> str:
    if abs_gap_pp <= 1.0:
        return "Very close"
    if abs_gap_pp <= 3.0:
        return "Close"
    if abs_gap_pp <= 5.0:
        return "Moderate gap"
    return "Large gap"


def metric_column(row: pd.Series, base_name: str) -> str:
    if base_name in row.index:
        return base_name
    clean_name = f"{base_name}_clean"
    if clean_name in row.index:
        return clean_name
    raise KeyError(base_name)


def build_comparison() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in PAIRS:
        clean = pd.read_csv(item["clean_path"])
        real = pd.read_csv(item["real_path"])
        merged = clean.merge(real, on="version", suffixes=("_clean", "_real"))

        for _, row in merged.iterrows():
            version = row["version"]
            clean_acc = pct(row[metric_column(row, "holdout_accuracy")])
            real_acc = pct(row["external_accuracy"])
            gap = round(real_acc - clean_acc, 2)
            abs_gap = round(abs(gap), 2)
            rows.append(
                {
                    "comparison_pair": item["pair"],
                    "version": version,
                    "feature_strategy": row.get("feature_strategy_clean", row.get("feature_strategy", "")),
                    "feature_count_clean_model": int(row[metric_column(row, "feature_count")]),
                    "clean_dataset": item["clean_label"],
                    "real_dataset": item["real_label"],
                    "clean_holdout_accuracy_pct": clean_acc,
                    "real_external_accuracy_pct": real_acc,
                    "accuracy_gap_real_minus_clean_pp": gap,
                    "absolute_gap_pp": abs_gap,
                    "stability": stability_label(abs_gap),
                    "clean_holdout_recall_pct": pct(row[metric_column(row, "holdout_recall")]),
                    "real_external_recall_pct": pct(row["external_recall"]),
                    "clean_holdout_f1_pct": pct(row[metric_column(row, "holdout_f1")]),
                    "real_external_f1_pct": pct(row["external_f1"]),
                    "clean_holdout_auc_pct": pct(row[metric_column(row, "holdout_auc")]),
                    "real_external_auc_pct": pct(row["external_auc"]),
                    "interpretation": VERSION_EXPLANATIONS.get(version, ""),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in df[columns].iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, divider, *rows])


def save_report(df: pd.DataFrame, out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary_cols = [
        "comparison_pair",
        "version",
        "feature_count_clean_model",
        "clean_holdout_accuracy_pct",
        "real_external_accuracy_pct",
        "accuracy_gap_real_minus_clean_pp",
        "absolute_gap_pp",
        "stability",
    ]
    best_by_stability = (
        df.sort_values(["absolute_gap_pp", "real_external_accuracy_pct"], ascending=[True, False])
        .groupby("comparison_pair")
        .head(1)
    )
    best_by_real_acc = (
        df.sort_values(["real_external_accuracy_pct", "absolute_gap_pp"], ascending=[False, True])
        .groupby("comparison_pair")
        .head(1)
    )

    lines = [
        "# LightGBM High-Signal SETC vs SETD Accuracy Stability Comparison",
        "",
        "This report compares the same LightGBM high-signal model version between clean holdout data and real external full-dataset data.",
        "",
        "- S1_vs_S3 compares SETC/S1 clean 5,000 holdout with SETD/S3 real 55,000 full test.",
        "- S2_vs_S4 compares SETC/S2 clean 50,000 holdout with SETD/S4 real 105,000 full test.",
        "",
        "This is the high-signal benchmark, so accuracy is expected to be higher than the normal-signal benchmark.",
        "",
        "## Summary Table",
        "",
        markdown_table(df, summary_cols),
        "",
        "## Best By Stability",
        "",
        markdown_table(best_by_stability, summary_cols),
        "",
        "## Best By Real External Accuracy",
        "",
        markdown_table(best_by_real_acc, summary_cols),
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def save_chart(df: pd.DataFrame, out_svg: Path) -> None:
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    versions = ["V1", "V2", "V3", "V4", "V5"]
    width, height = 1600, 720
    margin_top, margin_bottom = 105, 90
    panel_gap = 80
    panel_w = (width - 140 - panel_gap) / 2
    panel_h = height - margin_top - margin_bottom
    y_min, y_max = 65, 90
    clean_color = "#245B8A"
    real_color = "#C97018"
    grid_color = "#DADDE1"
    text_color = "#222222"

    def y_pos(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * panel_h

    def panel_x(idx: int) -> float:
        return 70 + idx * (panel_w + panel_gap)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="45" text-anchor="middle" font-family="Arial" font-size="30" font-weight="700" fill="{text_color}">LightGBM High-Signal Accuracy Stability</text>',
        f'<rect x="{width - 420}" y="25" width="18" height="18" fill="{clean_color}"/>',
        f'<text x="{width - 395}" y="40" font-family="Arial" font-size="16" fill="{text_color}">Clean holdout</text>',
        f'<rect x="{width - 250}" y="25" width="18" height="18" fill="{real_color}"/>',
        f'<text x="{width - 225}" y="40" font-family="Arial" font-size="16" fill="{text_color}">Real external</text>',
    ]

    for p_idx, pair in enumerate(["S1_vs_S3", "S2_vs_S4"]):
        sub = df[df["comparison_pair"] == pair].set_index("version").loc[versions]
        left = panel_x(p_idx)
        right = left + panel_w
        title = "SETC/S1 clean 5,000 vs SETD/S3 real 55,000" if pair == "S1_vs_S3" else "SETC/S2 clean 50,000 vs SETD/S4 real 105,000"
        svg.append(f'<text x="{left + panel_w / 2}" y="83" text-anchor="middle" font-family="Arial" font-size="21" font-weight="700" fill="{text_color}">{escape(title)}</text>')

        for tick in [65, 70, 75, 80, 85, 90]:
            y = y_pos(tick)
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{grid_color}" stroke-width="1"/>')
            svg.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#555555">{tick}%</text>')

        group_w = panel_w / len(versions)
        bar_w = 35
        for i, version in enumerate(versions):
            row = sub.loc[version]
            center = left + group_w * i + group_w / 2
            clean_val = float(row["clean_holdout_accuracy_pct"])
            real_val = float(row["real_external_accuracy_pct"])
            gap = real_val - clean_val
            for val, offset, color in [(clean_val, -bar_w / 1.7, clean_color), (real_val, bar_w / 1.7, real_color)]:
                bar_h = y_pos(y_min) - y_pos(val)
                x = center + offset - bar_w / 2
                y = y_pos(val)
                svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="4" fill="{color}"/>')
                svg.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700" fill="{text_color}">{val:.2f}%</text>')
            svg.append(f'<text x="{center:.1f}" y="{height - 58}" text-anchor="middle" font-family="Arial" font-size="17" font-weight="700" fill="{text_color}">{version}</text>')
            svg.append(f'<text x="{center:.1f}" y="{height - 34}" text-anchor="middle" font-family="Arial" font-size="13" fill="#555555">gap {gap:+.2f} pp</text>')

        svg.append(f'<line x1="{left}" y1="{y_pos(y_min):.1f}" x2="{right}" y2="{y_pos(y_min):.1f}" stroke="#333333" stroke-width="1.5"/>')

    svg.append("</svg>")
    out_svg.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    df = build_comparison()
    out_dir = EXP_DIR / "comparison_outputs"
    save_report(
        df,
        out_dir / "lightgbm_high_signal_setc_setd_accuracy_stability_comparison.csv",
        out_dir / "lightgbm_high_signal_setc_setd_accuracy_stability_comparison.md",
    )
    save_chart(df, out_dir / "images" / "lightgbm_high_signal_setc_setd_accuracy_stability_comparison.svg")


if __name__ == "__main__":
    main()
