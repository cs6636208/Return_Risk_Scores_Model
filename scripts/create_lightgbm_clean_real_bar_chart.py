from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "LightGBM" / "comparison_outputs" / "lightgbm_high_signal_setc_setd_accuracy_stability_comparison.csv"
OUT_DIR = ROOT / "docs" / "LightGBM" / "comparison_outputs" / "images"
OUT_PATH = OUT_DIR / "lightgbm_clean_vs_real_accuracy_bar_chart.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\tahomabd.ttf") if bold else Path(r"C:\Windows\Fonts\tahoma.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    s1_s3 = [row for row in rows if row["comparison_pair"] == "S1_vs_S3"]
    s2_s4 = [row for row in rows if row["comparison_pair"] == "S2_vs_S4"]
    order = {"V1": 1, "V2": 2, "V3": 3, "V4": 4, "V5": 5}
    return sorted(s1_s3, key=lambda r: order[r["version"]]), sorted(s2_s4, key=lambda r: order[r["version"]])


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, str]],
    box: tuple[int, int, int, int],
    title: str,
    clean_label: str,
    real_label: str,
) -> None:
    x0, y0, x1, y1 = box
    title_font = font(34, bold=True)
    axis_font = font(22)
    label_font = font(24, bold=True)
    value_font = font(21, bold=True)
    note_font = font(19)

    draw.text(((x0 + x1) // 2, y0), title, font=title_font, fill="#111111", anchor="ma")

    plot_left = x0 + 85
    plot_right = x1 - 35
    plot_top = y0 + 115
    plot_bottom = y1 - 75
    baseline = plot_bottom
    max_value = 90.0
    min_value = 78.0
    range_value = max_value - min_value

    # Background and grid.
    draw.rounded_rectangle((x0, y0 + 45, x1, y1), radius=18, fill="#FFFFFF", outline="#D8DEE9", width=2)
    for pct in [78, 80, 82, 84, 86, 88, 90]:
        y = baseline - int(((pct - min_value) / range_value) * (plot_bottom - plot_top))
        draw.line((plot_left, y, plot_right, y), fill="#E6EAF0", width=2)
        draw.text((plot_left - 12, y), f"{pct}%", font=axis_font, fill="#4B5563", anchor="rm")
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#333333", width=3)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#333333", width=3)

    clean_color = "#2563EB"
    real_color = "#F97316"
    clean_outline = "#1E3A8A"
    real_outline = "#9A3412"

    group_width = (plot_right - plot_left) / len(rows)
    bar_width = 42
    gap = 42

    for i, row in enumerate(rows):
        center = plot_left + group_width * (i + 0.5)
        clean = float(row["clean_holdout_accuracy_pct"])
        real = float(row["real_external_accuracy_pct"])
        gap_pp = float(row["accuracy_gap_real_minus_clean_pp"])
        for j, (value, color, outline) in enumerate([(clean, clean_color, clean_outline), (real, real_color, real_outline)]):
            bx0 = int(center - bar_width - gap / 2) if j == 0 else int(center + gap / 2)
            bx1 = bx0 + bar_width
            by1 = baseline
            by0 = baseline - int(((value - min_value) / range_value) * (plot_bottom - plot_top))
            draw.rounded_rectangle((bx0, by0, bx1, by1), radius=7, fill=color, outline=outline, width=2)
            label_x = ((bx0 + bx1) // 2) - 8 if j == 0 else ((bx0 + bx1) // 2) + 8
            anchor = "rb" if j == 0 else "lb"
            draw.text((label_x, by0 - 8), f"{value:.2f}%", font=value_font, fill="#111111", anchor=anchor)

        draw.text((center, plot_bottom + 18), row["version"], font=label_font, fill="#111111", anchor="ma")
        draw.text((center, plot_bottom + 48), f"Gap {gap_pp:+.2f} pp", font=note_font, fill="#374151", anchor="ma")

    # Legend.
    legend_y = y0 + 82
    legend_x = x0 + 110
    draw.rounded_rectangle((legend_x, legend_y - 10, legend_x + 28, legend_y + 18), radius=5, fill=clean_color)
    draw.text((legend_x + 40, legend_y + 4), clean_label, font=axis_font, fill="#111111", anchor="lm")
    legend_x2 = legend_x + 310
    draw.rounded_rectangle((legend_x2, legend_y - 10, legend_x2 + 28, legend_y + 18), radius=5, fill=real_color)
    draw.text((legend_x2 + 40, legend_y + 4), real_label, font=axis_font, fill="#111111", anchor="lm")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s1_s3, s2_s4 = load_rows()
    image = Image.new("RGB", (1900, 1250), "#F5F7FB")
    draw = ImageDraw.Draw(image)

    draw.text(
        (950, 42),
        "LightGBM High Signal: Clean Dataset vs Real Dataset Accuracy",
        font=font(42, bold=True),
        fill="#111111",
        anchor="ma",
    )
    draw.text(
        (950, 92),
        "Version-by-version comparison with percentage labels and Real-Clean gap",
        font=font(25),
        fill="#4B5563",
        anchor="ma",
    )

    draw_panel(
        draw,
        s1_s3,
        (70, 145, 1830, 620),
        "SETC/S1 Clean 5,000 vs SETD/S3 Real 55,000",
        "Clean S1 Accuracy",
        "Real S3 Accuracy",
    )
    draw_panel(
        draw,
        s2_s4,
        (70, 700, 1830, 1175),
        "SETC/S2 Clean 50,000 vs SETD/S4 Real 105,000",
        "Clean S2 Accuracy",
        "Real S4 Accuracy",
    )

    image.save(OUT_PATH, quality=95)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
