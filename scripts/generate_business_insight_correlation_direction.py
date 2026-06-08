from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "version 1" / "data" / "features" / "df_engineered.csv"
BUSINESS_DIR = ROOT / "reports" / "business_insights"
OLD_IMPORTANCE_PNG = BUSINESS_DIR / "05_feature_importance.png"
OUT_CSV = BUSINESS_DIR / "05_feature_importance_correlation_direction.csv"
OUT_PNG = BUSINESS_DIR / "05_feature_importance_correlation_direction.png"
OUT_PDF = (
    ROOT
    / "notebooks"
    / "eda"
    / "business_insights_report_feature_importance_correlation_supplement.pdf"
)
SOURCE_PDF = ROOT / "notebooks" / "eda" / "business_insights_report.pdf"
OUT_MERGED_PDF = ROOT / "notebooks" / "eda" / "business_insights_report_with_correlation_direction.pdf"


# These features come from the first chart in business_insights_report.pdf:
# reports/business_insights/05_feature_importance.png
TOP_IMPORTANCE_FEATURES = [
    "customer_return_ratio",
    "hist_return_rate_180d",
    "total_returns_before",
    "days_since_last_return",
    "delivery_time_expected_days",
    "hist_return_rate_60d",
    "hist_order_count_60d",
    "product_rating",
    "order_dayofweek",
    "customer_age_days",
    "hist_return_rate_30d",
    "days_since_last_order",
    "hist_order_count_180d",
    "hist_order_count_30d",
    "age",
    "return_rate_by_category",
    "hist_spend_sum_30d",
    "hist_spend_sum_180d",
    "promo_discount_pct",
    "total_orders_before",
]


def direction_from_corr(value: float) -> str:
    if pd.isna(value):
        return "not_available"
    if value > 0.01:
        return "positive"
    if value < -0.01:
        return "negative"
    return "near_zero"


def make_interpretation(feature: str, corr: float) -> str:
    direction = direction_from_corr(corr)
    if direction == "positive":
        return f"{feature} สูงขึ้น มีแนวโน้มสัมพันธ์กับโอกาสคืนสินค้าที่สูงขึ้น"
    if direction == "negative":
        return f"{feature} สูงขึ้น มีแนวโน้มสัมพันธ์กับโอกาสคืนสินค้าที่ลดลง"
    if direction == "near_zero":
        return f"{feature} มีความสัมพันธ์เชิงเส้นกับการคืนสินค้าน้อย แต่ยังอาจช่วยโมเดลผ่าน interaction/non-linear pattern"
    return f"{feature} คำนวณ correlation ไม่ได้จากข้อมูลชุดนี้"


def compute_correlation_table() -> pd.DataFrame:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source dataset: {SOURCE}")

    df = pd.read_csv(SOURCE)
    if "is_returned" not in df.columns:
        raise ValueError("Source dataset must contain target column: is_returned")

    target = pd.to_numeric(df["is_returned"], errors="coerce")
    rows: list[dict[str, object]] = []

    for rank, feature in enumerate(TOP_IMPORTANCE_FEATURES, start=1):
        if feature not in df.columns:
            rows.append(
                {
                    "importance_rank": rank,
                    "feature": feature,
                    "pearson_corr": np.nan,
                    "spearman_corr": np.nan,
                    "direction": "missing_feature",
                    "mean_not_returned": np.nan,
                    "mean_returned": np.nan,
                    "delta_return_minus_not": np.nan,
                    "non_null_rows": 0,
                    "interpretation": f"{feature} ไม่พบใน df_engineered.csv",
                }
            )
            continue

        feature_values = pd.to_numeric(df[feature], errors="coerce")
        tmp = pd.DataFrame({"feature_value": feature_values, "target": target}).dropna()
        if tmp.empty or tmp["feature_value"].nunique() <= 1 or tmp["target"].nunique() <= 1:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = tmp["feature_value"].corr(tmp["target"], method="pearson")
            # Pandas delegates Spearman to scipy in some environments. Ranking first
            # gives the same correlation definition without adding a dependency.
            spearman = tmp["feature_value"].rank().corr(tmp["target"].rank(), method="pearson")

        mean_not_returned = tmp.loc[tmp["target"] == 0, "feature_value"].mean()
        mean_returned = tmp.loc[tmp["target"] == 1, "feature_value"].mean()
        delta = mean_returned - mean_not_returned

        rows.append(
            {
                "importance_rank": rank,
                "feature": feature,
                "pearson_corr": pearson,
                "spearman_corr": spearman,
                "direction": direction_from_corr(pearson),
                "mean_not_returned": mean_not_returned,
                "mean_returned": mean_returned,
                "delta_return_minus_not": delta,
                "non_null_rows": int(len(tmp)),
                "interpretation": make_interpretation(feature, pearson),
            }
        )

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    return out


def plot_correlation_direction(corr_df: pd.DataFrame) -> None:
    plot_df = corr_df.dropna(subset=["pearson_corr"]).copy()
    plot_df = plot_df.sort_values("pearson_corr", ascending=True)
    colors = {"positive": "#2E7D32", "negative": "#C62828", "near_zero": "#78909C"}

    width, height = 1800, 1300
    image = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    title_font = load_font(46)
    subtitle_font = load_font(24)
    axis_font = load_font(22)
    label_font = load_font(21)
    small_font = load_font(19)

    draw.text(
        (width // 2, 42),
        "Correlation Direction of Top Feature-Importance Features",
        font=title_font,
        fill="#111111",
        anchor="ma",
    )
    draw.text(
        (width // 2, 102),
        "Positive = higher feature value tends to increase return probability | Negative = tends to reduce return probability",
        font=subtitle_font,
        fill="#455A64",
        anchor="ma",
    )

    left, right, top, bottom = 520, 145, 190, 130
    plot_x0, plot_y0 = left, top
    plot_x1, plot_y1 = width - right, height - bottom
    plot_w, plot_h = plot_x1 - plot_x0, plot_y1 - plot_y0

    max_abs = max(0.05, float(np.nanmax(np.abs(plot_df["pearson_corr"]))))
    max_abs *= 1.18

    def x_to_px(value: float) -> int:
        return int(plot_x0 + ((value + max_abs) / (2 * max_abs)) * plot_w)

    # Vertical grid and x-axis labels.
    ticks = np.linspace(-max_abs, max_abs, 7)
    for tick in ticks:
        x = x_to_px(float(tick))
        fill = "#263238" if abs(tick) < 1e-12 else "#E0E0E0"
        line_w = 3 if abs(tick) < 1e-12 else 1
        draw.line((x, plot_y0, x, plot_y1), fill=fill, width=line_w)
        draw.text((x, plot_y1 + 18), f"{tick:+.2f}", font=small_font, fill="#37474F", anchor="ma")

    n = len(plot_df)
    row_h = plot_h / n
    bar_h = max(20, min(35, int(row_h * 0.62)))
    zero_x = x_to_px(0)

    for i, row in enumerate(plot_df.itertuples(index=False)):
        y_mid = int(plot_y0 + row_h * i + row_h / 2)
        value = float(row.pearson_corr)
        val_x = x_to_px(value)
        color = colors.get(row.direction, "#78909C")

        # Feature label.
        draw.text((plot_x0 - 18, y_mid), str(row.feature), font=label_font, fill="#111111", anchor="rm")

        # Bar.
        y0, y1 = y_mid - bar_h // 2, y_mid + bar_h // 2
        x0, x1 = sorted((zero_x, val_x))
        draw.rounded_rectangle((x0, y0, x1, y1), radius=5, fill=color)

        # Value label.
        label = f"{value:+.3f}"
        if value >= 0:
            draw.text((val_x + 10, y_mid), label, font=label_font, fill="#111111", anchor="lm")
        else:
            draw.text((val_x - 10, y_mid), label, font=label_font, fill="#111111", anchor="rm")

    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill="#B0BEC5", width=2)
    draw.text(
        ((plot_x0 + plot_x1) // 2, height - 52),
        "Pearson correlation with is_returned",
        font=axis_font,
        fill="#263238",
        anchor="ma",
    )

    # Legend.
    legend_y = 145
    legend_items = [("Positive", "#2E7D32"), ("Negative", "#C62828"), ("Near zero", "#78909C")]
    legend_x = 1120
    for label, color in legend_items:
        draw.rounded_rectangle((legend_x, legend_y - 10, legend_x + 26, legend_y + 16), radius=5, fill=color)
        draw.text((legend_x + 36, legend_y + 3), label, font=small_font, fill="#263238", anchor="lm")
        legend_x += 170

    image.save(OUT_PNG)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = Path("C:/Windows/Fonts/tahoma.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: str = "#111111",
    max_chars: int = 92,
    line_spacing: int = 8,
) -> int:
    x, y = xy
    for paragraph in text.split("\n"):
        if not paragraph:
            y += font.size + line_spacing
            continue
        for line in wrap(paragraph, max_chars):
            draw.text((x, y), line, font=font, fill=fill)
            y += font.size + line_spacing
    return y


def paste_scaled(
    canvas: Image.Image,
    image_path: Path,
    box: tuple[int, int, int, int],
    border: bool = True,
) -> None:
    if not image_path.exists():
        return
    img = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = box
    max_w, max_h = x2 - x1, y2 - y1
    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    px = x1 + (max_w - img.width) // 2
    py = y1 + (max_h - img.height) // 2
    canvas.paste(img, (px, py))
    if border:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((px, py, px + img.width, py + img.height), outline="#CFD8DC", width=2)


def make_pdf_supplement(corr_df: pd.DataFrame) -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    page_w, page_h = 1800, 1300
    bg = "#FFFFFF"
    title_font = load_font(42)
    subtitle_font = load_font(27)
    body_font = load_font(24)
    small_font = load_font(20)

    pages: list[Image.Image] = []

    page1 = Image.new("RGB", (page_w, page_h), bg)
    d1 = ImageDraw.Draw(page1)
    d1.text((80, 55), "Business Insight Graph 1: เพิ่มมุมมอง Correlation Direction", font=title_font, fill="#111111")
    y = draw_wrapped(
        d1,
        "กราฟ Feature Importance เดิมบอกว่า feature ไหนมีผลต่อการแยกกลุ่มมากที่สุด แต่ไม่ได้บอกว่า feature นั้นสัมพันธ์กับการคืนสินค้าในทิศทางบวกหรือลบ "
        "จึงเพิ่มกราฟ correlation เพื่ออธิบายเหตุผลของความสำคัญให้ชัดขึ้น.",
        (80, 125),
        body_font,
        fill="#263238",
        max_chars=105,
    )
    d1.text((80, y + 15), "กราฟเดิม: Feature Importance", font=subtitle_font, fill="#0D47A1")
    paste_scaled(page1, OLD_IMPORTANCE_PNG, (80, y + 60, 1720, 1240))
    pages.append(page1)

    page2 = Image.new("RGB", (page_w, page_h), bg)
    d2 = ImageDraw.Draw(page2)
    d2.text((80, 55), "Correlation Direction ของ Top Features เทียบกับ is_returned", font=title_font, fill="#111111")
    y = draw_wrapped(
        d2,
        "วิธีที่ใช้: คำนวณ Pearson correlation ระหว่าง feature แต่ละตัวกับ target is_returned. "
        "ค่าบวกหมายถึง feature สูงขึ้นแล้วมีแนวโน้มพบการคืนสินค้ามากขึ้น ส่วนค่าลบหมายถึง feature สูงขึ้นแล้วมีแนวโน้มพบการคืนสินค้าน้อยลง. "
        "Correlation ไม่ใช่ causation แต่ช่วยอธิบายทิศทางของ signal ก่อนเข้าโมเดลได้ดี.",
        (80, 125),
        body_font,
        fill="#263238",
        max_chars=105,
    )
    paste_scaled(page2, OUT_PNG, (60, y + 35, 1740, 1240))
    pages.append(page2)

    page3 = Image.new("RGB", (page_w, page_h), bg)
    d3 = ImageDraw.Draw(page3)
    d3.text((80, 55), "สรุปค่าที่ควรใช้อธิบายอาจารย์", font=title_font, fill="#111111")
    d3.text((80, 130), "Top positive correlation", font=subtitle_font, fill="#2E7D32")
    positive = corr_df.dropna(subset=["pearson_corr"]).sort_values("pearson_corr", ascending=False).head(8)
    y = 180
    for _, row in positive.iterrows():
        y = draw_wrapped(
            d3,
            f"- {row['feature']}: corr={row['pearson_corr']:+.3f} | {row['interpretation']}",
            (95, y),
            small_font,
            fill="#111111",
            max_chars=115,
            line_spacing=6,
        )
    d3.text((80, y + 25), "Top negative correlation", font=subtitle_font, fill="#C62828")
    y += 75
    negative = corr_df.dropna(subset=["pearson_corr"]).sort_values("pearson_corr", ascending=True).head(8)
    for _, row in negative.iterrows():
        y = draw_wrapped(
            d3,
            f"- {row['feature']}: corr={row['pearson_corr']:+.3f} | {row['interpretation']}",
            (95, y),
            small_font,
            fill="#111111",
            max_chars=115,
            line_spacing=6,
        )
    y += 15
    draw_wrapped(
        d3,
        "หมายเหตุ: Feature Importance และ Correlation ตอบคนละคำถามกัน. Importance บอกว่า feature ช่วยโมเดลแยกกลุ่มได้มากแค่ไหน "
        "ส่วน Correlation บอกทิศทางความสัมพันธ์กับ target แบบเส้นตรง. ถ้า importance สูงแต่ correlation ใกล้ศูนย์ แปลว่า feature นั้นอาจช่วยผ่าน interaction หรือ pattern แบบไม่เป็นเส้นตรง.",
        (80, min(y + 25, 1120)),
        small_font,
        fill="#455A64",
        max_chars=120,
    )
    pages.append(page3)

    pages[0].save(OUT_PDF, save_all=True, append_images=pages[1:], resolution=150)


def make_merged_report_pdf() -> None:
    if not SOURCE_PDF.exists() or not OUT_PDF.exists():
        return

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for pdf_path in [SOURCE_PDF, OUT_PDF]:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            writer.add_page(page)

    with OUT_MERGED_PDF.open("wb") as f:
        writer.write(f)


def main() -> None:
    corr_df = compute_correlation_table()
    plot_correlation_direction(corr_df)
    make_pdf_supplement(corr_df)
    make_merged_report_pdf()
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved PDF: {OUT_PDF}")
    print(f"Saved merged PDF: {OUT_MERGED_PDF}")


if __name__ == "__main__":
    main()
