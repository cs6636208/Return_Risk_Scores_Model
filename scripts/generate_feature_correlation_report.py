from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = ROOT / "data" / "processed" / "clean_dataset_v2.csv"
DEFAULT_OUT_DIR = ROOT / "docs" / "model_feature_extension" / "feature_correlation"
TARGET = "is_returned"

WINDOWS = (7, 30, 60, 90, 180, 365)

LEAKAGE_FEATURES = {
    "return_id",
    "return_date",
    "return_reason",
    "return_scenario",
    "item_condition",
    "return_status",
    "refund_amount",
    "shap_values",
}

POST_EVENT_FEATURES = {
    "delivery_date",
    "delivery_days",
    "delay_days",
}

MODEL_GENERATED_FEATURES = {
    "risk_score",
    "risk_tier",
    "score_id",
    "scored_at",
}

warnings.simplefilter("ignore", PerformanceWarning)


BASE_MEANINGS: dict[str, tuple[str, str, str]] = {
    "age": ("Customer", "อายุลูกค้า ใช้ดูพฤติกรรม return ตามช่วงวัย", "Order-time"),
    "customer_age_days": ("Customer", "จำนวนวันที่ลูกค้าอยู่ในระบบตั้งแต่วันสมัครถึงวันสั่งซื้อ", "Order-time"),
    "product_rating": ("Product", "คะแนนสินค้า ยิ่งต่ำมักสะท้อนคุณภาพ/ความพึงพอใจต่ำ", "Order-time"),
    "is_fragile": ("Product", "สินค้าแตกหักง่าย ใช้ดูความเสี่ยงจากความเสียหายระหว่างขนส่ง", "Order-time"),
    "avg_delivery_days": ("Delivery", "ค่าเฉลี่ยระยะเวลาส่งของ courier/product profile", "Order-time"),
    "damage_rate": ("Product", "อัตราความเสียหายของสินค้า/ผู้ขายจาก historical profile", "Order-time"),
    "promo_discount_rate": ("Promotion", "อัตราส่วนลดของโปรโมชัน", "Order-time"),
    "quantity": ("Order", "จำนวนสินค้าที่สั่งใน order", "Order-time"),
    "unit_price": ("Order", "ราคาต่อหน่วยก่อนคูณจำนวน", "Order-time"),
    "tier_discount_pct": ("Promotion", "ส่วนลดจากระดับสมาชิก", "Order-time"),
    "campaign_discount_pct": ("Promotion", "ส่วนลดจาก campaign", "Order-time"),
    "total_discount_pct": ("Promotion", "เปอร์เซ็นต์ส่วนลดรวม", "Order-time"),
    "discount_applied_amount": ("Promotion", "มูลค่าส่วนลดที่ใช้จริง", "Order-time"),
    "total_amount": ("Order", "มูลค่า order หลังส่วนลด", "Order-time"),
    "delivery_time_expected_days": ("Delivery", "จำนวนวันที่คาดว่าจะใช้ส่ง", "Order-time"),
    "delivery_days": ("Delivery", "จำนวนวันส่งจริง เป็นข้อมูลหลังส่ง ไม่ควรใช้ตอน order เพิ่งเข้า", "Post-event"),
    "delay_days": ("Delivery", "จำนวนวันที่ส่งช้า เป็นข้อมูลหลังส่ง ไม่ควรใช้ตอน order เพิ่งเข้า", "Post-event"),
    "is_repurchased_item": ("Customer", "ลูกค้าเคยซื้อสินค้านี้ซ้ำหรือไม่", "Order-time"),
    "order_hour": ("Order", "ชั่วโมงที่สั่งซื้อ ใช้จับ pattern ช่วงเวลาสั่ง", "Order-time"),
    "days_since_last_order": ("Customer", "จำนวนวันตั้งแต่ order ก่อนหน้าของลูกค้าคนเดียวกัน", "Order-time"),
    "hist_order_count": ("Customer history", "จำนวน order ก่อนหน้าของลูกค้าคนนี้แบบ lifetime", "Order-time"),
    "hist_return_rate": ("Customer history", "อัตราคืนสินค้าในอดีตของลูกค้า = return ก่อนหน้า / order ก่อนหน้า", "Order-time"),
    "refund_amount": ("Leakage", "ยอดเงินคืนหลังเกิด return ใช้แล้วทำให้ผลหลอกสูง ห้ามใช้ train จริง", "Leakage"),
    "risk_score": ("Generated risk", "คะแนน risk ที่มีอยู่ใน dataset เป็น signal แรงแต่ต้องระวังที่มาของคะแนน", "Conditional"),
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0)


def add_group_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
    prefix: str,
    windows: tuple[int, ...],
    add_spend: bool = False,
    add_delay: bool = False,
) -> list[str]:
    """Create point-in-time rolling features using only rows before current order."""
    created: list[str] = []
    sort_cols = group_cols + ["order_date", "order_id"]
    ordered = df.sort_values(sort_cols)

    group_key = group_cols[0] if len(group_cols) == 1 else group_cols
    for _, group in ordered.groupby(group_key, sort=False, dropna=False):
        idx = group.index
        dates = group["order_date"]
        ones = pd.Series(1.0, index=dates)
        returns = pd.Series(group[TARGET].astype(float).to_numpy(), index=dates)
        amount = pd.Series(group["total_amount"].astype(float).to_numpy(), index=dates)
        delay_flag = pd.Series((group["delay_days"].fillna(0).astype(float) > 0).astype(float).to_numpy(), index=dates)

        for days in windows:
            window = f"{days}D"
            order_col = f"{prefix}_order_count_{days}d"
            return_col = f"{prefix}_return_count_{days}d"
            rate_col = f"{prefix}_return_rate_{days}d"

            order_count = ones.shift(1).rolling(window, min_periods=1).sum().fillna(0).to_numpy()
            return_count = returns.shift(1).rolling(window, min_periods=1).sum().fillna(0).to_numpy()

            df.loc[idx, order_col] = order_count
            df.loc[idx, return_col] = return_count
            df.loc[idx, rate_col] = np.divide(
                return_count,
                order_count,
                out=np.zeros_like(return_count, dtype=float),
                where=order_count != 0,
            )
            created.extend([order_col, return_col, rate_col])

            if add_spend:
                spend_col = f"{prefix}_spend_sum_{days}d"
                df.loc[idx, spend_col] = amount.shift(1).rolling(window, min_periods=1).sum().fillna(0).to_numpy()
                created.append(spend_col)

            if add_delay:
                delay_count_col = f"{prefix}_delay_count_{days}d"
                delay_rate_col = f"{prefix}_delay_rate_{days}d"
                delay_count = delay_flag.shift(1).rolling(window, min_periods=1).sum().fillna(0).to_numpy()
                df.loc[idx, delay_count_col] = delay_count
                df.loc[idx, delay_rate_col] = np.divide(
                    delay_count,
                    order_count,
                    out=np.zeros_like(delay_count, dtype=float),
                    where=order_count != 0,
                )
                created.extend([delay_count_col, delay_rate_col])

    return sorted(set(created))


def add_engineered_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[str, str, str]]]:
    meanings = dict(BASE_MEANINGS)

    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    df["expected_delivery_date"] = pd.to_datetime(df["expected_delivery_date"], errors="coerce")

    median_amount = float(df["total_amount"].median())
    q75_discount = float(df["total_discount_pct"].quantile(0.75))

    df["order_month"] = df["order_date"].dt.month.fillna(0)
    df["order_dayofweek"] = df["order_date"].dt.dayofweek.fillna(0)
    df["is_weekend_order"] = df["order_dayofweek"].isin([5, 6]).astype(int)
    df["total_amount_log"] = np.log1p(df["total_amount"].clip(lower=0))
    df["high_value_flag"] = (df["total_amount"] >= median_amount).astype(int)
    df["high_discount_flag"] = (df["total_discount_pct"] >= q75_discount).astype(int)
    df["cod_flag"] = df["payment_method"].astype(str).str.contains("cod|cash", case=False, regex=True).astype(int)
    df["cod_high_value_flag"] = ((df["cod_flag"] == 1) & (df["high_value_flag"] == 1)).astype(int)
    df["has_promo_flag"] = (~df["promo_id"].astype(str).str.contains("NONE", case=False, na=False)).astype(int)
    df["no_previous_order_flag"] = (df["hist_order_count"].fillna(0) == 0).astype(int)
    df["low_rating_flag"] = (df["product_rating"].fillna(df["product_rating"].median()) <= 3.5).astype(int)
    df["discount_amount_per_item"] = safe_divide(df["discount_applied_amount"], df["quantity"].replace(0, np.nan))

    meanings.update(
        {
            "order_month": ("Order", "เดือนที่เกิด order ใช้จับ seasonality ของ return", "Order-time"),
            "order_dayofweek": ("Order", "วันในสัปดาห์ที่สั่งซื้อ", "Order-time"),
            "is_weekend_order": ("Order", "flag ว่า order เกิดวันเสาร์/อาทิตย์หรือไม่", "Order-time"),
            "total_amount_log": ("Order", "แปลงมูลค่า order ด้วย log เพื่อลดผล outlier ของยอดเงินสูงมาก", "Order-time"),
            "high_value_flag": ("Order", "flag order มูลค่าสูงกว่าค่ากลาง", "Order-time"),
            "high_discount_flag": ("Promotion", "flag ส่วนลดสูงกว่าควอไทล์ 75 ของ dataset", "Order-time"),
            "cod_flag": ("Payment", "flag การชำระแบบ COD/เงินสดปลายทาง", "Order-time"),
            "cod_high_value_flag": ("Interaction", "interaction ระหว่าง COD กับ order มูลค่าสูง", "Order-time"),
            "has_promo_flag": ("Promotion", "flag ว่า order ใช้โปรโมชันหรือไม่", "Order-time"),
            "no_previous_order_flag": ("Customer history", "flag ลูกค้าใหม่ที่ยังไม่มี order ก่อนหน้า", "Order-time"),
            "low_rating_flag": ("Product", "flag สินค้าคะแนนต่ำหรือปานกลาง ใช้จับสินค้าเสี่ยง", "Order-time"),
            "discount_amount_per_item": ("Promotion", "ส่วนลดเฉลี่ยต่อชิ้น", "Order-time"),
        }
    )

    created = add_group_rolling_features(df, ["customer_id"], "cust", WINDOWS, add_spend=True)
    for col in created:
        if "_order_count_" in col:
            meanings[col] = ("Customer rolling", f"จำนวน order ก่อนหน้าของลูกค้าภายใน {col.rsplit('_', 1)[-1]}", "Order-time")
        elif "_return_count_" in col:
            meanings[col] = ("Customer rolling", f"จำนวน return ก่อนหน้าของลูกค้าภายใน {col.rsplit('_', 1)[-1]}", "Order-time")
        elif "_return_rate_" in col:
            meanings[col] = ("Customer rolling", f"อัตรา return ย้อนหลังของลูกค้าภายใน {col.rsplit('_', 1)[-1]}", "Order-time")
        elif "_spend_sum_" in col:
            meanings[col] = ("Customer rolling", f"ยอดใช้จ่ายย้อนหลังของลูกค้าภายใน {col.rsplit('_', 1)[-1]}", "Order-time")

    rolling_specs = [
        (["product_id"], "product", (30, 90, 180), False, False, "Product rolling"),
        (["category"], "category", (30, 90, 180), False, False, "Category rolling"),
        (["brand"], "brand", (90, 180), False, False, "Brand rolling"),
        (["supplier_id"], "supplier", (90, 180), False, False, "Supplier rolling"),
        (["courier_id"], "courier", (30, 90, 180), False, True, "Courier rolling"),
        (["province"], "province", (90, 180), False, False, "Province rolling"),
        (["promo_id"], "promo", (90, 180), False, False, "Promotion rolling"),
        (["payment_method"], "payment_method", (90, 180), False, False, "Payment rolling"),
        (["channel_type"], "channel_type", (90, 180), False, False, "Channel rolling"),
        (["province", "category"], "province_category", (90, 180), False, False, "Interaction rolling"),
        (["category", "payment_method"], "category_payment", (90, 180), False, False, "Interaction rolling"),
        (["channel_type", "category"], "channel_category", (90, 180), False, False, "Interaction rolling"),
        (["province", "courier_id"], "province_courier", (90, 180), False, True, "Interaction rolling"),
    ]

    for group_cols, prefix, windows, add_spend, add_delay, group_name in rolling_specs:
        new_cols = add_group_rolling_features(df, group_cols, prefix, windows, add_spend=add_spend, add_delay=add_delay)
        for col in new_cols:
            if "_order_count_" in col:
                meanings[col] = (group_name, f"จำนวน order ย้อนหลังของกลุ่ม {prefix} ภายใน {col.rsplit('_', 1)[-1]}", "Order-time")
            elif "_return_count_" in col:
                meanings[col] = (group_name, f"จำนวน return ย้อนหลังของกลุ่ม {prefix} ภายใน {col.rsplit('_', 1)[-1]}", "Order-time")
            elif "_return_rate_" in col:
                meanings[col] = (group_name, f"อัตรา return ย้อนหลังของกลุ่ม {prefix} ภายใน {col.rsplit('_', 1)[-1]}", "Order-time")
            elif "_delay_count_" in col:
                meanings[col] = (group_name, f"จำนวนเคสส่งช้าย้อนหลังของกลุ่ม {prefix} ภายใน {col.rsplit('_', 1)[-1]}", "Order-time historical")
            elif "_delay_rate_" in col:
                meanings[col] = (group_name, f"อัตราส่งช้าย้อนหลังของกลุ่ม {prefix} ภายใน {col.rsplit('_', 1)[-1]}", "Order-time historical")

    return df, meanings


def feature_status(feature: str, availability: str) -> tuple[str, str]:
    if feature in LEAKAGE_FEATURES:
        return "ห้ามใช้", "Leakage หลังรู้ผล return แล้ว ทำให้โมเดลเก่งหลอก"
    if feature in POST_EVENT_FEATURES:
        return "ใช้เฉพาะ post-delivery", "รู้หลังส่งสินค้าแล้ว ไม่เหมาะกับ prediction ตอน order เพิ่งเข้า"
    if feature in MODEL_GENERATED_FEATURES:
        return "ใช้ได้แบบมีเงื่อนไข", "ต้องตรวจที่มาของ score ว่าไม่ได้สร้างจาก target/leakage"
    if availability == "Order-time historical":
        return "ใช้ได้ถ้ามี feature store", "คำนวณจากประวัติในอดีต ไม่ใช้ order ปัจจุบัน"
    return "แนะนำใช้", "รู้ได้ก่อนหรือขณะ order เข้า"


def correlation_table(df: pd.DataFrame, meanings: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(float)
    rows = []

    for col in df.columns:
        if col == TARGET:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() < 10 or s.nunique(dropna=True) <= 1:
            continue

        s = s.replace([np.inf, -np.inf], np.nan).fillna(s.median())
        pearson = float(s.corr(y, method="pearson"))
        spearman = float(s.corr(y, method="spearman"))
        if math.isnan(pearson) and math.isnan(spearman):
            continue

        group, meaning, availability = meanings.get(col, ("Raw numeric", "numeric feature จาก dataset ตั้งต้น", "Order-time"))
        use_status, note = feature_status(col, availability)
        direction = "positive" if pearson > 0 else "negative" if pearson < 0 else "neutral"
        direction_th = "ค่าสูงขึ้นสัมพันธ์กับโอกาส return สูงขึ้น" if pearson > 0 else "ค่าสูงขึ้นสัมพันธ์กับโอกาส return ลดลง" if pearson < 0 else "ไม่เห็นทิศทางชัด"

        rows.append(
            {
                "feature": col,
                "feature_group": group,
                "meaning_th": meaning,
                "availability": availability,
                "use_status": use_status,
                "pearson_corr_with_is_returned": pearson,
                "spearman_corr_with_is_returned": spearman,
                "abs_pearson_corr": abs(pearson),
                "direction": direction,
                "direction_meaning_th": direction_th,
                "model_note_th": note,
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values("abs_pearson_corr", ascending=False).reset_index(drop=True)


def categorical_return_summary(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [
        "gender",
        "membership_tier",
        "preferred_channel",
        "province",
        "category",
        "brand",
        "supplier_id",
        "courier_type",
        "promo_type",
        "channel_type",
        "payment_method",
        "risk_tier",
    ]
    global_rate = df[TARGET].mean()
    rows = []
    for col in cat_cols:
        if col not in df.columns:
            continue
        grouped = df.groupby(col, dropna=False)[TARGET].agg(["count", "mean"]).reset_index()
        grouped = grouped[grouped["count"] >= 30]
        for _, row in grouped.iterrows():
            lift = float(row["mean"] - global_rate)
            rows.append(
                {
                    "feature": col,
                    "value": row[col],
                    "count": int(row["count"]),
                    "return_rate": float(row["mean"]),
                    "global_return_rate": float(global_rate),
                    "lift_vs_global": lift,
                    "direction": "higher_than_average" if lift > 0 else "lower_than_average" if lift < 0 else "neutral",
                }
            )
    return pd.DataFrame(rows).sort_values("lift_vs_global", key=lambda s: s.abs(), ascending=False)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_name = "tahomabd.ttf" if bold else "tahoma.ttf"
    candidates = [
        Path("C:/Windows/Fonts") / font_name,
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def corr_color(value: float) -> tuple[int, int, int]:
    value = max(min(value, 1.0), -1.0)
    if value >= 0:
        # white to dark red: higher positive correlation means higher return-risk signal.
        strength = value
        r = int(255 * (1 - strength) + 166 * strength)
        g = int(247 * (1 - strength) + 20 * strength)
        b = int(247 * (1 - strength) + 34 * strength)
    else:
        # blue-gray for negative correlation: higher feature value is associated with lower return risk.
        strength = abs(value)
        r = int(255 * (1 - strength) + 52 * strength)
        g = int(247 * (1 - strength) + 92 * strength)
        b = int(247 * (1 - strength) + 160 * strength)
    return r, g, b


def compact_feature_label(feature: str) -> str:
    replacements = {
        "is_returned": "target_is_returned",
        "province_courier": "prov_courier",
        "delivery_time_expected_days": "expected_delivery_days",
        "return_rate": "ret_rate",
        "return_count": "ret_count",
        "order_count": "ord_count",
        "customer": "cust",
    }
    label = feature
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def save_bar_plot(corr: pd.DataFrame, out_path: Path) -> None:
    plot_df = corr[
        ~corr["use_status"].eq("ห้ามใช้")
    ].head(30).sort_values("pearson_corr_with_is_returned")
    values = plot_df["pearson_corr_with_is_returned"].to_numpy()
    max_abs = max(float(np.nanmax(np.abs(values))), 0.01)

    width, height = 1700, 1150
    left, right, top, bottom = 520, 90, 110, 90
    chart_w = width - left - right
    row_h = (height - top - bottom) / max(len(plot_df), 1)
    zero_x = left + chart_w / 2
    scale = (chart_w / 2) / max_abs

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(42, bold=True)
    label_font = load_font(22)
    small_font = load_font(20)
    draw.text((width / 2, 35), "Top Feature Correlation with Return Target", anchor="mm", fill="#111111", font=title_font)
    draw.line((zero_x, top - 10, zero_x, height - bottom + 10), fill="#222222", width=2)

    for tick in np.linspace(-max_abs, max_abs, 9):
        x = zero_x + tick * scale
        draw.line((x, top - 6, x, height - bottom), fill="#e5e7eb", width=1)
        draw.text((x, height - bottom + 24), f"{tick:+.2f}", anchor="mm", fill="#333333", font=small_font)

    for i, (_, row) in enumerate(plot_df.iterrows()):
        y = top + i * row_h + row_h / 2
        value = float(row["pearson_corr_with_is_returned"])
        bar_x = zero_x + value * scale
        color = "#2e7d32" if value >= 0 else "#c62828"
        draw.text((left - 15, y), str(row["feature"]), anchor="rm", fill="#111111", font=small_font)
        draw.rounded_rectangle((min(zero_x, bar_x), y - row_h * 0.33, max(zero_x, bar_x), y + row_h * 0.33), radius=6, fill=color)
        text_x = bar_x + (10 if value >= 0 else -10)
        anchor = "lm" if value >= 0 else "rm"
        draw.text((text_x, y), f"{value:+.3f}", anchor=anchor, fill="#111111", font=label_font)

    draw.text((left + chart_w / 2, height - 28), "Pearson correlation with is_returned", anchor="mm", fill="#333333", font=label_font)
    img.save(out_path)


def save_heatmap(df: pd.DataFrame, corr: pd.DataFrame, out_path: Path) -> None:
    safe = corr[~corr["use_status"].isin(["ห้ามใช้"])].copy()
    positive = safe.sort_values("pearson_corr_with_is_returned", ascending=False).head(9)["feature"].tolist()
    negative = safe.sort_values("pearson_corr_with_is_returned", ascending=True).head(6)["feature"].tolist()
    selected = []
    for feature in positive + negative:
        if feature in df.columns and feature not in selected:
            selected.append(feature)
    features = [TARGET] + selected
    matrix = df[features].apply(pd.to_numeric, errors="coerce").corr(method="pearson")

    n = len(features)
    cell = 66
    left, top = 570, 170
    width = left + n * cell + 135
    height = top + n * cell + 150

    img = Image.new("RGB", (width, height), "#fbfbfb")
    draw = ImageDraw.Draw(img)
    title_font = load_font(36, bold=True)
    label_font = load_font(18)
    value_font = load_font(17)
    draw.rounded_rectangle((28, 24, width - 28, height - 28), radius=26, fill="white", outline="#e5e7eb", width=2)
    draw.text((width / 2, 48), "Return-Risk Feature Correlation Heatmap", anchor="mm", fill="#111111", font=title_font)
    draw.text((width / 2, 90), "Dark red = strong return-risk signal, light red = weak risk signal, blue-gray = lower-risk direction", anchor="mm", fill="#4b5563", font=load_font(20))
    draw.text((left + n * cell / 2, 128), "Column numbers match the numbered feature list on the left", anchor="mm", fill="#6b7280", font=load_font(18))

    for i, feature in enumerate(features):
        y = top + i * cell + cell / 2
        numbered_label = f"{i + 1:02d}. {compact_feature_label(feature)}"
        draw.text((left - 14, y), numbered_label, anchor="rm", fill="#111111", font=label_font)
        x = left + i * cell + cell / 2
        draw.text((x, top - 24), str(i + 1), anchor="mm", fill="#111111", font=label_font)

    for i in range(n):
        for j in range(n):
            value = matrix.iloc[i, j]
            color = corr_color(float(value)) if pd.notna(value) else (230, 230, 230)
            x0 = left + j * cell
            y0 = top + i * cell
            draw.rounded_rectangle((x0 + 1, y0 + 1, x0 + cell - 1, y0 + cell - 1), radius=4, fill=color, outline="#ffffff")
            if pd.notna(value):
                text_color = "#ffffff" if abs(float(value)) >= 0.72 else "#111111"
                draw.text((x0 + cell / 2, y0 + cell / 2), f"{value:.2f}", anchor="mm", fill=text_color, font=value_font)

    legend_y = height - 65
    legend_x = left
    draw.rounded_rectangle((legend_x, legend_y, legend_x + 30, legend_y + 18), radius=4, fill="#345ca0")
    draw.text((legend_x + 40, legend_y + 9), "negative / lower-risk direction", anchor="lm", fill="#374151", font=load_font(17))
    draw.rounded_rectangle((legend_x + 310, legend_y, legend_x + 340, legend_y + 18), radius=4, fill="#ffffff", outline="#d1d5db")
    draw.text((legend_x + 350, legend_y + 9), "near zero", anchor="lm", fill="#374151", font=load_font(17))
    draw.rounded_rectangle((legend_x + 480, legend_y, legend_x + 510, legend_y + 18), radius=4, fill="#f9d7d7")
    draw.text((legend_x + 520, legend_y + 9), "light red / weak risk", anchor="lm", fill="#374151", font=load_font(17))
    draw.rounded_rectangle((legend_x + 730, legend_y, legend_x + 760, legend_y + 18), radius=4, fill="#a61422")
    draw.text((legend_x + 770, legend_y + 9), "dark red / strong risk", anchor="lm", fill="#374151", font=load_font(17))

    img.save(out_path)


def save_category_lift_plot(cat: pd.DataFrame, out_path: Path) -> None:
    if cat.empty:
        return
    plot_df = cat.head(25).copy()
    plot_df["label"] = plot_df["feature"].astype(str) + "=" + plot_df["value"].astype(str)
    plot_df = plot_df.sort_values("lift_vs_global")
    values = (plot_df["lift_vs_global"] * 100).to_numpy()
    max_abs = max(float(np.nanmax(np.abs(values))), 0.1)

    width, height = 1700, 1050
    left, right, top, bottom = 560, 100, 110, 90
    chart_w = width - left - right
    row_h = (height - top - bottom) / max(len(plot_df), 1)
    zero_x = left + chart_w / 2
    scale = (chart_w / 2) / max_abs

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(42, bold=True)
    label_font = load_font(22)
    small_font = load_font(19)
    draw.text((width / 2, 35), "Categorical Return Rate Lift vs Global Average", anchor="mm", fill="#111111", font=title_font)
    draw.line((zero_x, top - 10, zero_x, height - bottom + 10), fill="#222222", width=2)

    for tick in np.linspace(-max_abs, max_abs, 9):
        x = zero_x + tick * scale
        draw.line((x, top - 6, x, height - bottom), fill="#e5e7eb", width=1)
        draw.text((x, height - bottom + 24), f"{tick:+.1f}", anchor="mm", fill="#333333", font=small_font)

    for i, (_, row) in enumerate(plot_df.iterrows()):
        y = top + i * row_h + row_h / 2
        value = float(row["lift_vs_global"] * 100)
        bar_x = zero_x + value * scale
        color = "#1565c0" if value >= 0 else "#ef6c00"
        draw.text((left - 15, y), str(row["label"])[:58], anchor="rm", fill="#111111", font=small_font)
        draw.rounded_rectangle((min(zero_x, bar_x), y - row_h * 0.33, max(zero_x, bar_x), y + row_h * 0.33), radius=6, fill=color)
        text_x = bar_x + (10 if value >= 0 else -10)
        anchor = "lm" if value >= 0 else "rm"
        draw.text((text_x, y), f"{value:+.1f}pp", anchor=anchor, fill="#111111", font=label_font)

    draw.text((left + chart_w / 2, height - 28), "Return rate lift percentage points", anchor="mm", fill="#333333", font=label_font)
    img.save(out_path)


def save_meaning_dictionary(corr: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "feature",
        "feature_group",
        "meaning_th",
        "availability",
        "use_status",
        "direction_meaning_th",
        "model_note_th",
    ]
    corr[cols].to_csv(out_path, index=False, encoding="utf-8-sig")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_ไม่มีข้อมูล_"
    text_df = df.fillna("").astype(str)
    columns = text_df.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).replace("\n", " ") for col in columns) + " |"
        for _, row in text_df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def save_report(corr: pd.DataFrame, cat: pd.DataFrame, out_path: Path, dataset_path: Path | str = DEFAULT_INPUT_CSV) -> None:
    safe_corr = corr[~corr["use_status"].eq("ห้ามใช้")]
    top_positive = safe_corr.sort_values("pearson_corr_with_is_returned", ascending=False).head(10)
    top_negative = safe_corr.sort_values("pearson_corr_with_is_returned", ascending=True).head(10)
    leakage = corr[corr["use_status"].eq("ห้ามใช้")].head(10)

    def table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_ไม่มีข้อมูล_"
        show = df[[
            "feature",
            "feature_group",
            "pearson_corr_with_is_returned",
            "direction_meaning_th",
            "use_status",
        ]].copy()
        show["pearson_corr_with_is_returned"] = show["pearson_corr_with_is_returned"].map(lambda x: f"{x:+.4f}")
        return markdown_table(show)

    recommended = safe_corr[
        safe_corr["use_status"].isin(["แนะนำใช้", "ใช้ได้ถ้ามี feature store"])
    ].sort_values("abs_pearson_corr", ascending=False).head(25)

    recommended_table = recommended[[
        "feature",
        "feature_group",
        "meaning_th",
        "pearson_corr_with_is_returned",
        "use_status",
    ]].copy()
    recommended_table["pearson_corr_with_is_returned"] = recommended_table["pearson_corr_with_is_returned"].map(lambda x: f"{x:+.4f}")

    cat_show = cat.head(15).copy()
    if not cat_show.empty:
        cat_show["return_rate"] = cat_show["return_rate"].map(lambda x: f"{x:.2%}")
        cat_show["lift_vs_global"] = cat_show["lift_vs_global"].map(lambda x: f"{x:+.2%}")
        cat_table = markdown_table(cat_show[["feature", "value", "count", "return_rate", "lift_vs_global", "direction"]])
    else:
        cat_table = "_ไม่มีข้อมูล_"

    text = f"""# Feature Correlation Report

Dataset: `{dataset_path}`

Target: `is_returned`

## วิธีอ่านค่า correlation

- ค่าเป็นบวก: feature สูงขึ้นแล้วสัมพันธ์กับโอกาส return สูงขึ้น
- ค่าเป็นลบ: feature สูงขึ้นแล้วสัมพันธ์กับโอกาส return ลดลง
- ค่าใกล้ 0: ความสัมพันธ์เชิงเส้นกับ target ยังไม่ชัด อาจยังมีประโยชน์ผ่าน interaction หรือ model tree
- Correlation ไม่ได้แปลว่าเป็นสาเหตุโดยตรง แต่ใช้คัด feature candidate ได้ดี

## Top Positive Correlation

{table(top_positive)}

## Top Negative Correlation

{table(top_negative)}

## Leakage / Post-event ที่ต้องระวัง

{table(leakage)}

## Feature ที่แนะนำพิจารณาเพิ่มเข้า model version ถัดไป

{markdown_table(recommended_table)}

## Categorical Insight แบบ Return Rate Lift

{cat_table}

## สรุปเชิง model

1. Feature ที่เป็น customer/product/category/province/courier rolling history คือกลุ่มที่เหมาะกับ production มากที่สุด เพราะคำนวณจากอดีตก่อน order ปัจจุบัน
2. Feature ที่เป็น `refund_amount`, `return_date`, `return_reason` เป็น leakage ห้ามใช้ train model จริง แม้ correlation จะสูงมาก
3. Feature ที่เป็น `delivery_days` และ `delay_days` ใช้ได้เฉพาะ model หลังส่งสินค้าแล้ว ไม่เหมาะกับ real-time prediction ตอน order เพิ่งเข้า
4. ถ้าจะเพิ่ม feature ใหม่เข้า model ต้อง retrain model version ใหม่ ไม่ใช่ส่ง column เพิ่มเข้า model เดิมแล้วให้ model เข้าใจเอง
5. Feature Store ควรเก็บ rolling feature เช่น `cust_return_rate_90d`, `product_return_rate_90d`, `province_category_return_rate_90d` เพื่อให้ inference เร็วและไม่ต้อง scan ทั้ง dataset
"""
    out_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate feature correlation report and heatmap.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="Input CSV path.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--label", default="v2", help="Filename label, for example lightgbm_setd_s3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    if not input_csv.is_absolute():
        input_csv = ROOT / input_csv
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    df, meanings = add_engineered_features(df)
    corr = correlation_table(df, meanings)
    cat = categorical_return_summary(df)

    label = str(args.label).strip() or "v2"
    corr_path = out_dir / f"feature_target_correlation_{label}.csv"
    cat_path = out_dir / f"categorical_return_rate_lift_{label}.csv"
    dict_path = out_dir / f"feature_meaning_dictionary_{label}.csv"
    report_path = out_dir / "feature_correlation_report.md"

    corr.to_csv(corr_path, index=False, encoding="utf-8-sig")
    cat.to_csv(cat_path, index=False, encoding="utf-8-sig")
    save_meaning_dictionary(corr, dict_path)
    save_report(corr, cat, report_path, dataset_path=input_csv.relative_to(ROOT) if input_csv.is_relative_to(ROOT) else input_csv)

    save_bar_plot(corr, image_dir / f"top_feature_target_correlation_{label}.png")
    save_heatmap(df, corr, image_dir / f"feature_correlation_heatmap_{label}.png")
    save_category_lift_plot(cat, image_dir / f"categorical_return_rate_lift_{label}.png")

    print(f"Saved: {corr_path}")
    print(f"Saved: {cat_path}")
    print(f"Saved: {dict_path}")
    print(f"Saved: {report_path}")
    print(f"Saved images: {image_dir}")


if __name__ == "__main__":
    main()
