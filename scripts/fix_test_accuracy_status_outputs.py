from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "docs" / "test"
PLAN_PATH = TEST_ROOT / "dataset_5000_50000_v1_to_v5_evaluation_plan.csv"
RESULTS_PATH = TEST_ROOT / "model_full_test_results" / "full_test_model_evaluation_results.csv"


def font(size: int) -> ImageFont.ImageFont:
    font_path = Path("C:/Windows/Fonts/tahoma.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def pct(value: object) -> str:
    try:
        if value == "" or pd.isna(value):
            return "-"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    fnt: ImageFont.ImageFont,
    fill: str = "#111111",
    max_chars: int = 92,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            y += getattr(fnt, "size", 22) + line_gap
            continue
        for line in wrap(paragraph, max_chars):
            draw.text((x, y), line, font=fnt, fill=fill)
            y += getattr(fnt, "size", 22) + line_gap
    return y


def normalize_plan() -> pd.DataFrame:
    plan = pd.read_csv(PLAN_PATH)
    pending_cols = [
        "new_full_test_accuracy",
        "new_full_test_recall",
        "new_full_test_precision",
        "new_full_test_f1",
        "new_full_test_auc",
        "new_full_test_cost",
        "accuracy_delta",
    ]
    for col in pending_cols:
        if col not in plan.columns:
            plan[col] = ""
        plan[col] = ""
    plan["status"] = "pending_saved_model_inference"
    plan["reason"] = (
        "Original artifact accuracy is shown for reference only. "
        "New full-test accuracy is not calculated yet because each version must run its own feature builder "
        "before loading the saved model and predicting every row."
    )
    plan.to_csv(PLAN_PATH, index=False, encoding="utf-8-sig")

    for dataset_size in [5000, 50000]:
        dataset_dir = TEST_ROOT / f"dataset_{dataset_size}"
        subset = plan[plan["dataset_size"].eq(dataset_size)].copy()
        subset.to_csv(dataset_dir / f"dataset_{dataset_size}_v1_to_v5_evaluation_plan.csv", index=False, encoding="utf-8-sig")
        for _, row in subset.iterrows():
            version_number = str(row["version"]).replace("V", "")
            version_dir = dataset_dir / f"version_{version_number}"
            version_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(
                version_dir / f"{str(row['version']).lower()}_dataset_{dataset_size}_evaluation_plan.csv",
                index=False,
                encoding="utf-8-sig",
            )
            (version_dir / "README.md").write_text(
                f"""# {row['version']} on Dataset {dataset_size}

Model: `{row['model']}`

Version id: `{row['version_id']}`

Test input: `{row['test_file']}`

Original artifact Accuracy: `{pct(row['original_accuracy'])}`

New full-test Accuracy: `pending`

สำคัญ: ค่านี้ยังไม่ใช่ผลที่ยิง test data ใหม่เข้า saved model จริง ตัวเลขที่เห็นเป็นค่าเดิมจากการ train/evaluate ของ version นั้น ๆ เท่านั้น

ทำไมยัง pending:

1. ต้องสร้าง feature จาก test data ด้วย feature builder ของ version นี้ก่อน
2. ต้อง align column/order/encoding ให้ตรงกับ saved model
3. ต้อง predict ทุก row แล้วค่อยคำนวณ Accuracy ใหม่
4. ถ้ายังไม่ทำขั้นตอนนี้ การนำ `original_accuracy` ไปวาดซ้ำจะทำให้ดูเหมือน dataset_5000/dataset_50000 ค่าเท่ากัน
""",
                encoding="utf-8",
            )
    return plan


def draw_status_chart(dataset_size: int, subset: pd.DataFrame) -> None:
    dataset_dir = TEST_ROOT / f"dataset_{dataset_size}"
    image_dir = dataset_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1900, 1120
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    draw.text((width // 2, 44), f"Dataset {dataset_size}: Original vs New Full-Test Accuracy", font=font(42), fill="#111111", anchor="ma")
    draw.text((width // 2, 94), "สิ่งที่เท่ากันคือค่าเดิมที่เอามาอ้างอิง ไม่ใช่ผลทดสอบใหม่", font=font(25), fill="#B71C1C", anchor="ma")

    left_x0, left_y0, left_x1, left_y1 = 110, 185, 1020, 780
    right_x0, right_y0, right_x1, right_y1 = 1130, 185, 1810, 780

    draw.text(((left_x0 + left_x1) // 2, 145), "Original Artifact Accuracy", font=font(30), fill="#263238", anchor="ma")
    draw.text(((right_x0 + right_x1) // 2, 145), "New Full-Test Accuracy", font=font(30), fill="#263238", anchor="ma")

    for x0, y0, x1, y1 in [(left_x0, left_y0, left_x1, left_y1), (right_x0, right_y0, right_x1, right_y1)]:
        draw.line((x0, y1, x1, y1), fill="#263238", width=2)
        draw.line((x0, y0, x0, y1), fill="#263238", width=2)
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            yy = int(y1 - tick * (y1 - y0))
            draw.line((x0 - 6, yy, x1, yy), fill="#ECEFF1", width=1)
            draw.text((x0 - 12, yy), f"{int(tick * 100)}", font=font(18), fill="#546E7A", anchor="rm")

    colors = ["#6D7C85", "#2E7D32", "#F9A825", "#1565C0", "#8E24AA"]
    rows = subset.reset_index(drop=True)
    bar_space = (left_x1 - left_x0 - 90) // len(rows)
    for i, row in rows.iterrows():
        acc = float(row["original_accuracy"])
        bx = left_x0 + 55 + i * bar_space
        bw = 95
        by = int(left_y1 - acc * (left_y1 - left_y0))
        draw.rounded_rectangle((bx, by, bx + bw, left_y1), radius=6, fill=colors[i % len(colors)])
        draw.text((bx + bw // 2, by - 8), pct(acc), font=font(21), fill="#111111", anchor="ms")
        draw.text((bx + bw // 2, left_y1 + 18), str(row["version"]), font=font(22), fill="#111111", anchor="ma")

        px = right_x0 + 55 + i * ((right_x1 - right_x0 - 90) // len(rows))
        draw.rounded_rectangle((px, right_y0 + 210, px + bw, right_y1), radius=6, fill="#CFD8DC", outline="#90A4AE", width=2)
        draw.text((px + bw // 2, right_y0 + 195), "PENDING", font=font(18), fill="#B71C1C", anchor="ma")
        draw.text((px + bw // 2, right_y1 + 18), str(row["version"]), font=font(22), fill="#111111", anchor="ma")

    note = (
        "ถ้าต้องการเห็นความต่างจริง ต้องทำ full-test inference: "
        "เอา test CSV ทั้งก้อนผ่าน feature engineering ของแต่ละ version แล้วใช้ saved model predict ทุก row. "
        "ก่อนหน้านี้กราฟใช้ original_accuracy ซ้ำ จึงไม่มีทางเห็นความต่างระหว่าง dataset_5000 กับ dataset_50000."
    )
    draw_wrapped(draw, note, (110, 850), font(26), fill="#263238", max_chars=116)

    img.save(image_dir / f"dataset_{dataset_size}_accuracy_by_version.png")


def write_explanation_doc(dataset_size: int, subset: pd.DataFrame) -> None:
    dataset_dir = TEST_ROOT / f"dataset_{dataset_size}"
    lines = [
        f"# Dataset {dataset_size} Accuracy Status",
        "",
        "## ทำไมก่อนหน้านี้ Accuracy ดูเท่ากัน",
        "",
        "เพราะกราฟเดิมใช้ `original_accuracy` ของแต่ละ model artifact มาวาดซ้ำ ไม่ได้ใช้ผล predict จาก test data ใหม่ทั้งชุด",
        "",
        "ดังนั้น dataset_5000 และ dataset_50000 จะดูเหมือนกัน เพราะทั้งสองชุดยังไม่ได้ถูกส่งเข้าโมเดลจริงเพื่อคำนวณ `new_full_test_accuracy`",
        "",
        "## สถานะปัจจุบัน",
        "",
        "| Version | Model | Original Accuracy | New Full-Test Accuracy | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in subset.iterrows():
        lines.append(
            f"| {row['version']} | {row['model']} | {pct(row['original_accuracy'])} | pending | {row['status']} |"
        )
    lines += [
        "",
        "## ต้องทำอะไรต่อเพื่อให้เห็นความต่างจริง",
        "",
        "1. สร้าง feature ของ test data ด้วย feature engineering ของแต่ละ version",
        "2. Align column, encoding, scaling ให้ตรงกับตอน train model",
        "3. Load saved model ของ version นั้น",
        "4. Predict ทุก row ใน test data",
        "5. คำนวณ Accuracy, Recall, Precision, F1, AUC, Cost ใหม่",
        "",
        "สรุป: ตอนนี้ความต่างของ feature/model มีอยู่ แต่ความต่างของผลบน test data ใหม่ยังไม่ถูกคำนวณ จึงไม่ควรอ่านกราฟเดิมว่าเป็นผลทดสอบใหม่",
    ]
    (dataset_dir / f"dataset_{dataset_size}_accuracy_status_explanation.md").write_text("\n".join(lines), encoding="utf-8")


def write_root_explanation(plan: pd.DataFrame) -> None:
    lines = [
        "# Why Accuracy Looked The Same",
        "",
        "ไฟล์นี้อธิบายประเด็นที่กราฟ dataset_5000 และ dataset_50000 เคยดูเหมือน Accuracy เท่ากันหมด",
        "",
        "## คำตอบสั้น ๆ",
        "",
        "เพราะตัวเลขที่แสดงเป็น `original_accuracy` ของ model แต่ละ version ไม่ใช่ `new_full_test_accuracy` จากการยิง test data ใหม่เข้า model",
        "",
        "## หลักฐานจากไฟล์ CSV",
        "",
        f"- แผนประเมินอยู่ที่ `{PLAN_PATH.relative_to(ROOT)}`",
        "- คอลัมน์ `original_accuracy` มีค่า",
        "- คอลัมน์ `new_full_test_accuracy` ยังว่าง",
        "- คอลัมน์ `status` เป็น `pending_saved_model_inference`",
        "",
        "## แปลว่าอะไร",
        "",
        "ถ้าเราเอาค่าเดิมไปวาดซ้ำใน dataset_5000 และ dataset_50000 กราฟจะไม่มีทางเปลี่ยน เพราะมันไม่ได้คำนวณจากข้อมูลชุดใหม่",
        "",
        "## สิ่งที่ต้องทำถ้าจะได้ Accuracy ใหม่จริง",
        "",
        "ต้องรัน saved-model inference ต่อ version โดยใช้ feature builder ของ version นั้น ๆ ก่อน แล้วค่อยเติมค่าใน `new_full_test_accuracy`",
        "",
        "## Current Rows",
        "",
        "| Dataset | Version | Original Accuracy | New Full-Test Accuracy | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in plan.iterrows():
        lines.append(
            f"| {int(row['dataset_size'])} | {row['version']} | {pct(row['original_accuracy'])} | pending | {row['status']} |"
        )
    (TEST_ROOT / "WHY_ACCURACY_LOOKED_THE_SAME.md").write_text("\n".join(lines), encoding="utf-8")


def update_results_file(plan: pd.DataFrame) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    plan = normalize_plan()
    for dataset_size in [5000, 50000]:
        subset = plan[plan["dataset_size"].eq(dataset_size)].copy()
        draw_status_chart(dataset_size, subset)
        write_explanation_doc(dataset_size, subset)
    write_root_explanation(plan)
    update_results_file(plan)
    print("Updated accuracy status outputs. New full-test accuracy is clearly marked as pending.")


if __name__ == "__main__":
    main()
