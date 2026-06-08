from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "docs" / "test"
DATASETS_ROOT = TEST_ROOT / "datasets"
COMPARISON = ROOT / "docs" / "Comparison Version" / "version_1_to_4_selected_model_comparison.csv"
V5_METRICS = (
    ROOT
    / "docs"
    / "version 5"
    / "v2_xgboost_safe_plus_rolling_HIGH_ACCURACY"
    / "reports"
    / "metrics_v2_xgboost_safe_plus_rolling_HIGH_ACCURACY.csv"
)


@dataclass
class ModelVersion:
    display_version: str
    version_id: str
    model: str
    feature_policy: str
    dataset_origin: str
    original_accuracy: float
    original_recall: float
    original_precision: float
    original_f1: float
    original_auc: float
    original_cost: float
    explanation: str


def font(size: int) -> ImageFont.ImageFont:
    font_path = Path("C:/Windows/Fonts/tahoma.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def pct(value: float | str | None) -> str:
    if value in ("", None):
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], fnt: ImageFont.ImageFont, fill: str = "#111111", max_chars: int = 95, line_gap: int = 7) -> int:
    x, y = xy
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            y += getattr(fnt, "size", 20) + line_gap
            continue
        for line in wrap(paragraph, max_chars):
            draw.text((x, y), line, font=fnt, fill=fill)
            y += getattr(fnt, "size", 20) + line_gap
    return y


def collect_versions() -> list[ModelVersion]:
    comparison = pd.read_csv(COMPARISON)
    versions: list[ModelVersion] = []
    explanations = {
        "V1": "V1 เป็น baseline XGBoost จาก clean_dataset.csv ใช้ feature จำนวนมากและมี customer history พื้นฐาน แต่ rolling history ยังไม่ละเอียดเท่า V2 จึงได้ Accuracy ระดับกลางและ Recall ต่ำ เพราะโมเดลค่อนข้าง conservative กับคลาส Return",
        "V2": "V2 คือ production candidate ปัจจุบัน ใช้ XGBoost safe plus rolling ตัด leakage หลังเหตุการณ์ และเพิ่ม rolling history 30/60/90/180/365 วัน ทำให้ Recall และ F1 ดีขึ้นชัดจาก V1 แม้ Accuracy เพิ่มไม่มาก",
        "V3": "V3 ใช้ stacking model จาก XGBoost, LightGBM และ CatBoost จุดเด่นคือจับเคส Return ได้มากขึ้นกว่า V1 แต่ Accuracy ลดลงเพราะโมเดลยอมทาย Return มากขึ้น ทำให้ false positive เพิ่ม",
        "V4": "V4 เป็น XGBoost + SMOTE + Optuna บน generated-data experiment ได้ Accuracy และ AUC สูงขึ้นจากการจูนและสมดุลคลาส แต่ Recall ไม่สูงเท่า V2/V3 จึงเหมาะเป็น experiment มากกว่าตัว production หลัก",
    }
    policies = {
        "V1": "baseline engineered features, partial order-time safe",
        "V2": "order-time-safe rolling history 30/60/90/180/365d",
        "V3": "stacking model using engineered feature set",
        "V4": "generated data + SMOTE + Optuna tuning",
    }
    origins = {
        "V1": "clean_dataset.csv, existing V1 evaluation artifact",
        "V2": "clean_dataset.csv, current V2 evaluation artifact",
        "V3": "V2 engineered feature set, stacking evaluation artifact",
        "V4": "generated-data experiment evaluation artifact",
    }
    for _, row in comparison.iterrows():
        display = str(row["display_version"])
        versions.append(
            ModelVersion(
                display_version=display,
                version_id=str(row["version"]),
                model=str(row["model"]),
                feature_policy=policies.get(display, ""),
                dataset_origin=origins.get(display, ""),
                original_accuracy=float(row["accuracy"]),
                original_recall=float(row["recall"]),
                original_precision=float(row["precision"]) if "precision" in row and pd.notna(row["precision"]) else 0.0,
                original_f1=float(row["f1"]),
                original_auc=float(row["auc"]),
                original_cost=float(row["cost"]),
                explanation=explanations.get(display, ""),
            )
        )

    if V5_METRICS.exists():
        v5 = pd.read_csv(V5_METRICS).iloc[0]
        versions.append(
            ModelVersion(
                display_version="V5",
                version_id=str(v5["version"]),
                model=str(v5["model"]),
                feature_policy="high-signal/generated dataset safe rolling experiment",
                dataset_origin="clean_dataset_v2_high_signal.csv / 50k high-signal experiment",
                original_accuracy=float(v5["accuracy"]),
                original_recall=float(v5["recall"]),
                original_precision=float(v5["precision"]),
                original_f1=float(v5["f1"]),
                original_auc=float(v5["auc"]),
                original_cost=float(v5["cost"]),
                explanation=(
                    "V5 คือ HIGH_ACCURACY archive/experiment ได้ Accuracy สูงเพราะใช้ชุดข้อมูล high-signal/generated ที่ pattern ระหว่าง feature กับ target ชัดกว่า "
                    "จึงทำให้ XGBoost จับสัญญาณได้ดีมาก แต่ไม่ควรนับเป็น V2 production current โดยตรง"
                ),
            )
        )
    return versions


def dataset_specs() -> list[dict]:
    return [
        {
            "folder": "dataset_5000",
            "dataset_size": 5000,
            "source": DATASETS_ROOT / "clean_dataset_5000" / "clean_dataset_5000_full_test.csv",
            "input_name": "clean_dataset_5000_full_test.csv",
            "title": "Dataset 5000",
            "description": "ใช้ clean_dataset.csv เดิมครบ 5,000 rows เป็น external full-test dataset",
        },
        {
            "folder": "dataset_50000",
            "dataset_size": 50000,
            "source": DATASETS_ROOT / "clean_dataset_generated_50000" / "clean_dataset_generated_50000_full_test.csv",
            "input_name": "clean_dataset_generated_50000_full_test.csv",
            "title": "Dataset 50000",
            "description": "generated แบบ stratified bootstrap จาก clean_dataset.csv จำนวน 50,000 rows เป็น external full-test dataset",
        },
    ]


def write_version_folder(base: Path, spec: dict, version: ModelVersion, shared_input: Path) -> dict:
    folder = base / f"version_{version.display_version[1:]}"
    folder.mkdir(parents=True, exist_ok=True)
    plan = {
        "dataset_size": spec["dataset_size"],
        "test_file": str(shared_input.relative_to(ROOT)),
        "version": version.display_version,
        "version_id": version.version_id,
        "model": version.model,
        "feature_policy": version.feature_policy,
        "dataset_origin": version.dataset_origin,
        "original_accuracy": version.original_accuracy,
        "original_recall": version.original_recall,
        "original_precision": version.original_precision,
        "original_f1": version.original_f1,
        "original_auc": version.original_auc,
        "original_cost": version.original_cost,
        "new_full_test_accuracy": "",
        "new_full_test_recall": "",
        "new_full_test_precision": "",
        "new_full_test_f1": "",
        "new_full_test_auc": "",
        "new_full_test_cost": "",
        "accuracy_delta": "",
        "status": "pending_saved_model_inference",
        "reason": "Need version-specific feature builder + saved model runtime dependencies before scoring full test data.",
    }
    pd.DataFrame([plan]).to_csv(folder / f"{version.display_version.lower()}_{spec['folder']}_evaluation_plan.csv", index=False, encoding="utf-8-sig")
    (folder / "model_input_path.txt").write_text(str(shared_input.relative_to(ROOT)), encoding="utf-8")
    (folder / "README.md").write_text(
        f"""# {version.display_version} on {spec['title']}

Model: `{version.model}`

Version id: `{version.version_id}`

Test input: `{shared_input.relative_to(ROOT)}`

Original Accuracy: `{pct(version.original_accuracy)}`

New full-test Accuracy: pending

Reason:

{version.explanation}

Next step:

1. Run feature engineering for this version on the full test file.
2. Load this version's saved model artifact.
3. Predict every row in the full test file.
4. Fill `new_full_test_accuracy` and `accuracy_delta` in the evaluation CSV.
""",
        encoding="utf-8",
    )
    return plan


def draw_accuracy_chart(base: Path, spec: dict, rows: list[dict]) -> None:
    images = base / "images"
    images.mkdir(parents=True, exist_ok=True)
    width, height = 1700, 980
    img = Image.new("RGB", (width, height), "#FFFFFF")
    d = ImageDraw.Draw(img)
    d.text((width // 2, 45), f"{spec['title']} Accuracy by Model Version", font=font(44), fill="#111111", anchor="ma")
    d.text((width // 2, 100), "Solid bars = original artifact accuracy | Full-test accuracy is pending saved-model inference", font=font(22), fill="#455A64", anchor="ma")

    x0, y0, x1, y1 = 145, 170, 1580, 800
    d.line((x0, y1, x1, y1), fill="#263238", width=2)
    d.line((x0, y0, x0, y1), fill="#263238", width=2)
    for tick in [0, 0.25, 0.50, 0.75, 1.0]:
        yy = int(y1 - tick * (y1 - y0))
        d.line((x0 - 6, yy, x1, yy), fill="#ECEFF1", width=1)
        d.text((x0 - 12, yy), f"{int(tick * 100)}", font=font(20), fill="#546E7A", anchor="rm")

    colors = ["#6D7C85", "#2E7D32", "#F9A825", "#1565C0", "#8E24AA"]
    bar_space = (x1 - x0 - 80) // len(rows)
    for i, row in enumerate(rows):
        acc = float(row["original_accuracy"])
        bx = x0 + 55 + i * bar_space
        bw = 115
        by = int(y1 - acc * (y1 - y0))
        d.rounded_rectangle((bx, by, bx + bw, y1), radius=7, fill=colors[i % len(colors)])
        d.text((bx + bw // 2, by - 10), pct(acc), font=font(24), fill="#111111", anchor="ms")
        d.text((bx + bw // 2, y1 + 18), row["version"], font=font(25), fill="#111111", anchor="ma")
        d.text((bx + bw // 2, y1 + 50), "pending", font=font(17), fill="#78909C", anchor="ma")

    d.text((145, 875), "หมายเหตุ: ยังไม่ได้ยิง full-test data เข้า saved model จริง เพราะ runtime ปัจจุบันขาด joblib/sklearn/xgboost และต้องผ่าน feature builder เฉพาะ version ก่อน", font=font(21), fill="#C62828")
    img.save(images / f"{spec['folder']}_accuracy_by_version.png")


def write_pdf_report(base: Path, spec: dict, rows: list[dict], versions: list[ModelVersion]) -> None:
    pages: list[Image.Image] = []
    page_w, page_h = 1800, 1300

    cover = Image.new("RGB", (page_w, page_h), "#FFFFFF")
    d = ImageDraw.Draw(cover)
    d.text((80, 55), f"{spec['title']} Accuracy Explanation Report", font=font(42), fill="#111111")
    y = draw_wrapped(
        d,
        f"{spec['description']} จุดประสงค์คือเอาข้อมูลทั้งก้อนไปทดสอบกับ Model Version 1-5 แล้วเปรียบเทียบ Accuracy ใหม่กับ Accuracy เดิมของแต่ละ version.",
        (80, 125),
        font(25),
        fill="#263238",
        max_chars=105,
    )
    chart = base / "images" / f"{spec['folder']}_accuracy_by_version.png"
    if chart.exists():
        im = Image.open(chart).convert("RGB")
        im.thumbnail((1640, 820), Image.Resampling.LANCZOS)
        cover.paste(im, ((page_w - im.width) // 2, y + 40))
    pages.append(cover)

    page = Image.new("RGB", (page_w, page_h), "#FFFFFF")
    d = ImageDraw.Draw(page)
    d.text((80, 55), "Model Version 1-5: ความแตกต่างและเหตุผลของ Accuracy", font=font(40), fill="#111111")
    y = 125
    for version in versions:
        y = draw_wrapped(
            d,
            f"{version.display_version} - {version.model}\nAccuracy เดิม: {pct(version.original_accuracy)} | Recall: {pct(version.original_recall)} | F1: {pct(version.original_f1)} | AUC: {pct(version.original_auc)}\n{version.explanation}\n",
            (80, y),
            font(21),
            fill="#111111",
            max_chars=120,
            line_gap=5,
        )
        y += 14
        if y > 1120:
            pages.append(page)
            page = Image.new("RGB", (page_w, page_h), "#FFFFFF")
            d = ImageDraw.Draw(page)
            y = 70
    y = draw_wrapped(
        d,
        "ข้อควรระวัง: ค่า Accuracy ที่แสดงในกราฟตอนนี้คือ original artifact accuracy ไม่ใช่ค่าใหม่จาก full-test dataset. ค่าใหม่จะถูกเติมหลังจากรัน inference ด้วย saved model และ feature builder ของแต่ละ version สำเร็จ.",
        (80, min(y + 10, 1140)),
        font(22),
        fill="#C62828",
        max_chars=110,
    )
    pages.append(page)

    pages[0].save(base / f"{spec['folder']}_accuracy_explanation_report.pdf", save_all=True, append_images=pages[1:], resolution=150)


def write_markdown_report(base: Path, spec: dict, versions: list[ModelVersion]) -> None:
    lines = [
        f"# {spec['title']} Accuracy Explanation Report",
        "",
        spec["description"],
        "",
        "## Current Status",
        "",
        "ยังไม่ได้ยิง full-test data เข้า saved model จริง เพราะ runtime ปัจจุบันขาด `joblib`, `sklearn`, `xgboost` และแต่ละ version ต้องผ่าน feature engineering ของตัวเองก่อน",
        "",
        "## Version Explanation",
        "",
    ]
    for version in versions:
        lines += [
            f"### {version.display_version}: {version.version_id}",
            "",
            f"- Model: `{version.model}`",
            f"- Original Accuracy: `{pct(version.original_accuracy)}`",
            f"- Original Recall: `{pct(version.original_recall)}`",
            f"- Original F1: `{pct(version.original_f1)}`",
            f"- Original AUC: `{pct(version.original_auc)}`",
            f"- Feature policy: `{version.feature_policy}`",
            "",
            version.explanation,
            "",
        ]
    (base / f"{spec['folder']}_accuracy_explanation_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_dataset_folder(spec: dict, versions: list[ModelVersion]) -> pd.DataFrame:
    base = TEST_ROOT / spec["folder"]
    input_dir = base / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shared_input = input_dir / spec["input_name"]
    shutil.copy2(spec["source"], shared_input)

    rows = []
    for version in versions:
        rows.append(write_version_folder(base, spec, version, shared_input))
    plan = pd.DataFrame(rows)
    plan.to_csv(base / f"{spec['folder']}_v1_to_v5_evaluation_plan.csv", index=False, encoding="utf-8-sig")
    draw_accuracy_chart(base, spec, rows)
    write_markdown_report(base, spec, versions)
    write_pdf_report(base, spec, rows, versions)

    (base / "README.md").write_text(
        f"""# {spec['title']}

{spec['description']}

Shared input file:

- `input/{spec['input_name']}`

Version folders:

- `version_1`
- `version_2`
- `version_3`
- `version_4`
- `version_5`

Main files:

- `{spec['folder']}_v1_to_v5_evaluation_plan.csv`
- `images/{spec['folder']}_accuracy_by_version.png`
- `{spec['folder']}_accuracy_explanation_report.pdf`
- `{spec['folder']}_accuracy_explanation_report.md`

ตอนนี้ Accuracy ใหม่จาก full-test dataset ยัง pending เพราะต้องรัน saved-model inference ใน environment ที่มี `joblib`, `sklearn`, `xgboost` และต้องผ่าน feature builder ของแต่ละ version ก่อน
""",
        encoding="utf-8",
    )
    return plan


def main() -> None:
    versions = collect_versions()
    all_rows = []
    for spec in dataset_specs():
        plan = build_dataset_folder(spec, versions)
        all_rows.append(plan)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(TEST_ROOT / "dataset_5000_50000_v1_to_v5_evaluation_plan.csv", index=False, encoding="utf-8-sig")
    print(f"Built dataset test folders under {TEST_ROOT}")


if __name__ == "__main__":
    main()
