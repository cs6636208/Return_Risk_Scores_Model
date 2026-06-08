from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_lightgbm_setc_s1_train_test_split_pipeline.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("lightgbm_setc_base_pipeline", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base LightGBM pipeline: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lightgbm_setc_base_pipeline"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    pipeline = load_base_module()
    pipeline.SOURCE_CANDIDATES = [
        ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "clean_dataset_s2.csv",
        ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S2" / "clean_dataset_s2.csv",
    ]
    pipeline.OUT_ROOT = ROOT / "docs" / "LightGBM" / "SETC" / "clean_dataset" / "S2"
    pipeline.SOURCE_COPY = pipeline.OUT_ROOT / "clean_dataset_s2.csv"
    pipeline.SUMMARY_NAME = "lgbm_s2_v1_to_v5_train_test_split_summary"
    pipeline.MODEL_PREFIX = "lgbm_s2"
    pipeline.CHART_NAME = "lgbm_s2_train_test_split_accuracy_v1_to_v5.png"
    pipeline.DATASET_LABEL = "LightGBM_SETC_S2"
    pipeline.DATASET_DISPLAY = "LightGBM SETC S2"
    pipeline.EXPECTED_ROWS = 50_000
    pipeline.SOURCE_FILE_NAME = "clean_dataset_s2.csv"
    pipeline.BASE_LGBM_PARAMS = {
        **pipeline.BASE_LGBM_PARAMS,
        "n_estimators": 620,
        "learning_rate": 0.03,
        "num_leaves": 35,
        "min_child_samples": 70,
    }
    pipeline.main()


if __name__ == "__main__":
    main()
