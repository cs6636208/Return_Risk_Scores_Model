from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_PLAN = ROOT / "docs" / "test" / "dataset_5000_50000_v1_to_v5_evaluation_plan.csv"
OUT_DIR = ROOT / "docs" / "test" / "model_full_test_results"
OUT_CSV = OUT_DIR / "full_test_model_evaluation_results.csv"


REQUIRED_PACKAGES = ["joblib", "sklearn", "xgboost"]


def dependency_status() -> dict[str, bool]:
    import importlib.util

    return {package: importlib.util.find_spec(package) is not None for package in REQUIRED_PACKAGES}


def write_blocked_results(reason: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if TEST_PLAN.exists():
        plan = pd.read_csv(TEST_PLAN)
    else:
        plan = pd.DataFrame()

    if plan.empty:
        rows = [{"status": "blocked", "reason": reason}]
    else:
        rows = []
        for _, row in plan.iterrows():
            item = row.to_dict()
            item.update(
                {
                    "new_test_accuracy": "",
                    "new_test_recall": "",
                    "new_test_precision": "",
                    "new_test_f1": "",
                    "new_test_auc": "",
                    "new_test_cost": "",
                    "accuracy_delta": "",
                    "status": "blocked_missing_ml_dependency",
                    "reason": reason,
                }
            )
            rows.append(item)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(reason)
    print(f"Saved blocked status to {OUT_CSV}")


def main() -> None:
    status = dependency_status()
    missing = [package for package, exists in status.items() if not exists]
    if missing:
        write_blocked_results(
            "Cannot run saved-model inference because these packages are missing: "
            + ", ".join(missing)
            + ". Install requirements.txt in the project environment, then rerun this script."
        )
        return

    # The inference implementation intentionally stops here unless the project
    # environment has the ML stack. This prevents accidentally creating fake
    # metrics. The next step is to run each version-specific feature builder,
    # load each model artifact, predict every row in the full test CSV, then
    # write actual metrics into OUT_CSV.
    write_blocked_results(
        "ML dependencies are available, but version-specific raw-data feature adapters still need to be wired before scoring saved model artifacts. "
        "Use scripts/run_same_xgboost_feature_version_benchmark.py if the goal is same-model retraining comparison."
    )


if __name__ == "__main__":
    main()
