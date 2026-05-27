from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PRODUCTION_DROP_COLUMNS = {
    "return_id",
    "return_date",
    "return_reason",
    "refund_amount",
    "refund_date",
    "delivery_days",
    "delay_days",
    "risk_score",
    "risk_tier",
    "shap_values",
}


def _clean_key(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__").astype(str)


class SmoothTargetEncoder:
    def __init__(self, m: float = 10.0, smoothing: float | None = None) -> None:
        self.m = float(m if smoothing is None else smoothing)
        self.global_mean = 0.0
        self.category_means: dict[str, dict[str, float]] = {}

    def fit(self, x: pd.DataFrame, y: Iterable[int], columns: Iterable[str]) -> "SmoothTargetEncoder":
        frame = x.copy()
        target = pd.Series(y, index=frame.index).astype(float)
        self.global_mean = float(target.mean()) if len(target) else 0.0
        self.category_means = {}
        for col in columns:
            keys = _clean_key(frame[col])
            stats = target.groupby(keys).agg(["mean", "count"])
            encoded = (
                (stats["mean"] * stats["count"] + self.global_mean * self.m)
                / (stats["count"] + self.m)
            )
            self.category_means[col] = encoded.astype(float).to_dict()
        return self

    def transform_column(self, series: pd.Series, column: str) -> pd.Series:
        keys = _clean_key(series)
        category_means = getattr(self, "category_means", None) or getattr(self, "mappings_", {})
        global_mean = getattr(self, "global_mean", getattr(self, "global_mean_", 0.0))
        return keys.map(category_means.get(column, {})).fillna(global_mean).astype(float)


@dataclass
class TabularTargetEncodingPreprocessor:
    target_encode_columns: Iterable[str] | None = None
    fill_value: float = 0.0
    scaler: StandardScaler = field(default_factory=StandardScaler)
    numeric_medians: dict[str, float] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    encoder: SmoothTargetEncoder = field(default_factory=SmoothTargetEncoder)
    fitted: bool = False

    def fit(self, x: pd.DataFrame, y: Iterable[int]) -> "TabularTargetEncodingPreprocessor":
        frame = x.copy()
        target = pd.Series(y, index=frame.index).astype(float)
        self.global_mean_ = float(target.mean()) if len(target) else 0.0

        if self.target_encode_columns is None:
            self.target_encode_columns = [
                col
                for col in frame.columns
                if not pd.api.types.is_numeric_dtype(frame[col])
            ]
        else:
            self.target_encode_columns = [col for col in self.target_encode_columns if col in frame.columns]

        self.encoder = SmoothTargetEncoder()
        self.encoder.fit(frame, target, self.target_encode_columns)

        numeric_columns = [col for col in frame.columns if col not in self.target_encode_columns]
        self.numeric_medians = {}
        for col in numeric_columns:
            values = pd.to_numeric(frame[col], errors="coerce")
            median = values.median()
            self.numeric_medians[col] = float(median) if not pd.isna(median) else self.fill_value

        self.feature_columns = list(frame.columns)
        transformed = self._transform_unscaled(frame)
        self.scaler.fit(transformed)
        self.fitted = True
        return self

    def _transform_unscaled(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = x.copy()
        out = pd.DataFrame(index=frame.index)
        target_encode_columns = list(getattr(self, "target_encode_columns", []) or [])
        numeric_medians = getattr(self, "numeric_medians", {})
        for col in self.feature_columns:
            if col not in frame.columns:
                out[col] = self.fill_value
                continue
            if col in target_encode_columns:
                out[col] = self.encoder.transform_column(frame[col], col)
            else:
                out[col] = (
                    pd.to_numeric(frame[col], errors="coerce")
                    .fillna(numeric_medians.get(col, self.fill_value))
                    .astype(float)
                )
        return out

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        transformed = self._transform_unscaled(x)
        scaled = self.scaler.transform(transformed)
        return pd.DataFrame(scaled, columns=self.feature_columns, index=transformed.index)

    def fit_transform(self, x: pd.DataFrame, y: Iterable[int]) -> pd.DataFrame:
        return self.fit(x, y).transform(x)


@dataclass
class TargetEncodedModelPipeline:
    preprocessor: TabularTargetEncodingPreprocessor
    model: object

    def fit(self, x: pd.DataFrame, y: Iterable[int]) -> "TargetEncodedModelPipeline":
        transformed = self.preprocessor.fit_transform(x, y)
        self.model.fit(transformed, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.preprocessor.transform(x))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.preprocessor.transform(x))


class V2ProdPreprocessor:
    def __init__(self) -> None:
        self.encoder = TabularTargetEncodingPreprocessor()
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []

    def _select_features(self, x: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [col for col in PRODUCTION_DROP_COLUMNS.union({"is_returned"}) if col in x.columns]
        return x.drop(columns=drop_cols, errors="ignore")

    def fit_transform(self, x: pd.DataFrame, y: Iterable[int]) -> pd.DataFrame:
        selected = self._select_features(x)
        encoded = self.encoder.fit_transform(selected, y)
        self.feature_columns = list(encoded.columns)
        scaled = self.scaler.fit_transform(encoded)
        return pd.DataFrame(scaled, columns=self.feature_columns, index=encoded.index)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        selected = self._select_features(x)
        encoded = self.encoder.transform(selected)
        scaled = self.scaler.transform(encoded)
        return pd.DataFrame(scaled, columns=self.feature_columns, index=encoded.index)
