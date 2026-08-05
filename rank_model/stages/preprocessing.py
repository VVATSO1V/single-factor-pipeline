"""Train-only feature preprocessing for the rank-model pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from rank_model.stages.dataset import FORBIDDEN_FEATURE_COLUMNS, _is_forbidden_feature


UNKNOWN_INDUSTRY = "UNKNOWN"


def _normalized_industries(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    missing = values.isna() | values.eq("") | values.str.lower().eq("unknown")
    return values.mask(missing, UNKNOWN_INDUSTRY).astype("string")


def _validate_feature_columns(
    continuous_columns: list[str],
    industry_column: str,
) -> list[str]:
    columns = list(continuous_columns)
    feature_columns = [*columns, industry_column]
    if (
        not columns
        or not all(isinstance(column, str) and column for column in feature_columns)
        or len(feature_columns) != len(set(feature_columns))
    ):
        raise ValueError("feature columns must be non-empty and unique")
    forbidden = sorted(
        column
        for column in feature_columns
        if column.lower() in FORBIDDEN_FEATURE_COLUMNS or _is_forbidden_feature(column)
    )
    if forbidden:
        raise ValueError(f"feature columns contain forbidden values: {forbidden}")
    return columns


@dataclass
class RankPreprocessor:
    continuous_columns: list[str]
    industry_column: str
    industry_categories: list[str]
    scaler: StandardScaler

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        continuous_columns: list[str],
        industry_column: str,
    ) -> RankPreprocessor:
        """Fit continuous scaling and industry categories from training rows only."""
        columns = _validate_feature_columns(continuous_columns, industry_column)
        required = set([*columns, industry_column])
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"rank fit is missing columns: {missing}")
        if frame.empty:
            raise ValueError("rank preprocessor training frame is empty")
        continuous = frame[columns].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(continuous).all():
            raise ValueError("rank continuous training features must be finite")
        scaler = StandardScaler(copy=True).fit(continuous)
        observed = sorted(
            set(_normalized_industries(frame[industry_column]).astype(str))
            - {UNKNOWN_INDUSTRY}
        )
        return cls(
            continuous_columns=columns,
            industry_column=industry_column,
            industry_categories=[*observed, UNKNOWN_INDUSTRY],
            scaler=scaler,
        )

    @property
    def feature_names(self) -> list[str]:
        return [
            *self.continuous_columns,
            *(
                f"{self.industry_column}__{category}"
                for category in self.industry_categories
            ),
        ]

    def transform(self, frame: pd.DataFrame, scale_continuous: bool) -> np.ndarray:
        required = set([*self.continuous_columns, self.industry_column])
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"rank transform is missing columns: {missing}")
        continuous = frame[self.continuous_columns].to_numpy(
            dtype=np.float64, copy=True
        )
        if not np.isfinite(continuous).all():
            raise ValueError("rank continuous features must be finite")
        if scale_continuous:
            continuous = self.scaler.transform(continuous)
        industries = _normalized_industries(frame[self.industry_column])
        categories = set(self.industry_categories)
        industries = industries.where(industries.isin(categories), UNKNOWN_INDUSTRY)
        category_positions = {
            category: position
            for position, category in enumerate(self.industry_categories)
        }
        one_hot = np.zeros((len(frame), len(self.industry_categories)), dtype=np.float32)
        positions = industries.map(category_positions).to_numpy(dtype=np.int64)
        one_hot[np.arange(len(frame)), positions] = 1.0
        return np.concatenate(
            [continuous.astype(np.float32, copy=False), one_hot], axis=1
        )


def equal_date_weights(dates: pd.Series) -> np.ndarray:
    normalized = pd.to_datetime(dates, errors="raise").dt.normalize()
    counts = normalized.groupby(normalized).transform("size").to_numpy(dtype="float64")
    if np.any(counts <= 0):
        raise ValueError("date weights require non-empty groups")
    return 1.0 / counts


def predicted_percentiles(scores: np.ndarray, dates: pd.Series) -> np.ndarray:
    normalized_dates = pd.to_datetime(dates, errors="raise").dt.normalize()
    if normalized_dates.isna().any():
        raise ValueError("prediction dates must be non-missing")
    frame = pd.DataFrame({"date": normalized_dates, "score": scores})
    if not np.isfinite(frame["score"]).all():
        raise ValueError("prediction scores must be finite")
    return frame.groupby("date", sort=False)["score"].transform(
        lambda values: (values.rank(method="average") - 1.0)
        / max(len(values) - 1, 1)
    ).to_numpy(dtype="float64")
