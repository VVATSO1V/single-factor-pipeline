"""Leakage-safe daily cross-sectional feature preprocessing."""

from __future__ import annotations

from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd


KEY_COLUMNS = ("date", "stock_code")
MAD_NORMALIZATION = 1.4826


def derived_feature_columns(factor_columns: Sequence[str]) -> list[str]:
    """Return the stable output order for factor values and missing flags."""
    factors = list(factor_columns)
    if not factors or len(factors) != len(set(factors)):
        raise ValueError("factor_columns must be non-empty and unique")
    conflicts = sorted(set(factors).intersection(KEY_COLUMNS))
    if conflicts:
        raise ValueError(f"factor columns conflict with keys: {conflicts}")
    return [*factors, *(f"{factor}__missing" for factor in factors)]


def _validate_frame(
    frame: pd.DataFrame,
    factor_columns: list[str],
    expected_cross_section_size: int,
) -> pd.Series:
    required = {*KEY_COLUMNS, *factor_columns}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"missing preprocessing columns: {missing_columns}")
    if frame.empty:
        raise ValueError("preprocessing frame is empty")

    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    stock_codes = frame["stock_code"].astype("string").str.strip()
    if dates.isna().any() or stock_codes.isna().any() or stock_codes.eq("").any():
        raise ValueError("preprocessing keys cannot be missing")

    keys = pd.DataFrame({"date": dates, "stock_code": stock_codes})
    duplicate_mask = keys.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate_mask.any():
        sample = keys.loc[duplicate_mask].head(5).to_dict("records")
        raise ValueError(f"duplicate preprocessing keys: {sample}")

    counts = dates.value_counts(sort=False)
    incomplete = counts[counts.ne(expected_cross_section_size)]
    if not incomplete.empty:
        sample = {
            timestamp.strftime("%Y-%m-%d"): int(count)
            for timestamp, count in incomplete.head(5).items()
        }
        raise ValueError(
            "every daily cross-section must contain exactly "
            f"{expected_cross_section_size} rows: {sample}"
        )
    return dates


def _transform_one_date(
    values: np.ndarray,
    mad_width: float,
) -> np.ndarray:
    finite = np.isfinite(values)
    clean = np.where(finite, values, np.nan)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        median = np.nanmedian(clean, axis=0)
        mad = np.nanmedian(np.abs(clean - median), axis=0)

    clip_scale = mad_width * MAD_NORMALIZATION * mad
    clipable = np.isfinite(clip_scale) & (clip_scale > 0)
    lower = np.where(clipable, median - clip_scale, -np.inf)
    upper = np.where(clipable, median + clip_scale, np.inf)
    clipped = np.clip(clean, lower, upper)

    counts = np.isfinite(clipped).sum(axis=0)
    sums = np.nansum(clipped, axis=0)
    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float64),
        where=counts > 0,
    )
    squared_deviation = np.where(
        np.isfinite(clipped),
        np.square(clipped - means),
        0.0,
    )
    variances = np.divide(
        squared_deviation.sum(axis=0),
        counts,
        out=np.zeros_like(sums, dtype=np.float64),
        where=counts > 0,
    )
    standard_deviations = np.sqrt(variances)
    scalable = (
        (counts >= 2)
        & np.isfinite(standard_deviations)
        & (standard_deviations > 0)
    )

    zscores = np.zeros_like(clean, dtype=np.float64)
    zscores[:, scalable] = (
        clipped[:, scalable] - means[scalable]
    ) / standard_deviations[scalable]
    zscores[~finite] = 0.0
    return np.concatenate(
        [zscores, (~finite).astype(np.float64)],
        axis=1,
    ).astype(np.float32, copy=False)


def transform_daily_cross_sections(
    frame: pd.DataFrame,
    factor_columns: Sequence[str],
    *,
    mad_width: float,
    expected_cross_section_size: int,
    date_chunk_size: int,
) -> tuple[np.ndarray, list[str]]:
    """Transform complete T-day stock cross-sections without using labels."""
    factors = list(factor_columns)
    output_columns = derived_feature_columns(factors)
    if mad_width <= 0:
        raise ValueError("mad_width must be positive")
    if expected_cross_section_size <= 0:
        raise ValueError("expected_cross_section_size must be positive")
    if date_chunk_size <= 0:
        raise ValueError("date_chunk_size must be positive")

    dates = _validate_frame(frame, factors, expected_cross_section_size)
    try:
        raw_values = frame[factors].to_numpy(dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("factor values must be numeric") from exc

    result = np.empty(
        (len(frame), len(output_columns)),
        dtype=np.float32,
    )
    date_codes, unique_dates = pd.factorize(dates, sort=False)
    date_order = np.argsort(date_codes, kind="stable")
    date_counts = np.bincount(date_codes, minlength=len(unique_dates))
    date_offsets = np.concatenate(([0], np.cumsum(date_counts)))
    for chunk_start in range(0, len(unique_dates), date_chunk_size):
        chunk_end = min(chunk_start + date_chunk_size, len(unique_dates))
        for date_code in range(chunk_start, chunk_end):
            positions = date_order[
                date_offsets[date_code] : date_offsets[date_code + 1]
            ]
            result[positions] = _transform_one_date(
                raw_values[positions],
                mad_width,
            )
        print(
            f"preprocessing progress: {chunk_end}/{len(unique_dates)} dates",
            flush=True,
        )
    return result, output_columns
