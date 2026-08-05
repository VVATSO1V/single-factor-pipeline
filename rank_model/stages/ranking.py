"""Shared date-group layout helpers for native ranking models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sorted_group_layout(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a stable date/stock ordered frame and one group size per date."""
    required_columns = {"date", "stock_code"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"ranking data is missing group keys: {missing}")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    if result["date"].isna().any() or result["stock_code"].isna().any():
        raise ValueError("ranking group keys must be non-null")
    result = result.sort_values("stock_code", kind="mergesort")
    result = result.sort_values("date", kind="mergesort")
    result = result.reset_index(drop=True)
    group_sizes = (
        result.groupby("date", sort=False, observed=True).size().to_numpy(dtype="int64")
    )
    if int(group_sizes.sum()) != len(result):
        raise ValueError("ranking group sizes do not cover every row")
    return result, group_sizes


def lightgbm_relevance(rank_target: pd.Series) -> np.ndarray:
    """Map rank percentiles to LightGBM's fixed 100-level relevance labels."""
    values = pd.to_numeric(rank_target, errors="raise").to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("LightGBM ranking labels must be finite")
    return np.clip(np.floor(values * 100.0), 0, 99).astype("int32")
