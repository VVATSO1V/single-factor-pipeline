"""Static 2024-2025 dataset, inference, and evaluation for frozen rank models."""

from __future__ import annotations

from collections.abc import Sequence
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any, BinaryIO
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from rank_model.stages.dataset import (
    EXIT_DATE_COLUMN,
    KEY_COLUMNS,
    RANK_TARGET_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    _is_forbidden_feature,
    file_sha256,
    percentile_rank_from_returns,
)
from rank_model.stages.evaluation import (
    compare_runs,
    evaluate_predictions,
    write_evaluation,
)
from rank_model.stages.finalization import rank_schema_contract_sha256
from rank_model.stages.preprocessing import RankPreprocessor, predicted_percentiles
from rank_model.stages.training import _load_persisted_model, _predict_model


TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-12-31")
UNKNOWN_INDUSTRY = "UNKNOWN"
MAD_NORMALIZATION = 1.4826
LIMIT_TOLERANCE = 1e-8
LOCKED_MAD_WIDTH = 3.0
LOCKED_WINDOWS = (1, 5, 20)
LOCKED_CAP_COVERAGE_THRESHOLD = 0.95
LOCKED_MODEL_NAMES = (
    "ridge_rank_regression",
    "xgboost_rank_regression",
    "lightgbm_rank_regression",
    "lightgbm_lambdarank",
    "mlp_top100_hybrid_rank",
)
LEGACY_FEATURE_ENGINEERING_CONTRACTS: dict[str, str] = {
    # The first real locked-test artifact predates the explicit code-hash field.
    "a157a33d437b417e0fe43be540418b432118b8c4f8c233dcc83978dc709ec99d": (
        "c9bc828a2cd7e605af55ae0256d5e5dd2a1200c43e1fb3d7d546297981614a78"
    ),
}


def _acquire_process_lock(path: Path) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if handle.seek(0, os.SEEK_END) == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise FileExistsError(f"another locked-test command holds: {path}") from error
    return handle


def _release_process_lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _normalize_industry(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    missing = values.isna() | values.eq("") | values.str.lower().eq("unknown")
    return values.mask(missing, UNKNOWN_INDUSTRY).astype("string")


def _validate_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = sorted(set(KEY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing key columns: {missing}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["stock_code"] = result["stock_code"].astype("string").str.strip()
    if result[KEY_COLUMNS].isna().any().any() or result["stock_code"].eq("").any():
        raise ValueError(f"{label} contains missing keys")
    if result.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{label} contains duplicate keys")
    return result.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _validate_daily_size(
    frame: pd.DataFrame,
    expected_cross_section_size: int,
    label: str,
) -> None:
    counts = frame.groupby("date", sort=False).size()
    invalid = counts.loc[counts.ne(expected_cross_section_size)]
    if not invalid.empty:
        sample = {
            pd.Timestamp(timestamp).strftime("%Y-%m-%d"): int(count)
            for timestamp, count in invalid.head(5).items()
        }
        raise ValueError(
            f"{label} must contain exactly {expected_cross_section_size} rows "
            f"per date: {sample}"
        )


def _transform_one_date(values: np.ndarray, mad_width: float) -> np.ndarray:
    finite = np.isfinite(values)
    clean = np.where(finite, values, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        median = np.nanmedian(clean, axis=0)
        mad = np.nanmedian(np.abs(clean - median), axis=0)
    clip_scale = mad_width * MAD_NORMALIZATION * mad
    clipable = np.isfinite(clip_scale) & (clip_scale > 0)
    clipped = np.clip(
        clean,
        np.where(clipable, median - clip_scale, -np.inf),
        np.where(clipable, median + clip_scale, np.inf),
    )
    counts = np.isfinite(clipped).sum(axis=0)
    sums = np.nansum(clipped, axis=0)
    means = np.divide(
        sums, counts, out=np.zeros_like(sums, dtype=np.float64), where=counts > 0
    )
    squared = np.where(np.isfinite(clipped), np.square(clipped - means), 0.0)
    variances = np.divide(
        squared.sum(axis=0),
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
    return np.concatenate([zscores, (~finite).astype(np.float64)], axis=1).astype(
        np.float32, copy=False
    )


def _transform_daily_cross_sections(
    frame: pd.DataFrame,
    factor_columns: Sequence[str],
    *,
    mad_width: float,
    expected_cross_section_size: int,
    date_chunk_size: int,
) -> tuple[np.ndarray, list[str]]:
    factors = list(factor_columns)
    if not factors or len(factors) != len(set(factors)):
        raise ValueError("factor columns must be non-empty and unique")
    if mad_width <= 0 or date_chunk_size <= 0:
        raise ValueError("mad width and date chunk size must be positive")
    _validate_daily_size(frame, expected_cross_section_size, "test factor data")
    values = frame[factors].to_numpy(dtype=np.float64, copy=True)
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    output = np.empty((len(frame), len(factors) * 2), dtype=np.float32)
    date_codes, unique_dates = pd.factorize(dates, sort=False)
    date_order = np.argsort(date_codes, kind="stable")
    date_counts = np.bincount(date_codes, minlength=len(unique_dates))
    offsets = np.concatenate(([0], np.cumsum(date_counts)))
    for chunk_start in range(0, len(unique_dates), date_chunk_size):
        chunk_end = min(chunk_start + date_chunk_size, len(unique_dates))
        for date_code in range(chunk_start, chunk_end):
            positions = date_order[offsets[date_code] : offsets[date_code + 1]]
            output[positions] = _transform_one_date(values[positions], mad_width)
        print(
            f"locked-test preprocessing progress: {chunk_end}/{len(unique_dates)} dates",
            flush=True,
        )
    return output, [*factors, *(f"{column}__missing" for column in factors)]


def _compound_return(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=window).apply(
        lambda item: float(np.prod(1.0 + item) - 1.0), raw=True
    )


def _fill_numeric_missing(frame: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if column in keys or not pd.api.types.is_numeric_dtype(result[column]):
            continue
        result[f"{column}__missing"] = result[column].isna().astype(np.float32)
        result[column] = result[column].fillna(0.0).astype(np.float64)
    return result


def _build_daily_context(
    market: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    windows: Sequence[int],
    cap_coverage_threshold: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "date",
        "stock_code",
        "in_universe",
        "post_close",
        "raw_close",
        "limit_up",
        "limit_down",
        "is_st",
        "is_suspended",
        "industry",
        "market_cap",
    }
    missing = sorted(required.difference(market.columns))
    if missing:
        raise ValueError(f"market context input is missing columns: {missing}")
    selected_windows = tuple(int(value) for value in windows)
    if (
        not selected_windows
        or len(selected_windows) != len(set(selected_windows))
        or any(value <= 0 for value in selected_windows)
    ):
        raise ValueError("windows must contain unique positive integers")
    panel = _validate_keys(market, "market context")
    panel["industry"] = _normalize_industry(panel["industry"])
    for column in ("post_close", "raw_close", "limit_up", "limit_down", "market_cap"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    for column in ("in_universe", "is_st", "is_suspended"):
        panel[column] = panel[column].astype("boolean").fillna(False).astype(bool)
    panel = panel.sort_values(["stock_code", "date"]).reset_index(drop=True)
    previous_close = panel.groupby("stock_code", sort=False)["post_close"].shift(1)
    valid_price = panel["post_close"].gt(0) & previous_close.gt(0)
    panel["stock_return_1d"] = np.where(
        valid_price, panel["post_close"] / previous_close - 1.0, np.nan
    )
    panel["valid_market_cap"] = panel["market_cap"].where(panel["market_cap"].gt(0))
    panel["log_market_cap"] = np.log1p(panel["valid_market_cap"])
    universe = panel.loc[panel["in_universe"]].copy()
    if universe.empty:
        raise ValueError("market context has no in-universe rows")

    market_rows: list[dict[str, Any]] = []
    industry_rows: list[dict[str, Any]] = []
    for timestamp, daily in universe.groupby("date", sort=True):
        returns = daily["stock_return_1d"]
        valid_returns = returns.notna()
        caps = daily["valid_market_cap"]
        valid_cap_return = valid_returns & caps.notna()
        count = len(daily)
        cap_coverage = float(valid_cap_return.sum() / count)
        equal_return = float(returns.mean()) if valid_returns.any() else np.nan
        cap_return = (
            float(np.average(returns.loc[valid_cap_return], weights=caps.loc[valid_cap_return]))
            if valid_cap_return.any()
            else np.nan
        )
        cap_fallback = cap_coverage < cap_coverage_threshold or not np.isfinite(cap_return)
        if cap_fallback:
            cap_return = equal_return
        valid_limit = (
            daily["raw_close"].gt(0)
            & daily["limit_up"].gt(0)
            & daily["limit_down"].gt(0)
            & daily["limit_up"].gt(daily["limit_down"])
        )
        valid_log_caps = daily["log_market_cap"].dropna()
        market_rows.append(
            {
                "date": timestamp,
                "market_return_equal_daily": equal_return,
                "market_return_cap_daily": cap_return,
                "market_up_ratio": float(returns.loc[valid_returns].gt(0).mean())
                if valid_returns.any()
                else np.nan,
                "market_return_std": float(returns.loc[valid_returns].std(ddof=0))
                if valid_returns.any()
                else np.nan,
                "market_return_q10": float(returns.loc[valid_returns].quantile(0.10))
                if valid_returns.any()
                else np.nan,
                "market_return_q50": float(returns.loc[valid_returns].quantile(0.50))
                if valid_returns.any()
                else np.nan,
                "market_return_q90": float(returns.loc[valid_returns].quantile(0.90))
                if valid_returns.any()
                else np.nan,
                "market_limit_up_ratio": float(
                    daily.loc[valid_limit, "raw_close"]
                    .ge(daily.loc[valid_limit, "limit_up"] - LIMIT_TOLERANCE)
                    .mean()
                )
                if valid_limit.any()
                else np.nan,
                "market_limit_down_ratio": float(
                    daily.loc[valid_limit, "raw_close"]
                    .le(daily.loc[valid_limit, "limit_down"] + LIMIT_TOLERANCE)
                    .mean()
                )
                if valid_limit.any()
                else np.nan,
                "market_suspended_ratio": float(daily["is_suspended"].mean()),
                "market_st_ratio": float(daily["is_st"].mean()),
                "market_log_cap_mean": float(valid_log_caps.mean())
                if not valid_log_caps.empty
                else np.nan,
                "market_log_cap_median": float(valid_log_caps.median())
                if not valid_log_caps.empty
                else np.nan,
                "market_log_cap_std": float(valid_log_caps.std(ddof=0))
                if not valid_log_caps.empty
                else np.nan,
                "market_cap_coverage": cap_coverage,
                "market_limit_data_coverage": float(valid_limit.sum() / count),
                "market_return_coverage": float(valid_returns.sum() / count),
                "market_cap_weight_fallback": float(cap_fallback),
            }
        )
        total_cap = float(caps.sum(min_count=1))
        known = daily.loc[~daily["industry"].eq(UNKNOWN_INDUSTRY)]
        for industry, group in known.groupby("industry", sort=True, dropna=False):
            group_returns = group["stock_return_1d"]
            valid_group_returns = group_returns.notna()
            group_caps = group["valid_market_cap"]
            valid_group_caps = group_caps.notna()
            values = group_returns.loc[valid_group_returns]
            industry_rows.append(
                {
                    "date": timestamp,
                    "industry": str(industry),
                    "industry_return_equal_daily": float(values.mean())
                    if not values.empty
                    else np.nan,
                    "industry_up_ratio": float(values.gt(0).mean())
                    if not values.empty
                    else np.nan,
                    "industry_return_std": float(values.std(ddof=0))
                    if not values.empty
                    else np.nan,
                    "industry_return_qspread": float(
                        values.quantile(0.90) - values.quantile(0.10)
                    )
                    if not values.empty
                    else np.nan,
                    "industry_stock_count": float(len(group)),
                    "industry_log_cap_median": float(group["log_market_cap"].median())
                    if valid_group_caps.any()
                    else np.nan,
                    "industry_cap_share": float(group_caps.sum() / total_cap)
                    if np.isfinite(total_cap) and total_cap > 0
                    else np.nan,
                    "industry_suspended_ratio": float(group["is_suspended"].mean()),
                    "industry_return_coverage": float(valid_group_returns.sum() / len(group)),
                    "industry_cap_coverage": float(valid_group_caps.sum() / len(group)),
                }
            )

    official = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    market_context = pd.DataFrame(market_rows).set_index("date").reindex(official)
    market_context.index.name = "date"
    for window in selected_windows:
        market_context[f"market_return_equal_{window}d"] = _compound_return(
            market_context["market_return_equal_daily"], window
        )
        market_context[f"market_return_cap_{window}d"] = _compound_return(
            market_context["market_return_cap_daily"], window
        )
    market_context = market_context.drop(
        columns=["market_return_equal_daily", "market_return_cap_daily"]
    ).reset_index()

    industry_daily = pd.DataFrame(industry_rows)
    industries = pd.Index(sorted(industry_daily["industry"].unique()), name="industry")
    industry_index = pd.MultiIndex.from_product(
        [official, industries], names=["date", "industry"]
    )
    industry_context = industry_daily.set_index(["date", "industry"]).reindex(
        industry_index
    )
    market_raw = market_context.set_index("date")
    for window in selected_windows:
        industry_context[f"industry_return_equal_{window}d"] = industry_context.groupby(
            level="industry", sort=False
        )["industry_return_equal_daily"].transform(
            lambda values: _compound_return(values, window)
        )
        market_values = market_raw[f"market_return_equal_{window}d"].reindex(
            industry_context.index.get_level_values("date")
        ).to_numpy()
        industry_context[f"industry_excess_return_{window}d"] = (
            industry_context[f"industry_return_equal_{window}d"].to_numpy()
            - market_values
        )
    industry_context = industry_context.drop(
        columns="industry_return_equal_daily"
    ).reset_index()
    return (
        _fill_numeric_missing(market_context, {"date"}),
        _fill_numeric_missing(industry_context, {"date", "industry"}),
    )


def build_locked_test_frame(
    model: pd.DataFrame,
    sample: pd.DataFrame,
    market: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    factor_columns: Sequence[str],
    expected_continuous_columns: Sequence[str] | None,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    mad_width: float,
    expected_cross_section_size: int,
    date_chunk_size: int,
    windows: Sequence[int] = (1, 5, 20),
    cap_coverage_threshold: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a rank-model test frame without fitting any model component."""
    data = _validate_keys(model, "full model dataset")
    index = _validate_keys(sample, "sample index")
    start = pd.Timestamp(test_start).normalize()
    end = pd.Timestamp(test_end).normalize()
    if start != TEST_START or end != TEST_END:
        raise ValueError("locked test window must be 2024-01-01 through 2025-12-31")
    selected = data["date"].between(start, end)
    data = data.loc[selected].reset_index(drop=True)
    index = index.loc[index["date"].between(start, end)].reset_index(drop=True)
    if data.empty:
        raise ValueError("full model dataset has no locked-test rows")
    _validate_daily_size(data, expected_cross_section_size, "locked test data")
    if len(data) != len(index) or not data[KEY_COLUMNS].equals(index[KEY_COLUMNS]):
        raise ValueError("locked test model and sample-index keys differ")
    required_model = {
        *factor_columns,
        TARGET_COLUMN,
        "industry",
        "market_cap",
    }
    missing_model = sorted(required_model.difference(data.columns))
    if missing_model:
        raise ValueError(f"full model dataset is missing columns: {missing_model}")
    required_sample = {"base_period", SPLIT_COLUMN, EXIT_DATE_COLUMN}
    missing_sample = sorted(required_sample.difference(index.columns))
    if missing_sample:
        raise ValueError(f"sample index is missing columns: {missing_sample}")
    if not index["base_period"].astype("string").eq("test").all():
        raise ValueError("locked-test keys must belong to the test base period")

    transformed, factor_features = _transform_daily_cross_sections(
        data,
        factor_columns,
        mad_width=mad_width,
        expected_cross_section_size=expected_cross_section_size,
        date_chunk_size=date_chunk_size,
    )
    factor_frame = pd.DataFrame(transformed, columns=factor_features, index=data.index)
    data = pd.concat([data.drop(columns=list(factor_columns)), factor_frame], axis=1)
    data["industry"] = _normalize_industry(data["industry"])
    data["industry_missing"] = data["industry"].eq(UNKNOWN_INDUSTRY).astype(np.float32)
    raw_cap = pd.to_numeric(data["market_cap"], errors="coerce").where(
        lambda values: values.gt(0)
    )
    log_cap = np.log1p(raw_cap)
    cap_median = log_cap.groupby(data["date"], sort=False).transform("median")
    data["market_cap_missing"] = log_cap.isna().astype(np.float32)
    data["log_market_cap"] = log_cap.fillna(cap_median).fillna(0.0).astype(np.float64)
    data = data.drop(columns="market_cap")

    official = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    test_dates = official[(official >= start) & (official <= end)]
    if not pd.DatetimeIndex(data["date"].unique()).equals(test_dates):
        raise ValueError("locked-test dates do not match the trading calendar")
    market_context, industry_context = _build_daily_context(
        market,
        official,
        windows=windows,
        cap_coverage_threshold=cap_coverage_threshold,
    )
    market_context = market_context.loc[market_context["date"].between(start, end)]
    industry_context = industry_context.loc[
        industry_context["date"].between(start, end)
    ]
    market_features = [column for column in market_context if column != "date"]
    industry_features = [
        column for column in industry_context if column not in {"date", "industry"}
    ]
    data = data.merge(market_context, on="date", how="left", validate="many_to_one")
    data = data.merge(
        industry_context, on=["date", "industry"], how="left", validate="many_to_one"
    )
    unknown = data["industry"].eq(UNKNOWN_INDUSTRY)
    for column in industry_features:
        data.loc[unknown, column] = 1.0 if column.endswith("__missing") else 0.0

    split = index[SPLIT_COLUMN].astype("string").str.strip().str.lower()
    exits = pd.to_datetime(index[EXIT_DATE_COLUMN], errors="coerce").dt.normalize()
    valid_label = split.eq("test").fillna(False) & exits.notna() & exits.le(end)
    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    data[TARGET_COLUMN] = target.where(valid_label)
    data[EXIT_DATE_COLUMN] = exits.where(valid_label)
    data[SPLIT_COLUMN] = "test"
    ranks = pd.Series(np.nan, index=data.index, dtype=np.float64)
    coverage: list[dict[str, Any]] = []
    for timestamp, positions in data.groupby("date", sort=True).groups.items():
        values = data.loc[positions, TARGET_COLUMN]
        finite_count = int(np.isfinite(values.to_numpy(dtype=np.float64)).sum())
        if finite_count >= 2:
            ranks.loc[positions] = percentile_rank_from_returns(values)
        coverage.append(
            {
                "date": pd.Timestamp(timestamp).strftime("%Y-%m-%d"),
                "total_rows": int(len(values)),
                "valid_targets": finite_count,
                "coverage": float(finite_count / len(values)),
                "tie_rows": int(
                    values.loc[np.isfinite(values.to_numpy(dtype=np.float64))]
                    .duplicated(keep=False)
                    .sum()
                ),
            }
        )
    data[RANK_TARGET_COLUMN] = ranks
    continuous = [
        *factor_features,
        "log_market_cap",
        "market_cap_missing",
        "industry_missing",
        *market_features,
        *industry_features,
    ]
    if expected_continuous_columns is not None and continuous != list(
        expected_continuous_columns
    ):
        raise ValueError("locked-test continuous feature order differs from development")
    forbidden = sorted(column for column in continuous if _is_forbidden_feature(column))
    if forbidden:
        raise ValueError(f"locked-test features contain forbidden columns: {forbidden}")
    values = data[continuous].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("locked-test continuous features must be finite")
    output_columns = [
        *KEY_COLUMNS,
        *continuous,
        "industry",
        TARGET_COLUMN,
        RANK_TARGET_COLUMN,
        SPLIT_COLUMN,
        EXIT_DATE_COLUMN,
    ]
    result = data[output_columns].sort_values(KEY_COLUMNS).reset_index(drop=True)
    _validate_daily_size(result, expected_cross_section_size, "locked test output")
    metadata = {
        "row_count": int(len(result)),
        "date_count": int(result["date"].nunique()),
        "date_min": result["date"].min().date().isoformat(),
        "date_max": result["date"].max().date().isoformat(),
        "labelled_date_count": int(
            result.groupby("date")[RANK_TARGET_COLUMN]
            .apply(lambda values: values.notna().any())
            .sum()
        ),
        "continuous_feature_columns": continuous,
        "industry_column": "industry",
        "feature_semantics": {
            "mad_width": float(mad_width),
            "windows": [int(value) for value in windows],
            "cap_coverage_threshold": float(cap_coverage_threshold),
        },
        "coverage_rows": coverage,
    }
    return result, metadata


def feature_engineering_code_sha256() -> str:
    """Fingerprint every function that defines the locked-test feature semantics."""
    functions = (
        _normalize_industry,
        _validate_keys,
        _validate_daily_size,
        _transform_one_date,
        _transform_daily_cross_sections,
        _compound_return,
        _fill_numeric_missing,
        _build_daily_context,
        build_locked_test_frame,
        percentile_rank_from_returns,
        _is_forbidden_feature,
    )
    contract = json.dumps(
        {
            "unknown_industry": UNKNOWN_INDUSTRY,
            "mad_normalization": MAD_NORMALIZATION,
            "limit_tolerance": LIMIT_TOLERANCE,
            "mad_width": LOCKED_MAD_WIDTH,
            "windows": LOCKED_WINDOWS,
            "cap_coverage_threshold": LOCKED_CAP_COVERAGE_THRESHOLD,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    payload = (contract + "\n\n" + "\n\n".join(
        inspect.getsource(function) for function in functions
    )).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _temporary_path(path: Path, suffix: str = ".tmp") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=suffix
    )
    os.close(descriptor)
    return Path(name)


def _write_locked_test_bundle_unlocked(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    dataset_path: Path,
    schema_path: Path,
    coverage_path: Path,
    *,
    source_hashes: dict[str, str] | None = None,
    development_contract_sha256: str | None = None,
    development_schema_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Atomically publish an immutable locked-test dataset bundle."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    outputs = [Path(dataset_path), Path(schema_path), Path(coverage_path)]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"locked-test dataset is immutable: {existing}")
    temporary = [
        _temporary_path(outputs[0], ".parquet"),
        _temporary_path(outputs[1]),
        _temporary_path(outputs[2]),
    ]
    published: list[Path] = []
    try:
        schema = {
            "schema_version": 1,
            "purpose": "locked_test_2024_2025",
            "uses_entry_tradeable": False,
            "key_columns": KEY_COLUMNS,
            "target_column": TARGET_COLUMN,
            "rank_target_column": RANK_TARGET_COLUMN,
            "split_column": SPLIT_COLUMN,
            "exit_date_column": EXIT_DATE_COLUMN,
            "rank_formula": "(average_rank(target_10d)-1)/(finite_count-1)",
            "rank_direction": "ascending_return_higher_rank",
            "feature_engineering_code_sha256": feature_engineering_code_sha256(),
            "feature_semantics": metadata.get(
                "feature_semantics",
                {
                    "mad_width": LOCKED_MAD_WIDTH,
                    "windows": list(LOCKED_WINDOWS),
                    "cap_coverage_threshold": LOCKED_CAP_COVERAGE_THRESHOLD,
                },
            ),
            "source_hashes": source_hashes or {},
            "development_contract_sha256": development_contract_sha256,
            "development_schema_contract_sha256": development_schema_contract_sha256,
            **{key: value for key, value in metadata.items() if key != "coverage_rows"},
        }
        table = pa.Table.from_pandas(frame, preserve_index=False)
        embedded = dict(table.schema.metadata or {})
        embedded[b"rank_model_locked_test_schema"] = json.dumps(
            schema, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        pq.write_table(
            table.replace_schema_metadata(embedded), temporary[0], compression="zstd"
        )
        schema["parquet_sha256"] = file_sha256(temporary[0])
        temporary[1].write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        coverage_rows = metadata.get("coverage_rows")
        if not isinstance(coverage_rows, list):
            raise ValueError("locked-test metadata is missing coverage rows")
        with temporary[2].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "date",
                    "total_rows",
                    "valid_targets",
                    "coverage",
                    "tie_rows",
                ],
            )
            writer.writeheader()
            writer.writerows(coverage_rows)
        for source, destination in zip(temporary, outputs):
            os.replace(source, destination)
            published.append(destination)
        return schema
    except BaseException:
        for path in reversed(published):
            if path.exists():
                path.unlink()
        raise
    finally:
        for path in temporary:
            if path.exists():
                path.unlink()


def write_locked_test_bundle(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    dataset_path: Path,
    schema_path: Path,
    coverage_path: Path,
    *,
    source_hashes: dict[str, str] | None = None,
    development_contract_sha256: str | None = None,
    development_schema_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Publish one dataset bundle while excluding concurrent writers."""
    destination = Path(dataset_path)
    lock = _acquire_process_lock(destination.parent / ".locked-test-dataset.lock")
    try:
        return _write_locked_test_bundle_unlocked(
            frame,
            metadata,
            destination,
            Path(schema_path),
            Path(coverage_path),
            source_hashes=source_hashes,
            development_contract_sha256=development_contract_sha256,
            development_schema_contract_sha256=development_schema_contract_sha256,
        )
    finally:
        _release_process_lock(lock)


def _read_csv_window(
    path: Path,
    columns: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype={"date": "string", "stock_code": "string"},
        chunksize=250_000,
        low_memory=False,
    ):
        dates = pd.to_datetime(chunk["date"], errors="raise").dt.normalize()
        selected = dates.between(start, end)
        if selected.any():
            kept = chunk.loc[selected].copy()
            kept["date"] = dates.loc[selected]
            frames.append(kept)
        if dates.min() > end:
            break
    if not frames:
        raise ValueError(f"{path} has no rows in the required window")
    return pd.concat(frames, ignore_index=True)


def prepare_locked_test_dataset(
    *,
    model_dataset_path: Path,
    model_schema_path: Path,
    sample_index_path: Path,
    split_summary_path: Path,
    market_panel_path: Path,
    trading_calendar_path: Path,
    development_schema_path: Path,
    output_dataset_path: Path,
    output_schema_path: Path,
    output_coverage_path: Path,
    test_start: pd.Timestamp = TEST_START,
    test_end: pd.Timestamp = TEST_END,
    mad_width: float = 3.0,
    expected_cross_section_size: int = 1000,
    date_chunk_size: int = 60,
    windows: Sequence[int] = (1, 5, 20),
    cap_coverage_threshold: float = 0.95,
) -> dict[str, Any]:
    """Build the sealed 2024-2025 rank input entirely from local artifacts."""
    source_paths = {
        "model_dataset": Path(model_dataset_path),
        "model_schema": Path(model_schema_path),
        "sample_index": Path(sample_index_path),
        "split_summary": Path(split_summary_path),
        "market_panel": Path(market_panel_path),
        "trading_calendar": Path(trading_calendar_path),
        "development_schema": Path(development_schema_path),
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing locked-test inputs: {missing}")
    model_schema = json.loads(
        source_paths["model_schema"].read_text(encoding="utf-8")
    )
    split_summary = json.loads(
        source_paths["split_summary"].read_text(encoding="utf-8")
    )
    development_schema = json.loads(
        source_paths["development_schema"].read_text(encoding="utf-8")
    )
    audited_source_hashes = validate_locked_test_source_hashes(
        {
            "model_dataset": source_paths["model_dataset"],
            "sample_index": source_paths["sample_index"],
            "market_panel": source_paths["market_panel"],
            "trading_calendar": source_paths["trading_calendar"],
        },
        development_schema,
    )
    model_hash = file_sha256(source_paths["model_dataset"])
    sample_hash = file_sha256(source_paths["sample_index"])
    if model_schema.get("parquet_sha256") != model_hash:
        raise ValueError("full model dataset hash does not match its schema")
    if split_summary.get("source_model_dataset_sha256") != model_hash:
        raise ValueError("sample index references another model dataset")
    if split_summary.get("sample_index_sha256") != sample_hash:
        raise ValueError("sample index hash does not match split summary")
    if split_summary.get("uses_entry_tradeable") is not False:
        raise ValueError("locked-test model evaluation must not use entry_tradeable")
    factor_columns = model_schema.get("feature_columns")
    expected_continuous = development_schema.get("continuous_feature_columns")
    if not isinstance(factor_columns, list) or not factor_columns:
        raise ValueError("model schema has no feature columns")
    if not isinstance(expected_continuous, list) or not expected_continuous:
        raise ValueError("development schema has no continuous feature contract")

    model_columns = [
        *KEY_COLUMNS,
        *factor_columns,
        TARGET_COLUMN,
        "industry",
        "market_cap",
    ]
    test_filter = [
        ("date", ">=", pd.Timestamp(test_start).to_pydatetime()),
        ("date", "<=", pd.Timestamp(test_end).to_pydatetime()),
    ]
    model = pd.read_parquet(
        source_paths["model_dataset"], columns=model_columns, filters=test_filter
    )
    sample = pd.read_parquet(
        source_paths["sample_index"],
        columns=[*KEY_COLUMNS, "base_period", SPLIT_COLUMN, EXIT_DATE_COLUMN],
        filters=test_filter,
    )
    calendar_frame = pd.read_csv(
        source_paths["trading_calendar"], usecols=["date"], dtype="string"
    )
    all_calendar = pd.DatetimeIndex(
        pd.to_datetime(calendar_frame["date"], errors="raise")
        .dt.normalize()
        .sort_values()
        .unique()
    )
    relevant_dates = all_calendar[all_calendar <= pd.Timestamp(test_end)]
    if len(relevant_dates) < max(windows) + 1:
        raise ValueError("trading calendar lacks context warm-up history")
    first_test_position = int(relevant_dates.searchsorted(pd.Timestamp(test_start)))
    warmup_position = max(0, first_test_position - max(windows))
    warmup_start = relevant_dates[warmup_position]
    context_calendar = relevant_dates[warmup_position:]
    market_columns = [
        "date",
        "stock_code",
        "in_universe",
        "post_close",
        "raw_close",
        "limit_up",
        "limit_down",
        "is_st",
        "is_suspended",
        "industry",
        "market_cap",
    ]
    market = _read_csv_window(
        source_paths["market_panel"], market_columns, warmup_start, pd.Timestamp(test_end)
    )
    frame, metadata = build_locked_test_frame(
        model,
        sample,
        market,
        context_calendar,
        factor_columns=factor_columns,
        expected_continuous_columns=expected_continuous,
        test_start=pd.Timestamp(test_start),
        test_end=pd.Timestamp(test_end),
        mad_width=mad_width,
        expected_cross_section_size=expected_cross_section_size,
        date_chunk_size=date_chunk_size,
        windows=windows,
        cap_coverage_threshold=cap_coverage_threshold,
    )
    source_hashes = {name: file_sha256(path) for name, path in source_paths.items()}
    if any(
        source_hashes[name] != digest
        for name, digest in audited_source_hashes.items()
    ):
        raise RuntimeError("locked-test source changed while the dataset was built")
    return write_locked_test_bundle(
        frame,
        metadata,
        output_dataset_path,
        output_schema_path,
        output_coverage_path,
        source_hashes=source_hashes,
        development_contract_sha256=source_hashes["development_schema"],
        development_schema_contract_sha256=rank_schema_contract_sha256(
            development_schema
        ),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def validate_locked_test_source_hashes(
    paths: dict[str, Path], development_schema: dict[str, Any]
) -> dict[str, str]:
    """Require locked-test sources to be the same files audited in development."""
    development_sources = development_schema.get("sources")
    if not isinstance(development_sources, dict):
        raise ValueError("development schema has no source hash contract")
    actual: dict[str, str] = {}
    for name in ("model_dataset", "sample_index", "market_panel", "trading_calendar"):
        path = Path(paths[name])
        digest = file_sha256(path)
        expected = development_sources.get(f"{name}_sha256")
        if digest != expected:
            raise ValueError(
                f"locked-test source {name} differs from the development source"
            )
        actual[name] = digest
    return actual


def _verify_artifact_hashes(directory: Path, manifest: dict[str, Any]) -> None:
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"final model manifest has no artifact hashes: {directory}")
    for name, expected in hashes.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("final model artifact hashes are invalid")
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(path)
        if file_sha256(path) != expected:
            raise ValueError(f"final model artifact hash mismatch: {name}")


def _load_test_dataset(
    dataset_path: Path,
    schema_path: Path,
    development_schema_path: Path,
    trading_calendar_path: Path,
    expected_cross_section_size: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    schema = _read_json_object(schema_path)
    development_schema = _read_json_object(development_schema_path)
    if schema.get("purpose") != "locked_test_2024_2025":
        raise ValueError("locked-test schema purpose is invalid")
    if schema.get("uses_entry_tradeable") is not False:
        raise ValueError("locked-test model evaluation must not use entry_tradeable")
    if file_sha256(dataset_path) != schema.get("parquet_sha256"):
        raise ValueError("locked-test dataset hash does not match its schema")
    if file_sha256(development_schema_path) != schema.get(
        "development_contract_sha256"
    ):
        raise ValueError("locked-test development contract hash mismatch")
    current_feature_hash = feature_engineering_code_sha256()
    recorded_feature_hash = schema.get("feature_engineering_code_sha256")
    if recorded_feature_hash is None:
        recorded_feature_hash = LEGACY_FEATURE_ENGINEERING_CONTRACTS.get(
            str(schema.get("parquet_sha256"))
        )
    if recorded_feature_hash != current_feature_hash:
        raise ValueError("locked-test feature engineering code contract mismatch")
    expected_semantics = {
        "mad_width": LOCKED_MAD_WIDTH,
        "windows": list(LOCKED_WINDOWS),
        "cap_coverage_threshold": LOCKED_CAP_COVERAGE_THRESHOLD,
    }
    recorded_semantics = schema.get("feature_semantics")
    if recorded_semantics is None and str(schema.get("parquet_sha256")) in (
        LEGACY_FEATURE_ENGINEERING_CONTRACTS
    ):
        recorded_semantics = expected_semantics
    if recorded_semantics != expected_semantics:
        raise ValueError("locked-test feature semantics differ from development")
    continuous = schema.get("continuous_feature_columns")
    development_continuous = development_schema.get("continuous_feature_columns")
    if not isinstance(continuous, list) or not continuous:
        raise ValueError("locked-test schema has no continuous feature contract")
    if not isinstance(development_continuous, list) or not development_continuous:
        raise ValueError("development schema has no continuous feature contract")
    if continuous != development_continuous:
        raise ValueError("locked-test feature contract differs from development")
    industry = schema.get("industry_column")
    development_industry = development_schema.get("industry_column")
    if not isinstance(industry, str) or not industry:
        raise ValueError("locked-test schema has no industry feature contract")
    if not isinstance(development_industry, str) or not development_industry:
        raise ValueError("development schema has no industry feature contract")
    if industry != development_industry:
        raise ValueError("locked-test industry contract differs from development")
    required = [
        *KEY_COLUMNS,
        *continuous,
        industry,
        TARGET_COLUMN,
        RANK_TARGET_COLUMN,
        SPLIT_COLUMN,
        EXIT_DATE_COLUMN,
    ]
    frame = pd.read_parquet(dataset_path, columns=required)
    frame = _validate_keys(frame, "locked-test dataset")
    _validate_daily_size(frame, expected_cross_section_size, "locked-test dataset")
    if not frame[SPLIT_COLUMN].astype("string").eq("test").all():
        raise ValueError("locked-test dataset contains a non-test split")
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if dates.min() < TEST_START or dates.max() > TEST_END:
        raise ValueError("locked-test dates are outside 2024-2025")
    source_hashes = schema.get("source_hashes")
    if not isinstance(source_hashes, dict) or file_sha256(
        Path(trading_calendar_path)
    ) != source_hashes.get("trading_calendar"):
        raise ValueError("locked-test trading calendar hash mismatch")
    calendar = pd.read_csv(trading_calendar_path, usecols=["date"], dtype="string")
    expected_dates = pd.DatetimeIndex(
        pd.to_datetime(calendar["date"], errors="raise")
        .dt.normalize()
        .loc[lambda values: values.between(TEST_START, TEST_END)]
        .sort_values()
        .unique()
    )
    actual_dates = pd.DatetimeIndex(dates.sort_values().unique())
    if not actual_dates.equals(expected_dates):
        raise ValueError("locked-test dates do not fully match the trading calendar")
    if len(frame) != schema.get("row_count") or dates.nunique() != schema.get(
        "date_count"
    ):
        raise ValueError("locked-test dimensions differ from its schema")
    values = frame[continuous].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("locked-test continuous features must be finite")
    return frame, schema, development_schema


def _prediction_frame(test: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != len(test) or not np.isfinite(values).all():
        raise ValueError("locked-test scores must be finite and cover every key")
    result = test[KEY_COLUMNS].copy()
    result["split"] = "test"
    result["horizon"] = 10
    result[TARGET_COLUMN] = pd.to_numeric(test[TARGET_COLUMN], errors="coerce")
    result[RANK_TARGET_COLUMN] = pd.to_numeric(
        test[RANK_TARGET_COLUMN], errors="coerce"
    )
    result["score_raw"] = values
    result["pred_rank_pct"] = predicted_percentiles(values, result["date"])
    result["pred_rank_position"] = result.groupby("date", sort=False)[
        "score_raw"
    ].rank(method="average", ascending=True)
    return result


def predict_locked_test_model(
    model_name: str,
    *,
    test_dataset_path: Path,
    test_schema_path: Path,
    development_schema_path: Path,
    trading_calendar_path: Path,
    frozen_spec_path: Path,
    final_runs_directory: Path,
    output_runs_directory: Path,
    expected_cross_section_size: int = 1000,
) -> Path:
    """Run one frozen model while excluding concurrent writers for that model."""
    output_root = Path(output_runs_directory)
    lock = _acquire_process_lock(output_root / f".{model_name}.inference.lock")
    try:
        return _predict_locked_test_model_unlocked(
            model_name,
            test_dataset_path=test_dataset_path,
            test_schema_path=test_schema_path,
            development_schema_path=development_schema_path,
            trading_calendar_path=trading_calendar_path,
            frozen_spec_path=frozen_spec_path,
            final_runs_directory=final_runs_directory,
            output_runs_directory=output_root,
            expected_cross_section_size=expected_cross_section_size,
        )
    finally:
        _release_process_lock(lock)


def _predict_locked_test_model_unlocked(
    model_name: str,
    *,
    test_dataset_path: Path,
    test_schema_path: Path,
    development_schema_path: Path,
    trading_calendar_path: Path,
    frozen_spec_path: Path,
    final_runs_directory: Path,
    output_runs_directory: Path,
    expected_cross_section_size: int = 1000,
) -> Path:
    """Load one frozen final model and publish immutable test predictions."""
    if model_name not in LOCKED_MODEL_NAMES:
        raise ValueError(f"model is not in the frozen locked-test set: {model_name}")
    output_directory = Path(output_runs_directory) / model_name
    if output_directory.exists():
        raise FileExistsError(
            f"locked-test prediction run is immutable: {output_directory}"
        )
    test, test_schema, development_schema = _load_test_dataset(
        Path(test_dataset_path),
        Path(test_schema_path),
        Path(development_schema_path),
        Path(trading_calendar_path),
        expected_cross_section_size,
    )
    frozen_spec = _read_json_object(frozen_spec_path)
    if frozen_spec.get("schema_version") != 1:
        raise ValueError("unsupported frozen model specification")
    candidates = [
        value
        for value in frozen_spec.get("candidates", [])
        if isinstance(value, dict) and value.get("model_name") == model_name
    ]
    if len(candidates) != 1:
        raise ValueError(f"frozen specification has no unique candidate: {model_name}")
    candidate = candidates[0]
    refit_contract = frozen_spec.get("refit_dataset", {})
    frozen_development_hash = refit_contract.get("schema_contract_sha256")
    current_development_hash = file_sha256(development_schema_path)
    current_development_contract_hash = rank_schema_contract_sha256(
        development_schema
    )
    if test_schema.get(
        "development_schema_contract_sha256"
    ) != current_development_contract_hash:
        raise ValueError("locked-test logical development contract hash mismatch")
    if frozen_development_hash != current_development_contract_hash:
        raise ValueError("frozen and locked-test development contracts differ")
    if file_sha256(frozen_spec_path) != _read_json_object(
        Path(final_runs_directory) / model_name / "manifest.json"
    ).get("frozen_spec_sha256"):
        raise ValueError("final model was trained from another frozen specification")

    final_directory = Path(final_runs_directory) / model_name
    manifest = _read_json_object(final_directory / "manifest.json")
    if (
        manifest.get("status") != "completed"
        or manifest.get("purpose") != "final_refit_2019_2023"
        or manifest.get("model_name") != model_name
    ):
        raise ValueError("final model manifest is not a completed 2019-2023 refit")
    _verify_artifact_hashes(final_directory, manifest)
    feature_schema = _read_json_object(final_directory / "feature_schema.json")
    if feature_schema.get("continuous_feature_columns") != test_schema.get(
        "continuous_feature_columns"
    ) or feature_schema.get("industry_column") != test_schema.get("industry_column"):
        raise ValueError("final model and locked-test feature contracts differ")

    preprocessor = joblib.load(final_directory / "preprocessor.joblib")
    if not isinstance(preprocessor, RankPreprocessor):
        raise ValueError("final model preprocessor has an unexpected type")
    if preprocessor.feature_names != candidate.get("transformed_feature_columns"):
        raise ValueError("final preprocessor feature order differs from frozen candidate")
    if preprocessor.feature_names != feature_schema.get("transformed_feature_columns"):
        raise ValueError("final preprocessor feature order differs from feature schema")
    model = _load_persisted_model(final_directory)
    started = perf_counter()
    features = preprocessor.transform(
        test,
        scale_continuous=not isinstance(model, (xgb.Booster, lgb.Booster)),
    )
    scores = _predict_model(model, features)
    inference_seconds = float(perf_counter() - started)
    predictions = _prediction_frame(test, scores)
    _validate_daily_size(
        predictions, expected_cross_section_size, "locked-test predictions"
    )

    Path(output_runs_directory).mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{model_name}.", dir=output_runs_directory)
    )
    try:
        prediction_path = temporary / "predictions_10d.parquet"
        predictions.to_parquet(prediction_path, index=False)
        output_manifest = {
            "status": "completed",
            "purpose": "locked_test_static_inference_2024_2025",
            "run_id": model_name,
            "model_name": model_name,
            "role": candidate.get("role"),
            "uses_training": False,
            "uses_refit": False,
            "uses_validation": False,
            "uses_early_stopping": False,
            "uses_entry_tradeable": False,
            "test_dataset_sha256": test_schema.get("parquet_sha256"),
            "test_schema_sha256": file_sha256(test_schema_path),
            "development_schema_sha256": current_development_hash,
            "frozen_spec_sha256": file_sha256(frozen_spec_path),
            "final_model_manifest_sha256": file_sha256(
                final_directory / "manifest.json"
            ),
            "final_model_artifact_sha256": manifest.get("artifact_sha256"),
            "transformed_feature_columns": preprocessor.feature_names,
            "prediction_rows": int(len(predictions)),
            "prediction_dates": int(predictions["date"].nunique()),
            "prediction_start": predictions["date"].min().date().isoformat(),
            "prediction_end": predictions["date"].max().date().isoformat(),
            "inference_seconds": inference_seconds,
            "predictions_10d_sha256": file_sha256(prediction_path),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_directory


def evaluate_locked_test_model(
    model_name: str,
    *,
    output_runs_directory: Path,
    hac_lag: int = 10,
    top_k: int = 100,
) -> Path:
    """Evaluate one immutable locked-test prediction artifact exactly once."""
    run_directory = Path(output_runs_directory) / model_name
    if not run_directory.is_dir():
        raise FileNotFoundError(run_directory)
    manifest = _read_json_object(run_directory / "manifest.json")
    if (
        manifest.get("status") != "completed"
        or manifest.get("purpose") != "locked_test_static_inference_2024_2025"
        or manifest.get("model_name") != model_name
        or manifest.get("uses_training") is not False
        or manifest.get("uses_refit") is not False
        or manifest.get("uses_validation") is not False
        or manifest.get("uses_early_stopping") is not False
        or manifest.get("uses_entry_tradeable") is not False
    ):
        raise ValueError("locked-test prediction manifest is invalid")
    prediction_path = run_directory / "predictions_10d.parquet"
    if file_sha256(prediction_path) != manifest.get("predictions_10d_sha256"):
        raise ValueError("locked-test prediction hash does not match its manifest")
    predictions = pd.read_parquet(prediction_path)
    if "split" not in predictions or not predictions["split"].eq("test").all():
        raise ValueError("locked-test predictions must contain only the test split")
    bundle = evaluate_predictions(predictions, hac_lag=hac_lag, top_k=top_k)
    write_evaluation(bundle, run_directory)
    return run_directory


def compare_locked_test_models(
    *,
    output_runs_directory: Path,
    output_path: Path,
    model_names: Sequence[str] = LOCKED_MODEL_NAMES,
) -> pd.DataFrame:
    """Compare every declared locked-test model without selecting a winner."""
    names = tuple(model_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("locked-test comparison model names must be unique")
    destination = Path(output_path)
    lock = _acquire_process_lock(
        Path(output_runs_directory) / ".locked-test-comparison.lock"
    )
    try:
        if destination.exists():
            raise FileExistsError(f"locked-test comparison is immutable: {destination}")
        frame = compare_runs(
            [Path(output_runs_directory) / name for name in names], destination
        )
        if frame["model_name"].tolist() != list(names):
            raise RuntimeError("locked-test comparison order differs from frozen order")
        return frame
    finally:
        _release_process_lock(lock)
