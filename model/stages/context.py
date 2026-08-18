"""Build leakage-safe market/industry context and 10-day label components."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from model.stages.preprocessing import transform_daily_cross_sections


KEY_COLUMNS = ["date", "stock_code"]
UNKNOWN_INDUSTRY = "UNKNOWN"
LIMIT_TOLERANCE = 1e-8


def _normalize_industry(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    missing = values.isna() | values.eq("") | values.str.lower().eq("unknown")
    return values.mask(missing, UNKNOWN_INDUSTRY).astype("string")


def _compound_return(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=window).apply(
        lambda item: float(np.prod(1.0 + item) - 1.0),
        raw=True,
    )


def _fill_numeric_missing(frame: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    result = frame.copy()
    numeric_columns = [
        column
        for column in result.columns
        if column not in keys and pd.api.types.is_numeric_dtype(result[column])
    ]
    for column in numeric_columns:
        missing_column = f"{column}__missing"
        result[missing_column] = result[column].isna().astype(np.float32)
        result[column] = result[column].fillna(0.0).astype(np.float64)
    return result


def _validate_context_input(
    market: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
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
    result = market.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["stock_code"] = result["stock_code"].astype("string").str.strip()
    if result[KEY_COLUMNS].isna().any().any() or result["stock_code"].eq("").any():
        raise ValueError("market context keys cannot be missing")
    if result.duplicated(KEY_COLUMNS).any():
        raise ValueError("market context keys must be unique")
    official = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    if official.empty or official.hasnans:
        raise ValueError("trading calendar must be non-empty and valid")
    unknown_dates = pd.DatetimeIndex(result["date"].unique()).difference(official)
    if not unknown_dates.empty:
        raise ValueError("market context contains dates outside the calendar")
    result["industry"] = _normalize_industry(result["industry"])
    for column in (
        "post_close",
        "raw_close",
        "limit_up",
        "limit_down",
        "market_cap",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    for column in ("in_universe", "is_st", "is_suspended"):
        result[column] = result[column].astype("boolean").fillna(False).astype(bool)
    return result.sort_values(["stock_code", "date"]).reset_index(drop=True)


def build_daily_context(
    market: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    windows: Sequence[int] = (1, 5, 20),
    cap_coverage_threshold: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return filled market-date and industry-date point-in-time features."""
    selected_windows = tuple(int(window) for window in windows)
    if (
        not selected_windows
        or len(selected_windows) != len(set(selected_windows))
        or any(window <= 0 for window in selected_windows)
    ):
        raise ValueError("windows must contain unique positive integers")
    if not 0 < cap_coverage_threshold <= 1:
        raise ValueError("cap_coverage_threshold must be in (0, 1]")

    panel = _validate_context_input(market, calendar)
    official = pd.DatetimeIndex(calendar).normalize().sort_values().unique()
    previous_close = panel.groupby("stock_code", sort=False)["post_close"].shift(1)
    valid_price = panel["post_close"].gt(0) & previous_close.gt(0)
    panel["stock_return_1d"] = np.where(
        valid_price,
        panel["post_close"] / previous_close - 1.0,
        np.nan,
    )
    panel["valid_market_cap"] = panel["market_cap"].where(panel["market_cap"].gt(0))
    panel["log_market_cap"] = np.log1p(panel["valid_market_cap"])
    universe = panel.loc[panel["in_universe"]].copy()
    if universe.empty:
        raise ValueError("market context has no in-universe rows")

    market_rows: list[dict[str, Any]] = []
    industry_rows: list[dict[str, Any]] = []
    for date, daily in universe.groupby("date", sort=True):
        returns = daily["stock_return_1d"]
        valid_returns = returns.notna()
        caps = daily["valid_market_cap"]
        valid_cap_return = valid_returns & caps.notna()
        count = len(daily)
        cap_coverage = float(valid_cap_return.sum() / count)
        equal_return = float(returns.mean()) if valid_returns.any() else np.nan
        if valid_cap_return.any():
            weights = caps.loc[valid_cap_return]
            cap_return = float(
                np.average(returns.loc[valid_cap_return], weights=weights)
            )
        else:
            cap_return = np.nan
        cap_fallback = cap_coverage < cap_coverage_threshold or not np.isfinite(
            cap_return
        )
        if cap_fallback:
            cap_return = equal_return

        valid_limit = (
            daily["raw_close"].gt(0)
            & daily["limit_up"].gt(0)
            & daily["limit_down"].gt(0)
            & daily["limit_up"].gt(daily["limit_down"])
        )
        if valid_limit.any():
            limit_up_ratio = float(
                daily.loc[valid_limit, "raw_close"]
                .ge(daily.loc[valid_limit, "limit_up"] - LIMIT_TOLERANCE)
                .mean()
            )
            limit_down_ratio = float(
                daily.loc[valid_limit, "raw_close"]
                .le(daily.loc[valid_limit, "limit_down"] + LIMIT_TOLERANCE)
                .mean()
            )
        else:
            limit_up_ratio = np.nan
            limit_down_ratio = np.nan
        valid_log_caps = daily["log_market_cap"].dropna()
        market_rows.append(
            {
                "date": date,
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
                "market_limit_up_ratio": limit_up_ratio,
                "market_limit_down_ratio": limit_down_ratio,
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

        total_valid_cap = float(caps.sum(min_count=1))
        known_industries = daily.loc[
            ~daily["industry"].eq(UNKNOWN_INDUSTRY)
        ]
        for industry, group in known_industries.groupby(
            "industry", sort=True, dropna=False
        ):
            group_returns = group["stock_return_1d"]
            valid_group_returns = group_returns.notna()
            group_caps = group["valid_market_cap"]
            valid_group_caps = group_caps.notna()
            valid_values = group_returns.loc[valid_group_returns]
            industry_rows.append(
                {
                    "date": date,
                    "industry": str(industry),
                    "industry_return_equal_daily": float(valid_values.mean())
                    if not valid_values.empty
                    else np.nan,
                    "industry_up_ratio": float(valid_values.gt(0).mean())
                    if not valid_values.empty
                    else np.nan,
                    "industry_return_std": float(valid_values.std(ddof=0))
                    if not valid_values.empty
                    else np.nan,
                    "industry_return_qspread": float(
                        valid_values.quantile(0.90) - valid_values.quantile(0.10)
                    )
                    if not valid_values.empty
                    else np.nan,
                    "industry_stock_count": float(len(group)),
                    "industry_log_cap_median": float(
                        group["log_market_cap"].median()
                    )
                    if valid_group_caps.any()
                    else np.nan,
                    "industry_cap_share": float(group_caps.sum() / total_valid_cap)
                    if np.isfinite(total_valid_cap) and total_valid_cap > 0
                    else np.nan,
                    "industry_suspended_ratio": float(group["is_suspended"].mean()),
                    "industry_return_coverage": float(
                        valid_group_returns.sum() / len(group)
                    ),
                    "industry_cap_coverage": float(valid_group_caps.sum() / len(group)),
                }
            )

    market_context = (
        pd.DataFrame(market_rows)
        .set_index("date")
        .reindex(official)
        .rename_axis("date")
    )
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

    industry_daily = pd.DataFrame(
        industry_rows,
        columns=[
            "date",
            "industry",
            "industry_return_equal_daily",
            "industry_up_ratio",
            "industry_return_std",
            "industry_return_qspread",
            "industry_stock_count",
            "industry_log_cap_median",
            "industry_cap_share",
            "industry_suspended_ratio",
            "industry_return_coverage",
            "industry_cap_coverage",
        ],
    )
    for column in industry_daily.columns.difference(["date", "industry"]):
        industry_daily[column] = pd.to_numeric(
            industry_daily[column], errors="coerce"
        ).astype(np.float64)
    industries = pd.Index(sorted(industry_daily["industry"].unique()), name="industry")
    industry_index = pd.MultiIndex.from_product(
        [official, industries], names=["date", "industry"]
    )
    industry_context = industry_daily.set_index(["date", "industry"]).reindex(
        industry_index
    )
    market_raw = market_context.set_index("date")
    for window in selected_windows:
        industry_context[f"industry_return_equal_{window}d"] = (
            industry_context.groupby(level="industry", sort=False)[
                "industry_return_equal_daily"
            ].transform(lambda values: _compound_return(values, window))
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

    market_context = _fill_numeric_missing(market_context, {"date"})
    industry_context = _fill_numeric_missing(
        industry_context, {"date", "industry"}
    )
    return market_context, industry_context


def decompose_target(
    frame: pd.DataFrame,
    *,
    shrink_k: float = 10.0,
) -> pd.DataFrame:
    """Decompose raw 10-day targets into market, industry, and stock alpha."""
    required = {"date", "stock_code", "industry", "target_10d"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"target decomposition is missing columns: {missing}")
    if not np.isfinite(shrink_k) or shrink_k <= 0:
        raise ValueError("shrink_k must be finite and positive")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["industry"] = _normalize_industry(result["industry"])
    result["target_10d"] = pd.to_numeric(
        result["target_10d"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    result["market_target_10d"] = result.groupby("date", sort=False)[
        "target_10d"
    ].transform("mean")
    grouped = result.groupby(["date", "industry"], sort=False)["target_10d"]
    industry_mean = grouped.transform("mean")
    industry_count = grouped.transform("count").astype(np.float64)
    shrinkage = industry_count / (industry_count + float(shrink_k))
    industry_effect = shrinkage * (industry_mean - result["market_target_10d"])
    unknown = result["industry"].eq(UNKNOWN_INDUSTRY)
    result["industry_target_count"] = industry_count
    result["industry_target_shrinkage"] = shrinkage.mask(unknown, 0.0)
    result["industry_target_10d"] = industry_effect.mask(unknown, 0.0)
    result["alpha_target_10d"] = (
        result["target_10d"]
        - result["market_target_10d"]
        - result["industry_target_10d"]
    )
    valid = result["target_10d"].notna()
    reconstructed = (
        result.loc[valid, "market_target_10d"]
        + result.loc[valid, "industry_target_10d"]
        + result.loc[valid, "alpha_target_10d"]
    )
    if not np.allclose(
        result.loc[valid, "target_10d"],
        reconstructed,
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("target decomposition identity failed")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_development_parquet(
    path: Path,
    columns: list[str],
    development_end: pd.Timestamp,
) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[("date", "<=", development_end.to_pydatetime())],
    )
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["stock_code"] = frame["stock_code"].astype("string").str.strip()
    if frame[KEY_COLUMNS].isna().any().any() or frame["stock_code"].eq("").any():
        raise ValueError(f"{path} has missing keys")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{path} has duplicate keys")
    if not frame.empty and frame["date"].max() > development_end:
        raise RuntimeError(f"{path} leaked beyond the development end")
    return frame.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _read_market_development(
    path: Path,
    development_end: pd.Timestamp,
) -> pd.DataFrame:
    columns = [
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
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype={"date": "string", "stock_code": "string"},
        chunksize=250_000,
        low_memory=False,
    ):
        dates = pd.to_datetime(chunk["date"], errors="raise").dt.normalize()
        selected = dates.le(development_end)
        if selected.any():
            kept = chunk.loc[selected].copy()
            kept["date"] = dates.loc[selected]
            frames.append(kept)
        if dates.min() > development_end:
            break
    if not frames:
        raise ValueError(f"{path} has no development rows")
    result = pd.concat(frames, ignore_index=True)
    if result["date"].max() > development_end:
        raise RuntimeError("market panel leaked beyond the development end")
    return result


def _atomic_write_context_bundle(
    frame: pd.DataFrame,
    output_path: Path,
    schema_path: Path,
    schema: dict[str, Any],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_handle, parquet_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    schema_handle, schema_name = tempfile.mkstemp(
        dir=schema_path.parent,
        prefix=f".{schema_path.name}.",
        suffix=".tmp",
    )
    os.close(parquet_handle)
    os.close(schema_handle)
    temporary_parquet = Path(parquet_name)
    temporary_schema = Path(schema_name)
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[b"context_dataset_schema"] = json.dumps(
            schema, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, temporary_parquet, compression="zstd")
        context_hash = _file_sha256(temporary_parquet)
        published = {**schema, "context_dataset_sha256": context_hash}
        temporary_schema.write_text(
            json.dumps(published, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_parquet.replace(output_path)
        temporary_schema.replace(schema_path)
    finally:
        if temporary_parquet.exists():
            temporary_parquet.unlink()
        if temporary_schema.exists():
            temporary_schema.unlink()


def build_context_dataset(
    model_dataset_path: Path,
    model_schema_path: Path,
    sample_index_path: Path,
    split_summary_path: Path,
    market_panel_path: Path,
    trading_calendar_path: Path,
    output_path: Path,
    schema_path: Path,
    *,
    development_end: str,
    mad_width: float,
    expected_cross_section_size: int,
    date_chunk_size: int,
    windows: Sequence[int] = (1, 5, 20),
    shrink_k: float = 10.0,
    cap_coverage_threshold: float = 0.95,
) -> pd.DataFrame:
    """Build and publish the local 10-day development context dataset."""
    paths = [
        Path(model_dataset_path),
        Path(model_schema_path),
        Path(sample_index_path),
        Path(split_summary_path),
        Path(market_panel_path),
        Path(trading_calendar_path),
    ]
    missing_paths = [str(path) for path in paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"missing context inputs: {missing_paths}")
    output_path = Path(output_path)
    schema_path = Path(schema_path)
    end = pd.Timestamp(development_end).normalize()
    if end > pd.Timestamp("2023-12-31"):
        raise ValueError("context development_end cannot exceed 2023-12-31")

    model_schema = _read_json(Path(model_schema_path))
    split_summary = _read_json(Path(split_summary_path))
    model_hash = _file_sha256(Path(model_dataset_path))
    sample_hash = _file_sha256(Path(sample_index_path))
    if model_schema.get("parquet_sha256") != model_hash:
        raise ValueError("model dataset hash does not match its schema")
    if split_summary.get("source_model_dataset_sha256") != model_hash:
        raise ValueError("sample index references another model dataset")
    if split_summary.get("sample_index_sha256") != sample_hash:
        raise ValueError("sample index hash does not match split summary")
    if split_summary.get("uses_entry_tradeable") is not False:
        raise ValueError("context development must not use entry_tradeable")
    factor_columns = model_schema.get("feature_columns")
    if (
        not isinstance(factor_columns, list)
        or not factor_columns
        or not all(isinstance(column, str) and column for column in factor_columns)
    ):
        raise ValueError("model schema feature_columns are invalid")

    model_columns = [
        *KEY_COLUMNS,
        *factor_columns,
        "target_10d",
        "industry",
        "market_cap",
    ]
    data = _read_development_parquet(
        Path(model_dataset_path), model_columns, end
    )
    sample = _read_development_parquet(
        Path(sample_index_path),
        [*KEY_COLUMNS, "split_10d", "exit_date_10d"],
        end,
    )
    if len(data) != len(sample) or not data[KEY_COLUMNS].equals(sample[KEY_COLUMNS]):
        raise ValueError("context dataset and sample-index keys differ")
    data["split_10d"] = sample["split_10d"].astype("string")
    data["exit_date_10d"] = pd.to_datetime(
        sample["exit_date_10d"], errors="coerce"
    ).dt.normalize()

    feature_matrix, factor_feature_columns = transform_daily_cross_sections(
        data,
        factor_columns,
        mad_width=float(mad_width),
        expected_cross_section_size=int(expected_cross_section_size),
        date_chunk_size=int(date_chunk_size),
    )
    factor_frame = pd.DataFrame(
        feature_matrix,
        columns=factor_feature_columns,
        index=data.index,
    )
    data = pd.concat(
        [
            data.drop(columns=factor_columns),
            factor_frame,
        ],
        axis=1,
    )
    data["industry"] = _normalize_industry(data["industry"])
    data["industry_missing"] = data["industry"].eq(UNKNOWN_INDUSTRY).astype(
        np.float32
    )
    raw_cap = pd.to_numeric(data["market_cap"], errors="coerce").where(
        lambda values: values.gt(0)
    )
    log_cap = np.log1p(raw_cap)
    cap_median = log_cap.groupby(data["date"], sort=False).transform("median")
    data["market_cap_missing"] = log_cap.isna().astype(np.float32)
    data["log_market_cap"] = log_cap.fillna(cap_median).fillna(0.0).astype(
        np.float64
    )
    data = data.drop(columns="market_cap")

    calendar_frame = pd.read_csv(
        trading_calendar_path, usecols=["date"], dtype={"date": "string"}
    )
    calendar_dates = pd.to_datetime(
        calendar_frame["date"], errors="raise"
    ).dt.normalize()
    calendar = pd.DatetimeIndex(
        calendar_dates.loc[calendar_dates.le(end)].sort_values().unique(),
        name="date",
    )
    market = _read_market_development(Path(market_panel_path), end)
    market_context, industry_context = build_daily_context(
        market,
        calendar,
        windows=windows,
        cap_coverage_threshold=float(cap_coverage_threshold),
    )
    industry_feature_columns = [
        column
        for column in industry_context.columns
        if column not in {"date", "industry"}
    ]
    data = data.merge(
        market_context,
        on="date",
        how="left",
        validate="many_to_one",
    )
    data = data.merge(
        industry_context,
        on=["date", "industry"],
        how="left",
        validate="many_to_one",
    )
    unknown_industry = data["industry"].eq(UNKNOWN_INDUSTRY)
    for column in industry_feature_columns:
        data.loc[unknown_industry, column] = (
            1.0 if column.endswith("__missing") else 0.0
        )
    if data.duplicated(KEY_COLUMNS).any():
        raise RuntimeError("context joins created duplicate keys")
    data = decompose_target(data, shrink_k=float(shrink_k))

    market_feature_columns = [
        column for column in market_context.columns if column != "date"
    ]
    stock_feature_columns = [
        *factor_feature_columns,
        "log_market_cap",
        "market_cap_missing",
        "industry_missing",
    ]
    continuous_feature_columns = [
        *stock_feature_columns,
        *market_feature_columns,
        *industry_feature_columns,
    ]
    output_columns = [
        *KEY_COLUMNS,
        "industry",
        *continuous_feature_columns,
        "target_10d",
        "market_target_10d",
        "industry_target_10d",
        "alpha_target_10d",
        "industry_target_count",
        "industry_target_shrinkage",
        "split_10d",
        "exit_date_10d",
    ]
    if len(output_columns) != len(set(output_columns)):
        raise RuntimeError("context output columns are not unique")
    data = data[output_columns].sort_values(KEY_COLUMNS).reset_index(drop=True)
    if len(data) == 0 or data["date"].max() > end:
        raise RuntimeError("context output development boundary failed")
    daily_counts = data.groupby("date", sort=False).size()
    if not daily_counts.eq(expected_cross_section_size).all():
        raise RuntimeError("context output cross-section size changed")
    selected = data["split_10d"].isin(["train", "validation"])
    if data.loc[selected, "exit_date_10d"].isna().any():
        raise ValueError("selected context rows have missing exit dates")

    schema: dict[str, Any] = {
        "schema_version": 1,
        "development_end": end.strftime("%Y-%m-%d"),
        "row_count": int(len(data)),
        "date_count": int(data["date"].nunique()),
        "key_columns": KEY_COLUMNS,
        "industry_column": "industry",
        "factor_feature_columns": factor_feature_columns,
        "stock_feature_columns": stock_feature_columns,
        "market_feature_columns": market_feature_columns,
        "industry_feature_columns": industry_feature_columns,
        "continuous_feature_columns": continuous_feature_columns,
        "target_columns": [
            "target_10d",
            "market_target_10d",
            "industry_target_10d",
            "alpha_target_10d",
        ],
        "sample_columns": ["split_10d", "exit_date_10d"],
        "windows": [int(window) for window in windows],
        "industry_shrink_k": float(shrink_k),
        "cap_coverage_threshold": float(cap_coverage_threshold),
        "uses_entry_tradeable": False,
        "sources": {
            "model_dataset_sha256": model_hash,
            "sample_index_sha256": sample_hash,
            "market_panel_sha256": _file_sha256(Path(market_panel_path)),
            "trading_calendar_sha256": _file_sha256(Path(trading_calendar_path)),
        },
    }
    _atomic_write_context_bundle(data, output_path, schema_path, schema)
    print(
        "context dataset written: "
        f"{output_path.resolve()} shape={data.shape} max_date={data['date'].max():%Y-%m-%d}",
        flush=True,
    )
    return data
