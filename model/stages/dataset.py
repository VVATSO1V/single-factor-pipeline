"""Build the model dataset and leakage-safe sample time index.

The factor-wide table defines the T-day sample keys and model features.
Targets are joined on the same keys. T-day industry and market cap are added
as metadata, while T+1 market fields are retained only for execution checks.

Sample membership is assigned separately for each horizon from its true exit
date on the official trading calendar.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["date", "stock_code"]
TARGET_COLUMNS = ["target_1d", "target_5d", "target_10d"]
WINDOWS = (1, 5, 10)
TARGET_BY_WINDOW = {window: f"target_{window}d" for window in WINDOWS}
T_METADATA_COLUMNS = ["industry", "market_cap"]
ENTRY_SOURCE_COLUMNS = [
    "is_st",
    "is_suspended",
    "listing_days",
    "raw_open",
    "limit_up",
    "limit_down",
]
ENTRY_AVAILABILITY_SOURCE_COLUMNS = [
    "has_price_record",
    "has_status_record",
]
ENTRY_COLUMNS = [f"entry_{column}" for column in ENTRY_SOURCE_COLUMNS]
RESERVED_COLUMNS = {
    *KEYS,
    *TARGET_COLUMNS,
    *T_METADATA_COLUMNS,
    "entry_date",
    "entry_data_available",
    *ENTRY_COLUMNS,
    "entry_open_limit_up",
    "entry_open_limit_down",
    "entry_tradeable",
}

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}
LIMIT_TOLERANCE = 1e-10


def require_columns(path: Path, required: set[str]) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            columns = next(csv.reader(handle))
        except StopIteration as error:
            raise ValueError(f"{path} is empty") from error
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{path} contains duplicate column names")
    return columns


def normalize_keys(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["stock_code"] = result["stock_code"].astype("string").str.strip()
    if result[KEYS].isna().any().any() or result["stock_code"].eq("").any():
        raise ValueError(f"{path} contains missing date or stock_code values")
    return result


def require_unique_keys(frame: pd.DataFrame, path: Path) -> None:
    duplicates = frame.duplicated(KEYS, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, KEYS].head(5).to_dict("records")
        raise ValueError(f"{path} contains duplicate keys: {sample}")


def coerce_numeric(
    frame: pd.DataFrame,
    columns: list[str],
    path: Path,
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        original = result[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any():
            sample = original.loc[invalid].astype(str).head(5).tolist()
            raise ValueError(f"{path} has non-numeric {column} values: {sample}")
        result[column] = numeric.replace([np.inf, -np.inf], np.nan)
    return result


def parse_boolean(series: pd.Series, column: str, path: Path) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    known = TRUE_VALUES | FALSE_VALUES
    invalid = text.notna() & ~text.isin(known)
    if invalid.any():
        sample = series.loc[invalid].astype(str).head(5).tolist()
        raise ValueError(f"{path} has invalid {column} values: {sample}")
    parsed = text.map(
        {
            **{value: True for value in TRUE_VALUES},
            **{value: False for value in FALSE_VALUES},
        }
    )
    return parsed.astype("boolean")


def load_factor_wide(path: Path) -> tuple[pd.DataFrame, list[str]]:
    columns = require_columns(path, set(KEYS))
    factor_columns = [column for column in columns if column not in KEYS]
    if not factor_columns:
        raise ValueError(f"{path} does not contain factor columns")

    conflicts = sorted(set(factor_columns).intersection(RESERVED_COLUMNS))
    if conflicts:
        raise ValueError(f"{path} uses reserved factor column names: {conflicts}")

    frame = pd.read_csv(
        path,
        dtype={"date": "string", "stock_code": "string"},
        low_memory=False,
    )
    frame = normalize_keys(frame, path)
    require_unique_keys(frame, path)
    frame = coerce_numeric(frame, factor_columns, path)
    return frame.sort_values(KEYS).reset_index(drop=True), factor_columns


def load_targets(path: Path) -> pd.DataFrame:
    require_columns(path, {*KEYS, *TARGET_COLUMNS})
    frame = pd.read_csv(
        path,
        usecols=[*KEYS, *TARGET_COLUMNS],
        dtype={"date": "string", "stock_code": "string"},
        low_memory=False,
    )
    frame = normalize_keys(frame, path)
    require_unique_keys(frame, path)
    frame = coerce_numeric(frame, TARGET_COLUMNS, path)
    return frame


def load_trading_calendar(path: Path) -> pd.DatetimeIndex:
    require_columns(path, {"date"})
    frame = pd.read_csv(path, usecols=["date"], dtype={"date": "string"})
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"{path} contains missing or duplicate trading dates")
    calendar = pd.DatetimeIndex(dates.sort_values().unique(), name="date")
    if calendar.empty:
        raise ValueError(f"{path} does not contain trading dates")
    return calendar


def load_market_panel(path: Path) -> pd.DataFrame:
    required = {
        *KEYS,
        *T_METADATA_COLUMNS,
        *ENTRY_SOURCE_COLUMNS,
        *ENTRY_AVAILABILITY_SOURCE_COLUMNS,
    }
    require_columns(path, required)
    frame = pd.read_csv(
        path,
        usecols=[
            *KEYS,
            *T_METADATA_COLUMNS,
            *ENTRY_SOURCE_COLUMNS,
            *ENTRY_AVAILABILITY_SOURCE_COLUMNS,
        ],
        dtype={
            "date": "string",
            "stock_code": "string",
            "industry": "string",
            "is_st": "string",
            "is_suspended": "string",
            "has_price_record": "string",
            "has_status_record": "string",
        },
        low_memory=False,
    )
    frame = normalize_keys(frame, path)
    require_unique_keys(frame, path)
    numeric_columns = [
        "market_cap",
        "listing_days",
        "raw_open",
        "limit_up",
        "limit_down",
    ]
    frame = coerce_numeric(frame, numeric_columns, path)
    frame["is_st"] = parse_boolean(frame["is_st"], "is_st", path)
    frame["is_suspended"] = parse_boolean(
        frame["is_suspended"],
        "is_suspended",
        path,
    )
    for column in ENTRY_AVAILABILITY_SOURCE_COLUMNS:
        frame[column] = parse_boolean(frame[column], column, path)
    return frame


def join_targets(
    factors: pd.DataFrame,
    targets: pd.DataFrame,
    target_path: Path,
) -> pd.DataFrame:
    result = factors.merge(
        targets,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator="_target_match",
    )
    missing = result["_target_match"].ne("both")
    if missing.any() or len(targets) != len(factors):
        sample = result.loc[missing, KEYS].head(5).to_dict("records")
        raise ValueError(
            f"{target_path} keys do not exactly match factor-wide keys; "
            f"factor_rows={len(factors)} target_rows={len(targets)} "
            f"missing_sample={sample}"
        )
    return result.drop(columns="_target_match")


def add_market_metadata(
    dataset: pd.DataFrame,
    market: pd.DataFrame,
    market_panel_path: Path,
    trading_calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    calendar = pd.DatetimeIndex(
        trading_calendar,
        name="date",
    )
    market_dates = pd.DatetimeIndex(market["date"].unique())
    missing_market_dates = calendar.difference(market_dates)
    extra_market_dates = market_dates.difference(calendar)
    if not missing_market_dates.empty or not extra_market_dates.empty:
        raise ValueError(
            f"{market_panel_path} does not exactly cover the trading calendar; "
            f"missing={missing_market_dates[:5].strftime('%Y-%m-%d').tolist()} "
            f"extra={extra_market_dates[:5].strftime('%Y-%m-%d').tolist()}"
        )
    unknown_dates = dataset.loc[~dataset["date"].isin(calendar), "date"].unique()
    if len(unknown_dates):
        sample = pd.to_datetime(unknown_dates[:5]).strftime("%Y-%m-%d").tolist()
        raise ValueError(
            f"factor-wide dates are missing from {market_panel_path}: {sample}"
        )

    next_date_map = pd.Series(
        calendar[1:].to_numpy(),
        index=calendar[:-1],
    )
    result = dataset.copy()
    result["entry_date"] = result["date"].map(next_date_map)

    t_metadata = market[[*KEYS, *T_METADATA_COLUMNS]]
    result = result.merge(
        t_metadata,
        on=KEYS,
        how="left",
        validate="many_to_one",
        indicator="_t_market_match",
    )
    missing_t = result["_t_market_match"].ne("both")
    if missing_t.any():
        sample = result.loc[missing_t, KEYS].head(5).to_dict("records")
        raise ValueError(
            f"{market_panel_path} has no T-day row for factor keys: {sample}"
        )
    result = result.drop(columns="_t_market_match")

    entry_state = market[
        [*KEYS, *ENTRY_SOURCE_COLUMNS, *ENTRY_AVAILABILITY_SOURCE_COLUMNS]
    ].rename(
        columns={
            "date": "entry_date",
            **{
                column: f"entry_{column}"
                for column in [
                    *ENTRY_SOURCE_COLUMNS,
                    *ENTRY_AVAILABILITY_SOURCE_COLUMNS,
                ]
            },
        }
    )
    result = result.merge(
        entry_state,
        on=["entry_date", "stock_code"],
        how="left",
        validate="many_to_one",
        indicator="_entry_match",
    )
    result["entry_data_available"] = (
        result["_entry_match"].eq("both")
        & result["entry_has_price_record"].astype("boolean").fillna(False)
        & result["entry_has_status_record"].astype("boolean").fillna(False)
    )
    return result.drop(
        columns=[
            "_entry_match",
            "entry_has_price_record",
            "entry_has_status_record",
        ]
    )


def add_tradeability(
    dataset: pd.DataFrame,
    min_listing_days: int,
) -> pd.DataFrame:
    if min_listing_days < 0:
        raise ValueError("min_listing_days cannot be negative")

    result = dataset.copy()
    result["entry_open_limit_up"] = (
        result["entry_raw_open"]
        >= result["entry_limit_up"] - LIMIT_TOLERANCE
    ).fillna(False)
    result["entry_open_limit_down"] = (
        result["entry_raw_open"]
        <= result["entry_limit_down"] + LIMIT_TOLERANCE
    ).fillna(False)

    is_st = result["entry_is_st"].astype("boolean").fillna(True)
    is_suspended = (
        result["entry_is_suspended"].astype("boolean").fillna(True)
    )
    valid_prices = (
        result[["entry_raw_open", "entry_limit_up", "entry_limit_down"]]
        .notna()
        .all(axis=1)
        & result[["entry_raw_open", "entry_limit_up", "entry_limit_down"]]
        .gt(0)
        .all(axis=1)
        & result["entry_limit_up"].gt(result["entry_limit_down"])
    )
    result["entry_tradeable"] = (
        result["entry_data_available"]
        & ~is_st
        & ~is_suspended
        & result["entry_listing_days"].ge(min_listing_days).fillna(False)
        & valid_prices
        & ~result["entry_open_limit_up"]
        & ~result["entry_open_limit_down"]
    ).astype(bool)
    return result


def build_schema(
    output_columns: list[str],
    factor_columns: list[str],
    min_listing_days: int,
) -> dict[str, object]:
    execution_columns = [
        "entry_date",
        "entry_data_available",
        *ENTRY_COLUMNS,
        "entry_open_limit_up",
        "entry_open_limit_down",
        "entry_tradeable",
    ]
    return {
        "schema_version": 1,
        "key_columns": KEYS,
        "feature_columns": factor_columns,
        "target_columns": TARGET_COLUMNS,
        "t_metadata_columns": T_METADATA_COLUMNS,
        "execution_columns": execution_columns,
        "output_columns": output_columns,
        "min_listing_days": min_listing_days,
        "target_type": "absolute_open_to_open_return",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(
    frame: pd.DataFrame,
    output_path: Path,
    schema: dict[str, object],
) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[b"model_dataset_schema"] = json.dumps(
            schema,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        table = table.replace_schema_metadata(metadata)
        pq.write_table(
            table,
            temporary_path,
            compression="zstd",
        )
        parquet_sha256 = file_sha256(temporary_path)
        temporary_path.replace(output_path)
        return parquet_sha256
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_schema(
    schema_path: Path,
    schema: dict[str, object],
    parquet_sha256: str,
) -> None:
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    published_schema = {**schema, "parquet_sha256": parquet_sha256}
    descriptor, temporary_name = tempfile.mkstemp(
        dir=schema_path.parent,
        prefix=f".{schema_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(published_schema, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(schema_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_model_dataset(
    factor_wide_path: Path,
    target_path: Path,
    market_panel_path: Path,
    trading_calendar_path: Path,
    output_path: Path,
    schema_path: Path,
    min_listing_days: int = 120,
) -> pd.DataFrame:
    factor_wide_path = Path(factor_wide_path)
    target_path = Path(target_path)
    market_panel_path = Path(market_panel_path)
    trading_calendar_path = Path(trading_calendar_path)
    output_path = Path(output_path)
    schema_path = Path(schema_path)

    factors, factor_columns = load_factor_wide(factor_wide_path)
    targets = load_targets(target_path)
    dataset = join_targets(factors, targets, target_path)
    del factors, targets
    gc.collect()

    trading_calendar = load_trading_calendar(trading_calendar_path)
    market = load_market_panel(market_panel_path)
    dataset = add_market_metadata(
        dataset,
        market,
        market_panel_path,
        trading_calendar,
    )
    del market
    gc.collect()

    dataset = add_tradeability(dataset, min_listing_days)
    output_columns = [
        *KEYS,
        *factor_columns,
        *TARGET_COLUMNS,
        *T_METADATA_COLUMNS,
        "entry_date",
        "entry_data_available",
        *ENTRY_COLUMNS,
        "entry_open_limit_up",
        "entry_open_limit_down",
        "entry_tradeable",
    ]
    dataset = dataset[output_columns].sort_values(KEYS).reset_index(drop=True)
    schema = build_schema(
        output_columns,
        factor_columns,
        min_listing_days,
    )
    parquet_sha256 = write_parquet(dataset, output_path, schema)
    write_schema(
        schema_path,
        schema,
        parquet_sha256,
    )

    print(
        "model dataset written: "
        f"{output_path.resolve()} shape={dataset.shape} "
        f"factors={len(factor_columns)}"
    )
    print(
        "entry coverage: "
        f"data_available={dataset['entry_data_available'].mean():.2%} "
        f"tradeable={dataset['entry_tradeable'].mean():.2%}"
    )
    for column in TARGET_COLUMNS:
        print(f"{column} coverage={dataset[column].notna().mean():.2%}")
    print(f"model schema written: {schema_path.resolve()}")
    return dataset


def load_source_schema(
    model_dataset_path: Path,
    model_schema_path: Path,
) -> dict[str, object]:
    """Validate that a model dataset still matches its published schema."""
    if not model_dataset_path.exists():
        raise FileNotFoundError(model_dataset_path)
    if not model_schema_path.exists():
        raise FileNotFoundError(model_schema_path)
    schema = json.loads(model_schema_path.read_text(encoding="utf-8"))
    expected_hash = schema.get("parquet_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{model_schema_path} has no valid parquet_sha256")
    actual_hash = file_sha256(model_dataset_path)
    if actual_hash != expected_hash:
        raise ValueError(
            "model dataset does not match its schema: "
            f"expected={expected_hash} actual={actual_hash}"
        )
    required_targets = set(TARGET_BY_WINDOW.values())
    schema_targets = set(schema.get("target_columns", []))
    if not required_targets.issubset(schema_targets):
        missing = sorted(required_targets.difference(schema_targets))
        raise ValueError(f"{model_schema_path} is missing targets: {missing}")
    return schema


def load_model_targets(path: Path) -> pd.DataFrame:
    """Read only keys and labels needed to construct sample membership."""
    import pyarrow.parquet as pq

    required = [*KEYS, *TARGET_BY_WINDOW.values()]
    available = pq.read_schema(path).names
    missing = sorted(set(required).difference(available))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = pd.read_parquet(path, columns=required)
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    ).dt.normalize()
    frame["stock_code"] = frame["stock_code"].astype("string").str.strip()
    if frame[KEYS].isna().any().any() or frame["stock_code"].eq("").any():
        raise ValueError(f"{path} contains missing keys")
    if frame.duplicated(KEYS).any():
        sample = frame.loc[frame.duplicated(KEYS, keep=False), KEYS]
        raise ValueError(
            f"{path} contains duplicate keys: "
            f"{sample.head(5).to_dict('records')}"
        )
    for column in TARGET_BY_WINDOW.values():
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & values.isna()
        if invalid.any():
            raise ValueError(f"{path} has non-numeric {column} values")
        frame[column] = values.replace([np.inf, -np.inf], np.nan)
    return frame.sort_values(KEYS).reset_index(drop=True)


def normalize_split_periods(
    split_periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Normalize and validate non-overlapping development periods."""
    expected_names = ("train", "validation", "test")
    if tuple(split_periods) != expected_names:
        raise ValueError(
            "split_periods must be ordered as train, validation, test"
        )
    normalized: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for name in expected_names:
        start, end = split_periods[name]
        start_timestamp = pd.Timestamp(start).normalize()
        end_timestamp = pd.Timestamp(end).normalize()
        if start_timestamp > end_timestamp:
            raise ValueError(f"{name} split start is after end")
        normalized[name] = (start_timestamp, end_timestamp)
    for previous, current in (("train", "validation"), ("validation", "test")):
        if normalized[previous][1] >= normalized[current][0]:
            raise ValueError(f"{previous} and {current} split periods overlap")
    return normalized


def normalize_final_train_period(
    final_train_period: tuple[pd.Timestamp, pd.Timestamp],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = final_train_period
    normalized = (pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    if normalized[0] > normalized[1]:
        raise ValueError("final_train period start is after end")
    return normalized


def assign_base_period(
    dates: pd.Series,
    split_periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.Series:
    result = pd.Series(pd.NA, index=dates.index, dtype="string")
    for split, (start, end) in split_periods.items():
        mask = dates.between(start, end)
        if result.loc[mask].notna().any():
            raise RuntimeError(f"split periods overlap at {split}")
        result.loc[mask] = split
    return result


def calculate_exit_dates(
    dates: pd.Series,
    calendar: pd.DatetimeIndex,
    window: int,
) -> pd.Series:
    positions = pd.Series(
        np.arange(len(calendar), dtype=np.int64),
        index=calendar,
    )
    signal_positions = dates.map(positions)
    if signal_positions.isna().any():
        sample = dates.loc[signal_positions.isna()].head(5)
        raise ValueError(
            "model dataset contains dates outside the trading calendar: "
            f"{sample.dt.strftime('%Y-%m-%d').tolist()}"
        )
    exit_positions = signal_positions.to_numpy(dtype=np.int64) + window + 1
    valid = exit_positions < len(calendar)
    values = np.full(
        len(dates),
        np.datetime64("NaT", "ns"),
        dtype="datetime64[ns]",
    )
    values[valid] = calendar.to_numpy()[exit_positions[valid]]
    return pd.Series(values, index=dates.index, dtype="datetime64[ns]")


def assign_horizon_split(
    base_period: pd.Series,
    target: pd.Series,
    exit_dates: pd.Series,
    split_periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.Series:
    period_end = base_period.map(
        {split: end for split, (_, end) in split_periods.items()}
    )
    eligible = (
        base_period.notna()
        & target.notna()
        & exit_dates.notna()
        & exit_dates.le(period_end)
    )
    result = pd.Series(pd.NA, index=base_period.index, dtype="string")
    result.loc[eligible] = base_period.loc[eligible]
    return result


def assign_final_train(
    dates: pd.Series,
    target: pd.Series,
    exit_dates: pd.Series,
    final_train_period: tuple[pd.Timestamp, pd.Timestamp],
) -> pd.Series:
    start, end = final_train_period
    return (
        dates.between(start, end)
        & target.notna()
        & exit_dates.notna()
        & exit_dates.le(end)
    ).astype(bool)


def iso_date(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def build_summary(
    index: pd.DataFrame,
    targets: pd.DataFrame,
    source_hash: str,
    split_periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    final_train_period: tuple[pd.Timestamp, pd.Timestamp],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "schema_version": 1,
        "source_model_dataset_sha256": source_hash,
        "uses_entry_tradeable": False,
        "split_periods": {
            split: {
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
            }
            for split, (start, end) in split_periods.items()
        },
        "final_train_period": {
            "start": final_train_period[0].strftime("%Y-%m-%d"),
            "end": final_train_period[1].strftime("%Y-%m-%d"),
        },
        "row_count": len(index),
        "horizons": {},
    }
    horizons: dict[str, object] = {}
    for window in WINDOWS:
        suffix = f"{window}d"
        split_column = f"split_{suffix}"
        exit_column = f"exit_date_{suffix}"
        target_column = TARGET_BY_WINDOW[window]
        split_counts = {
            split: int(index[split_column].eq(split).sum())
            for split in split_periods
        }
        base_scope = index["base_period"].notna()
        period_end = index["base_period"].map(
            {split: end for split, (_, end) in split_periods.items()}
        )
        target_valid = targets[target_column].notna()
        boundary_purged = (
            base_scope
            & target_valid
            & (
                index[exit_column].isna()
                | index[exit_column].gt(period_end)
            )
        )
        boundaries = {}
        for split in split_periods:
            mask = index[split_column].eq(split)
            boundaries[split] = {
                "first_signal_date": iso_date(index.loc[mask, "date"].min()),
                "last_signal_date": iso_date(index.loc[mask, "date"].max()),
                "last_exit_date": iso_date(index.loc[mask, exit_column].max()),
            }
        horizons[suffix] = {
            "target_column": target_column,
            "exit_offset": window + 1,
            "split_counts": split_counts,
            "final_train_count": int(index[f"final_train_{suffix}"].sum()),
            "target_missing_count": int((base_scope & ~target_valid).sum()),
            "boundary_purged_count": int(boundary_purged.sum()),
            "boundaries": boundaries,
        }
    summary["horizons"] = horizons
    return summary


def write_output_bundle(
    index: pd.DataFrame,
    output_path: Path,
    summary_path: Path,
    summary: dict[str, object],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_descriptor, parquet_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(parquet_descriptor)
    summary_descriptor, summary_name = tempfile.mkstemp(
        dir=summary_path.parent,
        prefix=f".{summary_path.name}.",
        suffix=".tmp",
    )
    os.close(summary_descriptor)
    temporary_parquet = Path(parquet_name)
    temporary_summary = Path(summary_name)
    try:
        table = pa.Table.from_pandas(index, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[b"model_split_config"] = json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, temporary_parquet, compression="zstd")
        published_summary = {
            **summary,
            "sample_index_sha256": file_sha256(temporary_parquet),
        }
        temporary_summary.write_text(
            json.dumps(published_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_parquet.replace(output_path)
        temporary_summary.replace(summary_path)
    finally:
        if temporary_parquet.exists():
            temporary_parquet.unlink()
        if temporary_summary.exists():
            temporary_summary.unlink()


def build_sample_index(
    model_dataset_path: Path,
    model_schema_path: Path,
    trading_calendar_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    split_periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    final_train_period: tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    model_dataset_path = Path(model_dataset_path)
    model_schema_path = Path(model_schema_path)
    trading_calendar_path = Path(trading_calendar_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    normalized_splits = normalize_split_periods(split_periods)
    normalized_final_train = normalize_final_train_period(final_train_period)

    source_schema = load_source_schema(
        model_dataset_path,
        model_schema_path,
    )
    source_hash = str(source_schema["parquet_sha256"])
    calendar = load_trading_calendar(trading_calendar_path)
    targets = load_model_targets(model_dataset_path)

    index = targets[KEYS].copy()
    index["base_period"] = assign_base_period(
        index["date"],
        normalized_splits,
    )
    for window in WINDOWS:
        suffix = f"{window}d"
        exit_column = f"exit_date_{suffix}"
        split_column = f"split_{suffix}"
        final_train_column = f"final_train_{suffix}"
        index[exit_column] = calculate_exit_dates(
            index["date"],
            calendar,
            window,
        )
        index[split_column] = assign_horizon_split(
            index["base_period"],
            targets[TARGET_BY_WINDOW[window]],
            index[exit_column],
            normalized_splits,
        )
        index[final_train_column] = assign_final_train(
            index["date"],
            targets[TARGET_BY_WINDOW[window]],
            index[exit_column],
            normalized_final_train,
        )

    summary = build_summary(
        index,
        targets,
        source_hash,
        normalized_splits,
        normalized_final_train,
    )
    write_output_bundle(index, output_path, summary_path, summary)
    print(
        f"model sample index written: {output_path.resolve()} "
        f"shape={index.shape}"
    )
    for window in WINDOWS:
        suffix = f"{window}d"
        counts = index[f"split_{suffix}"].value_counts()
        print(
            f"{suffix}: "
            + " ".join(
                f"{split}={int(counts.get(split, 0))}"
                for split in normalized_splits
            )
            + f" final_train={int(index[f'final_train_{suffix}'].sum())}"
        )
    print(f"split summary written: {summary_path.resolve()}")
    return index
