"""Build absolute open-to-open return targets for the model dataset.

For a signal observed after the close on T:

    target_1d(T)  = post_open(T+2)  / post_open(T+1) - 1
    target_5d(T)  = post_open(T+6)  / post_open(T+1) - 1
    target_10d(T) = post_open(T+11) / post_open(T+1) - 1

Targets are calculated on the complete trading calendar before they are
aligned to the historical CSI1000 keys in factor_wide.csv. Missing prices are
preserved as missing targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["date", "stock_code"]
TARGET_WINDOWS = (1, 5, 10)


def require_columns(path: Path, required: set[str]) -> None:
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def normalize_keys(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["stock_code"] = result["stock_code"].astype("string").str.strip()
    if result[KEYS].isna().any().any() or result["stock_code"].eq("").any():
        raise ValueError(f"{path} contains missing date or stock_code values")
    return result


def duplicate_key_sample(frame: pd.DataFrame) -> list[dict[str, object]]:
    duplicates = frame.duplicated(KEYS, keep=False)
    return frame.loc[duplicates, KEYS].head(5).to_dict("records")


def load_market_prices(market_panel_path: Path) -> pd.DataFrame:
    if not market_panel_path.exists():
        raise FileNotFoundError(f"market panel not found: {market_panel_path}")
    require_columns(market_panel_path, {*KEYS, "post_open"})
    prices = pd.read_csv(
        market_panel_path,
        usecols=[*KEYS, "post_open"],
        dtype={"date": "string", "stock_code": "string"},
    )
    prices = normalize_keys(prices, market_panel_path)
    if prices.duplicated(KEYS).any():
        raise ValueError(
            f"{market_panel_path} has duplicate date/stock_code rows: "
            f"{duplicate_key_sample(prices)}"
        )

    original = prices["post_open"]
    numeric = pd.to_numeric(original, errors="coerce")
    invalid = original.notna() & numeric.isna()
    if invalid.any():
        sample = original.loc[invalid].astype(str).head(5).tolist()
        raise ValueError(f"{market_panel_path} has non-numeric post_open: {sample}")
    prices["post_open"] = numeric.replace([np.inf, -np.inf], np.nan)
    return prices


def load_factor_keys(factor_wide_path: Path) -> pd.DataFrame:
    if not factor_wide_path.exists():
        raise FileNotFoundError(f"factor-wide table not found: {factor_wide_path}")
    require_columns(factor_wide_path, set(KEYS))
    keys = pd.read_csv(
        factor_wide_path,
        usecols=KEYS,
        dtype={"date": "string", "stock_code": "string"},
    )
    keys = normalize_keys(keys, factor_wide_path)
    if keys.duplicated(KEYS).any():
        raise ValueError(
            f"{factor_wide_path} has duplicate date/stock_code rows: "
            f"{duplicate_key_sample(keys)}"
        )
    return keys


def load_trading_calendar(path: Path) -> pd.DatetimeIndex:
    if not path.exists():
        raise FileNotFoundError(f"trading calendar not found: {path}")
    require_columns(path, {"date"})
    frame = pd.read_csv(path, usecols=["date"], dtype={"date": "string"})
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"{path} contains missing or duplicate trading dates")
    calendar = pd.DatetimeIndex(dates.sort_values().unique(), name="date")
    if calendar.empty:
        raise ValueError(f"{path} does not contain trading dates")
    return calendar


def calculate_targets(
    prices: pd.DataFrame,
    factor_keys: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    windows: tuple[int, ...] = TARGET_WINDOWS,
) -> pd.DataFrame:
    if not windows or any(window <= 0 for window in windows):
        raise ValueError("windows must contain positive integers")

    calendar = pd.DatetimeIndex(trading_calendar).sort_values().unique()
    calendar = pd.DatetimeIndex(calendar, name="date")
    price_dates = pd.DatetimeIndex(prices["date"].unique())
    unknown_price_dates = price_dates.difference(calendar)
    if not unknown_price_dates.empty:
        raise ValueError(
            "market panel contains dates outside the trading calendar: "
            f"{unknown_price_dates[:5].strftime('%Y-%m-%d').tolist()}"
        )
    unknown_factor_dates = pd.DatetimeIndex(
        factor_keys["date"].unique()
    ).difference(calendar)
    if not unknown_factor_dates.empty:
        raise ValueError(
            "factor-wide contains dates outside the trading calendar: "
            f"{unknown_factor_dates[:5].strftime('%Y-%m-%d').tolist()}"
        )

    stocks = pd.Index(sorted(prices["stock_code"].unique()), name="stock_code")
    full_index = pd.MultiIndex.from_product(
        [stocks, calendar],
        names=["stock_code", "date"],
    )
    panel = (
        prices.set_index(["stock_code", "date"])[["post_open"]]
        .reindex(full_index)
        .sort_index()
    )
    grouped = panel.groupby(level="stock_code", group_keys=False)
    entry_open = grouped["post_open"].shift(-1)

    targets = pd.DataFrame(index=full_index)
    for window in windows:
        exit_open = grouped["post_open"].shift(-(window + 1))
        targets[f"target_{window}d"] = (
            exit_open / entry_open - 1.0
        ).replace([np.inf, -np.inf], np.nan)

    factor_index = pd.MultiIndex.from_frame(
        factor_keys[["stock_code", "date"]],
        names=["stock_code", "date"],
    )
    aligned = targets.reindex(factor_index).reset_index()
    result = aligned[["date", "stock_code", *targets.columns]].copy()
    if len(result) != len(factor_keys):
        raise RuntimeError(
            f"target row count changed during alignment: "
            f"{len(result)} != {len(factor_keys)}"
        )
    if result.duplicated(KEYS).any():
        raise RuntimeError("target table has duplicate date/stock_code rows")
    return result


def build_target_table(
    market_panel_path: Path,
    trading_calendar_path: Path,
    factor_wide_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    market_panel_path = Path(market_panel_path).resolve()
    trading_calendar_path = Path(trading_calendar_path).resolve()
    factor_wide_path = Path(factor_wide_path).resolve()
    output_path = Path(output_path).resolve()

    prices = load_market_prices(market_panel_path)
    trading_calendar = load_trading_calendar(trading_calendar_path)
    factor_keys = load_factor_keys(factor_wide_path)
    print(
        "target inputs loaded: "
        f"price_rows={len(prices)} factor_rows={len(factor_keys)} "
        f"dates={factor_keys['date'].nunique()}",
        flush=True,
    )
    targets = calculate_targets(prices, factor_keys, trading_calendar)
    for window in TARGET_WINDOWS:
        column = f"target_{window}d"
        valid = int(targets[column].notna().sum())
        print(
            f"{column}: valid={valid} coverage={valid / len(targets):.2%}",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    print(f"target table written: {output_path} shape={targets.shape}", flush=True)
    return targets
