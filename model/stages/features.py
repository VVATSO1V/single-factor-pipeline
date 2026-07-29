"""Build the historical CSI1000 factor-wide table.

The market panel defines the point-in-time universe. Each factor CSV is
aligned to the resulting unique (date, stock_code) index, and missing factor
values are preserved for later preprocessing.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEYS = ["date", "stock_code"]

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}


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


def load_universe_keys(market_panel_path: Path) -> pd.DataFrame:
    if not market_panel_path.exists():
        raise FileNotFoundError(f"market panel not found: {market_panel_path}")
    require_columns(market_panel_path, {*KEYS, "in_universe"})
    panel = pd.read_csv(
        market_panel_path,
        usecols=[*KEYS, "in_universe"],
        dtype={"date": "string", "stock_code": "string", "in_universe": "string"},
    )
    panel = normalize_keys(panel, market_panel_path)
    if panel.duplicated(KEYS).any():
        raise ValueError(
            f"{market_panel_path} has duplicate date/stock_code rows: "
            f"{duplicate_key_sample(panel)}"
        )

    universe_text = panel["in_universe"].str.strip().str.lower()
    known_values = TRUE_VALUES | FALSE_VALUES
    invalid = universe_text.notna() & ~universe_text.isin(known_values)
    if invalid.any():
        sample = sorted(universe_text.loc[invalid].dropna().unique())[:5]
        raise ValueError(
            f"{market_panel_path} has invalid in_universe values: {sample}"
        )

    master = (
        panel.loc[universe_text.isin(TRUE_VALUES), KEYS]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    if master.empty:
        raise ValueError(f"{market_panel_path} has no in-universe rows")
    if master.duplicated(KEYS).any():
        raise ValueError("filtered historical universe has duplicate keys")
    return master


def infer_factor_name(factor_path: Path) -> str:
    if factor_path.name.lower() != "factor.csv" or factor_path.parent.name != "data":
        raise ValueError(
            f"factor path must follow <factor_name>/data/factor.csv: {factor_path}"
        )
    factor_name = factor_path.parent.parent.name.strip()
    if not factor_name:
        raise ValueError(f"cannot infer factor name from path: {factor_path}")
    return factor_name


def load_factor_series(factor_path: Path, factor_name: str) -> pd.Series:
    if not factor_path.exists():
        raise FileNotFoundError(f"factor file not found: {factor_path}")
    require_columns(factor_path, {*KEYS, "factor_value"})
    factor = pd.read_csv(
        factor_path,
        usecols=[*KEYS, "factor_value"],
        dtype={"date": "string", "stock_code": "string"},
    )
    factor = normalize_keys(factor, factor_path)
    if factor.duplicated(KEYS).any():
        raise ValueError(
            f"{factor_path} has duplicate date/stock_code rows: "
            f"{duplicate_key_sample(factor)}"
        )

    original = factor["factor_value"]
    numeric = pd.to_numeric(original, errors="coerce")
    invalid = original.notna() & numeric.isna()
    if invalid.any():
        sample = original.loc[invalid].astype(str).head(5).tolist()
        raise ValueError(f"{factor_path} has non-numeric factor values: {sample}")
    factor["factor_value"] = numeric.replace([np.inf, -np.inf], np.nan)

    series = factor.set_index(KEYS)["factor_value"]
    series.name = factor_name
    return series


def normalize_factor_paths(factor_paths: Iterable[Path]) -> list[tuple[str, Path]]:
    named_paths = []
    seen_names = set()
    for raw_path in factor_paths:
        path = Path(raw_path).resolve()
        factor_name = infer_factor_name(path)
        if factor_name in seen_names:
            raise ValueError(f"duplicate inferred factor name: {factor_name}")
        seen_names.add(factor_name)
        named_paths.append((factor_name, path))
    if not named_paths:
        raise ValueError("factor_paths must contain at least one factor")
    return named_paths


def build_factor_table(
    market_panel_path: Path,
    factor_paths: Iterable[Path],
    output_path: Path,
) -> pd.DataFrame:
    market_panel_path = Path(market_panel_path).resolve()
    output_path = Path(output_path).resolve()
    named_paths = normalize_factor_paths(factor_paths)

    wide = load_universe_keys(market_panel_path)
    expected_rows = len(wide)
    master_index = pd.MultiIndex.from_frame(wide[KEYS])
    print(
        "historical universe loaded: "
        f"rows={expected_rows} dates={wide['date'].nunique()} "
        f"stocks={wide['stock_code'].nunique()}",
        flush=True,
    )

    for position, (factor_name, factor_path) in enumerate(named_paths, start=1):
        factor = load_factor_series(factor_path, factor_name)
        aligned = factor.reindex(master_index)
        wide[factor_name] = aligned.to_numpy()
        if len(wide) != expected_rows:
            raise RuntimeError(
                f"row count changed while adding {factor_name}: "
                f"{len(wide)} != {expected_rows}"
            )
        valid = int(wide[factor_name].notna().sum())
        print(
            f"factor progress: {position}/{len(named_paths)} "
            f"{factor_name} valid={valid} coverage={valid / expected_rows:.2%}",
            flush=True,
        )
        del factor, aligned
        gc.collect()

    if wide.duplicated(KEYS).any():
        raise RuntimeError("factor-wide table has duplicate keys")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    print(
        f"factor-wide table written: {output_path} shape={wide.shape}",
        flush=True,
    )
    return wide
