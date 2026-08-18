"""Build the point-in-time ROE quality factor.

    roe = parent_net_profit_ttm / average_parent_equity
    accruals = net_profit_ttm - operating_cash_flow_ttm
    factor(T) = roe - accruals / total_assets

Average equity uses the latest visible values for the current quarter and
the same quarter one year earlier.
Statements become visible on the trading date after their announcement.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
CSI1000 = "000852.XSHG"
KEYS = ["date", "stock_code"]
PIT_FIELDS = [
    "np_parent_company_ownersTTM",
    "net_profitTTM",
    "net_operate_cashflowTTM",
    "total_assets",
    "equity_parent_company",
]
HISTORY_QUARTERS = 8


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def init_rqdatac(env_path: Path) -> Any:
    load_env_file(env_path)
    username = os.getenv("RQDATA_USER")
    password = os.getenv("RQDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            f"Missing rqdatac credentials. Fill RQDATA_USER and RQDATA_PASSWORD in {env_path}."
        )
    import rqdatac as rq

    rq.init(username, password)
    return rq


def normalize_rq_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if isinstance(result.index, pd.MultiIndex) or result.index.name is not None:
        result = result.reset_index()
    result = result.rename(
        columns={
            "order_book_id": "stock_code",
            "order_book_ids": "stock_code",
            "instrument": "stock_code",
        }
    )
    return result


def build_universe(
    rq: Any, start_date: str, end_date: str, index_code: str
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    trading_dates = pd.DatetimeIndex(
        [pd.Timestamp(date).normalize() for date in rq.get_trading_dates(start_date, end_date)]
    )
    raw_components = rq.index_components(
        index_code, start_date=start_date, end_date=end_date
    )
    snapshot_map: dict[pd.Timestamp, list[str]] = {}
    if isinstance(raw_components, dict):
        snapshot_map = {
            pd.Timestamp(date).normalize(): list(stocks)
            for date, stocks in raw_components.items()
        }
    elif isinstance(raw_components, pd.DataFrame):
        components = normalize_rq_frame(raw_components)
        if "date" not in components.columns:
            for column in components.columns:
                if pd.api.types.is_datetime64_any_dtype(components[column]):
                    components = components.rename(columns={column: "date"})
                    break
        components["date"] = pd.to_datetime(components["date"]).dt.normalize()
        for date, group in components.groupby("date"):
            snapshot_map[date] = group["stock_code"].tolist()
    if not snapshot_map:
        raise RuntimeError("index_components did not return historical snapshots")

    rows = []
    known_dates = sorted(snapshot_map)
    pointer = 0
    current_stocks: list[str] | None = None
    for date in trading_dates:
        while pointer < len(known_dates) and known_dates[pointer] <= date:
            current_stocks = snapshot_map[known_dates[pointer]]
            pointer += 1
        if current_stocks is None:
            raise RuntimeError(f"No index component snapshot is available before {date}")
        rows.extend({"date": date, "stock_code": stock} for stock in current_stocks)
    universe = pd.DataFrame(rows).drop_duplicates(KEYS).sort_values(KEYS)
    return universe, trading_dates


def quarter_ordinal(value: str) -> int:
    text = str(value).lower()
    year, quarter = text.split("q")
    return int(year) * 4 + int(quarter) - 1


def quarter_name(ordinal: int) -> str:
    return f"{ordinal // 4}q{ordinal % 4 + 1}"


def date_quarter(value: str | pd.Timestamp) -> str:
    date = pd.Timestamp(value)
    return f"{date.year}q{(date.month - 1) // 3 + 1}"


def iter_stock_chunks(stocks: list[str], chunk_size: int = 100):
    for start in range(0, len(stocks), chunk_size):
        yield start, stocks[start : start + chunk_size]


def fetch_pit_financials(
    rq: Any,
    stocks: list[str],
    fields: list[str],
    start_quarter: str,
    end_quarter: str,
    end_date: str,
) -> pd.DataFrame:
    parts = []
    total = len(stocks)
    for offset, chunk in iter_stock_chunks(stocks):
        raw = rq.get_pit_financials_ex(
            chunk,
            fields=list(fields),
            start_quarter=start_quarter,
            end_quarter=end_quarter,
            date=end_date,
            statements="all",
        )
        if raw is not None and not raw.empty:
            parts.append(normalize_rq_frame(raw))
        done = min(offset + len(chunk), total)
        print(f"download PIT financials progress: {done}/{total} stocks", flush=True)
    if not parts:
        return pd.DataFrame(columns=["stock_code", "quarter", "info_date", "if_adjusted", *fields])
    return pd.concat(parts, ignore_index=True)


def prepare_pit(frame: pd.DataFrame, fields: list[str], end_date: str) -> pd.DataFrame:
    required = {"stock_code", "quarter", "info_date", "if_adjusted", *fields}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"PIT financial data is missing columns: {missing}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["info_date"], errors="coerce").dt.normalize()
    result = result[result["date"] <= pd.Timestamp(end_date)]
    result = result[pd.to_numeric(result["if_adjusted"], errors="coerce").fillna(0) == 0]
    result = result.dropna(subset=["stock_code", "quarter", "date"])
    result["quarter_ord"] = result["quarter"].map(quarter_ordinal)
    for field in fields:
        result[field] = pd.to_numeric(result[field], errors="coerce")
    sort_columns = ["stock_code", "date", "quarter_ord"]
    if "rice_create_tm" in result.columns:
        sort_columns.append("rice_create_tm")
    result = result.sort_values(sort_columns)
    return result.drop_duplicates(["stock_code", "date", "quarter_ord"], keep="last")


def compute_value(state: dict[int, pd.Series], current_quarter: int) -> float:
    row = state[current_quarter]
    previous = state.get(current_quarter - 4)
    if previous is None:
        return np.nan
    values = [
        row["np_parent_company_ownersTTM"],
        row["net_profitTTM"],
        row["net_operate_cashflowTTM"],
        row["total_assets"],
        row["equity_parent_company"],
        previous["equity_parent_company"],
    ]
    if not np.isfinite(values).all():
        return np.nan
    average_equity = (row["equity_parent_company"] + previous["equity_parent_company"]) / 2
    if average_equity <= 0 or row["total_assets"] <= 0:
        return np.nan
    roe = row["np_parent_company_ownersTTM"] / average_equity
    accrual_ratio = (
        row["net_profitTTM"] - row["net_operate_cashflowTTM"]
    ) / row["total_assets"]
    return float(roe - accrual_ratio)


def factor_events(
    pit: pd.DataFrame,
    fields: list[str],
    end_date: str,
    value_function: Callable[[dict[int, pd.Series], int], float],
) -> pd.DataFrame:
    frame = prepare_pit(pit, fields, end_date)
    rows = []
    grouped = frame.groupby("stock_code", sort=True)
    total = grouped.ngroups
    for stock_index, (stock_code, stock_frame) in enumerate(grouped, start=1):
        state: dict[int, pd.Series] = {}
        for date, releases in stock_frame.groupby("date", sort=True):
            for _, release in releases.iterrows():
                state[int(release["quarter_ord"])] = release
            current_quarter = max(state)
            value = value_function(state, current_quarter)
            rows.append({"date": date, "stock_code": stock_code, "factor_value": value})
        if stock_index % 100 == 0 or stock_index == total:
            print(f"calculate factor progress: {stock_index}/{total} stocks", flush=True)
    if not rows:
        return pd.DataFrame(columns=[*KEYS, "factor_value"])
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def daily_factor_from_events(
    events: pd.DataFrame,
    dates: pd.DatetimeIndex,
    stocks: list[str],
    start_date: str,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=[*KEYS, "factor_value"])
    events = events.sort_values(KEYS).drop_duplicates(KEYS, keep="last")
    output_dates = dates[dates >= pd.Timestamp(start_date)]
    event_groups = {
        stock_code: stock_events
        for stock_code, stock_events in events.groupby("stock_code", sort=False)
    }
    rows = []
    for stock_code in stocks:
        stock_events = event_groups.get(stock_code)
        if stock_events is None:
            continue
        event_dates = stock_events["date"].to_numpy(dtype="datetime64[ns]")
        event_values = stock_events["factor_value"].to_numpy(dtype=float)
        positions = np.searchsorted(
            event_dates, output_dates.to_numpy(dtype="datetime64[ns]"), side="left"
        ) - 1
        valid = positions >= 0
        values = np.full(len(output_dates), np.nan)
        values[valid] = event_values[positions[valid]]
        finite = np.isfinite(values)
        if finite.any():
            rows.append(
                pd.DataFrame(
                    {
                        "date": output_dates[finite],
                        "stock_code": stock_code,
                        "factor_value": values[finite],
                    }
                )
            )
    if not rows:
        return pd.DataFrame(columns=[*KEYS, "factor_value"])
    return pd.concat(rows, ignore_index=True)[[*KEYS, "factor_value"]].sort_values(KEYS)


def build_factor(
    pit: pd.DataFrame,
    dates: pd.DatetimeIndex,
    stocks: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    events = factor_events(pit, PIT_FIELDS, end_date, compute_value)
    return daily_factor_from_events(events, dates, stocks, start_date)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build roe_quality factor CSV.")
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--index-code", default=CSI1000)
    parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--output-path", type=Path, default=SCRIPT_DIR / "data" / "factor.csv")
    parser.add_argument("--sample-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rq = init_rqdatac(args.env_path)
    universe, dates = build_universe(rq, args.start_date, args.end_date, args.index_code)
    stocks = sorted(universe["stock_code"].unique())
    if args.sample_size:
        stocks = stocks[: args.sample_size]
    end_quarter = date_quarter(args.end_date)
    start_ordinal = quarter_ordinal(date_quarter(args.start_date)) - HISTORY_QUARTERS
    pit = fetch_pit_financials(
        rq, stocks, PIT_FIELDS, quarter_name(start_ordinal), end_quarter, args.end_date
    )
    factor = build_factor(pit, dates, stocks, args.start_date, args.end_date)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    factor.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    print(f"factor written: {args.output_path.resolve()} shape={factor.shape}")


if __name__ == "__main__":
    main()
