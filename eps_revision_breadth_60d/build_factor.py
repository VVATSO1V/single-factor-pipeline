"""Build factor.csv for the 60-day EPS revision breadth factor.

The factor is:

    eps_revision_breadth_60d(T)
        = (up institutes - down institutes) / comparable institutes

For each institute, only its latest comparable revision during the rolling
window is counted. A revision compares two visible EPS forecasts for the same
fiscal year, so first-time coverage is not treated as an up or down revision.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
CSI1000 = "000852.XSHG"
KEYS = ["date", "stock_code"]


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

    rename_map = {}
    for column in result.columns:
        lower = str(column).lower()
        if lower in {"order_book_id", "order_book_ids", "instrument"}:
            rename_map[column] = "stock_code"
        elif lower in {"datetime", "trading_date"}:
            rename_map[column] = "date"
    result = result.rename(columns=rename_map)

    if "date" not in result.columns:
        for column in result.columns:
            if np.issubdtype(result[column].dtype, np.datetime64):
                result = result.rename(columns={column: "date"})
                break

    if "stock_code" not in result.columns:
        for column in result.columns:
            if column == "date" or result[column].dtype != object:
                continue
            sample = result[column].dropna().astype(str).head(20)
            if sample.str.contains(r"\.XS(HG|HE)$", regex=True).any():
                result = result.rename(columns={column: "stock_code"})
                break

    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    return result


def build_universe(
    rq: Any,
    start_date: str,
    end_date: str,
    index_code: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    trading_dates = pd.DatetimeIndex(
        [pd.Timestamp(date).normalize() for date in rq.get_trading_dates(start_date, end_date)]
    )
    raw_components = rq.index_components(
        index_code,
        start_date=start_date,
        end_date=end_date,
    )

    snapshot_map: dict[pd.Timestamp, list[str]] = {}
    if isinstance(raw_components, dict):
        snapshot_map = {
            pd.Timestamp(date).normalize(): list(stocks)
            for date, stocks in raw_components.items()
        }
    elif isinstance(raw_components, pd.DataFrame):
        components = normalize_rq_frame(raw_components)
        for date, group in components.groupby("date"):
            snapshot_map[pd.Timestamp(date).normalize()] = group["stock_code"].tolist()

    if not snapshot_map:
        raise RuntimeError("index_components did not return historical snapshots")

    known_dates = sorted(snapshot_map)
    rows = []
    pointer = 0
    current_stocks: list[str] | None = None
    for date in trading_dates:
        while pointer < len(known_dates) and known_dates[pointer] <= date:
            current_stocks = snapshot_map[known_dates[pointer]]
            pointer += 1
        if current_stocks is None:
            raise RuntimeError(f"No index component snapshot is available before {date}")
        for stock in current_stocks:
            rows.append({"date": date, "stock_code": stock})

    universe = pd.DataFrame(rows).drop_duplicates(KEYS).sort_values(KEYS)
    return universe, trading_dates


def lookback_start_date(rq: Any, start_date: str, lookback_days: int) -> str:
    start = pd.Timestamp(start_date)
    calendar_start = (start - pd.DateOffset(days=max(365, lookback_days * 3))).date()
    dates = pd.to_datetime(rq.get_trading_dates(calendar_start, start.date()))
    if len(dates) <= lookback_days:
        return str(dates.min().date())
    return str(dates[-lookback_days - 1].date())


def iter_stock_chunks(stocks: list[str], chunk_size: int = 100):
    for start in range(0, len(stocks), chunk_size):
        yield start, stocks[start : start + chunk_size]


def fetch_eps_reports(
    rq: Any,
    stocks: list[str],
    start_date: str,
    end_date: str,
    eps_field: str,
) -> pd.DataFrame:
    parts = []
    total = len(stocks)
    for offset, chunk in iter_stock_chunks(stocks):
        raw = rq.consensus.get_indicator(
            chunk,
            fiscal_year=None,
            fields=[eps_field],
            start_date=start_date,
            end_date=end_date,
            date_rule="rpt_dt",
        )
        if raw is not None:
            frame = normalize_rq_frame(raw)
            frame[eps_field] = pd.to_numeric(frame[eps_field], errors="coerce")
            parts.append(frame)
        done = min(offset + len(chunk), total)
        print(f"download EPS report progress: {done}/{total} stocks", flush=True)

    if not parts:
        return pd.DataFrame(columns=[*KEYS, "institute", "fiscal_year", eps_field])
    result = pd.concat(parts, ignore_index=True)
    result = result.dropna(subset=["date", "stock_code", "institute", eps_field])
    if "fiscal_year" not in result.columns:
        result["fiscal_year"] = np.nan
    return result.sort_values(["stock_code", "institute", "fiscal_year", "date"])


def classify_revision_events(reports: pd.DataFrame, eps_field: str) -> pd.DataFrame:
    frame = reports.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame[eps_field] = pd.to_numeric(frame[eps_field], errors="coerce")
    frame = frame.dropna(
        subset=["date", "stock_code", "institute", "fiscal_year", eps_field]
    )
    frame = frame.sort_values(["stock_code", "institute", "fiscal_year", "date"])
    group_keys = ["stock_code", "institute", "fiscal_year"]
    frame["previous_eps"] = frame.groupby(group_keys, dropna=False)[eps_field].shift(1)
    frame = frame.dropna(subset=["previous_eps"])
    frame["direction"] = np.sign(frame[eps_field] - frame["previous_eps"])
    return frame[[*KEYS, "institute", "fiscal_year", "direction"]]


def rolling_latest_revision_breadth(
    events: pd.DataFrame,
    dates: pd.DatetimeIndex,
    stocks: list[str],
    window: int,
) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be positive")
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().sort_values().unique()
    result = pd.DataFrame(index=dates, columns=stocks, dtype=float)
    event_groups = {
        stock_code: stock_events
        for stock_code, stock_events in events.groupby("stock_code", sort=False)
    }
    for done, stock in enumerate(stocks, start=1):
        stock_events = event_groups.get(stock)
        if stock_events is not None and not stock_events.empty:
            stock_events = stock_events[
                ["date", "institute", "fiscal_year", "direction"]
            ].copy()
            positions = dates.searchsorted(
                pd.DatetimeIndex(stock_events["date"]),
                side="left",
            )
            stock_events = stock_events.loc[positions < len(dates)].copy()
            positions = positions[positions < len(dates)]
            if stock_events.empty:
                continue
            stock_events["effective_date"] = dates[positions]
            stock_events = (
                stock_events.sort_values(
                    ["effective_date", "date", "fiscal_year", "institute"]
                )
                .drop_duplicates(["effective_date", "institute"], keep="last")
            )
            matrix = (
                stock_events.pivot(
                    index="effective_date",
                    columns="institute",
                    values="direction",
                )
                .reindex(dates)
            )
            latest = matrix if window == 1 else matrix.ffill(limit=window - 1)
            covered = latest.notna().sum(axis=1)
            up = latest.gt(0).sum(axis=1)
            down = latest.lt(0).sum(axis=1)
            result[stock] = (up - down).div(covered.where(covered > 0))
        if done % 100 == 0 or done == len(stocks):
            print(f"build rolling breadth progress: {done}/{len(stocks)} stocks", flush=True)
    return result


def long_factor_from_wide(values: pd.DataFrame, start_date: str) -> pd.DataFrame:
    values = values.copy()
    values.index.name = "date"
    values.columns.name = "stock_code"
    result = (
        values[values.index >= pd.Timestamp(start_date)]
        .stack(future_stack=True)
        .rename("factor_value")
        .reset_index()
        .dropna(subset=["factor_value"])
    )
    return result[[*KEYS, "factor_value"]].sort_values(KEYS)


def build_factor(
    reports: pd.DataFrame,
    dates: pd.DatetimeIndex,
    stocks: list[str],
    eps_field: str,
    window: int,
    start_date: str,
) -> pd.DataFrame:
    events = classify_revision_events(reports, eps_field)
    if events.empty:
        return pd.DataFrame(columns=[*KEYS, "factor_value"])
    factor = rolling_latest_revision_breadth(events, dates, stocks, window)
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return long_factor_from_wide(factor, start_date)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build eps_revision_breadth_60d factor CSV.")
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--index-code", default=CSI1000)
    parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=SCRIPT_DIR / "data" / "factor.csv",
    )
    parser.add_argument("--eps-field", default="eps_t1")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--sample-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rq = init_rqdatac(args.env_path)
    query_start = lookback_start_date(rq, args.start_date, args.window * 2 + 5)
    universe, dates = build_universe(rq, args.start_date, args.end_date, args.index_code)
    stocks = sorted(universe["stock_code"].unique())
    if args.sample_size:
        stocks = stocks[: args.sample_size]
    reports = fetch_eps_reports(rq, stocks, query_start, args.end_date, args.eps_field)
    factor = build_factor(reports, dates, stocks, args.eps_field, args.window, args.start_date)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    factor.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    print(f"factor written: {args.output_path.resolve()} shape={factor.shape}")


if __name__ == "__main__":
    main()
