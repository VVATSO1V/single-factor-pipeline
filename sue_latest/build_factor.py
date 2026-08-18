"""Build factor.csv for the latest standardized unexpected earnings factor.

The event-level factor is:

    sue_latest = (actual_np - expected_np) / std(previous 8 earnings surprises)

Only the first official financial-report event for each fiscal quarter is used.
When RQData omits adjusted actual profit for annual reports, the profit
difference is reconstructed from profit_ex_rate and consensus profit.
The latest event value becomes effective on the next trading date and is then
forward-filled until the next valid quarterly event.
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
PERIOD_ORDER = {"q1": 1, "q2": 2, "q3": 3, "q4": 4}


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


def event_query_start(start_date: str, history_quarters: int) -> str:
    years = max(4, int(np.ceil((history_quarters + 4) / 4)))
    return str((pd.Timestamp(start_date) - pd.DateOffset(years=years)).date())


def iter_stock_chunks(stocks: list[str], chunk_size: int = 100):
    for start in range(0, len(stocks), chunk_size):
        yield start, stocks[start : start + chunk_size]


def fetch_expectation_events(
    rq: Any,
    stocks: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    parts = []
    total = len(stocks)
    for offset, chunk in iter_stock_chunks(stocks):
        raw = rq.consensus.get_expect_appr_exceed(
            chunk,
            start_date=start_date,
            end_date=end_date,
            report_year=None,
            report_periods=None,
            report_types=["financial_reports"],
            appraisal_results=None,
        )
        if raw is not None:
            parts.append(normalize_rq_frame(raw))
        done = min(offset + len(chunk), total)
        print(f"download SUE event progress: {done}/{total} stocks", flush=True)

    if not parts:
        return pd.DataFrame(columns=[*KEYS, "adjust_con_profit", "con_profit"])
    return pd.concat(parts, ignore_index=True).sort_values(KEYS)


def prepare_event_factor(
    events: pd.DataFrame,
    actual_column: str,
    expected_column: str,
    history_quarters: int,
) -> pd.DataFrame:
    if history_quarters < 2:
        raise ValueError("history_quarters must be at least 2")
    frame = events.copy()
    if frame.empty:
        return pd.DataFrame(columns=[*KEYS, "factor_value"])

    frame[actual_column] = pd.to_numeric(frame[actual_column], errors="coerce")
    frame[expected_column] = pd.to_numeric(frame[expected_column], errors="coerce")
    if "profit_ex_rate" in frame.columns:
        frame["profit_ex_rate"] = pd.to_numeric(
            frame["profit_ex_rate"], errors="coerce"
        )
    else:
        frame["profit_ex_rate"] = np.nan
    required = {"report_year", "report_period"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"SUE event data is missing columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["report_year"] = pd.to_numeric(frame["report_year"], errors="coerce")
    frame["report_period"] = frame["report_period"].astype(str).str.lower()
    frame["_period_order"] = frame["report_period"].map(PERIOD_ORDER)
    frame = frame.dropna(
        subset=[
            "date",
            "stock_code",
            "report_year",
            "_period_order",
            expected_column,
        ]
    )
    frame["raw_surprise"] = frame[actual_column] - frame[expected_column]
    reconstructed = frame["profit_ex_rate"] * frame[expected_column].abs()
    frame["raw_surprise"] = frame["raw_surprise"].fillna(reconstructed)
    frame = frame.dropna(subset=["raw_surprise"])
    frame["quarter_ord"] = (
        frame["report_year"].astype(int) * 4
        + frame["_period_order"].astype(int)
        - 1
    )
    if "appraisal_standard" in frame.columns:
        frame["_standard_order"] = pd.to_numeric(
            frame["appraisal_standard"], errors="coerce"
        ).fillna(99)
    else:
        frame["_standard_order"] = 99

    period_keys = ["stock_code", "report_year", "report_period"]
    frame = frame.sort_values([*period_keys, "date", "_standard_order"])
    frame = frame.drop_duplicates([*period_keys, "date"], keep="first")
    frame = frame.sort_values([*period_keys, "date"])
    frame = frame.drop_duplicates(period_keys, keep="first")

    rows = []
    for stock_code, stock_events in frame.groupby("stock_code", sort=True):
        surprise_by_quarter: dict[int, float] = {}
        for _, event in stock_events.sort_values("date").iterrows():
            quarter = int(event["quarter_ord"])
            previous = [
                surprise_by_quarter.get(prior_quarter)
                for prior_quarter in range(quarter - history_quarters, quarter)
            ]
            if all(value is not None and np.isfinite(value) for value in previous):
                surprise_std = np.std(previous, ddof=1)
                if np.isfinite(surprise_std) and surprise_std > 0:
                    rows.append(
                        {
                            "date": event["date"],
                            "stock_code": stock_code,
                            "factor_value": event["raw_surprise"] / surprise_std,
                        }
                    )
            surprise_by_quarter[quarter] = float(event["raw_surprise"])

    if not rows:
        return pd.DataFrame(columns=[*KEYS, "factor_value"])
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(
        subset=["factor_value"]
    )


def daily_factor_from_events(
    event_factor: pd.DataFrame,
    dates: pd.DatetimeIndex,
    stocks: list[str],
    start_date: str,
) -> pd.DataFrame:
    if event_factor.empty:
        return pd.DataFrame(columns=[*KEYS, "factor_value"])
    events = event_factor.sort_values(KEYS).drop_duplicates(KEYS, keep="last")
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
            event_dates,
            output_dates.to_numpy(dtype="datetime64[ns]"),
            side="left",
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
    events: pd.DataFrame,
    dates: pd.DatetimeIndex,
    stocks: list[str],
    actual_column: str,
    expected_column: str,
    history_quarters: int,
    start_date: str,
) -> pd.DataFrame:
    event_factor = prepare_event_factor(
        events,
        actual_column,
        expected_column,
        history_quarters,
    )
    return daily_factor_from_events(event_factor, dates, stocks, start_date)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sue_latest factor CSV.")
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--index-code", default=CSI1000)
    parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=SCRIPT_DIR / "data" / "factor.csv",
    )
    parser.add_argument("--actual-column", default="adjust_con_profit")
    parser.add_argument("--expected-column", default="con_profit")
    parser.add_argument("--history-quarters", type=int, default=8)
    parser.add_argument("--sample-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rq = init_rqdatac(args.env_path)
    query_start = event_query_start(args.start_date, args.history_quarters)
    universe, dates = build_universe(rq, args.start_date, args.end_date, args.index_code)
    stocks = sorted(universe["stock_code"].unique())
    if args.sample_size:
        stocks = stocks[: args.sample_size]
    events = fetch_expectation_events(rq, stocks, query_start, args.end_date)
    factor = build_factor(
        events,
        dates,
        stocks,
        args.actual_column,
        args.expected_column,
        args.history_quarters,
        args.start_date,
    )
    if factor.empty:
        raise RuntimeError(
            "SUE factor is empty; existing factor.csv was not overwritten. "
            "Check RQData event coverage and selected columns."
        )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    factor.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    print(f"factor written: {args.output_path.resolve()} shape={factor.shape}")


if __name__ == "__main__":
    main()
