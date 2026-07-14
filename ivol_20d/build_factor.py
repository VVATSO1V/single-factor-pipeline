"""Build factor.csv for the 20-day idiosyncratic volatility factor.

For each stock and date T, use the latest 20 visible daily returns ending at T:

    r_i = a + b * r_mkt_i + e_i
    ivol_20d(T) = std(e_i)
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
) -> pd.DataFrame:
    trading_dates = [
        pd.Timestamp(date).normalize()
        for date in rq.get_trading_dates(start_date, end_date)
    ]
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

    return pd.DataFrame(rows).drop_duplicates(KEYS).sort_values(KEYS)


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


def fetch_post_close(
    rq: Any,
    stocks: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    parts = []
    total = len(stocks)
    for offset, chunk in iter_stock_chunks(stocks):
        raw = rq.get_price(
            chunk,
            start_date=start_date,
            end_date=end_date,
            frequency="1d",
            fields=["close"],
            adjust_type="post",
            skip_suspended=False,
        )
        frame = normalize_rq_frame(raw).rename(columns={"close": "post_close"})
        parts.append(frame[[*KEYS, "post_close"]])
        done = min(offset + len(chunk), total)
        print(f"download post_close progress: {done}/{total} stocks", flush=True)
    if not parts:
        return pd.DataFrame(columns=[*KEYS, "post_close"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(KEYS).sort_values(KEYS)


def fetch_index_close(
    rq: Any,
    index_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    raw = rq.get_price(
        index_code,
        start_date=start_date,
        end_date=end_date,
        frequency="1d",
        fields=["close"],
        adjust_type="none",
        skip_suspended=False,
    )
    frame = normalize_rq_frame(raw).rename(columns={"close": "index_close"})
    if "date" not in frame.columns:
        raise RuntimeError("index price data has no date column")
    return frame[["date", "index_close"]].drop_duplicates("date").sort_values("date")


def wide_post_close(price: pd.DataFrame) -> pd.DataFrame:
    frame = price.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["post_close"] = pd.to_numeric(frame["post_close"], errors="coerce")
    return frame.pivot(index="date", columns="stock_code", values="post_close").sort_index()


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
    close: pd.DataFrame,
    index_close: pd.DataFrame,
    window: int,
    start_date: str,
) -> pd.DataFrame:
    stock_return = close.pct_change()
    market_return = (
        index_close.set_index("date")["index_close"]
        .sort_index()
        .astype(float)
        .pct_change()
    )
    market_return = market_return.reindex(stock_return.index)

    stock_var = stock_return.rolling(window=window, min_periods=window).var()
    market_var = market_return.rolling(window=window, min_periods=window).var()
    stock_market_cov = stock_return.rolling(window=window, min_periods=window).cov(
        market_return
    )
    residual_var = stock_var - stock_market_cov.pow(2).div(market_var, axis=0)
    residual_var = residual_var.mask(market_var <= 0, axis=0)
    residual_var = residual_var.where(residual_var >= 0, 0.0)
    factor = np.sqrt(residual_var)
    return long_factor_from_wide(factor, start_date)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ivol_20d factor CSV.")
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--index-code", default=CSI1000)
    parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=SCRIPT_DIR / "data" / "factor.csv",
    )
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--sample-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rq = init_rqdatac(args.env_path)
    price_start = lookback_start_date(rq, args.start_date, args.window + 2)
    universe = build_universe(rq, args.start_date, args.end_date, args.index_code)
    stocks = sorted(universe["stock_code"].unique())
    if args.sample_size:
        stocks = stocks[: args.sample_size]
    price = fetch_post_close(rq, stocks, price_start, args.end_date)
    index_close = fetch_index_close(rq, args.index_code, price_start, args.end_date)
    factor = build_factor(wide_post_close(price), index_close, args.window, args.start_date)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    factor.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    print(f"factor written: {args.output_path.resolve()} shape={factor.shape}")


if __name__ == "__main__":
    main()
