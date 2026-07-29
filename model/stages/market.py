"""Build the model market panel and official trading calendar from rqdatac."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def log(message: str) -> None:
    print(message, flush=True)


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


def stack_bool_matrix(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    result = frame.copy()
    if isinstance(result.index, pd.MultiIndex):
        result = normalize_rq_frame(result.reset_index())
        if value_name not in result.columns:
            value_columns = [column for column in result.columns if column not in KEYS]
            if len(value_columns) == 1:
                result = result.rename(columns={value_columns[0]: value_name})
        return result[[*KEYS, value_name]]

    result.index = pd.to_datetime(result.index).normalize()
    result = (
        result.rename_axis("date")
        .stack(dropna=False)
        .rename(value_name)
        .reset_index()
        .rename(columns={"level_1": "stock_code"})
    )
    return result[[*KEYS, value_name]]


def build_universe(
    rq: Any,
    start_date: str,
    end_date: str,
    index_code: str,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
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
            rows.append({"date": date, "stock_code": stock, "in_universe": True})

    universe = pd.DataFrame(rows).drop_duplicates(KEYS).sort_values(KEYS)
    return universe, trading_dates


def build_prices(
    rq: Any,
    stocks: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    raw = rq.get_price(
        stocks,
        start_date=start_date,
        end_date=end_date,
        frequency="1d",
        fields=["open", "close", "limit_up", "limit_down"],
        adjust_type="none",
        skip_suspended=False,
    )
    post = rq.get_price(
        stocks,
        start_date=start_date,
        end_date=end_date,
        frequency="1d",
        fields=["open", "close"],
        adjust_type="post",
        skip_suspended=False,
    )

    raw = normalize_rq_frame(raw).rename(
        columns={"open": "raw_open", "close": "raw_close"}
    )
    post = normalize_rq_frame(post).rename(
        columns={"open": "post_open", "close": "post_close"}
    )
    prices = raw.merge(
        post[[*KEYS, "post_open", "post_close"]],
        on=KEYS,
        how="left",
    )
    return prices[
        [
            "date",
            "stock_code",
            "post_open",
            "post_close",
            "raw_open",
            "raw_close",
            "limit_up",
            "limit_down",
        ]
    ].drop_duplicates(KEYS).sort_values(KEYS)


def build_status(
    rq: Any,
    stocks: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    st = stack_bool_matrix(
        rq.is_st_stock(stocks, start_date=start_date, end_date=end_date),
        "is_st",
    )
    suspended = stack_bool_matrix(
        rq.is_suspended(stocks, start_date=start_date, end_date=end_date),
        "is_suspended",
    )
    status = st.merge(suspended, on=KEYS, how="outer")

    instruments = normalize_rq_frame(rq.all_instruments(type="CS"))
    listed = (
        instruments[["stock_code", "listed_date"]]
        .dropna(subset=["stock_code", "listed_date"])
        .drop_duplicates("stock_code")
        .set_index("stock_code")["listed_date"]
    )
    listed_date = pd.to_datetime(status["stock_code"].map(listed)).dt.normalize()
    if listed_date.dropna().empty:
        raise RuntimeError("No listed_date values are available for selected stocks")

    full_calendar = pd.Index(
        pd.to_datetime(
            rq.get_trading_dates(listed_date.dropna().min().date(), end_date)
        ).normalize()
    )
    calendar_pos = pd.Series(np.arange(len(full_calendar)), index=full_calendar)
    listed_pos = full_calendar.searchsorted(listed_date, side="left")
    current_pos = pd.to_datetime(status["date"]).map(calendar_pos).astype("Int64")
    status["listing_days"] = (current_pos - listed_pos + 1).astype("Float64")
    status.loc[listed_date.isna(), "listing_days"] = np.nan
    status.loc[status["listing_days"] < 0, "listing_days"] = 0
    status["is_st"] = status["is_st"].fillna(True).astype(bool)
    status["is_suspended"] = status["is_suspended"].fillna(True).astype(bool)
    return status[[*KEYS, "is_st", "is_suspended", "listing_days"]].sort_values(KEYS)


def normalize_market_cap(frame: pd.DataFrame) -> pd.DataFrame:
    result = normalize_rq_frame(frame)
    if "market_cap" not in result.columns:
        value_columns = [column for column in result.columns if column not in KEYS]
        if len(value_columns) == 1:
            result = result.rename(columns={value_columns[0]: "market_cap"})
    return result[[*KEYS, "market_cap"]]


def build_exposures(
    rq: Any,
    universe: pd.DataFrame,
    stocks: list[str],
    start_date: str,
    end_date: str,
    industry_source: str,
) -> pd.DataFrame:
    log("download market_cap")
    market_cap = normalize_market_cap(
        rq.get_factor(
            stocks,
            "market_cap",
            start_date=start_date,
            end_date=end_date,
        )
    )

    log("download daily industry classification")
    industry_rows = []
    for index, (date, group) in enumerate(universe.groupby("date"), start=1):
        daily_stocks = group["stock_code"].tolist()
        industry = rq.get_instrument_industry(
            daily_stocks,
            source=industry_source,
            level=1,
            date=pd.Timestamp(date).date(),
        )
        industry = normalize_rq_frame(industry)
        if "industry" not in industry.columns:
            for candidate in ["first_industry_name", "industry_name", "citics_2019_1"]:
                if candidate in industry.columns:
                    industry = industry.rename(columns={candidate: "industry"})
                    break
        if "industry" not in industry.columns:
            value_columns = [column for column in industry.columns if column not in KEYS]
            if not value_columns:
                raise RuntimeError("industry classification has no industry column")
            industry = industry.rename(columns={value_columns[-1]: "industry"})
        industry["date"] = pd.Timestamp(date).normalize()
        industry_rows.append(industry[["date", "stock_code", "industry"]])
        if index % 100 == 0:
            log(f"  industry progress: {index} trading dates")

    industries = pd.concat(industry_rows, ignore_index=True)
    exposures = universe[KEYS].merge(industries, on=KEYS, how="left")
    exposures = exposures.merge(market_cap, on=KEYS, how="left")
    exposures["industry"] = exposures["industry"].fillna("Unknown")
    return exposures[[*KEYS, "industry", "market_cap"]].sort_values(KEYS)


def build_market_panel(
    *,
    env_path: Path,
    start_date: str,
    end_date: str,
    index_code: str,
    industry_source: str,
    sample_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rq = init_rqdatac(env_path)
    log("download universe snapshots")
    universe, trading_dates = build_universe(
        rq,
        start_date,
        end_date,
        index_code,
    )
    if sample_size:
        selected = sorted(universe["stock_code"].unique())[:sample_size]
        universe = universe[universe["stock_code"].isin(selected)].copy()
    stocks = sorted(universe["stock_code"].unique())

    log(f"download prices for {len(stocks)} stocks")
    prices = build_prices(rq, stocks, start_date, end_date)
    log("download status")
    status = build_status(rq, stocks, start_date, end_date)
    exposures = build_exposures(
        rq,
        universe,
        stocks,
        start_date,
        end_date,
        industry_source,
    )

    # The explicit official-calendar skeleton prevents a fully missing market
    # date from being mistaken for the next trading day downstream.
    skeleton = pd.MultiIndex.from_product(
        [trading_dates, stocks],
        names=KEYS,
    ).to_frame(index=False)
    panel = skeleton.merge(
        prices,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator="_price_match",
    )
    panel["has_price_record"] = panel["_price_match"].eq("both")
    panel = panel.drop(columns="_price_match")
    panel = panel.merge(universe, on=KEYS, how="left", validate="one_to_one")
    panel["in_universe"] = (
        panel["in_universe"].astype("boolean").fillna(False).astype(bool)
    )
    panel = panel.merge(
        status,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator="_status_match",
    )
    panel["has_status_record"] = panel["_status_match"].eq("both")
    panel = panel.drop(columns="_status_match")
    panel = panel.merge(exposures, on=KEYS, how="left", validate="one_to_one")
    panel["is_st"] = (
        panel["is_st"].astype("boolean").fillna(True).astype(bool)
    )
    panel["is_suspended"] = (
        panel["is_suspended"].astype("boolean").fillna(True).astype(bool)
    )
    panel["industry"] = panel["industry"].fillna("Unknown")
    panel = panel[
        [
            "date",
            "stock_code",
            "in_universe",
            "has_price_record",
            "has_status_record",
            "post_open",
            "post_close",
            "raw_open",
            "raw_close",
            "limit_up",
            "limit_down",
            "is_st",
            "is_suspended",
            "listing_days",
            "industry",
            "market_cap",
        ]
    ].sort_values(KEYS)
    calendar = pd.DataFrame(
        {"date": pd.DatetimeIndex(trading_dates).sort_values().unique()}
    )
    if set(panel["date"].unique()) != set(calendar["date"]):
        raise RuntimeError("market panel does not cover the official trading calendar")
    return panel, calendar


def write_market_outputs(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
    panel_path: Path,
    calendar_path: Path,
) -> None:
    """Write the two market source artifacts using the established formats."""
    panel_path = Path(panel_path)
    calendar_path = Path(calendar_path)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_csv(
        calendar_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )
    print(f"market_panel written: {panel_path.resolve()} shape={panel.shape}")
    print(
        "trading_calendar written: "
        f"{calendar_path.resolve()} shape={calendar.shape}"
    )
