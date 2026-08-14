"""Immutable configuration and input seals for static strategy execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, BinaryIO, Mapping, Sequence

import numpy as np
import pandas as pd

from rank_model.stages.dataset import file_sha256


LOCKED_MODEL_NAMES = (
    "ridge_rank_regression",
    "xgboost_rank_regression",
    "lightgbm_rank_regression",
    "lightgbm_lambdarank",
    "mlp_top100_hybrid_rank",
)


STRATEGY_START = pd.Timestamp("2024-01-01")
STRATEGY_END = pd.Timestamp("2025-12-31")
STRATEGY_TOP_K = 100
STRATEGY_CROSS_SECTION_SIZE = 1000
STRATEGY_MIN_LISTING_DAYS = 120
STRATEGY_INITIAL_NAV = 1.0
STRATEGY_COMMISSION_BPS = 1.0
STRATEGY_SLIPPAGE_BPS = 5.0
STRATEGY_SELL_STAMP_DUTY_BPS = 5.0
STRATEGY_LIMIT_TOLERANCE = 1e-8
STRATEGY_ANNUALIZATION_DAYS = 252
PREDICTION_COLUMNS = ("date", "stock_code", "score_raw", "split", "horizon")
MARKET_EXECUTION_COLUMNS = (
    "date",
    "stock_code",
    "has_price_record",
    "is_suspended",
    "is_st",
    "listing_days",
    "raw_open",
    "post_open",
    "limit_up",
    "limit_down",
)
MARKET_EXECUTION_DTYPES = {
    "date": "string",
    "stock_code": "string",
    "has_price_record": "boolean",
    "is_suspended": "boolean",
    "is_st": "boolean",
    "listing_days": "float64",
    "raw_open": "float64",
    "post_open": "float64",
    "limit_up": "float64",
    "limit_down": "float64",
}
MARKET_READ_CHUNK_SIZE = 100_000


@dataclass(frozen=True)
class StrategySettings:
    start: pd.Timestamp
    end: pd.Timestamp
    top_k: int
    expected_cross_section_size: int
    min_listing_days: int
    initial_nav: float
    commission_rate: float
    slippage_rate: float
    sell_stamp_duty_rate: float
    limit_tolerance: float
    annualization_days: int

    @property
    def buy_cost_rate(self) -> float:
        return round(self.commission_rate + self.slippage_rate, 12)

    @property
    def sell_cost_rate(self) -> float:
        return round(
            self.commission_rate + self.slippage_rate + self.sell_stamp_duty_rate,
            12,
        )


def _exact_number(settings: Mapping[str, Any], name: str, expected: float) -> float:
    try:
        value = float(settings[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"strategy {name} must be {expected}") from error
    if value != expected:
        raise ValueError(f"strategy {name} must be {expected}")
    return value


def _exact_integer(settings: Mapping[str, Any], name: str, expected: int) -> int:
    value = settings.get(name)
    if isinstance(value, bool) or value != expected:
        raise ValueError(f"strategy {name} must be {expected}")
    return expected


def load_strategy_settings(config: Mapping[str, Any]) -> StrategySettings:
    """Load the single permitted static-strategy configuration."""
    settings = config.get("strategy")
    if not isinstance(settings, Mapping):
        raise ValueError("config is missing [strategy]")
    if settings.get("start") != STRATEGY_START.strftime("%Y-%m-%d"):
        raise ValueError("strategy start must be 2024-01-01")
    if settings.get("end") != STRATEGY_END.strftime("%Y-%m-%d"):
        raise ValueError("strategy end must be 2025-12-31")

    return StrategySettings(
        start=STRATEGY_START,
        end=STRATEGY_END,
        top_k=_exact_integer(settings, "top_k", STRATEGY_TOP_K),
        expected_cross_section_size=_exact_integer(
            settings, "expected_cross_section_size", STRATEGY_CROSS_SECTION_SIZE
        ),
        min_listing_days=_exact_integer(
            settings, "min_listing_days", STRATEGY_MIN_LISTING_DAYS
        ),
        initial_nav=_exact_number(settings, "initial_nav", STRATEGY_INITIAL_NAV),
        commission_rate=_exact_number(
            settings, "commission_bps", STRATEGY_COMMISSION_BPS
        ) / 10_000,
        slippage_rate=_exact_number(settings, "slippage_bps", STRATEGY_SLIPPAGE_BPS)
        / 10_000,
        sell_stamp_duty_rate=_exact_number(
            settings, "sell_stamp_duty_bps", STRATEGY_SELL_STAMP_DUTY_BPS
        )
        / 10_000,
        limit_tolerance=_exact_number(
            settings, "limit_tolerance", STRATEGY_LIMIT_TOLERANCE
        ),
        annualization_days=_exact_integer(
            settings, "annualization_days", STRATEGY_ANNUALIZATION_DAYS
        ),
    )


def _calendar_dates(calendar: Any) -> pd.DatetimeIndex:
    if isinstance(calendar, pd.DataFrame):
        if "date" not in calendar:
            raise ValueError("trading calendar is missing date")
        values = calendar["date"]
    else:
        values = calendar
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise")).normalize()
    except (TypeError, ValueError) as error:
        raise ValueError("trading calendar dates are invalid") from error
    if dates.has_duplicates:
        raise ValueError("trading calendar contains duplicate dates")
    return dates.sort_values()


def _validate_predictions(
    predictions: pd.DataFrame,
    settings: StrategySettings,
    calendar: Any,
) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("predictions must be a DataFrame")
    missing = sorted(set(PREDICTION_COLUMNS).difference(predictions.columns))
    if missing:
        raise ValueError(f"predictions are missing required columns: {missing}")

    result = predictions.loc[:, list(PREDICTION_COLUMNS)].copy()
    try:
        result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    except (TypeError, ValueError) as error:
        raise ValueError("prediction dates are invalid") from error
    result["stock_code"] = result["stock_code"].astype("string").str.strip()
    if result.empty or result[["date", "stock_code"]].isna().any().any():
        raise ValueError("prediction keys must be present")
    if result["stock_code"].eq("").any():
        raise ValueError("prediction stock codes must be present")
    if result.duplicated(["date", "stock_code"]).any():
        raise ValueError("predictions contain duplicate date,stock_code keys")

    result["score_raw"] = pd.to_numeric(result["score_raw"], errors="coerce")
    if not np.isfinite(result["score_raw"].to_numpy(dtype="float64")).all():
        raise ValueError("prediction scores must be finite")
    if not result["split"].astype("string").eq("test").all():
        raise ValueError("predictions must contain only the test split")
    horizons = pd.to_numeric(result["horizon"], errors="coerce")
    if not horizons.eq(10).all():
        raise ValueError("predictions must contain exact horizon metadata 10")
    if result["date"].lt(settings.start).any() or result["date"].gt(settings.end).any():
        raise ValueError("prediction dates are outside the strategy period")

    counts = result.groupby("date", sort=False).size()
    if not counts.eq(settings.expected_cross_section_size).all():
        raise ValueError("prediction cross-section size is not exactly 1000")

    if calendar is None:
        raise ValueError("official trading calendar is required")
    official = _calendar_dates(calendar)
    if not result["date"].isin(official).all():
        raise ValueError("prediction dates do not match the trading calendar")
    return result


def validate_prediction_calendar(
    predictions: pd.DataFrame,
    calendar: Any,
    settings: StrategySettings,
) -> None:
    """Validate prediction keys, metadata, daily size, and official dates."""
    _validate_predictions(predictions, settings, calendar)


def select_daily_top(
    predictions: pd.DataFrame,
    settings: StrategySettings,
    calendar: Any,
) -> dict[pd.Timestamp, tuple[str, ...]]:
    """Select each date's deterministic highest-scoring stock codes."""
    normalized = _validate_predictions(predictions, settings, calendar)
    selected: dict[pd.Timestamp, tuple[str, ...]] = {}
    for date, group in normalized.groupby("date", sort=True):
        ordered = group.sort_values(
            ["score_raw", "stock_code"],
            ascending=[False, True],
            kind="mergesort",
        )
        selected[pd.Timestamp(date)] = tuple(
            ordered.head(settings.top_k)["stock_code"].astype(str)
        )
    return selected


def _row_value(row: Any, field: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(field)
    try:
        return row[field]
    except (KeyError, IndexError, TypeError):
        return None


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _boolean_field(row: Any, field: str) -> tuple[bool, str] | None:
    value = _row_value(row, field)
    if _missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value), ""
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value), ""
    return None


def _positive_number(row: Any, field: str) -> tuple[float | None, str | None]:
    value = _row_value(row, field)
    if _missing(value):
        return None, field
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, field
    if not np.isfinite(number) or number <= 0:
        return None, field
    return number, None


def _common_execution_data(row: Any) -> tuple[dict[str, float], tuple[bool, str] | None]:
    price_record = _boolean_field(row, "has_price_record")
    if price_record is None or not price_record[0]:
        return {}, (False, "price_record")
    suspended = _boolean_field(row, "is_suspended")
    if suspended is None:
        return {}, (False, "suspension_status")
    if suspended[0]:
        return {}, (False, "suspended")

    values: dict[str, float] = {}
    for field in ("raw_open", "post_open", "limit_down"):
        number, reason = _positive_number(row, field)
        if reason is not None:
            return {}, (False, reason)
        values[field] = number  # type: ignore[assignment]
    return values, None


def sell_decision(row: Any, settings: StrategySettings) -> tuple[bool, str]:
    """Return whether an existing position may be sold at this open."""
    values, blocked = _common_execution_data(row)
    if blocked is not None:
        return blocked
    if values["raw_open"] <= values["limit_down"] + settings.limit_tolerance:
        return False, "limit_down"
    return True, "eligible"


def buy_decision(row: Any, settings: StrategySettings) -> tuple[bool, str]:
    """Return whether a new position may be bought at this open."""
    values, blocked = _common_execution_data(row)
    if blocked is not None:
        return blocked

    st = _boolean_field(row, "is_st")
    if st is None:
        return False, "st_status"
    if st[0]:
        return False, "st"
    listing_days = _row_value(row, "listing_days")
    if _missing(listing_days):
        return False, "listing_age"
    try:
        listing_days_value = float(listing_days)
    except (TypeError, ValueError):
        return False, "listing_age"
    if not np.isfinite(listing_days_value) or listing_days_value < settings.min_listing_days:
        return False, "listing_age"

    limit_up, reason = _positive_number(row, "limit_up")
    if reason is not None:
        return False, reason
    if limit_up <= values["limit_down"]:
        return False, "limit_data"
    if values["raw_open"] >= limit_up - settings.limit_tolerance:
        return False, "limit_up"
    if values["raw_open"] <= values["limit_down"] + settings.limit_tolerance:
        return False, "limit_down"
    return True, "eligible"


@dataclass
class Position:
    units: float
    last_mark: float


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, Position]
    previous_nav: float


@dataclass
class TransitionResult:
    state: PortfolioState
    trades: pd.DataFrame
    diagnostics: dict[str, float | int]
    pre_trade_nav: float
    total_cost: float
    end_nav: float


@dataclass
class StrategyBundle:
    daily_nav: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    execution_diagnostics: pd.DataFrame
    ending_positions: pd.DataFrame
    metrics_summary: dict[str, float | int]


_TRADE_COLUMNS = (
    "stock_code",
    "signal_date",
    "execution_date",
    "side",
    "status",
    "reason",
    "requested_gross_notional",
    "gross_notional",
    "cost",
    "units",
    "raw_open",
    "post_open",
)


def _market_rows_by_stock(market: Any) -> dict[str, Any]:
    if not isinstance(market, pd.DataFrame):
        raise ValueError("execution market slice must be a DataFrame")
    if "stock_code" in market:
        stock_codes = market["stock_code"].astype("string")
    else:
        stock_codes = pd.Series(market.index, index=market.index, dtype="string")
    if stock_codes.isna().any() or stock_codes.str.strip().eq("").any():
        raise ValueError("execution market stock codes must be present")
    normalized_codes = stock_codes.str.strip()
    if normalized_codes.duplicated().any():
        raise ValueError("execution market contains duplicate stock codes")
    return {
        str(stock_code): market.iloc[index]
        for index, stock_code in enumerate(normalized_codes)
    }


def _mark_price(row: Any) -> float | None:
    if row is None:
        return None
    has_price_record = _boolean_field(row, "has_price_record")
    is_suspended = _boolean_field(row, "is_suspended")
    if (
        has_price_record is None
        or not has_price_record[0]
        or is_suspended is None
        or is_suspended[0]
    ):
        return None
    post_open, reason = _positive_number(row, "post_open")
    return None if reason is not None else post_open


def _validated_state(state: PortfolioState) -> PortfolioState:
    try:
        cash = float(state.cash)
        previous_nav = float(state.previous_nav)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("portfolio state is invalid") from error
    if not np.isfinite(cash) or cash < -1e-12:
        raise ValueError("portfolio cash must be non-negative and finite")
    if not np.isfinite(previous_nav):
        raise ValueError("portfolio previous_nav must be finite")

    positions: dict[str, Position] = {}
    for stock_code, position in state.positions.items():
        if not isinstance(stock_code, str) or not stock_code.strip():
            raise ValueError("portfolio position stock codes must be present")
        if stock_code in positions:
            raise ValueError("portfolio contains duplicate position keys")
        try:
            units = float(position.units)
            last_mark = float(position.last_mark)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("portfolio position is invalid") from error
        if not np.isfinite(units) or units <= 0:
            raise ValueError("portfolio contains non-positive units")
        if not np.isfinite(last_mark) or last_mark <= 0:
            raise ValueError("portfolio contains invalid marks")
        positions[stock_code] = Position(units=units, last_mark=last_mark)
    return PortfolioState(cash=max(cash, 0.0), positions=positions, previous_nav=previous_nav)


def _desired_stock_codes(desired: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(desired, tuple):
        raise ValueError("desired portfolio must be a tuple")
    normalized: list[str] = []
    for stock_code in desired:
        if not isinstance(stock_code, str) or not stock_code.strip():
            raise ValueError("desired stock codes must be present")
        normalized.append(stock_code.strip())
    if len(normalized) != len(set(normalized)):
        raise ValueError("desired portfolio contains duplicate order keys")
    return tuple(normalized)


def _trade_record(
    *,
    stock_code: str,
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    side: str,
    status: str,
    reason: str,
    requested_gross_notional: float,
    gross_notional: float,
    cost: float,
    units: float,
    row: Any,
) -> dict[str, Any]:
    raw_open, _ = _positive_number(row, "raw_open") if row is not None else (None, None)
    post_open, _ = _positive_number(row, "post_open") if row is not None else (None, None)
    return {
        "stock_code": stock_code,
        "signal_date": signal_date,
        "execution_date": execution_date,
        "side": side,
        "status": status,
        "reason": reason,
        "requested_gross_notional": requested_gross_notional,
        "gross_notional": gross_notional,
        "cost": cost,
        "units": units,
        "raw_open": raw_open,
        "post_open": post_open,
    }


def _trade_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    trades = pd.DataFrame(records, columns=_TRADE_COLUMNS).set_index("stock_code")
    if trades.index.has_duplicates:
        raise AssertionError("transition generated duplicate order keys")
    return trades


def transition_at_open(
    state: PortfolioState,
    desired: tuple[str, ...],
    market: pd.DataFrame,
    signal_date: Any,
    execution_date: Any,
    settings: StrategySettings,
) -> TransitionResult:
    """Execute one deterministic open-to-open set-difference rebalance."""
    result_state = _validated_state(state)
    desired_codes = _desired_stock_codes(desired)
    market_rows = _market_rows_by_stock(market)
    signal_timestamp = pd.Timestamp(signal_date).normalize()
    execution_timestamp = pd.Timestamp(execution_date).normalize()

    for stock_code, position in result_state.positions.items():
        mark = _mark_price(market_rows.get(stock_code))
        if mark is not None:
            position.last_mark = mark

    pre_trade_nav = result_state.cash + sum(
        position.units * position.last_mark
        for position in result_state.positions.values()
    )
    if not np.isfinite(pre_trade_nav) or pre_trade_nav < 0:
        raise AssertionError("pre-trade NAV is invalid")

    records: list[dict[str, Any]] = []
    desired_set = set(desired_codes)
    held_before_sales = set(result_state.positions)
    total_cost = 0.0

    for stock_code in sorted(held_before_sales - desired_set):
        position = result_state.positions[stock_code]
        row = market_rows.get(stock_code)
        eligible, reason = sell_decision(row, settings)
        requested_gross = position.units * position.last_mark
        if eligible:
            gross_notional = requested_gross
            cost = gross_notional * settings.sell_cost_rate
            result_state.cash += gross_notional - cost
            total_cost += cost
            del result_state.positions[stock_code]
            status = "executed"
            units = position.units
        else:
            gross_notional = 0.0
            cost = 0.0
            status = "blocked"
            units = 0.0
        records.append(
            _trade_record(
                stock_code=stock_code,
                signal_date=signal_timestamp,
                execution_date=execution_timestamp,
                side="sell",
                status=status,
                reason=reason,
                requested_gross_notional=requested_gross,
                gross_notional=gross_notional,
                cost=cost,
                units=units,
                row=row,
            )
        )

    entry_candidates = sorted(desired_set - set(result_state.positions))
    requested_buy_gross = pre_trade_nav / settings.top_k
    eligible_buys: dict[str, Any] = {}
    blocked_buys: dict[str, tuple[Any, str]] = {}
    for stock_code in entry_candidates:
        row = market_rows.get(stock_code)
        eligible, reason = buy_decision(row, settings)
        if eligible:
            eligible_buys[stock_code] = row
        else:
            blocked_buys[stock_code] = (row, reason)

    gross_buy_capacity = len(eligible_buys) * requested_buy_gross * (1 + settings.buy_cost_rate)
    buy_scale = min(1.0, result_state.cash / gross_buy_capacity) if gross_buy_capacity else 1.0
    buy_scale = max(0.0, buy_scale)
    for stock_code in entry_candidates:
        if stock_code in blocked_buys:
            row, reason = blocked_buys[stock_code]
            status = "blocked"
            gross_notional = 0.0
            cost = 0.0
            units = 0.0
        elif buy_scale == 0.0:
            row = eligible_buys[stock_code]
            reason = "cash"
            status = "blocked"
            gross_notional = 0.0
            cost = 0.0
            units = 0.0
        else:
            row = eligible_buys[stock_code]
            post_open, reason = _positive_number(row, "post_open")
            if reason is not None or post_open is None:
                raise AssertionError("eligible buy has no valid post_open")
            gross_notional = requested_buy_gross * buy_scale
            cost = gross_notional * settings.buy_cost_rate
            units = gross_notional / post_open
            if units <= 0:
                raise AssertionError("transition generated non-positive units")
            result_state.cash -= gross_notional + cost
            total_cost += cost
            result_state.positions[stock_code] = Position(units=units, last_mark=post_open)
            reason = "eligible"
            status = "executed"
        records.append(
            _trade_record(
                stock_code=stock_code,
                signal_date=signal_timestamp,
                execution_date=execution_timestamp,
                side="buy",
                status=status,
                reason=reason,
                requested_gross_notional=requested_buy_gross,
                gross_notional=gross_notional,
                cost=cost,
                units=units,
                row=row,
            )
        )

    trades = _trade_frame(records)
    intersection = held_before_sales.intersection(desired_set)
    if set(trades.index).intersection(intersection):
        raise AssertionError("transition generated intersection trades")
    if result_state.cash < -1e-12:
        raise AssertionError("transition generated negative cash")
    result_state.cash = max(result_state.cash, 0.0)
    for position in result_state.positions.values():
        if position.units <= 0:
            raise AssertionError("transition generated non-positive units")

    end_nav = result_state.cash + sum(
        position.units * position.last_mark
        for position in result_state.positions.values()
    )
    if abs(end_nav - (pre_trade_nav - total_cost)) > 1e-10:
        raise AssertionError("transition NAV accounting does not reconcile")
    result_state.previous_nav = end_nav

    diagnostics: dict[str, float | int] = {
        "pre_trade_nav": pre_trade_nav,
        "end_nav": end_nav,
        "total_cost": total_cost,
        "buy_scale": buy_scale,
        "attempted_order_count": len(trades),
        "executed_order_count": int(trades["status"].eq("executed").sum()),
        "blocked_order_count": int(trades["status"].eq("blocked").sum()),
        "replacement_buy_count": 0,
    }
    return TransitionResult(
        state=result_state,
        trades=trades,
        diagnostics=diagnostics,
        pre_trade_nav=pre_trade_nav,
        total_cost=total_cost,
        end_nav=end_nav,
    )


_DAILY_NAV_COLUMNS = (
    "signal_date",
    "pre_trade_nav",
    "nav",
    "gross_return",
    "net_return",
    "cash",
    "cash_ratio",
    "holding_count",
    "desired_count",
    "buy_notional",
    "sell_notional",
    "total_cost",
    "gross_turnover",
    "one_way_turnover",
    "attempted_order_count",
    "executed_order_count",
    "blocked_order_count",
)
_POSITION_COLUMNS = ("date", "stock_code", "units", "last_mark", "market_value")
_DIAGNOSTIC_COLUMNS = (
    "signal_date",
    "execution_date",
    "pre_trade_nav",
    "end_nav",
    "total_cost",
    "buy_scale",
    "attempted_order_count",
    "executed_order_count",
    "blocked_order_count",
    "replacement_buy_count",
)
_ENDING_POSITION_COLUMNS = (
    "asset_type",
    "stock_code",
    "units",
    "last_mark",
    "market_value",
)
_STRATEGY_OUTPUT_NAMES = (
    "daily_nav.csv",
    "trades.parquet",
    "positions.parquet",
    "execution_diagnostics.csv",
    "ending_positions.csv",
    "metrics_summary.json",
)
_STRATEGY_SOURCE_NAMES = (
    "prediction",
    "prediction_manifest",
    "locked_test_schema",
    "market_panel",
    "trading_calendar",
    "frozen_models",
    "locked_test_conclusion",
    "locked_test_comparison",
)
_SUMMARY_FORMULAS = {
    "elapsed_trading_observations": "count(daily_nav rows excluding initial state)",
    "cumulative_return": "ending_nav / initial_nav - 1",
    "cagr": (
        "(ending_nav / initial_nav) ** "
        "(annualization_days / elapsed_trading_observations) - 1"
    ),
    "annualized_return": "cagr",
    "annualized_volatility": (
        "sample_std(net_return over execution rows) * sqrt(annualization_days)"
    ),
    "sharpe_ratio": (
        "mean(net_return over execution rows) / sample_std(net_return over execution "
        "rows) * sqrt(annualization_days); 0 when sample_std is 0"
    ),
    "max_drawdown": "max(1 - nav / running_max_nav) over all NAV rows",
    "win_rate": (
        "count(net_return > 0 over execution rows) / elapsed_trading_observations"
    ),
    "average_cash_ratio": "mean(cash / nav) over all NAV rows including initial state",
    "gross_turnover": "sum(daily gross_turnover over execution rows)",
    "one_way_turnover": "sum(daily gross_turnover / 2 over execution rows)",
    "average_gross_turnover": "mean(daily gross_turnover over execution rows)",
    "average_one_way_turnover": (
        "mean(daily gross_turnover / 2 over execution rows)"
    ),
    "average_turnover": "average_one_way_turnover",
    "annualized_gross_turnover": (
        "average_gross_turnover * annualization_days"
    ),
    "annualized_one_way_turnover": (
        "average_one_way_turnover * annualization_days"
    ),
    "annualized_turnover": "annualized_one_way_turnover",
    "total_cost": "sum(total_cost over execution rows)",
    "buy_attempt_count": "count(trade rows with side = buy)",
    "sell_attempt_count": "count(trade rows with side = sell)",
    "buy_count": "count(trade rows with side = buy and status = executed)",
    "sell_count": "count(trade rows with side = sell and status = executed)",
    "buy_fill_rate": "buy_count / buy_attempt_count; 0 when buy_attempt_count is 0",
    "sell_fill_rate": (
        "sell_count / sell_attempt_count; 0 when sell_attempt_count is 0"
    ),
    "blocked_sale_days": (
        "count(distinct execution_date with at least one blocked sell order)"
    ),
    "ending_nav": "nav on the final daily_nav row",
}
_STRATEGY_SETTINGS_CONTRACT = {
    "start": "2024-01-01",
    "end": "2025-12-31",
    "top_k": 100,
    "expected_cross_section_size": 1000,
    "min_listing_days": 120,
    "initial_nav": 1.0,
    "commission_rate": 0.0001,
    "slippage_rate": 0.0005,
    "sell_stamp_duty_rate": 0.0005,
    "buy_cost_rate": 0.0006,
    "sell_cost_rate": 0.0011,
    "limit_tolerance": 1e-8,
    "annualization_days": 252,
}
_STRATEGY_OBSERVATION_CONVENTIONS = {
    "annualization_factor": 252,
    "initial_nav_row": (
        "included in max_drawdown and average_cash_ratio; excluded "
        "from return, turnover, cost, and win-rate observations"
    ),
    "return_denominator": "prior execution's ending NAV",
    "turnover_denominator": "same execution's marked pre-trade NAV",
    "fill_rate_denominator": "attempted order rows on the same side",
    "blocked_sale_day_count": (
        "distinct execution dates with at least one blocked sell"
    ),
}
_STRATEGY_MANIFEST_NAMES = {
    "schema_version",
    "status",
    "purpose",
    "model_name",
    "settings",
    "rows",
    "formulas",
    "observation_conventions",
    "input_sha256",
    "output_sha256",
}
_STRATEGY_ROW_NAMES = {
    "daily_nav",
    "execution_dates",
    "trades",
    "positions",
    "ending_positions",
}
_STRATEGY_INTEGER_METRICS = {
    "elapsed_trading_observations",
    "buy_attempt_count",
    "sell_attempt_count",
    "buy_count",
    "sell_count",
    "blocked_sale_days",
}
_COMMON_STRATEGY_SOURCE_NAMES = tuple(
    name
    for name in _STRATEGY_SOURCE_NAMES
    if name not in {"prediction", "prediction_manifest"}
)


def _load_strategy_market_panel(
    path: Path,
    calendar: Any,
    settings: StrategySettings,
) -> pd.DataFrame:
    """Stream only execution rows on the exact official strategy calendar."""
    official = _calendar_dates(calendar)
    period_dates = official[(official >= settings.start) & (official <= settings.end)]
    if period_dates.empty:
        raise ValueError("strategy period has no official trading dates")
    required_dates = set(period_dates)
    seen_keys: set[tuple[pd.Timestamp, str]] = set()
    frames: list[pd.DataFrame] = []
    try:
        reader = pd.read_csv(
            Path(path),
            usecols=list(MARKET_EXECUTION_COLUMNS),
            dtype=MARKET_EXECUTION_DTYPES,
            chunksize=MARKET_READ_CHUNK_SIZE,
        )
        with reader:
            for chunk in reader:
                chunk["date"] = pd.to_datetime(
                    chunk["date"], errors="raise"
                ).dt.normalize()
                chunk = chunk.loc[chunk["date"].isin(required_dates)].copy()
                if chunk.empty:
                    continue
                chunk["stock_code"] = chunk["stock_code"].str.strip()
                if chunk[["date", "stock_code"]].isna().any().any() or chunk[
                    "stock_code"
                ].eq("").any():
                    raise ValueError("market panel keys must be present")
                keys = list(zip(chunk["date"], chunk["stock_code"], strict=True))
                if len(keys) != len(set(keys)) or any(key in seen_keys for key in keys):
                    raise ValueError(
                        "market panel contains duplicate date,stock_code keys across chunks"
                    )
                seen_keys.update(keys)
                frames.append(chunk)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError("strategy market panel is invalid") from error
    if not frames:
        return pd.DataFrame(columns=MARKET_EXECUTION_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _normalize_market_panel(market: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(market, pd.DataFrame):
        raise ValueError("market panel must be a DataFrame")
    missing = sorted({"date", "stock_code"}.difference(market.columns))
    if missing:
        raise ValueError(f"market panel is missing key columns: {missing}")
    result = market.copy()
    try:
        result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    except (TypeError, ValueError) as error:
        raise ValueError("market panel dates are invalid") from error
    result["stock_code"] = result["stock_code"].astype("string").str.strip()
    if result[["date", "stock_code"]].isna().any().any():
        raise ValueError("market panel keys must be present")
    if result["stock_code"].eq("").any():
        raise ValueError("market panel stock codes must be present")
    if result.duplicated(["date", "stock_code"]).any():
        raise ValueError("market panel contains duplicate date,stock_code keys")
    return result


def _position_rows(state: PortfolioState, date: pd.Timestamp) -> list[dict[str, Any]]:
    return [
        {
            "date": date,
            "stock_code": stock_code,
            "units": position.units,
            "last_mark": position.last_mark,
            "market_value": position.units * position.last_mark,
        }
        for stock_code, position in sorted(state.positions.items())
    ]


def _ending_position_frame(state: PortfolioState) -> pd.DataFrame:
    rows = [
        {
            "asset_type": "position",
            "stock_code": stock_code,
            "units": position.units,
            "last_mark": position.last_mark,
            "market_value": position.units * position.last_mark,
        }
        for stock_code, position in sorted(state.positions.items())
    ]
    rows.append(
        {
            "asset_type": "cash",
            "stock_code": "CASH",
            "units": 1.0,
            "last_mark": state.cash,
            "market_value": state.cash,
        }
    )
    return pd.DataFrame(rows, columns=_ENDING_POSITION_COLUMNS)


def simulate_strategy(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    calendar: Any,
    settings: StrategySettings,
) -> StrategyBundle:
    """Run the static strategy from each official signal to the next open."""
    official = _calendar_dates(calendar)
    period_dates = official[(official >= settings.start) & (official <= settings.end)]
    if period_dates.empty:
        raise ValueError("strategy period has no official trading dates")
    desired_by_date = select_daily_top(predictions, settings, official)
    if set(desired_by_date) != set(period_dates):
        raise ValueError("prediction dates must exactly match the strategy trading calendar")
    market_panel = _normalize_market_panel(market)
    market_by_date = {
        pd.Timestamp(date): group.drop(columns="date").reset_index(drop=True)
        for date, group in market_panel.groupby("date", sort=False)
    }

    first_date = pd.Timestamp(period_dates[0])
    state = PortfolioState(cash=settings.initial_nav, positions={}, previous_nav=settings.initial_nav)
    daily_rows: list[dict[str, Any]] = [
        {
            "date": first_date,
            "signal_date": pd.NaT,
            "pre_trade_nav": settings.initial_nav,
            "nav": settings.initial_nav,
            "gross_return": 0.0,
            "net_return": 0.0,
            "cash": settings.initial_nav,
            "cash_ratio": 1.0,
            "holding_count": 0,
            "desired_count": settings.top_k,
            "buy_notional": 0.0,
            "sell_notional": 0.0,
            "total_cost": 0.0,
            "gross_turnover": 0.0,
            "one_way_turnover": 0.0,
            "attempted_order_count": 0,
            "executed_order_count": 0,
            "blocked_order_count": 0,
        }
    ]
    trade_frames: list[pd.DataFrame] = []
    position_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for index in range(1, len(period_dates)):
        signal_date = pd.Timestamp(period_dates[index - 1])
        execution_date = pd.Timestamp(period_dates[index])
        prior_nav = state.previous_nav
        transition = transition_at_open(
            state,
            desired_by_date[signal_date],
            market_by_date.get(execution_date, pd.DataFrame(columns=["stock_code"])),
            signal_date,
            execution_date,
            settings,
        )
        state = transition.state
        trades = transition.trades.reset_index()
        if not trades.empty:
            trade_frames.append(trades)
        executed = trades["status"].eq("executed")
        buy_notional = float(
            trades.loc[executed & trades["side"].eq("buy"), "gross_notional"].sum()
        )
        sell_notional = float(
            trades.loc[executed & trades["side"].eq("sell"), "gross_notional"].sum()
        )
        gross_turnover = (
            (buy_notional + sell_notional) / transition.pre_trade_nav
            if transition.pre_trade_nav > 0
            else 0.0
        )
        daily_rows.append(
            {
                "date": execution_date,
                "signal_date": signal_date,
                "pre_trade_nav": transition.pre_trade_nav,
                "nav": transition.end_nav,
                "gross_return": (
                    transition.pre_trade_nav / prior_nav - 1.0 if prior_nav > 0 else 0.0
                ),
                "net_return": transition.end_nav / prior_nav - 1.0 if prior_nav > 0 else 0.0,
                "cash": state.cash,
                "cash_ratio": state.cash / transition.end_nav if transition.end_nav > 0 else 0.0,
                "holding_count": len(state.positions),
                "desired_count": len(desired_by_date[signal_date]),
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "total_cost": transition.total_cost,
                "gross_turnover": gross_turnover,
                "one_way_turnover": gross_turnover / 2.0,
                "attempted_order_count": transition.diagnostics["attempted_order_count"],
                "executed_order_count": transition.diagnostics["executed_order_count"],
                "blocked_order_count": transition.diagnostics["blocked_order_count"],
            }
        )
        position_rows.extend(_position_rows(state, execution_date))
        diagnostic_rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                **transition.diagnostics,
            }
        )

    daily_nav = pd.DataFrame(daily_rows).set_index("date")
    daily_nav.index = pd.DatetimeIndex(daily_nav.index, name="date")
    daily_nav = daily_nav.loc[:, _DAILY_NAV_COLUMNS]
    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(columns=_TRADE_COLUMNS)
    )
    positions = pd.DataFrame(position_rows, columns=_POSITION_COLUMNS)
    diagnostics = pd.DataFrame(diagnostic_rows, columns=_DIAGNOSTIC_COLUMNS)
    bundle = StrategyBundle(
        daily_nav=daily_nav,
        trades=trades,
        positions=positions,
        execution_diagnostics=diagnostics,
        ending_positions=_ending_position_frame(state),
        metrics_summary={},
    )
    bundle.metrics_summary = summarize_strategy(bundle, settings)
    return bundle


def summarize_strategy(
    bundle: StrategyBundle,
    settings: StrategySettings,
) -> dict[str, float | int]:
    """Compute the fixed net-NAV, risk, execution, and turnover summary."""
    if bundle.daily_nav.empty:
        raise ValueError("strategy bundle has no daily NAV rows")
    observations = bundle.daily_nav.iloc[1:]
    returns = observations["net_return"].astype("float64")
    elapsed = len(returns)
    ending_nav = float(bundle.daily_nav["nav"].iloc[-1])
    cumulative_return = ending_nav / settings.initial_nav - 1.0
    if elapsed == 0:
        cagr = 0.0
    elif ending_nav <= 0:
        cagr = -1.0
    else:
        cagr = (ending_nav / settings.initial_nav) ** (
            settings.annualization_days / elapsed
        ) - 1.0
    daily_volatility = float(returns.std(ddof=1)) if elapsed > 1 else 0.0
    if not np.isfinite(daily_volatility):
        daily_volatility = 0.0
    annualized_volatility = daily_volatility * np.sqrt(settings.annualization_days)
    sharpe_ratio = (
        float(returns.mean()) / daily_volatility * np.sqrt(settings.annualization_days)
        if daily_volatility > 0
        else 0.0
    )
    nav = bundle.daily_nav["nav"].astype("float64")
    max_drawdown = float(-(nav / nav.cummax() - 1.0).min())
    average_cash_ratio = float(bundle.daily_nav["cash_ratio"].mean())
    average_gross_turnover = (
        float(observations["gross_turnover"].mean()) if elapsed else 0.0
    )
    average_one_way_turnover = (
        float(observations["one_way_turnover"].mean()) if elapsed else 0.0
    )

    trades = bundle.trades
    buy_attempts = int(trades["side"].eq("buy").sum()) if not trades.empty else 0
    sell_attempts = int(trades["side"].eq("sell").sum()) if not trades.empty else 0
    executed = trades["status"].eq("executed") if not trades.empty else pd.Series(dtype=bool)
    executed_buys = int((executed & trades["side"].eq("buy")).sum()) if not trades.empty else 0
    executed_sells = int((executed & trades["side"].eq("sell")).sum()) if not trades.empty else 0
    blocked_sales = (
        trades["side"].eq("sell") & trades["status"].eq("blocked")
        if not trades.empty
        else pd.Series(dtype=bool)
    )
    blocked_sale_days = (
        int(trades.loc[blocked_sales, "execution_date"].nunique()) if not trades.empty else 0
    )
    total_gross_turnover = float(observations["gross_turnover"].sum())
    total_one_way_turnover = float(observations["one_way_turnover"].sum())
    win_rate = float(returns.gt(0).mean()) if elapsed else 0.0
    return {
        "elapsed_trading_observations": elapsed,
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "annualized_return": cagr,
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "average_cash_ratio": average_cash_ratio,
        "gross_turnover": total_gross_turnover,
        "one_way_turnover": total_one_way_turnover,
        "average_gross_turnover": average_gross_turnover,
        "average_one_way_turnover": average_one_way_turnover,
        "average_turnover": average_one_way_turnover,
        "annualized_gross_turnover": average_gross_turnover * settings.annualization_days,
        "annualized_one_way_turnover": average_one_way_turnover * settings.annualization_days,
        "annualized_turnover": average_one_way_turnover * settings.annualization_days,
        "total_cost": float(observations["total_cost"].sum()),
        "buy_attempt_count": buy_attempts,
        "sell_attempt_count": sell_attempts,
        "buy_count": executed_buys,
        "sell_count": executed_sells,
        "buy_fill_rate": executed_buys / buy_attempts if buy_attempts else 0.0,
        "sell_fill_rate": executed_sells / sell_attempts if sell_attempts else 0.0,
        "blocked_sale_days": blocked_sale_days,
        "ending_nav": ending_nav,
    }


def _acquire_strategy_lock(path: Path) -> BinaryIO:
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
        raise FileExistsError(f"another strategy publication holds: {path}") from error
    return handle


def _release_strategy_lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _settings_manifest(settings: StrategySettings) -> dict[str, Any]:
    return {
        "start": settings.start.date().isoformat(),
        "end": settings.end.date().isoformat(),
        "top_k": settings.top_k,
        "expected_cross_section_size": settings.expected_cross_section_size,
        "min_listing_days": settings.min_listing_days,
        "initial_nav": settings.initial_nav,
        "commission_rate": settings.commission_rate,
        "slippage_rate": settings.slippage_rate,
        "sell_stamp_duty_rate": settings.sell_stamp_duty_rate,
        "buy_cost_rate": settings.buy_cost_rate,
        "sell_cost_rate": settings.sell_cost_rate,
        "limit_tolerance": settings.limit_tolerance,
        "annualization_days": settings.annualization_days,
    }


def _write_json_file(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _strategy_source_hashes(
    source_paths: Mapping[str, Path], model_name: str
) -> dict[str, str]:
    if not isinstance(source_paths, Mapping) or set(source_paths) != set(
        _STRATEGY_SOURCE_NAMES
    ):
        raise ValueError("strategy publication requires the exact source artifact set")
    try:
        paths = {name: Path(source_paths[name]) for name in _STRATEGY_SOURCE_NAMES}
    except (TypeError, ValueError) as error:
        raise ValueError("strategy source artifact is not a file") from error
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"strategy source artifact is not a file: {name}")
    if model_name not in LOCKED_MODEL_NAMES:
        raise ValueError(f"strategy model is not frozen: {model_name}")

    hashes = {name: file_sha256(path) for name, path in paths.items()}
    try:
        conclusion = json.loads(
            paths["locked_test_conclusion"].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("locked-test conclusion is invalid") from error
    if not isinstance(conclusion, Mapping):
        raise ValueError("locked-test conclusion must be an object")
    if conclusion.get("schema_version") != 1:
        raise ValueError("locked-test conclusion schema version must be 1")
    if conclusion.get("period") != {
        "start": STRATEGY_START.strftime("%Y-%m-%d"),
        "end": STRATEGY_END.strftime("%Y-%m-%d"),
    }:
        raise ValueError("locked-test conclusion period is invalid")
    if conclusion.get("selection_policy") != "no_test_based_selection":
        raise ValueError("locked-test conclusion selection policy is invalid")
    if conclusion.get("retuning_allowed") is not False:
        raise ValueError("locked-test conclusion must prohibit retuning")
    if tuple(conclusion.get("strategy_models", ())) != LOCKED_MODEL_NAMES:
        raise ValueError("locked-test conclusion must contain exactly the five frozen models")

    (
        expected_frozen_hash,
        expected_comparison_hash,
        expected_schema_hash,
        prediction_hashes,
        prediction_manifest_hashes,
    ) = _locked_test_artifact_hashes(conclusion)
    if hashes["locked_test_schema"] != expected_schema_hash:
        raise ValueError("locked-test conclusion locked-test schema hash does not match")
    if hashes["prediction_manifest"] != prediction_manifest_hashes[model_name]:
        raise ValueError(
            "locked-test conclusion locked-test prediction manifest hash does not match"
        )
    if hashes["prediction"] != prediction_hashes[model_name]:
        raise ValueError("locked-test conclusion prediction hash does not match")
    if hashes["frozen_models"] != expected_frozen_hash:
        raise ValueError("locked-test conclusion frozen spec hash does not match")
    if hashes["locked_test_comparison"] != expected_comparison_hash:
        raise ValueError("locked-test conclusion comparison hash does not match")

    try:
        prediction_manifest = json.loads(
            paths["prediction_manifest"].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("locked-test prediction manifest is invalid") from error
    if not isinstance(prediction_manifest, Mapping) or any(
        (
            prediction_manifest.get("status") != "completed",
            prediction_manifest.get("purpose")
            != "locked_test_static_inference_2024_2025",
            prediction_manifest.get("run_id") != model_name,
            prediction_manifest.get("model_name") != model_name,
            prediction_manifest.get("uses_training") is not False,
            prediction_manifest.get("uses_refit") is not False,
            prediction_manifest.get("uses_validation") is not False,
            prediction_manifest.get("uses_early_stopping") is not False,
            prediction_manifest.get("uses_entry_tradeable") is not False,
        )
    ):
        raise ValueError("locked-test prediction manifest is invalid")
    if hashes["prediction"] != prediction_manifest.get("predictions_10d_sha256"):
        raise ValueError("locked-test prediction hash does not match its manifest")
    if hashes["locked_test_schema"] != prediction_manifest.get("test_schema_sha256"):
        raise ValueError("locked-test schema hash does not match prediction manifest")
    if hashes["frozen_models"] != prediction_manifest.get("frozen_spec_sha256"):
        raise ValueError("frozen spec hash does not match prediction manifest")

    try:
        locked_test_schema = json.loads(
            paths["locked_test_schema"].read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("locked-test schema is invalid") from error
    if not isinstance(locked_test_schema, Mapping):
        raise ValueError("locked-test schema must be an object")
    if locked_test_schema.get("schema_version") != 1:
        raise ValueError("locked-test schema version must be 1")
    if locked_test_schema.get("purpose") != "locked_test_2024_2025":
        raise ValueError("locked-test schema purpose is invalid")
    if locked_test_schema.get("uses_entry_tradeable") is not False:
        raise ValueError("locked-test schema entry-tradeable contract is invalid")
    source_hashes = locked_test_schema.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("locked-test schema source hashes are invalid")
    if hashes["market_panel"] != source_hashes.get("market_panel"):
        raise ValueError("locked-test schema market panel hash does not match")
    if hashes["trading_calendar"] != source_hashes.get("trading_calendar"):
        raise ValueError("locked-test schema trading calendar hash does not match")
    return hashes


def write_strategy_run(
    bundle: StrategyBundle,
    destination: Path,
    settings: StrategySettings,
    *,
    model_name: str,
    source_paths: Mapping[str, Path],
) -> Path:
    """Atomically publish one immutable strategy report directory."""
    destination = Path(destination)
    input_hashes = _strategy_source_hashes(source_paths, model_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = _acquire_strategy_lock(destination.parent / ".strategy.lock")
    staging: Path | None = None
    try:
        if destination.exists():
            raise FileExistsError(f"strategy run already exists: {destination}")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        bundle.daily_nav.reset_index().to_csv(staging / "daily_nav.csv", index=False)
        bundle.trades.to_parquet(staging / "trades.parquet", index=False)
        bundle.positions.to_parquet(staging / "positions.parquet", index=False)
        bundle.execution_diagnostics.to_csv(
            staging / "execution_diagnostics.csv", index=False
        )
        bundle.ending_positions.to_csv(staging / "ending_positions.csv", index=False)
        _write_json_file(staging / "metrics_summary.json", bundle.metrics_summary)
        output_hashes = {
            name: file_sha256(staging / name) for name in _STRATEGY_OUTPUT_NAMES
        }
        if set(_SUMMARY_FORMULAS) != set(bundle.metrics_summary):
            raise ValueError("strategy summary formula contract is incomplete")
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "purpose": "frozen_rank_model_static_strategy_2024_2025",
            "model_name": model_name,
            "settings": _settings_manifest(settings),
            "rows": {
                "daily_nav": len(bundle.daily_nav),
                "execution_dates": len(bundle.execution_diagnostics),
                "trades": len(bundle.trades),
                "positions": len(bundle.positions),
                "ending_positions": len(bundle.ending_positions),
            },
            "formulas": _SUMMARY_FORMULAS,
            "observation_conventions": _STRATEGY_OBSERVATION_CONVENTIONS,
            "input_sha256": input_hashes,
            "output_sha256": output_hashes,
        }
        _write_json_file(staging / "manifest.json", manifest)
        os.replace(staging, destination)
        staging = None
    finally:
        try:
            if staging is not None and staging.exists():
                try:
                    shutil.rmtree(staging)
                except Exception:
                    # Cleanup is best-effort; preserve the publication error.
                    pass
        finally:
            _release_strategy_lock(lock)
    return destination


def _read_strategy_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object: {path}")
    return value


def backtest_locked_strategy(
    model_name: str,
    *,
    destination: Path,
    settings: StrategySettings,
    source_paths: Mapping[str, Path],
) -> Path:
    """Load sealed inputs, simulate one frozen model, and publish its run."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"strategy run already exists: {destination}")
    _strategy_source_hashes(source_paths, model_name)
    paths = {name: Path(source_paths[name]) for name in _STRATEGY_SOURCE_NAMES}

    frozen_spec = _read_strategy_json(paths["frozen_models"], "frozen specification")
    candidates = frozen_spec.get("candidates")
    candidate_names = (
        tuple(
            candidate.get("model_name")
            for candidate in candidates
            if isinstance(candidate, Mapping)
        )
        if isinstance(candidates, list)
        else ()
    )
    if (
        frozen_spec.get("schema_version") != 1
        or frozen_spec.get("candidate_count") != len(LOCKED_MODEL_NAMES)
        or candidate_names != LOCKED_MODEL_NAMES
    ):
        raise ValueError("frozen specification must contain exactly the five models")

    try:
        locked_comparison = pd.read_csv(paths["locked_test_comparison"])
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError("locked-test comparison is invalid") from error
    if (
        "model_name" not in locked_comparison
        or tuple(locked_comparison["model_name"].tolist()) != LOCKED_MODEL_NAMES
        or not {"winner", "accept", "champion"}.isdisjoint(
            locked_comparison.columns
        )
    ):
        raise ValueError(
            "locked-test comparison must contain five models and no decision fields"
        )

    try:
        predictions = pd.read_parquet(paths["prediction"])
    except (OSError, ValueError) as error:
        raise ValueError("locked-test prediction file is invalid") from error
    try:
        trading_calendar = pd.read_csv(
            paths["trading_calendar"],
            usecols=["date"],
            dtype={"date": "string"},
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError("strategy trading calendar is invalid") from error
    market_panel = _load_strategy_market_panel(
        paths["market_panel"], trading_calendar, settings
    )

    bundle = simulate_strategy(
        predictions,
        market_panel,
        trading_calendar,
        settings,
    )
    return write_strategy_run(
        bundle,
        destination,
        settings,
        model_name=model_name,
        source_paths=paths,
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_completed_strategy_manifest(
    manifest: Mapping[str, Any], model_name: str, run_directory: Path
) -> dict[str, str]:
    if set(manifest) != _STRATEGY_MANIFEST_NAMES:
        raise ValueError(f"strategy run manifest schema is invalid: {run_directory}")
    if any(
        (
            manifest.get("schema_version") != 1,
            manifest.get("status") != "completed",
            manifest.get("purpose")
            != "frozen_rank_model_static_strategy_2024_2025",
            manifest.get("model_name") != model_name,
        )
    ):
        raise ValueError(f"strategy run is not completed: {run_directory}")
    if manifest.get("settings") != _STRATEGY_SETTINGS_CONTRACT:
        raise ValueError(f"strategy run settings contract is invalid: {run_directory}")
    if manifest.get("formulas") != _SUMMARY_FORMULAS:
        raise ValueError(f"strategy run formula contract is invalid: {run_directory}")
    if manifest.get("observation_conventions") != _STRATEGY_OBSERVATION_CONVENTIONS:
        raise ValueError(
            f"strategy run observation contract is invalid: {run_directory}"
        )

    rows = manifest.get("rows")
    if (
        not isinstance(rows, Mapping)
        or set(rows) != _STRATEGY_ROW_NAMES
        or any(
            isinstance(rows[name], bool)
            or not isinstance(rows[name], int)
            or rows[name] < 0
            for name in _STRATEGY_ROW_NAMES
        )
        or rows["daily_nav"] != 485
        or rows["execution_dates"] != 484
        or rows["daily_nav"] != rows["execution_dates"] + 1
        or rows["ending_positions"] < 1
    ):
        raise ValueError(f"strategy run row counts are invalid: {run_directory}")

    input_hashes = manifest.get("input_sha256")
    if (
        not isinstance(input_hashes, Mapping)
        or set(input_hashes) != set(_STRATEGY_SOURCE_NAMES)
        or any(not _is_sha256(input_hashes[name]) for name in _STRATEGY_SOURCE_NAMES)
    ):
        raise ValueError(f"strategy run input hashes are invalid: {run_directory}")
    output_hashes = manifest.get("output_sha256")
    if (
        not isinstance(output_hashes, Mapping)
        or set(output_hashes) != set(_STRATEGY_OUTPUT_NAMES)
        or any(not _is_sha256(output_hashes[name]) for name in _STRATEGY_OUTPUT_NAMES)
    ):
        raise ValueError(f"strategy run output hashes are invalid: {run_directory}")
    return {name: input_hashes[name] for name in _STRATEGY_SOURCE_NAMES}


def _strategy_artifact_row_counts(run_directory: Path) -> dict[str, int]:
    try:
        import pyarrow.parquet as pq

        return {
            "daily_nav": len(pd.read_csv(run_directory / "daily_nav.csv")),
            "execution_dates": len(
                pd.read_csv(run_directory / "execution_diagnostics.csv")
            ),
            "trades": pq.ParquetFile(
                run_directory / "trades.parquet"
            ).metadata.num_rows,
            "positions": pq.ParquetFile(
                run_directory / "positions.parquet"
            ).metadata.num_rows,
            "ending_positions": len(
                pd.read_csv(run_directory / "ending_positions.csv")
            ),
        }
    except Exception as error:
        raise ValueError(
            f"strategy run output row counts are unreadable: {run_directory}"
        ) from error


def compare_strategy_runs(
    run_root: Path,
    output_path: Path,
    model_names: Sequence[str] = LOCKED_MODEL_NAMES,
) -> pd.DataFrame:
    """Publish descriptive metrics for all five completed frozen-model runs."""
    names = tuple(model_names)
    if names != LOCKED_MODEL_NAMES:
        raise ValueError("strategy comparison requires exactly the five frozen models")
    root = Path(run_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    destination = Path(output_path)
    rows: list[dict[str, Any]] = []
    common_source_hashes: dict[str, str] | None = None
    for model_name in names:
        run_directory = root / model_name
        if not run_directory.is_dir():
            raise FileNotFoundError(run_directory)
        manifest = _read_strategy_json(run_directory / "manifest.json", "manifest")
        input_hashes = _validate_completed_strategy_manifest(
            manifest, model_name, run_directory
        )
        current_common_hashes = {
            name: input_hashes[name] for name in _COMMON_STRATEGY_SOURCE_NAMES
        }
        if common_source_hashes is None:
            common_source_hashes = current_common_hashes
        elif current_common_hashes != common_source_hashes:
            raise ValueError("strategy runs have different common source hashes")
        hashes = manifest["output_sha256"]
        for name in _STRATEGY_OUTPUT_NAMES:
            artifact_path = run_directory / name
            if not artifact_path.is_file():
                raise FileNotFoundError(artifact_path)
            if file_sha256(artifact_path) != hashes[name]:
                raise ValueError(
                    f"strategy output hash mismatch for {name}: {run_directory}"
                )
        if _strategy_artifact_row_counts(run_directory) != manifest["rows"]:
            raise ValueError(
                f"strategy run output row counts do not match: {run_directory}"
            )
        summary_path = run_directory / "metrics_summary.json"
        summary = _read_strategy_json(summary_path, "strategy metrics summary")
        if set(summary) != set(_SUMMARY_FORMULAS):
            raise ValueError(f"strategy metric keys are invalid: {run_directory}")
        if any(
            isinstance(summary[name], bool)
            or not isinstance(summary[name], (int, float))
            or not np.isfinite(summary[name])
            for name in _SUMMARY_FORMULAS
        ):
            raise ValueError(f"strategy metric values are invalid: {run_directory}")
        if any(
            isinstance(summary[name], bool) or not isinstance(summary[name], int)
            for name in _STRATEGY_INTEGER_METRICS
        ):
            raise ValueError(f"strategy metric counts are invalid: {run_directory}")
        if summary["elapsed_trading_observations"] != manifest["rows"][
            "execution_dates"
        ]:
            raise ValueError(
                f"strategy metric row count is invalid: {run_directory}"
            )
        row = {
            "model_name": model_name,
            **{key: summary[key] for key in _SUMMARY_FORMULAS},
        }
        rows.append(row)

    lock = _acquire_strategy_lock(root / ".strategy-comparison.lock")
    temporary: Path | None = None
    try:
        if destination.exists():
            raise FileExistsError(f"strategy comparison is immutable: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        frame = pd.DataFrame(rows)
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
        temporary = None
        return frame
    finally:
        try:
            if temporary is not None and temporary.exists():
                try:
                    temporary.unlink()
                except Exception:
                    pass
        finally:
            _release_strategy_lock(lock)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"locked-test conclusion has invalid {label} hash")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"locked-test conclusion has invalid {label} hash") from error
    return value


def _locked_test_artifact_hashes(conclusion: Mapping[str, Any]):
    artifact_hashes = conclusion.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping):
        raise ValueError("locked-test conclusion is missing artifact hashes")
    expected_frozen_hash = _require_sha256(
        artifact_hashes.get("frozen_models"), "frozen spec"
    )
    expected_comparison_hash = _require_sha256(
        artifact_hashes.get("locked_test_comparison"), "comparison"
    )
    expected_schema_hash = _require_sha256(
        artifact_hashes.get("locked_test_schema"), "locked-test schema"
    )
    prediction_hashes = artifact_hashes.get("predictions")
    if not isinstance(prediction_hashes, Mapping) or set(prediction_hashes) != set(
        LOCKED_MODEL_NAMES
    ):
        raise ValueError(
            "locked-test conclusion must contain prediction hashes for five models"
        )
    prediction_manifest_hashes = artifact_hashes.get("prediction_manifests")
    if not isinstance(prediction_manifest_hashes, Mapping) or set(
        prediction_manifest_hashes
    ) != set(LOCKED_MODEL_NAMES):
        raise ValueError(
            "locked-test conclusion must contain prediction manifest hashes for five models"
        )
    checked_prediction_hashes = {
        model_name: _require_sha256(
            prediction_hashes[model_name], f"{model_name} prediction"
        )
        for model_name in LOCKED_MODEL_NAMES
    }
    checked_manifest_hashes = {
        model_name: _require_sha256(
            prediction_manifest_hashes[model_name],
            f"{model_name} prediction manifest",
        )
        for model_name in LOCKED_MODEL_NAMES
    }
    return (
        expected_frozen_hash,
        expected_comparison_hash,
        expected_schema_hash,
        checked_prediction_hashes,
        checked_manifest_hashes,
    )


def validate_locked_test_conclusion(
    *,
    conclusion: Mapping[str, Any],
    frozen_spec_path: Path,
    comparison_path: Path,
    prediction_paths: Mapping[str, Path],
    locked_test_schema_path: Path,
    prediction_manifest_paths: Mapping[str, Path],
) -> None:
    """Verify that the sealed conclusion still binds every strategy input."""
    if conclusion.get("schema_version") != 1:
        raise ValueError("locked-test conclusion schema version must be 1")
    if conclusion.get("period") != {
        "start": STRATEGY_START.strftime("%Y-%m-%d"),
        "end": STRATEGY_END.strftime("%Y-%m-%d"),
    }:
        raise ValueError("locked-test conclusion period must be 2024-01-01 through 2025-12-31")
    if conclusion.get("selection_policy") != "no_test_based_selection":
        raise ValueError("locked-test conclusion selection policy is invalid")
    if conclusion.get("retuning_allowed") is not False:
        raise ValueError("locked-test conclusion must prohibit retuning")
    if tuple(conclusion.get("strategy_models", ())) != LOCKED_MODEL_NAMES:
        raise ValueError("locked-test conclusion must contain exactly the five frozen models")
    if any(field in conclusion for field in ("winner", "recommendation")):
        raise ValueError("locked-test conclusion must not select a winner or recommendation")

    metrics = conclusion.get("locked_test_metrics")
    if not isinstance(metrics, list) or len(metrics) != len(LOCKED_MODEL_NAMES):
        raise ValueError("locked-test conclusion must contain five test metric rows")
    if tuple(row.get("model_name") for row in metrics if isinstance(row, Mapping)) != LOCKED_MODEL_NAMES:
        raise ValueError("locked-test conclusion metric rows must cover frozen models")

    (
        expected_frozen_hash,
        expected_comparison_hash,
        expected_schema_hash,
        prediction_hashes,
        prediction_manifest_hashes,
    ) = _locked_test_artifact_hashes(conclusion)

    if file_sha256(Path(frozen_spec_path)) != expected_frozen_hash:
        raise ValueError("locked-test conclusion frozen spec hash does not match")
    if file_sha256(Path(comparison_path)) != expected_comparison_hash:
        raise ValueError("locked-test conclusion comparison hash does not match")
    if file_sha256(Path(locked_test_schema_path)) != expected_schema_hash:
        raise ValueError("locked-test conclusion locked-test schema hash does not match")
    if set(prediction_paths) != set(LOCKED_MODEL_NAMES):
        raise ValueError("locked-test conclusion requires prediction paths for five models")
    if set(prediction_manifest_paths) != set(LOCKED_MODEL_NAMES):
        raise ValueError(
            "locked-test conclusion requires prediction manifest paths for five models"
        )
    for model_name in LOCKED_MODEL_NAMES:
        if file_sha256(Path(prediction_paths[model_name])) != prediction_hashes[
            model_name
        ]:
            raise ValueError(
                f"locked-test conclusion prediction hash does not match: {model_name}"
            )
        if file_sha256(
            Path(prediction_manifest_paths[model_name])
        ) != prediction_manifest_hashes[model_name]:
            raise ValueError(
                "locked-test conclusion prediction manifest hash does not match: "
                f"{model_name}"
            )
