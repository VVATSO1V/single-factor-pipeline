"""Immutable configuration and input seals for static strategy execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"locked-test conclusion has invalid {label} hash")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"locked-test conclusion has invalid {label} hash") from error
    return value


def validate_locked_test_conclusion(
    *,
    conclusion: Mapping[str, Any],
    frozen_spec_path: Path,
    comparison_path: Path,
    prediction_paths: Mapping[str, Path],
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

    artifact_hashes = conclusion.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping):
        raise ValueError("locked-test conclusion is missing artifact hashes")
    expected_frozen_hash = _require_sha256(artifact_hashes.get("frozen_models"), "frozen spec")
    expected_comparison_hash = _require_sha256(
        artifact_hashes.get("locked_test_comparison"), "comparison"
    )
    prediction_hashes = artifact_hashes.get("predictions")
    if not isinstance(prediction_hashes, Mapping) or set(prediction_hashes) != set(
        LOCKED_MODEL_NAMES
    ):
        raise ValueError("locked-test conclusion must contain prediction hashes for five models")

    if file_sha256(Path(frozen_spec_path)) != expected_frozen_hash:
        raise ValueError("locked-test conclusion frozen spec hash does not match")
    if file_sha256(Path(comparison_path)) != expected_comparison_hash:
        raise ValueError("locked-test conclusion comparison hash does not match")
    if set(prediction_paths) != set(LOCKED_MODEL_NAMES):
        raise ValueError("locked-test conclusion requires prediction paths for five models")
    for model_name in LOCKED_MODEL_NAMES:
        expected_prediction_hash = _require_sha256(
            prediction_hashes[model_name], f"{model_name} prediction"
        )
        if file_sha256(Path(prediction_paths[model_name])) != expected_prediction_hash:
            raise ValueError(f"locked-test conclusion prediction hash does not match: {model_name}")
