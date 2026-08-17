"""Schedule and simulate staggered ten-day strategy paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from rank_model.stages.strategy import (
    StrategyBundle,
    StrategySettings,
    PortfolioState,
    _DAILY_NAV_COLUMNS,
    _DIAGNOSTIC_COLUMNS,
    _ENDING_POSITION_COLUMNS,
    _POSITION_COLUMNS,
    _TRADE_COLUMNS,
    _calendar_dates,
    _ending_position_frame,
    _normalize_market_panel,
    _position_rows,
    select_daily_top,
    summarize_strategy,
    transition_at_open,
)


STAGGERED_HORIZON = 10
STAGGERED_OFFSET_COUNT = 10
STAGGERED_COMPLETE_SHIFT = 11
_FIRST_ELIGIBLE_SIGNAL = pd.Timestamp("2024-01-02")
_LAST_ELIGIBLE_SIGNAL = pd.Timestamp("2025-12-16")
_ELIGIBLE_SIGNAL_COUNT = 474


@dataclass(frozen=True)
class OffsetSchedule:
    offset: int
    signal_dates: Sequence[pd.Timestamp]
    execution_dates: Sequence[pd.Timestamp]
    final_horizon_date: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "signal_dates",
            tuple(pd.Timestamp(date).normalize() for date in self.signal_dates),
        )
        object.__setattr__(
            self,
            "execution_dates",
            tuple(pd.Timestamp(date).normalize() for date in self.execution_dates),
        )
        object.__setattr__(
            self,
            "final_horizon_date",
            pd.Timestamp(self.final_horizon_date).normalize(),
        )

    @property
    def first_execution_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.execution_dates[0])


@dataclass
class StaggeredPathBundle:
    offset: int
    schedule: OffsetSchedule
    strategy: StrategyBundle


def _validate_offset_schedules(
    schedules: Sequence[OffsetSchedule], period: pd.DatetimeIndex
) -> None:
    if len(schedules) != STAGGERED_OFFSET_COUNT:
        raise ValueError("staggered schedules must contain exactly 10 offsets")

    eligible = tuple(pd.Timestamp(date) for date in period[:-STAGGERED_COMPLETE_SHIFT])
    assigned = tuple(
        pd.Timestamp(date) for schedule in schedules for date in schedule.signal_dates
    )
    if len(assigned) != len(set(assigned)) or set(assigned) != set(eligible):
        raise ValueError("staggered schedules must partition every eligible signal date")

    expected_offsets = tuple(range(1, STAGGERED_OFFSET_COUNT + 1))
    if tuple(schedule.offset for schedule in schedules) != expected_offsets:
        raise ValueError("staggered schedule offsets must be ordered from 1 through 10")

    date_positions = {pd.Timestamp(date): index for index, date in enumerate(period)}
    for schedule in schedules:
        expected_signals = tuple(
            eligible[index]
            for index in range(schedule.offset - 1, len(eligible), STAGGERED_OFFSET_COUNT)
        )
        signals = tuple(pd.Timestamp(date) for date in schedule.signal_dates)
        executions = tuple(pd.Timestamp(date) for date in schedule.execution_dates)
        if signals != expected_signals:
            raise ValueError("staggered schedule signal dates do not match their offset")
        if len(executions) != len(signals):
            raise ValueError("staggered schedule execution dates do not match signal dates")
        expected_executions = tuple(period[date_positions[date] + 1] for date in signals)
        if executions != expected_executions:
            raise ValueError("staggered schedule execution dates must be T+1")
        expected_horizon = period[date_positions[signals[-1]] + STAGGERED_COMPLETE_SHIFT]
        if pd.Timestamp(schedule.final_horizon_date) != expected_horizon:
            raise ValueError("staggered schedule final horizon must be T+11")


def build_offset_schedules(
    calendar: Any,
    settings: StrategySettings,
) -> Sequence[OffsetSchedule]:
    """Partition the sealed complete signal period into ten daily offsets."""
    official = _calendar_dates(calendar)
    period = official[(official >= settings.start) & (official <= settings.end)]
    if len(period) < STAGGERED_COMPLETE_SHIFT + 1:
        raise ValueError("strategy period requires at least 12 official trading dates")

    eligible = period[:-STAGGERED_COMPLETE_SHIFT]
    if len(eligible) != _ELIGIBLE_SIGNAL_COUNT:
        raise ValueError("staggered strategy requires exactly 474 eligible signal dates")
    if eligible[0] != _FIRST_ELIGIBLE_SIGNAL or eligible[-1] != _LAST_ELIGIBLE_SIGNAL:
        raise ValueError("staggered strategy eligible dates must span 2024-01-02 through 2025-12-16")

    schedules: list[OffsetSchedule] = []
    for offset in range(1, STAGGERED_OFFSET_COUNT + 1):
        indices = list(range(offset - 1, len(eligible), STAGGERED_OFFSET_COUNT))
        schedules.append(
            OffsetSchedule(
                offset=offset,
                signal_dates=tuple(eligible[index] for index in indices),
                execution_dates=tuple(period[index + 1] for index in indices),
                final_horizon_date=period[indices[-1] + STAGGERED_COMPLETE_SHIFT],
            )
        )
    result = tuple(schedules)
    _validate_offset_schedules(result, period)
    return result


_STAGGERED_DAILY_COLUMNS = (
    "offset",
    "is_rebalance",
    "active_signal_date",
    *_DAILY_NAV_COLUMNS,
)
_STAGGERED_DIAGNOSTIC_COLUMNS = (
    "offset",
    "is_rebalance",
    "active_signal_date",
    *_DIAGNOSTIC_COLUMNS,
)


def _validate_path_schedule(
    schedule: OffsetSchedule, period: pd.DatetimeIndex
) -> None:
    if not isinstance(schedule, OffsetSchedule):
        raise ValueError("staggered path requires an offset schedule")
    if (
        isinstance(schedule.offset, bool)
        or not isinstance(schedule.offset, int)
        or not 1 <= schedule.offset <= STAGGERED_OFFSET_COUNT
    ):
        raise ValueError("staggered path offset must be from 1 through 10")

    signals = tuple(pd.Timestamp(date) for date in schedule.signal_dates)
    executions = tuple(pd.Timestamp(date) for date in schedule.execution_dates)
    if not signals or len(signals) != len(executions):
        raise ValueError("staggered path requires matching non-empty schedule dates")
    if len(signals) != len(set(signals)) or len(executions) != len(set(executions)):
        raise ValueError("staggered path schedule dates must be unique")

    positions = {pd.Timestamp(date): index for index, date in enumerate(period)}
    if any(date not in positions for date in (*signals, *executions)):
        raise ValueError("staggered path schedule dates must be official period dates")
    signal_positions = [positions[date] for date in signals]
    execution_positions = [positions[date] for date in executions]
    if signal_positions != sorted(signal_positions):
        raise ValueError("staggered path signal dates must be ordered")
    if any(execution != signal + 1 for signal, execution in zip(signal_positions, execution_positions)):
        raise ValueError("staggered path execution dates must be T+1")

    final_horizon_date = pd.Timestamp(schedule.final_horizon_date)
    expected_horizon_position = signal_positions[-1] + STAGGERED_COMPLETE_SHIFT
    if (
        final_horizon_date not in positions
        or positions[final_horizon_date] != expected_horizon_position
    ):
        raise ValueError("staggered path final horizon date must be T+11")


def _notional_totals(trades: pd.DataFrame) -> tuple[float, float]:
    if trades.empty:
        return 0.0, 0.0
    executed = trades["status"].eq("executed")
    buy_notional = float(
        trades.loc[executed & trades["side"].eq("buy"), "gross_notional"].sum()
    )
    sell_notional = float(
        trades.loc[executed & trades["side"].eq("sell"), "gross_notional"].sum()
    )
    return buy_notional, sell_notional


def simulate_staggered_offset(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    calendar: Any,
    settings: StrategySettings,
    schedule: OffsetSchedule,
) -> StaggeredPathBundle:
    """Value one offset daily and update its desired set only on schedule."""
    official = _calendar_dates(calendar)
    period = official[(official >= settings.start) & (official <= settings.end)]
    if period.empty:
        raise ValueError("strategy period has no official trading dates")
    _validate_path_schedule(schedule, period)

    desired_by_date = select_daily_top(predictions, settings, official)
    if set(desired_by_date) != set(period):
        raise ValueError("prediction dates must exactly match the strategy trading calendar")
    market_panel = _normalize_market_panel(market)
    market_by_date = {
        pd.Timestamp(date): group.drop(columns="date").reset_index(drop=True)
        for date, group in market_panel.groupby("date", sort=False)
    }

    final_horizon_date = pd.Timestamp(schedule.final_horizon_date)
    path_dates = period[period <= final_horizon_date]
    signal_by_execution = {
        pd.Timestamp(execution): pd.Timestamp(signal)
        for signal, execution in zip(schedule.signal_dates, schedule.execution_dates)
    }
    first_date = pd.Timestamp(path_dates[0])
    state = PortfolioState(
        cash=settings.initial_nav,
        positions={},
        previous_nav=settings.initial_nav,
    )
    active_signal_date = pd.NaT
    current_desired: tuple[str, ...] = ()
    daily_rows: list[dict[str, Any]] = [
        {
            "date": first_date,
            "offset": schedule.offset,
            "is_rebalance": False,
            "active_signal_date": active_signal_date,
            "signal_date": active_signal_date,
            "pre_trade_nav": settings.initial_nav,
            "nav": settings.initial_nav,
            "gross_return": 0.0,
            "net_return": 0.0,
            "cash": settings.initial_nav,
            "cash_ratio": 1.0,
            "holding_count": 0,
            "desired_count": 0,
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

    for date in path_dates[1:]:
        execution_date = pd.Timestamp(date)
        is_rebalance = execution_date in signal_by_execution
        if is_rebalance:
            active_signal_date = signal_by_execution[execution_date]
            current_desired = desired_by_date[active_signal_date]
        prior_nav = state.previous_nav
        transition_signal_date = (
            active_signal_date if not pd.isna(active_signal_date) else first_date
        )
        transition = transition_at_open(
            state,
            current_desired,
            market_by_date.get(execution_date, pd.DataFrame(columns=["stock_code"])),
            transition_signal_date,
            execution_date,
            settings,
            allow_buys=is_rebalance,
        )
        state = transition.state
        trades = transition.trades.reset_index()
        if not trades.empty:
            trade_frames.append(trades)
        buy_notional, sell_notional = _notional_totals(trades)
        gross_turnover = (
            (buy_notional + sell_notional) / transition.pre_trade_nav
            if transition.pre_trade_nav > 0
            else 0.0
        )
        daily_rows.append(
            {
                "date": execution_date,
                "offset": schedule.offset,
                "is_rebalance": is_rebalance,
                "active_signal_date": active_signal_date,
                "signal_date": active_signal_date,
                "pre_trade_nav": transition.pre_trade_nav,
                "nav": transition.end_nav,
                "gross_return": (
                    transition.pre_trade_nav / prior_nav - 1.0 if prior_nav > 0 else 0.0
                ),
                "net_return": transition.end_nav / prior_nav - 1.0 if prior_nav > 0 else 0.0,
                "cash": state.cash,
                "cash_ratio": state.cash / transition.end_nav if transition.end_nav > 0 else 0.0,
                "holding_count": len(state.positions),
                "desired_count": len(current_desired),
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
                "offset": schedule.offset,
                "is_rebalance": is_rebalance,
                "active_signal_date": active_signal_date,
                "signal_date": active_signal_date,
                "execution_date": execution_date,
                **transition.diagnostics,
            }
        )

    daily_nav = pd.DataFrame(daily_rows).set_index("date")
    daily_nav.index = pd.DatetimeIndex(daily_nav.index, name="date")
    daily_nav = daily_nav.loc[:, _STAGGERED_DAILY_COLUMNS]
    strategy = StrategyBundle(
        daily_nav=daily_nav,
        trades=(
            pd.concat(trade_frames, ignore_index=True)
            if trade_frames
            else pd.DataFrame(columns=_TRADE_COLUMNS)
        ),
        positions=pd.DataFrame(position_rows, columns=_POSITION_COLUMNS),
        execution_diagnostics=pd.DataFrame(
            diagnostic_rows, columns=_STAGGERED_DIAGNOSTIC_COLUMNS
        ),
        ending_positions=pd.DataFrame(
            _ending_position_frame(state), columns=_ENDING_POSITION_COLUMNS
        ),
        metrics_summary={},
    )
    strategy.metrics_summary = summarize_strategy(strategy, settings)
    return StaggeredPathBundle(
        offset=schedule.offset,
        schedule=schedule,
        strategy=strategy,
    )
