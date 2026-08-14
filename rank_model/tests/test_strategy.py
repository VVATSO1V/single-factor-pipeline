"""Temporary contract tests for the frozen strategy stage."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

from rank_model.stages.strategy import (
    load_strategy_settings,
    validate_locked_test_conclusion,
)


VALID_CONFIG = {
    "strategy": {
        "start": "2024-01-01",
        "end": "2025-12-31",
        "top_k": 100,
        "expected_cross_section_size": 1000,
        "min_listing_days": 120,
        "initial_nav": 1.0,
        "commission_bps": 1.0,
        "slippage_bps": 5.0,
        "sell_stamp_duty_bps": 5.0,
        "limit_tolerance": 1e-8,
        "annualization_days": 252,
    }
}
MODEL_NAMES = (
    "ridge_rank_regression",
    "xgboost_rank_regression",
    "lightgbm_rank_regression",
    "lightgbm_lambdarank",
    "mlp_top100_hybrid_rank",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StrategyContractTests(unittest.TestCase):
    def test_fixed_settings_are_loaded_exactly(self):
        settings = load_strategy_settings(VALID_CONFIG)

        self.assertEqual(settings.top_k, 100)
        self.assertEqual(settings.min_listing_days, 120)
        self.assertEqual(settings.buy_cost_rate, 0.0006)
        self.assertEqual(settings.sell_cost_rate, 0.0011)

    def test_conclusion_accepts_valid_sealed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            frozen_spec = directory / "frozen_models.json"
            comparison = directory / "locked_test_comparison.csv"
            prediction = directory / "predictions_10d.parquet"
            frozen_spec.write_text('{"candidate_count": 5}', encoding="utf-8")
            comparison.write_text("model_name\\n", encoding="utf-8")
            prediction.write_bytes(b"original prediction bytes")
            conclusion = {
                "schema_version": 1,
                "period": {"start": "2024-01-01", "end": "2025-12-31"},
                "selection_policy": "no_test_based_selection",
                "retuning_allowed": False,
                "strategy_models": list(MODEL_NAMES),
                "artifact_sha256": {
                    "frozen_models": sha256(frozen_spec),
                    "locked_test_comparison": sha256(comparison),
                    "predictions": {
                        model_name: sha256(prediction) for model_name in MODEL_NAMES
                    },
                },
                "locked_test_metrics": [
                    {"model_name": model_name} for model_name in MODEL_NAMES
                ],
            }

            result = validate_locked_test_conclusion(
                conclusion=conclusion,
                frozen_spec_path=frozen_spec,
                comparison_path=comparison,
                prediction_paths={model_name: prediction for model_name in MODEL_NAMES},
            )

        self.assertIsNone(result)

    def test_conclusion_rejects_changed_prediction_hash(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            frozen_spec = directory / "frozen_models.json"
            comparison = directory / "locked_test_comparison.csv"
            prediction = directory / "predictions_10d.parquet"
            changed_prediction = directory / "changed_predictions_10d.parquet"
            frozen_spec.write_text('{"candidate_count": 5}', encoding="utf-8")
            comparison.write_text("model_name\\n", encoding="utf-8")
            prediction.write_bytes(b"original prediction bytes")
            changed_prediction.write_bytes(b"changed prediction bytes")
            conclusion = {
                "schema_version": 1,
                "period": {"start": "2024-01-01", "end": "2025-12-31"},
                "selection_policy": "no_test_based_selection",
                "retuning_allowed": False,
                "strategy_models": list(MODEL_NAMES),
                "artifact_sha256": {
                    "frozen_models": sha256(frozen_spec),
                    "locked_test_comparison": sha256(comparison),
                    "predictions": {
                        model_name: sha256(prediction) for model_name in MODEL_NAMES
                    },
                },
                "locked_test_metrics": [
                    {"model_name": model_name} for model_name in MODEL_NAMES
                ],
            }

            with self.assertRaisesRegex(ValueError, "prediction hash"):
                validate_locked_test_conclusion(
                    conclusion=conclusion,
                    frozen_spec_path=frozen_spec,
                    comparison_path=comparison,
                    prediction_paths={
                        model_name: (
                            changed_prediction
                            if model_name == "ridge_rank_regression"
                            else prediction
                        )
                        for model_name in MODEL_NAMES
                    },
                )


SETTINGS = load_strategy_settings(VALID_CONFIG)
OFFICIAL_CALENDAR = pd.to_datetime(["2024-01-02"])


def prediction_frame(*, dates=("2024-01-02",), rows_per_date=1000):
    rows = []
    for date in dates:
        for index in range(rows_per_date):
            rows.append(
                {
                    "date": date,
                    "stock_code": f"S{index:04d}",
                    "split": "test",
                    "horizon": 10,
                    "score_raw": float(index),
                }
            )
    return pd.DataFrame(rows)


def market_row(
    *,
    is_st=False,
    is_suspended=False,
    raw_open=10.0,
    post_open=10.0,
    limit_up=11.0,
    limit_down=9.0,
    listing_days=120.0,
    has_price_record=True,
):
    return pd.Series(
        {
            "has_price_record": has_price_record,
            "is_suspended": is_suspended,
            "is_st": is_st,
            "listing_days": listing_days,
            "raw_open": raw_open,
            "post_open": post_open,
            "limit_up": limit_up,
            "limit_down": limit_down,
        }
    )


def market_slice(**rows):
    return pd.DataFrame.from_dict(
        {stock_code: row.to_dict() for stock_code, row in rows.items()},
        orient="index",
    )


class SignalAndRuleTests(unittest.TestCase):
    def test_top100_uses_score_then_stock_code(self):
        predictions = prediction_frame(rows_per_date=1000)
        predictions.loc[:2, "score_raw"] = 9999.0
        predictions.loc[:2, "stock_code"] = ["C", "A", "B"]

        from rank_model.stages.strategy import select_daily_top

        selected = select_daily_top(predictions, SETTINGS, OFFICIAL_CALENDAR)

        self.assertEqual(selected[pd.Timestamp("2024-01-02")][:3], ("A", "B", "C"))

    def test_buy_and_sell_rules_are_asymmetric(self):
        from rank_model.stages.strategy import buy_decision, sell_decision

        st_row = market_row(is_st=True)

        self.assertEqual(buy_decision(st_row, SETTINGS), (False, "st"))
        self.assertEqual(sell_decision(st_row, SETTINGS), (True, "eligible"))

    def test_limit_down_blocks_both_sides(self):
        from rank_model.stages.strategy import buy_decision, sell_decision

        row = market_row(raw_open=9.0, limit_down=9.0)

        self.assertEqual(buy_decision(row, SETTINGS)[1], "limit_down")
        self.assertEqual(sell_decision(row, SETTINGS)[1], "limit_down")

    def test_selection_rejects_invalid_prediction_contract(self):
        from rank_model.stages.strategy import select_daily_top

        duplicate = prediction_frame()
        duplicate.loc[1, "stock_code"] = duplicate.loc[0, "stock_code"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_daily_top(duplicate, SETTINGS, OFFICIAL_CALENDAR)

        non_finite = prediction_frame()
        non_finite.loc[0, "score_raw"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            select_daily_top(non_finite, SETTINGS, OFFICIAL_CALENDAR)

        wrong_size = prediction_frame(rows_per_date=999)
        with self.assertRaisesRegex(ValueError, "cross-section"):
            select_daily_top(wrong_size, SETTINGS, OFFICIAL_CALENDAR)

    def test_selection_rejects_non_test_predictions_and_off_calendar_dates(self):
        from rank_model.stages.strategy import select_daily_top, validate_prediction_calendar

        wrong_split = prediction_frame()
        wrong_split.loc[0, "split"] = "validation"
        with self.assertRaisesRegex(ValueError, "test split"):
            select_daily_top(wrong_split, SETTINGS, OFFICIAL_CALENDAR)

        wrong_date = prediction_frame(dates=("2024-01-03",))
        calendar = pd.to_datetime(["2024-01-02"])
        with self.assertRaisesRegex(ValueError, "calendar"):
            select_daily_top(wrong_date, SETTINGS, calendar=calendar)

        valid = prediction_frame()
        validate_prediction_calendar(valid, calendar, SETTINGS)

        wrong_horizon = prediction_frame()
        wrong_horizon.loc[0, "horizon"] = 5
        with self.assertRaisesRegex(ValueError, "horizon"):
            validate_prediction_calendar(wrong_horizon, calendar, SETTINGS)

    def test_selection_requires_official_calendar_membership(self):
        from rank_model.stages.strategy import select_daily_top

        with self.assertRaisesRegex(ValueError, "official trading calendar"):
            select_daily_top(prediction_frame(), SETTINGS, None)

        weekend = prediction_frame(dates=("2024-01-06",))
        with self.assertRaisesRegex(ValueError, "calendar"):
            select_daily_top(weekend, SETTINGS, OFFICIAL_CALENDAR)

    def test_sell_ignores_listing_age_and_limit_up(self):
        from rank_model.stages.strategy import sell_decision

        self.assertEqual(
            sell_decision(market_row(listing_days=1.0), SETTINGS),
            (True, "eligible"),
        )
        self.assertEqual(
            sell_decision(market_row(raw_open=11.0, limit_up=11.0), SETTINGS),
            (True, "eligible"),
        )

    def test_execution_requires_complete_data(self):
        from rank_model.stages.strategy import buy_decision, sell_decision

        missing = market_row().drop("post_open")
        self.assertEqual(buy_decision(missing, SETTINGS), (False, "post_open"))
        self.assertEqual(sell_decision(missing, SETTINGS), (False, "post_open"))

        suspended = market_row(is_suspended=True)
        self.assertEqual(buy_decision(suspended, SETTINGS), (False, "suspended"))
        self.assertEqual(sell_decision(suspended, SETTINGS), (False, "suspended"))

    def test_missing_and_malformed_suspension_status_blocks_both_sides(self):
        from rank_model.stages.strategy import buy_decision, sell_decision

        for row in (
            market_row().drop("is_suspended"),
            market_row(is_suspended="false"),
        ):
            self.assertEqual(buy_decision(row, SETTINGS), (False, "suspension_status"))
            self.assertEqual(sell_decision(row, SETTINGS), (False, "suspension_status"))

    def test_missing_and_malformed_st_status_blocks_buys_only(self):
        from rank_model.stages.strategy import buy_decision, sell_decision

        for row in (market_row().drop("is_st"), market_row(is_st="false")):
            self.assertEqual(buy_decision(row, SETTINGS), (False, "st_status"))
            self.assertEqual(sell_decision(row, SETTINGS), (True, "eligible"))


class TransitionTests(unittest.TestCase):
    signal_date = pd.Timestamp("2024-01-02")
    execution_date = pd.Timestamp("2024-01-03")

    def test_intersection_units_never_change(self):
        from rank_model.stages.strategy import (
            PortfolioState,
            Position,
            transition_at_open,
        )

        state = PortfolioState(
            cash=0.0,
            positions={
                "A": Position(units=0.1, last_mark=10.0),
                "B": Position(units=0.2, last_mark=10.0),
            },
            previous_nav=3.0,
        )
        market = market_slice(
            A=market_row(),
            B=market_row(post_open=15.0),
            C=market_row(),
        )

        result = transition_at_open(
            state, ("B", "C"), market, self.signal_date, self.execution_date, SETTINGS
        )

        self.assertEqual(result.state.positions["B"].units, state.positions["B"].units)
        self.assertEqual(result.state.positions["B"].last_mark, 15.0)
        self.assertNotIn("B", result.trades.index)

    def test_missing_and_malformed_suspension_status_carry_existing_marks(self):
        from rank_model.stages.strategy import (
            PortfolioState,
            Position,
            transition_at_open,
        )

        for row in (
            market_row(post_open=20.0).drop("is_suspended"),
            market_row(is_suspended="false", post_open=20.0),
        ):
            state = PortfolioState(
                cash=0.0,
                positions={"A": Position(units=0.1, last_mark=10.0)},
                previous_nav=1.0,
            )
            result = transition_at_open(
                state,
                ("A",),
                market_slice(A=row),
                self.signal_date,
                self.execution_date,
                SETTINGS,
            )

            self.assertEqual(result.state.positions["A"].last_mark, 10.0)

    def test_blocked_sale_is_carried_and_buy_is_not_replaced(self):
        from rank_model.stages.strategy import (
            PortfolioState,
            Position,
            transition_at_open,
        )

        state = PortfolioState(
            cash=0.0,
            positions={"A": Position(units=0.1, last_mark=10.0)},
            previous_nav=1.0,
        )
        market = market_slice(
            A=market_row(is_suspended=True),
            B=market_row(is_suspended=True),
        )

        result = transition_at_open(
            state, ("B",), market, self.signal_date, self.execution_date, SETTINGS
        )

        self.assertIn("A", result.state.positions)
        self.assertEqual(result.state.positions["A"].last_mark, 10.0)
        self.assertNotIn("B", result.state.positions)
        self.assertEqual(result.diagnostics["replacement_buy_count"], 0)
        self.assertEqual(result.trades.loc["A", "reason"], "suspended")
        self.assertEqual(result.trades.loc["B", "reason"], "suspended")
        self.assertEqual(result.trades.loc["A", "status"], "blocked")
        self.assertEqual(result.trades.loc["B", "status"], "blocked")

    def test_cash_shortage_scales_all_eligible_buys_equally(self):
        from rank_model.stages.strategy import (
            PortfolioState,
            Position,
            transition_at_open,
        )

        state = PortfolioState(
            cash=0.0006,
            positions={"C": Position(units=0.09994, last_mark=10.0)},
            previous_nav=1.0,
        )
        market = market_slice(
            A=market_row(),
            B=market_row(),
            C=market_row(is_suspended=True),
        )

        result = transition_at_open(
            state, ("A", "B"), market, self.signal_date, self.execution_date, SETTINGS
        )

        self.assertAlmostEqual(
            result.trades.loc["A", "gross_notional"],
            result.trades.loc["B", "gross_notional"],
        )
        self.assertLess(result.diagnostics["buy_scale"], 1.0)
        self.assertEqual(result.trades.loc["C", "status"], "blocked")

    def test_marking_and_all_execution_costs_reconcile_nav(self):
        from rank_model.stages.strategy import (
            PortfolioState,
            Position,
            transition_at_open,
        )

        state = PortfolioState(
            cash=0.0,
            positions={"A": Position(units=0.1, last_mark=10.0)},
            previous_nav=1.0,
        )
        market = market_slice(
            A=market_row(raw_open=10.0, post_open=20.0),
            B=market_row(raw_open=10.0, post_open=10.0),
        )

        result = transition_at_open(
            state, ("B",), market, self.signal_date, self.execution_date, SETTINGS
        )

        self.assertAlmostEqual(result.pre_trade_nav, 2.0)
        self.assertAlmostEqual(result.trades.loc["A", "cost"], 0.0022)
        self.assertAlmostEqual(result.trades.loc["B", "cost"], 0.000012)
        self.assertAlmostEqual(result.total_cost, 0.002212)
        self.assertAlmostEqual(result.end_nav, result.pre_trade_nav - result.total_cost)
        self.assertAlmostEqual(result.state.previous_nav, result.end_nav)

    def test_rejects_nonpositive_positions_and_duplicate_order_keys(self):
        from rank_model.stages.strategy import (
            PortfolioState,
            Position,
            transition_at_open,
        )

        market = market_slice(A=market_row())
        invalid_state = PortfolioState(
            cash=1.0,
            positions={"A": Position(units=0.0, last_mark=10.0)},
            previous_nav=1.0,
        )
        with self.assertRaisesRegex(ValueError, "non-positive units"):
            transition_at_open(
                invalid_state,
                (),
                market,
                self.signal_date,
                self.execution_date,
                SETTINGS,
            )

        valid_state = PortfolioState(cash=1.0, positions={}, previous_nav=1.0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            transition_at_open(
                valid_state,
                ("A", "A"),
                market,
                self.signal_date,
                self.execution_date,
                SETTINGS,
            )


def strategy_fixture(dates):
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    stock_codes = np.array([f"S{index:04d}" for index in range(1000)])
    predictions = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), len(stock_codes)),
            "stock_code": np.tile(stock_codes, len(dates)),
            "split": "test",
            "horizon": 10,
            "score_raw": np.tile(np.arange(1000, dtype="float64"), len(dates)),
        }
    )
    selected_codes = stock_codes[-SETTINGS.top_k :]
    execution_dates = dates[1:]
    market = pd.DataFrame(
        {
            "date": np.repeat(execution_dates.to_numpy(), len(selected_codes)),
            "stock_code": np.tile(selected_codes, len(execution_dates)),
            "has_price_record": True,
            "is_suspended": False,
            "is_st": False,
            "listing_days": 500.0,
            "raw_open": 10.0,
            "post_open": 10.0,
            "limit_up": 11.0,
            "limit_down": 9.0,
        }
    )
    return predictions, market, dates


def strategy_source_paths(directory: Path, model_name=MODEL_NAMES[0]):
    paths = {
        "prediction": directory / "predictions_10d.parquet",
        "market_panel": directory / "market_panel.csv",
        "trading_calendar": directory / "trading_calendar.csv",
        "frozen_models": directory / "frozen_models.json",
        "locked_test_conclusion": directory / "locked_test_conclusion.json",
        "locked_test_comparison": directory / "locked_test_comparison.csv",
    }
    paths["prediction"].write_bytes(b"sealed prediction")
    paths["market_panel"].write_text("date,stock_code\n", encoding="utf-8")
    paths["trading_calendar"].write_text("date\n", encoding="utf-8")
    paths["frozen_models"].write_text('{"candidate_count": 5}\n', encoding="utf-8")
    paths["locked_test_comparison"].write_text("model_name\n", encoding="utf-8")
    conclusion = {
        "schema_version": 1,
        "period": {"start": "2024-01-01", "end": "2025-12-31"},
        "selection_policy": "no_test_based_selection",
        "retuning_allowed": False,
        "strategy_models": list(MODEL_NAMES),
        "artifact_sha256": {
            "frozen_models": sha256(paths["frozen_models"]),
            "locked_test_comparison": sha256(paths["locked_test_comparison"]),
            "predictions": {
                name: sha256(paths["prediction"]) if name == model_name else "0" * 64
                for name in MODEL_NAMES
            },
        },
        "locked_test_metrics": [{"model_name": name} for name in MODEL_NAMES],
    }
    paths["locked_test_conclusion"].write_text(
        json.dumps(conclusion), encoding="utf-8"
    )
    return paths


class PeriodAndReportingTests(unittest.TestCase):
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )

    def test_signal_executes_on_next_official_date(self):
        from rank_model.stages.strategy import simulate_strategy

        predictions, market, calendar = strategy_fixture(self.dates)

        bundle = simulate_strategy(predictions, market, calendar, SETTINGS)

        self.assertEqual(
            bundle.trades.iloc[0]["execution_date"], pd.Timestamp("2024-01-03")
        )
        self.assertEqual(
            bundle.trades.iloc[0]["signal_date"], pd.Timestamp("2024-01-02")
        )

    def test_full_period_has_initial_row_and_484_executions(self):
        from rank_model.stages.strategy import simulate_strategy

        business_dates = pd.bdate_range("2024-01-02", "2025-12-31")
        chosen = np.linspace(0, len(business_dates) - 1, 485, dtype=int)
        calendar = business_dates[chosen]
        predictions, market, calendar = strategy_fixture(calendar)

        bundle = simulate_strategy(predictions, market, calendar, SETTINGS)

        self.assertEqual(len(bundle.daily_nav), 485)
        self.assertEqual(len(bundle.execution_diagnostics), 484)
        self.assertEqual(bundle.daily_nav.index[0], pd.Timestamp("2024-01-02"))
        self.assertEqual(bundle.daily_nav.index[-1], pd.Timestamp("2025-12-31"))
        self.assertEqual(
            bundle.execution_diagnostics.iloc[-1]["signal_date"], calendar[-2]
        )
        self.assertEqual(
            bundle.execution_diagnostics.iloc[-1]["execution_date"], calendar[-1]
        )

    def test_missing_suspended_mark_is_carried_until_reopen(self):
        from rank_model.stages.strategy import simulate_strategy

        predictions, market, calendar = strategy_fixture(self.dates)
        suspended_day = pd.Timestamp("2024-01-04")
        reopen_day = pd.Timestamp("2024-01-05")
        stock_code = "S0999"
        suspended = market["date"].eq(suspended_day) & market["stock_code"].eq(
            stock_code
        )
        market.loc[suspended, "has_price_record"] = False
        market.loc[suspended, "is_suspended"] = True
        market.loc[suspended, ["raw_open", "post_open", "limit_up", "limit_down"]] = np.nan
        reopened = market["date"].eq(reopen_day) & market["stock_code"].eq(stock_code)
        market.loc[reopened, "post_open"] = 20.0

        bundle = simulate_strategy(predictions, market, calendar, SETTINGS)

        self.assertEqual(bundle.daily_nav.loc[suspended_day, "gross_return"], 0.0)
        self.assertAlmostEqual(bundle.daily_nav.loc[reopen_day, "gross_return"], 0.01)
        final_position = bundle.ending_positions.loc[
            bundle.ending_positions["stock_code"].eq(stock_code)
        ].iloc[0]
        self.assertEqual(final_position["last_mark"], 20.0)

    def test_target_columns_cannot_change_strategy_output(self):
        from rank_model.stages.strategy import simulate_strategy

        predictions, market, calendar = strategy_fixture(self.dates)
        first = simulate_strategy(predictions, market, calendar, SETTINGS)
        changed = predictions.assign(target_10d=999.0, rank_target_10d=-999.0)
        second = simulate_strategy(changed, market, calendar, SETTINGS)

        pd.testing.assert_frame_equal(first.daily_nav, second.daily_nav)
        pd.testing.assert_frame_equal(first.trades, second.trades)
        pd.testing.assert_frame_equal(first.positions, second.positions)
        pd.testing.assert_frame_equal(
            first.execution_diagnostics, second.execution_diagnostics
        )
        pd.testing.assert_frame_equal(first.ending_positions, second.ending_positions)
        self.assertEqual(first.metrics_summary, second.metrics_summary)

    def test_summary_uses_elapsed_returns_and_exact_report_columns(self):
        from rank_model.stages.strategy import simulate_strategy, summarize_strategy

        predictions, market, calendar = strategy_fixture(self.dates)
        market.loc[market["date"].eq(self.dates[2]), "post_open"] = 11.0
        market.loc[market["date"].eq(self.dates[3]), "post_open"] = 9.0
        bundle = simulate_strategy(predictions, market, calendar, SETTINGS)

        summary = summarize_strategy(bundle, SETTINGS)
        initial_buy_nav = 1.0 / 1.0006
        expected_returns = np.array(
            [initial_buy_nav - 1.0, 0.1, 9.0 / 11.0 - 1.0], dtype="float64"
        )
        cumulative = 0.9 / 1.0006 - 1.0
        expected_cagr = (1.0 + cumulative) ** (SETTINGS.annualization_days / 3) - 1.0
        expected_daily_volatility = expected_returns.std(ddof=1)
        expected_volatility = expected_daily_volatility * np.sqrt(
            SETTINGS.annualization_days
        )
        first_day_gross_turnover = 1.0 / 1.0006

        np.testing.assert_allclose(
            bundle.daily_nav["net_return"].iloc[1:].to_numpy(), expected_returns
        )
        self.assertAlmostEqual(bundle.daily_nav.iloc[2]["gross_return"], 0.1)
        self.assertAlmostEqual(
            bundle.daily_nav.iloc[3]["gross_return"], 9.0 / 11.0 - 1.0
        )
        self.assertEqual(summary["elapsed_trading_observations"], 3)
        self.assertAlmostEqual(summary["cumulative_return"], cumulative)
        self.assertAlmostEqual(summary["cagr"], expected_cagr)
        self.assertAlmostEqual(summary["annualized_return"], expected_cagr)
        self.assertAlmostEqual(summary["annualized_volatility"], expected_volatility)
        self.assertAlmostEqual(
            summary["sharpe_ratio"],
            expected_returns.mean()
            / expected_daily_volatility
            * np.sqrt(SETTINGS.annualization_days),
        )
        self.assertAlmostEqual(summary["max_drawdown"], 2.0 / 11.0)
        self.assertAlmostEqual(summary["win_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["average_cash_ratio"], 0.25)
        self.assertAlmostEqual(summary["gross_turnover"], first_day_gross_turnover)
        self.assertAlmostEqual(
            summary["one_way_turnover"], first_day_gross_turnover / 2.0
        )
        self.assertAlmostEqual(
            summary["average_gross_turnover"], first_day_gross_turnover / 3.0
        )
        self.assertAlmostEqual(
            summary["average_one_way_turnover"], first_day_gross_turnover / 6.0
        )
        self.assertAlmostEqual(
            summary["annualized_gross_turnover"],
            first_day_gross_turnover / 3.0 * SETTINGS.annualization_days,
        )
        self.assertAlmostEqual(
            summary["annualized_one_way_turnover"],
            first_day_gross_turnover / 6.0 * SETTINGS.annualization_days,
        )
        self.assertAlmostEqual(
            summary["annualized_turnover"],
            first_day_gross_turnover / 6.0 * SETTINGS.annualization_days,
        )
        self.assertAlmostEqual(summary["total_cost"], 0.0006 / 1.0006)
        self.assertAlmostEqual(summary["ending_nav"], 0.9 / 1.0006)
        self.assertEqual(summary, bundle.metrics_summary)
        self.assertTrue(
            {
                "gross_return",
                "net_return",
                "cash_ratio",
                "gross_turnover",
                "one_way_turnover",
                "attempted_order_count",
                "executed_order_count",
                "blocked_order_count",
            }.issubset(bundle.daily_nav.columns)
        )

    def test_summary_reports_cost_fills_and_blocked_sale_days(self):
        from rank_model.stages.strategy import simulate_strategy

        predictions, market, calendar = strategy_fixture(self.dates)
        replacement_codes = [f"S{index:04d}" for index in range(800, 900)]
        replacement_signal = predictions["date"].eq(self.dates[1]) & predictions[
            "stock_code"
        ].isin(replacement_codes)
        predictions.loc[replacement_signal, "score_raw"] += 2000.0
        replacement_market = pd.DataFrame(
            {
                "date": np.repeat(self.dates[2:].to_numpy(), len(replacement_codes)),
                "stock_code": np.tile(replacement_codes, len(self.dates[2:])),
                "has_price_record": True,
                "is_suspended": False,
                "is_st": False,
                "listing_days": 500.0,
                "raw_open": 10.0,
                "post_open": 10.0,
                "limit_up": 11.0,
                "limit_down": 9.0,
            }
        )
        market = pd.concat([market, replacement_market], ignore_index=True)
        blocked_sale = market["date"].eq(self.dates[2]) & market["stock_code"].eq(
            "S0999"
        )
        market.loc[blocked_sale, "is_suspended"] = True

        bundle = simulate_strategy(predictions, market, calendar, SETTINGS)
        summary = bundle.metrics_summary

        self.assertEqual(summary["buy_attempt_count"], 299)
        self.assertEqual(summary["buy_count"], 299)
        self.assertEqual(summary["sell_attempt_count"], 200)
        self.assertEqual(summary["sell_count"], 199)
        self.assertEqual(summary["buy_fill_rate"], 1.0)
        self.assertEqual(summary["sell_fill_rate"], 199 / 200)
        self.assertEqual(summary["blocked_sale_days"], 1)
        self.assertAlmostEqual(summary["total_cost"], bundle.trades["cost"].sum())
        self.assertEqual(
            bundle.ending_positions["asset_type"].eq("position").sum(), 100
        )

    def test_publication_is_hashed_atomic_and_immutable(self):
        from rank_model.stages.strategy import (
            _acquire_strategy_lock,
            _release_strategy_lock,
            simulate_strategy,
            write_strategy_run,
        )

        predictions, market, calendar = strategy_fixture(self.dates)
        bundle = simulate_strategy(predictions, market, calendar, SETTINGS)

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            destination = directory / MODEL_NAMES[0]
            source_paths = strategy_source_paths(directory)
            missing = dict(source_paths)
            del missing["market_panel"]
            with self.assertRaisesRegex(ValueError, "exact source artifact set"):
                write_strategy_run(
                    bundle,
                    destination,
                    SETTINGS,
                    model_name=MODEL_NAMES[0],
                    source_paths=missing,
                )
            extra = {**source_paths, "fabricated": directory / "fabricated.txt"}
            extra["fabricated"].write_text("fabricated", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact source artifact set"):
                write_strategy_run(
                    bundle,
                    destination,
                    SETTINGS,
                    model_name=MODEL_NAMES[0],
                    source_paths=extra,
                )
            for invalid_path in (directory, None):
                with self.subTest(invalid_path=invalid_path):
                    non_file = {**source_paths, "market_panel": invalid_path}
                    with self.assertRaisesRegex(
                        ValueError, "source artifact is not a file"
                    ):
                        write_strategy_run(
                            bundle,
                            destination,
                            SETTINGS,
                            model_name=MODEL_NAMES[0],
                            source_paths=non_file,
                        )
            for source_name, expected_error in (
                ("prediction", "prediction hash"),
                ("frozen_models", "frozen spec hash"),
                ("locked_test_comparison", "comparison hash"),
            ):
                with self.subTest(changed_source=source_name):
                    source_paths = strategy_source_paths(directory)
                    source_paths[source_name].write_bytes(b"changed contents")
                    with self.assertRaisesRegex(ValueError, expected_error):
                        write_strategy_run(
                            bundle,
                            destination,
                            SETTINGS,
                            model_name=MODEL_NAMES[0],
                            source_paths=source_paths,
                        )
            self.assertFalse(destination.exists())
            source_paths = strategy_source_paths(directory)
            lock = _acquire_strategy_lock(destination.parent / ".strategy.lock")
            try:
                with self.assertRaisesRegex(FileExistsError, "publication"):
                    write_strategy_run(
                        bundle,
                        destination,
                        SETTINGS,
                        model_name=MODEL_NAMES[0],
                        source_paths=source_paths,
                    )
            finally:
                _release_strategy_lock(lock)
            result = write_strategy_run(
                bundle,
                destination,
                SETTINGS,
                model_name=MODEL_NAMES[0],
                source_paths=source_paths,
            )
            manifest = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(result, destination)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(set(manifest["formulas"]), set(bundle.metrics_summary))
            self.assertEqual(
                manifest["formulas"]["win_rate"],
                "count(net_return > 0 over execution rows) / elapsed_trading_observations",
            )
            self.assertEqual(
                manifest["formulas"]["buy_fill_rate"],
                "buy_count / buy_attempt_count; 0 when buy_attempt_count is 0",
            )
            self.assertEqual(
                manifest["formulas"]["blocked_sale_days"],
                "count(distinct execution_date with at least one blocked sell order)",
            )
            self.assertEqual(
                manifest["observation_conventions"],
                {
                    "annualization_factor": SETTINGS.annualization_days,
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
                },
            )
            self.assertEqual(
                set(manifest["output_sha256"]),
                {
                    "daily_nav.csv",
                    "trades.parquet",
                    "positions.parquet",
                    "execution_diagnostics.csv",
                    "ending_positions.csv",
                    "metrics_summary.json",
                },
            )
            for name, expected_hash in manifest["output_sha256"].items():
                self.assertEqual(sha256(destination / name), expected_hash)
            self.assertEqual(
                manifest["input_sha256"],
                {name: sha256(path) for name, path in source_paths.items()},
            )
            with self.assertRaises(FileExistsError):
                write_strategy_run(
                    bundle,
                    destination,
                    SETTINGS,
                    model_name=MODEL_NAMES[0],
                    source_paths=source_paths,
                )

    def test_publication_failures_leave_no_destination_and_release_lock(self):
        import rank_model.stages.strategy as strategy_stage

        predictions, market, calendar = strategy_fixture(self.dates)
        bundle = strategy_stage.simulate_strategy(
            predictions, market, calendar, SETTINGS
        )
        real_hash = strategy_stage.file_sha256
        real_json_write = strategy_stage._write_json_file

        for fault in ("artifact write", "artifact hash", "manifest write", "rename"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                destination = directory / MODEL_NAMES[0]
                source_paths = strategy_source_paths(directory)

                if fault == "artifact write":
                    patcher = mock.patch.object(
                        pd.DataFrame,
                        "to_parquet",
                        side_effect=RuntimeError(fault),
                    )
                elif fault == "artifact hash":
                    def fail_staged_hash(path):
                        if Path(path).parent != directory:
                            raise RuntimeError(fault)
                        return real_hash(Path(path))

                    patcher = mock.patch.object(
                        strategy_stage, "file_sha256", side_effect=fail_staged_hash
                    )
                elif fault == "manifest write":
                    def fail_manifest(path, value):
                        if Path(path).name == "manifest.json":
                            raise RuntimeError(fault)
                        return real_json_write(Path(path), value)

                    patcher = mock.patch.object(
                        strategy_stage, "_write_json_file", side_effect=fail_manifest
                    )
                else:
                    patcher = mock.patch.object(
                        strategy_stage.os,
                        "replace",
                        side_effect=RuntimeError(fault),
                    )

                with mock.patch.object(
                    strategy_stage,
                    "_release_strategy_lock",
                    wraps=strategy_stage._release_strategy_lock,
                ) as release_lock, patcher:
                    with self.assertRaisesRegex(RuntimeError, fault):
                        strategy_stage.write_strategy_run(
                            bundle,
                            destination,
                            SETTINGS,
                            model_name=MODEL_NAMES[0],
                            source_paths=source_paths,
                        )

                self.assertFalse(destination.exists())
                self.assertEqual(
                    [path for path in directory.iterdir() if path.name.startswith(".")],
                    [directory / ".strategy.lock"],
                )
                release_lock.assert_called_once()
                lock = strategy_stage._acquire_strategy_lock(
                    directory / ".strategy.lock"
                )
                strategy_stage._release_strategy_lock(lock)

    def test_cleanup_failure_preserves_primary_error_and_releases_lock(self):
        import rank_model.stages.strategy as strategy_stage

        predictions, market, calendar = strategy_fixture(self.dates)
        bundle = strategy_stage.simulate_strategy(
            predictions, market, calendar, SETTINGS
        )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            destination = directory / MODEL_NAMES[0]
            source_paths = strategy_source_paths(directory)
            with mock.patch.object(
                strategy_stage,
                "_release_strategy_lock",
                wraps=strategy_stage._release_strategy_lock,
            ) as release_lock:
                with mock.patch.object(
                    pd.DataFrame,
                    "to_parquet",
                    side_effect=RuntimeError("primary artifact write failure"),
                ), mock.patch.object(
                    strategy_stage.shutil,
                    "rmtree",
                    side_effect=RuntimeError("cleanup failure"),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "primary artifact write failure"
                    ):
                        strategy_stage.write_strategy_run(
                            bundle,
                            destination,
                            SETTINGS,
                            model_name=MODEL_NAMES[0],
                            source_paths=source_paths,
                        )

            self.assertFalse(destination.exists())
            release_lock.assert_called_once()
            lock = strategy_stage._acquire_strategy_lock(
                directory / ".strategy.lock"
            )
            strategy_stage._release_strategy_lock(lock)
