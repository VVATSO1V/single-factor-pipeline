"""Temporary synthetic contracts for the staggered strategy implementation."""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import rank_model.stages.strategy as strategy
import rank_model.stages.staggered_strategy as staggered
from rank_model.stages.dataset import file_sha256
from rank_model.stages.strategy import (
    LOCKED_MODEL_NAMES,
    LockedStrategyInputs,
    PortfolioState,
    Position,
    StrategyBundle,
    StrategySettings,
    select_daily_top,
    simulate_strategy,
    transition_at_open,
)
from rank_model.stages.staggered_strategy import (
    AVERAGE_METRICS,
    OFFSET_STATISTICS,
    OffsetSchedule,
    StaggeredPathBundle,
    _validate_staggered_publication,
    _validate_offset_schedules,
    backtest_staggered_strategy,
    build_average_nav,
    build_offset_metrics,
    build_offset_summary,
    build_offset_schedules,
    simulate_staggered_offset,
    summarize_average_nav,
)


def _settings(*, start: str = "2024-01-02", end: str = "2024-01-04") -> StrategySettings:
    return StrategySettings(
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        top_k=1,
        expected_cross_section_size=2,
        min_listing_days=120,
        initial_nav=1.0,
        commission_rate=0.0001,
        slippage_rate=0.0005,
        sell_stamp_duty_rate=0.0005,
        limit_tolerance=1e-8,
        annualization_days=252,
    )


def _market_row(stock_code: str, raw_open: float, post_open: float, **overrides: object) -> dict[str, object]:
    return {
        "stock_code": stock_code,
        "has_price_record": True,
        "is_suspended": False,
        "is_st": False,
        "listing_days": 200.0,
        "raw_open": raw_open,
        "post_open": post_open,
        "limit_up": raw_open * 1.2,
        "limit_down": raw_open * 0.8,
        **overrides,
    }


def _schedule_settings(start: pd.Timestamp, end: pd.Timestamp) -> StrategySettings:
    return StrategySettings(
        **{
            **_settings().__dict__,
            "start": start,
            "end": end,
        }
    )


def _sealed_synthetic_calendar() -> pd.DataFrame:
    signal_dates = pd.bdate_range("2024-01-02", periods=473).append(
        pd.DatetimeIndex([pd.Timestamp("2025-12-16")])
    )
    completion_dates = pd.bdate_range("2025-12-17", periods=11)
    return pd.DataFrame({"date": signal_dates.append(completion_dates)})


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_sealed_loader_inputs(root: Path, settings: StrategySettings) -> dict[str, Path]:
    model_name = LOCKED_MODEL_NAMES[0]
    calendar_path = root / "trading_calendar.csv"
    market_path = root / "market_panel.csv"
    prediction_path = root / "predictions_10d.parquet"
    prediction_manifest_path = root / "prediction_manifest.json"
    frozen_path = root / "frozen_models.json"
    comparison_path = root / "locked_test_comparison.csv"
    schema_path = root / "locked_test_schema.json"
    conclusion_path = root / "locked_test_conclusion.json"

    pd.DataFrame({"date": ["2024-01-02", "2024-01-03"]}).to_csv(
        calendar_path, index=False
    )
    pd.DataFrame(
        [
            {"date": "2024-01-02", **_market_row("A", 10.0, 10.0)},
            {"date": "2024-01-02", **_market_row("B", 10.0, 10.0)},
            {"date": "2024-01-03", **_market_row("A", 10.0, 10.0)},
            {"date": "2024-01-03", **_market_row("B", 10.0, 10.0)},
        ]
    ).to_csv(market_path, index=False)
    pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "stock_code": "A",
                "score_raw": 2.0,
                "split": "test",
                "horizon": 10,
                "target_10d": -99.0,
            },
            {
                "date": "2024-01-02",
                "stock_code": "B",
                "score_raw": 1.0,
                "split": "test",
                "horizon": 10,
                "target_10d": 99.0,
            },
            {
                "date": "2024-01-03",
                "stock_code": "A",
                "score_raw": 3.0,
                "split": "test",
                "horizon": 10,
                "target_10d": -88.0,
            },
            {
                "date": "2024-01-03",
                "stock_code": "B",
                "score_raw": 1.0,
                "split": "test",
                "horizon": 10,
                "target_10d": 88.0,
            },
        ]
    ).to_parquet(prediction_path, index=False)
    _write_json(
        frozen_path,
        {
            "schema_version": 1,
            "candidate_count": len(LOCKED_MODEL_NAMES),
            "candidates": [{"model_name": name} for name in LOCKED_MODEL_NAMES],
        },
    )
    pd.DataFrame({"model_name": list(LOCKED_MODEL_NAMES)}).to_csv(
        comparison_path, index=False
    )
    _write_json(
        schema_path,
        {
            "schema_version": 1,
            "purpose": "locked_test_2024_2025",
            "uses_entry_tradeable": False,
            "source_hashes": {
                "market_panel": file_sha256(market_path),
                "trading_calendar": file_sha256(calendar_path),
            },
        },
    )
    _write_json(
        prediction_manifest_path,
        {
            "status": "completed",
            "purpose": "locked_test_static_inference_2024_2025",
            "run_id": model_name,
            "model_name": model_name,
            "uses_training": False,
            "uses_refit": False,
            "uses_validation": False,
            "uses_early_stopping": False,
            "uses_entry_tradeable": False,
            "predictions_10d_sha256": file_sha256(prediction_path),
            "test_schema_sha256": file_sha256(schema_path),
            "frozen_spec_sha256": file_sha256(frozen_path),
        },
    )
    empty_hashes = {name: "0" * 64 for name in LOCKED_MODEL_NAMES}
    prediction_hashes = {**empty_hashes, model_name: file_sha256(prediction_path)}
    manifest_hashes = {
        **empty_hashes,
        model_name: file_sha256(prediction_manifest_path),
    }
    _write_json(
        conclusion_path,
        {
            "schema_version": 1,
            "period": {"start": "2024-01-01", "end": "2025-12-31"},
            "selection_policy": "no_test_based_selection",
            "retuning_allowed": False,
            "strategy_models": list(LOCKED_MODEL_NAMES),
            "artifact_sha256": {
                "frozen_models": file_sha256(frozen_path),
                "locked_test_comparison": file_sha256(comparison_path),
                "locked_test_schema": file_sha256(schema_path),
                "predictions": prediction_hashes,
                "prediction_manifests": manifest_hashes,
            },
        },
    )
    return {
        "prediction": prediction_path,
        "prediction_manifest": prediction_manifest_path,
        "locked_test_schema": schema_path,
        "market_panel": market_path,
        "trading_calendar": calendar_path,
        "frozen_models": frozen_path,
        "locked_test_conclusion": conclusion_path,
        "locked_test_comparison": comparison_path,
    }


def _aggregation_paths() -> tuple[StaggeredPathBundle, ...]:
    dates = pd.bdate_range("2024-01-02", periods=15)
    paths: list[StaggeredPathBundle] = []
    for offset in range(1, 11):
        path_dates = dates[: offset + 5]
        day = np.arange(len(path_dates), dtype="float64")
        wave = np.where(day % 3 == 0, -1.0, 1.0)
        nav = 1.0 + day * (0.002 + offset * 0.0002) + wave * day * 0.0005
        nav[0] = 1.0
        total_cost = day * offset * 0.000001
        pre_trade_nav = nav + total_cost
        cash = nav * (0.05 + offset * 0.01)
        buy_notional = day * (0.001 + offset * 0.0001)
        sell_notional = day * (0.0005 + offset * 0.00005)
        prior_nav = np.concatenate(([1.0], nav[:-1]))
        daily_nav = pd.DataFrame(
            {
                "signal_date": pd.NaT,
                "pre_trade_nav": pre_trade_nav,
                "nav": nav,
                "gross_return": pre_trade_nav / prior_nav - 1.0,
                "net_return": nav / prior_nav - 1.0,
                "cash": cash,
                "cash_ratio": cash / nav,
                "holding_count": offset,
                "desired_count": offset,
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "total_cost": total_cost,
                "gross_turnover": (buy_notional + sell_notional) / pre_trade_nav,
                "one_way_turnover": (buy_notional + sell_notional)
                / pre_trade_nav
                / 2.0,
                "attempted_order_count": offset,
                "executed_order_count": offset,
                "blocked_order_count": 0,
            },
            index=pd.DatetimeIndex(path_dates, name="date"),
        )
        bundle = StrategyBundle(
            daily_nav=daily_nav,
            trades=pd.DataFrame(columns=strategy._TRADE_COLUMNS),
            positions=pd.DataFrame(columns=strategy._POSITION_COLUMNS),
            execution_diagnostics=pd.DataFrame(columns=strategy._DIAGNOSTIC_COLUMNS),
            ending_positions=pd.DataFrame(columns=strategy._ENDING_POSITION_COLUMNS),
            metrics_summary={},
        )
        bundle.metrics_summary = strategy.summarize_strategy(
            bundle, _settings(end=str(path_dates[-1].date()))
        )
        paths.append(
            StaggeredPathBundle(
                offset=offset,
                schedule=OffsetSchedule(
                    offset=offset,
                    signal_dates=(dates[offset - 1],),
                    execution_dates=(dates[offset],),
                    final_horizon_date=path_dates[-1],
                ),
                strategy=bundle,
            )
        )
    return tuple(paths)


def _synthetic_locked_inputs(
    root: Path, paths: tuple[StaggeredPathBundle, ...]
) -> LockedStrategyInputs:
    source_paths: dict[str, Path] = {}
    for name in strategy._STRATEGY_SOURCE_NAMES:
        path = root / f"{name}.source"
        path.write_text(f"sealed {name}\n", encoding="utf-8")
        source_paths[name] = path
    return LockedStrategyInputs(
        predictions=pd.DataFrame(),
        market_panel=pd.DataFrame(),
        trading_calendar=pd.DataFrame(
            {"date": paths[-1].strategy.daily_nav.index.astype(str)}
        ),
        source_paths=source_paths,
        source_hashes={name: file_sha256(path) for name, path in source_paths.items()},
    )


def _windows_reparse_result(result: os.stat_result) -> object:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return mock.Mock(
        st_mode=result.st_mode,
        st_file_attributes=(
            getattr(result, "st_file_attributes", 0) | reparse_flag
        ),
    )


class TransitionModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = _settings()
        self.settings = StrategySettings(
            **{**self.settings.__dict__, "top_k": 2, "initial_nav": 24.1}
        )
        self.state = PortfolioState(
            cash=0.1,
            positions={
                "KEEP": Position(units=1.0, last_mark=10.0),
                "EXIT": Position(units=2.0, last_mark=5.0),
            },
            previous_nav=22.1,
        )

    def test_disabling_buys_marks_holdings_and_retries_sales(self) -> None:
        market = pd.DataFrame(
            [
                _market_row("KEEP", 12.0, 12.0),
                _market_row("EXIT", 15.0, 15.0, is_suspended=True),
                _market_row("MISSING", 8.0, 8.0),
            ]
        )

        result = transition_at_open(
            self.state,
            ("KEEP", "MISSING"),
            market,
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            self.settings,
            allow_buys=False,
        )

        self.assertFalse(result.trades["side"].eq("buy").any())
        self.assertEqual(result.state.positions["KEEP"].units, 1.0)
        self.assertEqual(result.state.positions["KEEP"].last_mark, 12.0)
        self.assertIn("EXIT", result.state.positions)
        exit_trades = result.trades.query("stock_code == 'EXIT'")
        self.assertEqual(len(exit_trades), 1)
        self.assertTrue(exit_trades["side"].eq("sell").all())
        self.assertTrue(exit_trades["status"].eq("blocked").all())

    def test_default_buy_mode_preserves_transition_snapshot(self) -> None:
        market = pd.DataFrame(
            [
                _market_row("KEEP", 12.0, 12.0),
                _market_row("EXIT", 6.0, 6.0),
                _market_row("BUY", 8.0, 8.0),
            ]
        )

        result = transition_at_open(
            self.state,
            ("KEEP", "BUY"),
            market,
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            self.settings,
        )

        expected_trades = pd.DataFrame(
            [
                {
                    "stock_code": "EXIT",
                    "signal_date": pd.Timestamp("2024-01-02"),
                    "execution_date": pd.Timestamp("2024-01-03"),
                    "side": "sell",
                    "status": "executed",
                    "reason": "eligible",
                    "requested_gross_notional": 12.0,
                    "gross_notional": 12.0,
                    "cost": 0.0132,
                    "units": 2.0,
                    "raw_open": 6.0,
                    "post_open": 6.0,
                },
                {
                    "stock_code": "BUY",
                    "signal_date": pd.Timestamp("2024-01-02"),
                    "execution_date": pd.Timestamp("2024-01-03"),
                    "side": "buy",
                    "status": "executed",
                    "reason": "eligible",
                    "requested_gross_notional": 12.05,
                    "gross_notional": 12.05,
                    "cost": 0.00723,
                    "units": 1.50625,
                    "raw_open": 8.0,
                    "post_open": 8.0,
                },
            ]
        ).set_index("stock_code")
        assert_frame_equal(result.trades, expected_trades)
        self.assertEqual(result.state.cash, 0.029569999999999652)
        self.assertEqual(result.state.positions["KEEP"].units, 1.0)
        self.assertEqual(result.state.positions["BUY"].units, 1.50625)
        self.assertEqual(result.total_cost, 0.02043)
        self.assertEqual(result.end_nav, 24.07957)

    def test_buy_mode_requires_a_boolean_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "allow_buys must be boolean"):
            transition_at_open(
                self.state,
                ("KEEP",),
                pd.DataFrame([_market_row("KEEP", 10.0, 10.0)]),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
                self.settings,
                allow_buys=1,
            )


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.settings = _settings(end="2024-01-03")
        self.source_paths = _write_sealed_loader_inputs(
            Path(self.temporary_directory.name), self.settings
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loader_seals_sources_and_selects_only_score_raw(self) -> None:
        inputs = strategy.load_locked_strategy_inputs(
            LOCKED_MODEL_NAMES[0], self.settings, self.source_paths
        )

        self.assertEqual(inputs.source_paths, self.source_paths)
        self.assertEqual(
            inputs.source_hashes,
            {name: file_sha256(path) for name, path in self.source_paths.items()},
        )
        self.assertEqual(
            select_daily_top(inputs.predictions, self.settings, inputs.trading_calendar),
            {
                pd.Timestamp("2024-01-02"): ("A",),
                pd.Timestamp("2024-01-03"): ("A",),
            },
        )

    def test_loader_rejects_a_prediction_that_does_not_match_its_seal(self) -> None:
        self.source_paths["prediction"].write_bytes(b"tampered")

        with self.assertRaisesRegex(
            ValueError, "locked-test conclusion prediction hash does not match"
        ):
            strategy.load_locked_strategy_inputs(
                LOCKED_MODEL_NAMES[0], self.settings, self.source_paths
            )

    def test_loader_rejects_a_model_outside_the_frozen_set(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "strategy model is not frozen: not-a-frozen-model"
        ):
            strategy.load_locked_strategy_inputs(
                "not-a-frozen-model", self.settings, self.source_paths
            )


class DailyRegressionTests(unittest.TestCase):
    def test_daily_simulation_snapshot_is_unchanged(self) -> None:
        settings = StrategySettings(
            **{
                **_settings().__dict__,
                "commission_rate": 0.0,
                "slippage_rate": 0.0,
                "sell_stamp_duty_rate": 0.0,
            }
        )
        calendar = pd.DataFrame(
            {"date": ["2024-01-02", "2024-01-03", "2024-01-04"]}
        )
        predictions = pd.DataFrame(
            [
                ("2024-01-02", "A", 2.0),
                ("2024-01-02", "B", 1.0),
                ("2024-01-03", "A", 1.0),
                ("2024-01-03", "B", 2.0),
                ("2024-01-04", "A", 2.0),
                ("2024-01-04", "B", 1.0),
            ],
            columns=["date", "stock_code", "score_raw"],
        ).assign(split="test", horizon=10)
        market = pd.DataFrame(
            [
                {"date": "2024-01-03", **_market_row("A", 10.0, 10.0)},
                {"date": "2024-01-03", **_market_row("B", 10.0, 10.0)},
                {"date": "2024-01-04", **_market_row("A", 11.0, 11.0)},
                {"date": "2024-01-04", **_market_row("B", 10.0, 10.0)},
            ]
        )

        bundle = simulate_strategy(predictions, market, calendar, settings)

        expected_daily_nav = pd.DataFrame(
            [
                (pd.Timestamp("2024-01-02"), pd.NaT, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0),
                (pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02"), 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1, 1, 1.0, 0.0, 0.0, 1.0, 0.5, 1, 1, 0),
                (pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-03"), 1.1, 1.1, 0.1, 0.1, 0.0, 0.0, 1, 1, 1.1, 1.1, 0.0, 2.0, 1.0, 2, 2, 0),
            ],
            columns=("date", *bundle.daily_nav.columns),
        ).set_index("date")
        expected_daily_nav.index = pd.DatetimeIndex(expected_daily_nav.index, name="date")
        expected_trades = pd.DataFrame(
            [
                ("A", "2024-01-02", "2024-01-03", "buy", "executed", "eligible", 1.0, 1.0, 0.0, 0.1, 10.0, 10.0),
                ("A", "2024-01-03", "2024-01-04", "sell", "executed", "eligible", 1.1, 1.1, 0.0, 0.1, 11.0, 11.0),
                ("B", "2024-01-03", "2024-01-04", "buy", "executed", "eligible", 1.1, 1.1, 0.0, 0.11, 10.0, 10.0),
            ],
            columns=bundle.trades.columns,
        )
        expected_trades["signal_date"] = pd.to_datetime(expected_trades["signal_date"])
        expected_trades["execution_date"] = pd.to_datetime(expected_trades["execution_date"])
        expected_positions = pd.DataFrame(
            [
                ("2024-01-03", "A", 0.1, 10.0, 1.0),
                ("2024-01-04", "B", 0.11, 10.0, 1.1),
            ],
            columns=bundle.positions.columns,
        )
        expected_positions["date"] = pd.to_datetime(expected_positions["date"])
        expected_diagnostics = pd.DataFrame(
            [
                ("2024-01-02", "2024-01-03", 1.0, 1.0, 0.0, 1.0, 1, 1, 0, 0),
                ("2024-01-03", "2024-01-04", 1.1, 1.1, 0.0, 1.0, 2, 2, 0, 0),
            ],
            columns=bundle.execution_diagnostics.columns,
        )
        expected_diagnostics["signal_date"] = pd.to_datetime(expected_diagnostics["signal_date"])
        expected_diagnostics["execution_date"] = pd.to_datetime(expected_diagnostics["execution_date"])
        expected_ending_positions = pd.DataFrame(
            [("position", "B", 0.11, 10.0, 1.1), ("cash", "CASH", 1.0, 0.0, 0.0)],
            columns=bundle.ending_positions.columns,
        )

        assert_frame_equal(bundle.daily_nav, expected_daily_nav)
        assert_frame_equal(bundle.trades, expected_trades)
        assert_frame_equal(bundle.positions, expected_positions)
        assert_frame_equal(bundle.execution_diagnostics, expected_diagnostics)
        assert_frame_equal(bundle.ending_positions, expected_ending_positions)
        self.assertEqual(
            bundle.metrics_summary,
            {
                "elapsed_trading_observations": 2,
                "cumulative_return": 0.10000000000000009,
                "cagr": 164238.77066398552,
                "annualized_return": 164238.77066398552,
                "annualized_volatility": 1.1224972160321836,
                "sharpe_ratio": 11.224972160321824,
                "max_drawdown": -0.0,
                "win_rate": 0.5,
                "average_cash_ratio": 0.3333333333333333,
                "gross_turnover": 3.0,
                "one_way_turnover": 1.5,
                "average_gross_turnover": 1.5,
                "average_one_way_turnover": 0.75,
                "average_turnover": 0.75,
                "annualized_gross_turnover": 378.0,
                "annualized_one_way_turnover": 189.0,
                "annualized_turnover": 189.0,
                "total_cost": 0.0,
                "buy_attempt_count": 2,
                "sell_attempt_count": 1,
                "buy_count": 2,
                "sell_count": 1,
                "buy_fill_rate": 1.0,
                "sell_fill_rate": 1.0,
                "blocked_sale_days": 0,
                "ending_nav": 1.1,
            },
        )


class AggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.paths = _aggregation_paths()
        self.settings = _settings(end="2024-01-22")

    def test_average_nav_uses_only_genuine_common_dates_and_economic_means(self) -> None:
        average = build_average_nav(self.paths, self.settings)

        common_end = min(path.schedule.final_horizon_date for path in self.paths)
        sample_date = pd.Timestamp("2024-01-05")
        self.assertEqual(average.index.min(), pd.Timestamp("2024-01-02"))
        self.assertEqual(average.index.max(), common_end)
        self.assertEqual(len(average), 6)
        self.assertAlmostEqual(
            average.loc[sample_date, "nav"],
            np.mean(
                [path.strategy.daily_nav.loc[sample_date, "nav"] for path in self.paths]
            ),
        )
        self.assertAlmostEqual(
            average.loc[sample_date, "total_cost"],
            np.mean(
                [
                    path.strategy.daily_nav.loc[sample_date, "total_cost"]
                    for path in self.paths
                ]
            ),
        )
        prior_nav = average["nav"].shift(1).loc[sample_date]
        self.assertAlmostEqual(
            average.loc[sample_date, "net_return"],
            average.loc[sample_date, "nav"] / prior_nav - 1.0,
        )
        self.assertAlmostEqual(
            average.loc[sample_date, "gross_return"],
            average.loc[sample_date, "pre_trade_nav"] / prior_nav - 1.0,
        )
        self.assertAlmostEqual(
            average.loc[sample_date, "gross_turnover"],
            (
                average.loc[sample_date, "buy_notional"]
                + average.loc[sample_date, "sell_notional"]
            )
            / average.loc[sample_date, "pre_trade_nav"],
        )
        self.assertAlmostEqual(
            average.loc[sample_date, "cash_ratio"],
            average.loc[sample_date, "cash"] / average.loc[sample_date, "nav"],
        )

    def test_average_metrics_are_recomputed_instead_of_averaging_path_metrics(self) -> None:
        average = build_average_nav(self.paths, self.settings)
        metrics = summarize_average_nav(average, self.settings)
        observations = average.iloc[1:]
        returns = observations["net_return"]
        daily_volatility = float(returns.std(ddof=1))
        expected_sharpe = (
            float(returns.mean())
            / daily_volatility
            * np.sqrt(self.settings.annualization_days)
        )

        self.assertTupleEqual(tuple(metrics), AVERAGE_METRICS)
        self.assertEqual(metrics["elapsed_trading_observations"], 5)
        self.assertAlmostEqual(
            metrics["cumulative_return"],
            float(average["nav"].iloc[-1]) / self.settings.initial_nav - 1.0,
        )
        self.assertAlmostEqual(
            metrics["annualized_volatility"],
            daily_volatility * np.sqrt(self.settings.annualization_days),
        )
        self.assertAlmostEqual(metrics["sharpe_ratio"], expected_sharpe)
        self.assertNotAlmostEqual(
            metrics["sharpe_ratio"],
            np.mean(
                [path.strategy.metrics_summary["sharpe_ratio"] for path in self.paths]
            ),
        )
        self.assertAlmostEqual(
            metrics["max_drawdown"],
            float(-(average["nav"] / average["nav"].cummax() - 1.0).min()),
        )
        self.assertAlmostEqual(
            metrics["average_cash_ratio"], float(average["cash_ratio"].mean())
        )
        self.assertAlmostEqual(
            metrics["gross_turnover"], float(observations["gross_turnover"].sum())
        )
        self.assertAlmostEqual(
            metrics["total_cost"], float(observations["total_cost"].sum())
        )

    def test_offset_reports_cover_every_path_metric_and_distribution_statistic(self) -> None:
        detail = build_offset_metrics(self.paths)
        summary = build_offset_summary(detail)

        self.assertEqual(len(detail), 10)
        self.assertListEqual(detail["offset"].tolist(), list(range(1, 11)))
        self.assertAlmostEqual(
            detail.loc[detail["offset"].eq(4), "ending_nav"].iloc[0],
            self.paths[3].strategy.metrics_summary["ending_nav"],
        )
        self.assertTupleEqual(tuple(summary.index), OFFSET_STATISTICS)
        self.assertTupleEqual(tuple(summary.columns), tuple(strategy._SUMMARY_FORMULAS))
        expected_std = detail["ending_nav"].std(ddof=1)
        self.assertAlmostEqual(summary.loc["std", "ending_nav"], expected_std)
        self.assertAlmostEqual(
            summary.loc["median", "total_cost"], detail["total_cost"].median()
        )


class PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.paths = _aggregation_paths()
        self.settings = _settings(end="2024-01-22")
        source_root = self.root / "sources"
        source_root.mkdir()
        self.inputs = _synthetic_locked_inputs(source_root, self.paths)
        self.model_name = LOCKED_MODEL_NAMES[0]
        self.run_root = self.root / "strategy_10d_runs"
        self.destination = self.run_root / self.model_name
        self.loader_calls: list[str] = []

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _publish(
        self,
        *,
        replace: object | None = None,
        writer: object | None = None,
    ) -> Path:
        def load_once(
            model_name: str,
            settings: StrategySettings,
            source_paths: dict[str, Path],
        ) -> LockedStrategyInputs:
            self.loader_calls.append(model_name)
            if len(self.loader_calls) > 1:
                raise AssertionError("locked inputs were loaded more than once")
            return self.inputs

        def simulated_path(
            predictions: pd.DataFrame,
            market: pd.DataFrame,
            calendar: pd.DataFrame,
            settings: StrategySettings,
            schedule: OffsetSchedule,
        ) -> StaggeredPathBundle:
            return self.paths[schedule.offset - 1]

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    staggered, "load_locked_strategy_inputs", side_effect=load_once
                )
            )
            stack.enter_context(
                mock.patch.object(
                    staggered,
                    "build_offset_schedules",
                    return_value=tuple(path.schedule for path in self.paths),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    staggered,
                    "simulate_staggered_offset",
                    side_effect=simulated_path,
                )
            )
            if replace is not None:
                stack.enter_context(mock.patch.object(staggered.os, "replace", replace))
            if writer is not None:
                stack.enter_context(
                    mock.patch.object(
                        staggered, "_write_staggered_offset", side_effect=writer
                    )
                )
            return backtest_staggered_strategy(
                self.model_name,
                destination=self.destination,
                settings=self.settings,
                source_paths=self.inputs.source_paths,
            )

    def test_publishes_one_complete_hash_bound_model_directory(self) -> None:
        real_replace = os.replace
        renames: list[tuple[Path, Path]] = []

        def recording_replace(source: object, destination: object) -> None:
            renames.append((Path(source), Path(destination)))
            real_replace(source, destination)

        published = self._publish(replace=recording_replace)

        self.assertEqual(published, self.destination)
        self.assertEqual(self.loader_calls, [self.model_name])
        self.assertEqual(len(renames), 1)
        self.assertEqual(renames[0][0].parent, self.destination.parent)
        self.assertEqual(renames[0][1], self.destination)
        self.assertSetEqual(
            {path.name for path in self.destination.iterdir()},
            {
                *(f"offset_{offset:02d}" for offset in range(1, 11)),
                "offset_metrics.csv",
                "offset_summary.csv",
                "average_nav.csv",
                "average_metrics.json",
                "manifest.json",
            },
        )

        model_manifest = json.loads(
            (self.destination / "manifest.json").read_text(encoding="utf-8")
        )
        model_files = {
            "offset_metrics.csv",
            "offset_summary.csv",
            "average_nav.csv",
            "average_metrics.json",
            "manifest.json",
        }
        offset_files = {
            "daily_nav.csv",
            "trades.parquet",
            "positions.parquet",
            "execution_diagnostics.csv",
            "ending_positions.csv",
            "metrics_summary.json",
            "manifest.json",
        }
        self.assertSetEqual(set(model_manifest["output_sha256"]), model_files)
        for name in model_files - {"manifest.json"}:
            self.assertEqual(
                model_manifest["output_sha256"][name],
                file_sha256(self.destination / name),
            )
        self.assertEqual(len(model_manifest["output_sha256"]["manifest.json"]), 64)
        self.assertEqual(model_manifest["input_sha256"], self.inputs.source_hashes)
        self.assertEqual(model_manifest["rows"]["offset_metrics"], 10)
        self.assertEqual(model_manifest["rows"]["offset_summary"], 5)
        self.assertEqual(len(model_manifest["offsets"]), 10)
        self.assertEqual(model_manifest["formulas"]["offset_metrics"], strategy._SUMMARY_FORMULAS)
        self.assertEqual(
            set(model_manifest["formulas"]["average_metrics"]), set(AVERAGE_METRICS)
        )

        for offset, descriptor in enumerate(model_manifest["offsets"], start=1):
            directory = self.destination / f"offset_{offset:02d}"
            self.assertSetEqual({path.name for path in directory.iterdir()}, offset_files)
            self.assertEqual(descriptor["offset"], offset)
            self.assertEqual(descriptor["path"], directory.name)
            self.assertEqual(descriptor["manifest_sha256"], file_sha256(directory / "manifest.json"))
            self.assertSetEqual(
                set(descriptor["output_sha256"]), offset_files - {"manifest.json"}
            )
            for name, digest in descriptor["output_sha256"].items():
                self.assertEqual(digest, file_sha256(directory / name))
            offset_manifest = json.loads(
                (directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(offset_manifest["schedule"], descriptor["schedule"])
            self.assertEqual(offset_manifest["rows"], descriptor["rows"])
            self.assertEqual(offset_manifest["output_sha256"], descriptor["output_sha256"])

        _validate_staggered_publication(
            self.destination,
            self.model_name,
            self.inputs.source_paths,
            self.inputs.source_hashes,
        )

    def test_rejects_existing_or_escaped_destinations_without_overwrite(self) -> None:
        self.destination.mkdir(parents=True)
        marker = self.destination / "partial.txt"
        marker.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            backtest_staggered_strategy(
                self.model_name,
                destination=self.destination,
                settings=self.settings,
                source_paths=self.inputs.source_paths,
            )
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

        escaped = self.run_root / ".." / self.model_name
        with self.assertRaisesRegex(ValueError, "destination"):
            backtest_staggered_strategy(
                self.model_name,
                destination=escaped,
                settings=self.settings,
                source_paths=self.inputs.source_paths,
            )

    def test_rejects_a_symlinked_output_root(self) -> None:
        target = self.root / "outside"
        target.mkdir()
        symlink_root = self.root / "symlink-case" / "strategy_10d_runs"
        symlink_root.parent.mkdir()
        try:
            os.symlink(target, symlink_root, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"directory symlinks unavailable: {error}")

        with self.assertRaisesRegex(ValueError, "symlink"):
            backtest_staggered_strategy(
                self.model_name,
                destination=symlink_root / self.model_name,
                settings=self.settings,
                source_paths=self.inputs.source_paths,
            )

    def test_rejects_an_output_root_that_becomes_a_junction_after_creation(
        self,
    ) -> None:
        real_lstat = Path.lstat

        def output_root_reparse_lstat(path: Path) -> object:
            result = real_lstat(path)
            if path == self.run_root and path.exists():
                return _windows_reparse_result(result)
            return result

        with mock.patch.object(Path, "lstat", new=output_root_reparse_lstat):
            with self.assertRaisesRegex(ValueError, "junction"):
                backtest_staggered_strategy(
                    self.model_name,
                    destination=self.destination,
                    settings=self.settings,
                    source_paths=self.inputs.source_paths,
                )

        self.assertFalse(self.destination.exists())

    def test_python_311_fallback_rejects_a_windows_reparse_output_root(self) -> None:
        real_lstat = Path.lstat

        def python_311_lstat(path: Path) -> object:
            result = real_lstat(path)
            if path == self.run_root and path.exists():
                return _windows_reparse_result(result)
            return result

        with mock.patch.object(
            staggered, "_is_windows_platform", return_value=True
        ), mock.patch.object(Path, "lstat", new=python_311_lstat):
            with self.assertRaisesRegex(ValueError, "reparse"):
                backtest_staggered_strategy(
                    self.model_name,
                    destination=self.destination,
                    settings=self.settings,
                    source_paths=self.inputs.source_paths,
                )

        self.assertFalse(self.destination.exists())

    def test_rejects_a_junction_in_the_staging_subtree(self) -> None:
        real_lstat = Path.lstat

        def staged_offset_reparse_lstat(path: Path) -> object:
            result = real_lstat(path)
            if (
                path.name == "offset_04"
                and path.parent.name.startswith(f".{self.model_name}.")
            ):
                return _windows_reparse_result(result)
            return result

        with mock.patch.object(Path, "lstat", new=staged_offset_reparse_lstat):
            with self.assertRaisesRegex(ValueError, "junction"):
                self._publish()

        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.run_root.glob(f".{self.model_name}.*")), [])

    def test_rechecks_destination_ancestry_immediately_before_rename(self) -> None:
        real_validator = staggered._validate_staggered_publication
        real_lstat = Path.lstat
        staged_validation_finished = False

        def validate_then_change_ancestry(*args: object, **kwargs: object) -> None:
            nonlocal staged_validation_finished
            real_validator(*args, **kwargs)
            staged_validation_finished = True

        def late_output_root_reparse_lstat(path: Path) -> object:
            result = real_lstat(path)
            if (
                path == self.run_root and staged_validation_finished
            ):
                return _windows_reparse_result(result)
            return result

        with mock.patch.object(
            staggered,
            "_validate_staggered_publication",
            side_effect=validate_then_change_ancestry,
        ), mock.patch.object(Path, "lstat", new=late_output_root_reparse_lstat):
            with self.assertRaisesRegex(ValueError, "junction"):
                self._publish()

        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.run_root.glob(f".{self.model_name}.*")), [])

    def test_rejects_a_symlinked_publication_lock(self) -> None:
        lock_path = self.run_root / ".strategy-10d.lock"
        real_is_symlink = Path.is_symlink

        def lock_is_symlink(path: Path) -> bool:
            return path == lock_path or real_is_symlink(path)

        with mock.patch.object(Path, "is_symlink", new=lock_is_symlink):
            with self.assertRaisesRegex(ValueError, "symlink"):
                backtest_staggered_strategy(
                    self.model_name,
                    destination=self.destination,
                    settings=self.settings,
                    source_paths=self.inputs.source_paths,
                )

    def test_validation_rejects_output_and_source_tampering(self) -> None:
        self._publish()

        path_file = self.destination / "offset_03" / "daily_nav.csv"
        original_path_file = path_file.read_bytes()
        path_file.write_bytes(original_path_file + b"tampered")
        with self.assertRaisesRegex(ValueError, "hash"):
            _validate_staggered_publication(
                self.destination,
                self.model_name,
                self.inputs.source_paths,
                self.inputs.source_hashes,
            )
        path_file.write_bytes(original_path_file)

        summary_file = self.destination / "offset_summary.csv"
        original_summary = summary_file.read_bytes()
        summary_file.write_bytes(original_summary + b"tampered")
        with self.assertRaisesRegex(ValueError, "hash"):
            _validate_staggered_publication(
                self.destination,
                self.model_name,
                self.inputs.source_paths,
                self.inputs.source_hashes,
            )
        summary_file.write_bytes(original_summary)

        for source_name in ("prediction_manifest", "trading_calendar"):
            source = self.inputs.source_paths[source_name]
            original_source = source.read_bytes()
            source.write_bytes(original_source + b"tampered")
            with self.assertRaisesRegex(ValueError, "source"):
                _validate_staggered_publication(
                    self.destination,
                    self.model_name,
                    self.inputs.source_paths,
                    self.inputs.source_hashes,
                )
            source.write_bytes(original_source)

    def test_interrupted_staging_never_publishes_a_partial_directory(self) -> None:
        real_writer = staggered._write_staggered_offset
        writes = 0

        def interrupted_writer(*args: object, **kwargs: object) -> object:
            nonlocal writes
            writes += 1
            result = real_writer(*args, **kwargs)
            if writes == 2:
                raise RuntimeError("injected publication interruption")
            return result

        with self.assertRaisesRegex(RuntimeError, "injected publication interruption"):
            self._publish(writer=interrupted_writer)

        self.assertFalse(self.destination.exists())
        self.assertEqual(
            list(self.run_root.glob(f".{self.model_name}.*")),
            [],
        )

    def test_cleanup_retry_preserves_the_original_publication_error(self) -> None:
        real_writer = staggered._write_staggered_offset
        real_rmtree = staggered.shutil.rmtree
        writes = 0
        cleanup_attempts = 0

        def interrupted_writer(*args: object, **kwargs: object) -> object:
            nonlocal writes
            writes += 1
            result = real_writer(*args, **kwargs)
            if writes == 2:
                raise RuntimeError("original publication failure")
            return result

        def transient_cleanup_failure(path: object) -> None:
            nonlocal cleanup_attempts
            cleanup_attempts += 1
            if cleanup_attempts == 1:
                raise OSError("transient staging cleanup failure")
            real_rmtree(path)

        with mock.patch.object(
            staggered.shutil,
            "rmtree",
            side_effect=transient_cleanup_failure,
        ):
            with self.assertRaisesRegex(RuntimeError, "original publication failure"):
                self._publish(writer=interrupted_writer)

        self.assertEqual(cleanup_attempts, 2)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.run_root.glob(f".{self.model_name}.*")), [])

    def test_persistent_cleanup_failure_leaves_staging_and_blocks_the_next_run(
        self,
    ) -> None:
        real_writer = staggered._write_staggered_offset
        real_rmtree = staggered.shutil.rmtree
        writes = 0
        cleanup_attempts = 0

        def interrupted_writer(*args: object, **kwargs: object) -> object:
            nonlocal writes
            writes += 1
            result = real_writer(*args, **kwargs)
            if writes == 2:
                raise RuntimeError("original persistent-cleanup publication failure")
            return result

        def persistent_cleanup_failure(path: object) -> None:
            nonlocal cleanup_attempts
            cleanup_attempts += 1
            raise OSError("persistent staging cleanup failure")

        with mock.patch.object(
            staggered.shutil,
            "rmtree",
            side_effect=persistent_cleanup_failure,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "original persistent-cleanup publication failure"
            ):
                self._publish(writer=interrupted_writer)

        stale_staging = list(self.run_root.glob(f".{self.model_name}.*"))
        try:
            self.assertEqual(cleanup_attempts, 2)
            self.assertEqual(len(stale_staging), 1)
            self.assertFalse(self.destination.exists())
            with self.assertRaisesRegex(FileExistsError, "partial.*staging"):
                backtest_staggered_strategy(
                    self.model_name,
                    destination=self.destination,
                    settings=self.settings,
                    source_paths=self.inputs.source_paths,
                )
        finally:
            for staging in stale_staging:
                real_rmtree(staging)


class ScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calendar = _sealed_synthetic_calendar()
        self.period = pd.DatetimeIndex(self.calendar["date"])
        self.settings = _schedule_settings(self.period[0], self.period[-1])

    def test_partitions_every_complete_signal_into_ten_offsets(self) -> None:
        schedules = build_offset_schedules(self.calendar, self.settings)

        self.assertEqual(len(schedules), 10)
        self.assertEqual(sum(len(schedule.signal_dates) for schedule in schedules), 474)
        self.assertEqual(
            {date for schedule in schedules for date in schedule.signal_dates},
            set(self.period[:-11]),
        )
        self.assertEqual(schedules[0].execution_dates[0], self.period[1])
        self.assertEqual(schedules[9].execution_dates[0], self.period[10])
        self.assertEqual(
            min(schedule.final_horizon_date for schedule in schedules),
            self.period[475],
        )
        self.assertEqual(self.period[475], pd.Timestamp("2025-12-18"))
        self.assertEqual(self.period[:-11][0], pd.Timestamp("2024-01-02"))
        self.assertEqual(self.period[:-11][-1], pd.Timestamp("2025-12-16"))

    def test_offset_schedule_normalizes_date_sequences_to_immutable_tuples(self) -> None:
        schedule = OffsetSchedule(
            offset=1,
            signal_dates=[self.period[0]],
            execution_dates=[self.period[1]],
            final_horizon_date=self.period[11],
        )

        self.assertIsInstance(schedule.signal_dates, tuple)
        self.assertIsInstance(schedule.execution_dates, tuple)

    def test_rejects_duplicate_dates_and_short_periods(self) -> None:
        duplicate_calendar = pd.concat(
            [self.calendar, self.calendar.iloc[[0]]], ignore_index=True
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_offset_schedules(duplicate_calendar, self.settings)

        short_period = self.calendar.iloc[:11].copy()
        short_settings = _schedule_settings(
            pd.Timestamp(short_period["date"].iloc[0]),
            pd.Timestamp(short_period["date"].iloc[-1]),
        )
        with self.assertRaisesRegex(ValueError, "at least 12"):
            build_offset_schedules(short_period, short_settings)

    def test_validator_rejects_missing_or_overlapping_offsets(self) -> None:
        schedules = build_offset_schedules(self.calendar, self.settings)
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            _validate_offset_schedules(schedules[:-1], self.period)

        overlapping = (*schedules[:1], schedules[0], *schedules[2:])
        with self.assertRaisesRegex(ValueError, "partition"):
            _validate_offset_schedules(overlapping, self.period)


class PathSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.bdate_range("2024-01-02", periods=34)
        self.settings = StrategySettings(
            **{
                **_settings().__dict__,
                "start": self.dates[0],
                "end": self.dates[-1],
                "top_k": 2,
                "expected_cross_section_size": 4,
                "commission_rate": 0.0,
                "slippage_rate": 0.0,
                "sell_stamp_duty_rate": 0.0,
            }
        )
        self.schedule = OffsetSchedule(
            offset=3,
            signal_dates=(self.dates[2], self.dates[12], self.dates[22]),
            execution_dates=(self.dates[3], self.dates[13], self.dates[23]),
            final_horizon_date=self.dates[33],
        )

    def _predictions(
        self, desired_by_date: dict[pd.Timestamp, tuple[str, str]]
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for date in self.dates:
            desired = desired_by_date.get(date, ("A", "B"))
            scores = {
                stock_code: float(4 - desired.index(stock_code))
                if stock_code in desired
                else 1.0
                for stock_code in ("A", "B", "C", "D")
            }
            rows.extend(
                {
                    "date": date,
                    "stock_code": stock_code,
                    "score_raw": scores[stock_code],
                    "split": "test",
                    "horizon": 10,
                }
                for stock_code in ("A", "B", "C", "D")
            )
        return pd.DataFrame(rows)

    def _market(
        self, overrides: dict[tuple[pd.Timestamp, str], dict[str, object]] | None = None
    ) -> pd.DataFrame:
        overrides = overrides or {}
        rows: list[dict[str, object]] = []
        for date in self.dates:
            for stock_code in ("A", "B", "C", "D"):
                row = {"date": date, **_market_row(stock_code, 10.0, 10.0)}
                row.update(overrides.get((date, stock_code), {}))
                rows.append(row)
        return pd.DataFrame(rows)

    def _simulate(
        self,
        desired_by_date: dict[pd.Timestamp, tuple[str, str]],
        overrides: dict[tuple[pd.Timestamp, str], dict[str, object]] | None = None,
    ) -> StaggeredPathBundle:
        return simulate_staggered_offset(
            self._predictions(desired_by_date),
            self._market(overrides),
            pd.DataFrame({"date": self.dates}),
            self.settings,
            self.schedule,
        )

    def test_rejects_a_manual_offset_mismatch_or_irregular_cadence(self) -> None:
        malformed_schedules = (
            OffsetSchedule(
                offset=2,
                signal_dates=(self.dates[2], self.dates[12], self.dates[22]),
                execution_dates=(self.dates[3], self.dates[13], self.dates[23]),
                final_horizon_date=self.dates[33],
            ),
            OffsetSchedule(
                offset=3,
                signal_dates=(self.dates[2], self.dates[11], self.dates[22]),
                execution_dates=(self.dates[3], self.dates[12], self.dates[23]),
                final_horizon_date=self.dates[33],
            ),
        )

        for schedule in malformed_schedules:
            with self.subTest(schedule=schedule):
                with self.assertRaisesRegex(ValueError, "offset cadence"):
                    simulate_staggered_offset(
                        self._predictions({}),
                        self._market(),
                        pd.DataFrame({"date": self.dates}),
                        self.settings,
                        schedule,
                    )

    def test_stays_in_cash_before_first_execution_and_marks_between_rebalances(self) -> None:
        bundle = self._simulate(
            {
                self.dates[2]: ("A", "B"),
                self.dates[4]: ("C", "D"),
                self.dates[12]: ("C", "D"),
                self.dates[22]: ("C", "D"),
            },
            {(self.dates[4], "A"): {"post_open": 12.0}},
        )

        daily = bundle.strategy.daily_nav
        for date in self.dates[:3]:
            self.assertEqual(daily.loc[date, "nav"], 1.0)
            self.assertEqual(daily.loc[date, "holding_count"], 0)
            self.assertFalse(daily.loc[date, "is_rebalance"])
        self.assertTrue(daily.loc[self.dates[3], "is_rebalance"])
        self.assertEqual(daily.loc[self.dates[3], "active_signal_date"], self.dates[2])
        self.assertFalse(daily.loc[self.dates[4], "is_rebalance"])
        self.assertEqual(daily.loc[self.dates[4], "active_signal_date"], self.dates[2])
        self.assertEqual(daily.loc[self.dates[4], "nav"], 1.1)
        self.assertTrue(
            bundle.strategy.trades.loc[
                bundle.strategy.trades["execution_date"].eq(self.dates[4])
            ].empty
        )
        self.assertSetEqual(
            set(
                bundle.strategy.trades.loc[
                    bundle.strategy.trades["execution_date"].eq(self.dates[13]),
                    "stock_code",
                ]
            ),
            {"A", "B", "C", "D"},
        )
        diagnostic = bundle.strategy.execution_diagnostics.loc[
            bundle.strategy.execution_diagnostics["execution_date"].eq(self.dates[4])
        ].iloc[0]
        self.assertEqual(diagnostic["offset"], 3)
        self.assertFalse(diagnostic["is_rebalance"])
        self.assertEqual(diagnostic["active_signal_date"], self.dates[2])

    def test_retries_a_blocked_sell_on_the_next_day(self) -> None:
        bundle = self._simulate(
            {
                self.dates[2]: ("A", "B"),
                self.dates[12]: ("C", "D"),
                self.dates[22]: ("C", "D"),
            },
            {(self.dates[13], "A"): {"is_suspended": True}},
        )

        sales = bundle.strategy.trades.loc[
            bundle.strategy.trades["stock_code"].eq("A")
            & bundle.strategy.trades["side"].eq("sell")
        ]
        self.assertEqual(sales["execution_date"].tolist()[:2], [self.dates[13], self.dates[14]])
        self.assertEqual(sales["status"].tolist()[:2], ["blocked", "executed"])
        self.assertNotIn(
            "A", bundle.strategy.ending_positions.loc[
                bundle.strategy.ending_positions["asset_type"].eq("position"), "stock_code"
            ].tolist(),
        )

    def test_does_not_retry_a_failed_buy_until_the_next_scheduled_rebalance(self) -> None:
        bundle = self._simulate(
            {
                self.dates[2]: ("A", "B"),
                self.dates[12]: ("A", "B"),
                self.dates[22]: ("A", "B"),
            },
            {(self.dates[3], "A"): {"is_st": True}},
        )

        buys = bundle.strategy.trades.loc[
            bundle.strategy.trades["stock_code"].eq("A")
            & bundle.strategy.trades["side"].eq("buy")
        ]
        self.assertEqual(buys["execution_date"].tolist(), [self.dates[3], self.dates[13]])
        self.assertEqual(buys["status"].tolist(), ["blocked", "executed"])

    def test_reentering_desired_set_cancels_pending_blocked_exit(self) -> None:
        suspended_dates = {
            (date, "A"): {"is_suspended": True}
            for date in self.dates[13:23]
        }
        bundle = self._simulate(
            {
                self.dates[2]: ("A", "B"),
                self.dates[12]: ("C", "D"),
                self.dates[22]: ("A", "D"),
            },
            suspended_dates,
        )

        final_rebalance = bundle.strategy.trades.loc[
            bundle.strategy.trades["execution_date"].eq(self.dates[23])
        ]
        self.assertNotIn("A", final_rebalance["stock_code"].tolist())
        self.assertIn(
            "A", bundle.strategy.ending_positions.loc[
                bundle.strategy.ending_positions["asset_type"].eq("position"), "stock_code"
            ].tolist(),
        )

    def test_preserves_units_for_the_desired_intersection(self) -> None:
        bundle = self._simulate(
            {
                self.dates[2]: ("A", "B"),
                self.dates[12]: ("B", "C"),
                self.dates[22]: ("B", "C"),
            }
        )

        positions = bundle.strategy.positions
        initial_units = positions.loc[
            positions["date"].eq(self.dates[3]) & positions["stock_code"].eq("B"), "units"
        ].iloc[0]
        rebalanced_units = positions.loc[
            positions["date"].eq(self.dates[13]) & positions["stock_code"].eq("B"), "units"
        ].iloc[0]
        self.assertEqual(initial_units, rebalanced_units)
        self.assertTrue(
            bundle.strategy.trades.loc[
                bundle.strategy.trades["execution_date"].eq(self.dates[13])
                & bundle.strategy.trades["stock_code"].eq("B")
            ].empty
        )

    def test_ends_at_the_horizon_without_forcing_terminal_liquidation(self) -> None:
        bundle = self._simulate(
            {
                self.dates[2]: ("A", "B"),
                self.dates[12]: ("C", "D"),
                self.dates[22]: ("C", "D"),
            }
        )

        self.assertEqual(bundle.strategy.daily_nav.index[-1], self.schedule.final_horizon_date)
        self.assertTrue(
            bundle.strategy.trades.loc[
                bundle.strategy.trades["execution_date"].eq(self.schedule.final_horizon_date)
            ].empty
        )
        self.assertSetEqual(
            set(
                bundle.strategy.ending_positions.loc[
                    bundle.strategy.ending_positions["asset_type"].eq("position"), "stock_code"
                ]
            ),
            {"C", "D"},
        )


if __name__ == "__main__":
    unittest.main()
