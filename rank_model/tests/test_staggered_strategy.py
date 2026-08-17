"""Temporary synthetic contracts for the staggered strategy implementation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

import rank_model.stages.strategy as strategy
from rank_model.stages.dataset import file_sha256
from rank_model.stages.strategy import (
    LOCKED_MODEL_NAMES,
    PortfolioState,
    Position,
    StrategySettings,
    select_daily_top,
    simulate_strategy,
    transition_at_open,
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
        self.assertTrue(result.trades.query("stock_code == 'EXIT'")["side"].eq("sell").all())
        self.assertTrue(result.trades.query("stock_code == 'EXIT'")["status"].eq("blocked").all())

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


if __name__ == "__main__":
    unittest.main()
