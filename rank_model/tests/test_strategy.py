"""Temporary contract tests for the frozen strategy stage."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
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
