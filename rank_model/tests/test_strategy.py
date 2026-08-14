"""Temporary contract tests for the frozen strategy stage."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

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
