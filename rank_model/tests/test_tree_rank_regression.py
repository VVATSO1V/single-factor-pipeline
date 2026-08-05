from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rank_model.stages.training import (
    MODEL_REGISTRY,
    train_lightgbm_rank_regression,
    train_registered_model,
    train_xgboost_rank_regression,
)


class TreeRankRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = {
            "continuous_feature_columns": ["factor_a", "factor_b"],
            "industry_column": "industry",
        }
        self.train = self._frame(pd.date_range("2019-01-02", periods=20, freq="B"))
        self.validation = self._frame(
            pd.date_range("2023-01-03", periods=2, freq="B")
        )

    @staticmethod
    def _frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
        row_count = len(dates) * 30
        positions = np.tile(np.arange(30), len(dates))
        return pd.DataFrame(
            {
                "date": np.repeat(dates, 30),
                "factor_a": np.arange(row_count, dtype="float64") / row_count,
                "factor_b": np.tile(np.arange(30, dtype="float64"), len(dates)),
                "industry": np.where(positions % 2 == 0, "A", "B"),
                "rank_target_10d": positions / 29.0,
            }
        )

    def test_xgboost_returns_finite_scores_and_fixed_metadata(self) -> None:
        outcome = train_xgboost_rank_regression(
            self.train, self.validation, self.schema, {}
        )

        self.assertEqual(outcome.score_validation.shape, (len(self.validation),))
        self.assertTrue(np.isfinite(outcome.score_validation).all())
        self.assertEqual(outcome.metadata["objective"], "reg:squarederror")
        self.assertEqual(outcome.metadata["boosting_rounds"], 81)
        self.assertIs(MODEL_REGISTRY["xgboost_rank_regression"], train_xgboost_rank_regression)

    def test_lightgbm_returns_finite_scores_and_fixed_metadata(self) -> None:
        outcome = train_lightgbm_rank_regression(
            self.train, self.validation, self.schema, {}
        )

        self.assertEqual(outcome.score_validation.shape, (len(self.validation),))
        self.assertTrue(np.isfinite(outcome.score_validation).all())
        self.assertEqual(outcome.metadata["objective"], "regression")
        self.assertEqual(outcome.metadata["boosting_rounds"], 21)
        self.assertIs(MODEL_REGISTRY["lightgbm_rank_regression"], train_lightgbm_rank_regression)

    def test_registered_tree_models_publish_native_reloaded_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            dataset_path = root / "rank_dataset.parquet"
            schema_path = root / "rank_schema.json"
            config_path = root / "config.toml"
            dataset = pd.concat(
                [
                    self.train.assign(
                        stock_code=[f"T{index}" for index in range(len(self.train))],
                        target_10d=self.train["rank_target_10d"],
                        split_10d="train",
                        exit_date_10d=pd.Timestamp("2019-01-16"),
                    ),
                    self.validation.assign(
                        stock_code=[f"V{index}" for index in range(len(self.validation))],
                        target_10d=self.validation["rank_target_10d"],
                        split_10d="validation",
                        exit_date_10d=pd.Timestamp("2023-01-17"),
                    ),
                ],
                ignore_index=True,
            )
            columns = [
                "date",
                "stock_code",
                "factor_a",
                "factor_b",
                "industry",
                "target_10d",
                "rank_target_10d",
                "split_10d",
                "exit_date_10d",
            ]
            dataset.loc[:, columns].to_parquet(dataset_path, index=False)
            schema_path.write_text(
                json.dumps(
                    {
                        **self.schema,
                        "key_columns": ["date", "stock_code"],
                        "target_column": "target_10d",
                        "rank_target_column": "rank_target_10d",
                        "split_column": "split_10d",
                        "output_columns": columns,
                        "parquet_sha256": hashlib.sha256(
                            dataset_path.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                """[paths]
rank_dataset = "rank_dataset.parquet"
rank_schema = "rank_schema.json"
runs_dir = "runs"

[models.xgboost_rank_regression]

[models.lightgbm_rank_regression]
""",
                encoding="utf-8",
            )
            config = {
                "paths": {
                    "rank_dataset": "rank_dataset.parquet",
                    "rank_schema": "rank_schema.json",
                    "runs_dir": "runs",
                },
                "models": {
                    "xgboost_rank_regression": {},
                    "lightgbm_rank_regression": {},
                },
            }

            for model_name, artifact_name in (
                ("xgboost_rank_regression", "model.json"),
                ("lightgbm_rank_regression", "model.txt"),
            ):
                run_directory = train_registered_model(
                    config, config_path, model_name, model_name
                )
                self.assertTrue((run_directory / artifact_name).exists())
                self.assertTrue((run_directory / "preprocessor.joblib").exists())
                predictions = pd.read_parquet(run_directory / "predictions_10d.parquet")
                self.assertTrue(np.isfinite(predictions["score_raw"]).all())


if __name__ == "__main__":
    unittest.main()
