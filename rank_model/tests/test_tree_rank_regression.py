from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

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
        self.learning_train = self._learning_frame(
            pd.bdate_range("2019-01-02", "2022-12-30")
        )
        self.learning_validation = self._learning_frame(
            pd.bdate_range("2023-01-03", periods=2)
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

    @staticmethod
    def _learning_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
        row_count = len(dates) * 30
        positions = np.tile(np.arange(30), len(dates))
        signal = positions / 29.0
        return pd.DataFrame(
            {
                "date": np.repeat(dates, 30),
                "factor_a": signal,
                "factor_b": np.arange(row_count, dtype="float64") / row_count,
                "industry": np.where(positions % 2 == 0, "A", "B"),
                "rank_target_10d": signal,
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

    def test_tree_models_learn_nonconstant_validation_scores(self) -> None:
        for trainer in (
            train_xgboost_rank_regression,
            train_lightgbm_rank_regression,
        ):
            outcome = trainer(
                self.learning_train, self.learning_validation, self.schema, {}
            )

            self.assertTrue(np.isfinite(outcome.score_validation).all())
            self.assertGreater(np.std(outcome.score_validation), 1e-12)

    def test_registered_tree_models_publish_reloaded_nonconstant_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            dataset_path = root / "rank_dataset.parquet"
            schema_path = root / "rank_schema.json"
            config_path = root / "config.toml"
            dataset = pd.concat(
                [
                    self.learning_train.assign(
                        stock_code=[
                            f"T{index}" for index in range(len(self.learning_train))
                        ],
                        target_10d=self.learning_train["rank_target_10d"],
                        split_10d="train",
                        exit_date_10d=pd.Timestamp("2023-01-13"),
                    ),
                    self.learning_validation.assign(
                        stock_code=[
                            f"V{index}" for index in range(len(self.learning_validation))
                        ],
                        target_10d=self.learning_validation["rank_target_10d"],
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
                self.assertGreater(np.std(predictions["score_raw"]), 1e-12)

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="^Setting the shape on a NumPy array has been deprecated",
                        category=DeprecationWarning,
                        module=r"^joblib\.numpy_pickle$",
                    )
                    preprocessor = joblib.load(run_directory / "preprocessor.joblib")
                features = preprocessor.transform(
                    self.learning_validation, scale_continuous=False
                )
                if model_name == "xgboost_rank_regression":
                    reloaded_model = xgb.Booster()
                    reloaded_model.load_model(run_directory / artifact_name)
                    reloaded_scores = reloaded_model.predict(xgb.DMatrix(features))
                else:
                    reloaded_model = lgb.Booster(
                        model_file=str(run_directory / artifact_name)
                    )
                    reloaded_scores = reloaded_model.predict(features)
                np.testing.assert_allclose(
                    reloaded_scores,
                    predictions["score_raw"].to_numpy(dtype="float64"),
                    rtol=1e-12,
                    atol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
