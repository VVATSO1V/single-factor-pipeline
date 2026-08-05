from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from typing import get_args, get_type_hints

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from rank_model.stages.ranking import lightgbm_relevance, sorted_group_layout
from rank_model.stages.training import (
    MODEL_REGISTRY,
    _native_ranking_training_inputs,
    train_lightgbm_lambdarank,
    train_registered_model,
    train_xgboost_pairwise_rank,
)


class NativeTreeRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = {
            "continuous_feature_columns": ["factor_a", "factor_b"],
            "industry_column": "industry",
        }
        self.train = self._frame(pd.bdate_range("2019-01-02", "2022-12-30"))
        self.validation = self._frame(pd.bdate_range("2023-01-03", periods=2))

    @staticmethod
    def _frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
        rows_per_date = 30
        row_count = len(dates) * rows_per_date
        positions = np.tile(np.arange(rows_per_date), len(dates))
        signal = positions / (rows_per_date - 1)
        return pd.DataFrame(
            {
                "date": np.repeat(dates, rows_per_date),
                "stock_code": [f"S{index % rows_per_date:04d}" for index in range(row_count)],
                "factor_a": signal,
                "factor_b": np.arange(row_count, dtype="float64") / row_count,
                "industry": np.where(positions % 2 == 0, "A", "B"),
                "rank_target_10d": signal,
            }
        )

    def test_sorted_group_layout_stably_orders_date_and_stock_code(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2023-01-04",
                        "2023-01-03",
                        "2023-01-05",
                        "2023-01-04",
                        "2023-01-05",
                        "2023-01-03",
                        "2023-01-05",
                        "2023-01-03",
                        "2023-01-05",
                    ]
                ),
                "stock_code": ["C", "B", "D", "A", "A", "A", "C", "C", "B"],
                "marker": list(range(9)),
            }
        )

        sorted_frame, group_sizes = sorted_group_layout(frame)

        self.assertEqual(group_sizes.tolist(), [3, 2, 4])
        self.assertEqual(
            sorted_frame.loc[:, ["date", "stock_code"]].values.tolist(),
            [
                [pd.Timestamp("2023-01-03"), "A"],
                [pd.Timestamp("2023-01-03"), "B"],
                [pd.Timestamp("2023-01-03"), "C"],
                [pd.Timestamp("2023-01-04"), "A"],
                [pd.Timestamp("2023-01-04"), "C"],
                [pd.Timestamp("2023-01-05"), "A"],
                [pd.Timestamp("2023-01-05"), "B"],
                [pd.Timestamp("2023-01-05"), "C"],
                [pd.Timestamp("2023-01-05"), "D"],
            ],
        )
        self.assertCountEqual(sorted_frame["marker"].tolist(), list(range(9)))

    def test_lightgbm_relevance_uses_hundred_fixed_linear_gains(self) -> None:
        actual = lightgbm_relevance(pd.Series([0.0, 0.009, 0.01, 0.999, 1.0]))

        np.testing.assert_array_equal(actual, [0, 0, 1, 99, 99])

    def test_native_ranking_input_annotation_matches_return_contract(self) -> None:
        return_types = get_args(
            get_type_hints(_native_ranking_training_inputs)["return"]
        )

        self.assertEqual(len(return_types), 10)
        self.assertIs(return_types[5], np.ndarray)
        self.assertIs(return_types[8], pd.Series)

    def test_native_tree_rankers_return_finite_scores_and_fixed_metadata(self) -> None:
        trainers = (
            (
                "xgboost_pairwise_rank",
                train_xgboost_pairwise_rank,
                "rank:pairwise",
                81,
            ),
            (
                "lightgbm_lambdarank",
                train_lightgbm_lambdarank,
                "lambdarank",
                21,
            ),
        )

        for name, trainer, objective, rounds in trainers:
            with self.subTest(name=name):
                outcome = trainer(self.train, self.validation, self.schema, {})

                self.assertEqual(outcome.score_validation.shape, (len(self.validation),))
                self.assertTrue(np.isfinite(outcome.score_validation).all())
                self.assertEqual(outcome.metadata["objective"], objective)
                self.assertEqual(outcome.metadata["boosting_rounds"], rounds)
                self.assertIs(MODEL_REGISTRY[name], trainer)

    def test_registered_native_tree_rankers_publish_reloaded_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            dataset_path = root / "rank_dataset.parquet"
            schema_path = root / "rank_schema.json"
            config_path = root / "config.toml"
            dataset = pd.concat(
                [
                    self.train.assign(
                        target_10d=self.train["rank_target_10d"],
                        split_10d="train",
                        exit_date_10d=pd.Timestamp("2023-01-13"),
                    ),
                    self.validation.assign(
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

[models.xgboost_pairwise_rank]

[models.lightgbm_lambdarank]
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
                    "xgboost_pairwise_rank": {},
                    "lightgbm_lambdarank": {},
                },
            }

            for model_name, artifact_name in (
                ("xgboost_pairwise_rank", "model.json"),
                ("lightgbm_lambdarank", "model.txt"),
            ):
                with self.subTest(model_name=model_name):
                    run_directory = train_registered_model(
                        config, config_path, model_name, model_name
                    )
                    predictions = pd.read_parquet(
                        run_directory / "predictions_10d.parquet"
                    )
                    self.assertTrue((run_directory / artifact_name).exists())
                    self.assertTrue(np.isfinite(predictions["score_raw"]).all())

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="^Setting the shape on a NumPy array has been deprecated",
                            category=DeprecationWarning,
                            module=r"^joblib\.numpy_pickle$",
                        )
                        preprocessor = joblib.load(run_directory / "preprocessor.joblib")
                    features = preprocessor.transform(
                        self.validation, scale_continuous=False
                    )
                    if model_name == "xgboost_pairwise_rank":
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
                        rtol=1e-6,
                        atol=1e-6,
                    )

    def test_shuffled_inputs_preserve_published_prediction_key_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            ordered_root = root / "ordered"
            shuffled_root = root / "shuffled"
            ordered_root.mkdir()
            shuffled_root.mkdir()
            ordered_dataset = self._published_dataset(self.train, self.validation)
            shuffled_dataset = self._published_dataset(
                self.train.sample(frac=1.0, random_state=17),
                self.validation.sample(frac=1.0, random_state=29),
            )
            ordered_config, ordered_config_path = self._write_training_inputs(
                ordered_root, ordered_dataset
            )
            shuffled_config, shuffled_config_path = self._write_training_inputs(
                shuffled_root, shuffled_dataset
            )

            expected_keys = (
                self.validation.loc[:, ["date", "stock_code"]]
                .sort_values(["date", "stock_code"])
                .reset_index(drop=True)
            )
            for model_name in ("xgboost_pairwise_rank", "lightgbm_lambdarank"):
                with self.subTest(model_name=model_name):
                    ordered_run = train_registered_model(
                        ordered_config, ordered_config_path, model_name, model_name
                    )
                    shuffled_run = train_registered_model(
                        shuffled_config, shuffled_config_path, model_name, model_name
                    )
                    ordered_predictions = pd.read_parquet(
                        ordered_run / "predictions_10d.parquet"
                    ).sort_values(["date", "stock_code"])
                    shuffled_predictions = pd.read_parquet(
                        shuffled_run / "predictions_10d.parquet"
                    ).sort_values(["date", "stock_code"])

                    pd.testing.assert_frame_equal(
                        shuffled_predictions.loc[:, ["date", "stock_code"]].reset_index(
                            drop=True
                        ),
                        expected_keys,
                    )
                    np.testing.assert_allclose(
                        shuffled_predictions["score_raw"].to_numpy(dtype="float64"),
                        ordered_predictions["score_raw"].to_numpy(dtype="float64"),
                        rtol=1e-6,
                        atol=1e-6,
                    )

    def _published_dataset(
        self, train: pd.DataFrame, validation: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.concat(
            [
                train.assign(
                    target_10d=train["rank_target_10d"],
                    split_10d="train",
                    exit_date_10d=pd.Timestamp("2023-01-13"),
                ),
                validation.assign(
                    target_10d=validation["rank_target_10d"],
                    split_10d="validation",
                    exit_date_10d=pd.Timestamp("2023-01-17"),
                ),
            ],
            ignore_index=True,
        )

    def _write_training_inputs(
        self, root: Path, dataset: pd.DataFrame
    ) -> tuple[dict[str, object], Path]:
        dataset_path = root / "rank_dataset.parquet"
        schema_path = root / "rank_schema.json"
        config_path = root / "config.toml"
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
                    "parquet_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        config_path.write_text(
            """[paths]
rank_dataset = "rank_dataset.parquet"
rank_schema = "rank_schema.json"
runs_dir = "runs"

[models.xgboost_pairwise_rank]

[models.lightgbm_lambdarank]
""",
            encoding="utf-8",
        )
        return (
            {
                "paths": {
                    "rank_dataset": "rank_dataset.parquet",
                    "rank_schema": "rank_schema.json",
                    "runs_dir": "runs",
                },
                "models": {
                    "xgboost_pairwise_rank": {},
                    "lightgbm_lambdarank": {},
                },
            },
            config_path,
        )


if __name__ == "__main__":
    unittest.main()
