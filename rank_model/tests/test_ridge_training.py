from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rank_model.stages.training import (
    train_registered_model,
    train_ridge_rank_regression,
)


class RidgeTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = {
            "continuous_feature_columns": ["factor_a", "factor_b"],
            "industry_column": "industry",
        }
        self.params = {"lambda": 1.0}
        self.train = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2019-01-02",
                        "2019-01-02",
                        "2020-01-02",
                        "2020-01-02",
                        "2022-12-30",
                        "2022-12-30",
                    ]
                ),
                "factor_a": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "factor_b": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "industry": ["A", "B", "A", "B", "A", "B"],
                "rank_target_10d": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            }
        )
        self.validation = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
                "factor_a": [0.25, 0.75],
                "factor_b": [0.25, 0.75],
                "industry": ["A", "C"],
                "rank_target_10d": [0.25, np.nan],
            }
        )

    def test_ridge_returns_finite_scores_and_records_lambda(self) -> None:
        outcome = train_ridge_rank_regression(
            self.train, self.validation, self.schema, self.params
        )

        self.assertEqual(outcome.score_validation.shape, (len(self.validation),))
        self.assertTrue(np.isfinite(outcome.score_validation).all())
        self.assertEqual(outcome.metadata["lambda"], 1.0)

    def test_existing_run_directory_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            run_directory = root / "runs" / "immutable-run"
            run_directory.mkdir(parents=True)
            marker = run_directory / "marker.txt"
            marker.write_text("preserve", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                train_registered_model(
                    {"paths": {"runs_dir": "runs"}},
                    root / "config.toml",
                    "ridge_rank_regression",
                    "immutable-run",
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_registered_ridge_publishes_verified_prediction_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            dataset_path = root / "rank_dataset.parquet"
            schema_path = root / "rank_schema.json"
            config_path = root / "config.toml"
            dataset = pd.concat(
                [
                    self.train.assign(
                        stock_code=[f"T{index}" for index in range(len(self.train))],
                        target_10d=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                        split_10d="train",
                        exit_date_10d=pd.Timestamp("2023-01-16"),
                    ),
                    self.validation.assign(
                        stock_code=[f"V{index}" for index in range(len(self.validation))],
                        target_10d=[0.25, np.nan],
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

[models.ridge_rank_regression]
lambda = 1.0
""",
                encoding="utf-8",
            )

            run_directory = train_registered_model(
                {
                    "paths": {
                        "rank_dataset": "rank_dataset.parquet",
                        "rank_schema": "rank_schema.json",
                        "runs_dir": "runs",
                    },
                    "models": {"ridge_rank_regression": self.params},
                },
                config_path,
                "ridge_rank_regression",
                "ridge-synthetic",
            )

            for filename in (
                "config_snapshot.toml",
                "feature_schema.json",
                "preprocessor.joblib",
                "model.joblib",
                "predictions_10d.parquet",
                "manifest.json",
            ):
                self.assertTrue((run_directory / filename).exists(), filename)
            predictions = pd.read_parquet(run_directory / "predictions_10d.parquet")
            self.assertEqual(len(predictions), len(self.validation))
            self.assertEqual(predictions["stock_code"].tolist(), ["V0", "V1"])
            self.assertTrue(predictions["rank_target_10d"].iloc[1] != predictions["rank_target_10d"].iloc[1])
            manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse({"accept", "reject", "champion"}.intersection(manifest))


if __name__ == "__main__":
    unittest.main()
