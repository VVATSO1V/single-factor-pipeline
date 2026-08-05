from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from rank_model.stages.training import (
    MODEL_REGISTRY,
    train_mlp_rank_regression,
    train_registered_model,
)


class MlpRankRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = {
            "continuous_feature_columns": ["factor_a", "factor_b"],
            "industry_column": "industry",
        }
        dates = pd.to_datetime(
            ["2019-01-02", "2020-01-02", "2021-01-04", "2022-01-03", "2023-01-03"]
        )
        self.dataset = self._frame(dates)
        self.train = self.dataset.loc[self.dataset["date"].dt.year.lt(2023)].copy()
        self.validation = self.dataset.loc[
            self.dataset["date"].dt.year.eq(2023)
        ].copy()

    @staticmethod
    def _frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
        rows_per_date = 20
        positions = np.tile(np.arange(rows_per_date), len(dates))
        signals = positions.astype("float64") / (rows_per_date - 1)
        return pd.DataFrame(
            {
                "date": np.repeat(dates, rows_per_date),
                "stock_code": [f"S{index:03d}" for index in range(len(positions))],
                "factor_a": signals,
                "factor_b": np.cos(signals * np.pi),
                "industry": np.where(positions % 2 == 0, "A", "B"),
                "target_10d": signals,
                "rank_target_10d": signals,
                "split_10d": np.where(
                    np.repeat(dates.year, rows_per_date) == 2023,
                    "validation",
                    "train",
                ),
                "exit_date_10d": np.repeat(dates, rows_per_date),
            }
        )

    def test_mlp_is_deterministic_and_persists_a_reloaded_state_dict(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)
        first = train_mlp_rank_regression(
            self.train, self.validation, self.schema, {}
        )
        np.random.seed(42)
        torch.manual_seed(42)
        second = train_mlp_rank_regression(
            self.train, self.validation, self.schema, {}
        )

        self.assertTrue(np.isfinite(first.score_validation).all())
        np.testing.assert_allclose(
            first.score_validation,
            second.score_validation,
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertEqual(first.metadata["hidden_layers"], [128, 64, 32])
        self.assertEqual(first.metadata["epochs"], 12)
        self.assertEqual(first.metadata["loss"], "mse")
        self.assertIs(MODEL_REGISTRY["mlp_rank_regression"], train_mlp_rank_regression)

        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
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
            self.dataset.loc[:, columns].to_parquet(dataset_path, index=False)
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

[models.mlp_rank_regression]
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
                    "models": {"mlp_rank_regression": {}},
                },
                config_path,
                "mlp_rank_regression",
                "mlp-synthetic",
            )

            self.assertTrue((run_directory / "model_architecture.json").exists())
            self.assertTrue((run_directory / "model_state_dict.pt").exists())
            architecture = json.loads(
                (run_directory / "model_architecture.json").read_text(encoding="utf-8")
            )
            self.assertEqual(architecture["hidden_layers"], [128, 64, 32])
            predictions = pd.read_parquet(run_directory / "predictions_10d.parquet")
            np.testing.assert_allclose(
                predictions["score_raw"].to_numpy(dtype="float64"),
                first.score_validation,
                rtol=1e-6,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
