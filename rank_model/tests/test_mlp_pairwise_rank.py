from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from rank_model.stages.ranking import pairwise_logistic_loss, sample_date_pairs
from rank_model.stages.training import (
    MLP_EPOCHS,
    MODEL_REGISTRY,
    train_mlp_pairwise_rank,
    train_registered_model,
)


class MlpPairwiseRankTests(unittest.TestCase):
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
        rows_per_date = 10
        positions = np.tile(np.arange(rows_per_date), len(dates))
        signals = positions.astype("float64") / (rows_per_date - 1)
        return pd.DataFrame(
            {
                "date": np.repeat(dates, rows_per_date),
                "stock_code": [f"S{index:04d}" for index in range(len(positions))],
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

    def test_sampler_is_same_date_deterministic_and_has_eight_pairs_per_anchor(
        self,
    ) -> None:
        dates = pd.Series(
            np.repeat(pd.to_datetime(["2020-01-02", "2020-01-03"]), 10)
        )
        targets = np.tile(np.linspace(0.0, 1.0, 10), 2)

        first = sample_date_pairs(dates, targets, 8, 0.5, 42)
        second = sample_date_pairs(dates, targets, 8, 0.5, 42)
        changed = sample_date_pairs(dates, targets, 8, 0.5, 43)
        left, right, direction = first

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])
        self.assertTrue(
            not np.array_equal(right, changed[1]),
            "changing the seed must change at least one sampled opponent",
        )
        np.testing.assert_array_equal(
            np.bincount(left, minlength=len(dates)), np.full(len(dates), 8)
        )
        self.assertTrue(np.all(dates.iloc[left].to_numpy() == dates.iloc[right].to_numpy()))
        self.assertTrue(np.all(left != right))
        self.assertTrue(np.all(targets[left] != targets[right]))
        self.assertTrue(np.all(np.isin(direction, [-1, 1])))
        np.testing.assert_array_equal(
            direction, np.sign(targets[left] - targets[right]).astype("int8")
        )

    def test_sampler_rejects_a_date_without_a_valid_opponent(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid opponent"):
            sample_date_pairs(
                pd.Series(pd.to_datetime(["2020-01-02", "2020-01-02"])),
                np.array([0.5, 0.5]),
                8,
                0.5,
                42,
            )

    def test_pairwise_logistic_loss_rewards_correct_ordering(self) -> None:
        good = pairwise_logistic_loss(
            torch.tensor([2.0, 0.0]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([1.0]),
        )
        bad = pairwise_logistic_loss(
            torch.tensor([0.0, 2.0]),
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([1.0]),
        )

        self.assertLess(float(good), float(bad))

    def test_pairwise_mlp_is_deterministic_and_publishes_reloaded_scores(self) -> None:
        first = train_mlp_pairwise_rank(self.train, self.validation, self.schema, {})
        second = train_mlp_pairwise_rank(self.train, self.validation, self.schema, {})

        self.assertTrue(np.isfinite(first.score_validation).all())
        np.testing.assert_allclose(
            first.score_validation, second.score_validation, rtol=1e-6, atol=1e-6
        )
        self.assertEqual(first.metadata["epochs"], MLP_EPOCHS)
        self.assertEqual(first.metadata["pairs_per_stock"], 8)
        self.assertEqual(first.metadata["adjacent_fraction"], 0.5)
        self.assertEqual(first.metadata["dates_per_batch"], 8)
        self.assertEqual(first.metadata["pair_count"], len(self.train) * 8)
        self.assertIs(MODEL_REGISTRY["mlp_pairwise_rank"], train_mlp_pairwise_rank)

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

[models.mlp_pairwise_rank]
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
                    "models": {"mlp_pairwise_rank": {}},
                },
                config_path,
                "mlp_pairwise_rank",
                "mlp-pairwise-synthetic",
            )

            manifest = json.loads(
                (run_directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["metadata"]["pair_count"], len(self.train) * 8)
            self.assertEqual(manifest["metadata"]["pairs_per_stock"], 8)
            self.assertTrue((run_directory / "model_state_dict.pt").exists())
            predictions = pd.read_parquet(run_directory / "predictions_10d.parquet")
            np.testing.assert_allclose(
                predictions["score_raw"].to_numpy(dtype="float64"),
                first.score_validation,
                rtol=1e-6,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
